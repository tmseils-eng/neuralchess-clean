"""Input encoding: a position (plus history) as an ``119 x 8 x 8`` tensor.

Layout, following AlphaZero:

======  ===========================================================
planes  meaning
======  ===========================================================
0-111   eight history steps, most recent first.  Each step uses 14
        planes: 6 for the mover's pieces, 6 for the opponent's, and
        2 repetition flags (position seen once before / twice).
112     side to move (1.0 when black is to move)
113     fullmove number, scaled
114     mover's king-side castling right
115     mover's queen-side castling right
116     opponent's king-side castling right
117     opponent's queen-side castling right
118     halfmove (no-progress) clock, scaled
======  ===========================================================

The board is always rendered from the mover's point of view: when black is
to move the board is flipped vertically and the colours are swapped, so the
network only ever has to learn "my pieces move up the board".  This halves
the effective input space and is why the policy head needs no colour flag.

Search touches this code once per simulation, so history is kept as
:class:`Frame` snapshots - twelve integers and four scalars - rather than
full board copies.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..chess.bitboard import BLACK, WHITE, scan
from ..chess.board import CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ, Board

HISTORY_LENGTH = 8          # AlphaZero's depth, and the default here
PLANES_PER_STEP = 14        # 6 our pieces + 6 theirs + 2 repetition flags
EXTRA_PLANES = 7            # side to move, move number, 4 castling, no-progress
BOARD_SIZE = 8


def input_planes(history_length: int = HISTORY_LENGTH) -> int:
    """Number of input planes for a given history depth (8 steps -> 119)."""
    return history_length * PLANES_PER_STEP + EXTRA_PLANES


INPUT_PLANES = input_planes(HISTORY_LENGTH)  # 119

_MOVE_SCALE = 1.0 / 100.0
_NO_PROGRESS_SCALE = 1.0 / 100.0


class Frame:
    """An immutable, cheap snapshot of everything the encoder reads."""

    __slots__ = ("pieces", "turn", "castling", "halfmove", "fullmove", "repetitions")

    def __init__(self, pieces, turn, castling, halfmove, fullmove, repetitions):
        self.pieces = pieces
        self.turn = turn
        self.castling = castling
        self.halfmove = halfmove
        self.fullmove = fullmove
        self.repetitions = repetitions

    @classmethod
    def of(cls, board: Board) -> "Frame":
        return cls(
            (tuple(board.pieces[WHITE]), tuple(board.pieces[BLACK])),
            board.turn,
            board.castling,
            board.halfmove_clock,
            board.fullmove_number,
            board.history.count(board.zobrist),
        )


def _fill_piece_planes(out: np.ndarray, offset: int, frame: Frame, mover: int, flip: bool) -> None:
    for rel, color in enumerate((mover, 1 - mover)):
        planes = frame.pieces[color]
        for ptype in range(6):
            plane = out[offset + rel * 6 + ptype]
            for sq in scan(planes[ptype]):
                s = sq ^ 56 if flip else sq
                plane[s >> 3, s & 7] = 1.0


def encode_frames(history: Sequence[Frame],
                  history_length: int = HISTORY_LENGTH) -> np.ndarray:
    """Encode a frame history (oldest first, current position last).

    ``history_length`` is a property of the *network*, not of the caller: a
    model trained with two history steps must always be fed two, so callers
    read it off the model's config rather than assuming the default.
    """
    if not history:
        raise ValueError("history must contain at least the current position")
    current = history[-1]
    mover = current.turn
    flip = mover == BLACK

    out = np.zeros((input_planes(history_length), BOARD_SIZE, BOARD_SIZE),
                   dtype=np.float32)

    recent = list(history[-history_length:])
    recent.reverse()  # index 0 == current position
    for step, frame in enumerate(recent):
        offset = step * PLANES_PER_STEP
        _fill_piece_planes(out, offset, frame, mover, flip)
        if frame.repetitions >= 2:
            out[offset + 12].fill(1.0)
        if frame.repetitions >= 3:
            out[offset + 13].fill(1.0)

    base = history_length * PLANES_PER_STEP
    if mover == BLACK:
        out[base].fill(1.0)
    out[base + 1].fill(min(current.fullmove, 200) * _MOVE_SCALE)

    if mover == WHITE:
        rights = (CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ)
    else:
        rights = (CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ)
    for i, bit in enumerate(rights):
        if current.castling & bit:
            out[base + 2 + i].fill(1.0)

    out[base + 6].fill(min(current.halfmove, 100) * _NO_PROGRESS_SCALE)
    return out


def encode_position(history: Sequence,
                    history_length: int = HISTORY_LENGTH) -> np.ndarray:
    """Convenience wrapper accepting :class:`Board` objects or frames."""
    frames = [f if isinstance(f, Frame) else Frame.of(f) for f in history]
    return encode_frames(frames, history_length)


class PositionHistory:
    """Rolling window of :class:`Frame` snapshots used to build network inputs."""

    __slots__ = ("_frames", "window")

    def __init__(self, board: Board = None, frames: Sequence[Frame] = (),
                 window: int = HISTORY_LENGTH):
        # The window is the *deepest* history any consumer might ask for.
        # Encoding then slices to whatever the network was trained on, so one
        # history object can feed models with different depths.
        self.window = max(1, window)
        self._frames: List[Frame] = list(frames)
        if board is not None:
            self._frames.append(Frame.of(board))
        self._trim()

    def _trim(self) -> None:
        if len(self._frames) > self.window:
            del self._frames[: len(self._frames) - self.window]

    def push(self, board: Board) -> None:
        self._frames.append(Frame.of(board))
        self._trim()

    def push_frame(self, frame: Frame) -> None:
        self._frames.append(frame)
        self._trim()

    def pop(self) -> Frame:
        return self._frames.pop()

    def encode(self, history_length: int = HISTORY_LENGTH) -> np.ndarray:
        return encode_frames(self._frames, history_length)

    def frames(self) -> List[Frame]:
        return list(self._frames)

    def clone(self) -> "PositionHistory":
        return PositionHistory(frames=self._frames, window=self.window)

    def __len__(self) -> int:
        return len(self._frames)


def legal_move_mask(board: Board) -> np.ndarray:
    """Boolean mask over the 4672 policy outputs for the legal moves."""
    from .policy_map import POLICY_SIZE, move_to_index

    mask = np.zeros(POLICY_SIZE, dtype=bool)
    flip = board.turn == BLACK
    for move in board.legal_moves():
        mask[move_to_index(move, flip)] = True
    return mask
