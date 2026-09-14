"""The 4672-way policy head: a bijection between moves and output indices.

AlphaZero represents chess moves as an ``8 x 8 x 73`` stack.  Every origin
square owns 73 planes:

* **0-55  queen-like moves.**  Eight ray directions x seven distances.
  These cover king, queen, rook, bishop and single/double pawn pushes.
* **56-63 knight moves.**  The eight L-shaped offsets.
* **64-72 underpromotions.**  Three pawn directions (capture left, straight
  ahead, capture right) x three pieces (knight, bishop, rook).

Queen promotions are *not* given their own planes; a pawn reaching the last
rank via a queen-like plane is promoted to a queen by convention.  That is
what makes 73 rather than 76 planes sufficient.

Everything here is expressed in the *canonical* orientation, i.e. from the
point of view of the side to move, so pawns always march toward rank 8.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..chess.bitboard import BISHOP, KNIGHT, QUEEN, ROOK, square, square_file, square_rank
from ..chess.move import move_from, move_promotion, move_to

NUM_MOVE_PLANES = 73
POLICY_SIZE = 64 * NUM_MOVE_PLANES  # 4672

# Ray directions, in the order used by the reference implementation.
QUEEN_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHT_DIRECTIONS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
UNDERPROMOTIONS = (KNIGHT, BISHOP, ROOK)
UNDERPROMOTION_FILES = (-1, 0, 1)  # capture left, push, capture right

# (from, to, promotion) -> flat policy index, and the inverse.
_MOVE_TO_INDEX: Dict[Tuple[int, int, Optional[int]], int] = {}
_INDEX_TO_MOVE: Dict[int, Tuple[int, int, Optional[int]]] = {}


def _register(frm: int, to: int, promo: Optional[int], plane: int) -> None:
    index = frm * NUM_MOVE_PLANES + plane
    _MOVE_TO_INDEX[(frm, to, promo)] = index
    # The inverse keeps the *first* registration, so a queen-promotion alias
    # never shadows the plain queen-like move that shares its plane.  Callers
    # that need an exact move resolve the ambiguity against the legal list
    # (see :func:`index_to_move`).
    _INDEX_TO_MOVE.setdefault(index, (frm, to, promo))


def _build() -> None:
    for frm in range(64):
        f, r = square_file(frm), square_rank(frm)

        # Queen-like rays.
        for d, (df, dr) in enumerate(QUEEN_DIRECTIONS):
            for dist in range(1, 8):
                nf, nr = f + df * dist, r + dr * dist
                if not (0 <= nf < 8 and 0 <= nr < 8):
                    break
                to = square(nf, nr)
                plane = d * 7 + (dist - 1)
                _register(frm, to, None, plane)
                # A pawn reaching the last rank on a queen-like plane promotes
                # to a queen; register that alias on the same plane.
                if nr == 7 and r == 6:
                    _register(frm, to, QUEEN, plane)

        # Knight jumps.
        for k, (df, dr) in enumerate(KNIGHT_DIRECTIONS):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                _register(frm, square(nf, nr), None, 56 + k)

        # Underpromotions (only meaningful from rank 7 in canonical view).
        if r == 6:
            for i, df in enumerate(UNDERPROMOTION_FILES):
                nf = f + df
                if not 0 <= nf < 8:
                    continue
                to = square(nf, 7)
                for j, piece in enumerate(UNDERPROMOTIONS):
                    _register(frm, to, piece, 64 + i * 3 + j)


_build()


def mirror_square(sq: int) -> int:
    """Vertical flip: ``a1 <-> a8``.  Turns a black-to-move board white-to-move."""
    return sq ^ 56


def move_to_index(move: int, flip: bool) -> int:
    """Flat policy index for ``move``; ``flip`` when black is to move."""
    frm, to = move_from(move), move_to(move)
    if flip:
        frm, to = mirror_square(frm), mirror_square(to)
    promo = move_promotion(move)
    try:
        return _MOVE_TO_INDEX[(frm, to, promo)]
    except KeyError as exc:  # pragma: no cover - indicates an encoding bug
        raise KeyError(f"move {(frm, to, promo)} has no policy plane") from exc


def index_to_move_squares(index: int, flip: bool):
    """Inverse of :func:`move_to_index`, returning ``(from, to, promotion)``."""
    frm, to, promo = _INDEX_TO_MOVE[index]
    if flip:
        frm, to = mirror_square(frm), mirror_square(to)
    return frm, to, promo


def plane_of(index: int) -> int:
    return index % NUM_MOVE_PLANES


def coverage() -> int:
    """Number of distinct (from, to, promo) triples the map can represent."""
    return len(_MOVE_TO_INDEX)


def index_to_move(index: int, legal_moves, flip: bool) -> Optional[int]:
    """Resolve a policy index to an actual legal move, or ``None``.

    Queen promotions share a plane with the queen-like move of the same
    geometry, so the only reliable inverse is one that filters the legal
    move list.  This is what the search and the UCI adapter use.
    """
    frm, to, promo = index_to_move_squares(index, flip)
    fallback = None
    for move in legal_moves:
        if move_from(move) != frm or move_to(move) != to:
            continue
        mp = move_promotion(move)
        if mp == promo:
            return move
        if promo is None and mp == QUEEN:
            fallback = move  # queen-promotion alias
    return fallback
