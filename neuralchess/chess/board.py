"""A complete, perft-verified chess position.

The implementation is deliberately self-contained: no third-party chess
library is used anywhere in this project.  Move generation is pseudo-legal
plus a make/undo legality filter, which is simple to get right and fast
enough for a search whose cost is dominated by network evaluation.

Everything needed by the learning pipeline lives here: full legal move
generation, make/unmake with an undo stack, Zobrist hashing, threefold
repetition, the fifty-move rule, and insufficient-material detection.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

from .bitboard import (
    BISHOP,
    BLACK,
    FULL,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    PAWN,
    PAWN_ATTACKS,
    PIECE_SYMBOLS,
    QUEEN,
    ROOK,
    SQUARE_INDEX,
    SQUARE_NAMES,
    WHITE,
    bishop_attacks,
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
    scan,
    square,
    square_file,
    square_rank,
)
from .move import (
    CASTLE,
    DOUBLE_PUSH,
    EN_PASSANT,
    PROMO_PIECES,
    make_move,
    move_flag,
    move_from,
    move_promotion,
    move_to,
    move_uci,
    parse_uci_squares,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Castling right bits.
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8
_CASTLE_CHARS = (("K", CASTLE_WK), ("Q", CASTLE_WQ), ("k", CASTLE_BK), ("q", CASTLE_BQ))

# Rights that survive a move touching a given square.
_CASTLE_MASK = [15] * 64
_CASTLE_MASK[SQUARE_INDEX["a1"]] = 15 ^ CASTLE_WQ
_CASTLE_MASK[SQUARE_INDEX["h1"]] = 15 ^ CASTLE_WK
_CASTLE_MASK[SQUARE_INDEX["e1"]] = 15 ^ CASTLE_WK ^ CASTLE_WQ
_CASTLE_MASK[SQUARE_INDEX["a8"]] = 15 ^ CASTLE_BQ
_CASTLE_MASK[SQUARE_INDEX["h8"]] = 15 ^ CASTLE_BK
_CASTLE_MASK[SQUARE_INDEX["e8"]] = 15 ^ CASTLE_BK ^ CASTLE_BQ

_RNG = random.Random(0x9E3779B97F4A7C15)
ZOBRIST_PIECES = [[[_RNG.getrandbits(64) for _ in range(64)] for _ in range(6)] for _ in range(2)]
ZOBRIST_SIDE = _RNG.getrandbits(64)
ZOBRIST_CASTLE = [_RNG.getrandbits(64) for _ in range(16)]
ZOBRIST_EP_FILE = [_RNG.getrandbits(64) for _ in range(8)]

_PROMO_RANK = (7, 0)
_PAWN_START_RANK = (1, 6)


class Board:
    """A mutable chess position with an undo stack."""

    __slots__ = (
        "pieces",
        "occupancy",
        "occupied",
        "mailbox",
        "turn",
        "castling",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
        "zobrist",
        "history",
        "_undo",
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, fen: str = START_FEN):
        self.set_fen(fen)

    def set_fen(self, fen: str) -> None:
        parts = fen.split()
        if len(parts) < 4:
            raise ValueError(f"malformed FEN: {fen!r}")
        placement, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1

        self.pieces = [[0] * 6, [0] * 6]
        self.occupancy = [0, 0]
        self.mailbox = [-1] * 64

        rank = 7
        file = 0
        for ch in placement:
            if ch == "/":
                rank -= 1
                file = 0
            elif ch.isdigit():
                file += int(ch)
            else:
                color = WHITE if ch.isupper() else BLACK
                ptype = PIECE_SYMBOLS.index(ch.lower())
                self._put(square(file, rank), color, ptype)
                file += 1

        self.turn = WHITE if side == "w" else BLACK
        self.castling = 0
        for ch, bit in _CASTLE_CHARS:
            if ch in castling:
                self.castling |= bit
        self.ep_square = None if ep == "-" else SQUARE_INDEX[ep]
        self.halfmove_clock = halfmove
        self.fullmove_number = fullmove
        self.zobrist = self._compute_zobrist()
        self.history: List[int] = [self.zobrist]
        self._undo: List[tuple] = []

    def copy(self) -> "Board":
        other = Board.__new__(Board)
        other.pieces = [self.pieces[0][:], self.pieces[1][:]]
        other.occupancy = self.occupancy[:]
        other.occupied = self.occupied
        other.mailbox = self.mailbox[:]
        other.turn = self.turn
        other.castling = self.castling
        other.ep_square = self.ep_square
        other.halfmove_clock = self.halfmove_clock
        other.fullmove_number = self.fullmove_number
        other.zobrist = self.zobrist
        other.history = self.history[:]
        other._undo = []
        return other

    # ------------------------------------------------------------------
    # Low-level piece placement
    # ------------------------------------------------------------------
    def _put(self, sq: int, color: int, ptype: int) -> None:
        bit = 1 << sq
        self.pieces[color][ptype] |= bit
        self.occupancy[color] |= bit
        self.mailbox[sq] = color * 6 + ptype
        self.occupied = self.occupancy[WHITE] | self.occupancy[BLACK]

    def _remove(self, sq: int, color: int, ptype: int) -> None:
        bit = 1 << sq
        self.pieces[color][ptype] ^= bit
        self.occupancy[color] ^= bit
        self.mailbox[sq] = -1
        self.occupied = self.occupancy[WHITE] | self.occupancy[BLACK]

    def piece_at(self, sq: int) -> Optional[Tuple[int, int]]:
        code = self.mailbox[sq]
        if code < 0:
            return None
        return divmod(code, 6)

    def king_square(self, color: int) -> int:
        bb = self.pieces[color][KING]
        if not bb:
            raise ValueError(f"position has no {'white' if color == WHITE else 'black'} king")
        return lsb(bb)

    def is_valid(self) -> bool:
        """True when the position could occur in a real game.

        Checks both kings exist and that the side *not* to move is not in
        check - the two conditions that make a FEN unplayable.
        """
        if not self.pieces[WHITE][KING] or not self.pieces[BLACK][KING]:
            return False
        return not self.is_attacked(self.king_square(1 - self.turn), self.turn)

    def _compute_zobrist(self) -> int:
        h = 0
        for sq in range(64):
            code = self.mailbox[sq]
            if code >= 0:
                color, ptype = divmod(code, 6)
                h ^= ZOBRIST_PIECES[color][ptype][sq]
        if self.turn == BLACK:
            h ^= ZOBRIST_SIDE
        h ^= ZOBRIST_CASTLE[self.castling]
        if self.ep_square is not None:
            h ^= ZOBRIST_EP_FILE[square_file(self.ep_square)]
        return h

    # ------------------------------------------------------------------
    # Attacks
    # ------------------------------------------------------------------
    def attackers_to(self, sq: int, by_color: int, occ: Optional[int] = None) -> int:
        """Bitboard of ``by_color`` pieces attacking ``sq``."""
        if occ is None:
            occ = self.occupied
        p = self.pieces[by_color]
        attackers = PAWN_ATTACKS[1 - by_color][sq] & p[PAWN]
        attackers |= KNIGHT_ATTACKS[sq] & p[KNIGHT]
        attackers |= KING_ATTACKS[sq] & p[KING]
        attackers |= bishop_attacks(sq, occ) & (p[BISHOP] | p[QUEEN])
        attackers |= rook_attacks(sq, occ) & (p[ROOK] | p[QUEEN])
        return attackers

    def is_attacked(self, sq: int, by_color: int, occ: Optional[int] = None) -> bool:
        return self.attackers_to(sq, by_color, occ) != 0

    def is_check(self) -> bool:
        return self.is_attacked(self.king_square(self.turn), 1 - self.turn)

    def gives_check(self, move: int) -> bool:
        self.push(move)
        result = self.is_check()
        self.pop()
        return result

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------
    def pseudo_legal_moves(self) -> List[int]:
        """Every move that follows the piece movement rules, ignoring pins.

        Captures of the enemy king are deliberately never generated.  Such a
        move can only arise from an illegal position (one where the side *not*
        to move is already in check), and generating it would leave the board
        without a king and corrupt every downstream consumer.  Excluding it
        costs nothing in legal play and makes the engine total on arbitrary
        FEN input.
        """
        moves: List[int] = []
        us, them = self.turn, 1 - self.turn
        own = self.occupancy[us]
        enemy = self.occupancy[them] & ~self.pieces[them][KING]
        occ = self.occupied
        empty = FULL ^ occ
        blocked = own | self.pieces[them][KING]
        p = self.pieces[us]

        # --- pawns -----------------------------------------------------
        forward = 8 if us == WHITE else -8
        promo_rank = _PROMO_RANK[us]
        start_rank = _PAWN_START_RANK[us]
        for frm in scan(p[PAWN]):
            r = square_rank(frm)
            one = frm + forward
            if 0 <= one < 64 and (empty >> one) & 1:
                if square_rank(one) == promo_rank:
                    for promo in PROMO_PIECES:
                        moves.append(make_move(frm, one, promo))
                else:
                    moves.append(make_move(frm, one))
                    if r == start_rank:
                        two = one + forward
                        if (empty >> two) & 1:
                            moves.append(make_move(frm, two, None, DOUBLE_PUSH))
            targets = PAWN_ATTACKS[us][frm] & enemy
            for to in scan(targets):
                if square_rank(to) == promo_rank:
                    for promo in PROMO_PIECES:
                        moves.append(make_move(frm, to, promo))
                else:
                    moves.append(make_move(frm, to))
            if self.ep_square is not None and (PAWN_ATTACKS[us][frm] >> self.ep_square) & 1:
                moves.append(make_move(frm, self.ep_square, None, EN_PASSANT))

        # --- knights ---------------------------------------------------
        for frm in scan(p[KNIGHT]):
            for to in scan(KNIGHT_ATTACKS[frm] & ~blocked):
                moves.append(make_move(frm, to))

        # --- bishops / rooks / queens ----------------------------------
        for frm in scan(p[BISHOP]):
            for to in scan(bishop_attacks(frm, occ) & ~blocked):
                moves.append(make_move(frm, to))
        for frm in scan(p[ROOK]):
            for to in scan(rook_attacks(frm, occ) & ~blocked):
                moves.append(make_move(frm, to))
        for frm in scan(p[QUEEN]):
            for to in scan(queen_attacks(frm, occ) & ~blocked):
                moves.append(make_move(frm, to))

        # --- king ------------------------------------------------------
        ksq = self.king_square(us)
        for to in scan(KING_ATTACKS[ksq] & ~blocked):
            moves.append(make_move(ksq, to))
        moves.extend(self._castling_moves(us, them, occ, ksq))
        return moves

    def _castling_moves(self, us: int, them: int, occ: int, ksq: int) -> List[int]:
        out: List[int] = []
        if us == WHITE:
            king_side, queen_side = CASTLE_WK, CASTLE_WQ
            e, f, g, d, c, b = (
                SQUARE_INDEX["e1"], SQUARE_INDEX["f1"], SQUARE_INDEX["g1"],
                SQUARE_INDEX["d1"], SQUARE_INDEX["c1"], SQUARE_INDEX["b1"],
            )
        else:
            king_side, queen_side = CASTLE_BK, CASTLE_BQ
            e, f, g, d, c, b = (
                SQUARE_INDEX["e8"], SQUARE_INDEX["f8"], SQUARE_INDEX["g8"],
                SQUARE_INDEX["d8"], SQUARE_INDEX["c8"], SQUARE_INDEX["b8"],
            )
        if ksq != e:
            return out
        if self.castling & king_side:
            if not (occ >> f) & 1 and not (occ >> g) & 1:
                if not self.is_attacked(e, them) and not self.is_attacked(f, them) \
                        and not self.is_attacked(g, them):
                    out.append(make_move(e, g, None, CASTLE))
        if self.castling & queen_side:
            if not (occ >> d) & 1 and not (occ >> c) & 1 and not (occ >> b) & 1:
                if not self.is_attacked(e, them) and not self.is_attacked(d, them) \
                        and not self.is_attacked(c, them):
                    out.append(make_move(e, c, None, CASTLE))
        return out

    def legal_moves(self) -> List[int]:
        """All strictly legal moves in the current position."""
        us = self.turn
        legal = []
        for move in self.pseudo_legal_moves():
            self.push(move)
            if not self.is_attacked(self.king_square(us), 1 - us):
                legal.append(move)
            self.pop()
        return legal

    def is_legal(self, move: int) -> bool:
        return move in self.legal_moves()

    # ------------------------------------------------------------------
    # Make / unmake
    # ------------------------------------------------------------------
    def push(self, move: int) -> None:
        us, them = self.turn, 1 - self.turn
        frm, to = move_from(move), move_to(move)
        flag = move_flag(move)
        promo = move_promotion(move)
        moving = self.mailbox[frm] % 6
        captured = self.mailbox[to]
        h = self.zobrist

        self._undo.append(
            (move, captured, self.castling, self.ep_square, self.halfmove_clock,
             self.fullmove_number, self.zobrist)
        )

        # Clear old en-passant hash contribution.
        if self.ep_square is not None:
            h ^= ZOBRIST_EP_FILE[square_file(self.ep_square)]
        h ^= ZOBRIST_CASTLE[self.castling]

        if captured >= 0:
            cap_type = captured % 6
            self._remove(to, them, cap_type)
            h ^= ZOBRIST_PIECES[them][cap_type][to]
        elif flag == EN_PASSANT:
            cap_sq = to - 8 if us == WHITE else to + 8
            self._remove(cap_sq, them, PAWN)
            h ^= ZOBRIST_PIECES[them][PAWN][cap_sq]

        self._remove(frm, us, moving)
        h ^= ZOBRIST_PIECES[us][moving][frm]
        landed = promo if promo is not None else moving
        self._put(to, us, landed)
        h ^= ZOBRIST_PIECES[us][landed][to]

        if flag == CASTLE:
            if to > frm:  # king side
                rook_from, rook_to = to + 1, to - 1
            else:         # queen side
                rook_from, rook_to = to - 2, to + 1
            self._remove(rook_from, us, ROOK)
            self._put(rook_to, us, ROOK)
            h ^= ZOBRIST_PIECES[us][ROOK][rook_from] ^ ZOBRIST_PIECES[us][ROOK][rook_to]

        self.castling &= _CASTLE_MASK[frm] & _CASTLE_MASK[to]
        h ^= ZOBRIST_CASTLE[self.castling]

        if flag == DOUBLE_PUSH:
            self.ep_square = (frm + to) // 2
            h ^= ZOBRIST_EP_FILE[square_file(self.ep_square)]
        else:
            self.ep_square = None

        if moving == PAWN or captured >= 0 or flag == EN_PASSANT:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1
        if us == BLACK:
            self.fullmove_number += 1

        self.turn = them
        h ^= ZOBRIST_SIDE
        self.zobrist = h
        self.history.append(h)

    def pop(self) -> int:
        (move, captured, castling, ep_square, halfmove, fullmove, zob) = self._undo.pop()
        self.history.pop()
        them = self.turn
        us = 1 - them
        frm, to = move_from(move), move_to(move)
        flag = move_flag(move)
        promo = move_promotion(move)
        landed = promo if promo is not None else self.mailbox[to] % 6

        self._remove(to, us, landed)
        moving = PAWN if promo is not None else landed
        self._put(frm, us, moving)

        if flag == CASTLE:
            if to > frm:
                rook_from, rook_to = to + 1, to - 1
            else:
                rook_from, rook_to = to - 2, to + 1
            self._remove(rook_to, us, ROOK)
            self._put(rook_from, us, ROOK)
        elif flag == EN_PASSANT:
            cap_sq = to - 8 if us == WHITE else to + 8
            self._put(cap_sq, them, PAWN)
        elif captured >= 0:
            self._put(to, them, captured % 6)

        self.turn = us
        self.castling = castling
        self.ep_square = ep_square
        self.halfmove_clock = halfmove
        self.fullmove_number = fullmove
        self.zobrist = zob
        return move

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------
    def is_insufficient_material(self) -> bool:
        if self.pieces[WHITE][PAWN] or self.pieces[BLACK][PAWN]:
            return False
        if self.pieces[WHITE][ROOK] or self.pieces[BLACK][ROOK]:
            return False
        if self.pieces[WHITE][QUEEN] or self.pieces[BLACK][QUEEN]:
            return False
        minors = 0
        bishops = 0
        bishop_colors = set()
        for color in (WHITE, BLACK):
            for sq in scan(self.pieces[color][BISHOP]):
                bishops += 1
                minors += 1
                bishop_colors.add((square_file(sq) + square_rank(sq)) & 1)
            minors += popcount(self.pieces[color][KNIGHT])
        if minors <= 1:
            return True
        # Any number of bishops, all on one color complex, cannot mate.
        if bishops == minors and len(bishop_colors) == 1:
            return True
        return False

    def is_repetition(self, count: int = 3) -> bool:
        return self.history.count(self.zobrist) >= count

    def is_fifty_moves(self) -> bool:
        return self.halfmove_clock >= 100

    def outcome(self) -> Optional[str]:
        """``'1-0'``, ``'0-1'``, ``'1/2-1/2'`` or ``None`` if the game goes on."""
        if not self.legal_moves():
            if self.is_check():
                return "0-1" if self.turn == WHITE else "1-0"
            return "1/2-1/2"
        if self.is_fifty_moves() or self.is_repetition() or self.is_insufficient_material():
            return "1/2-1/2"
        return None

    def terminal_value(self) -> Optional[float]:
        """Game result from the perspective of the side to move (+1/0/-1)."""
        return self.terminal_value_with_moves(self.legal_moves())

    def terminal_value_with_moves(self, moves: Sequence[int]) -> Optional[float]:
        """As :meth:`terminal_value`, reusing a move list the caller already has.

        Generating legal moves is the single most expensive rules operation,
        and the search needs both the move list and the terminal test at every
        leaf.  Passing the list in halves that cost.
        """
        if not moves:
            return -1.0 if self.is_check() else 0.0
        if self.is_fifty_moves() or self.is_repetition() or self.is_insufficient_material():
            return 0.0
        return None

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def fen(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row = ""
            empty = 0
            for file in range(8):
                code = self.mailbox[square(file, rank)]
                if code < 0:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                color, ptype = divmod(code, 6)
                sym = PIECE_SYMBOLS[ptype]
                row += sym.upper() if color == WHITE else sym
            if empty:
                row += str(empty)
            rows.append(row)
        placement = "/".join(rows)
        side = "w" if self.turn == WHITE else "b"
        rights = "".join(ch for ch, bit in _CASTLE_CHARS if self.castling & bit) or "-"
        ep = SQUARE_NAMES[self.ep_square] if self.ep_square is not None else "-"
        return f"{placement} {side} {rights} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def parse_uci(self, uci: str) -> int:
        """Resolve a UCI string against the current position (sets flags)."""
        frm, to, promo = parse_uci_squares(uci)
        for move in self.legal_moves():
            if move_from(move) == frm and move_to(move) == to and move_promotion(move) == promo:
                return move
        raise ValueError(f"illegal move {uci!r} in position {self.fen()}")

    def push_uci(self, uci: str) -> int:
        move = self.parse_uci(uci)
        self.push(move)
        return move

    def san(self, move: int) -> str:
        """Standard Algebraic Notation for a legal move in this position.

        Full disambiguation rules are implemented (file, rank, or both), along
        with castling, en passant, promotion and the check/mate suffix, so the
        games this engine plays can be pasted straight into any chess GUI.
        """
        frm, to = move_from(move), move_to(move)
        flag = move_flag(move)
        promo = move_promotion(move)
        moving = self.mailbox[frm] % 6
        captured = self.mailbox[to] >= 0 or flag == EN_PASSANT

        if flag == CASTLE:
            text = "O-O" if to > frm else "O-O-O"
        elif moving == PAWN:
            text = SQUARE_NAMES[frm][0] + "x" + SQUARE_NAMES[to] if captured \
                else SQUARE_NAMES[to]
            if promo is not None:
                text += "=" + PIECE_SYMBOLS[promo].upper()
        else:
            same = [
                m for m in self.legal_moves()
                if move_to(m) == to and m != move
                and self.mailbox[move_from(m)] is not None
                and self.mailbox[move_from(m)] % 6 == moving
            ]
            disambiguator = ""
            if same:
                files = {square_file(move_from(m)) for m in same}
                ranks = {square_rank(move_from(m)) for m in same}
                if square_file(frm) not in files:
                    disambiguator = SQUARE_NAMES[frm][0]
                elif square_rank(frm) not in ranks:
                    disambiguator = SQUARE_NAMES[frm][1]
                else:
                    disambiguator = SQUARE_NAMES[frm]
            text = PIECE_SYMBOLS[moving].upper() + disambiguator + ("x" if captured else "") \
                + SQUARE_NAMES[to]

        self.push(move)
        if self.is_check():
            text += "#" if not self.legal_moves() else "+"
        self.pop()
        return text

    def san_line(self, moves: Sequence[int]) -> List[str]:
        """SAN for a sequence of moves, played out from this position."""
        out = []
        for move in moves:
            out.append(self.san(move))
            self.push(move)
        for _ in moves:
            self.pop()
        return out

    def __str__(self) -> str:
        rows = []
        for rank in range(7, -1, -1):
            row = []
            for file in range(8):
                code = self.mailbox[square(file, rank)]
                if code < 0:
                    row.append(".")
                else:
                    color, ptype = divmod(code, 6)
                    sym = PIECE_SYMBOLS[ptype]
                    row.append(sym.upper() if color == WHITE else sym)
            rows.append(f"{rank + 1} " + " ".join(row))
        rows.append("  a b c d e f g h")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"


def perft(board: Board, depth: int) -> int:
    """Count leaf nodes of the legal move tree - the standard correctness test."""
    if depth == 0:
        return 1
    if depth == 1:
        return len(board.legal_moves())
    total = 0
    for move in board.legal_moves():
        board.push(move)
        total += perft(board, depth - 1)
        board.pop()
    return total


def perft_divide(board: Board, depth: int):
    """Per-root-move perft, the standard way to localise a movegen bug."""
    out = {}
    for move in board.legal_moves():
        board.push(move)
        out[move_uci(move)] = perft(board, depth - 1)
        board.pop()
    return out
