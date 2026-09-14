"""Bitboard primitives and precomputed attack tables.

Squares are indexed 0..63 with ``a1 == 0`` and ``h8 == 63`` so that
``rank = sq >> 3`` and ``file = sq & 7``.  Bitboards are plain Python
integers; bit ``i`` set means square ``i`` is occupied.

Sliding-piece attacks use the *classical ray* method: for every square and
every one of the eight directions we precompute the full ray.  At query
time the ray is masked against the occupancy, the nearest blocker is found
with a bit scan, and the segment beyond that blocker is subtracted.  This
avoids the memory cost (and the table-generation complexity) of magic
bitboards while remaining fast enough for a search that is dominated by
neural-network evaluation.
"""

from __future__ import annotations

from typing import List

# --------------------------------------------------------------------------
# Basic constants
# --------------------------------------------------------------------------

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

PIECE_SYMBOLS = "pnbrqk"
PIECE_NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")

FULL = (1 << 64) - 1

FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7
FILES = [FILE_A << i for i in range(8)]

RANK_1 = 0xFF
RANKS = [RANK_1 << (8 * i) for i in range(8)]

NOT_FILE_A = FULL ^ FILE_A
NOT_FILE_H = FULL ^ FILE_H

SQUARE_NAMES = [f"{'abcdefgh'[s & 7]}{(s >> 3) + 1}" for s in range(64)]
SQUARE_INDEX = {name: i for i, name in enumerate(SQUARE_NAMES)}


def square(file: int, rank: int) -> int:
    """Square index from 0-based ``file`` and ``rank``."""
    return (rank << 3) | file


def square_file(sq: int) -> int:
    return sq & 7


def square_rank(sq: int) -> int:
    return sq >> 3


def lsb(bb: int) -> int:
    """Index of the least significant set bit."""
    return (bb & -bb).bit_length() - 1


def msb(bb: int) -> int:
    """Index of the most significant set bit."""
    return bb.bit_length() - 1


def pop_lsb(bb: int):
    """Return ``(index, bb_without_that_bit)``."""
    low = bb & -bb
    return low.bit_length() - 1, bb ^ low


def popcount(bb: int) -> int:
    return bin(bb).count("1")


def scan(bb: int):
    """Iterate over the indices of the set bits, lowest first."""
    while bb:
        low = bb & -bb
        yield low.bit_length() - 1
        bb ^= low


def bb_str(bb: int) -> str:
    """Human-readable 8x8 rendering, rank 8 on top (debugging aid)."""
    rows = []
    for r in range(7, -1, -1):
        rows.append(" ".join("1" if bb >> square(f, r) & 1 else "." for f in range(8)))
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Leaper attack tables
# --------------------------------------------------------------------------

def _shift(bb: int, delta: int) -> int:
    """Shift with wrap-around protection on the a/h files."""
    if delta == 1:
        return (bb & NOT_FILE_H) << 1
    if delta == -1:
        return (bb & NOT_FILE_A) >> 1
    if delta == 9:
        return (bb & NOT_FILE_H) << 9
    if delta == 7:
        return (bb & NOT_FILE_A) << 7
    if delta == -7:
        return (bb & NOT_FILE_H) >> 7
    if delta == -9:
        return (bb & NOT_FILE_A) >> 9
    if delta == 8:
        return (bb << 8) & FULL
    if delta == -8:
        return bb >> 8
    raise ValueError(f"unsupported shift {delta}")


def _leaper_table(offsets) -> List[int]:
    table = [0] * 64
    for sq in range(64):
        f, r = square_file(sq), square_rank(sq)
        acc = 0
        for df, dr in offsets:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                acc |= 1 << square(nf, nr)
        table[sq] = acc
    return table


KNIGHT_ATTACKS = _leaper_table(
    [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
)
KING_ATTACKS = _leaper_table(
    [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
)

PAWN_ATTACKS = [[0] * 64, [0] * 64]
for _sq in range(64):
    _f, _r = square_file(_sq), square_rank(_sq)
    _w = _b = 0
    for _df in (-1, 1):
        if 0 <= _f + _df < 8:
            if _r + 1 < 8:
                _w |= 1 << square(_f + _df, _r + 1)
            if _r - 1 >= 0:
                _b |= 1 << square(_f + _df, _r - 1)
    PAWN_ATTACKS[WHITE][_sq] = _w
    PAWN_ATTACKS[BLACK][_sq] = _b


# --------------------------------------------------------------------------
# Sliding attack rays
# --------------------------------------------------------------------------

# Direction order: N, NE, E, SE, S, SW, W, NW
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
_POSITIVE = (True, True, True, False, False, False, False, True)

RAYS = [[0] * 64 for _ in range(8)]
for _d, (_df, _dr) in enumerate(_DIRECTIONS):
    for _sq in range(64):
        _f, _r = square_file(_sq), square_rank(_sq)
        _acc = 0
        while True:
            _f += _df
            _r += _dr
            if not (0 <= _f < 8 and 0 <= _r < 8):
                break
            _acc |= 1 << square(_f, _r)
        RAYS[_d][_sq] = _acc

_ROOK_DIRS = (0, 2, 4, 6)
_BISHOP_DIRS = (1, 3, 5, 7)

# Empty-board attack masks, handy for cheap "could ever reach" tests.
ROOK_MASKS = [0] * 64
BISHOP_MASKS = [0] * 64
for _sq in range(64):
    ROOK_MASKS[_sq] = RAYS[0][_sq] | RAYS[2][_sq] | RAYS[4][_sq] | RAYS[6][_sq]
    BISHOP_MASKS[_sq] = RAYS[1][_sq] | RAYS[3][_sq] | RAYS[5][_sq] | RAYS[7][_sq]


def _ray_attacks(sq: int, occ: int, dirs) -> int:
    acc = 0
    for d in dirs:
        ray = RAYS[d][sq]
        blockers = ray & occ
        if blockers:
            nearest = lsb(blockers) if _POSITIVE[d] else msb(blockers)
            ray ^= RAYS[d][nearest]
        acc |= ray
    return acc


def rook_attacks(sq: int, occ: int) -> int:
    return _ray_attacks(sq, occ, _ROOK_DIRS)


def bishop_attacks(sq: int, occ: int) -> int:
    return _ray_attacks(sq, occ, _BISHOP_DIRS)


def queen_attacks(sq: int, occ: int) -> int:
    return _ray_attacks(sq, occ, _ROOK_DIRS) | _ray_attacks(sq, occ, _BISHOP_DIRS)


# Squares strictly between two aligned squares (0 when not aligned).  Used for
# check-evasion masks and pin detection.
BETWEEN = [[0] * 64 for _ in range(64)]
for _a in range(64):
    for _d in range(8):
        _ray = RAYS[_d][_a]
        _bb = _ray
        while _bb:
            _b, _bb = pop_lsb(_bb)
            BETWEEN[_a][_b] = _ray & RAYS[_d ^ 4][_b]

LINE = [[0] * 64 for _ in range(64)]
for _a in range(64):
    for _d in range(8):
        for _b in scan(RAYS[_d][_a]):
            LINE[_a][_b] = (1 << _a) | RAYS[_d][_a] | RAYS[_d ^ 4][_a]
