"""Compact move encoding.

A move is a single Python ``int`` so that move lists stay cheap to build and
hash inside the search.  Layout (little end first)::

    bits  0- 5   origin square
    bits  6-11   destination square
    bits 12-14   promotion piece type (0 = none, 1..4 = N,B,R,Q)
    bits 15-17   flag (see below)
"""

from __future__ import annotations

from .bitboard import BISHOP, KNIGHT, QUEEN, ROOK, SQUARE_INDEX, SQUARE_NAMES

NORMAL = 0
DOUBLE_PUSH = 1
EN_PASSANT = 2
CASTLE = 3

PROMO_PIECES = (KNIGHT, BISHOP, ROOK, QUEEN)
_PROMO_TO_CODE = {KNIGHT: 1, BISHOP: 2, ROOK: 3, QUEEN: 4}
_CODE_TO_PROMO = {1: KNIGHT, 2: BISHOP, 3: ROOK, 4: QUEEN}
_PROMO_CHARS = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}
_CHAR_TO_PROMO = {v: k for k, v in _PROMO_CHARS.items()}

NULL_MOVE = 0


def make_move(frm: int, to: int, promo: int | None = None, flag: int = NORMAL) -> int:
    code = 0 if promo is None else _PROMO_TO_CODE[promo]
    return frm | (to << 6) | (code << 12) | (flag << 15)


def move_from(move: int) -> int:
    return move & 0x3F


def move_to(move: int) -> int:
    return (move >> 6) & 0x3F


def move_promotion(move: int):
    code = (move >> 12) & 0x7
    return _CODE_TO_PROMO.get(code)


def move_flag(move: int) -> int:
    return (move >> 15) & 0x7


def move_uci(move: int) -> str:
    """Long algebraic (UCI) notation, e.g. ``e2e4`` or ``a7a8q``."""
    if move == NULL_MOVE:
        return "0000"
    s = SQUARE_NAMES[move_from(move)] + SQUARE_NAMES[move_to(move)]
    promo = move_promotion(move)
    return s + _PROMO_CHARS[promo] if promo is not None else s


def parse_uci_squares(uci: str):
    """Decode a UCI string into ``(from, to, promotion)`` without a board."""
    uci = uci.strip()
    if len(uci) not in (4, 5):
        raise ValueError(f"malformed uci move: {uci!r}")
    frm = SQUARE_INDEX[uci[0:2]]
    to = SQUARE_INDEX[uci[2:4]]
    promo = _CHAR_TO_PROMO[uci[4]] if len(uci) == 5 else None
    return frm, to, promo
