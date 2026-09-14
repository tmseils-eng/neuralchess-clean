"""Self-contained chess rules: bitboards, move encoding, and a legal-move board."""

from .bitboard import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    PAWN,
    PIECE_SYMBOLS,
    QUEEN,
    ROOK,
    SQUARE_INDEX,
    SQUARE_NAMES,
    WHITE,
    popcount,
    scan,
    square,
    square_file,
    square_rank,
)
from .board import START_FEN, Board, perft, perft_divide
from .move import (
    CASTLE,
    DOUBLE_PUSH,
    EN_PASSANT,
    NORMAL,
    make_move,
    move_flag,
    move_from,
    move_promotion,
    move_to,
    move_uci,
    parse_uci_squares,
)

__all__ = [
    "Board", "START_FEN", "perft", "perft_divide",
    "WHITE", "BLACK", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING",
    "PIECE_SYMBOLS", "SQUARE_NAMES", "SQUARE_INDEX",
    "square", "square_file", "square_rank", "scan", "popcount",
    "make_move", "move_from", "move_to", "move_promotion", "move_flag", "move_uci",
    "parse_uci_squares", "NORMAL", "DOUBLE_PUSH", "EN_PASSANT", "CASTLE",
]
