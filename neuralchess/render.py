"""Render a position as a standalone SVG.

Used for documentation and for eyeballing self-play games without opening a
GUI.  Pieces are drawn as vector shapes rather than Unicode glyphs so the
output looks identical everywhere, including inside a GitHub README.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from .chess.bitboard import PIECE_SYMBOLS
from .chess.board import START_FEN, Board
from .chess.move import move_from, move_to

LIGHT = "#dfe5f1"
DARK = "#6f7ea6"
HIGHLIGHT = "#f2c14e"
WHITE_FILL = "#ffffff"
BLACK_FILL = "#1b2030"
OUTLINE = "#161a24"

# Original piece silhouettes on a 45x45 grid.  Deliberately simple geometric
# shapes rather than a traditional Staunton set: the goal is a rendering that
# is unambiguous at README size and carries no third-party artwork.
_BASE = "M9 39 h27 v-3 q0 -3 -4 -3 h-19 q-4 0 -4 3 z"

_SHAPES: Dict[str, List[str]] = {
    "p": [
        "M22.5 7.5 a5.5 5.5 0 1 1 -0.01 0 z",
        "M17 19 q5.5 3.5 11 0 l2.5 14 h-16 z",
        _BASE,
    ],
    "r": [
        "M11 9 h5.5 v3.5 h4 V9 h4 v3.5 h4 V9 h5.5 v8 h-23 z",
        "M14 17 h17 l2 16 h-21 z",
        _BASE,
    ],
    "n": [
        "M13 33 q0 -13 8 -18 l-3 -3 2.5 -3.5 4 3.5 q8 2 9.5 10 1.2 6 1.2 11 z",
        "M25.5 15.5 a1.4 1.4 0 1 1 -0.01 0 z",
        _BASE,
    ],
    "b": [
        "M22.5 7 q6.5 6.5 6.5 11.5 0 5.5 -6.5 9.5 -6.5 -4 -6.5 -9.5 0 -5 6.5 -11.5 z",
        "M15 29 h15 l1.5 4 h-18 z",
        _BASE,
    ],
    "q": [
        "M10 28 l1.5 -14 5 9.5 3 -13.5 3 13.5 3 -13.5 3 13.5 5 -9.5 1.5 14 z",
        "M12 31 h21 l1 3 h-23 z",
        _BASE,
    ],
    "k": [
        "M21 5 h3 v3 h3 v3 h-3 v4.5 h-3 V11 h-3 V8 h3 z",
        "M13 33 q-3.5 -13 4 -16 3.2 -1.2 5.5 2 2.3 -3.2 5.5 -2 7.5 3 4 16 z",
        _BASE,
    ],
}


def board_svg(board: Board, size: int = 400, flipped: bool = False,
              last_move: Optional[int] = None, coordinates: bool = True,
              caption: str = "") -> str:
    """Render ``board`` as a self-contained SVG string."""
    cell = size / 8.0
    pad = 26 if coordinates else 0
    height = size + (26 if coordinates else 0) + (26 if caption else 0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size + pad}" height="{height}" '
        f'viewBox="0 0 {size + pad} {height}" '
        f'font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{size + pad}" height="{height}" fill="#ffffff"/>',
    ]
    highlight = set()
    if last_move is not None:
        highlight = {move_from(last_move), move_to(last_move)}

    for rank in range(8):
        for file in range(8):
            sq = (rank << 3) | file
            col = file if not flipped else 7 - file
            row = 7 - rank if not flipped else rank
            x, y = pad + col * cell, row * cell
            fill = LIGHT if (file + rank) % 2 else DARK
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" '
                         f'height="{cell:.1f}" fill="{fill}"/>')
            if sq in highlight:
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell:.1f}" '
                             f'height="{cell:.1f}" fill="{HIGHLIGHT}" opacity="0.45"/>')
            code = board.mailbox[sq]
            if code >= 0:
                color, ptype = divmod(code, 6)
                symbol = PIECE_SYMBOLS[ptype]
                scale = cell / 45.0
                fill_color = WHITE_FILL if color == 0 else BLACK_FILL
                shapes = "".join(
                    f'<path d="{d}" fill="{fill_color}" stroke="{OUTLINE}" '
                    f'stroke-width="1.5" stroke-linejoin="round"/>'
                    for d in _SHAPES[symbol]
                )
                parts.append(f'<g transform="translate({x:.1f},{y:.1f}) '
                             f'scale({scale:.4f})">{shapes}</g>')

    if coordinates:
        for file in range(8):
            col = file if not flipped else 7 - file
            parts.append(f'<text x="{pad + col * cell + cell / 2:.1f}" y="{size + 17:.1f}" '
                         f'font-size="12" fill="#5a6072" text-anchor="middle">'
                         f'{"abcdefgh"[file]}</text>')
        for rank in range(8):
            row = 7 - rank if not flipped else rank
            parts.append(f'<text x="{pad - 8:.1f}" y="{row * cell + cell / 2 + 4:.1f}" '
                         f'font-size="12" fill="#5a6072" text-anchor="end">{rank + 1}</text>')
    if caption:
        parts.append(f'<text x="{pad}" y="{size + 43:.1f}" font-size="12" '
                     f'fill="#3a3a42">{caption}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_board_svg(path: str, board: Board, **kwargs) -> str:
    with open(path, "w") as fh:
        fh.write(board_svg(board, **kwargs))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="render a position as SVG")
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument("--moves", nargs="*", default=[], help="UCI moves to play first")
    parser.add_argument("--output", "-o", default="board.svg")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--caption", default="")
    args = parser.parse_args(argv)

    board = Board(args.fen)
    last = None
    for uci in args.moves:
        last = board.push_uci(uci)
    write_board_svg(args.output, board, size=args.size, flipped=args.flip,
                    last_move=last, caption=args.caption)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
