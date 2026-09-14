"""Minimal PGN export so self-play and arena games can be reviewed anywhere."""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Sequence

from .board import START_FEN, Board


def to_pgn(moves: Sequence[int], result: str = "*", headers: Optional[Dict[str, str]] = None,
           start_fen: str = START_FEN, width: int = 80) -> str:
    """Render a move list as a PGN game."""
    board = Board(start_fen)
    tags = {
        "Event": "neuralchess self-play",
        "Site": "local",
        "Date": datetime.date.today().strftime("%Y.%m.%d"),
        "Round": "-",
        "White": "neuralchess",
        "Black": "neuralchess",
        "Result": result,
    }
    if start_fen != START_FEN:
        tags["FEN"] = start_fen
        tags["SetUp"] = "1"
    tags.update(headers or {})

    tokens: List[str] = []
    for i, move in enumerate(moves):
        if board.turn == 0:
            tokens.append(f"{board.fullmove_number}.")
        elif i == 0:
            tokens.append(f"{board.fullmove_number}...")
        tokens.append(board.san(move))
        board.push(move)
    tokens.append(result)

    lines: List[str] = []
    current = ""
    for token in tokens:
        if len(current) + len(token) + 1 > width:
            lines.append(current)
            current = token
        else:
            current = f"{current} {token}".strip()
    if current:
        lines.append(current)

    header_block = "\n".join(f'[{k} "{v}"]' for k, v in tags.items())
    return header_block + "\n\n" + "\n".join(lines) + "\n"


def write_pgn(path: str, games: Sequence[dict]) -> str:
    """Write several games; each dict needs ``moves`` and ``result`` keys."""
    with open(path, "w") as fh:
        for game in games:
            fh.write(to_pgn(game["moves"], game.get("result", "*"),
                            game.get("headers"), game.get("start_fen", START_FEN)))
            fh.write("\n")
    return path
