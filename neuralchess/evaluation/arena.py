"""Head-to-head matches between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..chess.board import START_FEN, Board
from ..chess.move import move_uci
from ..encoding.planes import PositionHistory
from .baselines import material_balance
from .elo import MatchRecord, elo_difference, likelihood_of_superiority


@dataclass
class GameLog:
    white: str
    black: str
    result: str
    plies: int
    reason: str
    moves: List[str] = field(default_factory=list)
    start_fen: str = START_FEN


def _random_opening(rng: np.random.Generator, plies: int) -> str:
    """A short random opening so that repeated matches are not identical."""
    board = Board()
    for _ in range(plies):
        moves = board.legal_moves()
        if not moves:
            break
        board.push(int(rng.choice(moves)))
    return board.fen()


def play_pair(agent_a, agent_b, start_fen: str = START_FEN,
              max_moves: int = 300, adjudicate_margin: Optional[int] = None) -> GameLog:
    """One game with ``agent_a`` as white.

    ``adjudicate_margin`` (in centipawns) awards a game that hits the move
    limit to whichever side is ahead by at least that much material.  This is
    standard practice in engine testing and it matters here: a weakly trained
    network often reaches a completely winning position and then fails to
    force mate before the limit, which would otherwise be scored as a draw and
    understate its strength.  Leave it ``None`` to score strictly by the rules.
    """
    board = Board(start_fen)
    history = PositionHistory(board)
    agent_a.reset()
    agent_b.reset()
    moves: List[str] = []
    for _ in range(max_moves):
        if board.terminal_value() is not None:
            break
        agent = agent_a if board.turn == 0 else agent_b
        move = agent.select_move(board, history)
        if move is None:
            break
        moves.append(move_uci(move))
        board.push(move)
        history.push(board)
    outcome = board.outcome()
    if outcome is None:
        outcome, reason = "1/2-1/2", "move limit"
        if adjudicate_margin is not None:
            white_score = material_balance(board)
            if white_score >= adjudicate_margin:
                outcome, reason = "1-0", "adjudicated on material"
            elif white_score <= -adjudicate_margin:
                outcome, reason = "0-1", "adjudicated on material"
    elif not board.legal_moves():
        reason = "checkmate" if board.is_check() else "stalemate"
    elif board.is_repetition():
        reason = "threefold repetition"
    elif board.is_fifty_moves():
        reason = "fifty-move rule"
    else:
        reason = "insufficient material"
    return GameLog(getattr(agent_a, "name", "A"), getattr(agent_b, "name", "B"),
                   outcome, len(moves), reason, moves, start_fen)


def match(agent_a, agent_b, games: int = 20, seed: int = 0,
          opening_plies: int = 2, max_moves: int = 300,
          adjudicate_margin: Optional[int] = None,
          progress: Optional[Callable[[int, MatchRecord], None]] = None
          ) -> Tuple[MatchRecord, List[GameLog]]:
    """Play ``games`` games with alternating colours and paired openings.

    Colours alternate and each opening is played twice - once from each side -
    so that an opening advantage cannot bias the result.
    """
    rng = np.random.default_rng(seed)
    name_a = getattr(agent_a, "name", "A")
    name_b = getattr(agent_b, "name", "B")
    record = MatchRecord(name_a, name_b)
    logs: List[GameLog] = []

    pairs = (games + 1) // 2
    for _pair in range(pairs):
        fen = _random_opening(rng, opening_plies) if opening_plies else START_FEN
        for flip in (False, True):
            if len(logs) >= games:
                break
            white, black = (agent_b, agent_a) if flip else (agent_a, agent_b)
            log = play_pair(white, black, fen, max_moves, adjudicate_margin)
            logs.append(log)
            a_is_white = not flip
            if log.result == "1/2-1/2":
                record.draws += 1
            elif (log.result == "1-0") == a_is_white:
                record.wins += 1
            else:
                record.losses += 1
            if progress:
                progress(len(logs), record)
    return record, logs


def match_summary(record: MatchRecord) -> Dict[str, float]:
    elo, lo, hi = elo_difference(record.score, record.games)
    return {
        "games": record.games,
        "wins": record.wins,
        "draws": record.draws,
        "losses": record.losses,
        "score": record.score,
        "elo": elo,
        "elo_low": lo,
        "elo_high": hi,
        "los": likelihood_of_superiority(record.score, record.games),
    }


def round_robin(agents: Sequence, games_per_pair: int = 10, seed: int = 0,
                opening_plies: int = 2, max_moves: int = 300,
                adjudicate_margin: Optional[int] = None,
                progress: Optional[Callable[[str], None]] = None
                ) -> Tuple[List[MatchRecord], List[GameLog]]:
    """Every agent against every other; feed the result to :func:`fit_elo`."""
    records: List[MatchRecord] = []
    logs: List[GameLog] = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if progress:
                progress(f"{getattr(agents[i], 'name', i)} vs {getattr(agents[j], 'name', j)}")
            record, game_logs = match(agents[i], agents[j], games_per_pair,
                                      seed=seed + 1000 * i + j,
                                      opening_plies=opening_plies,
                                      max_moves=max_moves,
                                      adjudicate_margin=adjudicate_margin)
            records.append(record)
            logs.extend(game_logs)
    return records, logs
