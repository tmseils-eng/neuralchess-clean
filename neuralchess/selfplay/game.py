"""Self-play game generation.

Each game produces one training sample per recorded position: the encoded
input planes, the MCTS visit distribution as the policy target, and the
eventual game result as the value target.

Two refinements from KataGo are used because they buy a large amount of
learning per unit of compute, which matters when the whole run has to fit on
a laptop:

**Playout-cap randomisation.**  Most moves are played with a small visit
budget and are *not* recorded; a minority get the full budget plus root noise
and are.  Search quality where it matters is preserved while the number of
positions generated per second roughly triples.

**Forced-playout / policy-target pruning is deliberately omitted**, but
resignation with a periodic no-resign audit is included so that the value
head still sees a calibrated fraction of played-out endings.

**Material adjudication of indecisive games.**  A weak network shuffles: in a
first run of this loop, 72% of self-play games ended at the move limit or by
repetition, every value target was therefore 0, and the value head learned
that all positions are drawn.  With no evaluation signal the search collapses
to sampling from the policy, and training the policy on that search's visit
counts just reinforces the network's own arbitrary initial preferences - the
policy entropy fell from 2.34 to 1.68 nats while the network got *weaker*.

The fix is to adjudicate indecisive endings on material, exactly as engine
testing frameworks do.  This shapes the *training target* only; the evaluation
arena still scores games by the rules.  See ``docs/training.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ..chess.board import START_FEN, Board
from ..chess.move import move_uci
from ..encoding.planes import HISTORY_LENGTH, PositionHistory
from ..encoding.policy_map import move_to_index
from ..search.mcts import MCTS, MCTSConfig, Node


@dataclass
class SelfPlayConfig:
    """Everything that controls how a self-play game is produced."""

    simulations: int = 128
    fast_simulations: int = 32
    full_search_probability: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 20
    endgame_temperature: float = 0.15
    max_moves: int = 220
    resign_threshold: float = -0.92
    resign_consecutive: int = 4
    resign_disable_probability: float = 0.1
    adjudicate_margin: int = 300
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    reuse_tree: bool = True
    opening_random_plies: int = 0


@dataclass
class Sample:
    """One training position."""

    planes: np.ndarray                 # (119, 8, 8) float32
    policy_indices: np.ndarray         # int32 indices into the 4672 policy
    policy_probs: np.ndarray           # float32, sums to 1
    value: float = 0.0                 # filled in once the game ends
    ply: int = 0


@dataclass
class GameResult:
    samples: List[Sample] = field(default_factory=list)
    moves: List[int] = field(default_factory=list)
    result: str = "*"
    plies: int = 0
    resigned: bool = False
    adjudicated: bool = False
    terminal_reason: str = ""

    def uci_moves(self) -> List[str]:
        return [move_uci(m) for m in self.moves]


def _terminal_reason(board: Board) -> str:
    if not board.legal_moves():
        return "checkmate" if board.is_check() else "stalemate"
    if board.is_fifty_moves():
        return "fifty-move rule"
    if board.is_repetition():
        return "threefold repetition"
    if board.is_insufficient_material():
        return "insufficient material"
    return "move limit"


def play_game(evaluator, config: Optional[SelfPlayConfig] = None,
              mcts_config: Optional[MCTSConfig] = None,
              rng: Optional[np.random.Generator] = None,
              start_fen: str = START_FEN) -> GameResult:
    """Play one self-play game and return its samples and result."""
    config = config or SelfPlayConfig()
    rng = rng or np.random.default_rng()
    mcts_config = mcts_config or MCTSConfig()
    mcts_config.dirichlet_alpha = config.dirichlet_alpha
    mcts_config.dirichlet_epsilon = config.dirichlet_epsilon

    board = Board(start_fen)
    history_length = getattr(evaluator, "history_length", HISTORY_LENGTH)
    history = PositionHistory(board, window=history_length)
    search = MCTS(evaluator, mcts_config, rng=rng)

    out = GameResult()
    allow_resign = rng.random() > config.resign_disable_probability
    resign_streak = 0
    resign_color: Optional[int] = None
    root: Optional[Node] = None

    for ply in range(config.max_moves):
        if board.terminal_value() is not None:
            break

        if ply < config.opening_random_plies:
            move = rng.choice(board.legal_moves())
            out.moves.append(int(move))
            board.push(int(move))
            history.push(board)
            root = None
            continue

        full = rng.random() < config.full_search_probability
        sims = config.simulations if full else config.fast_simulations
        root = search.run(board, history.frames(), simulations=sims,
                          root=root if config.reuse_tree else None,
                          add_noise=full)
        if not root.moves:
            break

        if full:
            indices = np.fromiter(
                (move_to_index(m, board.turn == 1) for m in root.moves),
                dtype=np.int32, count=len(root.moves),
            )
            probs = root.visit_policy(1.0).astype(np.float32)
            out.samples.append(
                Sample(planes=history.encode(history_length), policy_indices=indices,
                       policy_probs=probs, ply=ply)
            )

        temp = config.temperature if ply < config.temperature_moves else config.endgame_temperature
        policy = root.visit_policy(temp)
        choice = int(rng.choice(len(policy), p=policy)) if temp > 1e-3 else int(np.argmax(policy))
        move = root.moves[choice]

        # Resignation: require the same side to be lost for several plies.
        root_value = root.value
        if allow_resign and full:
            if root_value < config.resign_threshold:
                if resign_color == board.turn:
                    resign_streak += 1
                else:
                    resign_color, resign_streak = board.turn, 1
                if resign_streak >= config.resign_consecutive:
                    out.result = "0-1" if board.turn == 0 else "1-0"
                    out.resigned = True
                    out.terminal_reason = "resignation"
                    break
            else:
                resign_streak, resign_color = 0, None

        out.moves.append(int(move))
        board.push(int(move))
        history.push(board)
        root = MCTS.advance(root, move) if config.reuse_tree else None

    out.plies = len(out.moves)
    if not out.resigned:
        outcome = board.outcome()
        if outcome is None:
            out.result, out.terminal_reason = "1/2-1/2", "move limit"
        else:
            out.result, out.terminal_reason = outcome, _terminal_reason(board)
        _adjudicate(out, board, config.adjudicate_margin)

    _assign_values(out)
    return out


# Draws that reflect the engine's inability to make progress, rather than a
# genuinely drawn position.  Stalemate and insufficient material are real
# draws and are never adjudicated.
_INDECISIVE = ("move limit", "threefold repetition", "fifty-move rule")


def _adjudicate(game: GameResult, board: Board, margin: int) -> None:
    """Award an indecisive ending to whichever side is clearly ahead."""
    if margin <= 0 or game.result != "1/2-1/2":
        return
    if game.terminal_reason not in _INDECISIVE:
        return
    from ..evaluation.baselines import material_balance

    white_score = material_balance(board)
    if white_score >= margin:
        game.result = "1-0"
    elif white_score <= -margin:
        game.result = "0-1"
    else:
        return
    game.adjudicated = True
    game.terminal_reason = f"adjudicated ({game.terminal_reason})"


def _assign_values(game: GameResult) -> None:
    """Propagate the final result back to every sample, sign-flipped by side."""
    if game.result == "1-0":
        white_score = 1.0
    elif game.result == "0-1":
        white_score = -1.0
    else:
        white_score = 0.0
    for sample in game.samples:
        mover_is_white = sample.ply % 2 == 0
        sample.value = white_score if mover_is_white else -white_score


def play_match_game(white_agent, black_agent, start_fen: str = START_FEN,
                    max_moves: int = 300) -> Tuple[str, List[int], str]:
    """Play one game between two agents; used by the evaluation arena."""
    board = Board(start_fen)
    history = PositionHistory(board)
    moves: List[int] = []
    for _ply in range(max_moves):
        if board.terminal_value() is not None:
            break
        agent = white_agent if board.turn == 0 else black_agent
        move = agent.select_move(board, history)
        if move is None:
            break
        moves.append(int(move))
        board.push(int(move))
        history.push(board)
    outcome = board.outcome()
    return (outcome if outcome is not None else "1/2-1/2"), moves, _terminal_reason(board)
