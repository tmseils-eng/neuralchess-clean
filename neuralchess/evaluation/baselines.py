"""Opponents to measure against.

A learned engine is only as interesting as the ladder it is measured on, so
the arena ships four reference players spanning roughly 1200 Elo of strength:
a random mover, a one-ply material grabber, a classical alpha-beta searcher
with piece-square tables, and the network's own raw policy with no search at
all.  The last one is the most informative: comparing it against the full
MCTS agent isolates how much strength the *search* contributes on top of the
*network*.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..chess.bitboard import BLACK, WHITE, popcount, scan
from ..chess.board import Board
from ..encoding.planes import PositionHistory
from ..encoding.policy_map import move_to_index
from ..search.mcts import MCTS, MCTSConfig

PIECE_VALUES = (100, 320, 330, 500, 900, 0)

# Piece-square tables, white's point of view, a1 first.  Standard simplified
# evaluation values; they give the classical baseline a sense of development
# and king safety without any tuning of our own.
_PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
_PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
_PST_ROOK = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
_PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
_PST_KING = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
]
PST = (_PST_PAWN, _PST_KNIGHT, _PST_BISHOP, _PST_ROOK, _PST_QUEEN, _PST_KING)

MATE_SCORE = 100_000


def material_balance(board: Board) -> int:
    """Pure material count in centipawns, from white's point of view.

    Deliberately excludes the piece-square tables: this is the quantity used
    to adjudicate indecisive games, where a positional bonus tipping the
    verdict would be indefensible.
    """
    score = 0
    for ptype in range(5):
        value = PIECE_VALUES[ptype]
        score += value * popcount(board.pieces[WHITE][ptype])
        score -= value * popcount(board.pieces[BLACK][ptype])
    return score


def evaluate_material(board: Board) -> int:
    """Static evaluation in centipawns, from the side to move's point of view."""
    score = 0
    for ptype in range(6):
        table = PST[ptype]
        value = PIECE_VALUES[ptype]
        for sq in scan(board.pieces[WHITE][ptype]):
            score += value + table[sq]
        for sq in scan(board.pieces[BLACK][ptype]):
            score -= value + table[sq ^ 56]
    return score if board.turn == WHITE else -score


class Agent:
    """Common interface used by the arena and the UCI adapter."""

    name = "agent"

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        moves = board.legal_moves()
        return int(self.rng.choice(moves)) if moves else None


class MaterialAgent(Agent):
    """One-ply greedy: take the move with the best immediate static score.

    ``noise`` adds Gaussian error (in centipawns) to each candidate score,
    which turns the same agent into a weaker, blundering version of itself.
    That is the cheapest way to put an extra rung on the strength ladder
    between random play and clean material evaluation.
    """

    def __init__(self, seed: int = 0, noise: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.name = "material" if noise <= 0 else f"material-noisy{int(noise)}"

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        moves = board.legal_moves()
        if not moves:
            return None
        best, best_score = [], -10 ** 9
        for move in moves:
            board.push(move)
            score = -evaluate_material(board)
            if board.terminal_value() == -1.0:
                score = MATE_SCORE
            board.pop()
            if self.noise:
                score += self.rng.normal(0, self.noise)
            if score > best_score:
                best, best_score = [move], score
            elif score == best_score:
                best.append(move)
        return int(self.rng.choice(best))


class AlphaBetaAgent(Agent):
    """Classical negamax with alpha-beta pruning and MVV-LVA move ordering."""

    def __init__(self, depth: int = 2, seed: int = 0):
        self.depth = depth
        self.name = f"alphabeta-d{depth}"
        self.rng = np.random.default_rng(seed)
        self.nodes = 0

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        moves = self._ordered(board)
        if not moves:
            return None
        self.nodes = 0
        best_score, best_moves = -MATE_SCORE * 2, []
        alpha, beta = -MATE_SCORE * 2, MATE_SCORE * 2
        for move in moves:
            board.push(move)
            score = -self._search(board, self.depth - 1, -beta, -alpha)
            board.pop()
            if score > best_score:
                best_score, best_moves = score, [move]
            elif score == best_score:
                best_moves.append(move)
            alpha = max(alpha, score)
        return int(self.rng.choice(best_moves))

    def _search(self, board: Board, depth: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        moves = self._ordered(board)
        if not moves:
            return -MATE_SCORE - depth if board.is_check() else 0
        if board.is_fifty_moves() or board.is_repetition() or board.is_insufficient_material():
            return 0
        if depth <= 0:
            return evaluate_material(board)
        best = -MATE_SCORE * 2
        for move in moves:
            board.push(move)
            score = -self._search(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    @staticmethod
    def _ordered(board: Board) -> List[int]:
        from ..chess.move import move_from, move_to

        moves = board.legal_moves()

        def key(move: int) -> int:
            victim = board.mailbox[move_to(move)]
            if victim < 0:
                return 0
            attacker = board.mailbox[move_from(move)] % 6
            return 10 * PIECE_VALUES[victim % 6] - PIECE_VALUES[attacker]

        return sorted(moves, key=key, reverse=True)


class PolicyAgent(Agent):
    """The network's raw policy head, no tree search at all."""

    name = "policy-only"

    def __init__(self, model, temperature: float = 0.0, seed: int = 0):
        self.model = model
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        moves = board.legal_moves()
        if not moves:
            return None
        depth = getattr(self.model.config, 'history_length', None)
        planes = history.encode(depth) if depth else history.encode()
        logits, _ = self.model.predict(planes[None])
        flip = board.turn == BLACK
        idx = [move_to_index(m, flip) for m in moves]
        scores = logits[0][idx]
        if self.temperature <= 1e-3:
            return int(moves[int(np.argmax(scores))])
        scaled = scores / self.temperature
        scaled -= scaled.max()
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(moves[int(self.rng.choice(len(moves), p=probs))])


class MCTSAgent(Agent):
    """The full engine: network priors plus PUCT search."""

    def __init__(self, evaluator, simulations: int = 200,
                 config: Optional[MCTSConfig] = None, temperature: float = 0.0,
                 seed: int = 0, name: Optional[str] = None):
        self.evaluator = evaluator
        self.simulations = simulations
        self.config = config or MCTSConfig(simulations=simulations)
        self.config.simulations = simulations
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.search = MCTS(evaluator, self.config, rng=self.rng)
        self.name = name or f"mcts-{simulations}"
        self.last_root = None

    def select_move(self, board: Board, history: PositionHistory) -> Optional[int]:
        if not board.legal_moves():
            return None
        move, root = self.search.search_move(board, history.frames(),
                                             self.simulations, self.temperature)
        self.last_root = root
        return int(move)

    def reset(self) -> None:
        self.evaluator.clear_cache()
