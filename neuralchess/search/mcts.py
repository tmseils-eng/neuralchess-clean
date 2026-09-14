"""Batched PUCT Monte-Carlo tree search.

The search follows AlphaZero: each edge stores a prior from the policy head
and an action value averaged from the value head, and selection maximises

.. math:: Q(s,a) + c_{puct}(s) \\, P(s,a) \\frac{\\sqrt{N(s)}}{1 + N(s,a)}

with the exploration constant growing slowly with the parent visit count,

.. math:: c_{puct}(s) = \\log\\frac{1 + N(s) + c_{base}}{c_{base}} + c_{init}

Three refinements matter in practice and are all implemented here:

**Batched leaf collection.**  A pure-NumPy network is dominated by per-call
overhead, so the search gathers several leaves before evaluating.  Paths are
kept apart by a *virtual loss* that temporarily makes an in-flight edge look
bad, exactly as in the distributed AlphaZero setup.

**First-play urgency.**  An unvisited child is optimistically valued at the
parent's own value minus a penalty that grows with how much prior mass has
already been explored.  Without it a fresh network wastes most of its
simulations sampling every legal move once.

**Edge-array storage.**  Children live in NumPy arrays on the parent rather
than in per-child objects, so selecting a move is one vectorised argmax
instead of a Python loop over up to 218 children.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..chess.board import Board
from ..chess.move import move_uci
from ..encoding.planes import HISTORY_LENGTH, Frame, encode_frames
from ..encoding.policy_map import move_to_index


@dataclass
class MCTSConfig:
    """Search hyper-parameters."""

    simulations: int = 200
    c_puct_init: float = 1.25
    c_puct_base: float = 19652.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    fpu_reduction: float = 0.25
    virtual_loss: float = 1.0
    batch_size: int = 16
    policy_softmax_temp: float = 1.0
    max_batch_collisions: int = 8


def _mark_terminal(node: "Node", value: float) -> None:
    """Freeze a node at a proven game result; it is never evaluated again."""
    node.terminal_value = value
    node.expanded = True
    node.priors = np.zeros(0, dtype=np.float32)
    node.child_visits = np.zeros(0, dtype=np.float32)
    node.child_values = np.zeros(0, dtype=np.float32)


class Node:
    """One position in the tree; children are stored as parallel arrays."""

    __slots__ = (
        "moves", "priors", "child_visits", "child_values", "children",
        "visits", "expanded", "terminal_value", "pending",
    )

    def __init__(self):
        self.moves: List[int] = []
        self.priors: Optional[np.ndarray] = None
        self.child_visits: Optional[np.ndarray] = None
        self.child_values: Optional[np.ndarray] = None
        self.children: List[Optional["Node"]] = []
        self.visits: int = 0
        self.expanded: bool = False
        self.terminal_value: Optional[float] = None
        self.pending: bool = False

    # -- properties -----------------------------------------------------
    @property
    def value(self) -> float:
        """Mean action value from this node's mover's point of view."""
        if self.visits == 0 or self.child_values is None:
            return 0.0
        return float(self.child_values.sum() / self.visits)

    def expand(self, moves: Sequence[int], priors: np.ndarray) -> None:
        self.moves = list(moves)
        n = len(self.moves)
        self.priors = priors.astype(np.float32)
        self.child_visits = np.zeros(n, dtype=np.float32)
        self.child_values = np.zeros(n, dtype=np.float32)
        self.children = [None] * n
        self.expanded = True

    def child_q(self, fpu: float) -> np.ndarray:
        visited = self.child_visits > 0
        q = np.where(visited, self.child_values / np.maximum(self.child_visits, 1.0), fpu)
        return q

    def select(self, config: MCTSConfig) -> int:
        c_puct = (
            math.log((1.0 + self.visits + config.c_puct_base) / config.c_puct_base)
            + config.c_puct_init
        )
        explored_prior = float(self.priors[self.child_visits > 0].sum())
        fpu = self.value - config.fpu_reduction * math.sqrt(max(explored_prior, 0.0))
        q = self.child_q(fpu)
        u = c_puct * self.priors * math.sqrt(max(self.visits, 1)) / (1.0 + self.child_visits)
        return int(np.argmax(q + u))

    def visit_policy(self, temperature: float = 1.0) -> np.ndarray:
        """Normalised search policy over ``self.moves``."""
        counts = self.child_visits.astype(np.float64)
        if counts.sum() <= 0:
            return np.full(len(self.moves), 1.0 / max(1, len(self.moves)))
        if temperature <= 1e-3:
            out = np.zeros_like(counts)
            out[int(np.argmax(counts))] = 1.0
            return out
        scaled = counts ** (1.0 / temperature)
        return scaled / scaled.sum()

    def best_move(self) -> int:
        return self.moves[int(np.argmax(self.child_visits))]

    def principal_variation(self, depth: int = 8) -> List[int]:
        pv, node = [], self
        while node is not None and node.expanded and node.child_visits is not None \
                and node.child_visits.sum() > 0 and len(pv) < depth:
            i = int(np.argmax(node.child_visits))
            pv.append(node.moves[i])
            node = node.children[i]
        return pv


class MCTS:
    """Monte-Carlo tree search driven by a policy-value network."""

    def __init__(self, evaluator, config: Optional[MCTSConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()
        self.history_length = getattr(evaluator, "history_length", HISTORY_LENGTH)
        self.nodes_created = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, board: Board, frames: Sequence[Frame], simulations: Optional[int] = None,
            root: Optional[Node] = None, add_noise: bool = False) -> Node:
        """Search from ``board`` and return the populated root node."""
        sims = simulations if simulations is not None else self.config.simulations
        work = board.copy()
        base_frames = list(frames)

        if root is None:
            root = Node()
            self.nodes_created += 1
        if not root.expanded:
            self._expand_root(root, work, base_frames)
        if root.terminal_value is not None:
            return root
        if add_noise:
            self._add_dirichlet_noise(root)

        while root.visits < sims:
            budget = min(self.config.batch_size, sims - root.visits)
            before = root.visits
            leaves = self._collect(root, work, base_frames, budget)
            if leaves:
                self._evaluate_and_backup(leaves)
            if root.visits == before:
                # Nothing progressed (every path collided); further spinning
                # would not change the tree.
                break
        return root

    def search_move(self, board: Board, frames: Sequence[Frame],
                    simulations: Optional[int] = None, temperature: float = 0.0):
        """Convenience wrapper returning ``(move, root)``."""
        root = self.run(board, frames, simulations)
        policy = root.visit_policy(temperature)
        if temperature <= 1e-3:
            index = int(np.argmax(policy))
        else:
            index = int(self.rng.choice(len(policy), p=policy))
        return root.moves[index], root

    @staticmethod
    def advance(root: Optional[Node], move: int) -> Optional[Node]:
        """Reuse the subtree that follows ``move`` after it is played."""
        if root is None or not root.expanded:
            return None
        for i, m in enumerate(root.moves):
            if m == move:
                child = root.children[i]
                if child is not None:
                    child.pending = False
                return child
        return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _priors_for(self, moves: Sequence[int], flip: bool,
                    logits: np.ndarray) -> np.ndarray:
        """Softmax of the policy head restricted to the legal moves."""
        idx = np.fromiter((move_to_index(m, flip) for m in moves), dtype=np.int64,
                          count=len(moves))
        selected = logits[idx] / self.config.policy_softmax_temp
        selected -= selected.max()
        priors = np.exp(selected)
        total = priors.sum()
        if total <= 0:
            return np.full(len(moves), 1.0 / max(1, len(moves)), dtype=np.float32)
        return (priors / total).astype(np.float32)

    def _expand_root(self, root: Node, board: Board, frames: Sequence[Frame]) -> None:
        moves = board.legal_moves()
        terminal = board.terminal_value_with_moves(moves)
        if terminal is not None:
            _mark_terminal(root, terminal)
            return
        key = board.zobrist
        cached = self.evaluator.lookup(key)
        if cached is None:
            logits, values = self.evaluator.evaluate_batch(
                [encode_frames(list(frames), self.history_length)])
            priors = self._priors_for(moves, board.turn == 1, logits[0])
            self.evaluator.store(key, moves, priors, float(values[0]))
        else:
            moves, priors, _value = cached
        root.expand(moves, priors)

    def _add_dirichlet_noise(self, root: Node) -> None:
        n = len(root.moves)
        if n == 0:
            return
        noise = self.rng.dirichlet([self.config.dirichlet_alpha] * n).astype(np.float32)
        eps = self.config.dirichlet_epsilon
        root.priors = (1.0 - eps) * root.priors + eps * noise

    def _collect(self, root: Node, board: Board, base_frames: Sequence[Frame], budget: int):
        """Descend ``budget`` times, applying virtual loss to keep paths apart."""
        leaves = []
        collisions = 0
        done = 0  # simulations resolved this batch, whether or not they need the net
        vl = self.config.virtual_loss
        while done < budget and collisions <= self.config.max_batch_collisions:
            node = root
            path: List[Tuple[Node, int]] = []
            frames = list(base_frames)
            depth = 0

            while True:
                if node.terminal_value is not None:
                    break
                if not node.expanded:
                    break
                i = node.select(self.config)
                path.append((node, i))
                node.child_visits[i] += vl
                node.child_values[i] -= vl
                board.push(node.moves[i])
                frames.append(Frame.of(board))
                depth += 1
                child = node.children[i]
                if child is None:
                    child = Node()
                    self.nodes_created += 1
                    node.children[i] = child
                node = child
                if not node.expanded and node.terminal_value is None:
                    break

            if node.terminal_value is None and not node.expanded:
                # One legal-move generation serves both the terminal test and
                # (if the position is live) the expansion that follows.
                moves = board.legal_moves()
                terminal = board.terminal_value_with_moves(moves)
                if terminal is not None:
                    _mark_terminal(node, terminal)
            else:
                moves = None

            if node.terminal_value is not None:
                # A proven result needs no network call; it resolves immediately.
                self._backup(path, node.terminal_value, vl)
                done += 1
                for _ in range(depth):
                    board.pop()
                continue

            if node.pending:
                collisions += 1
                self._undo_virtual(path, vl)
                for _ in range(depth):
                    board.pop()
                continue

            key = board.zobrist
            cached = self.evaluator.lookup(key)
            if cached is not None:
                cached_moves, priors, value = cached
                node.expand(cached_moves, priors)
                self._backup(path, value, vl)
                done += 1
                for _ in range(depth):
                    board.pop()
                continue

            node.pending = True
            done += 1
            leaves.append((node, path, moves, board.turn == 1,
                           encode_frames(frames, self.history_length), key))
            for _ in range(depth):
                board.pop()
        return leaves

    def _evaluate_and_backup(self, leaves) -> None:
        planes = [leaf[4] for leaf in leaves]
        logits, values = self.evaluator.evaluate_batch(planes)
        vl = self.config.virtual_loss
        for k, (node, path, moves, flip, _planes, key) in enumerate(leaves):
            value = float(values[k])
            priors = self._priors_for(moves, flip, logits[k])
            self.evaluator.store(key, moves, priors, value)
            if not node.expanded:
                node.expand(moves, priors)
            node.pending = False
            self._backup(path, value, vl)

    @staticmethod
    def _backup(path: Sequence[Tuple[Node, int]], value: float, vl: float) -> None:
        v = value
        for node, i in reversed(path):
            node.child_visits[i] -= vl
            node.child_values[i] += vl
            v = -v
            node.child_visits[i] += 1.0
            node.child_values[i] += v
            node.visits += 1
        if not path:
            return

    @staticmethod
    def _undo_virtual(path: Sequence[Tuple[Node, int]], vl: float) -> None:
        for node, i in path:
            node.child_visits[i] -= vl
            node.child_values[i] += vl


def describe_root(root: Node, top: int = 6) -> str:
    """Readable summary of the root's move distribution (for logs and UCI)."""
    if not root.expanded or not root.moves:
        return "(no search)"
    order = np.argsort(-root.child_visits)[:top]
    parts = []
    for i in order:
        visits = int(root.child_visits[i])
        q = float(root.child_values[i] / max(root.child_visits[i], 1.0))
        parts.append(f"{move_uci(root.moves[i])} N={visits} Q={q:+.3f} P={root.priors[i]:.3f}")
    return " | ".join(parts)
