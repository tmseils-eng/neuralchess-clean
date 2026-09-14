"""Search behaviour: forced wins, tree bookkeeping, and board integrity."""

import numpy as np

from neuralchess.chess import Board, move_uci
from neuralchess.encoding import PositionHistory
from neuralchess.nn import NetConfig, PolicyValueNet
from neuralchess.search import MCTS, Evaluator, MCTSConfig, RandomEvaluator, describe_root


def mating_moves(fen):
    """Ground truth computed directly from the rules, not from the search."""
    board = Board(fen)
    found = []
    for move in board.legal_moves():
        board.push(move)
        if not board.legal_moves() and board.is_check():
            found.append(move_uci(move))
        board.pop()
    return found


def search_best(fen, simulations=400, evaluator=None, seed=0):
    board = Board(fen)
    history = PositionHistory(board)
    search = MCTS(evaluator or RandomEvaluator(),
                  MCTSConfig(simulations=simulations, batch_size=8),
                  rng=np.random.default_rng(seed))
    root = search.run(board, history.frames())
    assert board.fen() == fen, "search must not mutate the caller's board"
    return move_uci(root.best_move()), root


def test_finds_mate_in_one():
    positions = [
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",       # back rank
        "3r2k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", # rook trade into mate
        "7k/6pp/8/8/8/8/8/4R1K1 w - - 0 1",        # rook to the eighth
    ]
    for fen in positions:
        expected = mating_moves(fen)
        assert expected, f"test position has no mate in one: {fen}"
        best, root = search_best(fen, 400)
        assert best in expected, f"{fen}: played {best}, mates are {expected}"
        assert root.value > 0.5


def test_avoids_hanging_a_queen():
    # White queen is attacked; only capturing the attacker keeps material.
    fen = "7k/8/8/8/8/8/6q1/7K w - - 0 1"
    best, _ = search_best(fen, 200)
    assert best == "h1g2"


def test_visit_policy_is_a_distribution():
    _, root = search_best("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", 200)
    for temperature in (1.0, 0.5, 0.0):
        policy = root.visit_policy(temperature)
        assert abs(policy.sum() - 1.0) < 1e-6
        assert (policy >= 0).all()
    assert len(root.moves) == len(root.child_visits)
    assert int(root.child_visits.sum()) <= root.visits


def test_tree_reuse_keeps_statistics():
    board = Board()
    history = PositionHistory(board)
    search = MCTS(RandomEvaluator(), MCTSConfig(simulations=120, batch_size=8),
                  rng=np.random.default_rng(0))
    root = search.run(board, history.frames())
    move = root.best_move()
    child = MCTS.advance(root, move)
    assert child is not None and child.expanded
    inherited = child.visits
    board.push(move)
    history.push(board)
    root2 = search.run(board, history.frames(), 200, root=child)
    assert root2 is child
    assert root2.visits >= inherited


def test_network_evaluator_runs_and_caches():
    model = PolicyValueNet(NetConfig(channels=16, blocks=1))
    evaluator = Evaluator(model)
    best, root = search_best("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                             150, evaluator)
    assert best in {move_uci(m) for m in Board().legal_moves()}
    stats = evaluator.stats()
    assert stats["positions"] > 0
    assert describe_root(root, 3).count("|") == 2


def test_terminal_root_is_handled():
    fen = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"     # black is checkmated
    board = Board(fen)
    search = MCTS(RandomEvaluator(), MCTSConfig(simulations=50))
    root = search.run(board, PositionHistory(board).frames())
    assert root.terminal_value == -1.0
    assert root.moves == []
