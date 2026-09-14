"""Input planes and the policy-index bijection."""

import random

import numpy as np

from neuralchess.chess import Board, move_from, move_promotion, move_to
from neuralchess.encoding import (
    HISTORY_LENGTH,
    INPUT_PLANES,
    POLICY_SIZE,
    PositionHistory,
    index_to_move,
    legal_move_mask,
    move_to_index,
)
from neuralchess.encoding.policy_map import coverage


def test_input_shape_and_constant_planes():
    board = Board()
    planes = PositionHistory(board).encode()
    assert planes.shape == (INPUT_PLANES, 8, 8) == (119, 8, 8)
    # Start position: 32 pieces + four full castling planes.
    assert planes[:12].sum() == 32
    assert planes[112].mean() == 0.0                    # white to move
    assert all(planes[114 + i].mean() == 1.0 for i in range(4))


def test_encoding_is_from_the_movers_point_of_view():
    board = Board()
    history = PositionHistory(board)
    board.push_uci("e2e4")
    history.push(board)
    planes = history.encode()
    assert planes[112].mean() == 1.0                    # black to move
    # "Our" pawns (black) must appear on row 1 after the vertical flip.
    rows = sorted({int(r) for r in np.argwhere(planes[0] == 1)[:, 0]})
    assert rows == [1]
    # The opponent's e-pawn advanced two squares, so it sits on row 4.
    their_rows = sorted({int(r) for r in np.argwhere(planes[6] == 1)[:, 0]})
    assert their_rows == [4, 6]


def test_history_window_is_bounded():
    board = Board()
    history = PositionHistory(board)
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6",
                "e1g1", "g8f6"):
        board.push_uci(uci)
        history.push(board)
    assert len(history) == HISTORY_LENGTH
    assert history.encode().shape == (INPUT_PLANES, 8, 8)


def test_policy_map_is_injective_on_legal_moves():
    rng = random.Random(3)
    board = Board()
    for _ in range(400):
        moves = board.legal_moves()
        if not moves:
            board = Board()
            continue
        flip = board.turn == 1
        indices = [move_to_index(m, flip) for m in moves]
        assert len(set(indices)) == len(indices)
        for move, index in zip(moves, indices):
            assert 0 <= index < POLICY_SIZE
            recovered = index_to_move(index, moves, flip)
            assert move_from(recovered) == move_from(move)
            assert move_to(recovered) == move_to(move)
            assert move_promotion(recovered) == move_promotion(move)
        board.push(rng.choice(moves))


def test_policy_map_covers_every_geometry():
    # 1880 distinct (from, to, promotion) triples are reachable on 8x8.
    assert coverage() == 1880
    assert POLICY_SIZE == 4672


def test_legal_move_mask_matches_move_list():
    board = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    mask = legal_move_mask(board)
    assert mask.sum() == len(board.legal_moves()) == 48


def test_history_depth_is_a_hyperparameter_not_a_constant():
    from neuralchess.encoding import input_planes

    assert input_planes(8) == 119
    assert input_planes(2) == 35
    assert input_planes(1) == 21

    board = Board()
    history = PositionHistory(board)
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
        history.push(board)

    # One history object can feed networks of different depths: it keeps the
    # deepest window and the encoder slices to what the network was trained on.
    for depth in (8, 4, 2, 1):
        planes = history.encode(depth)
        assert planes.shape == (input_planes(depth), 8, 8)
        # The seven trailing scalar planes are identical at every depth.
        assert planes[input_planes(depth) - 7].mean() == history.encode(8)[112].mean()

    # The most recent step is always step 0, whatever the depth.
    assert np.array_equal(history.encode(2)[:12], history.encode(8)[:12])


def test_shallow_history_model_round_trips_through_a_checkpoint():
    import os
    import tempfile

    from neuralchess.nn import NetConfig, PolicyValueNet

    model = PolicyValueNet(NetConfig(channels=16, blocks=1, history_length=2))
    assert model.config.input_planes == 35
    assert model.stem_conv.weight.shape[1] == 35

    board = Board()
    history = PositionHistory(board)
    planes = history.encode(model.config.history_length)[None]
    before = model.predict(planes)

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "t2.npz")
        model.save(path)
        restored = PolicyValueNet.load(path)
    assert restored.config.history_length == 2
    assert np.allclose(before[0], restored.predict(planes)[0], atol=1e-5)


def test_checkpoints_written_before_the_field_existed_default_to_eight():
    """Backward compatibility: a config dict with no history_length is T=8."""
    from neuralchess.nn import NetConfig

    legacy = {"channels": 48, "blocks": 5, "use_se": True, "wdl": True}
    assert NetConfig.from_dict(legacy).history_length == 8
    assert NetConfig.from_dict(legacy).input_planes == 119
