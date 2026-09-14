"""Board <-> tensor encoding for the policy-value network."""

from .planes import (
    BOARD_SIZE,
    EXTRA_PLANES,
    HISTORY_LENGTH,
    INPUT_PLANES,
    PLANES_PER_STEP,
    Frame,
    PositionHistory,
    encode_frames,
    encode_position,
    input_planes,
    legal_move_mask,
)
from .policy_map import (
    NUM_MOVE_PLANES,
    POLICY_SIZE,
    index_to_move,
    index_to_move_squares,
    mirror_square,
    move_to_index,
)

__all__ = [
    "INPUT_PLANES", "HISTORY_LENGTH", "PLANES_PER_STEP", "EXTRA_PLANES",
    "BOARD_SIZE", "input_planes",
    "Frame", "encode_frames", "encode_position", "PositionHistory", "legal_move_mask",
    "POLICY_SIZE", "NUM_MOVE_PLANES", "move_to_index", "index_to_move",
    "index_to_move_squares", "mirror_square",
]
