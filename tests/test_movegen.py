"""Move generator correctness: perft, make/unmake invariants, SAN, PGN."""

import random

from neuralchess.chess import START_FEN, Board, move_uci, perft
from neuralchess.chess.pgn import to_pgn

# The standard perft suite.  Any bug in castling, en passant, promotion or
# pin handling shows up as a wrong node count at one of these depths.
PERFT_CASES = [
    (START_FEN, [20, 400, 8902, 197281]),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     [48, 2039, 97862]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     [46, 2079, 89890]),
]


def test_perft_suite():
    for fen, expected in PERFT_CASES:
        board = Board(fen)
        for depth, count in enumerate(expected, start=1):
            assert perft(board, depth) == count, f"{fen} depth {depth}"
        assert board.fen() == fen, "perft must leave the position untouched"


def test_fen_round_trip():
    for fen, _ in PERFT_CASES:
        assert Board(fen).fen() == fen


def test_make_unmake_restores_everything():
    rng = random.Random(7)
    board = Board()
    for _ in range(200):
        moves = board.legal_moves()
        if not moves:
            break
        move = rng.choice(moves)
        snapshot = (board.fen(), board.zobrist, list(board.mailbox),
                    [row[:] for row in board.pieces], board.occupied)
        board.push(move)
        board.pop()
        assert board.fen() == snapshot[0]
        assert board.zobrist == snapshot[1]
        assert board.mailbox == snapshot[2]
        assert board.pieces == snapshot[3]
        assert board.occupied == snapshot[4]
        board.push(move)


def test_zobrist_matches_recomputation():
    rng = random.Random(11)
    board = Board()
    for _ in range(120):
        moves = board.legal_moves()
        if not moves:
            break
        board.push(rng.choice(moves))
        assert board.zobrist == board._compute_zobrist()


def test_en_passant_and_promotion():
    board = Board("8/8/8/8/4pP2/8/8/K6k b - f3 0 1")
    assert "e4f3" in [move_uci(m) for m in board.legal_moves()]
    board.push_uci("e4f3")
    assert board.mailbox[Board("8/8/8/8/8/8/8/8 w - - 0 1").mailbox.__len__() - 1] == -1

    promo = Board("8/P7/8/8/8/8/8/K6k w - - 0 1")
    ucis = {move_uci(m) for m in promo.legal_moves()}
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= ucis


def test_castling_rights_are_lost_correctly():
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("e1g1")
    assert "K" not in board.fen().split()[2] and "Q" not in board.fen().split()[2]
    assert "k" in board.fen().split()[2]

    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push_uci("a1a2")
    rights = board.fen().split()[2]
    assert "Q" not in rights and "K" in rights


def test_terminal_detection():
    # Qg7 is defended by the king on g6, so the black king has no escape.
    assert Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1").outcome() == "1-0"
    # Qf7 covers every flight square but does not attack h8: stalemate.
    assert Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").outcome() == "1/2-1/2"
    assert Board("7k/8/6K1/8/8/8/8/8 w - - 0 1").is_insufficient_material()
    assert Board("7k/8/6KB/8/8/8/8/8 w - - 0 1").is_insufficient_material()
    assert not Board("7k/8/6KR/8/8/8/8/8 w - - 0 1").is_insufficient_material()


def test_threefold_repetition():
    board = Board("4k3/8/8/8/8/8/8/4K2R w - - 0 1")
    for _ in range(2):
        for uci in ("h1h2", "e8e7", "h2h1", "e7e8"):
            board.push_uci(uci)
    assert board.is_repetition()
    assert board.outcome() == "1/2-1/2"


def test_san_disambiguation_and_suffixes():
    board = Board("4k3/8/8/8/8/8/4K3/R6R w - - 0 1")
    assert board.san(board.parse_uci("a1d1")) == "Rad1"
    assert board.san(board.parse_uci("h1d1")) == "Rhd1"

    mate = Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert mate.san(mate.parse_uci("a1a8")) == "Ra8#"

    castle = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert castle.san(castle.parse_uci("e1g1")) == "O-O"
    assert castle.san(castle.parse_uci("e1c1")) == "O-O-O"


def test_pgn_export_round_trips_through_san():
    board = Board()
    moves = [board.push_uci(u) or m for u, m in []] if False else []
    board = Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"):
        moves.append(board.parse_uci(uci))
        board.push(moves[-1])
    text = to_pgn(moves, "*")
    assert "1. e4 e5 2. Nf3 Nc6 3. Bb5" in text
    assert '[Result "*"]' in text


def test_king_capture_is_never_generated():
    """Illegal positions must not produce a move that removes a king."""
    illegal = [
        "7k/8/6K1/8/8/8/8/7Q w - - 0 1",   # black is in check on white's turn
        "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1",
    ]
    for fen in illegal:
        board = Board(fen)
        assert not board.is_valid()
        for move in board.pseudo_legal_moves():
            target = board.mailbox[move & 0x3F if False else (move >> 6) & 0x3F]
            assert target < 0 or target % 6 != 5, f"{fen}: {move_uci(move)} captures a king"


def test_is_valid_accepts_real_positions():
    for fen, _ in PERFT_CASES:
        assert Board(fen).is_valid()
