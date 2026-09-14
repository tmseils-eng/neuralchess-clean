"""UCI protocol and the web API."""

from neuralchess.chess import Board, move_uci
from neuralchess.server import EngineService
from neuralchess.uci import UCIEngine, value_to_centipawns


class CapturingEngine(UCIEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lines = []

    def send(self, line):
        self.lines.append(line)


def run_commands(commands, simulations=40):
    engine = CapturingEngine(simulations=simulations)
    for command in commands:
        engine.handle(command)
    return engine


def test_uci_handshake():
    engine = run_commands(["uci", "isready"])
    assert "uciok" in engine.lines
    assert "readyok" in engine.lines
    assert any(line.startswith("id name") for line in engine.lines)


def test_uci_returns_a_legal_move():
    engine = run_commands(["ucinewgame", "position startpos moves e2e4 e7e5", "go nodes 40"])
    bestmove = [line for line in engine.lines if line.startswith("bestmove")][-1].split()[1]
    board = Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert bestmove in {move_uci(m) for m in board.legal_moves()}
    assert any(line.startswith("info depth") for line in engine.lines)


def test_uci_accepts_fen_positions():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    engine = run_commands([f"position fen {fen}", "go nodes 200"])
    assert [line for line in engine.lines if line.startswith("bestmove")][-1] == "bestmove a1a8"


def test_uci_option_parsing():
    engine = run_commands(["setoption name Simulations value 123"])
    assert engine.options["Simulations"] == 123


def test_centipawn_mapping_is_monotone():
    values = [-0.9, -0.4, 0.0, 0.4, 0.9]
    scores = [value_to_centipawns(v) for v in values]
    assert scores == sorted(scores)
    assert value_to_centipawns(0.0) == 0


def test_web_api_endpoints():
    service = EngineService(simulations=30)
    state = service.state(["e2e4", "e7e5"])
    assert state["turn"] == "white"
    assert "g1f3" in state["legal"]
    assert state["san"] == ["e4", "e5"]
    assert state["outcome"] is None

    result = service.engine_move(["e2e4", "e7e5"], simulations=30)
    assert result["bestmove"] in state["legal"]
    assert result["visits"] > 0
    assert len(result["top"]) > 0

    finished = service.state(["f2f3", "e7e5", "g2g4", "d8h4"])
    assert finished["outcome"] == "0-1"
    assert finished["reason"] == "checkmate"


def test_static_files_cannot_escape_the_web_root():
    """A request path must never resolve outside web/, prefix tricks included."""
    import os

    from neuralchess.server import WEB_DIR, resolve_static

    assert resolve_static("/") == os.path.join(WEB_DIR, "index.html")
    assert resolve_static("/index.html") is not None
    for hostile in ("/../server.py", "/../../etc/passwd", "/..%2fserver.py",
                    "/../web_secrets/key", "/subdir/../../cli.py"):
        assert resolve_static(hostile) is None, hostile


def test_engine_reports_real_telemetry_not_placeholders():
    service = EngineService(simulations=40)
    result = service.engine_move(["d2d4", "d7d5"], simulations=40)
    assert result["nodes"] == result["visits"] == 40
    assert result["elapsedMs"] > 0 and result["nps"] > 0
    assert result["depth"] == len(result["pv"]) == len(result["pvSan"])
    assert abs(sum(result["wdl"].values()) - 1.0) < 1e-4
    assert result["bestmoveSan"] and not result["bestmoveSan"][0].isdigit()
    shares = [t["share"] for t in result["top"]]
    assert all(0 < s <= 1 for s in shares)
