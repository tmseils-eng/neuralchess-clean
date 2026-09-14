"""Local web server for the play-against-the-engine UI.

Deliberately built on :mod:`http.server` so that the whole project keeps a
zero-dependency install path.  The browser owns the game state as a list of
UCI moves; every request replays it, which costs microseconds and removes any
possibility of the client and server disagreeing about the position.

Endpoints
---------
``GET  /``            the single-page UI
``POST /api/state``   legal moves and status for a move list
``POST /api/move``    validate and apply a human move
``POST /api/engine``  run a search and return the move plus its statistics
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

import numpy as np

from . import __version__
from .chess.board import START_FEN, Board
from .chess.move import move_uci
from .chess.pgn import to_pgn
from .encoding.planes import PositionHistory
from .evaluation.baselines import evaluate_material
from .nn.modules import NetConfig, PolicyValueNet
from .search.evaluator import Evaluator
from .search.mcts import MCTS, MCTSConfig
from .uci import ENGINE_NAME, value_to_centipawns

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class EngineService:
    """Thread-safe wrapper around a network and a search."""

    def __init__(self, checkpoint: Optional[str] = None, simulations: int = 300):
        self.lock = threading.Lock()
        self.checkpoint = checkpoint
        self.model = (PolicyValueNet.load(checkpoint) if checkpoint
                      else PolicyValueNet(NetConfig(channels=32, blocks=4)))
        self.model.eval()
        self.evaluator = Evaluator(self.model, cache_size=100_000)
        self.simulations = simulations

    def replay(self, moves: List[str]):
        board = Board(START_FEN)
        history = PositionHistory(board)
        for uci in moves:
            board.push_uci(uci)
            history.push(board)
        return board, history

    def san_history(self, moves: List[str]) -> List[str]:
        board = Board(START_FEN)
        out = []
        for uci in moves:
            move = board.parse_uci(uci)
            out.append(board.san(move))
            board.push(move)
        return out

    def state(self, moves: List[str]) -> Dict[str, Any]:
        board, _ = self.replay(moves)
        legal = board.legal_moves()
        outcome = board.outcome()
        return {
            "fen": board.fen(),
            "turn": "white" if board.turn == 0 else "black",
            "legal": [move_uci(m) for m in legal],
            "check": board.is_check(),
            "outcome": outcome,
            "reason": self._reason(board, outcome),
            "material": evaluate_material(board) * (1 if board.turn == 0 else -1),
            "moveNumber": board.fullmove_number,
            "san": self.san_history(moves),
        }

    @staticmethod
    def _reason(board: Board, outcome) -> Optional[str]:
        if outcome is None:
            return None
        if not board.legal_moves():
            return "checkmate" if board.is_check() else "stalemate"
        if board.is_repetition():
            return "threefold repetition"
        if board.is_fifty_moves():
            return "fifty-move rule"
        return "insufficient material"

    def meta(self) -> Dict[str, Any]:
        """Engine identity, shown in the header instead of invented numbers."""
        config = self.model.config
        return {
            "name": ENGINE_NAME,
            "version": __version__,
            "checkpoint": os.path.basename(self.checkpoint) if self.checkpoint else None,
            "trained": bool(self.checkpoint),
            "blocks": config.blocks,
            "channels": config.channels,
            "squeezeExcitation": config.use_se,
            "valueHead": "WDL" if config.wdl else "scalar",
            "historyLength": config.history_length,
            "inputPlanes": config.input_planes,
            "parameters": self.model.num_parameters(),
            "defaultSimulations": self.simulations,
        }

    def engine_move(self, moves: List[str], simulations: Optional[int] = None,
                    temperature: float = 0.0) -> Dict[str, Any]:
        board, history = self.replay(moves)
        if board.terminal_value() is not None:
            return {"bestmove": None, "outcome": board.outcome()}
        sims = simulations or self.simulations
        with self.lock:
            config = MCTSConfig(simulations=sims, batch_size=16)
            search = MCTS(self.evaluator, config, rng=np.random.default_rng())
            started = time.perf_counter()
            root = search.run(board, history.frames(), sims)
            elapsed = max(time.perf_counter() - started, 1e-6)
            # predict_wdl flips the shared model between train/eval mode, so it
            # has to stay inside the lock: two browser tabs would otherwise
            # race on it.
            wdl = self.model.predict_wdl(
                history.encode(self.model.config.history_length)[None])[0]
        order = np.argsort(-root.child_visits)[:6]
        total_visits = max(float(root.child_visits.sum()), 1.0)
        top = [
            {
                "uci": move_uci(root.moves[i]),
                "san": board.san(root.moves[i]),
                "visits": int(root.child_visits[i]),
                "share": float(root.child_visits[i] / total_visits),
                "q": float(root.child_values[i] / max(root.child_visits[i], 1.0)),
                "prior": float(root.priors[i]),
            }
            for i in order if root.child_visits[i] > 0
        ]
        if temperature > 1e-3:
            policy = root.visit_policy(temperature)
            choice = int(np.random.default_rng().choice(len(policy), p=policy))
            best = root.moves[choice]
        else:
            best = root.best_move()
        pv = root.principal_variation(10)
        return {
            "bestmove": move_uci(best),
            "bestmoveSan": board.san(best),
            "value": root.value,
            "cp": value_to_centipawns(root.value),
            "visits": int(root.visits),
            "nodes": int(root.visits),
            "depth": len(pv),
            "elapsedMs": round(elapsed * 1000),
            "nps": int(root.visits / elapsed),
            "wdl": {"win": float(wdl[0]), "draw": float(wdl[1]), "loss": float(wdl[2])},
            "cacheHitRate": float(self.evaluator.stats()["hit_rate"]),
            "top": top,
            "pv": [move_uci(m) for m in pv],
            "pvSan": board.san_line(pv),
        }


def resolve_static(path: str) -> Optional[str]:
    """Map a request path to a file inside ``web/``, or ``None`` if it escapes.

    Pulled out of the handler so the traversal rule is unit-testable without
    opening a socket.
    """
    relative = "index.html" if path in ("/", "") else path.lstrip("/")
    full = os.path.normpath(os.path.join(WEB_DIR, relative))
    # startswith() alone would also accept a sibling directory whose name
    # merely begins with WEB_DIR; compare path components instead.
    if os.path.commonpath([full, WEB_DIR]) != WEB_DIR:
        return None
    return full if os.path.isfile(full) else None


class Handler(BaseHTTPRequestHandler):
    service: EngineService = None  # type: ignore[assignment]

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/api/meta":
            self._send(self.service.meta())
            return
        full = resolve_static(self.path.split("?", 1)[0])
        if full is None:
            self.send_error(404)
            return
        ctype = "text/html" if full.endswith(".html") else "application/octet-stream"
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send({"error": "bad json"}, 400)
            return
        moves = payload.get("moves", [])
        try:
            if self.path == "/api/state":
                self._send(self.service.state(moves))
            elif self.path == "/api/move":
                board, _ = self.service.replay(moves)
                board.parse_uci(payload["uci"])  # validates
                self._send(self.service.state(moves + [payload["uci"]]))
            elif self.path == "/api/pgn":
                board = Board(START_FEN)
                parsed = []
                for uci in moves:
                    move = board.parse_uci(uci)
                    parsed.append(move)
                    board.push(move)
                outcome = board.outcome() or "*"
                self._send({"pgn": to_pgn(parsed, outcome,
                                          payload.get("headers"))})
            elif self.path == "/api/meta":
                self._send(self.service.meta())
            elif self.path == "/api/engine":
                self._send(self.service.engine_move(
                    moves, payload.get("simulations"), payload.get("temperature", 0.0)))
            else:
                self._send({"error": "unknown endpoint"}, 404)
        except (ValueError, KeyError) as exc:
            self._send({"error": str(exc)}, 400)


def serve(checkpoint: Optional[str] = None, port: int = 8000,
          simulations: int = 300, open_browser: bool = False) -> None:
    Handler.service = EngineService(checkpoint, simulations)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"neuralchess UI on {url}  (checkpoint: {checkpoint or 'random weights'})")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Play against neuralchess in a browser")
    parser.add_argument("--checkpoint")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--open", action="store_true", help="open a browser window")
    args = parser.parse_args(argv)
    serve(args.checkpoint, args.port, args.simulations, args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
