# Playing against it

Three front ends share one engine.

## Browser UI

```bash
python -m neuralchess serve --checkpoint runs/laptop/checkpoints/best.npz --open
```

A single-page app on `http://127.0.0.1:8000` with click-or-drag movement,
legal-move markers, a promotion picker, an evaluation bar, undo, board flip,
a hint button, and one-click PGN export. The right-hand panel shows what the
search is actually doing: the top moves by visit count, their `Q` values, and
the principal variation. Two sliders control engine strength (simulations per
move) and randomness (sampling temperature).

The server is `http.server` from the standard library — no Flask, no
dependencies. The browser owns the game as a list of UCI moves and the server
replays it on every request, which costs microseconds and makes it impossible
for the two to disagree about the position.

| endpoint | purpose |
| --- | --- |
| `POST /api/state` | legal moves, SAN history, check/outcome for a move list |
| `POST /api/move` | validate and apply a human move |
| `POST /api/engine` | run a search; returns the move, value, top moves, PV |
| `POST /api/pgn` | export the game |

## UCI engine

```bash
python -m neuralchess uci --checkpoint runs/laptop/checkpoints/best.npz
```

Speaks enough of the Universal Chess Interface to load in Arena, Cute Chess,
Banksia or a Lichess bot bridge. Supports `position startpos|fen … moves …`,
`go nodes|movetime|depth`, and `setoption` for simulations, `c_puct`, batch
size and temperature. Search output is reported as `info` lines with a
centipawn score mapped from the value head, so a GUI's evaluation bar behaves
the way a user expects:

```
> position fen 6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1
> go nodes 200
info depth 1 nodes 200 nps 11839 time 16 score cp 981 pv a1a8
bestmove a1a8
```

## Terminal

```bash
python -m neuralchess play --checkpoint runs/laptop/checkpoints/best.npz --color white
```

Prints the board, takes UCI moves, and shows the search summary after each
engine reply.

## Everything else

```bash
python -m neuralchess --help

train      run the self-play training loop
selfplay   generate games (optionally exported as PGN)
arena      play agents against each other and fit Elo
play       play in the terminal
serve      play in a browser
uci        run as a UCI engine
perft      move generator correctness and speed
bench      inference and search throughput on this machine
render     draw a position as an SVG
info       describe a checkpoint
```

Agents are named by a small spec language, so any two of them can be matched
without writing code:

```bash
python -m neuralchess arena \
  random material alphabeta:2 \
  policy:runs/laptop/checkpoints/best.npz \
  mcts:runs/laptop/checkpoints/best.npz \
  --games 30 --simulations 200 --pgn ladder.pgn
```
