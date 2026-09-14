# The chess engine underneath

Nothing in this project imports a chess library. The rules — move generation,
legality, repetition, SAN, PGN — are implemented in
[`neuralchess/chess/`](../neuralchess/chess) and verified against the standard
perft suite.

## Board representation

A position is twelve 64-bit integers (`pieces[color][piece_type]`), two
occupancy masks, and a 64-entry mailbox array that answers "what is on this
square?" in one lookup. Squares are numbered `a1 = 0 … h8 = 63`, so
`rank = sq >> 3` and `file = sq & 7`.

Keeping *both* bitboards and a mailbox is redundant, and deliberately so.
Bitboards make "all squares this rook attacks" a handful of integer
operations; the mailbox makes "what did I just capture?" a single array
index. Move generation needs the first, make/unmake needs the second.

## Sliding pieces: classical rays

Rook and bishop attacks use the classical ray method. For each of the 64
squares and each of the 8 directions the full ray is precomputed. At query
time:

```python
ray = RAYS[direction][square]
blockers = ray & occupancy
if blockers:
    nearest = lsb(blockers) if direction_is_positive else msb(blockers)
    ray ^= RAYS[direction][nearest]     # cut off everything past the blocker
```

Magic bitboards would be faster, but this search spends well over 90% of its
time in the neural network, so the extra complexity would buy nothing
measurable. That trade-off is the whole reason the choice is worth stating.

## Legality

Moves are generated pseudo-legally and then filtered by making the move and
asking whether our own king is attacked. This is the simplest formulation
that is *obviously* correct, which matters more here than the ~20% that
incremental pin detection would save.

One deliberate deviation: **captures of the enemy king are never generated.**
They can only arise from an illegal position — one where the side *not* to
move is already in check — and generating one leaves the board kingless and
silently corrupts every downstream consumer. Excluding them makes the engine
total on arbitrary FEN input, which matters when a UI or a UCI GUI can send
anything. `Board.is_valid()` reports the condition explicitly.

## Correctness: perft

`perft(n)` counts the leaves of the legal move tree at depth `n`. It is the
standard correctness test because a single mishandled edge case — an en
passant capture that exposes a pin, a castling right lost by a rook capture —
changes the count by an amount no eyeball review would catch.

```
$ python -m neuralchess perft --depth 5
perft(1) =           20      0.00s
perft(2) =          400      0.00s
perft(3) =        8,902      0.04s
perft(4) =      197,281      0.82s
perft(5) =    4,865,609     20.9s
```

The test suite checks six positions chosen to exercise every rule that is
easy to get wrong:

| position | depths verified |
| --- | --- |
| start position | 20 / 400 / 8,902 / 197,281 |
| "Kiwipete" (castling, pins, promotions) | 48 / 2,039 / 97,862 |
| pawn endgame with en passant | 14 / 191 / 2,812 / 43,238 |
| promotion tactics (both mirrors) | 6 / 264 / 9,467 |
| under-promotion and discovered check | 44 / 1,486 / 62,379 |
| dense middlegame | 46 / 2,079 / 89,890 |

Alongside perft the suite checks that make/unmake restores every field
byte-for-byte, that the incrementally updated Zobrist hash always equals a
full recomputation, and that SAN disambiguation, castling notation and the
check/mate suffix are right.

## Hashing and draws

Zobrist hashing is updated incrementally inside `push`/`pop`: XOR out the
moving piece, XOR in its destination, and toggle the side-to-move, castling
and en-passant keys. The hash serves three purposes — threefold repetition
detection, the search's transposition cache, and a cheap invariant to test
against.

Draw detection covers all four rules: stalemate, the fifty-move rule,
threefold repetition, and insufficient material (including the "any number of
bishops on one colour complex" case that naive implementations miss).
