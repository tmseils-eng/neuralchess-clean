# Turning a position into a tensor

Two conversions sit between the rules and the network: the position must
become an input tensor, and moves must map onto output indices.

## Input: 119 planes of 8×8

| planes | contents |
| --- | --- |
| 0–111 | eight history steps × 14 planes (6 our pieces, 6 theirs, 2 repetition flags) |
| 112 | side to move |
| 113 | fullmove number, scaled |
| 114–117 | castling rights (ours king/queen side, then theirs) |
| 118 | halfmove clock, scaled |

Two design points are worth calling out.

**Everything is from the mover's point of view.** When black is to move the
board is flipped vertically (`sq ^ 56`) and the colour planes are swapped, so
"my pawns" always march up the board. The network never has to learn the
black-side mirror image of anything it learned for white, which roughly
halves the effective size of the input distribution. It is also why the
policy head needs no colour flag.

**History is eight steps deep** because chess is not quite Markovian:
threefold repetition and the fifty-move rule depend on the past, and a
network that cannot see repetitions will happily shuffle into a draw it
believes is winning. The two repetition planes per step make that visible
directly rather than asking the network to infer it.

For speed, history is stored as `Frame` snapshots — twelve integers and four
scalars — rather than full board copies. The search touches this code once
per simulation, so the difference is not academic: frames are roughly
two orders of magnitude cheaper to create than a `Board.copy()`.

## Output: 4,672 move indices

The policy head emits an `8 × 8 × 73` stack. Every origin square owns 73
planes:

| planes | move type |
| --- | --- |
| 0–55 | queen-like: 8 ray directions × 7 distances |
| 56–63 | the 8 knight jumps |
| 64–72 | underpromotions: 3 pawn directions × {knight, bishop, rook} |

`8 × 8 × 73 = 4,672`. Queen promotions get no planes of their own: a pawn
reaching the last rank along a queen-like plane is promoted to a queen by
convention, which is why 73 rather than 76 planes suffice.

Of those 4,672 slots, 1,880 correspond to a geometrically possible
`(from, to, promotion)` triple; the rest are unreachable and are masked out
before every softmax. Masking rather than pruning keeps the output shape
fixed, which keeps the head a single convolution.

The map is a bijection on legal moves, and the test suite verifies exactly
that: over a random walk of hundreds of positions, every legal move maps to a
distinct index and decodes back to itself. The inverse needs the legal move
list to resolve the queen-promotion alias, which is what
`index_to_move(index, legal_moves, flip)` is for.
