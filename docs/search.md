# Monte-Carlo tree search

The search is PUCT, as in AlphaZero. Each edge stores a prior `P` from the
policy head, a visit count `N`, and an action value `Q` averaged from the
value head. Selection maximises

$$Q(s,a) + c_{\text{puct}}(s)\, P(s,a) \frac{\sqrt{N(s)}}{1 + N(s,a)}$$

with the exploration constant growing slowly with the parent's visit count:

$$c_{\text{puct}}(s) = \log \frac{1 + N(s) + c_{\text{base}}}{c_{\text{base}}} + c_{\text{init}}$$

Early on, the prior dominates and the search follows the network's intuition;
as visits accumulate, the empirical `Q` takes over. That is the whole
mechanism by which search improves on the raw policy.

## Three things that matter in practice

**Batched leaf collection.** A pure-NumPy network is dominated by per-call
overhead — evaluating 16 positions costs barely more than evaluating one.
The search therefore gathers a batch of leaves before calling the network.
Paths are kept from collapsing onto each other by a *virtual loss*: an
in-flight edge is temporarily credited with a loss, making it unattractive to
the next descent, and the adjustment is undone during backup. This is the
same trick that lets AlphaZero's search run across many threads.

Measured on the 5×48 network (single core, NumPy backend):

| leaf batch | simulations / second |
| --- | --- |
| 1 | 510 |
| 4 | 769 |
| 16 | 928 |
| 32 | 1,094 |

Better than **2× throughput for free**, and the larger the network the more
the batching wins.

**First-play urgency.** An unvisited child has no `Q`, so something must be
assumed. Optimism ("assume a win") makes the search sample all 30-odd legal
moves before deepening anywhere. This implementation uses FPU *reduction*:
an unvisited child inherits the parent's value minus a penalty that grows
with how much prior mass has already been explored. A freshly initialised
network — where every prior is near-uniform — is exactly the case where this
matters most.

**Edge-array storage.** Children live in NumPy arrays on the parent
(`priors`, `child_visits`, `child_values`) rather than as objects. Selecting
a move is one vectorised `argmax` over up to 218 children instead of a Python
loop, and the arrays are what get updated during backup.

## Root behaviour

At the root of a self-play search, Dirichlet noise is mixed into the priors
(`(1−ε)P + εη`, with `η ~ Dir(0.3)` and `ε = 0.25`). Without it the network's
own policy is the only thing ever explored, and self-play stops generating
new information. Noise is added during training only — the search plays
deterministically when you play against it.

Move selection uses a temperature schedule: sampling proportionally to visit
counts for the first ~20 plies (opening diversity), then near-greedy after
that (so the recorded game is actually decided by strength).

Subtrees are reused across moves: after a move is played, the corresponding
child becomes the new root and keeps every visit it accumulated.

## Caching, and a memory trap worth naming

Within a search the same position is reached by many move orders, so results
are cached by Zobrist key. The obvious implementation caches the raw 4,672-wide
logit vector the network emits — and that is a trap: at 18 KB per entry, a
100k-entry cache is **1.9 GB per process**, which on a two-worker self-play run
is the difference between fitting in RAM and thrashing. (This was not
hypothetical; the first version of this training loop crawled at 12% CPU with a
3 GB resident set until the cache was the suspect.)

The cache instead stores the *expanded* result: the legal move list and its
normalised priors, about 600 bytes. Same hit rate, 30× less memory. The leaf
handler also generates legal moves exactly once and reuses that list for the
terminal test, the priors and the expansion — together these took a self-play
game at 128 simulations per move from ~25 s to under 2 s.

## Terminal positions

Proven results are stored on the node and never re-evaluated. This matters
more than it sounds: once a search finds a forced mate, every subsequent
descent down that line resolves without a network call. An early version of
this code spun forever collecting a batch that could never fill, because
terminal hits were not counted toward the batch budget — the kind of bug that
only appears when the search actually starts finding things.

## What the search buys

The `policy-only` baseline plays the raw `argmax` of the policy head with no
search at all. Comparing it against the full MCTS agent isolates the
contribution of search from the contribution of the network — a comparison
that is easy to build and surprisingly often skipped:

```
python -m neuralchess arena policy:runs/laptop/checkpoints/best.npz \
                            mcts:runs/laptop/checkpoints/best.npz \
                            --games 40 --simulations 200
```
