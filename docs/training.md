# The training loop

One iteration is:

1. **Self-play.** Generate games with the current network, in parallel worker
   processes.
2. **Store.** Append every recorded position to a fixed-size replay window.
3. **Learn.** Take gradient steps on batches drawn uniformly from that window.
4. **Measure.** Periodically play the fixed baseline ladder, and (optionally)
   gate the new network against the previous best.

Each recorded position becomes one training example: the encoded input
planes, the MCTS visit distribution as the policy target, and the eventual
game result as the value target. The loss is

$$\ell = -\pi^\top \log p \;+\; \text{CE}(z, \text{WDL}) \;+\; c\lVert\theta\rVert^2$$

where the L2 term lives in the optimiser as weight decay.

## Why the visit distribution, not the played move

The search visits moves in proportion to how good they turn out to be after
exploration. Training the policy on that distribution, rather than on the
single move eventually played, is what makes the loop self-improving: the
network learns to predict the *output of a search* it did not have to run,
so next iteration's search starts from a better prior and produces an even
better distribution. Policy improvement and policy evaluation, in the
classical reinforcement-learning sense.

## Playout-cap randomisation

Most moves are played with a small visit budget (32) and are **not** recorded;
a minority (30%) get the full budget (128) plus root noise and are. This is
KataGo's trick, and it matters here because the whole run has to fit on a
laptop: the moves that shape the game still get a real search, but the
positions-per-second rate roughly triples compared to searching every move at
full budget. Training data quality is set by the *recorded* searches; game
quality is set by the average one.

## The draw trap, and how the first run failed

The first full run of this loop got *worse*. Forty iterations, 800 self-play
games, a policy loss that fell from 2.33 to 1.67 — and in a round-robin the
final network lost to the checkpoint from **iteration 1**, using the same
architecture and the same search. The training loop was actively destroying
strength.

The cause is visible in one number: **72% of self-play games ended in a
draw**, almost all of them by hitting the move limit or repeating. A drawn
game gives every position in it a value target of exactly 0. With three
quarters of the data labelled 0, the value head learned the only thing that
data supports — that every position is drawn — and by the end of the run
self-play was 100% drawn and *every* value target was 0.

From there the collapse is mechanical. MCTS combines a prior with an action
value; if the value head returns 0 everywhere, the search has nothing to
optimise and degenerates into sampling from the policy. Training the policy on
that search's visit counts is then training the policy on *itself*. The
network converges on its own arbitrary initial preferences and becomes
confident about them — policy entropy fell from 2.34 to 1.68 nats while
playing strength went down. Loss curves looked healthy the entire time,
because the loss was measuring how well the network predicted a search that
the network itself was driving.

The metric that would have caught it early is not in the loss at all: it is
the fraction of self-play games with a decisive result. That is now logged
every iteration as `decisive_fraction`.

**The fix: adjudicate indecisive games on material.** When a game ends by move
limit, repetition or the fifty-move rule and one side is ahead by at least
150 centipawns of pure material (piece values only — no piece-square bonuses,
which would be indefensible as a verdict), the training target records that
side as the winner. Stalemate and insufficient material are real draws and are
never touched.

This shapes the *training target*, not the rules. It is the same adjudication
that engine-testing frameworks apply, and it is a compute decision: with
enough self-play, an AlphaZero run generates decisive games on its own, but at
laptop scale the value head has to be given something to learn from before the
policy can start improving. The evaluation arena scores games by the rules,
with adjudication only at the move limit and only at a full rook, so the
measurement stays independent of the shaping.

Effect on the value signal, measured on the same untrained network:

| | fraction of value targets equal to 0 |
| --- | --- |
| no adjudication | 1.00 |
| 300 cp margin | 0.53 |
| 150 cp margin | 0.12 |

## Resignation, and why it is audited

A side that the search evaluates below −0.92 for four consecutive recorded
plies resigns. This saves a large fraction of total compute — dead-lost
endgames are long and teach the value head almost nothing new.

It also introduces a bias: if the threshold is ever wrong, the value head
never sees the positions that would have corrected it. So 10% of games
disable resignation entirely. Those games are the audit set that keeps the
threshold honest.

## Gating

`gate_every > 0` makes each newly trained network play a match against the
current best and only replaces it on a score above threshold. This is the
AlphaGo Zero design; it stops a bad training step from poisoning the data
that trains the next one.

The AlphaZero paper dropped it, and for short runs so does
`configs/laptop.json` — with a small gating match the score estimate is noisy
enough that the promotion decision is close to a coin flip, and the compute
buys more by going into self-play. `configs/cluster.json` turns it back on,
where 60-game gating matches are affordable and the protection is real.

## Memorisation, and the metric that catches it

The run reported in [results](results.md) ends with a training value loss of
**0.246** — an average 78% probability assigned to the correct game outcome,
which looks excellent. It is not. Measured on 525 positions from games the
optimiser never saw:

| | training window | held out |
| --- | --- | --- |
| value cross-entropy | 0.246 | **2.238** |
| value top-1 accuracy | 79% | **41.5%** (majority class: 39.8%) |
| correlation of predicted value with the actual outcome | — | **0.045** |

The value head is no better than guessing the majority class on new
positions, and it is confidently wrong: it predicts a draw 1.3% of the time
when 21% of games are drawn. A direct probe makes it obvious — the network
returns the *same* distribution (W 0.56 / D 0.06 / L 0.38) whether White is a
queen up, a queen down, or material is level.

Three things cause this, and all three are worth knowing about:

**The replay window is reused far too aggressively.** At the laptop
configuration the optimiser draws `80 steps × 96 batch = 7,680` samples per
iteration from a window that gains only ~630 new positions. Every fresh
position is trained on roughly **twelve times** before it ages out. That
number is now logged as `reuse_ratio`; above about four, a run is memorising.

**The history planes are a memorisation channel.** 112 of the 119 input
planes encode the previous eight positions, which is very close to a unique
fingerprint of the specific game. All positions in a game share one outcome
label, so a network with 566K parameters and 28K positions can learn "which
game is this" and recall the answer.

**Nothing measured generalisation.** The loop reported only training loss, so
there was no number that could distinguish learning from recall.

The loop now holds out a fraction of self-play games — **whole games, never
individual positions**, since a position-level split leaks the label straight
across a game — and reports `val_policy_loss`, `val_value_loss`,
`val_value_accuracy` and `generalisation_gap` every iteration. Batch-norm runs
in eval mode during that pass so validation data never touches the running
statistics.

### What actually fixed it, and what did not

Two hypotheses, tested rather than assumed.

**Reuse ratio: not the cause.** A run at 48 games and 30 steps per iteration
brings the reuse ratio from 12.6x down to ~2x — each new position is seen
twice instead of twelve times. Nine iterations in, the value head was
diverging exactly as before:

| iteration | value train | value held out | policy train | policy held out |
| --- | --- | --- | --- | --- |
| 1 | 0.817 | 1.058 | 2.707 | 2.551 |
| 5 | 0.632 | 1.684 | 2.142 | 2.307 |
| 9 | 0.581 | **1.795** | 1.997 | 2.128 |

Note which head is at fault. The **policy** generalises fine — its held-out
loss tracks training within 0.13 and both fall. The **value** head is where
training and held-out losses move in opposite directions. That is worth
knowing on its own: the two heads share a tower but fail differently, and any
diagnosis that only watches the total loss will miss it. The run was stopped
at iteration 9 rather than spending two more hours confirming a negative.

**History depth: the actual channel.** A full training run cannot separate
history depth from everything else that differs between runs, so
`scripts/ablate_history.py` trains four depths on the *same* 120 games, the
same game-level split, the same 400 steps and the same seed. Positions are
stored as move lists and re-encoded per variant — encoding once at depth 8 and
slicing would leak the deeper representation into the shallow ones.

![history depth ablation](../assets/ablation_history.svg)

| depth | planes | train | held out | gap | held-out value accuracy |
| --- | --- | --- | --- | --- | --- |
| T=8 (AlphaZero) | 119 | 1.906 | 3.633 | +1.727 | 0.59 |
| T=4 | 63 | 1.903 | 3.415 | +1.512 | 0.60 |
| **T=2** | **35** | 1.946 | **2.996** | **+1.050** | **0.69** |
| T=1 | 21 | 1.962 | 3.476 | +1.514 | 0.57 |

Training loss is flat across all four — every variant fits its data equally
well. Only generalisation moves. Cutting history from eight steps to two
reduces the gap by 39% and lifts held-out value accuracy by ten points.

The result is **not monotonic**, which is the interesting part: T=1 is as bad
as T=4. One step encodes only the current position, so the network cannot see
*what just changed* — which piece the opponent moved, whether a capture just
happened. Two steps restore that while still being far too little to
fingerprint a specific game. The useful signal in history is the last move;
the rest is mostly a label-recall channel.

Shallower is also simply cheaper: at 5x48, T=2 gives 2x the inference
throughput and 2.4x the training throughput of T=8, for 6% fewer parameters.

### Decoupled weight decay

The runs above were configured with `weight_decay: 1e-4` on Adam, which adds
`wd * p` to the gradient — where the adaptive denominator immediately
normalises it away. Measured on a parameter with no other gradient signal,
raising the setting tenfold changes the shrinkage by:

| optimiser | shrinkage at wd=0.01 | at wd=0.1 | ratio |
| --- | --- | --- | --- |
| Adam (L2 in the gradient) | 0.0983 | 0.0983 | **1.0x** |
| AdamW (decoupled) | 0.0010 | 0.0100 | **9.9x** |

Under Adam the setting is very nearly inert — it was not regularising
anything. `optimizer: adamw` applies the decay directly to the weights, where
it is proportional to the learning rate alone and means what it says.

## Reading the metrics honestly

`policy_accuracy` is the network's top-1 agreement with the search's most
visited move. It starts high and that is not a sign of a strong network: the
search is *guided by the same network's priors*, so early in training the two
agree because both are close to uniform-with-noise, not because either is
good. The metrics that are not circular are `policy_entropy` (which should
fall as the network becomes decisive), `value_loss` against the actual game
outcome, and above all the scores against the fixed baseline ladder, which
never changes.

## Measuring strength

Elo is estimated two ways:

- `elo_difference(score, games)` inverts the logistic curve for a single
  head-to-head result and reports a Wald confidence interval plus a
  likelihood-of-superiority. This is what gating uses.
- `fit_elo(records)` solves the full multi-player maximum-likelihood problem
  (Bradley–Terry with draws as half-wins, solved by minorisation–maximisation)
  so a whole ladder of checkpoints and baselines lands on one scale from a
  sparse round-robin. A virtual prior draw keeps an undefeated player from
  diverging to infinity.

The fixed ladder is `random`, `material` (one-ply greedy with piece-square
tables), `alphabeta:2` (classical negamax with MVV-LVA ordering), and
`policy-only` (the network with no search). Reporting against opponents that
never change is what makes the numbers comparable across a run.

## Reproducibility and interruption

Every run writes `config.json`, an append-only `metrics.jsonl`, SVG plots, and
a checkpoint per iteration. Runs resume from
`runs/<name>/checkpoints/best.npz` automatically, which is what makes the
HTCondor and SLURM submit scripts safe to pre-empt: an evicted job loses at
most one iteration.
