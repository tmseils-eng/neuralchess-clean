"""Elo estimation from match results.

Two estimators are provided.  :func:`elo_difference` inverts the logistic
Elo curve for a single head-to-head score and reports a Wald confidence
interval, which is what the training loop uses for gating decisions.
:func:`fit_elo` solves the full multi-player maximum-likelihood problem, so a
whole ladder of checkpoints and baselines can be placed on one scale from a
sparse round-robin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

SCALE = 400.0
LOG10 = math.log(10.0)


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))


def elo_difference(score: float, games: int, confidence: float = 0.95):
    """Elo difference implied by a score rate, with a confidence interval.

    ``score`` is points per game in ``[0, 1]`` (win 1, draw 0.5).  Returns
    ``(elo, low, high)``; infinities are clamped to +/-800 so that a clean
    sweep of a short match does not produce an unusable number.
    """
    if games <= 0:
        return 0.0, 0.0, 0.0
    eps = 1.0 / (2.0 * games)
    p = min(max(score, eps), 1.0 - eps)
    elo = -SCALE * math.log10(1.0 / p - 1.0)
    se = math.sqrt(max(p * (1.0 - p), 1e-9) / games)
    z = 1.959964 if confidence >= 0.95 else 1.644854
    lo_p = min(max(p - z * se, eps), 1 - eps)
    hi_p = min(max(p + z * se, eps), 1 - eps)
    lo = -SCALE * math.log10(1.0 / lo_p - 1.0)
    hi = -SCALE * math.log10(1.0 / hi_p - 1.0)
    def clamp(value: float) -> float:
        return max(-800.0, min(800.0, value))

    return clamp(elo), clamp(lo), clamp(hi)


def likelihood_of_superiority(score: float, games: int) -> float:
    """Probability the challenger is genuinely stronger (normal approximation)."""
    if games <= 0:
        return 0.5
    se = math.sqrt(max(score * (1 - score), 1e-9) / games)
    if se <= 0:
        return 1.0 if score > 0.5 else 0.0
    z = (score - 0.5) / se
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class MatchRecord:
    player_a: str
    player_b: str
    wins: int = 0      # for player_a
    draws: int = 0
    losses: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games if self.games else 0.0

    def summary(self) -> str:
        elo, lo, hi = elo_difference(self.score, self.games)
        return (f"{self.player_a} vs {self.player_b}: "
                f"+{self.wins} ={self.draws} -{self.losses} "
                f"({self.score:.1%}, {elo:+.0f} Elo [{lo:+.0f}, {hi:+.0f}])")


def fit_elo(records: Sequence[MatchRecord], anchor: Optional[str] = None,
            anchor_rating: float = 0.0, iterations: int = 500,
            prior_games: float = 1.0) -> Dict[str, float]:
    """Maximum-likelihood Elo over a set of pairwise results.

    Uses the Bradley-Terry model with draws counted as half a win each way,
    solved by the standard minorisation-maximisation update.  ``prior_games``
    adds a virtual draw against a fictitious average player, which keeps
    undefeated players from diverging to infinity.
    """
    players = sorted({p for r in records for p in (r.player_a, r.player_b)})
    if not players:
        return {}
    gamma = {p: 1.0 for p in players}

    wins: Dict[Tuple[str, str], float] = {}
    games: Dict[Tuple[str, str], float] = {}
    for r in records:
        a, b = r.player_a, r.player_b
        wins[(a, b)] = wins.get((a, b), 0.0) + r.wins + 0.5 * r.draws
        wins[(b, a)] = wins.get((b, a), 0.0) + r.losses + 0.5 * r.draws
        games[(a, b)] = games.get((a, b), 0.0) + r.games
        games[(b, a)] = games.get((b, a), 0.0) + r.games

    for _ in range(iterations):
        for p in players:
            numerator = prior_games * 0.5
            denominator = prior_games / (gamma[p] + 1.0)
            for q in players:
                if q == p or (p, q) not in games:
                    continue
                numerator += wins[(p, q)]
                denominator += games[(p, q)] / (gamma[p] + gamma[q])
            if denominator > 0:
                gamma[p] = max(numerator / denominator, 1e-9)

    ratings = {p: SCALE * math.log10(g) for p, g in gamma.items()}
    if anchor and anchor in ratings:
        shift = anchor_rating - ratings[anchor]
    else:
        shift = -sum(ratings.values()) / len(ratings)
    return {p: r + shift for p, r in ratings.items()}


def format_ladder(ratings: Dict[str, float]) -> str:
    lines = ["  rating  player", "  ------  ------"]
    for name, rating in sorted(ratings.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {rating:6.0f}  {name}")
    return "\n".join(lines)
