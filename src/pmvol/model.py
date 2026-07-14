"""Martingale-coherent distribution forecasts for bounded binary prices.

The central objects are:

``r``
    The fraction of remaining Bernoulli uncertainty released over the forecast
    horizon, so ``Var(P_next | F) = r * p * (1-p)``.

``q``
    The conditional probability of an active update. Bounded-martingale
    coherence requires ``0 <= r <= q <= 1``.

The :class:`HurdleBeta` construction realizes every interior pair ``0 < r < q``
with a point mass at the current price and a mean-preserving beta component.
The equality case ``r == q`` is the limiting active Bernoulli distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, log
from typing import Literal

import numpy as np
from scipy.special import betainc, expit
from scipy.stats import beta as beta_dist

Activity = Literal["constant", "log1p", "sqrt", "linear"]
_EPS = 1e-12


def _activity(volume: np.ndarray, kind: Activity) -> np.ndarray:
    volume = np.maximum(np.asarray(volume, dtype=float), 0.0)
    if kind == "constant":
        return np.ones_like(volume)
    if kind == "log1p":
        return np.log1p(volume)
    if kind == "sqrt":
        return np.sqrt(volume)
    if kind == "linear":
        return volume
    raise ValueError(f"unknown activity transform: {kind}")


def dr_as_linear_variance(
    price: np.ndarray | float,
    time_to_resolution: np.ndarray | float,
    spread: np.ndarray | float = 0.0,
    volume: np.ndarray | float = 0.0,
    *,
    k: float = 0.0,
    delta: float = 1.0,
    activity: Activity = "sqrt",
) -> np.ndarray:
    """The additive Euler DR-AS variance used by Xi et al. (2026).

    This function intentionally does not cap the result at remaining binary
    uncertainty; it is retained as the exact baseline being audited.
    """

    p, tau, s, v = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(time_to_resolution, dtype=float),
        np.asarray(spread, dtype=float),
        np.asarray(volume, dtype=float),
    )
    if delta <= 0:
        raise ValueError("delta must be positive")
    if k < 0:
        raise ValueError("k must be nonnegative")
    uncertainty = p * (1.0 - p)
    safe_tau = np.maximum(tau, _EPS)
    return (uncertainty / safe_tau + k * _activity(v, activity) * s**2 / 4.0) * delta


def clock_release(
    price: np.ndarray | float,
    time_to_resolution: np.ndarray | float,
    spread: np.ndarray | float = 0.0,
    volume: np.ndarray | float = 0.0,
    *,
    k: float = 0.0,
    delta: float = 1.0,
    activity: Activity = "sqrt",
) -> np.ndarray:
    """Return a finite-horizon normalized variance release in ``[0, 1]``.

    The deadline clock is integrated exactly over the step:

    ``Lambda_DR = log(tau / (tau-delta))`` for ``tau > delta``.

    The reduced-form order-flow variance is interpreted as a local information
    intensity after division by remaining uncertainty. Independent clock
    intensities add, and a Wright-Fisher information interval of length
    ``Lambda`` releases the fraction ``1-exp(-Lambda)`` of current uncertainty.

    For small steps, multiplying the returned value by ``p(1-p)`` agrees with
    additive DR-AS to first order. Unlike the Euler expression, it can never
    spend more than all remaining binary uncertainty.
    """

    p, tau, s, v = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(time_to_resolution, dtype=float),
        np.asarray(spread, dtype=float),
        np.asarray(volume, dtype=float),
    )
    if delta <= 0:
        raise ValueError("delta must be positive")
    if k < 0:
        raise ValueError("k must be nonnegative")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("price must lie in [0, 1]")

    uncertainty = p * (1.0 - p)
    interior = uncertainty > _EPS

    with np.errstate(divide="ignore", invalid="ignore"):
        deadline_clock = np.where(
            tau <= delta,
            np.inf,
            -np.log1p(-delta / tau),
        )
        order_flow_clock = np.where(
            interior,
            k * _activity(v, activity) * s**2 * delta / (4.0 * uncertainty),
            0.0,
        )
        total_clock = deadline_clock + order_flow_clock
        release = -np.expm1(-total_clock)

    # At an absorbed boundary there is no remaining variance to release. The
    # normalized ratio is formally 0/0; returning zero is the useful convention.
    release = np.where(interior, release, 0.0)
    return np.clip(release, 0.0, 1.0)


def feasible_hazard(release: np.ndarray | float, linear_predictor: np.ndarray | float) -> np.ndarray:
    """Map an unconstrained predictor to an update hazard satisfying ``q >= r``."""

    r, eta = np.broadcast_arrays(
        np.asarray(release, dtype=float), np.asarray(linear_predictor, dtype=float)
    )
    if np.any((r < 0) | (r > 1)):
        raise ValueError("release must lie in [0, 1]")
    return r + (1.0 - r) * expit(eta)


@dataclass(frozen=True)
class HurdleBeta:
    """A martingale-coherent hurdle distribution on ``[0, 1]``.

    Parameters are scalar to keep the distribution API explicit. Vectorized
    fitting and scoring utilities construct these parameters in arrays and use
    the same formulas.
    """

    price: float
    release: float
    update_probability: float

    def __post_init__(self) -> None:
        p, r, q = self.price, self.release, self.update_probability
        if not 0.0 < p < 1.0:
            raise ValueError("price must be strictly inside (0, 1)")
        if not 0.0 < r <= q <= 1.0:
            raise ValueError("parameters must satisfy 0 < release <= update_probability <= 1")

    @property
    def is_terminal_limit(self) -> bool:
        return isclose(self.release, self.update_probability, rel_tol=1e-10, abs_tol=1e-12)

    @property
    def concentration(self) -> float:
        """Concentration of the active beta component; zero is its endpoint limit."""

        return max(self.update_probability / self.release - 1.0, 0.0)

    @property
    def alpha(self) -> float:
        return self.price * self.concentration

    @property
    def beta(self) -> float:
        return (1.0 - self.price) * self.concentration

    @property
    def mean(self) -> float:
        return self.price

    @property
    def variance(self) -> float:
        return self.release * self.price * (1.0 - self.price)

    @property
    def no_update_probability(self) -> float:
        return 1.0 - self.update_probability

    def cdf(self, value: float) -> float:
        """CDF of the mixed law, including its atom at the current price."""

        y = float(value)
        if y < 0.0:
            return 0.0
        if y >= 1.0:
            return 1.0
        p, q = self.price, self.update_probability
        if self.is_terminal_limit:
            mass_zero = q * (1.0 - p)
            if y < p:
                return mass_zero
            return mass_zero + (1.0 - q)
        active_cdf = float(beta_dist.cdf(y, self.alpha, self.beta))
        return q * active_cdf + ((1.0 - q) if y >= p else 0.0)

    def ppf(self, probability: float) -> float:
        """Generalized inverse CDF, used for calibrated central intervals."""

        u = float(probability)
        if not 0.0 <= u <= 1.0:
            raise ValueError("probability must lie in [0, 1]")
        if u == 0.0:
            return 0.0
        if u == 1.0:
            return 1.0

        p, q = self.price, self.update_probability
        if self.is_terminal_limit:
            left = q * (1.0 - p)
            right = left + 1.0 - q
            if u <= left:
                return 0.0
            if u <= right:
                return p
            return 1.0

        active_at_p = float(beta_dist.cdf(p, self.alpha, self.beta))
        left = q * active_at_p
        right = left + 1.0 - q
        if u <= left:
            return float(beta_dist.ppf(u / q, self.alpha, self.beta))
        if u <= right:
            return p
        return float(beta_dist.ppf((u - (1.0 - q)) / q, self.alpha, self.beta))

    def central_interval(self, level: float = 0.95) -> tuple[float, float]:
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        tail = (1.0 - level) / 2.0
        return self.ppf(tail), self.ppf(1.0 - tail)

    def bin_probability(self, value: float, tick_size: float = 0.005) -> float:
        """Probability of the observed price cell around ``value``.

        This is the recommended likelihood contribution for lattice-valued
        mid-quotes. It includes both the no-update atom and active-component
        probability that rounds into the same cell.
        """

        if tick_size <= 0:
            raise ValueError("tick_size must be positive")
        y = float(value)
        if not 0.0 <= y <= 1.0:
            return 0.0
        lower = max(0.0, y - tick_size / 2.0)
        upper = min(1.0, y + tick_size / 2.0)
        p, q = self.price, self.update_probability
        atom = (1.0 - q) if lower <= p <= upper else 0.0

        if self.is_terminal_limit:
            active = 0.0
            if lower <= 0.0 <= upper:
                active += q * (1.0 - p)
            if lower <= 1.0 <= upper:
                active += q * p
            return min(max(atom + active, 0.0), 1.0)

        # scipy.special.betainc is the regularized beta CDF and is stable for
        # the small shape parameters encountered close to the terminal limit.
        active = q * (
            float(betainc(self.alpha, self.beta, upper))
            - float(betainc(self.alpha, self.beta, lower))
        )
        return min(max(atom + active, 0.0), 1.0)

    def binned_negative_log_score(self, value: float, tick_size: float = 0.005) -> float:
        return -log(max(self.bin_probability(value, tick_size), np.finfo(float).tiny))

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if size < 0:
            raise ValueError("size must be nonnegative")
        rng = rng or np.random.default_rng()
        active = rng.random(size) < self.update_probability
        draws = np.full(size, self.price, dtype=float)
        n_active = int(active.sum())
        if n_active == 0:
            return draws
        if self.is_terminal_limit:
            draws[active] = (rng.random(n_active) < self.price).astype(float)
        else:
            draws[active] = rng.beta(self.alpha, self.beta, size=n_active)
        return draws


def normal_reference_interval(
    price: np.ndarray | float,
    variance: np.ndarray | float,
    *,
    z: float = 1.959963984540054,
) -> tuple[np.ndarray, np.ndarray]:
    """The paper's symmetric normal-reference interval clipped to ``[0, 1]``."""

    p, var = np.broadcast_arrays(np.asarray(price, dtype=float), np.asarray(variance, dtype=float))
    scale = np.sqrt(np.maximum(var, 0.0))
    return np.maximum(0.0, p - z * scale), np.minimum(1.0, p + z * scale)


def interval_score(
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    realized: np.ndarray | float,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Winkler interval score for a central ``1-alpha`` interval."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    lo, hi, y = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        np.asarray(realized, dtype=float),
    )
    if np.any(lo > hi):
        raise ValueError("lower endpoint exceeds upper endpoint")
    return (
        hi
        - lo
        + (2.0 / alpha) * (lo - y) * (y < lo)
        + (2.0 / alpha) * (y - hi) * (y > hi)
    )

