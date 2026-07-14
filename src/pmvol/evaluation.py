"""Vectorized proper scores and intervals for model comparison."""

from __future__ import annotations

import numpy as np
from scipy.special import betainc, ndtr
from scipy.stats import beta as beta_dist

from .model import interval_score


def _cells(realized: np.ndarray, tick_size: float) -> tuple[np.ndarray, np.ndarray]:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    y = np.asarray(realized, dtype=float)
    return np.maximum(0.0, y - tick_size / 2.0), np.minimum(1.0, y + tick_size / 2.0)


def hurdle_beta_cell_probability(
    price: np.ndarray,
    realized: np.ndarray,
    release: np.ndarray,
    hazard: np.ndarray,
    *,
    tick_size: float = 0.005,
) -> np.ndarray:
    """Rounding-aware cell probabilities for the hurdle-beta forecast."""

    p, y, r, q = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(realized, float),
        np.asarray(release, float),
        np.asarray(hazard, float),
    )
    r = np.clip(r, 1e-10, 1.0 - 1e-10)
    q = np.clip(np.maximum(q, r + 1e-10), r + 1e-10, 1.0)
    concentration = np.maximum(q / r - 1.0, 1e-8)
    alpha = np.clip(p * concentration, 1e-10, None)
    beta = np.clip((1.0 - p) * concentration, 1e-10, None)
    lower, upper = _cells(y, tick_size)
    active = q * (betainc(alpha, beta, upper) - betainc(alpha, beta, lower))
    atom = (1.0 - q) * ((lower <= p) & (p <= upper))
    return np.clip(active + atom, np.finfo(float).tiny, 1.0)


def clipped_normal_cell_probability(
    price: np.ndarray,
    realized: np.ndarray,
    variance: np.ndarray,
    *,
    tick_size: float = 0.005,
) -> np.ndarray:
    """Cell probability after clipping a latent Gaussian price to ``[0,1]``."""

    p, y, var = np.broadcast_arrays(
        np.asarray(price, float), np.asarray(realized, float), np.asarray(variance, float)
    )
    scale = np.sqrt(np.maximum(var, 1e-12))
    lower, upper = _cells(y, tick_size)
    z_lower = (lower - p) / scale
    z_upper = (upper - p) / scale
    probability = ndtr(z_upper) - ndtr(z_lower)
    probability = np.where(lower <= 0.0, ndtr(z_upper), probability)
    probability = np.where(upper >= 1.0, 1.0 - ndtr(z_lower), probability)
    return np.clip(probability, np.finfo(float).tiny, 1.0)


def hurdle_beta_interval(
    price: np.ndarray,
    release: np.ndarray,
    hazard: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized central interval from the mixed distribution's quantiles."""

    if not 0 < level < 1:
        raise ValueError("level must lie in (0,1)")
    p, r, q = np.broadcast_arrays(
        np.asarray(price, float), np.asarray(release, float), np.asarray(hazard, float)
    )
    r = np.clip(r, 1e-10, 1.0 - 1e-10)
    q = np.clip(np.maximum(q, r + 1e-10), r + 1e-10, 1.0)
    concentration = np.maximum(q / r - 1.0, 1e-8)
    alpha_shape = np.clip(p * concentration, 1e-10, None)
    beta_shape = np.clip((1.0 - p) * concentration, 1e-10, None)
    active_at_p = beta_dist.cdf(p, alpha_shape, beta_shape)
    atom_left = q * active_at_p
    atom_right = atom_left + 1.0 - q

    def quantile(u: float) -> np.ndarray:
        below = u <= atom_left
        in_atom = (u > atom_left) & (u <= atom_right)
        active_probability = np.where(
            below,
            u / q,
            (u - (1.0 - q)) / q,
        )
        active_probability = np.clip(active_probability, 0.0, 1.0)
        result = beta_dist.ppf(active_probability, alpha_shape, beta_shape)
        return np.where(in_atom, p, result)

    tail = (1.0 - level) / 2.0
    return quantile(tail), quantile(1.0 - tail)


def summarize_forecast(
    probability: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    realized: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict[str, float]:
    y = np.asarray(realized, float)
    w = np.asarray(weights, float)
    w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
    score = interval_score(lower, upper, y, alpha=alpha)
    covered = (lower <= y) & (y <= upper)
    return {
        "negative_log_score": float(np.sum(w * -np.log(probability))),
        "interval_score": float(np.sum(w * score)),
        "coverage": float(np.sum(w * covered)),
        "width": float(np.sum(w * (upper - lower))),
    }

