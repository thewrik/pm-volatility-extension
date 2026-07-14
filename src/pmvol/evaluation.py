"""Vectorized proper scores and intervals for model comparison."""

from __future__ import annotations

import numpy as np
from scipy.special import betainc, ndtr
from scipy.stats import beta as beta_dist

from .model import interval_score


def _coherent_beta_parameters(
    price: np.ndarray,
    release: np.ndarray,
    hazard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numerically stable parameters, including the terminal Bernoulli limit."""

    p, r, q = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(release, float),
        np.asarray(hazard, float),
    )
    r = np.clip(r, 1e-10, 1.0)
    q = np.clip(np.maximum(q, r), r, 1.0)
    concentration = np.maximum(q / r - 1.0, 0.0)
    terminal = concentration <= 1e-8
    safe_concentration = np.maximum(concentration, 1e-8)
    alpha = np.clip(p * safe_concentration, 1e-10, None)
    beta = np.clip((1.0 - p) * safe_concentration, 1e-10, None)
    return p, r, q, alpha, beta, terminal


def _cells(
    realized: np.ndarray, tick_size: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    y, tick = np.broadcast_arrays(
        np.asarray(realized, dtype=float), np.asarray(tick_size, dtype=float)
    )
    if np.any(tick <= 0):
        raise ValueError("tick_size must be positive")
    return np.maximum(0.0, y - tick / 2.0), np.minimum(1.0, y + tick / 2.0)


def hurdle_beta_cell_probability(
    price: np.ndarray,
    realized: np.ndarray,
    release: np.ndarray,
    hazard: np.ndarray,
    *,
    tick_size: np.ndarray | float = 0.005,
) -> np.ndarray:
    """Rounding-aware cell probabilities for the hurdle-beta forecast."""

    p, y, r, q = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(realized, float),
        np.asarray(release, float),
        np.asarray(hazard, float),
    )
    p, r, q, alpha, beta, terminal = _coherent_beta_parameters(p, r, q)
    lower, upper = _cells(y, tick_size)
    active_beta = q * (betainc(alpha, beta, upper) - betainc(alpha, beta, lower))
    active_terminal = q * (
        (1.0 - p) * ((lower <= 0.0) & (0.0 <= upper))
        + p * ((lower <= 1.0) & (1.0 <= upper))
    )
    active = np.where(terminal, active_terminal, active_beta)
    atom = (1.0 - q) * ((lower <= p) & (p <= upper))
    return np.clip(active + atom, np.finfo(float).tiny, 1.0)


def clipped_normal_cell_probability(
    price: np.ndarray,
    realized: np.ndarray,
    variance: np.ndarray,
    *,
    tick_size: np.ndarray | float = 0.005,
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


def hurdle_beta_randomized_cell_pit(
    price: np.ndarray,
    realized: np.ndarray,
    release: np.ndarray,
    hazard: np.ndarray,
    uniform: np.ndarray,
    *,
    tick_size: np.ndarray | float = 0.005,
) -> np.ndarray:
    """Randomized PIT for the observed lattice cell under the mixed law."""

    p, y, r, q, u = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(realized, float),
        np.asarray(release, float),
        np.asarray(hazard, float),
        np.asarray(uniform, float),
    )
    p, r, q, alpha, beta, terminal = _coherent_beta_parameters(p, r, q)
    lower, _ = _cells(y, tick_size)
    beta_before = q * betainc(alpha, beta, lower)
    terminal_before = q * (1.0 - p) * (0.0 < lower)
    active_before = np.where(terminal, terminal_before, beta_before)
    cdf_before = active_before + (1.0 - q) * (p < lower)
    cell_probability = hurdle_beta_cell_probability(
        p, y, r, q, tick_size=tick_size
    )
    return np.clip(cdf_before + u * cell_probability, 0.0, 1.0)


def clipped_normal_randomized_cell_pit(
    price: np.ndarray,
    realized: np.ndarray,
    variance: np.ndarray,
    uniform: np.ndarray,
    *,
    tick_size: np.ndarray | float = 0.005,
) -> np.ndarray:
    """Randomized PIT for a lattice cell under a clipped latent Gaussian."""

    p, y, var, u = np.broadcast_arrays(
        np.asarray(price, float),
        np.asarray(realized, float),
        np.asarray(variance, float),
        np.asarray(uniform, float),
    )
    scale = np.sqrt(np.maximum(var, 1e-12))
    lower, _ = _cells(y, tick_size)
    cdf_before = np.where(lower <= 0.0, 0.0, ndtr((lower - p) / scale))
    cell_probability = clipped_normal_cell_probability(
        p, y, var, tick_size=tick_size
    )
    return np.clip(cdf_before + u * cell_probability, 0.0, 1.0)


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
    p, r, q, alpha_shape, beta_shape, terminal = _coherent_beta_parameters(p, r, q)
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
        beta_result = beta_dist.ppf(active_probability, alpha_shape, beta_shape)
        terminal_left = q * (1.0 - p)
        terminal_right = terminal_left + 1.0 - q
        terminal_result = np.where(u <= terminal_left, 0.0, np.where(u <= terminal_right, p, 1.0))
        result = np.where(terminal, terminal_result, beta_result)
        return np.where(in_atom & ~terminal, p, result)

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
