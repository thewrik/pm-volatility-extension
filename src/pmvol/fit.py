"""Leakage-safe estimation for structural release and update hazards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

from .model import clock_release, dr_as_linear_variance, feasible_hazard

HAZARD_FEATURES = (
    "log_volume",
    "log_open_interest",
    "log_tau",
    "log_spread",
    "uncertainty",
    "lag_updated",
)


def _weights(frame: pd.DataFrame, weighting: str) -> np.ndarray:
    if weighting == "equal":
        result = np.ones(len(frame), dtype=float)
    elif weighting == "volume":
        result = np.maximum(frame["volume"].to_numpy(float), 0.0)
    elif weighting == "log_volume":
        result = 1.0 + np.log1p(np.maximum(frame["volume"].to_numpy(float), 0.0))
    else:
        raise ValueError(f"unknown weighting: {weighting}")
    mean = result.mean()
    return result / mean if mean > 0 else np.ones_like(result)


def fit_order_flow_k(
    frame: pd.DataFrame,
    *,
    finite_clock: bool,
    active_only: bool,
    weighting: str = "equal",
) -> float:
    """Fit the nonnegative DR-AS scale by Gaussian quasi-likelihood."""

    sample = frame.loc[frame["updated"].astype(bool)] if active_only else frame
    if sample.empty:
        raise ValueError("cannot estimate K on an empty sample")
    p = sample["price"].to_numpy(float)
    tau = sample["time_to_resolution"].to_numpy(float)
    spread = sample["spread"].to_numpy(float)
    volume = sample["volume"].to_numpy(float)
    innovation2 = np.square(sample["innovation"].to_numpy(float))
    weights = _weights(sample, weighting)

    def objective(log_k: float) -> float:
        k = float(np.exp(log_k))
        if finite_clock:
            variance = clock_release(p, tau, spread, volume, k=k) * p * (1.0 - p)
        else:
            variance = dr_as_linear_variance(p, tau, spread, volume, k=k)
        variance = np.maximum(variance, 1e-10)
        terms = np.log(variance) + innovation2 / variance
        return float(np.average(terms, weights=weights))

    result = minimize_scalar(objective, bounds=(-20.0, 20.0), method="bounded")
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"K optimization failed: {result}")
    return float(np.exp(result.x))


def hazard_raw_features(frame: pd.DataFrame) -> np.ndarray:
    volume = np.maximum(frame["volume"].to_numpy(float), 0.0)
    open_interest = np.maximum(frame["open_interest"].fillna(0).to_numpy(float), 0.0)
    tau = np.maximum(frame["time_to_resolution"].to_numpy(float), 1e-6)
    spread = np.maximum(frame["spread"].to_numpy(float), 0.0)
    price = frame["price"].to_numpy(float)
    lag = frame["lag_updated"].fillna(0).to_numpy(float)
    return np.column_stack(
        [
            np.log1p(volume),
            np.log1p(open_interest),
            np.log1p(tau),
            np.log(spread + 0.005),
            price * (1.0 - price),
            lag,
        ]
    )


@dataclass(frozen=True)
class HazardModel:
    coefficient: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    constrained: bool

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        raw = hazard_raw_features(frame)
        standardized = (raw - self.feature_mean) / self.feature_scale
        return np.column_stack([np.ones(len(frame)), standardized])

    def predict(self, frame: pd.DataFrame, release: np.ndarray) -> np.ndarray:
        eta = self.design(frame) @ self.coefficient
        if self.constrained:
            return feasible_hazard(release, eta)
        return expit(eta)


def fit_hazard(
    frame: pd.DataFrame,
    release: np.ndarray,
    *,
    constrained: bool,
    weighting: str = "equal",
    l2: float = 1e-4,
) -> HazardModel:
    """Fit a Bernoulli update model, optionally enforcing ``q >= release``."""

    z = frame["updated"].to_numpy(float)
    r = np.clip(np.asarray(release, dtype=float), 0.0, 1.0 - 1e-9)
    if r.shape != z.shape:
        raise ValueError("release and frame lengths differ")
    raw = hazard_raw_features(frame)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = np.column_stack([np.ones(len(frame)), (raw - mean) / scale])
    weights = _weights(frame, weighting)

    if constrained:
        residual_rate = np.mean(np.clip((z - r) / np.maximum(1.0 - r, 1e-9), 1e-4, 1 - 1e-4))
        intercept = float(logit(np.clip(residual_rate, 1e-4, 1 - 1e-4)))
    else:
        intercept = float(logit(np.clip(z.mean(), 1e-4, 1 - 1e-4)))
    initial = np.zeros(x.shape[1])
    initial[0] = intercept

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        eta = x @ coef
        s = expit(eta)
        if constrained:
            q = r + (1.0 - r) * s
            dq = (1.0 - r) * s * (1.0 - s)
        else:
            q = s
            dq = s * (1.0 - s)
        q = np.clip(q, 1e-10, 1.0 - 1e-10)
        nll_terms = -(z * np.log(q) + (1.0 - z) * np.log1p(-q))
        loss = float(np.average(nll_terms, weights=weights) + 0.5 * l2 * (coef[1:] @ coef[1:]))
        derivative_eta = (q - z) * dq / (q * (1.0 - q))
        gradient = (x.T @ (weights * derivative_eta)) / weights.sum()
        gradient[1:] += l2 * coef[1:]
        return loss, gradient

    result = minimize(
        lambda coef: objective(coef),
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"hazard optimization failed: {result.message}")
    return HazardModel(result.x, mean, scale, constrained)

