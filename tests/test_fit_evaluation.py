import numpy as np
import pandas as pd
import pytest

from pmvol.evaluation import (
    clipped_normal_cell_probability,
    hurdle_beta_cell_probability,
    hurdle_beta_interval,
)
from pmvol.fit import fit_hazard


def _frame(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    volume = rng.exponential(20, n)
    spread = rng.uniform(0.01, 0.2, n)
    price = rng.uniform(0.05, 0.95, n)
    eta = -0.5 + 0.7 * np.log1p(volume) - 5 * spread
    release = rng.uniform(0.001, 0.08, n)
    q = release + (1 - release) / (1 + np.exp(-eta))
    updated = rng.random(n) < q
    return (
        pd.DataFrame(
            {
                "price": price,
                "price_next": price,
                "innovation": np.zeros(n),
                "updated": updated.astype(int),
                "lag_updated": rng.integers(0, 2, n),
                "volume": volume,
                "open_interest": rng.exponential(100, n),
                "spread": spread,
                "time_to_resolution": rng.uniform(2, 1000, n),
            }
        ),
        release,
    )


def test_constrained_hazard_predictions_are_feasible_and_calibrated_in_mean():
    frame, release = _frame()
    model = fit_hazard(frame, release, constrained=True)
    predicted = model.predict(frame, release)
    assert np.all(predicted >= release)
    assert predicted.mean() == pytest.approx(frame.updated.mean(), abs=0.015)


def test_unconstrained_fit_can_violate_bound():
    frame, release = _frame()
    release[:] = 0.8
    model = fit_hazard(frame, release, constrained=False)
    predicted = model.predict(frame, release)
    assert np.mean(predicted < release) > 0.5


def test_vectorized_probabilities_and_intervals_are_valid():
    p = np.array([0.05, 0.4, 0.9])
    y = np.array([0.0, 0.4, 1.0])
    r = np.array([0.01, 0.03, 0.2])
    q = np.array([0.2, 0.4, 0.8])
    hb = hurdle_beta_cell_probability(p, y, r, q)
    normal = clipped_normal_cell_probability(p, y, r * p * (1 - p))
    assert np.all((hb > 0) & (hb <= 1))
    assert np.all((normal > 0) & (normal <= 1))
    lo, hi = hurdle_beta_interval(p, r, q)
    assert np.all((0 <= lo) & (lo <= hi) & (hi <= 1))
