import numpy as np
import pandas as pd
import pytest

from pmvol.evaluation import (
    clipped_normal_cell_probability,
    clipped_normal_randomized_cell_pit,
    hurdle_beta_cell_probability,
    hurdle_beta_interval,
    hurdle_beta_randomized_cell_pit,
)
from pmvol.backtest import (
    active_replication_bootstrap,
    contract_cluster_bootstrap,
    hazard_score_table,
    subgroup_score_table,
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

    warm = fit_hazard(frame, release, constrained=True, initial_model=model)
    np.testing.assert_allclose(warm.predict(frame, release), predicted, atol=2e-5)


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

    varying_ticks = hurdle_beta_cell_probability(
        p, y, r, q, tick_size=np.array([0.0005, 0.005, 0.01])
    )
    assert np.all((varying_ticks > 0) & (varying_ticks <= 1))


def test_randomized_cell_pits_stay_in_unit_interval():
    p = np.array([0.05, 0.4, 0.9])
    y = np.array([0.0, 0.4, 1.0])
    r = np.array([0.01, 0.03, 0.2])
    q = np.array([0.2, 0.4, 0.8])
    u = np.array([0.1, 0.5, 0.9])
    hb = hurdle_beta_randomized_cell_pit(p, y, r, q, u)
    normal = clipped_normal_randomized_cell_pit(p, y, r * p * (1 - p), u)
    assert np.all((0 <= hb) & (hb <= 1))
    assert np.all((0 <= normal) & (normal <= 1))


def test_vectorized_terminal_limit_is_three_point_law():
    p = np.array([0.3, 0.3, 0.3])
    y = np.array([0.0, 0.3, 1.0])
    probability = hurdle_beta_cell_probability(
        p,
        y,
        np.full(3, 0.2),
        np.full(3, 0.2),
        tick_size=0.005,
    )
    np.testing.assert_allclose(probability, [0.14, 0.8, 0.06], rtol=1e-12, atol=1e-12)
    lower, upper = hurdle_beta_interval(
        np.array([0.3]), np.array([0.2]), np.array([0.2]), level=0.95
    )
    np.testing.assert_array_equal(lower, [0.0])
    np.testing.assert_array_equal(upper, [1.0])


def test_subgroups_and_cluster_bootstrap_preserve_paired_improvements():
    frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "category": ["Macro", "Macro", "Sports", "Sports"],
            "updated": [0, 1, 0, 1],
            "volume": [1.0, 2.0, 3.0, 4.0],
        }
    )
    for slug in ("dr_normal", "dras_normal", "beta_no_hurdle", "hurdle_posthoc", "mhb"):
        frame[f"nll_{slug}"] = 1.0 if slug == "mhb" else 2.0
        frame[f"is_{slug}"] = 0.5 if slug == "mhb" else 0.75

    subgroups = subgroup_score_table(frame)
    overall_active = subgroups.loc[
        (subgroups["category"] == "All")
        & (subgroups["regime"] == "active")
        & (subgroups["model_slug"] == "mhb")
        & (subgroups["weighting"] == "equal")
    ]
    assert overall_active.iloc[0]["n"] == 2
    assert overall_active.iloc[0]["negative_log_score"] == pytest.approx(1.0)

    boot = contract_cluster_bootstrap(frame, draws=99, seed=5)
    assert np.allclose(boot.loc[boot["metric"] == "nll", "difference_b_minus_a"], 1.0)
    assert np.allclose(boot.loc[boot["metric"] == "is", "difference_b_minus_a"], 0.25)

    replication = active_replication_bootstrap(frame, draws=99, seed=6)
    assert np.allclose(
        replication.loc[replication["metric"] == "nll", "difference_b_minus_a"],
        0.0,
    )

    frame["q_unconstrained"] = [0.2, 0.8, 0.2, 0.8]
    frame["q_posthoc"] = frame["q_unconstrained"]
    frame["q_mhb"] = frame["q_unconstrained"]
    hazards = hazard_score_table(frame)
    assert len(hazards) == 9
    assert np.allclose(hazards["observed_update_rate"].iloc[:3], 0.5)
