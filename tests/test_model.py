import numpy as np
import pytest

from pmvol.model import (
    HurdleBeta,
    clock_release,
    dr_as_linear_variance,
    feasible_hazard,
    interval_score,
    normal_reference_interval,
)


@pytest.mark.parametrize(
    "price,release,hazard",
    [(0.5, 0.05, 0.4), (0.1, 0.01, 0.8), (0.9, 0.3, 1.0)],
)
def test_hurdle_beta_moments_monte_carlo(price, release, hazard):
    law = HurdleBeta(price, release, hazard)
    draws = law.sample(500_000, np.random.default_rng(71099))
    assert draws.mean() == pytest.approx(price, abs=2e-3)
    assert draws.var() == pytest.approx(release * price * (1 - price), rel=0.025)
    assert np.mean(draws != price) == pytest.approx(hazard, abs=2e-3)


def test_terminal_limit_is_exact_three_point_law():
    law = HurdleBeta(price=0.3, release=0.2, update_probability=0.2)
    draws = law.sample(300_000, np.random.default_rng(42))
    assert set(np.unique(draws)) == {0.0, 0.3, 1.0}
    assert draws.mean() == pytest.approx(0.3, abs=2e-3)
    assert draws.var() == pytest.approx(0.2 * 0.3 * 0.7, abs=2e-3)


def test_hazard_map_enforces_coherence():
    release = np.linspace(0, 1, 101)
    hazard = feasible_hazard(release, np.linspace(-50, 50, 101))
    assert np.all(hazard >= release)
    assert np.all(hazard <= 1)
    assert hazard[-1] == 1


def test_deadline_clock_is_exact_delta_over_tau_without_order_flow():
    p = np.array([0.1, 0.5, 0.9])
    tau = np.array([2.0, 10.0, 100.0])
    release = clock_release(p, tau, delta=1.0)
    np.testing.assert_allclose(release, 1.0 / tau, rtol=1e-12, atol=1e-12)


def test_clock_is_bounded_where_linear_euler_can_be_infeasible():
    p, tau, spread, volume, k = 0.5, 1.1, 0.8, 10_000.0, 10.0
    linear = dr_as_linear_variance(p, tau, spread, volume, k=k)
    release = clock_release(p, tau, spread, volume, k=k)
    assert linear > p * (1 - p)
    assert 0 <= release <= 1


def test_clock_matches_linear_dr_as_to_first_order():
    p, tau, spread, volume, k = 0.37, 1000.0, 0.02, 4.0, 0.3
    delta = 1e-5
    linear = dr_as_linear_variance(p, tau, spread, volume, k=k, delta=delta)
    exact = clock_release(p, tau, spread, volume, k=k, delta=delta) * p * (1 - p)
    assert exact == pytest.approx(float(linear), rel=2e-6)


def test_ppf_respects_atom_and_interval_contains_price_when_atom_is_large():
    law = HurdleBeta(price=0.42, release=0.01, update_probability=0.10)
    assert law.ppf(0.5) == 0.42
    lo, hi = law.central_interval(0.95)
    assert lo <= 0.42 <= hi


def test_binned_probability_is_valid_at_boundaries_and_atom():
    law = HurdleBeta(price=0.25, release=0.05, update_probability=0.4)
    for y in [0.0, 0.25, 1.0]:
        assert 0 < law.bin_probability(y, tick_size=0.01) <= 1
    assert law.bin_probability(0.25, tick_size=0.01) >= 0.6


def test_interval_score_and_normal_reference_baseline():
    lo, hi = normal_reference_interval([0.01, 0.99], [0.04, 0.04])
    np.testing.assert_array_equal(lo, [0.0, pytest.approx(0.5980072031)])
    np.testing.assert_array_equal(hi, [pytest.approx(0.4019927969), 1.0])
    score = interval_score(0.2, 0.8, np.array([0.5, 0.1, 0.9]))
    np.testing.assert_allclose(score, [0.6, 4.6, 4.6])


def test_invalid_parameters_fail_loudly():
    with pytest.raises(ValueError):
        HurdleBeta(0.5, 0.3, 0.2)
    with pytest.raises(ValueError):
        HurdleBeta(0.0, 0.1, 0.2)
    with pytest.raises(ValueError):
        clock_release(1.2, 10)

