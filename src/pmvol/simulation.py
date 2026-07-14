"""Simulation checks for identification, coherence, and forecast calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import hurdle_beta_cell_probability, hurdle_beta_interval
from .fit import fit_hazard, fit_order_flow_k, hazard_raw_features
from .model import clock_release, feasible_hazard, interval_score


def simulate_panel(n: int, seed: int, true_k: float = 0.7) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    price = rng.beta(1.8, 1.8, n) * 0.98 + 0.01
    tau = np.exp(rng.uniform(np.log(2.0), np.log(2500.0), n))
    volume = rng.lognormal(2.0, 1.4, n)
    zero_volume = rng.random(n) < 0.35
    volume[zero_volume] = 0.0
    spread = np.clip(rng.lognormal(-3.3, 0.8, n), 0.005, 0.4)
    open_interest = rng.lognormal(4.0, 1.2, n)
    lag_updated = rng.integers(0, 2, n)
    release = clock_release(price, tau, spread, volume, k=true_k)

    skeleton = pd.DataFrame(
        {
            "price": price,
            "time_to_resolution": tau,
            "spread": spread,
            "volume": volume,
            "open_interest": open_interest,
            "lag_updated": lag_updated,
        }
    )
    raw = hazard_raw_features(skeleton)
    standardized = (raw - raw.mean(axis=0)) / raw.std(axis=0)
    coefficient = np.array([-0.6, 0.7, 0.15, -0.25, -0.35, 0.4, 0.55])
    eta = np.column_stack([np.ones(n), standardized]) @ coefficient
    hazard = feasible_hazard(release, eta)

    active = rng.random(n) < hazard
    concentration = np.maximum(hazard / np.maximum(release, 1e-12) - 1.0, 1e-8)
    alpha = price * concentration
    beta = (1.0 - price) * concentration
    price_next = price.copy()
    price_next[active] = rng.beta(alpha[active], beta[active])
    innovation = price_next - price
    skeleton["price_next"] = price_next
    skeleton["innovation"] = innovation
    skeleton["updated"] = active.astype(int)
    return skeleton, release, hazard


def run_once(n_train: int, n_test: int, seed: int, true_k: float) -> dict[str, float]:
    train, train_release_true, _ = simulate_panel(n_train, seed, true_k)
    test, test_release_true, test_hazard_true = simulate_panel(n_test, seed + 1, true_k)

    fitted_k = fit_order_flow_k(
        train, finite_clock=True, active_only=False, weighting="equal"
    )
    train_release = clock_release(
        train.price,
        train.time_to_resolution,
        train.spread,
        train.volume,
        k=fitted_k,
    )
    hazard_model = fit_hazard(train, train_release, constrained=True)
    test_release = clock_release(
        test.price,
        test.time_to_resolution,
        test.spread,
        test.volume,
        k=fitted_k,
    )
    test_hazard = hazard_model.predict(test, test_release)
    probability = hurdle_beta_cell_probability(
        test.price.to_numpy(),
        test.price_next.to_numpy(),
        test_release,
        test_hazard,
        tick_size=0.005,
    )
    lower, upper = hurdle_beta_interval(test.price.to_numpy(), test_release, test_hazard)
    score = interval_score(lower, upper, test.price_next.to_numpy())
    covered = (lower <= test.price_next.to_numpy()) & (test.price_next.to_numpy() <= upper)
    return {
        "seed": seed,
        "true_k": true_k,
        "fitted_k": fitted_k,
        "relative_k_error": (fitted_k - true_k) / true_k,
        "release_mae": float(np.mean(np.abs(test_release - test_release_true))),
        "hazard_mae": float(np.mean(np.abs(test_hazard - test_hazard_true))),
        "hazard_bias": float(np.mean(test_hazard - test_hazard_true)),
        "update_rate_error": float(test_hazard.mean() - test.updated.mean()),
        "coverage_95": float(covered.mean()),
        "mean_interval_score": float(score.mean()),
        "mean_negative_log_score": float(-np.log(probability).mean()),
        "minimum_coherence_margin": float(np.min(test_hazard - test_release)),
    }


def reliability_plot(predictions: np.ndarray, outcomes: np.ndarray, output: Path) -> None:
    bins = np.linspace(0, 1, 11)
    index = np.clip(np.digitize(predictions, bins) - 1, 0, len(bins) - 2)
    rows = []
    for i in range(len(bins) - 1):
        mask = index == i
        if mask.sum() < 20:
            continue
        rows.append((predictions[mask].mean(), outcomes[mask].mean(), mask.sum()))
    points = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", label="ideal")
    ax.scatter(points[:, 0], points[:, 1], s=np.sqrt(points[:, 2]) * 10, color="#176B87")
    ax.set(xlabel="Predicted update probability", ylabel="Observed update rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("MHB update-hazard calibration (simulation)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/simulation"))
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--train", type=int, default=30_000)
    parser.add_argument("--test", type=int, default=15_000)
    parser.add_argument("--true-k", type=float, default=0.7)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    records = [
        run_once(args.train, args.test, seed=9200 + i, true_k=args.true_k)
        for i in range(args.replications)
    ]
    results = pd.DataFrame(records)
    results.to_csv(args.output / "replications.csv", index=False)
    summary = results.agg(["mean", "std", "min", "max"]).T
    summary.to_csv(args.output / "summary.csv")

    demo, release, _ = simulate_panel(40_000, 8871, args.true_k)
    fitted_k = fit_order_flow_k(demo, finite_clock=True, active_only=False)
    fitted_release = clock_release(
        demo.price, demo.time_to_resolution, demo.spread, demo.volume, k=fitted_k
    )
    fitted_hazard = fit_hazard(demo, fitted_release, constrained=True).predict(
        demo, fitted_release
    )
    reliability_plot(
        fitted_hazard, demo.updated.to_numpy(float), args.output / "hazard_reliability.png"
    )
    metadata = {
        "replications": args.replications,
        "n_train": args.train,
        "n_test": args.test,
        "true_k": args.true_k,
        "all_coherent": bool((results.minimum_coherence_margin >= -1e-12).all()),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()

