"""Expanding-window out-of-sample comparison on the full hourly panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    clipped_normal_cell_probability,
    hurdle_beta_cell_probability,
    hurdle_beta_interval,
    summarize_forecast,
)
from .fit import fit_hazard, fit_order_flow_k
from .model import clock_release, dr_as_linear_variance, normal_reference_interval


def _prepare(frame: pd.DataFrame, max_spread: float) -> pd.DataFrame:
    valid = frame["valid_forecast"]
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().eq("true")
    result = frame.loc[valid & (frame["spread"] <= max_spread)].copy()
    for column in ("timestamp", "next_timestamp", "deadline"):
        result[column] = pd.to_datetime(result[column], utc=True)
    result["updated"] = result["updated"].astype(int)
    result["lag_updated"] = result["lag_updated"].astype(int)
    return result.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def _model_scores(
    test: pd.DataFrame,
    release: np.ndarray,
    hazard: np.ndarray,
    *,
    tick_size: float,
    level: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = hurdle_beta_cell_probability(
        test["price"].to_numpy(float),
        test["price_next"].to_numpy(float),
        release,
        hazard,
        tick_size=tick_size,
    )
    lower, upper = hurdle_beta_interval(
        test["price"].to_numpy(float), release, hazard, level=level
    )
    return probability, lower, upper


def expanding_backtest(
    panel: pd.DataFrame,
    *,
    max_spread: float = 0.20,
    min_training_rows: int = 1000,
    tick_size: float = 0.005,
    level: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _prepare(panel, max_spread)
    months = sorted(data["forecast_month"].unique())
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for month in months:
        test = data.loc[data["forecast_month"] == month]
        train = data.loc[data["timestamp"] < test["timestamp"].min()]
        if len(train) < min_training_rows or test.empty:
            continue

        # Match the target paper for the normal DR-AS baseline: active-update
        # QMLE. MHB instead fits the unconditional variance budget on all hours.
        k_original = fit_order_flow_k(
            train, finite_clock=False, active_only=True, weighting="equal"
        )
        k_mhb = fit_order_flow_k(
            train, finite_clock=True, active_only=False, weighting="equal"
        )
        train_release = clock_release(
            train["price"],
            train["time_to_resolution"],
            train["spread"],
            train["volume"],
            k=k_mhb,
        )
        constrained = fit_hazard(train, train_release, constrained=True)
        unconstrained = fit_hazard(train, train_release, constrained=False)

        p = test["price"].to_numpy(float)
        y = test["price_next"].to_numpy(float)
        test_release = clock_release(
            p,
            test["time_to_resolution"],
            test["spread"],
            test["volume"],
            k=k_mhb,
        )
        q_mhb = constrained.predict(test, test_release)
        q_raw = unconstrained.predict(test, test_release)
        q_posthoc = np.maximum(q_raw, test_release + 1e-10)

        dr_variance = dr_as_linear_variance(p, test["time_to_resolution"])
        dras_variance = dr_as_linear_variance(
            p,
            test["time_to_resolution"],
            test["spread"],
            test["volume"],
            k=k_original,
        )
        normal_models = {
            "DR normal": dr_variance,
            "DR-AS normal": dras_variance,
        }
        beta_models = {
            "Beta, no hurdle": np.ones_like(test_release),
            "Hurdle beta, post-hoc coherence": q_posthoc,
            "MHB": q_mhb,
        }
        weight_sets = {
            "equal": np.ones(len(test)),
            "volume": np.maximum(test["volume"].to_numpy(float), 0.0),
        }

        for name, variance in normal_models.items():
            probability = clipped_normal_cell_probability(
                p, y, variance, tick_size=tick_size
            )
            lower, upper = normal_reference_interval(p, variance)
            for weighting, weights in weight_sets.items():
                summary = summarize_forecast(
                    probability, lower, upper, y, weights, alpha=1.0 - level
                )
                records.append(
                    {
                        "month": month,
                        "model": name,
                        "weighting": weighting,
                        "n": len(test),
                        "k_original": k_original,
                        "k_mhb": k_mhb,
                        **summary,
                    }
                )

        for name, hazard in beta_models.items():
            probability, lower, upper = _model_scores(
                test, test_release, hazard, tick_size=tick_size, level=level
            )
            for weighting, weights in weight_sets.items():
                summary = summarize_forecast(
                    probability, lower, upper, y, weights, alpha=1.0 - level
                )
                records.append(
                    {
                        "month": month,
                        "model": name,
                        "weighting": weighting,
                        "n": len(test),
                        "k_original": k_original,
                        "k_mhb": k_mhb,
                        **summary,
                    }
                )

        prediction = test[
            ["ticker", "series", "category", "timestamp", "price", "price_next", "updated", "volume"]
        ].copy()
        prediction["month"] = month
        prediction["release"] = test_release
        prediction["q_mhb"] = q_mhb
        prediction["q_unconstrained"] = q_raw
        prediction["unconstrained_violation"] = q_raw < test_release
        predictions.append(prediction)
        print(
            f"{month}: train={len(train):,} test={len(test):,} "
            f"K(original)={k_original:.4g} K(MHB)={k_mhb:.4g} "
            f"violations={np.mean(q_raw < test_release):.3%}",
            flush=True,
        )

    if not records:
        raise RuntimeError("no month had enough training observations")
    return pd.DataFrame(records), pd.concat(predictions, ignore_index=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/empirical"))
    parser.add_argument("--max-spread", type=float, default=0.20)
    parser.add_argument("--min-training-rows", type=int, default=1000)
    parser.add_argument("--tick-size", type=float, default=0.005)
    args = parser.parse_args(argv)

    panel = pd.read_csv(args.panel)
    scores, predictions = expanding_backtest(
        panel,
        max_spread=args.max_spread,
        min_training_rows=args.min_training_rows,
        tick_size=args.tick_size,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "monthly_scores.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv.gz", index=False, compression="gzip")

    aggregate = (
        scores.groupby(["model", "weighting"], as_index=False)
        .apply(
            lambda group: pd.Series(
                {
                    metric: np.average(group[metric], weights=group["n"])
                    for metric in ("negative_log_score", "interval_score", "coverage", "width")
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    aggregate.to_csv(args.output / "aggregate_scores.csv", index=False)
    metadata = {
        "panel": str(args.panel),
        "max_spread": args.max_spread,
        "min_training_rows": args.min_training_rows,
        "tick_size": args.tick_size,
        "months": sorted(scores["month"].unique().tolist()),
        "test_predictions": int(len(predictions)),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()

