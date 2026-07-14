"""Expanding-window out-of-sample comparison on the full hourly panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import (
    clipped_normal_cell_probability,
    clipped_normal_randomized_cell_pit,
    hurdle_beta_cell_probability,
    hurdle_beta_interval,
    hurdle_beta_randomized_cell_pit,
    summarize_forecast,
)
from .fit import fit_hazard, fit_order_flow_k
from .model import (
    clock_release,
    dr_as_linear_variance,
    interval_score,
    normal_reference_interval,
)

MODEL_NAMES = {
    "dr_normal": "DR normal",
    "dras_normal": "DR-AS normal",
    "beta_no_hurdle": "Beta, no hurdle",
    "hurdle_posthoc": "Hurdle beta, post-hoc coherence",
    "mhb": "MHB",
}


def _prepare(
    frame: pd.DataFrame,
    max_spread: float,
    minimum_contract_forecasts: int,
) -> pd.DataFrame:
    valid = frame["valid_forecast"]
    if valid.dtype != bool:
        valid = valid.astype(str).str.lower().eq("true")
    valid_counts = frame.loc[valid].groupby("ticker", sort=False).size()
    eligible = valid_counts.loc[valid_counts >= minimum_contract_forecasts].index
    result = frame.loc[
        valid & frame["ticker"].isin(eligible) & (frame["spread"] <= max_spread)
    ].copy()
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
    tick_size: np.ndarray | float,
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


def _observation_ticks(frame: pd.DataFrame, override: float | None) -> np.ndarray:
    if override is not None:
        return np.full(len(frame), override, dtype=float)
    if "tick_size_next" in frame:
        return frame["tick_size_next"].to_numpy(float)
    return np.full(len(frame), 0.005, dtype=float)


def _evaluation_weights(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    volume = np.maximum(frame["volume"].to_numpy(float), 0.0)
    work = pd.DataFrame({"ticker": frame["ticker"].to_numpy(), "volume": volume})
    total_volume = work.groupby("ticker", sort=False)["volume"].transform("sum").to_numpy()
    contract_rows = work.groupby("ticker", sort=False)["volume"].transform("size").to_numpy()
    contract_balanced = np.where(
        total_volume > 0,
        volume / np.maximum(total_volume, 1e-12),
        1.0 / contract_rows,
    )
    return {
        "equal": np.ones(len(frame)),
        "volume": volume,
        "contract_balanced": contract_balanced,
    }


def expanding_backtest(
    panel: pd.DataFrame,
    *,
    max_spread: float = 0.20,
    min_training_rows: int = 1000,
    tick_size: float | None = None,
    level: float = 0.95,
    estimation_weighting: str = "volume",
    minimum_contract_forecasts: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _prepare(panel, max_spread, minimum_contract_forecasts)
    months = sorted(data["forecast_month"].unique())
    records: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    previous_k_original: float | None = None
    previous_k_mhb: float | None = None
    previous_constrained = None
    previous_unconstrained = None

    for month in months:
        test = data.loc[data["forecast_month"] == month]
        train = data.loc[data["timestamp"] < test["timestamp"].min()]
        if len(train) < min_training_rows or test.empty:
            continue

        # Match the target paper for the normal DR-AS baseline: active-update
        # QMLE. MHB instead fits the unconditional variance budget on all hours.
        k_original = fit_order_flow_k(
            train,
            finite_clock=False,
            active_only=True,
            weighting=estimation_weighting,
            initial_k=previous_k_original,
        )
        k_mhb = fit_order_flow_k(
            train,
            finite_clock=True,
            active_only=False,
            weighting=estimation_weighting,
            initial_k=previous_k_mhb,
        )
        train_release = clock_release(
            train["price"],
            train["time_to_resolution"],
            train["spread"],
            train["volume"],
            k=k_mhb,
        )
        constrained = fit_hazard(
            train,
            train_release,
            constrained=True,
            weighting=estimation_weighting,
            initial_model=previous_constrained,
        )
        unconstrained = fit_hazard(
            train,
            train_release,
            constrained=False,
            weighting=estimation_weighting,
            initial_model=previous_unconstrained,
        )
        previous_k_original = k_original
        previous_k_mhb = k_mhb
        previous_constrained = constrained
        previous_unconstrained = unconstrained

        p = test["price"].to_numpy(float)
        y = test["price_next"].to_numpy(float)
        observation_ticks = _observation_ticks(test, tick_size)
        test_release = clock_release(
            p,
            test["time_to_resolution"],
            test["spread"],
            test["volume"],
            k=k_mhb,
        )
        q_mhb = constrained.predict(test, test_release)
        q_raw = unconstrained.predict(test, test_release)
        q_posthoc = np.clip(np.maximum(q_raw, test_release), 0.0, 1.0)

        dr_variance = dr_as_linear_variance(p, test["time_to_resolution"])
        dras_variance = dr_as_linear_variance(
            p,
            test["time_to_resolution"],
            test["spread"],
            test["volume"],
            k=k_original,
        )
        normal_models = {
            "dr_normal": dr_variance,
            "dras_normal": dras_variance,
        }
        beta_models = {
            "beta_no_hurdle": np.ones_like(test_release),
            "hurdle_posthoc": q_posthoc,
            "mhb": q_mhb,
        }
        weight_sets = _evaluation_weights(test)

        prediction = test[
            [
                "ticker",
                "series",
                "category",
                "timestamp",
                "price",
                "price_next",
                "updated",
                "volume",
            ]
        ].copy()
        prediction["tick_size_next"] = observation_ticks
        prediction["month"] = month
        prediction["release"] = test_release
        prediction["q_mhb"] = q_mhb
        prediction["q_unconstrained"] = q_raw
        prediction["q_posthoc"] = q_posthoc
        prediction["unconstrained_violation"] = q_raw < test_release
        prediction["variance_dr_normal"] = dr_variance
        prediction["variance_dras_normal"] = dras_variance

        for slug, variance in normal_models.items():
            probability = clipped_normal_cell_probability(
                p, y, variance, tick_size=observation_ticks
            )
            lower, upper = normal_reference_interval(p, variance)
            prediction[f"nll_{slug}"] = -np.log(probability)
            prediction[f"is_{slug}"] = interval_score(
                lower, upper, y, alpha=1.0 - level
            )
            for weighting, weights in weight_sets.items():
                summary = summarize_forecast(
                    probability, lower, upper, y, weights, alpha=1.0 - level
                )
                records.append(
                    {
                        "month": month,
                        "model": MODEL_NAMES[slug],
                        "model_slug": slug,
                        "weighting": weighting,
                        "n": len(test),
                        "weight_sum": float(weights.sum()),
                        "k_original": k_original,
                        "k_mhb": k_mhb,
                        **summary,
                    }
                )

        for slug, hazard in beta_models.items():
            probability, lower, upper = _model_scores(
                test,
                test_release,
                hazard,
                tick_size=observation_ticks,
                level=level,
            )
            prediction[f"nll_{slug}"] = -np.log(probability)
            prediction[f"is_{slug}"] = interval_score(
                lower, upper, y, alpha=1.0 - level
            )
            for weighting, weights in weight_sets.items():
                summary = summarize_forecast(
                    probability, lower, upper, y, weights, alpha=1.0 - level
                )
                records.append(
                    {
                        "month": month,
                        "model": MODEL_NAMES[slug],
                        "model_slug": slug,
                        "weighting": weighting,
                        "n": len(test),
                        "weight_sum": float(weights.sum()),
                        "k_original": k_original,
                        "k_mhb": k_mhb,
                        **summary,
                    }
                )

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


def calibration_table(
    predictions: pd.DataFrame,
    *,
    levels: tuple[float, ...] = (0.50, 0.80, 0.95),
) -> pd.DataFrame:
    """Pooled interval calibration at pre-specified nominal levels."""

    p = predictions["price"].to_numpy(float)
    y = predictions["price_next"].to_numpy(float)
    release = predictions["release"].to_numpy(float)
    hazards = {
        "beta_no_hurdle": np.ones(len(predictions)),
        "hurdle_posthoc": predictions["q_posthoc"].to_numpy(float),
        "mhb": predictions["q_mhb"].to_numpy(float),
    }
    variances = {
        "dr_normal": predictions["variance_dr_normal"].to_numpy(float),
        "dras_normal": predictions["variance_dras_normal"].to_numpy(float),
    }
    weights_by_name = _evaluation_weights(predictions)
    rows: list[dict[str, object]] = []
    from scipy.stats import norm

    for level in levels:
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for slug, variance in variances.items():
            intervals[slug] = normal_reference_interval(
                p, variance, z=float(norm.ppf((1.0 + level) / 2.0))
            )
        for slug, hazard in hazards.items():
            intervals[slug] = hurdle_beta_interval(p, release, hazard, level=level)

        for slug, (lower, upper) in intervals.items():
            scores = interval_score(lower, upper, y, alpha=1.0 - level)
            covered = (lower <= y) & (y <= upper)
            for weighting, weights in weights_by_name.items():
                total = weights.sum()
                if total <= 0:
                    continue
                rows.append(
                    {
                        "model": MODEL_NAMES[slug],
                        "model_slug": slug,
                        "weighting": weighting,
                        "level": level,
                        "coverage": float(np.sum(weights * covered) / total),
                        "interval_score": float(np.sum(weights * scores) / total),
                        "width": float(np.sum(weights * (upper - lower)) / total),
                    }
                )
    return pd.DataFrame(rows)


def pit_table(
    predictions: pd.DataFrame,
    *,
    tick_size: float | None,
    seed: int = 41731,
) -> pd.DataFrame:
    """Randomized cell-PIT diagnostics without invalid iid p-values."""

    rng = np.random.default_rng(seed)
    uniform = rng.random(len(predictions))
    p = predictions["price"].to_numpy(float)
    y = predictions["price_next"].to_numpy(float)
    observation_ticks = _observation_ticks(predictions, tick_size)
    pit_by_model = {
        "dr_normal": clipped_normal_randomized_cell_pit(
            p,
            y,
            predictions["variance_dr_normal"].to_numpy(float),
            uniform,
            tick_size=observation_ticks,
        ),
        "dras_normal": clipped_normal_randomized_cell_pit(
            p,
            y,
            predictions["variance_dras_normal"].to_numpy(float),
            uniform,
            tick_size=observation_ticks,
        ),
        "mhb": hurdle_beta_randomized_cell_pit(
            p,
            y,
            predictions["release"].to_numpy(float),
            predictions["q_mhb"].to_numpy(float),
            uniform,
            tick_size=observation_ticks,
        ),
    }
    grid = np.linspace(0.05, 0.95, 19)
    rows = []
    for slug, pit in pit_by_model.items():
        cvm_grid = float(np.mean([(np.mean(pit <= point) - point) ** 2 for point in grid]))
        rows.append(
            {
                "model": MODEL_NAMES[slug],
                "model_slug": slug,
                "mean": float(pit.mean()),
                "variance": float(pit.var()),
                "grid_cvm": cvm_grid,
            }
        )
    return pd.DataFrame(rows)


def contract_cluster_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int = 999,
    seed: int = 77123,
) -> pd.DataFrame:
    """Paired contract-cluster bootstrap of score improvements over MHB."""

    rng = np.random.default_rng(seed)
    comparisons = [slug for slug in MODEL_NAMES if slug != "mhb"]
    rows: list[dict[str, object]] = []
    for weighting, base_weights in _evaluation_weights(predictions).items():
        for metric in ("nll", "is"):
            for other in comparisons:
                work = pd.DataFrame(
                    {
                        "ticker": predictions["ticker"].to_numpy(),
                        "weighted_difference": base_weights
                        * (
                            predictions[f"{metric}_{other}"].to_numpy(float)
                            - predictions[f"{metric}_mhb"].to_numpy(float)
                        ),
                        "weight": base_weights,
                    }
                )
                clusters = work.groupby("ticker", sort=False).sum(numeric_only=True)
                clusters = clusters.loc[clusters["weight"] > 0]
                numerator = clusters["weighted_difference"].to_numpy(float)
                denominator = clusters["weight"].to_numpy(float)
                point = float(numerator.sum() / denominator.sum())
                indices = rng.integers(0, len(clusters), size=(draws, len(clusters)))
                boot = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
                rows.append(
                    {
                        "model_a": "MHB",
                        "model_b": MODEL_NAMES[other],
                        "metric": metric,
                        "weighting": weighting,
                        "difference_b_minus_a": point,
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                        "clusters": len(clusters),
                        "draws": draws,
                    }
                )
    return pd.DataFrame(rows)


def active_replication_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int = 999,
    seed: int = 44021,
) -> pd.DataFrame:
    """Paired cluster intervals for DR-AS versus DR on active updates."""

    active = predictions.loc[predictions["updated"].to_numpy(int) == 1]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for weighting, weights in _evaluation_weights(active).items():
        for metric in ("nll", "is"):
            work = pd.DataFrame(
                {
                    "ticker": active["ticker"].to_numpy(),
                    "weighted_difference": weights
                    * (
                        active[f"{metric}_dr_normal"].to_numpy(float)
                        - active[f"{metric}_dras_normal"].to_numpy(float)
                    ),
                    "weight": weights,
                }
            )
            clusters = work.groupby("ticker", sort=False).sum(numeric_only=True)
            clusters = clusters.loc[clusters["weight"] > 0]
            numerator = clusters["weighted_difference"].to_numpy(float)
            denominator = clusters["weight"].to_numpy(float)
            indices = rng.integers(0, len(clusters), size=(draws, len(clusters)))
            boot = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
            rows.append(
                {
                    "model_a": "DR-AS normal",
                    "model_b": "DR normal",
                    "regime": "active",
                    "metric": metric,
                    "weighting": weighting,
                    "difference_b_minus_a": float(numerator.sum() / denominator.sum()),
                    "ci_low": float(np.quantile(boot, 0.025)),
                    "ci_high": float(np.quantile(boot, 0.975)),
                    "clusters": len(clusters),
                    "draws": draws,
                }
            )
    return pd.DataFrame(rows)


def subgroup_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Proper-score summaries by update regime and market category.

    The active-update rows provide a direct independent replication target for
    the conditional evaluation in the source paper. The all-hour and inactive
    rows assess the new unconditional forecasting task.
    """

    frames: list[tuple[str, str, pd.DataFrame]] = []
    regimes = {
        "all": np.ones(len(predictions), dtype=bool),
        "active": predictions["updated"].to_numpy(int) == 1,
        "inactive": predictions["updated"].to_numpy(int) == 0,
    }
    for regime, mask in regimes.items():
        frames.append(("All", regime, predictions.loc[mask]))
        for category, group in predictions.loc[mask].groupby("category", sort=True):
            frames.append((str(category), regime, group))

    rows: list[dict[str, object]] = []
    for category, regime, frame in frames:
        if frame.empty:
            continue
        for weighting, weights in _evaluation_weights(frame).items():
            total = float(weights.sum())
            if total <= 0:
                continue
            for slug, model in MODEL_NAMES.items():
                rows.append(
                    {
                        "category": category,
                        "regime": regime,
                        "model": model,
                        "model_slug": slug,
                        "weighting": weighting,
                        "n": len(frame),
                        "weight_sum": total,
                        "negative_log_score": float(
                            np.average(frame[f"nll_{slug}"], weights=weights)
                        ),
                        "interval_score": float(
                            np.average(frame[f"is_{slug}"], weights=weights)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def hazard_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Proper scores and calibration for the update-incidence margin."""

    outcome = predictions["updated"].to_numpy(float)
    forecasts = {
        "unconstrained": predictions["q_unconstrained"].to_numpy(float),
        "posthoc": predictions["q_posthoc"].to_numpy(float),
        "mhb": predictions["q_mhb"].to_numpy(float),
    }
    labels = {
        "unconstrained": "Separate hazard",
        "posthoc": "Post-hoc coherent hazard",
        "mhb": "MHB coherent hazard",
    }
    rows: list[dict[str, object]] = []
    for weighting, weights in _evaluation_weights(predictions).items():
        if weights.sum() <= 0:
            continue
        for slug, raw_forecast in forecasts.items():
            forecast = np.clip(raw_forecast, 1e-10, 1.0 - 1e-10)
            log_loss = -(outcome * np.log(forecast) + (1.0 - outcome) * np.log1p(-forecast))
            rows.append(
                {
                    "model": labels[slug],
                    "model_slug": slug,
                    "weighting": weighting,
                    "brier_score": float(np.average((forecast - outcome) ** 2, weights=weights)),
                    "log_loss": float(np.average(log_loss, weights=weights)),
                    "predicted_update_rate": float(np.average(forecast, weights=weights)),
                    "observed_update_rate": float(np.average(outcome, weights=weights)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_score_table(
    predictions: pd.DataFrame, calibration: pd.DataFrame
) -> pd.DataFrame:
    """Pooled scores under each evaluation weighting.

    Computing from observations rather than averaging monthly summaries is
    essential for contract-balanced weights, which normalize over a contract's
    full out-of-sample history rather than separately within every month.
    """

    rows: list[dict[str, object]] = []
    for weighting, weights in _evaluation_weights(predictions).items():
        if weights.sum() <= 0:
            continue
        for slug, model in MODEL_NAMES.items():
            diagnostic = calibration.loc[
                (calibration["model_slug"] == slug)
                & (calibration["weighting"] == weighting)
                & np.isclose(calibration["level"], 0.95)
            ].iloc[0]
            rows.append(
                {
                    "model": model,
                    "model_slug": slug,
                    "weighting": weighting,
                    "negative_log_score": float(
                        np.average(predictions[f"nll_{slug}"], weights=weights)
                    ),
                    "interval_score": float(
                        np.average(predictions[f"is_{slug}"], weights=weights)
                    ),
                    "coverage": diagnostic["coverage"],
                    "width": diagnostic["width"],
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/empirical"))
    parser.add_argument("--max-spread", type=float, default=0.20)
    parser.add_argument("--min-training-rows", type=int, default=1000)
    parser.add_argument("--minimum-contract-forecasts", type=int, default=48)
    parser.add_argument(
        "--tick-size",
        type=float,
        default=None,
        help="override market-specific midpoint tick sizes",
    )
    parser.add_argument(
        "--estimation-weighting",
        choices=("equal", "volume", "log_volume"),
        default="volume",
    )
    args = parser.parse_args(argv)

    panel = pd.read_csv(args.panel)
    scores, predictions = expanding_backtest(
        panel,
        max_spread=args.max_spread,
        min_training_rows=args.min_training_rows,
        tick_size=args.tick_size,
        estimation_weighting=args.estimation_weighting,
        minimum_contract_forecasts=args.minimum_contract_forecasts,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "monthly_scores.csv", index=False)
    predictions.to_csv(args.output / "predictions.csv.gz", index=False, compression="gzip")

    calibration = calibration_table(predictions)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    aggregate = aggregate_score_table(predictions, calibration)
    aggregate.to_csv(args.output / "aggregate_scores.csv", index=False)
    pits = pit_table(predictions, tick_size=args.tick_size)
    pits.to_csv(args.output / "pit.csv", index=False)
    bootstrap = contract_cluster_bootstrap(predictions)
    bootstrap.to_csv(args.output / "cluster_bootstrap.csv", index=False)
    replication = active_replication_bootstrap(predictions)
    replication.to_csv(args.output / "active_replication_bootstrap.csv", index=False)
    subgroups = subgroup_score_table(predictions)
    subgroups.to_csv(args.output / "subgroup_scores.csv", index=False)
    hazards = hazard_score_table(predictions)
    hazards.to_csv(args.output / "hazard_scores.csv", index=False)
    metadata = {
        "panel": str(args.panel),
        "max_spread": args.max_spread,
        "min_training_rows": args.min_training_rows,
        "minimum_contract_forecasts": args.minimum_contract_forecasts,
        "tick_size": args.tick_size if args.tick_size is not None else "market_specific",
        "estimation_weighting": args.estimation_weighting,
        "months": sorted(scores["month"].unique().tolist()),
        "test_predictions": int(len(predictions)),
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
