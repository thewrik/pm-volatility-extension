"""Generate manuscript tables and a machine-readable empirical summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ORDER = [
    "dr_normal",
    "dras_normal",
    "beta_no_hurdle",
    "hurdle_posthoc",
    "mhb",
]

MODEL_TEX = {
    "dr_normal": "DR normal",
    "dras_normal": "DR--AS normal",
    "beta_no_hurdle": "Beta, no hurdle",
    "hurdle_posthoc": "Hurdle beta, post-hoc",
    "mhb": "MHB",
}


def _number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _interval(low: float, high: float, digits: int = 4) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def _percent(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}\\%"


def _comparison_sentence(row: pd.Series, label: str, metric: str) -> str:
    point = float(row["difference_b_minus_a"])
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    unit = "tick log score" if metric == "nll" else "95\\% interval score"
    if low > 0:
        interpretation = "lower"
    elif high < 0:
        interpretation = "higher"
    else:
        interpretation = "not distinguishable from zero by the cluster interval"
    if interpretation in {"lower", "higher"}:
        reported_low, reported_high = (low, high) if low > 0 else (-high, -low)
        return (
            f"Against {label}, MHB's {unit} is {interpretation} by "
            f"{abs(point):.4f} (95\\% contract-cluster interval "
            f"{_interval(reported_low, reported_high)})."
        )
    return (
        f"Against {label}, the {label.removeprefix('the ')}-minus-MHB {unit} difference is {point:.4f}; "
        f"the 95\\% contract-cluster interval {_interval(low, high)} includes zero."
    )


def build_report(
    result_dir: Path,
    manifest_path: Path,
    output_tex: Path,
    sensitivity_root: Path | None,
) -> dict[str, object]:
    aggregate = pd.read_csv(result_dir / "aggregate_scores.csv")
    bootstrap = pd.read_csv(result_dir / "cluster_bootstrap.csv")
    replication = pd.read_csv(result_dir / "active_replication_bootstrap.csv")
    calibration = pd.read_csv(result_dir / "calibration.csv")
    hazards = pd.read_csv(result_dir / "hazard_scores.csv")
    subgroups = pd.read_csv(result_dir / "subgroup_scores.csv")
    predictions = pd.read_csv(result_dir / "predictions.csv.gz")
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    volume_scores = aggregate.loc[aggregate["weighting"] == "volume"].set_index(
        "model_slug"
    )
    equal_scores = aggregate.loc[aggregate["weighting"] == "equal"].set_index(
        "model_slug"
    )
    contract_scores = aggregate.loc[
        aggregate["weighting"] == "contract_balanced"
    ].set_index("model_slug")
    prediction_volume = np.maximum(predictions["volume"].to_numpy(float), 0.0)
    update = predictions["updated"].to_numpy(float)
    weighted_update_rate = float(np.average(update, weights=prediction_volume))
    panel_summary = (
        predictions.groupby("category", sort=True)
        .agg(
            forecast_origins=("ticker", "size"),
            contracts=("ticker", "nunique"),
            active_updates=("updated", "sum"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    panel_summary.to_csv(result_dir / "panel_summary.csv", index=False)
    contract_volume = predictions.groupby("ticker", sort=False)["volume"].sum().sort_values(
        ascending=False
    )
    top_two_volume_share = float(contract_volume.head(2).sum() / contract_volume.sum())

    lines = [
        "\\subsection{Locked out-of-sample results}",
        "",
        (
            f"The public-API acquisition selected {manifest['markets_selected']:,} contracts and "
            f"successfully retrieved {manifest['markets_succeeded']:,}. After forecast-validity, "
            f"quote-quality, minimum-history, and expanding-window rules, the locked evaluation "
            f"contains {len(predictions):,} hourly forecasts on {predictions['ticker'].nunique():,} "
            f"contracts across {predictions['month'].nunique()} test months. The unweighted update "
            f"rate is {_percent(predictions['updated'].mean())}; its forecast-origin "
            f"volume-weighted counterpart is {_percent(weighted_update_rate)}."
        ),
        "",
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Full-panel out-of-sample distribution forecasts. Lower is better for both scores.}",
        "\\label{tab:main-results}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Tick log score & 95\\% IS & Coverage & Width \\\\",
        "\\midrule",
    ]
    for slug in MODEL_ORDER:
        row = volume_scores.loc[slug]
        lines.append(
            f"{MODEL_TEX[slug]} & {_number(row['negative_log_score'])} & "
            f"{_number(row['interval_score'])} & {_number(row['coverage'], 3)} & "
            f"{_number(row['width'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )

    replication_is = replication.loc[
        (replication["metric"] == "is") & (replication["weighting"] == "volume")
    ].iloc[0]
    active_volume = subgroups.loc[
        (subgroups["category"] == "All")
        & (subgroups["regime"] == "active")
        & (subgroups["weighting"] == "volume")
    ].set_index("model_slug")
    lines.extend(
        [
            (
                "On active updates, the independent replication preserves the source paper's "
                f"structural ordering: DR--AS posts a volume-weighted 95\\% interval score of "
                f"{active_volume.loc['dras_normal', 'interval_score']:.4f}, versus "
                f"{active_volume.loc['dr_normal', 'interval_score']:.4f} for deadline resolution "
                f"alone. The DR-minus-DR--AS improvement is {replication_is['difference_b_minus_a']:.4f} "
                f"with a 95\\% contract-cluster interval "
                f"{_interval(replication_is['ci_low'], replication_is['ci_high'])}."
            ),
            "",
        ]
    )
    lines.extend(
        [
            (
                f"Volume is concentrated: the two largest contracts contribute "
                f"{_percent(top_two_volume_share)} of evaluation weight. Under the pre-specified "
                f"contract-balanced robustness rule, MHB's tick log score is "
                f"{contract_scores.loc['mhb', 'negative_log_score']:.4f}, versus "
                f"{contract_scores.loc['dras_normal', 'negative_log_score']:.4f} for DR--AS "
                f"normal and {contract_scores.loc['hurdle_posthoc', 'negative_log_score']:.4f} "
                f"for the post-hoc hurdle; the corresponding interval scores are "
                f"{contract_scores.loc['mhb', 'interval_score']:.4f}, "
                f"{contract_scores.loc['dras_normal', 'interval_score']:.4f}, and "
                f"{contract_scores.loc['hurdle_posthoc', 'interval_score']:.4f}. Under equal "
                f"observation weights, by contrast, MHB has the best tick log score "
                f"({equal_scores.loc['mhb', 'negative_log_score']:.4f} versus "
                f"{equal_scores.loc['dras_normal', 'negative_log_score']:.4f} for DR--AS) but a "
                f"worse interval score ({equal_scores.loc['mhb', 'interval_score']:.4f} versus "
                f"{equal_scores.loc['dras_normal', 'interval_score']:.4f})."
            ),
            "",
        ]
    )

    categories = sorted(
        category for category in subgroups["category"].unique() if category != "All"
    )
    lines.extend(
        [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Volume-weighted category results (descriptive).}",
            "\\label{tab:category-results}",
            "\\small",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Category & DR--AS log & MHB log & DR--AS IS & MHB IS \\\\",
            "\\midrule",
        ]
    )
    for category in categories:
        subset = subgroups.loc[
            (subgroups["category"] == category)
            & (subgroups["regime"] == "all")
            & (subgroups["weighting"] == "volume")
        ].set_index("model_slug")
        lines.append(
            f"{category} & {_number(subset.loc['dras_normal', 'negative_log_score'])} & "
            f"{_number(subset.loc['mhb', 'negative_log_score'])} & "
            f"{_number(subset.loc['dras_normal', 'interval_score'])} & "
            f"{_number(subset.loc['mhb', 'interval_score'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
            (
                "The MHB density advantage over DR--AS appears in Economics but reverses in "
                "Politics and Sports; interval scores do not improve. These category results are "
                "descriptive, not multiplicity-adjusted. The two pre-specified daily weather "
                "series contribute no contracts after the 48-valid-origin rule, which is a "
                "limitation of this public-API replication rather than a basis for changing the "
                "locked filter."
            ),
            "",
        ]
    )

    for other, label in (
        ("dras_normal", "the active-fit DR--AS normal benchmark"),
        ("beta_no_hurdle", "the beta law without a hurdle"),
        ("hurdle_posthoc", "the post-hoc coherent hurdle"),
    ):
        for metric in ("nll", "is"):
            row = bootstrap.loc[
                (bootstrap["model_b"] == MODEL_TEX[other].replace("--", "-"))
                & (bootstrap["metric"] == metric)
                & (bootstrap["weighting"] == "volume")
            ]
            # The CSV uses display labels from backtest.py; fall back to slug order.
            if row.empty:
                display = {
                    "dras_normal": "DR-AS normal",
                    "beta_no_hurdle": "Beta, no hurdle",
                    "hurdle_posthoc": "Hurdle beta, post-hoc coherence",
                }[other]
                row = bootstrap.loc[
                    (bootstrap["model_b"] == display)
                    & (bootstrap["metric"] == metric)
                    & (bootstrap["weighting"] == "volume")
                ]
            lines.append(_comparison_sentence(row.iloc[0], label, metric))
        lines.append("")

    lines.extend(
        [
            "\\begin{table}[ht]",
            "\\centering",
            "\\caption{Volume-weighted scores by update regime.}",
            "\\label{tab:regime-results}",
            "\\small",
            "\\begin{tabular}{llrr}",
            "\\toprule",
            "Regime & Model & Tick log score & 95\\% IS \\\\",
            "\\midrule",
        ]
    )
    for regime in ("all", "active", "inactive"):
        for slug in ("dras_normal", "beta_no_hurdle", "hurdle_posthoc", "mhb"):
            row = subgroups.loc[
                (subgroups["category"] == "All")
                & (subgroups["regime"] == regime)
                & (subgroups["model_slug"] == slug)
                & (subgroups["weighting"] == "volume")
            ].iloc[0]
            label = regime.capitalize() if slug == "dras_normal" else ""
            lines.append(
                f"{label} & {MODEL_TEX[slug]} & {_number(row['negative_log_score'])} & "
                f"{_number(row['interval_score'])} \\\\"
            )
        if regime != "inactive":
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    mhb_calibration = calibration.loc[
        (calibration["model_slug"] == "mhb") & (calibration["weighting"] == "volume")
    ].sort_values("level")
    coverage_text = ", ".join(
        f"{int(100 * row.level)}\\%: {_percent(row.coverage)}"
        for row in mhb_calibration.itertuples()
    )
    mhb_hazard = hazards.loc[
        (hazards["model_slug"] == "mhb") & (hazards["weighting"] == "volume")
    ].iloc[0]
    raw_hazard = hazards.loc[
        (hazards["model_slug"] == "unconstrained") & (hazards["weighting"] == "volume")
    ].iloc[0]
    violation_rate = float(predictions["unconstrained_violation"].mean())
    lines.extend(
        [
            (
                f"MHB central-interval coverage at the pre-specified levels is {coverage_text}. "
                f"Its volume-weighted update-hazard Brier score is {mhb_hazard['brier_score']:.4f}, "
                f"versus {raw_hazard['brier_score']:.4f} for the separately fitted hazard. The "
                f"separate hazard violates $q\\ge r$ at {_percent(violation_rate, 2)} of test "
                "origins before "
                "post-hoc repair."
            ),
            "",
        ]
    )

    sensitivity_rows: list[dict[str, object]] = []
    for slug in ("hurdle_posthoc", "mhb"):
        row = volume_scores.loc[slug]
        sensitivity_rows.append(
            {
                "threshold": run["max_spread"],
                "model_slug": slug,
                "negative_log_score": row["negative_log_score"],
                "interval_score": row["interval_score"],
            }
        )
    if sensitivity_root is not None and sensitivity_root.exists():
        for directory in sorted(sensitivity_root.glob("spread-*")):
            path = directory / "aggregate_scores.csv"
            if not path.exists():
                continue
            scores = pd.read_csv(path)
            for slug in ("hurdle_posthoc", "mhb"):
                row = scores.loc[
                    (scores["model_slug"] == slug) & (scores["weighting"] == "volume")
                ].iloc[0]
                candidate = {
                    "threshold": json.loads((directory / "run.json").read_text())["max_spread"],
                    "model_slug": slug,
                    "negative_log_score": row["negative_log_score"],
                    "interval_score": row["interval_score"],
                }
                sensitivity_rows = [
                    existing
                    for existing in sensitivity_rows
                    if not (
                        existing["threshold"] == candidate["threshold"]
                        and existing["model_slug"] == slug
                    )
                ]
                sensitivity_rows.append(
                    {
                        **candidate,
                    }
                )
        if len({row["threshold"] for row in sensitivity_rows}) > 1:
            sensitivity = pd.DataFrame(sensitivity_rows)
            sensitivity.to_csv(result_dir / "sensitivity_summary.csv", index=False)
            lines.extend(
                [
                    "\\begin{table}[ht]",
                    "\\centering",
                    "\\caption{Pre-specified spread-threshold sensitivity.}",
                    "\\label{tab:spread-sensitivity}",
                    "\\small",
                    "\\begin{tabular}{lrrrr}",
                    "\\toprule",
                    "Max spread & Post-hoc log & MHB log & Post-hoc IS & MHB IS \\\\",
                    "\\midrule",
                ]
            )
            for threshold in sorted(sensitivity["threshold"].unique()):
                subset = sensitivity.loc[sensitivity["threshold"] == threshold].set_index(
                    "model_slug"
                )
                lines.append(
                    f"{threshold:.2f} & {_number(subset.loc['hurdle_posthoc', 'negative_log_score'])} & "
                    f"{_number(subset.loc['mhb', 'negative_log_score'])} & "
                    f"{_number(subset.loc['hurdle_posthoc', 'interval_score'])} & "
                    f"{_number(subset.loc['mhb', 'interval_score'])} \\\\"
                )
            lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    summary = {
        "result_dir": str(result_dir),
        "manifest_sha256": manifest["output_sha256"],
        "test_predictions": len(predictions),
        "test_contracts": int(predictions["ticker"].nunique()),
        "test_months": run["months"],
        "unweighted_update_rate": float(predictions["updated"].mean()),
        "volume_weighted_update_rate": weighted_update_rate,
        "unconstrained_violation_rate": violation_rate,
        "top_two_volume_share": top_two_volume_share,
        "volume_scores": volume_scores[
            ["negative_log_score", "interval_score", "coverage", "width"]
        ].to_dict(orient="index"),
        "sensitivity_runs": len({row["threshold"] for row in sensitivity_rows}),
    }
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/empirical/main"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/core_panel.csv.manifest.json"),
    )
    parser.add_argument("--sensitivity-root", type=Path, default=Path("results/empirical"))
    parser.add_argument("--output", type=Path, default=Path("paper/generated/results.tex"))
    args = parser.parse_args(argv)
    summary = build_report(args.results, args.manifest, args.output, args.sensitivity_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
