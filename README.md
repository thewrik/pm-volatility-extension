# When Nothing Happens

This repository develops a distributional extension of Xi, Moallemi, Pai, and
Wang (2026), *Volatility in Prediction Markets: A Structural Approach*
([arXiv:2607.08199](https://arxiv.org/abs/2607.08199)). The working contribution
is a **martingale-coherent hurdle-beta (MHB)** model for unconditional
prediction-market price changes.

This is a one-shotting exercise for **5.6-Sol**. The repository is intended as
an auditable research artifact, but the accuracy, correctness, and novelty of
model-generated outputs remain subject to independent verification.

The original DR-AS model forecasts the scale of a move conditional on an active
price update. MHB instead forecasts the entire next-period distribution,
including the atom at no change. It enforces a restriction that follows from
bounded martingale prices:

\[
0 \le r_t := \frac{\operatorname{Var}(p_{t+1}\mid\mathcal F_t)}
                     {p_t(1-p_t)}
\le q_t := \Pr(p_{t+1}\ne p_t\mid\mathcal F_t) \le 1.
\]

The model uses

\[
p_{t+1}\mid\mathcal F_t \sim
(1-q_t)\,\delta_{p_t}
+q_t\,\mathrm{Beta}\!\left(p_t\kappa_t,(1-p_t)\kappa_t\right),
\qquad \kappa_t=q_t/r_t-1,
\]

which has conditional mean exactly `p_t`, unconditional variance exactly
`r_t p_t(1-p_t)`, and no-change probability exactly `1-q_t` in the continuous
price idealization. A rounding-aware score is provided for observed tick data.

## Locked findings

The repository contains a completed theorem/simulation audit and a frozen
public-API backtest with 1,007,560 out-of-sample hourly forecasts on 1,404
contracts across 50 months.

- The active-update replication confirms that DR-AS improves on deadline
  resolution alone: the volume-weighted 95% interval-score gain is 0.1667, with
  contract-cluster interval `[0.0158, 0.2620]`.
- MHB improves volume-weighted tick log score over the beta-without-hurdle and
  post-hoc-coherence ablations. The paired improvement intervals are
  `[0.620, 1.066]` and `[0.133, 1.420]`.
- MHB does **not** dominate the active-fit DR-AS normal benchmark. DR-AS is
  better on the primary volume-weighted log and interval scores, and on the
  contract-balanced versions. MHB has the best equal-observation log score.
- Category results are heterogeneous: MHB's density advantage over DR-AS
  appears in Economics and reverses in Politics and Sports. These subgroup
  results are descriptive.

Claims and rejected hypotheses are tracked in [`docs/claims.md`](docs/claims.md);
protocol amendments and threats to validity are explicit in
[`docs/audit.md`](docs/audit.md). The manuscript treats the negative boundary as
evidence that coherence is necessary but not sufficient without a richer jump
or scheduled-event component.

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest

# Small public-API validation panel (cached and restartable)
.venv/bin/pmv-download --config config/core_panel.json

# Theory/simulation checks and expanding-window empirical evaluation
.venv/bin/pmv-simulate --output results/simulation
.venv/bin/pmv-backtest --panel data/processed/core_panel.csv.gz \
  --output results/empirical/main
.venv/bin/pmv-report --results results/empirical/main
```

The downloader uses only public, unauthenticated Kalshi market-data endpoints.
Raw and processed market data are intentionally not committed; a manifest with
request parameters and hashes is produced for provenance.

## Repository map

- `src/pmvol/`: model, fitting, data acquisition, simulation, and backtesting
- `tests/`: mathematical, numerical, and leakage-regression tests
- `docs/`: audit, novelty search, claims ledger, and reproducibility protocol
- `paper/`: manuscript source and generated figures/tables
- `config/`: frozen empirical panel definitions

## Scope and non-claims

The beta active component is a moment-coherent forecasting kernel, not claimed
to be the exact finite-time Wright-Fisher transition law. The empirical study
uses public candlesticks and therefore need not exactly reproduce the authors'
private or separately constructed hourly panel. Results must be described as an
independent replication unless row-level equality can be established.
