# Novelty search protocol

## Candidate contribution

An unconditional, boundary-aware distribution forecast for prediction-market
prices that couples information release and the no-update hazard through the
martingale restriction `r <= q`, with proper mixed-distribution scoring.

## Closest work to distinguish

- Xi et al. (2026): structural DR-AS variance; conditions on active updates;
  names but does not build the hazard extension.
- Dalen (2025): logit jump-diffusion and derivative kernel; not an explicit
  no-update hurdle coupled by a bounded-martingale variance inequality.
- Xu (2026): local-volatility theory for absorbed event-market martingales;
  must be checked carefully for an equivalent hazard/variance restriction.
- Catania, Di Mari, and Santucci de Magistris (2019): zero-inflated dynamic
  mixtures for high-frequency asset prices; not specific to bounded posterior
  martingales or prediction-market deadline clocks.
- Aktug and Torul (2026): quote-update hazards in a prediction-market event
  study; no joint distributional volatility model.

## Search requirements before a novelty claim

Search title/abstract/full text for combinations of:

- prediction market + hurdle / zero-inflated / no-update / stale price
- bounded martingale + update probability + conditional variance
- prediction market + density forecast / distribution forecast / CRPS
- Wright-Fisher + atom / hurdle / beta approximation / transition density
- event market + information clock + hazard

Record the exact query, date, database, candidate papers screened, and the
feature-level distinction. Absence from a search result is not proof of novelty.

## Search snapshot: 2026-07-15

Databases searched through web indexing: arXiv, SSRN, Crossref-indexed journal
pages, and general scholarly web search. Queries included:

- `prediction market volatility jump diffusion probability forecast martingale`
- `prediction market update hazard`
- `prediction market zero-inflated`
- `prediction market distributional forecast volatility`
- `Wright Fisher transition density prediction market forecast interval`
- `bounded martingale update probability conditional variance`
- `hurdle beta martingale`
- `Beta mixture no change price martingale`
- `"A Local-Volatility Theory of Prediction Markets" PDF Xu 7012278`
- `"normalized variance" "update probability" martingale`
- `"hurdle beta" martingale prediction market`
- `"variance release" "update hazard" prediction market`

Screened primary papers and working papers:

| Work | Relevant overlap | Distinction from this project |
|---|---|---|
| [Xi et al. (2026)](https://arxiv.org/abs/2607.08199) | Wright-Fisher deadline variance, order-flow variance, full-panel robustness. | Explicitly leaves the update/no-update hazard for future work; no mixed predictive law or `r <= q` restriction. |
| [Dalen (2025)](https://arxiv.org/abs/2510.15205) | Martingale logit jump-diffusion, jump intensity, scheduled events. | Continuous-time kernel and derivative layer; no observed no-change atom coupled to normalized variance release. |
| [Xu (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7012278) | Absorbed event-market martingales, boundary-degenerate local volatility, information clocks, and a simplex structure theorem. | The full 25-page indexed abstract focuses on admissible local-volatility shapes and transient-mispricing autocorrelation; it reports no hurdle law or update-incidence inequality. Full PDF access remained blocked by SSRN and must be checked before a final novelty claim. |
| [Catania et al. (2019)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3349118) | Zero-inflated discrete price changes and density forecasts. | General high-frequency equity model with latent Skellam mixtures; no binary-payoff boundary or martingale coupling between zero mass and variance. |
| [Hol\'y (2023)](https://arxiv.org/abs/2211.12376) | Zero-inflated Skellam GARCH for discrete prices. | General intraday prices; support is not `[0,1]` and the extensive/intensive margins are not constrained by remaining Bernoulli uncertainty. |
| [Han and Irie (2024)](https://arxiv.org/abs/2403.10945) | Zero-inflated stochastic volatility and interval calibration. | CPI components with exact zeros, not prediction-market posterior martingales. |
| [Roa et al. (2022)](https://arxiv.org/abs/2212.11442) | Wright-Fisher transition-density approximations and critique of Gaussian/Beta approximations near fixation. | Population-genetics transition approximation; no update hurdle or prediction-market empirical model. It limits, rather than duplicates, the beta-kernel claim here. |
| [Aktug and Torul (2026)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6642687) | Quote-update hazard in a prediction-market event study. | Identifies informational delay around one CPI leak; no joint volatility distribution or coherence law. |

Provisional conclusion: zero-inflated volatility, bounded event-market
martingales, and prediction-market jump kernels all exist separately. The
specific inequality linking normalized variance release to the observed update
hazard, its sharp distributional construction, and its use as a constrained
extension of DR-AS were not located. This remains a provisional negative-search
finding until the full Xu paper, citation chains, and a fresh search at
submission time are checked.
