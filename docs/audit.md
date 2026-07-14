# Audit of the target paper

Target: Xi, Moallemi, Pai, and Wang (2026), arXiv:2607.08199v2.

## Reproduction status

- The v2 PDF and TeX source were acquired from arXiv on 2026-07-15.
- The paper reports 880,719 active-update out-of-sample contract-hours from
  August 2021 through April 2026 and uses a private or unlinked constructed
  Kalshi panel. No code or row-level data link appears in v2.
- Kalshi's public historical candlestick API now supplies hourly bid, ask,
  last-price, volume, and open-interest fields. This repository uses that API
  for an independent replication. Exact row equality is not assumed.

## Verified model ingredients

The closed-form DR-AS variance is

\[
h_i^2 = \left[\frac{p_i(1-p_i)}{\tau_i}
+K\nu(V_i)\frac{s_i^2}{4}\right]\Delta_i.
\]

The headline target removes observations with exactly zero next-hour price
innovation, retains contracts with at least 48 hourly observations, estimates
parameters in monthly expanding windows, and scores clipped symmetric
normal-reference 95% intervals with a volume-weighted Winkler score.

## Gaps that motivate the extension

1. **No unconditional law.** The main target is update size conditional on a
   move. Appendix D adds zero observations back to the same scale rules but does
   not model update incidence. The conclusion explicitly leaves a hazard model
   for future work.
2. **Variance-to-interval map is not structural.** The Wright-Fisher mechanism
   is bounded, skewed near the boundary, and can absorb. The evaluation instead
   uses a symmetric Gaussian multiplier and clips the result to `[0,1]`.
3. **Finite-step variance budget.** The additive Euler rule is a local
   approximation and can exceed remaining binary uncertainty when the deadline
   and order-flow terms are jointly large. A finite-step information-clock map
   should keep normalized variance in `[0,1]`.
4. **Conditional/unconditional scale mismatch.** If the full conditional
   variance release is `r p(1-p)` and a move occurs with probability `q`, the
   active-update variance is not generally the same object; bounded-martingale
   coherence requires `r <= q`.
5. **Jump-heavy categories.** The paper documents event-concentrated Sports
   moves and explicitly proposes an event clock or jump component. The hurdle
   law provides a minimal bridge between update incidence and move magnitude,
   but scheduled-event features remain an additional empirical extension.

## Threats to validity for this repository

- Public API history may differ from the paper's archived snapshots or cleaning.
- Mid-quote candlesticks can contain extremely wide, economically uninformative
  books; spread and quote-quality filters must be fixed before test evaluation.
- Exact no-change is partly economic inactivity and partly the price lattice.
  Both mixed-measure and rounding-aware scores must be reported.
- Category and series selection must remain frozen after outcome evaluation.
- Contract-level dependence requires cluster resampling; observation-level
  standard errors are invalid.

## Metadata-only sampling amendment (2026-07-15)

The first core selection returned 12,270 qualifying markets, of which 9,854
were daily weather strike contracts. Before any core-panel outcomes were scored,
the protocol added a cap of 1,000 markets per series. When a series exceeds the
cap, markets are selected at evenly spaced ranks in settlement-time order. This
systematic sample preserves the full date range and uses metadata only. The
smoke configuration retains its separate most-recent sampling rule and is never
used for manuscript evidence.
