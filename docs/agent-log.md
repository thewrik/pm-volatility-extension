# Agent Activity Log

This log is a public, sanitized process record for the 5.6-Sol one-shotting
exercise. It is not a private chain-of-thought transcript. Instead, it records
the observable parsing path: inputs reviewed, decisions made, claims weakened,
verification performed, and remaining checks for independent readers.

## Scope

- Target paper: Xi, Moallemi, Pai, and Wang (2026), *Volatility in Prediction
  Markets: A Structural Approach*, arXiv:2607.08199.
- Goal: produce a novel, significant, correct, and defensible research
  extension in a new repository.
- Final artifact: a paper, code, tests, locked empirical outputs, public README,
  and layperson-friendly web explainer.
- Required caveat: this is a one-shotting exercise for 5.6-Sol; model output
  accuracy, correctness, and novelty remain subject to independent verification.

## Parsing and Work Log

### 1. Problem framing

- Parsed the source paper as a deadline-resolution plus active-search
  volatility model for prediction-market price changes.
- Identified a gap: the original conditional volatility framing focuses on move
  scale once an active update occurs, while many observed market intervals have
  no price change.
- Chose a distributional extension rather than another scalar volatility
  regression, because the no-change atom is a distributional feature.

### 2. Theoretical constraint

- Formulated the bounded-martingale restriction for prices in `[0, 1]`: the
  normalized unconditional variance release `r` cannot exceed the update
  probability `q`.
- Kept the claim narrow: `0 <= r <= q <= 1` for bounded martingale prices, with
  explicit equality and edge-case characterization in the manuscript.
- Avoided claiming that the beta active component is the exact finite-time
  Wright-Fisher transition law.

### 3. Model construction

- Built the martingale-coherent hurdle-beta model:
  `(1-q) delta_p + q Beta(p*kappa, (1-p)*kappa)`.
- Set `kappa = q/r - 1`, giving exact conditional mean `p`, exact unconditional
  variance `r p(1-p)`, and exact no-change mass `1-q` in the continuous-price
  idealization.
- Added tick-aware scoring because observed prediction-market prices live on
  rounded price grids, not a continuous interval.

### 4. Empirical protocol

- Used public Kalshi historical market data rather than private data.
- Locked an expanding-window hourly backtest before writing final claims.
- Preserved a manifest for processed data provenance while excluding large raw
  and processed datasets from Git.
- Treated the study as an independent replication, not a row-identical
  reproduction of the original authors' panel.

### 5. Results parsing

- Preserved the positive active-update replication result: DR-AS improves on
  deadline resolution alone.
- Reported MHB's positive density result only where supported: it improves
  tick-aware log score over beta-without-hurdle and post-hoc-coherence
  ablations.
- Preserved the negative boundary: MHB does not dominate the active-fit DR-AS
  normal benchmark on the primary volume-weighted and contract-balanced scores.
- Marked subgroup patterns as descriptive, not confirmatory.

### 6. Novelty check

- Screened the closest accessible prior work and documented the search in
  `docs/novelty.md`.
- Marked novelty as provisional because one close SSRN paper had an inaccessible
  full PDF during the audit.
- Avoided claiming peer-reviewed novelty; the repo asks readers to refresh the
  literature search before citation.

### 7. Verification performed

- Ran the Python test suite: 21 tests passed.
- Ran linting during the paper build workflow.
- Built the final PDF and visually inspected rendered pages.
- Built the web explainer and verified local and live page rendering.
- Published the repository to GitHub and the public explainer to GitHub Pages.
- Removed the temporary Surge deployment after switching to GitHub Pages.

### 8. Remaining verification burden

- Independently check the theorem proof and equality cases.
- Re-run the full data download and backtest from a clean environment.
- Re-check the novelty search, especially inaccessible or newly posted papers.
- Compare against the original authors' private or separately constructed panel
  if row-level data become available.
- Stress-test the model on scheduled-event clusters where jump mixtures may be
  more appropriate than a single beta active component.

## Public Links

- Explainer: https://thewrik.github.io/pm-volatility-extension/
- Repository: https://github.com/thewrik/pm-volatility-extension
- Original paper: https://arxiv.org/abs/2607.08199
- Claims ledger: `docs/claims.md`
- Novelty notes: `docs/novelty.md`
- Audit notes: `docs/audit.md`
