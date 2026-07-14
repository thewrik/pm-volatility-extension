# Claims ledger

Every manuscript claim must appear here with an evidence state.

| ID | Candidate claim | State | Required evidence |
|---|---|---|---|
| T1 | Any bounded martingale with update probability `q` and normalized variance release `r` satisfies `r <= q`. | proved | Formal proof and edge cases in manuscript; unit tests for constructions. |
| T2 | The MHB kernel has mean `p`, variance `r p(1-p)`, and atom `1-q` at `p`. | proved | Algebraic proof plus numerical/Monte Carlo tests. |
| T3 | The finite-step clock release is in `[0,1]` and matches DR-AS to first order. | proved | Expansion and numerical regression test. |
| N1 | Applying the coherence inequality and MHB distribution to prediction-market volatility is novel. | provisional | Structured search of arXiv, SSRN, Crossref/Scholar; compare closest papers. |
| E1 | MHB improves unconditional proper scores over DR-AS normal intervals. | rejected | MHB has a better equal-weight tick log score, but DR-AS is better on the primary volume-weighted and contract-balanced log and interval scores; paired cluster intervals exclude zero for the volume-weighted losses. |
| E2 | MHB's performance relative to DR-AS is heterogeneous across market categories. | tested | Descriptive subgroups show a volume-weighted density gain in Economics and reversals in Politics and Sports; no multiplicity-adjusted category claim is made. |
| E3 | Coherence constraints improve calibration rather than only density fit. | rejected | MHB improves tick log score over the post-hoc coherent hurdle, but does not improve interval score or hazard Brier score. |
| E4 | MHB improves full-panel tick log score over the beta-without-hurdle and post-hoc-hurdle ablations. | tested | Volume-weighted paired contract-cluster intervals for comparator minus MHB are `[0.620, 1.066]` and `[0.133, 1.420]`; the result is stable across spread thresholds. |
| E5 | The public panel independently reproduces the active-update DR-AS improvement over deadline resolution alone. | tested | Volume-weighted 95% interval-score improvement is `0.1667`, with contract-cluster interval `[0.0158, 0.2620]`. |

States: `proposed`, `provisional`, `proved`, `tested`, `rejected`.
