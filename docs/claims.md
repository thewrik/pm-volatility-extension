# Claims ledger

Every manuscript claim must appear here with an evidence state.

| ID | Candidate claim | State | Required evidence |
|---|---|---|---|
| T1 | Any bounded martingale with update probability `q` and normalized variance release `r` satisfies `r <= q`. | proved | Formal proof and edge cases in manuscript; unit tests for constructions. |
| T2 | The MHB kernel has mean `p`, variance `r p(1-p)`, and atom `1-q` at `p`. | proved | Algebraic proof plus numerical/Monte Carlo tests. |
| T3 | The finite-step clock release is in `[0,1]` and matches DR-AS to first order. | proved | Expansion and numerical regression test. |
| N1 | Applying the coherence inequality and MHB distribution to prediction-market volatility is novel. | provisional | Structured search of arXiv, SSRN, Crossref/Scholar; compare closest papers. |
| E1 | MHB improves unconditional proper scores over DR-AS normal intervals. | untested | Locked expanding-window backtest, contract-cluster confidence intervals. |
| E2 | The gain is concentrated in inactive/boundary/event-heavy regimes. | untested | Pre-specified subgroup analysis with multiplicity control or scoped interpretation. |
| E3 | Coherence constraints improve calibration rather than only sharpness. | untested | Reliability curves, multi-level coverage, PIT/randomized PIT, and score decomposition. |

States: `proposed`, `provisional`, `proved`, `tested`, `rejected`.

