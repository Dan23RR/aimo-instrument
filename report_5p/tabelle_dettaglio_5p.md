# Tabelle di dettaglio del report da 5 pagine (2/11/2026)

Le Tabelle 1, 2 e 3 del testo principale come erano nella versione verificata del 28/10-2/11 (corpo v3), copiate parola per parola dal corpo v3 con uno script, il 2/11. Il testo principale ora le condensa in Table 1, Figure 1 e Table 2; costruisci_5p.py le mette per intero nell'Appendice J come Tables J.1 a J.3.

**Table 1.** Channels tested against ground truth (DeepSeek and the public pairs unless stated). Accuracy differences are in points against the baseline named in each pre-registration. H1, H7 and T are computed on the 335 published pairs, counting the 52 duplicated DeepSeek pairs twice; counted once, H1 is −7.4, H7 −1.4 and T +7.1, and no verdict changes.

| Channel | Entry | Result | Verdict |
|---|---|---|---|
| Problem text, 32 features, grouped 5-fold | H1 | 0.675 vs 0.761 (−8.7) | dead |
| Robustness as a property of the problem | H2 | −1.3 (z = 0.88) with duplicated rows counted once; +3.4 (z = 4.04) only when counted twice | dead |
| Labels, base accuracy, difficulty order across models | H3 to H5, H7 | −7.3, −11.8, −25.4, −1.2 | dead |
| Official uncertainty features, logistic regression, grouped CV | H10 | 0.7744 vs 0.756, inside one standard error | reaches the proposal's accuracy (method substituted) |
| Cached features for other models; per-target thresholds | H12, H13 | −16.5; +7.3 from thresholds chosen on the scored items, −2.8 when forced to rank | dead; ceiling withdrawn |
| Our families; translation; self-consistency; the organizers' families regenerated (pilot) | H19, H22, H33, H32 | kappa 0.02; kappa −0.10; AUROC 0.502; kappa −0.001 | dead |
| Representation shift under perturbation; length control | H30, H31 | AUROC 0.615 (planned threshold 0.655; character count 0.705); 0.555 | dead |
| Our linear probe on the residual stream, nested CV | H36 | AUROC 0.710; about +0.02 over character count | survives, largely explained |
| The organizers' published probe artifact | H35 | AUROC 0.739 (pre-registered aggregation) and 0.709 (its own), 0.62 and 0.57 residualized on log character count; shuffled-label controls 0.68 | survives, largely explained |
| Trivial-robust rule (robust iff base accuracy < 0.05), post hoc | T | +6.0 on the 335 public pairs, +0.0 on `val-sample` | trap |
| Identical bundles re-executed (warmup, 17 items) | H18 | 11, 9, 11 and 11 correct | noise confirmed |

**Table 2.** Forty public problems, twenty-three executions in six configurations; generated text hashed per item. Items crossing: items whose prediction changes across the runs of a configuration (crossings grow with the number of runs; per pair of runs, 15.3 under contention, 13.3 with batch size 1, 12.7 on the second card, 11.4 by default). Deterministic algorithms: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)` and deterministic cuDNN; compilation disabled: `TORCH_COMPILE_DISABLE=1`.

| Configuration | Runs | Byte-identical items | Items crossing | Seconds per item |
|---|---|---|---|---|
| Default (batch 8) | 5 | 0/40 | 24/40 | 26.0 |
| Batch size 1 | 3 | 1/40 | 20/40 | 72.3 |
| Two concurrent processes on one card | 6 | 0/40 | 32/40 | 54.7 |
| Second physical card | 3 | 0/40 | 19/40 | 26.0 |
| Deterministic algorithms and cuBLAS workspace setting | 3 | 40/40 | 0 | 27.4 |
| Compilation disabled | 3 | 40/40 | 0 | 27.8 |

**Table 3.** Agreement of regenerated labels (Cohen's kappa; in brackets, 95% percentile bootstrap intervals over problems, 2,000 resamples). Robust rates are over all 164 problems (the 4 without accepted variants count as not robust); kappas are over the problems labeled in both series (159 for A and B, 160 for A′ and for the official labels); the main-phase column uses the problems labeled in both draws under the 0.10/0.25 thresholds.

| Model | Robust in draw A | kappa(A, B), conjunction | kappa(A, A′), same variants, new seed | kappa(A, B), main-phase rule | kappa with official labels |
|---|---|---|---|---|---|
| Skywork | 72.6% | 0.439 [0.28, 0.59] | 0.789 [0.67, 0.89] | 0.49 [0.17, 0.76] (132 problems) | no earlier labels |
| DeepSeek | 37.8% | 0.413 [0.26, 0.55] | 0.464* [0.33, 0.59] | 0.56 [0.36, 0.73] (100) | 0.025 [−0.12, 0.18]; 0.02 [−0.09, 0.13] under the main-phase rule (133) |
| Olmo | 65.9% | 0.378 [0.21, 0.54] | 0.653 [0.52, 0.78] | 0.41 [0.09, 0.69] (133) | no earlier labels |
| Qwen3.5-4B | 9.1% | 0.178 [−0.06, 0.39] | not run | 0.34 [0.13, 0.53] (86) | no earlier labels |

\* On DeepSeek, A′ also received the answer-format line of Section 5.3, so this entry mixes seed and prompt.
