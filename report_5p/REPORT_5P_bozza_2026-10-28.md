# The Instrument Behind the Leaderboard: Re-Execution Noise and Label Reliability in the AIMO Interpretability Challenge

**Daniel Culotta** (Codabench: `dan23rr`), independent researcher

*Five-page version, 28 October 2026, condensed from the nine-page draft of 22 October, whose Sections 2 to 7 are given as Appendix J, and from the full report (draft v2.4, 23 September 2026); these remain the reference for every figure below. Results of the per-problem validation protocol and of the final test set will be added when available; their places are marked [TBD].*

## Abstract

We treat the AIMO Interpretability benchmark, which asks whether a language model's answer to a math problem survives perturbations of the problem, as a measurement instrument, and report three negative results. First, with DeepSeek-R1-0528-Qwen3-8B on the evaluation workers' GPU model, inside the official runtime image and pipeline, 24 of 40 items change side of the decision threshold across five executions on the same inputs; twenty-three executions fall into fourteen numerical states, byte-identical within a state. Second, the label, regenerated under the rule its public labels follow with a language-model rewriter and our own prompts, decoding and answer key, agrees with itself at Cohen's kappa 0.18 to 0.44 across four models and with the official DeepSeek labels at 0.03; a simulation of the organizers' process overstates by about 0.4 when run on our labels, so the reliability of the official label remains unmeasured, and an unspecified answer-format line moves the share of Qwen3.5-4B's problems passing the label's base condition from 0 to 88%. Third, of thirty-one hypotheses recorded in dated files and tested mostly on DeepSeek, twenty-one died by their kill criterion, and the two probes that survived, ours and the organizers' published artifact, are largely redundant with problem length. We list fifty-four corrections to our own statements.

## 1 Introduction

A leaderboard is a measurement: a score is read off an instrument made of a label, an evaluation set, an inference pipeline and the hardware it runs on, and when the quantity to be predicted is itself produced by a stochastic process, the instrument can fail in ways that no single submission can see. We study the instrument of the AIMO Interpretability Challenge, in which a submission predicts, for a given model and math problem, whether the model's answer is robust to perturbations of the problem [1]. We approached the task as a falsification campaign: every hypothesis was recorded in a dated file, and every kill is reported. Of thirty-one entries, twenty-seven carried a kill criterion fixed before they ran (three of them in a written plan rather than a frozen pre-registration; for H2 it was written after its concordance had been observed): twenty-one died by it, five survived it and one is not established (Appendix A); Appendix C lists fifty-four corrections to our own statements. Our findings concern the instrument: re-execution noise with structure (Section 4), a label that our regeneration does not reproduce and whose reliability is unmeasured (Section 5), and apparent signals that trace back, wholly or mostly, to something other than the model's internals, or are largely redundant with problem length (Section 3). Nondeterminism of LLM inference is documented in general [2, 3]; we add its structure and consequences on this benchmark's hardware and pipeline, and two candidate remedies. The organizers reached a compatible conclusion: the best leaderboard scores did not generalize to unseen problems, and relying on the balanced distribution of the validation labels is "a sort of overfitting" [6]. We make no claim about any participant's method. Appendix J gives an extended version of Sections 2 to 7.

## 2 The task, the labels and the public data

**Task and labels.** A submission is a function that receives a model identifier, a reasoning effort and a list of problems, and returns one robustness verdict per problem; it runs offline on a single GPU within one hour [4, 5] (70 minutes since 15 October, when evaluation moved to one call per problem [6]). Five models are evaluated: Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think and DeepSeek-R1-0528-Qwen3-8B in both tracks, and gpt-oss-120b in the Main track only. Since 16 September the private validation tasks have 21 items (Main) and 14 (Small); the final test will use 30 to 50 new problems [5, 6]. On all 379 labeled rows of the earlier public datasets, the label equals the conjunction "no perturbation lowered accuracy" (a majority rule fits them equally well); the exact criterion is not published [4]. The main-phase train set thresholds the largest accuracy drop across perturbation families: robust at most 0.10, non-robust at least 0.25 [5].

**Public data.** Six datasets aggregate to 335 (model, problem) pairs over 174 problems; 52 pairs repeat the DeepSeek measurements under a second model name. DeepSeek is the only evaluated model with substantial ground truth under the earlier rule: 164 labeled pairs, 26 of them derived by us under the conjunction (a majority rule would call 21 of these 26 robust rather than 4). In the three aggregated datasets the field `permutation_type` is a JSON list if and only if the item is robust, an artifact of the aggregation that is absent at inference; and the public validation sample, `val-sample`, has no labeled pair with base accuracy below 0.05, while 20 of the 80 robust published pairs have base accuracy 0 and are robust only trivially under the conjunction. On the 21 and 14 validation items the binomial standard error of an accuracy near one half is 10.9 and 13.4 points, and with robust base rates of 10/21 and 7/14 a constant scores 11/21 and 7/14 at best.

## 3 The falsification campaign

Table 1 condenses the campaign; Appendix A gives every entry with its baseline and caveats.

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

**Three traps.** Rule T gains six points on the public pairs and nothing on `val-sample`, where the trivially robust subpopulation is absent. The oracle H13 chose each target's threshold on the very items it was scored on; forced to rank them, it gave −2.8. And H2, the only warmup hypothesis about a property of the problem that had survived, owed its survival to one measurement published under two model names and counted twice.

**Length and probes.** The character count alone orders the DeepSeek labels at AUROC 0.705. Our logistic probe on DeepSeek's last-token hidden states (layer and regularization chosen by inner cross-validation) reaches 0.710 out of fold and survives its permutation null (60 permutations where 500 were pre-registered), but correlates with length at Spearman +0.55: over 20 second-level splits, character count alone gives 0.696, the probe 0.697 and both together 0.714. The organizers' published DeepSeek probe artifact orders our labels at 0.709 under its own inference aggregation but 0.57 once residualized on log character count; its controls trained on shuffled labels give 0.68, and at its layer each seed's probe points the same way whether trained on true or shuffled labels (cosine 0.93 to 0.998). These length controls and artifact analyses were post hoc. Under its own aggregation the artifact thus orders the labels mostly through length, and our probe is largely redundant with length; what our probe reads is not established. Apart from rule T, no channel we scored in accuracy beats the majority constant by more than about one standard error on public ground truth, mostly DeepSeek's.

## 4 Re-execution noise on the evaluation workers' GPU model

On a virtual machine that the organizers provided on 8 September, with two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (the evaluation workers' model), all runs used the official runtime image and the unmodified official extraction and regressor, on DeepSeek.

**Table 2.** Forty public problems, twenty-three executions in six configurations; generated text hashed per item. Items crossing: items whose prediction changes across the runs of a configuration (crossings grow with the number of runs; per pair of runs, 15.3 under contention, 13.3 with batch size 1, 12.7 on the second card, 11.4 by default). Deterministic algorithms: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True, warn_only=True)` and deterministic cuDNN; compilation disabled: `TORCH_COMPILE_DISABLE=1`.

| Configuration | Runs | Byte-identical items | Items crossing | Seconds per item |
|---|---|---|---|---|
| Default (batch 8) | 5 | 0/40 | 24/40 | 26.0 |
| Batch size 1 | 3 | 1/40 | 20/40 | 72.3 |
| Two concurrent processes on one card | 6 | 0/40 | 32/40 | 54.7 |
| Second physical card | 3 | 0/40 | 19/40 | 26.0 |
| Deterministic algorithms and cuBLAS workspace setting | 3 | 40/40 | 0 | 27.4 |
| Compilation disabled | 3 | 40/40 | 0 | 27.8 |

**Fourteen states.** The twenty-three executions fall into fourteen numerical states, byte-identical within a state; 61 of the 91 pairs of states share no generation. The state belongs to the process, not to the card: two runs launched in the same second on one card landed in different states, while two of three runs on the second card reproduced the first card's default state exactly. We infer, without proof, that the state is fixed when the process starts. Two pre-registered predictions failed: batch size 1 gave three states in three runs, and disabling compilation gave a state of its own rather than a default one. As only 7 of the 40 problems are robust and crossings largely cancel, accuracy moved much less than predictions (21 to 24 correct under the default configuration on either card, 24 to 28 under contention), always below the 33 of the majority constant.

**Candidate remedies.** In three sequential runs on one card, either deterministic algorithms with a fixed cuBLAS workspace or disabling compilation made generation byte-identical at 5 to 7% runtime cost. Neither was tested under contention or across cards, and the default configuration also gave three identical sequential runs before its fourth diverged: these are candidates, not fixes. Nor does determinism buy correctness: the two configurations, run on different cards, settled on different states, predicting 15 and 12 of 40 problems robust and agreeing on only 21.

**The official score channel.** In an exploratory measurement, not pre-registered, the official regressor score was extracted three times for the 174 public problems with the same code and image. The two runs without the cuBLAS workspace setting correlate at Spearman 0.24 over the 164 labeled problems and give AUROC 0.487 and 0.602, a difference that is not significant (paired bootstrap 95% interval −0.015 to 0.246): the ordering does not reproduce, which is not a demonstrated change of signal. The shipped threshold predicts 25.6% of problems robust in one state and 33.5% in the other (base rate 24.4%). As 137 of the 164 problems are in the regressor's training set, this concerns the stability of its inputs, not new problems.

## 5 Can the label be reproduced?

We identified the organizers' perturbation families in the public data (rephrase, rename, domain, distract, typos), regenerated them with gpt-oss-120b prompted with the public examples, and gated each variant with gpt-oss-120b (the text must differ, the length ratio lie between 0.7 and 4.0 and the solver's answer not change; prompts in Appendix I). An answer counts as correct when it matches gpt-oss-120b's answer on the original problem (the public data carry no answers). On the organizers' machine, with vLLM 0.28.0 outside the official image, we labeled four models with ten variants per family in two independent draws (A and B; about 40 accepted variants per problem), ten samples on the base problem and one per variant, and re-sampled draw A's variants with a new seed (A′).

**Table 3.** Agreement of regenerated labels (Cohen's kappa; in brackets, 95% percentile bootstrap intervals over problems, 2,000 resamples). Robust rates are over all 164 problems (the 4 without accepted variants count as not robust); kappas are over the problems labeled in both series (159 for A and B, 160 for A′ and for the official labels); the main-phase column uses the problems labeled in both draws under the 0.10/0.25 thresholds.

| Model | Robust in draw A | kappa(A, B), conjunction | kappa(A, A′), same variants, new seed | kappa(A, B), main-phase rule | kappa with official labels |
|---|---|---|---|---|---|
| Skywork | 72.6% | 0.439 [0.28, 0.59] | 0.789 [0.67, 0.89] | 0.49 [0.17, 0.76] (132 problems) | no earlier labels |
| DeepSeek | 37.8% | 0.413 [0.26, 0.55] | 0.464* [0.33, 0.59] | 0.56 [0.36, 0.73] (100) | 0.025 [−0.12, 0.18]; 0.02 [−0.09, 0.13] under the main-phase rule (133) |
| Olmo | 65.9% | 0.378 [0.21, 0.54] | 0.653 [0.52, 0.78] | 0.41 [0.09, 0.69] (133) | no earlier labels |
| Qwen3.5-4B | 9.1% | 0.178 [−0.06, 0.39] | not run | 0.34 [0.13, 0.53] (86) | no earlier labels |

\* On DeepSeek, A′ also received the answer-format line of Section 5.3, so this entry mixes seed and prompt.

**5.1 More variants do not help, and our noise does not explain the gap.** Subsampling DeepSeek's variants from 2 to 40 per problem, the mean kappa(A, B) stays between 0.36 and 0.47 with no upward trend, and the mean concordance with the official label between −0.02 and +0.03. Corrected for our process's unreliability, the observed 0.025 becomes 0.039 (about 0.27 at the upper end of its interval): our noise could hide at most a weak agreement. Under the main-phase rule, pre-registered on the same answers, test-retest stays below 0.60 on every model; post hoc, our DeepSeek robust rate is about the same whether the official label is robust or not (77.5% and 74.2% in draw A), and the largest accuracy drops do not correlate (Spearman −0.09 and −0.01), so no stricter threshold on our side would recover theirs.

**5.2 What drives the instability.** On Skywork, re-sampling the same variants with a new seed agrees far more than drawing new variants (0.79 against 0.44), so there the label depends mainly on which variants are drawn; on Olmo the seed alone accounts for about half of the instability. A plug-in simulation of the organizers' process from their published prediction counts gives a test-retest kappa of 0.94, but it resamples answers to the same variants and never lets a robust item fail; on our draw A it gives 0.64 to 0.84 against a measured 0.18 to 0.44, overstating by about 0.4 on every model, so the reliability of the official label remains unmeasured. The official DeepSeek labels also come from a different sampling regime [5] and decoding [4]: what the data show is narrower than a verdict on the label, namely that variants produced by our process measure something else. The organizers have decided not to publish their labeling code [6].

**5.3 An unspecified prompt moves the label.** Fed the bare problem text, Qwen3.5-4B closes its answer in the expected format in only 5 of its 1,640 base samples in draw A; with one instruction line requesting the format, the share of problems passing the label's base condition (at least 6 of 10 samples agreeing with the reference answer) moved from 0 to 145 of 164 (88%). A label defined by correctness is not comparable across prompts that the benchmark leaves unspecified.

## 6 Our entries, and the evaluations the organizers requested

**V8, the entry that does nothing.** Pre-registered on 13 September, its entire logic is `return [len(p) >= 98 for p in problems]`, with the threshold at the lower median of the 174 public problems and the direction fixed on the public pairs. On the original main-phase tasks it scored 11/21 and 10/19, identical to our measuring entry v6 (the official extraction and regressor on each model's own greedy generation) and to the organizers' own entries on the leaderboard (0.52 and 0.53); on 21 and 19 items these ties are compatible with length carrying little signal (95% Wilson intervals 0.32 to 0.72 and 0.32 to 0.73) without showing that it carries none, and none of these entries scored above the majority constant. A one-line rule that ties a pipeline loading up to five models for 8.5 to 14 minutes is a measurement of the pipeline. On the replacement tasks of 16 September it predicted robust for every item: 10/21 and 7/14, the base rates.

**The evaluations requested for reports [5].** On the public main-phase train set (revision of 17 September), V8 scores 46/71 (0.648), below the 52/71 (0.732) of the all-robust constant. After the change of 15 October [6], we resubmitted V8 unchanged on 28 October (Codabench 1016713, Main; 1016730, Small) with an all-robust constant as control (1016732, 1016735): V8, now called once per problem through its legacy signature, again predicted robust for every item and scored 10/21 and 7/14, as did the constant. Final test: [TBD: nominated IDs and scores]. Our pre-registered predictions will be scored there (Appendix B): P1, that the top-ranked entry's final accuracy will lie within two binomial standard errors of the test set's majority constant, and P2; if either fails, the label has learnable structure that we did not find.

## 7 Recommendations

**R1. Fix the numerical state and test the fix under load.** Verify either remedy of Section 4 under the workers' real load and across cards, and document the chosen setting, which selects the fixed point; isolating submissions on a card may matter as much. **R2. Repeat only where it decides something.** At the warmup's re-execution standard deviation (1.0 item on 17, from four runs, itself uncertain), a one-item gap needs about eight runs: repeat only entries near a prize boundary. **R3. Publish the labeling prompt,** at least after the close. **R4. Measure the label's reliability directly,** with two independent draws of variants labeled by the official process. **R5. Show the floors and the noise on the leaderboard:** a row per constant predictor, a re-execution standard deviation and binomial intervals. **R6. Require a length control in white-box ablations:** without it, removing the white-box component cannot separate its signal from length. **R7. Remove the leaking field** `permutation_type` from the aggregated public datasets.

**Code and data.** Our code, bundles, pre-registrations, outcome files and full report (v2.4) will be released at https://github.com/Dan23RR/aimo-instrument; Appendix K gives release terms, bundle hashes and our Codabench submissions.

## Limitations

Ground truth under the earlier rule exists for two evaluated models, and every probe and every concordance with the official labels in Sections 3 to 5 is measured on DeepSeek, 26 of whose 164 labels we derived; these results are assumed, not shown, for the other models. The main-phase train set, at the revision we used (Appendix H), labels all five models but has 3 to 37 labeled rows per model, and apart from the evaluation of V8 and a descriptive comparison with our regenerated labels on the five labeled rows that fall on our problems and models, we do not use it. Section 4 uses 40 and 174 problems and one model; the fourteen-state structure is established, its cause inferred, and the remedies were tested only in sequential runs, each on one card and the two on different cards, so configuration and card are not separated. The regeneration of Section 5 used one generator, gated by itself, with our own answer key, sampling counts, decoding and prompts, so its negative concordance is a fact about our process, not about the families. Several analyses were first run from scripts that were not saved and were re-created and saved later, and a few figures rest only on a dated record (Appendix D). Nothing here estimates performance on the final held-out problems, beyond the predictions of Appendix B. The measurements of Sections 3 (probes), 4 and 5 were taken on the organizers' virtual machine; on 7 September we asked them whether such measurements may appear in this report, and we have not yet received an answer (Appendix K). The five-page main text condenses Appendix J, which states the caveats of each section more fully.

## AI usage statement

This work was carried out with an AI assistant (Claude, by Anthropic, through Claude Code). It drafted the pre-registrations, including hypotheses, predictions and kill criteria; wrote the solution code of our bundles and the analysis and extraction scripts, and ran them; operated the organizers' virtual machine over SSH; made most of our Codabench uploads through the author's browser at the author's instruction, with two exceptions: one resubmission (925822, 14 September), which it made without a specific instruction, and one submission that the author had not approved (1016734, 28 October): uploading the constant's Small bundle, it sent it to both tasks, probably because the form had re-selected them after its check; read the public records, the organizers' Discord channels and the author's correspondence with them; drafted the correspondence and this report, including its condensation to five pages; and ran automated multi-agent adversarial reviews of hypotheses, drafts and code, which found most of the corrections in Appendix C. The author directed the experiments and the submissions, with the exceptions stated, and is responsible for every claim.

## Acknowledgments

We thank Michal Štefánik, Róbert Belanec and the AIMO Interpretability team for the compute grant and for answering our questions. [TBD, to be confirmed by the author: the organizers provided a virtual machine with two GPUs; rented GPU time on Modal was funded personally or by platform credit (Appendix D); no other funding; no competing interests.]

## References

[1] M. Štefánik, P. Mondorf, A. Waldis, M. Spiegel, R. Belanec, C. Yang, M. Matveev, J. Mazak, Q. Liu, M. Schepers, J. Kuchař, M. Kadlčík, A. Vawda-Oomerjee, C. Liu, S. Frieder, B. Plank, F. Barez and P. Stenetorp. AIMO Interpretability Challenge. arXiv:2607.13899, 2026.

[2] H. He and Thinking Machines Lab. Defeating Nondeterminism in LLM Inference. Thinking Machines Lab: Connectionism, 2025.

[3] J. Yuan, H. Li, X. Ding, W. Xie, Y.-J. Li, W. Zhao, K. Wan, J. Shi, X. Hu and Z. Liu. Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. NeurIPS 2025. arXiv:2506.09501 (first posted as "Give Me FP32 or Give Me Death? Challenges and Solutions for Reproducible Reasoning").

[4] AIMO Interpretability Challenge website and FAQ. https://aimo-interp.github.io (accessed 28 October 2026), and its source repository aimo-interp/aimo-interp.github.io on GitHub.

[5] aimo-interp/getting-started and aimo-interp/baselines, GitHub repositories: README and issues #6 and #10 of getting-started; aimo-interp datasets on Hugging Face, including `train-main-v2` and `val-sample`.

[6] AIMO Interpretability Challenge Discord, #announcements and #qa: 5, 15 and 22 September, 9, 15, 16 and 20 October 2026.

[7] D. Hendrycks et al. Measuring Mathematical Problem Solving With the MATH Dataset. NeurIPS 2021 Datasets and Benchmarks Track. arXiv:2103.03874.

[8] W. Kwon et al. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023.

---

## Appendix A. The hypothesis table

[Generated from the full report, Section 4, by appendici_5p.py.]

## Appendix B. Pre-registered predictions to be scored against the final evaluation

P1 and P2 were written on 7 September 2026 (PREREG_E12), P-V8 on 13 September (PREREG_V8) and the P-V8b predictions on 21 September (PREREG_V8b). They are translated from the dated files and otherwise unchanged, except that P2 there names the participant then leading and quotes those entries' figures, which we do not. P2 asks whether a leaderboard figure survives new problems, not how it was obtained, and no inference about any participant's method will be drawn whatever the outcome.

- **P1.** The top-ranked entry's accuracy on the final held-out problems will lie within two binomial standard errors of that set's majority-class constant.
- **P2.** The final accuracy of the participant whose entries led both tracks on 7 September will be lower than those entries' leaderboard figures of that day by more than two standard errors.
- **P-V8.** If the organizers had neutralized length, the deterministic entry would score between 0.45 and 0.55 on both tracks (stated beforehand as the most likely outcome); if they had not, and the set was balanced, about 0.66; in either case with coverage 1.0, no invalid predictions and a runtime under 60 seconds. Main 0.52 and Small 0.53 fall inside the first band; on 21 and 19 items, however, 0.66 lies inside the 95% intervals of both scores, so the scores do not exclude the second. The clause on coverage, invalid predictions and runtime was not scored at the time: each evaluation on the task the bundle was built for had coverage 1.0 and no invalid predictions, and the bundle logs no runtime.
- **P-V8b**, on the replacement tasks: the old interface works with the new ingestion; the same bundles score 6 to 15 of 21 and 4 to 10 of 14; each finishes in under 60 seconds. Observed: coverage 1.0 and no invalid predictions; 10/21 and 7/14, inside the bands but degenerately, since the rule predicted robust for every item; each evaluation was marked finished within about a minute of upload, queue and scoring included, an upper bound on the runtime.

## Appendix C. Corrections to our earlier statements

[Generated from the full report, Section 11 and Appendix D, by appendici_5p.py.]

## Appendix D. Limitations, deviations, data provenance and compute

[Generated from the full report, Sections 12 and 13, by appendici_5p.py.]

[Appendices E to K are generated: E, section map between the full report and this report; F, floor algebra; G, the allocation episode; H, assets and licenses; I, prompts (appendici_5p.py); J, the nine-page text of Sections 2 to 7; K, Codabench submissions, bundles and release (costruisci_5p.py).]
