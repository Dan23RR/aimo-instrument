# The Instrument Behind the Leaderboard: Re-Execution Noise and Label Reliability in the AIMO Interpretability Challenge

**Daniel Culotta** (Codabench: `dan23rr`), independent researcher

*Five-page version, 28 October 2026, condensed from the nine-page draft of 22 October, whose Sections 2 to 7 are given as Appendix J, and from the full report (draft v2.4, 23 September 2026); these remain the reference for every figure below. Results of the per-problem validation protocol and of the final test set will be added when available; their places are marked [TBD].*

## Abstract

We treat the AIMO Interpretability benchmark, which asks whether a language model's answer to a math problem survives perturbations of the problem, as a measurement instrument, and report three negative results. First, on DeepSeek-R1-0528-Qwen3-8B, inside the official runtime image and pipeline and on the evaluation workers' GPU model, 24 of 40 items change prediction across five executions on the same inputs, and twenty-three executions fall into fourteen numerical states. Second, labels we regenerated under the rule the public labels follow, with our own prompts, decoding and answer key, agree with themselves at Cohen's kappa 0.18 to 0.44 across four models and with the official DeepSeek labels at 0.03; run on our labels, a simulation of the organizers' process overstates this test-retest agreement by about 0.4, so the reliability of the official label remains unmeasured. Third, of thirty-one hypotheses recorded in dated files and tested mostly on DeepSeek, twenty-one died by their kill criterion, and the two probes that survived are largely redundant with problem length. We close with seven recommendations and list fifty-four corrections to our own statements.

## 1 Introduction

A leaderboard is a measurement: its score is read off an instrument made of a label, an evaluation set, an inference pipeline and the hardware it runs on. When the quantity to be predicted is itself produced by a stochastic process, the instrument can fail in ways that no single submission can see.

We study the instrument of the AIMO Interpretability Challenge, in which a submission predicts, for a given model and math problem, whether the model's answer is robust to perturbations of the problem [1]. We approached the task as a falsification campaign: every hypothesis was recorded in a dated file, and every kill is reported. Of thirty-one entries, twenty-seven carried a kill criterion fixed before they ran⟦Three of them in a written plan rather than a frozen pre-registration; for H2 the criterion was written after its concordance had been observed (Appendix A).⟧: twenty-one died by it, five survived it and one is not established.

```latex
\begin{center}
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}L{3.2cm}L{3.2cm}L{3.2cm}L{3.2cm}@{}}
\toprule
{\LARGE 24 of 40} & {\LARGE 0.18 to 0.44} & {\LARGE 0.03} & {\LARGE 21 of 27} \\[2pt]
\footnotesize items whose prediction changes across five identical executions (DeepSeek, default configuration; Section 4) & \footnotesize test-retest kappa of our regenerated labels on four models (Section 5) & \footnotesize kappa between our regenerated DeepSeek labels and the official ones (Section 5) & \footnotesize entries with a kill criterion that died by it; the surviving probes are largely redundant with length (Section 3) \\
\bottomrule
\end{tabular}
\end{center}
```

Our findings concern the instrument: re-execution noise with structure (Section 4), a label that our regeneration does not reproduce and whose reliability is unmeasured (Section 5), and apparent signals that trace back, wholly or mostly, to something other than the model's internals, or are largely redundant with problem length (Section 3). Nondeterminism of LLM inference is documented in general [2, 3]; we add its structure and consequences on this benchmark's hardware and pipeline, and two candidate remedies. The organizers reached a compatible conclusion: the best leaderboard scores did not generalize to unseen problems [6]. We make no claim about any participant's method. Appendix J gives the long version of Sections 2 to 7, and Appendix C our fifty-four corrections.

## 2 The task, the labels and the public data

```latex
\begin{center}
\small
\begin{tabular}{@{}L{2.1cm}L{11.3cm}@{}}
\toprule
\textbf{Task} & A submission receives a model identifier, a reasoning effort and a list of problems, and returns one robustness verdict per problem, offline on one GPU within one hour [4, 5]; 70 minutes since 15 October, when evaluation moved to one call per problem [6]. \\
\midrule
\textbf{Models} & Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think and DeepSeek-R1-0528-Qwen3-8B in both tracks; gpt-oss-120b in the Main track only. \\
\midrule
\textbf{Evaluation} & Since 16 September, private validation tasks of 21 items (Main) and 14 (Small), on which the binomial standard error of an accuracy near one half is 10.9 and 13.4 points. The final test uses 30 to 50 new problems [5, 6]. \\
\midrule
\textbf{Label} & Its exact criterion is not published [4]. On all 379 labeled rows of the earlier public datasets it equals the conjunction ``no perturbation lowered accuracy'' (a majority rule fits them equally well). The main-phase train set thresholds the largest accuracy drop across perturbation families: robust at most 0.10, non-robust at least 0.25 [5]. \\
\midrule
\textbf{Public data} & Six datasets aggregate to 335 (model, problem) pairs over 174 problems; 52 pairs repeat the DeepSeek measurements under a second model name. DeepSeek is the only evaluated model with substantial ground truth under the earlier rule: 164 labeled pairs, 26 of them derived by us under the conjunction (a majority rule would call 21 of these 26 robust rather than 4). \\
\midrule
\textbf{Two pitfalls} & In the three aggregated datasets, the field \texttt{permutation\_type} is a JSON list if and only if the item is robust, an artifact of the aggregation that is absent at inference. And 20 of the 80 robust published pairs have base accuracy 0 and are robust only trivially under the conjunction, while the public validation sample, \texttt{val-sample}, has no labeled pair with base accuracy below 0.05. \\
\bottomrule
\end{tabular}
\end{center}
```

## 3 The falsification campaign

Table 1 summarizes the campaign; Appendix A gives all thirty-one entries with their baselines, numbers and caveats.

```latex
\begin{table}[t]
\caption{The campaign in brief. Ground truth is DeepSeek's and the public pairs' unless stated; accuracy differences are in points against each pre-registration's baseline. H1, H7 and T count the 52 duplicated DeepSeek pairs twice; counted once, no verdict changes. Table J.1 gives every result.}
\label{tab:campaign}
\centering
\small
\begin{tabular}{@{}L{5.5cm}L{1.9cm}L{5.6cm}@{}}
\toprule
\textbf{What we tested} & \textbf{Entries} & \textbf{Outcome} \\
\midrule
Text features; robustness as a property of the problem; labels, base accuracy and difficulty order across models & H1 to H5, H7 & dead; H2 gained only when the duplicated pairs were counted twice \\
Official uncertainty features, logistic regression & H10 & reaches the proposal's accuracy, within its 0.692 ± 0.101 (method substituted): 0.7744 against a majority constant of 0.756 \\
Features cached for other models; per-target thresholds & H12, H13 & H12 dead; the H13 ceiling withdrawn, −2.8 points when forced to rank \\
Our perturbation families, translation, self-consistency, the organizers' families regenerated (pilot), representation shift & H19, H22, H30 to H33 & dead: kappa at most 0.02, AUROC at most 0.615 \\
Probes on DeepSeek's hidden states: ours and the organizers' published artifact & H36, H35 & survive, largely redundant with length \\
Trivial-robust rule, post hoc & T & a trap: +6.0 points on the public pairs, +0.0 on \texttt{val-sample} \\
Identical bundles re-executed (warmup, 17 items) & H18 & noise confirmed: 11, 9, 11 and 11 correct \\
\bottomrule
\end{tabular}
\end{table}
```

**Three traps.** Rule T gains six points on the public pairs and nothing on `val-sample`, where the trivially robust subpopulation is absent. The oracle H13 chose each target's threshold on the very items it was scored on; forced to rank them, it gave −2.8. And H2, the only warmup hypothesis about a property of the problem that had survived, owed its survival to one measurement published under two model names and counted twice.

**Length and probes.** The character count alone orders the DeepSeek labels at AUROC 0.705. Our logistic probe on DeepSeek's last-token hidden states reaches 0.710 out of fold and survives its permutation null⟦With 60 permutations, where 500 were pre-registered; layer and regularization were chosen by inner cross-validation.⟧, but it correlates with length at Spearman +0.55: over 20 second-level splits, character count alone gives 0.696, the probe 0.697 and both together 0.714. The organizers' published DeepSeek probe artifact orders our labels at 0.709 under its own aggregation, but at 0.57 once residualized on log character count, and its controls trained on shuffled labels give 0.68. These length controls and artifact analyses were post hoc. Under its own aggregation the artifact thus orders the labels mostly through length; our probe is largely redundant with length, and what it reads is not established. Apart from rule T, no channel we scored in accuracy beats the majority constant by more than about one standard error on public ground truth, mostly DeepSeek's.

## 4 Re-execution noise on the evaluation workers' GPU model

On a virtual machine lent by the organizers, with two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (the evaluation workers' model), we ran forty public problems through the official runtime image, extraction and regressor, unmodified, on DeepSeek (Figure 1).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{fig_rumore.pdf}
\caption{Predictions of the official pipeline for forty public problems (columns) in twenty-three executions on DeepSeek (rows), by configuration; the last row is the label, dark cells are robust, and columns are sorted by how often they are predicted robust. At right: items whose prediction changes across the configuration's runs (more runs, more crossings), items whose generated text is byte-identical in all of them, and seconds per item. Table J.2 gives the mean number of crossings per pair of runs.}
\label{fig:noise}
\end{figure}
```

**Fourteen states.** The twenty-three executions fall into fourteen numerical states, byte-identical within a state. The state belongs to the process, not to the card: two runs launched in the same second on one card landed in different states, while two of three runs on the second card reproduced the first card's default state exactly. We infer, without proof, that the state is fixed when the process starts; two pre-registered predictions about the states failed (Appendix J). As only 7 of the 40 problems are robust and crossings largely cancel, accuracy moved much less than predictions (21 to 24 correct under the default configuration on either card, 24 to 28 with two processes on one card), always below the 33 of the majority constant.

**Candidate remedies.** In three sequential runs on one card, either deterministic algorithms (PyTorch's, in warn-only mode, with deterministic cuDNN and a fixed cuBLAS workspace; flags in Table J.2) or disabling compilation made generation byte-identical at 5 to 7% runtime cost. Neither was tested under contention or across cards, and the default configuration also gave three identical sequential runs before its fourth diverged: these are candidates, not fixes. Nor does determinism buy correctness: the two configurations, run on different cards, settled on different states, predicting 15 and 12 of 40 problems robust and agreeing on only 21.

**The official score channel.** In an exploratory measurement, not pre-registered, we extracted the official regressor score three times for the 174 public problems with the same code and image. One of the three runs set the cuBLAS workspace variable; the other two correlate at Spearman 0.24 over the 164 labeled problems and give AUROC 0.487 and 0.602, a difference that is not significant: the ordering does not reproduce, which is not a demonstrated change of signal. As 137 of the 164 problems are in the regressor's training set, this concerns the stability of its inputs, not new problems.

## 5 Can the label be reproduced?

We identified the organizers' perturbation families in the public data (rephrase, rename, domain, distract, typos), regenerated them with gpt-oss-120b prompted with the public examples, and gated each variant with gpt-oss-120b⟦The text must differ, the length ratio lie between 0.7 and 4.0 and the solver's answer not change; prompts in Appendix I.⟧. An answer counts as correct when it matches gpt-oss-120b's answer on the original problem, as the public data carry no answers. On the organizers' machine, with vLLM 0.28.0 outside the official image, we labeled four models with ten variants per family in two independent draws (A and B; about 40 accepted variants per problem), ten samples on the base problem and one per variant, and re-sampled draw A's variants with a new seed (A′).

```latex
\begin{table}[t]
\caption{Agreement of regenerated labels under the conjunction: Cohen's kappa, with 95\% percentile bootstrap intervals over problems. Robust rates are over all 164 problems (the 4 without accepted variants count as not robust); kappas over the problems labeled in both series (159 for A and B, 160 for A′ and the official labels). Among these models, official labels under the earlier rule exist for DeepSeek only; Table J.3 adds the main-phase rule.}
\label{tab:label}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{@{}lllll@{}}
\toprule
\textbf{Model} & \textbf{Robust in A} & \textbf{A and B} & \textbf{A and A′ (new seed)} & \textbf{Official labels} \\
\midrule
Skywork & 72.6\% & 0.439 [0.28, 0.59] & 0.789 [0.67, 0.89] & none \\
DeepSeek & 37.8\% & 0.413 [0.26, 0.55] & 0.464 [0.33, 0.59]$^{*}$ & 0.025 [−0.12, 0.18] \\
Olmo & 65.9\% & 0.378 [0.21, 0.54] & 0.653 [0.52, 0.78] & none \\
Qwen3.5-4B & 9.1\% & 0.178 [−0.06, 0.39] & not run & none \\
\bottomrule
\end{tabular}
\par\smallskip{\footnotesize $^{*}$On DeepSeek, A′ also received the answer-format line of Section 5.3, so this entry mixes seed and prompt.}
\end{table}
```

**5.1 More variants do not help, and our noise does not explain the gap.** Subsampling DeepSeek's variants from 2 to 40 per problem, the mean kappa(A, B) stays between 0.36 and 0.47 with no upward trend, and the mean concordance with the official label between −0.02 and +0.03. Corrected for our own noise, 0.025 becomes 0.039 (about 0.27 at the top of its 95% interval): our noise could hide at most a weak agreement. Under the main-phase rule, pre-registered on the same answers, test-retest stays below 0.60 on every model.

**5.2 What drives the instability.** On Skywork, re-sampling the same variants with a new seed agrees far more than drawing new variants (0.79 against 0.44), so there the label depends mainly on which variants are drawn. A plug-in simulation of the organizers' process from their published prediction counts gives a test-retest kappa of 0.94, but it resamples answers to the same variants and never lets a robust item fail; on our draw A it gives 0.64 to 0.84 against a measured 0.18 to 0.44, overstating by about 0.4 on every model. The reliability of the official label therefore remains unmeasured. The official DeepSeek labels also come from a different sampling regime [5] and decoding [4], so the data show something narrower than a verdict on the label: variants produced by our process measure something else. The organizers have decided not to publish their labeling code [6].

**5.3 An unspecified prompt moves the label.** Fed the bare problem text, Qwen3.5-4B closes its answer in the expected format in only 5 of its 1,640 base samples in draw A; with one instruction line requesting the format, the share of problems passing the label's base condition⟦At least 6 of 10 samples agreeing with the reference answer.⟧ moved from 0 to 145 of 164 (88%). A label defined by correctness is not comparable across prompts that the benchmark leaves unspecified.

## 6 Our entries, and the evaluations the organizers requested

**V8, the entry that does nothing.** Pre-registered on 13 September, its entire logic is `return [len(p) >= 98 for p in problems]`, with the threshold at the lower median of the 174 public problems and the direction fixed on the public pairs. On the original main-phase tasks it scored 11/21 and 10/19, identical to our measuring entry v6 (the official extraction and regressor on each model's own greedy generation) and to the organizers' own entries on the leaderboard; none of these entries scored above the majority constant. On 21 and 19 items these ties are compatible with length carrying little signal (95% Wilson intervals 0.32 to 0.72 and 0.32 to 0.73) without showing that it carries none. A one-line rule that ties a pipeline loading up to five models for 8.5 to 14 minutes is a measurement of the pipeline.

**The evaluations requested for reports [5].** On the public main-phase train set (revision of 17 September), V8 scores 46/71, below the 52/71 of the all-robust constant. On the replacement tasks of 16 September it predicted robust for every item: 10/21 and 7/14, the base rates. On 28 October, after the change of 15 October [6], we resubmitted V8 unchanged (Codabench 1016713, Main; 1016730, Small) with an all-robust constant as control (1016732, 1016735); both predicted robust for every item and scored 10/21 and 7/14. For the final test we nominated V8 (1016713) and the constant (1016732), both on the Main track; their scores were not yet available when we submitted this report. Our pre-registered predictions will be scored there (Appendix B): P1, that the top-ranked entry's final accuracy will lie within two binomial standard errors of the test set's majority constant, and P2; if either fails, the label has learnable structure that we did not find.

## 7 Recommendations

- **R1. Fix the numerical state, and test the fix under load.** Verify either remedy of Section 4 under the workers' real load and across cards, and document the chosen setting, which selects the fixed point; isolating submissions on a card may matter as much.
- **R2. Repeat only where it decides something.** At the warmup's re-execution standard deviation (1.0 item on 17, from four runs, itself uncertain), a one-item gap needs about eight runs: repeat only entries near a prize boundary.
- **R3. Publish the labeling prompt,** at least after the close.
- **R4. Measure the label's reliability directly,** with two independent draws of variants labeled by the official process.
- **R5. Show the floors and the noise on the leaderboard:** a row per constant predictor, a re-execution standard deviation and binomial intervals.
- **R6. Require a length control in white-box ablations:** without it, removing the white-box component cannot separate its signal from length.
- **R7. Remove the leaking field** `permutation_type` from the aggregated public datasets.

**Code and data.** Our code, bundles, pre-registrations, outcome files and full report (v2.4) will be released at https://github.com/Dan23RR/aimo-instrument; Appendix K gives release terms, bundle hashes and our Codabench submissions.

## Limitations

Ground truth under the earlier rule exists for two evaluated models, and every probe and every concordance with the official labels in Sections 3 to 5 is measured on DeepSeek, 26 of whose 164 labels we derived; these results are assumed, not shown, for the other models. The main-phase train set, at the revision we used (Appendix H), labels all five models but has 3 to 37 labeled rows per model, and apart from the evaluation of V8 and a descriptive comparison with our regenerated labels on the five labeled rows that fall on our problems and models, we do not use it. Section 4 uses 40 and 174 problems and one model; the fourteen-state structure is established, its cause inferred, and the remedies were tested only in sequential runs, each on one card and the two on different cards, so configuration and card are not separated. The regeneration of Section 5 used one generator, gated by itself, with our own answer key, sampling counts, decoding and prompts, so its negative concordance is a fact about our process, not about the families. Several analyses were first run from scripts that were not saved and were re-created and saved later, and a few figures rest only on a dated record (Appendix D). Nothing here estimates performance on the final held-out problems, beyond the predictions of Appendix B. The measurements of Sections 3 (probes), 4 and 5 were taken on the organizers' virtual machine; on 7 September, and again on 2 November, we asked them whether such measurements may appear in this report, and we have not yet received an answer (Appendix K). The five-page main text condenses Appendix J, which states the caveats of each section more fully.

## AI usage statement

In this work, we used one generative AI tool, Claude (Anthropic), through Claude Code, for ideation, coding, analysis and writing. Ideation: it drafted the pre-registrations, including hypotheses, predictions and kill criteria, and searched and summarized the literature and the public records, including the organizers' Discord channels and the author's correspondence with them. Coding: it wrote the solution code of our bundles and the analysis and extraction scripts, ran them, and operated the organizers' virtual machine over SSH. Analysis: it computed and interpreted the results and ran automated multi-agent adversarial reviews of hypotheses, drafts and code, which found most of the corrections in Appendix C. Writing: it drafted the correspondence and this report, including its condensation to five pages. It also made most of our Codabench uploads through the author's browser at the author's instruction, with two exceptions: one resubmission (925822, 14 September), which it made without a specific instruction, and one submission that the author had not approved (1016734, 28 October): uploading the constant's Small bundle, it sent it to both tasks, probably because the form had re-selected them after its check. Language models were also components of our method: gpt-oss-120b generated and gated the perturbation variants and supplied the reference answers of Section 5, and phi-4 translated problems for H22 (Appendices H and I). We did not use generative AI tools to make the final decisions on the direction of the work, on the submissions (with the two exceptions above), on the nominations for the final test and on the claims of this report, which were the author's, and the author sent all correspondence personally. The author has reviewed all AI-assisted work and takes full responsibility for the final content of this report, including AI-assisted text, claims, code and other artifacts.

## Acknowledgments

We thank Michal Štefánik, Róbert Belanec and the AIMO Interpretability team for the compute grant, a virtual machine with two GPUs, and for answering our questions. Other GPU time was rented on Modal and paid by the author or covered by Modal's platform credit (Appendix D). This work received no other funding, and the author has no competing interests.

## References

[1] M. Štefánik, P. Mondorf, A. Waldis, Q. Liu, C. Yang, M. Spiegel, J. Kuchař, M. Kadlčík, A. Vawda-Oomerjee, C. Liu, S. Frieder, B. Plank, F. Barez and P. Stenetorp. AIMO Interpretability Challenge. CoRR, abs/2607.13899, 2026. doi:10.48550/arXiv.2607.13899.

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
