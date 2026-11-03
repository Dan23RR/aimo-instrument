"""Compila la checklist NeurIPS 2026 per il report da 5 pagine (28/10/2026), copiato da report_9p/checklist_compila.py.

Uso (dalla cartella report_5p): python checklist_compila_5p.py
Legge ../report_9p/neurips_2026/checklist.tex (ufficiale), toglie il blocco di istruzioni come chiede il
modello, sostituisce in ordine le 16 coppie \\answerTODO{} / \\justificationTODO{} e scrive
latex/checklist_compilata.tex. Domande e linee guida restano invariate. Le note interne (terzo
elemento delle tuple) diventano commenti LaTeX e non compaiono nel PDF.
"""
import io
import os
import re

QUI = os.path.dirname(os.path.abspath(__file__))
t = io.open(os.path.join(QUI, "..", "report_9p", "neurips_2026", "checklist.tex"), encoding="utf-8").read()

a = t.index("%%% BEGIN INSTRUCTIONS %%%")
b = t.index("%%% END INSTRUCTIONS %%%") + len("%%% END INSTRUCTIONS %%%")
t = t[:a] + t[b:]

RISPOSTE = [
    ("Yes", "The abstract states the three negative results and their scope (the re-execution measurement on DeepSeek, a regeneration with our own prompts, decoding and answer key, hypotheses tested mostly on DeepSeek), and Section 1 states the same contributions; Sections 3 to 6 support them and the Limitations section states their limits."),
    ("Yes", "The Limitations section, after Section 7, is devoted to limitations, Sections 4 and 5 qualify each result where it is stated (exploratory measurements, remedies tested only in sequential runs, ground truth on one model), and Appendix J states the caveats of each section in full; Appendix D lists the figures that rest only on a dated record."),
    ("NA", "The paper has no theoretical results; Appendix F, carried over from the full report and not used in the main text, restates an elementary rescaling of accuracy against a floor and the number of items it implies, without theorems or proofs."),
    ("Yes", "Sections 2 to 6 and Appendix J describe the data, procedures, hardware, software versions, configurations and decision rules, Appendix I gives the prompts, Appendix D lists the deviations from the pre-registrations and gives the rule, parameters and bundle hashes of our entry, and Appendix K gives the bundle hashes again with our Codabench submissions; our entry loads no model and is deterministic. The measurements of Section 4 ran on the organizers' virtual machine inside the official runtime image, while the regeneration and labeling of Section 5 and the hidden-state extraction for the probes of Section 3 ran on the same machine in our own Python environment, whose versions Section 5 and Appendix J give, so reproducing them needs comparable GPUs."),
    ("Yes", "The analysis scripts, dated pre-registrations, extraction outputs, submission bundles and our perturbation variants are public at https://github.com/Dan23RR/aimo-instrument (Section 7, Code and data; Appendix K), where the file paths cited in this report are valid. Its README gives the steps that recompute our entry's parameters and reproduce its score, maps the experiments to the folders that hold their scripts and outputs, and points to the record of the virtual machine's environment; rerunning the GPU experiments needs comparable GPUs and the models of Appendix H, and the organizers' datasets are not redistributed (\\texttt{dati.py} fetches the six earlier ones)."),
    ("Yes", "Data splits and grouping (Sections 2 and 3, Appendix A), decision thresholds and how they were chosen (Sections 3 and 6), the selection of the probe's layer and regularization by inner cross-validation (Section 3), the decoding, sampling counts and prompts of the regeneration (Section 5, Appendices I and J) and the runtime configurations (Section 4, Table 2) are given."),
    ("Yes", "We report 95\\% percentile bootstrap intervals over problems for every agreement statistic in Table 2 and Table J.3, Wilson intervals for V8's scores on the original tasks (Section 6), a paired bootstrap interval for the difference between the two score-channel AUROCs (Section 4), a permutation null for our probe (Section 3) and the binomial standard errors of the validation tasks (Section 2); threshold crossings across runs (Figure 1) are reported as counts, and the other results of Tables 1 and J.1 (accuracy differences, kappas and AUROCs) without intervals, apart from the z values of H2 and the comparison of H10 with one standard error."),
    ("Yes", "Section 4 and Appendix J.4 give the hardware (two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 96 GB each, on the organizers' virtual machine), the runtime image and the seconds per item of each configuration, and Appendix J.6 the runtimes of v6; Appendix D gives the GPU time documented on Modal (about 12 to 13 GPU-hours) and an estimate of about 100 card-hours for E13, and Appendix C (item 54) the invalidated Qwen runs (about fifteen card-hours), while the totals for E14 to E16 and the billed totals are still to be itemized from the run logs and billing records."),
    ("Yes", "The work analyses public benchmark data, our own submissions, snapshots of the public Codabench leaderboard and the aggregate facts about the private evaluation sets that our submissions logged, all disclosed together with the entry built on them (Appendices D and G); it involves no human-subjects research and makes no claim about other participants' methods, and the released files, the leaderboard snapshots included, have other participants' names, usernames and details removed (Appendix K). On 7 September, and again on 2 November, we asked the organizers whether measurements taken on their virtual machine may appear in this report, and on 2 November they answered that this is acceptable."),
    ("No", "The paper does not discuss societal impacts: it measures the reliability of a competition benchmark, and its recommendations concern evaluation practice (Section 7); the ways it documents to obtain misleading scores, a field that leaks the label in development data and re-execution noise, come with the fixes we recommend (R1, R2, R5 and R7)."),
    ("NA", "We release no models and no data with a high risk of misuse: the release (Appendix D, Release, and Appendix K) consists of our scripts and their outputs, our labels and per-variant answers, our submission bundles as submitted, with the organizers' files they contain, our dated files and snapshots of the Codabench leaderboard, with other participants' names and usernames removed, and, with the organizers' permission, perturbed versions of public math problems."),
    ("Yes", "Appendix H credits the models, including those of the warmup, the datasets with the sources of their problems and the revision we used, and the code we used, with the licenses stated on their pages; where no license is stated (all six earlier datasets, Skywork-OR1-Math-7B, the getting-started repository) we say so and say how we use and redistribute them."),
    ("Yes", "The regenerated perturbation variants, our labels, the extraction outputs and the analysis scripts are released at https://github.com/Dan23RR/aimo-instrument (Appendix K), alongside the dated pre-registrations and outcome files, with an English README that describes the folders and states the licenses (MIT for code, CC BY 4.0 for our text and data) and a manifest giving each file's size, SHA-256 and origin; the variants are released with the organizers' permission (Appendix H)."),
    ("NA", "The paper involves no crowdsourcing and no research with human subjects."),
    ("NA", "The paper involves no human subjects."),
    ("Yes", "Language models are the objects of study (Section 2); in our label regeneration gpt-oss-120b generated and gated the perturbation variants and supplied the reference answers (Section 5, Appendix I), and phi-4 translated problems for H22 (Appendix H). An AI assistant (Claude, through Claude Code) drafted the pre-registrations, including hypotheses, predictions and kill criteria, wrote and ran the code, operated the virtual machine, made most of our uploads and ran the adversarial reviews behind most corrections, as disclosed in the AI usage statement."),
]

risp = list(re.finditer(r"\\answerTODO\{\}", t))
giust = list(re.finditer(r"\\justificationTODO\{\}", t))
assert len(risp) == 16 and len(giust) == 16, (len(risp), len(giust))
assert len(RISPOSTE) == 16
out, pos = [], 0
for (r, g), voce in zip(zip(risp, giust), RISPOSTE):
    ans, just = voce[0], voce[1]
    out.append(t[pos:r.start()])
    out.append("\\answer" + ans + "{}")
    out.append(t[r.end():g.start()])
    out.append(just + ("  % " + voce[2] if len(voce) > 2 else ""))
    pos = g.end()
out.append(t[pos:])
t = "".join(out)
t = re.sub(r" % Replace by \\answerYes\{\}, \\answerNo\{\}, or \\answerNA\{\}\.", "", t)
io.open(os.path.join(QUI, "latex", "checklist_compilata.tex"), "w", encoding="utf-8", newline="\n").write(t)
print("checklist compilata:", len(re.findall(r"\\answer(?:Yes|No|NA)\{\}", t)) - len(re.findall(r"answer \\answer", t)), "macro di risposta;",
      "TODO rimasti:", len(re.findall(r"TODO\{\}", t)))
