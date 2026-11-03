# aimo-instrument

Code, data and dated records behind the technical report *The Instrument Behind the Leaderboard: Re-Execution Noise and Label Reliability in the AIMO Interpretability Challenge* by Daniel Culotta (independent researcher; Codabench user `dan23rr`), written for the AIMO Interpretability Challenge (https://aimo-interp.github.io). The report is `report_5p/latex/main.pdf`.

The repository keeps the layout of the author's working folder, so the file paths cited in the report (for example `vm/controlli_verifica3.py` or `vm_risultati/e13_analisi_2026-09-14.txt`) are valid here. Most records and code comments are in Italian, the author's working language; the report is in English.

## Our entry, and how to reproduce its score

Our entry on the current validation tasks is V8, with an all-robust constant as control.

| Bundle | What it does | SHA-256 |
|---|---|---|
| `sonde/solution-v8-main.zip` | `are_robust(model_id, problems)` returns `[len(p) >= 98 for p in problems]` | `7b70f15c51752dcf4285e477d58c541d0ecf052d833c325c0702dbbd054ffb5f` |
| `sonde/solution-v8-small.zip` | the same, with the empty marker file `small.txt` | `0b83b55f94dc16b9852a7c2b35d71ac5787e43d81afd6396dee00fb5e011e500` |
| `sonde/solution-costante-main.zip` | returns True for every problem | `35c47bbef717a5d4c5b6c7786bddbab9d87a02483e4f417cd34fd4ecf669c67d` |
| `sonde/solution-costante-small.zip` | the same, with `small.txt` | `f798c2979b136a84c22e6d994bf0228bbfeffa044ac68f7868e13859b4c6075a` |

Upload each bundle to Codabench (competition 16180) with only its own task selected: the Main bundles on the Main task, the Small bundles on the Small task. No bundle loads a model or reads data at run time. On 28 October 2026 V8 and the constant both scored 10/21 on the Main task and 7/14 on the Small task (submissions 1016713, 1016730, 1016732 and 1016735); `dati/invii_dan23rr_2026-10-28_sera.json` lists all our submissions. The sources are in `soluzione_v8/` and `soluzione_costante/`.

V8 has no trained weights. Its two parameters were fixed on the six public `aimo-interp` datasets: the threshold, 98, is the lower median of the character counts of their 174 distinct problems, and the direction (longer problems robust) is the one with the higher balanced accuracy on their labeled pairs. To recompute them:

1. `python dati.py` downloads the six datasets into `dati/` through the public Hugging Face datasets-server API;
2. `python aggrega.py` rebuilds `dati/tabella_aggregata.json` with the problem texts (the released copy has them removed);
3. `python vm/controlli_verifica3.py` recomputes the threshold and `python vm/dati_pubblici_controlli.py` the balanced accuracies; the latter reads the dataset copies from `vm_backup/estratto/work/aimo/dati/`, so copy the downloaded files there first.

## Map

- `report_5p/`: the five-page report (Markdown source, build scripts, LaTeX sources and PDF). `report_9p/`: the nine-page draft that forms Appendix J of the report, with the records of its adversarial verifications. `versioni_writeup/WRITEUP_v2.4_2026-09-23.md`: the full report, frozen on 23 September 2026.
- `PREREG_*.md` and `PIANO_E11_definitivo.md`: dated pre-registrations and plans, with predictions and kill criteria, and their outcome files (`*_ESITO.md`, `*_DEVIAZIONE.md`). Corrections are appended at the end of each file, never edited in place.
- `RISULTATI_*.md`, `OSSERVATO*.md`, `NOTA_TECNICA_nondeterminismo_2026-09-04.md`, `PILOTA_COLAB_ESITO.md`: dated records of results.
- Top-level `*.py`: analyses of the public data (E1 to E8, floors, labels).
- `gpu/`: runs on rented GPUs, with outputs in `gpu/out/`.
- `vm/`: runs on the virtual machine that the organizers lent us (E13 to E16), with outputs in `vm_risultati/` and the environment record in `vm_backup/AMBIENTE_VM_2026-09-17.txt`. The hidden states used by the probes of Section 3 of the report are in `vm_backup/estratto/work/aimo/e15/`.
- `sonde/`, `soluzione/`, `soluzione_v8/`, `soluzione_costante/`: our submission bundles, exactly as submitted, and their sources, including `sonde/solution-v4-allocazione.zip`, the V4 allocation entry (submission 909843, report Appendix G), which the author submitted once as a measurement against its pre-registered rule (`PREREG_V4.md`).
- `vm_risultati/e13/varianti_D1*.parquet`, `vm_risultati/e13/varianti_D2*.parquet` (E13, one set of files per draw) and `gpu/out/e12_s2_varianti.parquet` (E12): our perturbed variants of the public problems, with gpt-oss-120b's answers, the reference answers and the gate's decisions; the files `*_gen_parziale` and `*_sol_parziale` hold the generation and solving steps. Rows of the family `original` keep the problem identifier but not its text.
- `dati/`: our Codabench submissions, the competition metadata, and snapshots of the public leaderboard with other participants pseudonymized.
- `MANIFEST.tsv`: every file with its size, SHA-256, source path in the author's working folder and the transformation applied. `REDACTIONS.md`: the redactions.

## What is not here, and why

- Copies of the organizers' datasets: the `aimo-interp` datasets on Hugging Face, whose cards state no license (`python dati.py` fetches the six earlier ones), and `train-main-v2`. Copies of the three files of the `baselines` repository that we used (Apache-2.0, available there). Files of the `getting-started` repository other than those inside our bundles, such as the probe artifact and its inference code: fetch them from https://github.com/aimo-interp/getting-started.
- The organizers' problem texts. In our variant files the rows of the family `original` keep the problem identifier but not its text, `dati/tabella_aggregata.json` has the field `problema` removed, and `soluzione/calibrazione_gptoss.json` gives the ten calibration items of v6 by `problem_id` in `aimo-interp/val-sample`. The exception is the three v6 bundles in `sonde/`, released exactly as submitted, whose `calibrazione_gptoss.json` holds the texts of those items (five problems). Also left out: the few-shot examples of E12 (`gpu/e12_fewshot.json`), which are the organizers' own perturbed texts, the list of the 164 base problems of E12 with their texts (`gpu/out/_base164.parquet`), and the column holding the tail of the model's output in `gpu/out/feature_greedy*.parquet`.
- Correspondence, internal status notes and plans (one of them contains the address of the organizers' machine), drafts of the full report before v2.4, and one record about other participants' entries.
- Other participants' names, usernames, personal data and passages about their entries: redacted in place, as listed in `REDACTIONS.md`.

The measurements of Sections 3 to 5 of the report were taken on the organizers' virtual machine. On 7 September 2026 we asked the organizers whether such measurements may appear in the report, and we asked again on 2 November, together with whether we may release the `getting-started` files inside our bundles and our perturbed variants; the same day the organizers answered yes to both.

## License

Code: MIT License (`LICENSE`). Text and data written or produced by us: CC BY 4.0. Third-party files keep their own terms (the NeurIPS 2026 style files in `report_9p/neurips_2026/`). The `getting-started` files inside the bundles in `sonde/` (`uncertainty_profile/`, `uncertainty_artifacts/`) are the organizers', who state no license for them; we redistribute them with their permission of 2 November 2026, and our licenses do not cover them. Our perturbed variants derive from the organizers' public problems and are released with the same permission; CC BY 4.0 covers only our contribution.

## Use of AI assistants

Most of the code and of the records were written by an AI assistant (Claude, by Anthropic, through Claude Code) under the author's direction; see the AI usage statement of the report.
