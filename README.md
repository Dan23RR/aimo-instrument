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
- `sonde/`, `soluzione/`, `soluzione_v8/`, `soluzione_costante/`: our submission bundles and their sources, including `sonde/solution-v4-allocazione/`, the V4 allocation entry (submission 909843, report Appendix G), which the author submitted once as a measurement against its pre-registered rule (`PREREG_V4.md`).
- `dati/`: our Codabench submissions, the competition metadata, and snapshots of the public leaderboard with other participants pseudonymized.
- `MANIFEST.tsv`: every file with its size, SHA-256, source path in the author's working folder and the transformation applied. `REDACTIONS.md`: the redactions.

## What is not here, and why

- Files of the organizers that state no license: the copies of the `aimo-interp` datasets, the `getting-started` code (`uncertainty_profile`, the DeepSeek regressor, the probe artifact and its inference code) and copies of files of the `baselines` repository. The bundles that contained `getting-started` files are given with our files only; `ORGANIZERS_FILES.txt` in each lists the omitted files with their SHA-256.
- Problem texts, and our perturbed variants of the public problems (E12, E13): the variants will be released only with the organizers' agreement. The ten calibration problems of v6 are given by `problem_id` in `aimo-interp/val-sample`, and the column holding the tail of the model's output is removed from `gpu/out/feature_greedy*.parquet`.
- Correspondence, internal status notes and plans (one of them contains the address of the organizers' machine), and one record about other participants' entries.
- Other participants' names, usernames, personal data and passages about their entries: redacted in place, as listed in `REDACTIONS.md`.
- Drafts of the full report before v2.4.

The measurements of Sections 3 to 5 of the report were taken on the organizers' virtual machine. On 7 September 2026 we asked the organizers whether such measurements may appear in the report; at the time of this release we have not received an answer.

## License

Code: MIT License (`LICENSE`). Text and data written or produced by us: CC BY 4.0. Third-party files keep their own terms (the NeurIPS 2026 style files in `report_9p/neurips_2026/`).

## Use of AI assistants

Most of the code and of the records were written by an AI assistant (Claude, by Anthropic, through Claude Code) under the author's direction; see the AI usage statement of the report.
