# aimo-instrument

This repository holds the code, data and dated records behind the report *The Instrument Behind the Leaderboard:
Re-Execution Noise and Label Regeneration in the AIMO Interpretability Challenge*. The author is Daniel Culotta,
independent researcher. The report was written for the AIMO Interpretability Challenge at NeurIPS 2026
(https://aimo-interp.github.io), and this is its revision of November 2026.

## What is here

| Folder or file | What it holds |
|---|---|
| `paper/` | The report, `main.pdf`, and its LaTeX sources. |
| `reproduce/` | English scripts that recompute the main numbers of the report from saved outputs, on a laptop. |
| `archive/` | The author's working folder: every script, log and output, and our submission bundles, with the dated pre-registrations and outcome files gathered in `archive/records/`. The file paths cited in the report are relative to it, except `reproduce/` and the files at the top. |
| `CORRECTIONS.md` | The 54 corrections made to our own statements before submission. |
| `MANIFEST.tsv` | Every other file, with its size and SHA-256. |
| `REDACTIONS.md` | What was redacted before release, with the SHA-256 of each original. |

## Revision of November 2026

The report was conditionally accepted on 21 November 2026. Compared with the submitted version, the revision:

- separates the preliminary public datasets from the official training and validation data, says that our
  regenerated labels come from our own protocol, and says that most results are on DeepSeek-R1-0528-Qwen3-8B;
- reports the share of chains that reach our 16,384-token cap, agreement under three treatments of truncated
  chains, and agreement with the 138 published DeepSeek labels separately from the 160 that include our derived ones;
- replaces a combination of probe and length that leaked labels across folds with a recalculation on common outer
  folds (E15-R), and extends the probe's permutation null to 1,000 permutations;
- presents the two candidate remedies for re-execution noise with their limits, and flags the organizers' probe
  artifact (H35) as probably in-sample;
- has a shorter appendix with a run manifest, and moves the 54 corrections of September to `CORRECTIONS.md`.

## Recompute the main numbers

From the repository root, with Python 3.11, NumPy, pandas, PyArrow and SciPy:

```
python archive/dati.py                  # once: the six preliminary datasets (about 2 MB)
python archive/aggrega.py               # once: the aggregated table used by V8
python reproduce/labels_table.py        # Table 2
python reproduce/reexecution_table.py   # Table 1 and Figure 1
python reproduce/probe_vs_length.py     # Section 5
python reproduce/v8_threshold.py        # Section 6
python reproduce/compute_hours.py       # Appendix C
python reproduce/fewshot_pool.py        # Appendix C
```

`reproduce/README.md` says what each script recomputes, which inputs it reads, and which runs need GPUs.

## Reproduce our leaderboard score

Our entry is V8, with an all-robust constant as control. Neither bundle loads a model or reads data at run time.

| Bundle in `archive/sonde/` | What it does | SHA-256 |
|---|---|---|
| `solution-v8-main.zip` | returns `[len(p) >= 98 for p in problems]` | `7b70f15c51752dcf4285e477d58c541d0ecf052d833c325c0702dbbd054ffb5f` |
| `solution-v8-small.zip` | the same, with the empty marker file `small.txt` | `0b83b55f94dc16b9852a7c2b35d71ac5787e43d81afd6396dee00fb5e011e500` |
| `solution-costante-main.zip` | returns True for every problem | `35c47bbef717a5d4c5b6c7786bddbab9d87a02483e4f417cd34fd4ecf669c67d` |
| `solution-costante-small.zip` | the same, with `small.txt` | `f798c2979b136a84c22e6d994bf0228bbfeffa044ac68f7868e13859b4c6075a` |

Upload each bundle to Codabench (competition 16180) with only its own task selected: the Main bundles to task 36237,
the Small bundles to task 36241. On 28 October 2026 V8 and the constant both scored 10/21 on the Main task and 7/14 on
the Small task. V8's two parameters were fixed on the six preliminary datasets, and `reproduce/v8_threshold.py`
recomputes them.

## Reading the archive

- Most records and code comments are in Italian, the author's working language; the glossary at the end translates
  the words in the file names.
- Dated files are append-only: a correction is added at the end of its file, never made in place. The only in-place
  changes are the redactions listed in `REDACTIONS.md`.
- `archive/records/` holds the dated records: the pre-registrations (`PREREG_*.md`), with their predictions and kill
  criteria; their outcomes and deviations (`*_ESITO.md`, `*_DEVIAZIONE.md`, `RISULTATI_*.md`); and the other dated
  notes. They cite each other and the scripts by name.
- The scripts in `archive/vm/` written up to 14 September ran on the virtual machine that the organizers lent us,
  and their paths point there; the later ones ran on the author's computer from the working folder. Many also need
  files of the organizers that are not released here; a script stops with the missing path.

## Not included, and why

- **The organizers' datasets and files.** `archive/dati.py` downloads the six preliminary datasets; `train-main-v2` is
  on Hugging Face; the `getting-started` and `baselines` repositories are on GitHub under `aimo-interp`.
- **The organizers' problem texts.** Our variant files and tables keep the problem identifiers but not the texts, and
  phi-4's translations of the problems (H22) are left out. The exception is five problems inside the v6 bundles,
  released exactly as submitted.
- **Correspondence, internal notes and other participants' data.** Other participants' names and passages about their
  entries are redacted in place, as `REDACTIONS.md` lists.
- **Earlier drafts of the report.** The submitted version is on OpenReview; the drafts are in commit 47be35d of this
  repository.

## Permissions and licenses

The measurements on the organizers' virtual machine appear with their permission, and the `getting-started` files
inside our bundles and our perturbed variants are redistributed with their permission of 2 November 2026. Our code is
under the MIT License (`LICENSE`), and the text and data we wrote or produced under CC BY 4.0. Third-party files keep
their own terms: the NeurIPS 2026 style file in `paper/`, and the `getting-started` files inside the bundles, for which
the organizers state no license.

## Use of AI assistants

Most of the code and of the records were written by an AI assistant (Claude, by Anthropic, through Claude Code) under
the author's direction; the AI usage statement of the report gives the details.

## Glossary: Italian words in file names

| Word | Meaning | Word | Meaning |
|---|---|---|---|
| `affidabilita` | reliability | `percorso ufficiale` | official path |
| `aggrega`, `aggregata` | aggregate, aggregated | `piano` | plan |
| `allocazione` | allocation | `pilota` | pilot |
| `ambiente` | environment | `pubblici` | public |
| `analisi` | analysis | `pulizia` | clean-up |
| `artefatto` | artifact | `regola nuova` | new rule |
| `calibrata` | calibrated | `ricalcola`, `ricalcolo`, `ricalcoli` | recompute, recalculation, recalculations |
| `campioni` | samples | `riprendi` | resume |
| `classifica` | leaderboard | `ripresa` | resumption |
| `coda` | queue | `riproducibilita` | reproducibility |
| `collaudo locale` | local test | `risultati` | results |
| `competizione` | competition | `rumore` | noise |
| `controllo`, `controlli` | check, checks | `scartati` | discarded |
| `costante` | constant | `scelte` | choices |
| `cronometro` | timer | `segnale` | signal |
| `dati` | data | `simulazione` | simulation |
| `definitivo` | final | `sintesi` | summary |
| `deviazione` | deviation | `soluzione` | solution |
| `esito` | outcome | `sonda` | probe |
| `esperimento` | experiment | `sonde` | probes (our submission bundles) |
| `estrai`, `estratto` | extract, extracted | `sottocampionamento` | subsampling |
| `etichetta`, `etich` | label | `stadio` | stage |
| `fattibilita` | feasibility | `strumentata` | instrumented |
| `invii` | submissions | `tabella` | table |
| `lunghezza` | length | `tetto` | ceiling |
| `modello` | model | `traduzioni` | translations |
| `nidificato` | nested | `valuta` | evaluate |
| `nota tecnica` | technical note | `varianti`, `varianza` | variants, variance |
| `nudo`, `esplicito` | bare, explicit (prompt) | `verdetto` | verdict |
| `organizzatori` | organizers | `verifica` | verification |
| `osservato` | observed | `verifica tutto` | verify everything |
| `parziale` | partial | `versioni` | versions |
| `perche scartati` | why discarded |  |  |
