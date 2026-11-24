# reproduce: the main numbers of the report, from saved outputs

Every script runs from the repository root on a laptop, without a GPU, in seconds. It reads released files in
`archive/` and the six preliminary public datasets, which `archive/dati.py` downloads (about 2 MB). Each script checks
its inputs and says what to run if one is missing. Requirements: Python 3.11 with NumPy, pandas, PyArrow and SciPy.

```
python archive/dati.py                  # once: the six preliminary datasets, into archive/dati/
python archive/aggrega.py               # once: the aggregated table used by V8
python reproduce/labels_table.py        # Section 4: Table 2
python reproduce/reexecution_table.py   # Section 3: Table 1, the numbers of Figure 1 and the 14 distinct outputs
python reproduce/probe_vs_length.py     # Section 5: the probe against problem length, and the E15-R recalculation
python reproduce/v8_threshold.py        # Section 6: the threshold and direction of V8
python reproduce/compute_hours.py       # Appendix C: the GPU-hours of the main runs
python reproduce/fewshot_pool.py        # Appendix C: the few-shot pool of the variant generator
```

## What each script recomputes

The inputs are in `archive/`.

| Script | In the report | Inputs |
|---|---|---|
| `labels_table.py` | Table 2 and the agreement numbers of Section 4: kappa between draws A and B under three treatments of truncated chains, kappa between A and A', the share of chains at the 16,384-token cap, kappa with the 138 published DeepSeek labels and with the 160 that include our derived ones | `vm_risultati/e13/etich_*.parquet`, `gpu/out/feature_greedy.parquet`, `dati/*.json` |
| `reexecution_table.py` | Table 1 and the numbers of Figure 1: crossings, identical outputs, correct predictions and cost of each configuration | `vm_risultati/e14/out/*.jsonl`, `vm_risultati/e14/log/*.log`, `gpu/out/feature_greedy.parquet` |
| `probe_vs_length.py` | Section 5: AUROC of the probe and of the character count on the 164, 138 and 26 labels, and the results of E15-R with the p-value of the permutation null | `vm_risultati/e15_oof_sonda_2026-09-23.npz`, `dati/*.json`, `vm_risultati/e15r/` |
| `v8_threshold.py` | Section 6: the two parameters of V8 | `dati/tabella_aggregata.json`, from `aggrega.py` |
| `compute_hours.py` | Appendix C: GPU-hours | `vm_risultati/e13/log`, `vm_risultati/e14/log`, `vm_risultati/vm_logs` |
| `fewshot_pool.py` | Appendix C: the few-shot pool of the variant generator, rebuilt from the public datasets and checked by SHA-256 | `vm_risultati/e13/fewshot_pool_ids.json`, `dati/*.json` |

The other numbers of the report come from the dated files in `archive/`: the pre-registrations and their outcomes in
`records/`, and `vm_risultati/`, `gpu/out/` and the Codabench records in `dati/`.

## What needs more than a laptop

- **A CPU and time.** The nested cross-validation of our probe, its permutation null and its combination with length
  (`archive/vm/e15r_ricalcolo.py`) read the released hidden states in `archive/vm_backup/estratto/work/aimo/e15/`
  (about 50 MB each, loadable without pickle). The run took 14.5 hours of computation on 5 cores of a laptop, in two
  parts because the computer restarted; `archive/vm/e15r_riprendi.py` resumes it.
- **GPUs and the original models.** The re-execution study E14 (official runtime image, two NVIDIA RTX PRO 6000
  Blackwell GPUs), the variant generation and labeling E13 (vLLM 0.28.0), the hidden-state extraction E15 and the score
  extraction E16. Their scripts are in `archive/vm/`; Appendix C of the report gives the commands and pins the model
  revisions by their Hugging Face snapshot hashes.

## The six preliminary datasets

`archive/dati.py` downloads them from the Hugging Face datasets server. The analyses used these files:

| File | SHA-256 |
|---|---|
| `aimo-interp-challenge-sample-full.json` | `b8a57039d902fbdd03313ea8bbdabb853626f5e4bee91650ece70f6e61f9eae4` |
| `val-sample.json` | `afae22a0b05dcf6d0b48376b510367e29c7cb1951172e2dc12bb2d0e354be000` |
| `augmented-sample-math-agg.json` | `3d168c2953ad8467c5b326eca33d49f5752fbdc94f2a9becb935eba03e5e8db7` |
| `augmented-sample-math-agg-filtered.json` | `a96bf90d9e18816dc1083bfdf75aeff263ad27ae8d15f584fbf11ff206712d7f` |
| `augmented-sample-math.json` | `991b74c0cbc6368bb76764b587b94fbdc0e007cd8ffa135eb3c9a780cbecfc0a` |
| `augmented-sample-math-full-filtered.json` | `1b62215f83ee17eb58b5577827a73385a5b50d8e1d5abe212ad4176f1d434b24` |
