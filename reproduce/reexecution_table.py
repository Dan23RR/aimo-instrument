"""Section 3 and Table 1: repeatability of the official pipeline on DeepSeek, 40 public problems (experiment E14).

Usage, from the repository root:  python reproduce/reexecution_table.py
Inputs: vm_risultati/e14/out/e14_<condition>_rep<r>_gpu<g>_<tag>.jsonl (one line per problem: prediction and SHA-256 of
the generated text), vm_risultati/e14/log/*.log (timings), gpu/out/feature_greedy.parquet (the labels of the 40 problems),
and, if present, dati/*.json (the six preliminary datasets, to tell published labels from those we derived).
Conditions: C0 default (batch 8), C1 batch 1, C2 deterministic algorithms, C3 two processes on one card,
C4 default on the second card, C5 compilation disabled.
"""
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd

from common import DATA, E14, OFFICIAL_DS, PRELIMINARY, need, official_deepseek

NAMES = {"C0": "Default (batch 8)", "C1": "Batch size 1", "C3": "Two processes on one card",
         "C4": "Second physical card", "C2": "Deterministic algorithms", "C5": "Compilation disabled"}


def main():
    need(os.path.join(E14, "out"), "the E14 outputs are part of the release (vm_risultati/e14)")
    runs = {}
    for f in sorted(glob.glob(os.path.join(E14, "out", "e14_*.jsonl"))):
        key = os.path.basename(f)[4:-6]
        runs[key] = {r["problem_id"]: r for r in map(json.loads, open(f, encoding="utf-8"))}
    sec = {}
    for f in glob.glob(os.path.join(E14, "log", "*.log")):
        m = re.search(r"gen (\d+)s = ([\d.]+) s/item", open(f, encoding="utf-8", errors="replace").read())
        if m:
            sec[os.path.basename(f)[:-4]] = float(m.group(2))
    label = pd.read_parquet(OFFICIAL_DS).set_index("problem_id").robust.astype(bool)
    by = defaultdict(list)
    for k in runs:
        by[k.split("_")[0]].append(k)
    base = sum(sec[k] for k in by["C0"]) / len(by["C0"])
    print("%-28s %-5s %-4s %-9s %-9s %-26s %s" % ("configuration", "card", "runs", "crossings", "identical", "correct per run", "s/item, cost"))
    for c in ("C0", "C1", "C3", "C4", "C2", "C5"):   # the order of Figure 1
        ks = sorted(by[c])
        pids = sorted(runs[ks[0]])
        cross = sum(1 for p in pids if len({runs[k][p]["robust_pred"] for k in ks}) > 1)
        ident = sum(1 for p in pids if len({runs[k][p]["sha256_testo"] for k in ks}) == 1)
        correct = [sum(1 for p in pids if bool(runs[k][p]["robust_pred"]) == bool(label[p])) for k in ks]
        cards = sorted({k.split("_")[2][3:] for k in ks})
        s = sum(sec[k] for k in ks) / len(ks)
        print("%-28s %-5s %-4d %-9s %-9s %-26s %.1f, %+.0f%%" % (NAMES[c], ",".join(cards), len(ks), "%d/40" % cross,
              "%d/40" % ident, ", ".join(map(str, correct)), s, 100 * (s / base - 1)))
    states = defaultdict(list)
    for k, r in runs.items():
        states[tuple(r[p]["sha256_testo"] for p in sorted(r))].append(k)
    print("\n%d executions, %d distinct outputs (byte-identical generated text on all 40 problems within each):" % (len(runs), len(states)))
    for i, ks in enumerate(sorted(states.values(), key=lambda v: sorted(v)[0])):
        print("   output %2d: %s" % (i, ", ".join(sorted(ks))))
    majority = sum(1 for p in sorted(runs[by["C0"][0]]) if not label[p])
    print("majority constant (always non-robust): %d/40 correct; robust labels: %d/40" % (majority, 40 - majority))
    if all(os.path.exists(os.path.join(DATA, n + ".json")) for n in PRELIMINARY):
        sub = official_deepseek().loc[sorted(runs[by["C0"][0]])]
        print("labels of the 40 problems: %d published, %d derived with the conjunction; robust: %d, of which %d derived" % (
            int(sub.published.sum()), int((~sub.published).sum()), int(sub.robust.sum()), int(sub[~sub.published].robust.sum())))
    else:
        print("which of the 40 labels are published: run first  python archive/dati.py")


if __name__ == "__main__":
    main()
