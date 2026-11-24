"""Section 5: our probe on DeepSeek's hidden states against problem length (experiments E15 and E15-R).

Usage, from the repository root:  python archive/dati.py   (once)   then   python reproduce/probe_vs_length.py [E15R_FOLDER]
Inputs: vm_risultati/e15_oof_sonda_2026-09-23.npz (out-of-fold probe scores of the nested cross-validation, labels and
character counts of the 164 problems), dati/*.json (to know which labels are published), and optionally the output
folder of vm/e15r_ricalcolo.py, vm_risultati/e15r by default (second-stage combination on common outer folds, probe
retrained on the 138 published labels, permutation null with up to 1,000 permutations).
To rerun the probe from the released hidden states (vm_backup/estratto/work/aimo/e15/hs_*_sys.npz): vm/e15r_ricalcolo.py.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

from common import PROBE_OOF, ROOT, auroc, need, official_deepseek


def main():
    need(PROBE_OOF, "vm_risultati/e15_oof_sonda_2026-09-23.npz is part of the release")
    z = np.load(PROBE_OOF)
    pids, oof, y, nchar = list(z["problem_id"]), z["oof"], z["y"], z["n_char"]
    pub = official_deepseek().reindex(pids).published.to_numpy()
    print("Spearman correlation between probe score and character count: %+.2f" % spearmanr(oof, nchar).correlation)
    for name, m in (("all 164 labels", np.ones(len(y), bool)), ("138 published labels", pub), ("26 labels derived by us", ~pub)):
        print("%-24s n=%3d robust %2d | AUROC probe %.3f | AUROC character count %.3f" % (
            name, m.sum(), y[m].sum(), auroc(oof[m], y[m]), auroc(nchar[m], y[m])))
    cartelle = sys.argv[1:] or [c for c in [os.path.join(ROOT, "vm_risultati", "e15r")] if os.path.isdir(c)]
    for c in cartelle:
        s = os.path.join(c, "sintesi.json")
        if os.path.isfile(s):
            print("\nE15-R results in %s:" % os.path.relpath(c, ROOT))
            for riga in open(s, encoding="utf-8"):
                print("   " + riga.strip())
        n = os.path.join(c, "parte2_null.jsonl")
        if os.path.isfile(n):
            nul = np.array([json.loads(r)["auroc"] for r in open(n, encoding="utf-8")])
            stat = auroc(oof, y)
            print("   permutation null: %d permutations, mean %.4f, 95th percentile %.4f, %d at or above %.4f, p = %.4f" % (
                len(nul), nul.mean(), np.percentile(nul, 95), int((nul >= stat).sum()), stat, (1 + (nul >= stat).sum()) / (1 + len(nul))))


if __name__ == "__main__":
    main()
