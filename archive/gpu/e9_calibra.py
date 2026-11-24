"""E9 stadio 2 — soglia per-modello dalle etichette NOSTRE, con la regola pre-registrata.

Uso: python e9_calibra.py stadio2 Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think ...

Per ogni modello incrocia out/e9_<tag>_<model>.parquet (etichette con perturbazioni nostre)
e out/e9feat_<tag>_<model>.parquet (punteggi del regressore ufficiale). Regola identica a v6
(PREREG_V6/E9): soglia = massima accuratezza sull'etichetta, a parita' la piu' vicina a 0,576.
Scrive soluzione/soglie_per_modello.json, che solution.py usa se presente.

Riporta anche, per onesta', quante coppie hanno etichetta definita e il tasso robust_ours.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
OUT = QUI / "out"
DEST = QUI.parent / "soluzione" / "soglie_per_modello.json"
EREDITATA = 0.576


def taratura(scores, robust, ereditata=EREDITATA):
    s = np.asarray(scores, float)
    y = np.asarray(robust, bool)
    xs = sorted(set(s.tolist()))
    cand = [ereditata] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[0] - 1e-6, xs[-1] + 1e-6]
    acc = lambda t: int(((s < t) == y).sum())
    return float(max(cand, key=lambda t: (acc(t), -abs(t - ereditata)))), acc


def main(tag, modelli, prefisso="e9", colonna="robust_ours"):
    """prefisso: 'e9' (perturbazioni di forma) o 'e10' (traduzione); colonna: etichetta da usare."""
    tabella = {}
    if DEST.is_file():
        tabella = json.loads(DEST.read_text(encoding="utf-8"))
    for m in modelli:
        safe = m.replace("/", "__")
        fl = OUT / ("%s_%s_%s.parquet" % (prefisso, tag, safe))
        ff = OUT / ("e9feat_%s_%s.parquet" % (tag, safe))
        if not (fl.is_file() and ff.is_file()):
            print("%-40s manca %s" % (m, "etichette" if not fl.is_file() else "feature"))
            continue
        lab = pd.read_parquet(fl)[["problem_id", "ha_risposta", colonna]]
        feat = pd.read_parquet(ff)
        d = feat.merge(lab, on="problem_id", how="inner", suffixes=("_feat", ""))
        d = d[d["ha_risposta"].astype(bool)].reset_index(drop=True)
        y = d[colonna].astype(bool)
        if len(d) < 20 or y.nunique() < 2:
            print("%-40s n=%d: troppo poco o degenere, soglia NON calibrata" % (m, len(d)))
            continue
        t, acc = taratura(d.score, y)
        n = len(d)
        print("%-40s n=%3d %s=%.0f%%  soglia=%.3f (ereditata %.3f)  acc_in-sample %.3f vs %.3f"
              % (m, n, colonna, 100 * y.mean(), t, EREDITATA, acc(t) / n, acc(EREDITATA) / n))
        tabella[m] = {"soglia": round(t, 4), "n": int(n), colonna: round(float(y.mean()), 4),
                      "tag": tag, "fonte": prefisso}
    DEST.write_text(json.dumps(tabella, indent=1, ensure_ascii=False), encoding="utf-8")
    print("scritto", DEST, "con", len(tabella), "modelli")


if __name__ == "__main__":
    # uso: python e9_calibra.py <tag> <modello>... [--prefisso e9|e10] [--colonna robust_ours|robust_trad]
    argv = sys.argv[1:]
    prefisso, colonna = "e9", "robust_ours"
    if "--prefisso" in argv:
        i = argv.index("--prefisso"); prefisso = argv[i + 1]; del argv[i:i + 2]
    if "--colonna" in argv:
        i = argv.index("--colonna"); colonna = argv[i + 1]; del argv[i:i + 2]
    main(argv[0], argv[1:], prefisso=prefisso, colonna=colonna)
