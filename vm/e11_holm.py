"""E11: p-value permutazionali di H30 e H31 con il null dello script originale, e Holm su 2 test.

gpu/e11_s2_analisi.py dichiara nel docstring una correzione di Holm su 2 test primari ma non la
applica: il verdetto confronta ogni AUROC con il p95 del null di una singola statistica. Qui si
usa lo stesso null (20.000 punteggi casuali, orientamento libero, seme 20260829) per calcolare il
p-value di ciascuna statistica e applicare Holm (23/9/2026, verifica del report).

Uso (dalla cartella AIMO): python vm/e11_holm.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "gpu"))
sys.argv = [sys.argv[0]]
import e11_s2_analisi as A  # noqa: E402


def main():
    v = json.load(open(os.path.join(BASE, "gpu", "out", "e11_s2_verdetto.json"), encoding="utf-8"))
    print("verdetto salvato:", {k: v[k] for k in v if k in ("n", "robust", "null_p95", "M0_TESTO")})
    righe = [(k, x) for k, x in v["risultati"].items() if k in ("Z_C1", "Z_C1b")]
    for k, x in righe:
        print("   %s: AUROC %.4f, orientamento %s, vs null %s" % (k, x["auroc"], x.get("orientamento"), x.get("vs_null")))
    d = pd.read_parquet(os.path.join(BASE, "gpu", "out", "e11_s2_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet"))
    y = d.robust.astype(int).to_numpy() if "robust" in d.columns else None
    n = int(v["n"])
    rng = np.random.RandomState(A.SEED)
    s = rng.normal(size=(A.N_PERM, n))
    yy = y if y is not None else np.r_[np.ones(int(v["robust"])), np.zeros(n - int(v["robust"]))].astype(int)
    nul = np.array([max(A.auroc(s[i], yy), 1 - A.auroc(s[i], yy)) for i in range(A.N_PERM)])
    print("null ricostruito: p95 %.4f (salvato %.4f)" % (np.percentile(nul, 95), v["null_p95"]))
    ps = []
    for k, x in righe:
        p = float((nul >= x["auroc"]).mean())
        ps.append((p, k, x["auroc"]))
        print("   %s: p permutazionale %.4f" % (k, p))
    ps.sort()
    m = len(ps)
    print("Holm su %d test (alfa 0,05):" % m)
    for i, (p, k, a) in enumerate(ps):
        soglia = 0.05 / (m - i)
        print("   %s: p %.4f contro %.4f -> %s" % (k, p, soglia, "supera" if p <= soglia else "NON supera"))
        if p > soglia:
            break


if __name__ == "__main__":
    main()
