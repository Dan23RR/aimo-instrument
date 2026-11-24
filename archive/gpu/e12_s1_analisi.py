"""E12 / Stadio 1 — verdetto su H33 (self-consistency), protocollo di PREREG_E12.md.

Uso: python e12_s1_analisi.py [file_parquet]

Regole fissate PRIMA:
  - statistica primaria: UNA, AUROC di `accordo` contro l'etichetta loro, orientamento libero
    dichiarato, null permutazionale a 20.000 permutazioni (stesso codice di E11);
  - criterio di morte: AUROC(accordo) < 0.65;
  - informazione aggiuntiva (non decide): accuratezza bilanciata di una soglia su `accordo`
    in CV stratificata a 5 fold (una riga per problema = raggruppata), soglia scelta sul solo
    train fold; le feature secondarie `risposte`, `accordo_pesato`; confronto con n_char.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
SEED = 20260829
N_PERM = 20000
SOGLIA_MORTE = 0.65


def auroc(s, y):
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    o = np.argsort(s, kind="mergesort")
    yy = y[o]
    ss = s[o]
    # rango medio per i pareggi
    r = np.empty(len(ss))
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    p = yy.sum()
    n = len(yy) - p
    if p == 0 or n == 0:
        return float("nan")
    return (r[yy == 1].sum() - p * (p + 1) / 2) / (p * n)


def auroc_bidir(s, y):
    a = auroc(s, y)
    return max(a, 1 - a), (1 if a >= 0.5 else -1)


def null_p95(y, n_perm=N_PERM):
    rng = np.random.RandomState(SEED)
    y = np.asarray(y, int)
    vals = []
    for _ in range(n_perm):
        a = auroc(rng.normal(size=len(y)), y)
        vals.append(max(a, 1 - a))
    v = np.array(vals)
    return float(np.percentile(v, 95)), float(v.mean())


def acc_bilanciata(pred, y):
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    tpr = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
    tnr = (pred[y == 0] == 0).mean() if (y == 0).any() else float("nan")
    return (tpr + tnr) / 2


def cv_soglia(s, y, orient, k=5):
    """Soglia scelta sul train fold (massima acc. bilanciata), valutata sul test fold."""
    rng = np.random.RandomState(SEED)
    s = np.asarray(s, float) * orient
    y = np.asarray(y, int)
    idx1 = rng.permutation(np.where(y == 1)[0])
    idx0 = rng.permutation(np.where(y == 0)[0])
    folds = [np.concatenate([idx1[i::k], idx0[i::k]]) for i in range(k)]
    pred = np.zeros(len(y), int)
    soglie = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        cand = np.unique(s[tr])
        best, bt = -1, None
        for t in cand:
            b = acc_bilanciata((s[tr] >= t).astype(int), y[tr])
            if b > best:
                best, bt = b, t
        soglie.append(float(bt))
        pred[te] = (s[te] >= bt).astype(int)
    return acc_bilanciata(pred, y), (pred == y).mean(), soglie


def main():
    f = Path(sys.argv[1]) if len(sys.argv) > 1 else QUI / "out" / "e12_s1_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet"
    if not f.is_file():
        print("manca %s" % f)
        return
    s1 = pd.read_parquet(f)
    e6 = pd.read_parquet(QUI / "out" / "feature_greedy.parquet")[["problem_id", "robust"]]
    tab = {r["problem_id"]: r for r in json.loads((QUI.parent / "dati" / "tabella_aggregata.json").read_text(encoding="utf-8"))}
    d = s1.merge(e6, on="problem_id", how="inner").reset_index(drop=True)
    d["n_char"] = d.problem_id.map(lambda p: len(tab[p]["problema"]) if p in tab and tab[p].get("problema") else np.nan)
    y = d.robust.astype(int).to_numpy()
    n = len(d)
    print("=" * 74)
    print("E12 / STADIO 1 — VERDETTO H33  (n=%d, robust=%d = %.1f%%)" % (n, y.sum(), 100 * y.mean()))
    print("=" * 74)
    print("  campioni per problema: %s | accordo medio %.3f | risposte medie %.3f | finiti %.3f"
          % (sorted(d.k.unique().tolist()), d.accordo.mean(), d.risposte.mean(), d.frac_finiti.mean()))
    print("  accordo per classe: robust %.3f  spurious %.3f" % (d.accordo[y == 1].mean(), d.accordo[y == 0].mean()))
    print("  risposte per classe: robust %.3f  spurious %.3f" % (d.risposte[y == 1].mean(), d.risposte[y == 0].mean()))
    print()
    p95, media = null_p95(y)
    print("  null permutazionale (%d perm., orientamento libero): media %.4f, p95 = %.4f" % (N_PERM, media, p95))
    print("  criterio di morte fissato prima: AUROC(accordo) < %.2f" % SOGLIA_MORTE)
    print()
    print("  %-22s %8s %6s %9s  %9s  %s" % ("canale", "AUROC", "orient", "vs null", "acc.bil.CV", "esito"))
    ris = {}
    for nome, col, primario in (("H33  accordo", "accordo", True),
                                ("(sec) risposte", "risposte", False),
                                ("(sec) accordo_pesato", "accordo_pesato", False),
                                ("(sec) n_distinte", "n_distinte", False),
                                ("(sec) tok_medio", "tok_medio", False),
                                ("(info) n_char", "n_char", False)):
        x = d[col].to_numpy(float)
        ok = ~np.isnan(x)
        a, orient = auroc_bidir(x[ok], y[ok])
        ab, acc, soglie = cv_soglia(x[ok], y[ok], orient)
        if primario:
            esito = "SOPRAVVIVE" if a >= SOGLIA_MORTE else "MORTA"
        else:
            esito = "secondaria"
        print("  %-22s %8.4f %6s %9s  %9.4f  %s" % (nome, a, "+" if orient > 0 else "-",
                                                    "SI" if a >= p95 else "no", ab, esito))
        ris[col] = {"auroc": float(a), "orientamento": int(orient), "sopra_null": bool(a >= p95),
                    "acc_bilanciata_cv": float(ab), "acc_cv": float(acc), "soglie_cv": soglie, "esito": esito}
    print()
    # distribuzione di accordo per classe, per leggere il dato e non solo il numero
    print("  distribuzione di accordo (k=8):")
    for v in sorted(d.accordo.unique()):
        m = d.accordo == v
        print("    accordo=%.3f  n=%3d  robust=%2d  spurious=%2d" % (v, m.sum(), (m & (y == 1)).sum(), (m & (y == 0)).sum()))
    a33 = ris["accordo"]["auroc"]
    print()
    print("=" * 74)
    print("VERDETTO H33: %s  (AUROC %.4f contro soglia %.2f; null p95 %.4f)"
          % (ris["accordo"]["esito"], a33, SOGLIA_MORTE, p95))
    print("=" * 74)
    (QUI / "out" / "e12_s1_verdetto.json").write_text(
        json.dumps({"n": int(n), "robust": int(y.sum()), "null_p95": p95, "soglia_morte": SOGLIA_MORTE,
                    "risultati": ris}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
