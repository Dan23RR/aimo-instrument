"""E15, controllo sulla lunghezza della sonda nostra: script ricostruito il 23/9/2026.

Il controllo era descritto in PREREG_E15.md (esito del 14/9) ma lo script non era stato salvato.
Qui la procedura e' fissata e scritta PRIMA di eseguirla:

1. Sonda nostra: punteggi out-of-fold con la stessa CV nidificata di e15_valuta.py
   (funzione cv_nidificata importata, stato gli stessi dati con system prompt, seme 20260914).
   Controllo: l'AUROC OOF deve tornare 0,7097 come nel log del 14/9, altrimenti ci si ferma.
2. Lunghezze: n_char = caratteri del testo originale del problema (stesse fonti di
   e15_estrai.py); n_token = token del prompt formattato, salvati nel file .npz.
3. Statistiche:
   - Spearman fra punteggio OOF della sonda e n_char;
   - AUROC del residuo della sonda regredita linearmente su log(n_token);
   - AUROC della sonda dentro i terzili di n_token (terzili per rango: 54/55/55 problemi);
   - CV di secondo livello: 5 fold stratificati (costruiti come in cv_nidificata, seme 20260915),
     regressione logistica (class_weight balanced, C=1, feature standardizzate) su
     (a) log(n_char), (b) punteggio OOF della sonda, (c) entrambi. Si riporta l'AUROC OOF.
     Stessa procedura con 20 semi diversi (20260915..20260934), per mostrare quanto dipende
     dalla divisione in fold.

Uso (dalla cartella AIMO): python vm/e15_controllo_lunghezza.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
NPZ = os.path.join(ARCH, "e15", "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz")
FG = os.path.join(ARCH, "gpu", "out", "feature_greedy.parquet")
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]

sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e15_valuta as V  # noqa: E402  (solo le funzioni; gli argomenti di default non si usano)


def testi():
    t = {}
    for nome in FONTI:
        for r in json.load(open(os.path.join(ARCH, "dati", "%s.json" % nome), encoding="utf-8")):
            pid, x = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(x, str) and pid not in t:
                t[pid] = x
    return t


def cv2(F, y, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    idx1 = rng.permutation(np.where(y == 1)[0]); idx0 = rng.permutation(np.where(y == 0)[0])
    folds = [np.concatenate([idx1[i::5], idx0[i::5]]) for i in range(5)]
    oof = np.zeros(len(y))
    for i in range(5):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(F[te]))
    return V.auroc(oof, y)


def main():
    lab = pd.read_parquet(FG)[["problem_id", "robust"]]
    d = np.load(NPZ, allow_pickle=True)
    X, ids, ntok = d["X"].astype(np.float32), list(d["ids"]), np.asarray(d["n_token"])
    pos = {p: i for i, p in enumerate(ids)}
    keep = [p for p in lab.problem_id if p in pos]
    idx = [pos[p] for p in keep]
    X, ntok = X[idx], ntok[idx]
    y = lab.set_index("problem_id").loc[keep, "robust"].astype(int).to_numpy()
    tx = testi()
    nchar = np.array([len(tx[p]) for p in keep])
    print("n=%d | robust %d | token del prompt: mediana %d" % (len(y), y.sum(), int(np.median(ntok))))

    oof, scelte = V.cv_nidificata(X, y, np.random.RandomState(20260914))
    a = V.auroc(oof, y)
    print("sonda nostra, AUROC OOF %.4f (atteso 0,7097) | layer/C per fold %s" % (a, scelte))
    if abs(a - 0.7097) > 5e-4:
        print("STOP: la sonda non si riproduce, il controllo non prosegue."); sys.exit(1)

    from scipy.stats import spearmanr
    print("Spearman(sonda, n_char) %+.3f | Spearman(sonda, n_token) %+.3f" % (
        spearmanr(oof, nchar).correlation, spearmanr(oof, ntok).correlation))
    L = np.log(ntok.astype(float))
    beta = np.polyfit(L, oof, 1)
    print("AUROC del residuo della sonda su log(n_token): %.4f" % V.auroc(oof - np.polyval(beta, L), y))
    np.savez(os.path.join(BASE, "vm_risultati", "e15_oof_sonda_2026-09-23.npz"), problem_id=np.array(keep), oof=oof, y=y, n_char=nchar, n_token=ntok)
    ordine = np.argsort(np.argsort(ntok, kind="mergesort"), kind="mergesort")
    terz = np.minimum(ordine * 3 // len(y), 2)
    print("terzili per rango di n_token (55/55/54, pareggi spezzati dall'ordine dei dati):")
    for k, nome in enumerate(["corti", "medi", "lunghi"]):
        m = terz == k
        print("   terzile %-6s n=%3d robust %2d | AUROC sonda %.3f | AUROC n_char %.3f" % (
            nome, m.sum(), y[m].sum(), V.auroc(oof[m], y[m]), V.auroc(nchar[m], y[m])))
    q1, q2 = np.quantile(ntok, [1 / 3, 2 / 3])
    print("terzili per soglia sui quantili di n_token (%.1f, %.1f; i pareggi alla soglia vanno nel terzile superiore):" % (q1, q2))
    for nome, m in (("corti", ntok < q1), ("medi", (ntok >= q1) & (ntok < q2)), ("lunghi", ntok >= q2)):
        print("   terzile %-6s n=%3d robust %2d | AUROC sonda %.3f | AUROC n_char %.3f" % (
            nome, m.sum(), y[m].sum(), V.auroc(oof[m], y[m]), V.auroc(nchar[m], y[m])))
    print("AUROC diretto di n_char (nessun adattamento): %.4f" % V.auroc(nchar, y))

    F_char = np.log(nchar)[:, None]; F_sonda = oof[:, None]; F_due = np.column_stack([np.log(nchar), oof])
    print("CV di secondo livello, seme 20260915: n_char %.4f | sonda %.4f | entrambi %.4f" % (
        cv2(F_char, y, 20260915), cv2(F_sonda, y, 20260915), cv2(F_due, y, 20260915)))
    r = np.array([[cv2(F, y, s) for F in (F_char, F_sonda, F_due)] for s in range(20260915, 20260935)])
    for j, nome in enumerate(["n_char", "sonda", "entrambi"]):
        print("   20 semi, %-8s media %.4f  min %.4f  max %.4f" % (nome, r[:, j].mean(), r[:, j].min(), r[:, j].max()))
    dd = r[:, 2] - r[:, 0]
    print("   guadagno di 'entrambi' su n_char: media %+.4f  min %+.4f  max %+.4f" % (dd.mean(), dd.min(), dd.max()))


if __name__ == "__main__":
    main()
