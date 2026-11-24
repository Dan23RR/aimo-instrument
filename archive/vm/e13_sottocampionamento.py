"""E13, curva di sottocampionamento su DeepSeek: script ricostruito il 23/9/2026.

La curva del 9/9 (PREREG_E13.md, tabella «varianti per problema 2, 5, 10, 20, 40») veniva da
una sola estrazione casuale per taglia, senza script salvato. Qui la procedura e' fissata prima
di eseguirla: regola vecchia di E13 (base corretta in almeno 6 campioni su 10 e ogni variante
estratta corretta); per ogni problema si estraggono k varianti accettate senza reinserimento
(tutte, se sono meno di k), indipendentemente nelle condizioni A e B; kappa(A,B) sui problemi
con almeno una variante in entrambe; kappa(A, ufficiale) con l'etichetta ufficiale vecchia.
20 estrazioni per taglia, seme 20260923. Si riportano media, minimo e massimo.

Uso (dalla cartella AIMO): python vm/e13_sottocampionamento.py
"""
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
TAG = "deepseek-ai__DeepSeek-R1-0528-Qwen3-8B"
TAGLIE = (2, 5, 10, 20, 40)
N_ESTR, SEME, BASE_OK = 20, 20260923, 6


def kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def struttura(df):
    out = {}
    for pid, g in df.groupby("problem_id"):
        b = g[g.idx == -1]
        if len(b) == 0:
            continue
        v = g[g.idx >= 0]
        out[pid] = (int(b.iloc[0].n_corrette) >= BASE_OK, (v.n_corrette.to_numpy() > 0))
    return out


def etichette(s, k, rng):
    lab = {}
    for pid, (ok, corr) in s.items():
        if len(corr) == 0:
            continue
        sel = corr if len(corr) <= k else corr[rng.choice(len(corr), k, replace=False)]
        lab[pid] = bool(ok and sel.all())
    return lab


def main():
    A = struttura(pd.read_parquet(os.path.join(ARCH, "e13", "etich_%s_A.parquet" % TAG)))
    B = struttura(pd.read_parquet(os.path.join(ARCH, "e13", "etich_%s_B.parquet" % TAG)))
    uff = pd.read_parquet(os.path.join(ARCH, "gpu", "out", "feature_greedy.parquet")).set_index("problem_id").robust.to_dict()
    rng = np.random.RandomState(SEME)
    print("DeepSeek, regola vecchia (base >= %d/10 e ogni variante estratta corretta), %d estrazioni per taglia" % (BASE_OK, N_ESTR))
    print("%-8s | %-26s | %-26s | %-26s" % ("k", "robust A (media, min-max)", "kappa(A,B)", "kappa(A,ufficiale)"))
    for k in TAGLIE:
        rA, kAB, kAU = [], [], []
        for _ in range(N_ESTR):
            la, lb = etichette(A, k, rng), etichette(B, k, rng)
            com = sorted(set(la) & set(lb))
            kAB.append(kappa([la[p] for p in com], [lb[p] for p in com]))
            cu = sorted(p for p in la if p in uff)
            kAU.append(kappa([la[p] for p in cu], [int(uff[p]) for p in cu]))
            rA.append(np.mean(list(la.values())))
        f = lambda x: "%.4f (%.4f, %.4f)" % (np.mean(x), np.min(x), np.max(x))
        print("%-8d | %-26s | %-26s | %-26s" % (k, f(rA), f(kAB), f(kAU)))
    la, lb = etichette(A, 10 ** 6, rng), etichette(B, 10 ** 6, rng)
    com = sorted(set(la) & set(lb))
    print("tutte le varianti: robust A %.3f | kappa(A,B) %.4f (n=%d)" % (np.mean(list(la.values())), kappa([la[p] for p in com], [lb[p] for p in com]), len(com)))


if __name__ == "__main__":
    main()
