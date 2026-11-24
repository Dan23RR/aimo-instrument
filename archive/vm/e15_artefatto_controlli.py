"""E15: controlli sull'artefatto-sonda degli organizzatori e sulla nostra sonda (23/9/2026).

Non pre-registrato: chiesto dalla seconda verifica avversaria del report. Tutto dai file salvati:
probe_artifact.pkl (artefatto ufficiale per DeepSeek), stati nascosti con il system prompt
dell'artefatto (hs_..._sys.npz), 164 etichette di feature_greedy.parquet, punteggi OOF della nostra
sonda (vm_risultati/e15_oof_sonda_2026-09-23.npz).

Uso (dalla cartella AIMO): python vm/e15_artefatto_controlli.py
"""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]


def cos(u, v):
    return float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))


def margini(a, X, gruppi, layer=None):
    out = []
    for g in gruppi:
        L = layer if layer is not None else int(g["best_layer_index"])
        pr = g["probes"].get(L, g["probes"].get(str(L)))
        W = np.asarray(pr["weights"], np.float32)
        b = np.asarray(pr["bias"], np.float32).reshape(-1)
        t = np.asarray(pr["threshold"], np.float32).reshape(-1)
        out.append(X[:, L, :] @ W.T + b - t)
    return np.concatenate(out, 1).mean(1)


def controllo_lunghezza(nome, s, y, nchar, ntok):
    Lt, Lc = np.log(ntok.astype(float)), np.log(nchar.astype(float))
    rt = s - np.polyval(np.polyfit(Lt, s, 1), Lt)
    rc = s - np.polyval(np.polyfit(Lc, s, 1), Lc)
    lungo = ntok >= np.quantile(ntok, 2 / 3)
    print("   %-38s AUROC %.3f | Spearman con n_char %+.3f | residuo su log(n_token) %.3f, su log(n_char) %.3f | terzile lungo (n=%d, robust %d) %.3f" % (
        nome, roc_auc_score(y, s), spearmanr(s, nchar).correlation, roc_auc_score(y, rt), roc_auc_score(y, rc),
        lungo.sum(), y[lungo].sum(), roc_auc_score(y[lungo], s[lungo])))


def main():
    a = pickle.load(open(os.path.join(E, "e15", "probe_artifacts", "probe_artifact.pkl"), "rb"))
    ps = a["probe_scores"]
    print("1. PUNTEGGI DI VALIDAZIONE SALVATI NELL'ARTEFATTO")
    for ct in ("NONE", "RANDOMIZATION"):
        sc = [p["score"] for p in ps if p["control_task"] == ct]
        print("   %-14s probe %d | esattamente 0,500: %d" % (ct, len(sc), sum(s == 0.5 for s in sc)))
    print("   migliore per fold (NONE): %s" % [round(max(p["score"] for p in ps if p["control_task"] == "NONE" and p["fold_index"] == f), 4) for f in range(4)])
    fl = defaultdict(list)
    lay = defaultdict(list)
    for p in ps:
        if p["control_task"] == "NONE":
            fl[(p["fold_index"], p["layer_index"])].append(p["score"])
            lay[p["layer_index"]].append(p["score"])
    k = max(fl, key=lambda x: np.mean(fl[x]))
    kl = max(lay, key=lambda x: np.mean(lay[x]))
    print("   media sui semi per (fold, layer), massimo: %.4f a %s | media su fold e semi per layer, massimo: %.4f al layer %d" % (np.mean(fl[k]), k, np.mean(lay[kl]), kl))
    ls = a["layer_scores"]
    kb = max(ls, key=ls.get)
    print("   layer_scores dell'artefatto (media che include i controlli): massimo %.4f al layer %s" % (ls[kb], kb))

    d = np.load(os.path.join(E, "e15", "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz"), allow_pickle=True)
    X, ids, ntok_all = d["X"].astype(np.float32), list(d["ids"]), np.asarray(d["n_token"])
    lab = pd.read_parquet(os.path.join(E, "gpu", "out", "feature_greedy.parquet")).set_index("problem_id").robust
    keep = [i for i, p in enumerate(ids) if p in lab.index]
    y = np.array([int(lab[ids[i]]) for i in keep])
    Xk = X[keep]
    tx = {}
    for nome in FONTI:
        for r in json.load(open(os.path.join(E, "dati", nome + ".json"), encoding="utf-8")):
            pid, t = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(t, str) and pid not in tx:
                tx[pid] = t
    nchar = np.array([len(tx[ids[i]]) for i in keep])
    ntok = ntok_all[keep]
    G = a["groups"]
    none = [g for g in G if g["control_task"] == "NONE"]
    ctrl = [g for g in G if g["control_task"] != "NONE"]

    print("\n2. MARGINI DELL'ARTEFATTO SULLE NOSTRE 164 ETICHETTE, E CONTROLLO DI LUNGHEZZA")
    uff = margini(a, Xk, G, layer=31)
    pre = margini(a, Xk, none)
    print("   aggregazione del codice di inferenza (layer 31, 40 probe): predetti robust %.1f%%" % (100 * (uff >= 0).mean()))
    print("   aggregazione pre-registrata (gruppi NONE al proprio layer): predetti robust %.1f%%" % (100 * (pre >= 0).mean()))
    controllo_lunghezza("codice di inferenza", uff, y, nchar, ntok)
    controllo_lunghezza("pre-registrata", pre, y, nchar, ntok)
    controllo_lunghezza("soli gruppi NONE al layer 31", margini(a, Xk, none, layer=31), y, nchar, ntok)
    controllo_lunghezza("soli gruppi di CONTROLLO al layer 31", margini(a, Xk, ctrl, layer=31), y, nchar, ntok)
    print("   AUROC per gruppo di controllo al layer 31: %s" % [round(roc_auc_score(y, margini(a, Xk, [g], layer=31)), 3) for g in ctrl])

    print("\n3. DIREZIONI DEI PROBE AL LAYER 31")
    W = [(g["control_task"], g["fold_index"], np.asarray(g["probes"][31]["weights"], np.float64)) for g in G]
    stesso_nr, diversi = [], []
    for (c1, f1, w1), (c2, f2, w2) in itertools.combinations(W, 2):
        for s in range(w1.shape[0]):
            if c1 != c2:
                stesso_nr.append(cos(w1[s], w2[s]))
        for s, t in itertools.product(range(w1.shape[0]), range(w2.shape[0])):
            if s != t:
                diversi.append(cos(w1[s], w2[t]))
    for c, f, w in W:
        for s, t in itertools.combinations(range(w.shape[0]), 2):
            diversi.append(cos(w[s], w[t]))
    print("   stesso seme, gruppo con etichette vere contro gruppo di controllo: coseno da %.3f a %.3f (mediana %.3f)" % (min(stesso_nr), max(stesso_nr), np.median(stesso_nr)))
    print("   semi diversi, qualunque gruppo: coseno da %.3f a %.3f (mediana %.3f)" % (min(diversi), max(diversi), np.median(diversi)))
    rng = np.random.default_rng(20260923)
    au = np.array([roc_auc_score(y, Xk[:, 31, :] @ rng.normal(size=Xk.shape[2])) for _ in range(500)])
    print("   500 direzioni casuali al layer 31: AUROC p5 %.3f, p95 %.3f; quota con |AUROC - 0,5| >= 0,18: %.3f" % (
        np.percentile(au, 5), np.percentile(au, 95), (np.abs(au - 0.5) >= 0.18).mean()))

    print("\n4. LA NOSTRA SONDA (punteggi OOF salvati)")
    o = np.load(os.path.join(BASE, "vm_risultati", "e15_oof_sonda_2026-09-23.npz"), allow_pickle=True)
    oof, yy, nc, nt = o["oof"], o["y"], o["n_char"], o["n_token"]
    controllo_lunghezza("sonda nostra", oof, yy, nc, nt)
    Lc = np.log(nc.astype(float))
    rc = Lc - np.polyval(np.polyfit(oof, Lc, 1), oof)
    print("   log(n_char) al netto del punteggio della sonda: AUROC %.3f" % roc_auc_score(yy, rc))


if __name__ == "__main__":
    main()
