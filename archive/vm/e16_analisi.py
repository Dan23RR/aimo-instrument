"""E16 e affini: tutte le cifre delle sezioni 2.3(b), 8.2 e 8.4 del report, da output salvati.

Scritto il 23/9/2026 (seconda verifica avversaria): prima non esisteva uno script di analisi.
Input: vm_risultati/e16_*.jsonl (estrazione con la sola CUBLAS_WORKSPACE_CONFIG, quattro modelli),
vm_risultati/e16nodet/rep0.jsonl e rep1.jsonl (DeepSeek, senza flag), vm_risultati/e14/out/*.jsonl,
il regressore ufficiale (soluzione/uncertainty_artifacts/...joblib) e le nostre feature A100
(gpu/out/feature_greedy.parquet). Etichette: le 164 di feature_greedy.parquet.

Uso (dalla cartella AIMO): python vm/e16_analisi.py
"""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(BASE, "vm_risultati")
FG = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "gpu", "out", "feature_greedy.parquet")
ART = os.path.join(BASE, "soluzione", "uncertainty_artifacts", "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib")
AGG = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati", "augmented-sample-math-agg.json")
SEME = 20260923


def leggi(f):
    return {json.loads(l)["problem_id"]: json.loads(l) for l in open(f, encoding="utf-8") if l.strip()}


def main():
    fg = pd.read_parquet(FG)
    lab = dict(zip(fg.problem_id, fg.robust.astype(int)))
    art = joblib.load(ART)
    soglia = float(art["decision_threshold"])
    print("regressore: %s | soglia %.4f | feature %d" % (type(art["estimator"]).__name__, soglia, len(art["feature_names"])))
    runs = {"flag (solo CUBLAS)": leggi(os.path.join(R, "e16_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.jsonl")),
            "senza flag rep0": leggi(os.path.join(R, "e16nodet", "rep0.jsonl")),
            "senza flag rep1": leggi(os.path.join(R, "e16nodet", "rep1.jsonl"))}
    P = sorted(p for p in runs["senza flag rep0"] if p in lab)
    y = np.array([lab[p] for p in P])
    maggioranza = max(y.mean(), 1 - y.mean())
    print("\n8.2  n=%d etichettati (robust %d) | costante di maggioranza %.3f" % (len(P), y.sum(), maggioranza))
    s = {}
    for k, r in runs.items():
        s[k] = np.array([r[p]["score"] for p in P])
        pred = s[k] < soglia
        tutti = np.array([r[p]["score"] for p in r])
        print("   %-20s AUROC grezza %.4f | nel verso della regola %.4f | predetti robust %d/%d = %.1f%% (su 174: %.1f%%) | accuratezza della regola %.3f" % (
            k, roc_auc_score(y, s[k]), roc_auc_score(y, -s[k]), pred.sum(), len(P), 100 * pred.mean(), 100 * (tutti < soglia).mean(), (pred == y).mean()))
    a, b, c = runs["flag (solo CUBLAS)"], runs["senza flag rep0"], runs["senza flag rep1"]
    ids = sorted(a)
    print("   score identici: flag vs rep0 %d/%d | rep0 vs rep1 %d/%d | differenza assoluta mediana rep0-rep1 %.3f" % (
        sum(a[p]["score"] == b[p]["score"] for p in ids), len(ids), sum(b[p]["score"] == c[p]["score"] for p in ids), len(ids),
        np.median([abs(b[p]["score"] - c[p]["score"]) for p in ids])))
    print("   Spearman fra rep0 e rep1: %.3f sui 164 etichettati, %.3f sui 174" % (
        spearmanr(s["senza flag rep0"], s["senza flag rep1"]).correlation, spearmanr([b[p]["score"] for p in ids], [c[p]["score"] for p in ids]).correlation))
    rng = np.random.default_rng(SEME)
    o1 = -s["senza flag rep1"]; o0 = -s["senza flag rep0"]
    a1 = roc_auc_score(y, o1)
    perm = np.array([roc_auc_score(rng.permutation(y), o1) for _ in range(20000)])
    print("   rep1 nel verso della regola %.4f: p permutazionale unilaterale %.4f, bilaterale %.4f (20000 permutazioni)" % (
        a1, (perm >= a1).mean(), (np.abs(perm - 0.5) >= abs(a1 - 0.5)).mean()))
    d = []
    for _ in range(5000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], o1[i]) - roc_auc_score(y[i], o0[i]))
    print("   differenza rep1 - rep0 nel verso della regola %.4f, IC95 bootstrap appaiato [%.3f, %.3f]" % (
        a1 - roc_auc_score(y, o0), np.percentile(d, 2.5), np.percentile(d, 97.5)))
    tr = {r["problem_id"] for r in json.load(open(AGG, encoding="utf-8"))}
    print("   problemi etichettati presenti nell'addestramento del regressore (augmented-sample-math-agg): %d/%d" % (sum(p in tr for p in P), len(P)))

    print("\n8.4  quote predette robust con il regressore e la soglia di DeepSeek sulle feature di ciascun modello")
    for f in sorted(glob.glob(os.path.join(R, "e16_*.jsonl"))):
        r = leggi(f)
        sc = np.array([x["score"] for x in r.values()])
        print("   %-48s n=%d | %.1f%% sotto la soglia | score min %.3f max %.3f" % (os.path.basename(f), len(sc), 100 * (sc < soglia).mean(), sc.min(), sc.max()))

    print("\n2.3(b)  generation_frac_selected_is_top1 nel percorso ufficiale e soglie degli split della foresta")
    est = art["estimator"]
    rf = est.steps[-1][1] if hasattr(est, "steps") else est
    j = list(art["feature_names"]).index("generation_frac_selected_is_top1")
    thr = [t.tree_.threshold[t.tree_.feature == j] for t in rf.estimators_]
    thr = np.concatenate(thr)
    tot_split = sum((t.tree_.feature >= 0).sum() for t in rf.estimators_)
    print("   split sulla feature %d su %d (%.1f%%) | soglie da %.6f a %.6f | importanza %.3f" % (
        len(thr), tot_split, 100 * len(thr) / tot_split, thr.min(), thr.max(), rf.feature_importances_[j]))
    v_all, v_ds = [], []
    for f in glob.glob(os.path.join(R, "e14", "out", "*.jsonl")):
        for r in leggi(f).values():
            v_all.append(r["generation_frac_selected_is_top1"]); v_ds.append(r["generation_frac_selected_is_top1"])
    for f in glob.glob(os.path.join(R, "e16_*.jsonl")) + [os.path.join(R, "e16nodet", "rep0.jsonl"), os.path.join(R, "e16nodet", "rep1.jsonl")]:
        for r in leggi(f).values():
            v_all.append(r["generation_frac_selected_is_top1"])
            if "deepseek" in f.lower() or "e16nodet" in f:
                v_ds.append(r["generation_frac_selected_is_top1"])
    for nome, v in (("quattro modelli", np.array(v_all)), ("solo DeepSeek", np.array(v_ds))):
        dentro = ((v >= thr.min()) & (v <= thr.max())).mean()
        print("   %-16s n=%d | uguali a 1,0: %d | min %.5f max %.5f | dentro [soglia minima, soglia massima] %.1f%%" % (
            nome, len(v), int((v == 1.0).sum()), v.min(), v.max(), 100 * dentro))
    print("   estrazione A100 (feature_greedy): frac_top1 uguale a 1,0 su %d/%d" % (int((fg.frac_top1 == 1.0).sum()), len(fg)))

    print("\n8.2  lo stesso regressore sulle nostre feature A100 (estrattore nostro, non ufficiale)")
    m = {"generation_mean_logprob": "mean_logprob", "generation_mean_nll": "mean_nll", "generation_ppl": "ppl",
         "generation_min_logprob": "min_logprob", "generation_max_logprob": "max_logprob", "generation_std_logprob": "std_logprob",
         "generation_min_k_logprob": "min_k_logprob", "generation_mean_entropy": "mean_entropy", "generation_mean_top1_prob": "mean_top1",
         "generation_mean_top2_margin": "mean_margin", "generation_frac_selected_is_top1": "frac_top1",
         "generation_frac_high_conf_tokens": "frac_alta", "generation_frac_low_conf_tokens": "frac_bassa"}
    ya = fg.robust.astype(int).to_numpy()
    for nome, mp in (("mean_prob = media top-1 (come l'estrattore ufficiale in greedy)", fg["mean_top1"]),
                     ("mean_prob = exp(mean_logprob) (bozze precedenti)", np.exp(fg["mean_logprob"]))):
        X = pd.DataFrame({k: fg[v] for k, v in m.items()})
        X["generation_mean_prob"] = mp
        sc = art["estimator"].predict(X[art["feature_names"]])
        pred = sc < soglia
        print("   %-62s AUROC grezza %.4f, nel verso della regola %.4f | predetti robust %.1f%% | accuratezza %.3f" % (
            nome, roc_auc_score(ya, sc), roc_auc_score(ya, -sc), 100 * pred.mean(), (pred == ya).mean()))


if __name__ == "__main__":
    main()
