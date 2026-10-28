"""E15 — verdetto: sonda LORO (H35) e sonda NOSTRA (H36) contro le 164 etichette vere.

Uso: python e15_valuta.py [--npz-sys ...] [--npz-nosys ...] [--n-perm 500]
Protocollo in PREREG_E15.md, scritto prima. Il layer e la regolarizzazione della sonda nostra
si scelgono SOLO dentro la CV interna; il null permutazionale ripete l'intera procedura.
"""
import argparse
import pickle

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--npz-sys", default="/work/aimo/e15/hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz")
ap.add_argument("--npz-nosys", default="/work/aimo/e15/hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.npz")
ap.add_argument("--artefatto", default="/work/aimo/e15/probe_artifacts/probe_artifact.pkl")
ap.add_argument("--etichette", default="/work/aimo/gpu/out/feature_greedy.parquet")
ap.add_argument("--n-perm", type=int, default=500)
ap.add_argument("--seed", type=int, default=20260914)
args = ap.parse_args()


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); o = np.argsort(s, kind="mergesort"); yy = y[o]; ss = s[o]
    r = np.empty(len(ss)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    p = yy.sum(); n = len(yy) - p
    return float((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")


def bal_acc(pred, y):
    pred = np.asarray(pred, int); y = np.asarray(y, int)
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def carica(npz):
    d = np.load(npz, allow_pickle=True)
    return d["X"].astype(np.float32), list(d["ids"])


def allinea(X, ids, lab):
    m = {p: i for i, p in enumerate(ids)}
    keep = [p for p in lab.problem_id if p in m]
    idx = [m[p] for p in keep]
    y = lab.set_index("problem_id").loc[keep, "robust"].astype(int).to_numpy()
    return X[idx], y, keep


# ---------------- H35: la sonda loro, applicata come probe_inference.py ----------------
def margine_ufficiale(a, X):
    """X: [n, L+1, hidden]. Media dei margini su tutti i gruppi NONE e tutti i semi, al layer di ciascun gruppo."""
    margini = []
    for g in a["groups"]:
        if g.get("control_task") != "NONE":
            continue
        L = int(g["best_layer_index"]); pr = g["probes"][L]
        for w, b, t in zip(pr["weights"], pr["bias"], pr["threshold"]):
            w = np.asarray(w, np.float32)
            margini.append(X[:, L, :] @ w + float(b) - float(t))
    return np.mean(np.stack(margini, 0), 0)


# ---------------- H36: la sonda nostra, CV nidificata ----------------
def _fit_logreg(Xtr, ytr, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return lambda Z: clf.decision_function(sc.transform(Z))


def cv_nidificata(X, y, rng, k_out=5, k_in=4, Cs=(0.001, 0.01, 0.1, 1.0)):
    """Restituisce i punteggi OOF: per ogni fold esterno, layer e C scelti con CV interna sul solo train."""
    n, L1, _ = X.shape
    idx1 = rng.permutation(np.where(y == 1)[0]); idx0 = rng.permutation(np.where(y == 0)[0])
    folds = [np.concatenate([idx1[i::k_out], idx0[i::k_out]]) for i in range(k_out)]
    oof = np.zeros(n); scelte = []
    for i in range(k_out):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(k_out) if j != i])
        # CV interna: sub-fold del train
        t1 = rng.permutation(tr[y[tr] == 1]); t0 = rng.permutation(tr[y[tr] == 0])
        inner = [np.concatenate([t1[j::k_in], t0[j::k_in]]) for j in range(k_in)]
        best = (-1, None, None)
        for L in range(1, L1):  # il layer 0 e' l'embedding
            for C in Cs:
                s = np.zeros(len(tr)); pos = {v: q for q, v in enumerate(tr)}
                for j in range(k_in):
                    ite = inner[j]; itr = np.concatenate([inner[q] for q in range(k_in) if q != j])
                    f = _fit_logreg(X[itr, L, :], y[itr], C)
                    s[[pos[v] for v in ite]] = f(X[ite, L, :])
                a = auroc(s, y[tr])
                if a > best[0]:
                    best = (a, L, C)
        _, L, C = best
        f = _fit_logreg(X[tr, L, :], y[tr], C)
        oof[te] = f(X[te, L, :]); scelte.append((L, C, round(best[0], 3)))
    return oof, scelte


def main():
    lab = pd.read_parquet(args.etichette)[["problem_id", "robust"]]
    a = pickle.load(open(args.artefatto, "rb"))
    print("=" * 76); print("E15 — VERDETTO"); print("=" * 76)
    Xs, ids_s = carica(args.npz_sys); Xs, y, keep = allinea(Xs, ids_s, lab)
    print("n=%d problemi con etichetta | robust %.1f%% | layer %d | hidden %d" % (len(y), 100 * y.mean(), Xs.shape[1], Xs.shape[2]))

    # null permutazionale per una statistica AUROC su n, orientamento libero
    rng = np.random.RandomState(args.seed)
    nul = np.array([max(a_, 1 - a_) for a_ in (auroc(rng.normal(size=len(y)), y) for _ in range(20000))])
    p95 = float(np.percentile(nul, 95)); print("null permutazionale AUROC (20000, orientamento libero): p95 = %.4f" % p95); print()

    # ---- H35
    m = margine_ufficiale(a, Xs)
    a35 = auroc(m, y); pred35 = (m >= 0).astype(int)
    print("H35  sonda LORO (ensemble ufficiale, system prompt loro)")
    print("     AUROC %.4f (orientato %.4f) | acc. bilanciata regola margine>=0: %.4f | predice robust %.1f%%"
          % (a35, max(a35, 1 - a35), bal_acc(pred35, y), 100 * pred35.mean()))
    print("     esito: %s" % ("SOPRAVVIVE" if max(a35, 1 - a35) >= p95 else "MORTA (sotto il null)"))
    print()

    # ---- H36, con e senza system prompt
    for nome, npz in (("con system prompt", args.npz_sys), ("senza system prompt", args.npz_nosys)):
        X, ids = carica(npz); X, yy, _ = allinea(X, ids, lab)
        rng = np.random.RandomState(args.seed)
        oof, scelte = cv_nidificata(X, yy, rng)
        a36 = auroc(oof, yy)
        print("H36  sonda NOSTRA, CV nidificata, %s" % nome)
        print("     AUROC OOF %.4f | acc. bilanciata OOF (soglia 0) %.4f | layer/C scelti per fold: %s" % (a36, bal_acc((oof >= 0).astype(int), yy), scelte))
        # null nidificato: la stessa procedura su etichette permutate (costoso: n_perm ridotto)
        nperm = min(args.n_perm, 60)
        rngp = np.random.RandomState(args.seed + 1); nulls = []
        for q in range(nperm):
            yp = rngp.permutation(yy)
            o, _ = cv_nidificata(X, yp, np.random.RandomState(args.seed + 100 + q), Cs=(0.01, 1.0))
            nulls.append(auroc(o, yp))
        nulls = np.array(nulls); p95n = float(np.percentile(nulls, 95))
        print("     null NIDIFICATO (%d permutazioni, stessa procedura): media %.4f, p95 %.4f" % (nperm, nulls.mean(), p95n))
        print("     esito: %s" % ("SOPRAVVIVE" if a36 >= p95n else "MORTA"))
        # curva per layer, solo informativa (CV semplice, NON e' la statistica)
        rng2 = np.random.RandomState(args.seed)
        idx1 = rng2.permutation(np.where(yy == 1)[0]); idx0 = rng2.permutation(np.where(yy == 0)[0])
        folds = [np.concatenate([idx1[i::5], idx0[i::5]]) for i in range(5)]
        per_layer = []
        for L in range(1, X.shape[1]):
            s = np.zeros(len(yy))
            for i in range(5):
                te = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
                s[te] = _fit_logreg(X[tr, L, :], yy[tr], 0.1)(X[te, L, :])
            per_layer.append(auroc(s, yy))
        print("     (info) AUROC per layer, CV semplice a C fisso: max %.3f al layer %d, mediana %.3f  <- include selezione, NON e' il risultato"
              % (max(per_layer), 1 + int(np.argmax(per_layer)), float(np.median(per_layer))))
        print()
    print("=" * 76)


if __name__ == "__main__":
    main()
