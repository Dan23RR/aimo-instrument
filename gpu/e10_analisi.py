"""E10 stadio 1 — l'etichetta "robusto alla traduzione" concorda con quella della gara? H22, H23.

Uso: python e10_analisi.py out/e10_stadio1_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet
Incrocia con out/feature_greedy.parquet (E6: punteggi del regressore + etichetta LORO).
Protocollo fissato in PREREG_E10.md prima del run. Stesso schema di e9_analisi.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260829
QUI = Path(__file__).resolve().parent


def kappa(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fold_di(problemi):
    rng = np.random.RandomState(SEED)
    p = sorted(problemi)
    perm = rng.permutation(len(p))
    return {p[i]: (k % 5) for k, i in enumerate(perm)}


def punteggio_regressore(df):
    import joblib
    art = joblib.load(QUI.parent / "soluzione" / "uncertainty_artifacts"
                      / "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib")
    X = df.reindex(columns=list(art["feature_names"])).fillna(0.0)
    return np.asarray(art["estimator"].predict(X), float), float(art["decision_threshold"])


def main(p):
    ours = pd.read_parquet(p)
    e6 = pd.read_parquet(QUI / "out" / "feature_greedy.parquet")
    tieni = ["problem_id", "robust_trad", "ha_risposta", "n_lingue", "n_stabili"] + \
            [c for c in ours.columns if c.startswith("stab_")]
    ours = ours[tieni].rename(columns={"ha_risposta": "ha_risposta_trad"})
    m = e6.merge(ours, on="problem_id", how="inner")
    m = m[m["ha_risposta_trad"].astype(bool) & (m["n_lingue"] > 0)].reset_index(drop=True)
    print("coppie con etichetta LORO e TRADUZIONE definita: %d (su %d E6)" % (len(m), len(e6)))
    loro = m["robust"].astype(bool).to_numpy()
    nostra = m["robust_trad"].astype(bool).to_numpy()
    print("robust LORO %.1f%% | robust TRADUZIONE %.1f%%" % (100 * loro.mean(), 100 * nostra.mean()))

    acc = (loro == nostra).mean()
    magg = max(loro.mean(), 1 - loro.mean())
    se = math.sqrt(0.25 / len(m))
    k = kappa(loro, nostra)
    print("\nH22  accordo %.4f | maggioranza %.4f | %+.1f punti (SE %.1f) | kappa %.3f"
          % (acc, magg, 100 * (acc - magg), 100 * se, k))
    h22 = (acc - magg) >= se and k >= 0.30
    print("     -> %s" % ("SOPRAVVIVE" if h22 else "MORTA"))

    s, t_er = punteggio_regressore(m)
    fd = fold_di(m.problem_id)
    f = m.problem_id.map(fd).to_numpy()
    g_cal = g_er = 0
    for kf in range(5):
        tr, te = f != kf, f == kf
        xs = sorted(set(s[tr].tolist()) | {t_er})
        mids = [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[0] - 1e-6, xs[-1] + 1e-6, t_er]
        best = max(mids, key=lambda t: (((s[tr] < t) == nostra[tr]).sum(), -abs(t - t_er)))
        g_cal += ((s[te] < best) == loro[te]).sum()
        g_er += ((s[te] < t_er) == loro[te]).sum()
    n = len(m)
    diff = (g_cal - g_er) / n
    print("\nH23  soglia tarata su TRADUZIONE -> acc su LORO: %.4f | ereditata 0.576: %.4f | %+.1f punti"
          % (g_cal / n, g_er / n, 100 * diff))
    h23 = diff > 0
    print("     -> %s" % ("SOPRAVVIVE" if h23 else "MORTA"))
    print("\nVERDETTO STADIO 1: %s" % ("PROXY VALIDO, si passa allo stadio 2"
                                        if (h22 and h23) else "PROXY NON VALIDO, stadio 2 NON si fa"))
    for c in [c for c in m.columns if c.startswith("stab_")]:
        v = m[c].dropna().astype(bool)
        print("  %-10s stabile nel %.0f%% (n=%d)" % (c.replace("stab_", "lingua "), 100 * v.mean(), len(v)))
    # dettaglio: la traduzione e' informativa quando ROMPE un item che l'originale risolve
    rotti = m[~m.robust_trad.astype(bool)]
    if len(rotti):
        print("  fra gli item rotti dalla traduzione, robust LORO: %.0f%% (n=%d)"
              % (100 * rotti["robust"].astype(bool).mean(), len(rotti)))
    stabili = m[m.robust_trad.astype(bool)]
    if len(stabili):
        print("  fra gli item stabili alla traduzione, robust LORO: %.0f%% (n=%d)"
              % (100 * stabili["robust"].astype(bool).mean(), len(stabili)))


if __name__ == "__main__":
    main(sys.argv[1])
