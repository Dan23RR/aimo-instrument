"""E9 stadio 1 — il proxy (etichetta con perturbazioni nostre) vale? H19 e H20.

Uso: python e9_analisi.py out/e9_stadio1_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet
Incrocia con out/feature_greedy.parquet (E6: punteggi del regressore + etichetta LORO).
Protocollo fissato in PREREG_E9.md prima del run.
"""
import sys, math
import numpy as np, pandas as pd
from pathlib import Path

SEED = 20260829
BASE = ["mean_logprob","min_logprob","max_logprob","std_logprob","mean_nll","ppl","min_k_logprob",
        "mean_entropy","mean_top1","mean_margin","frac_top1","frac_alta","frac_bassa"]
QUI = Path(__file__).resolve().parent


def kappa(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def fold_di(problemi):
    rng = np.random.RandomState(SEED); p = sorted(problemi); perm = rng.permutation(len(p))
    return {p[i]: (k % 5) for k, i in enumerate(perm)}


def punteggio_regressore(df):
    """Ricostruisce lo score del regressore ufficiale dalle feature E6 (stesso artefatto v6)."""
    import joblib, zipfile, io, os
    art = QUI.parent / "soluzione" / "uncertainty_artifacts" / "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib"
    a = joblib.load(art)
    est = a["estimator"] if isinstance(a, dict) else getattr(a, "estimator", a)
    names = list(a["feature_names"]) if isinstance(a, dict) else list(getattr(a, "feature_names", BASE))
    X = df.reindex(columns=names).fillna(0.0)
    return np.asarray(est.predict(X), float), float(a["decision_threshold"] if isinstance(a, dict) else a.decision_threshold)


def main(p):
    ours = pd.read_parquet(p)
    e6 = pd.read_parquet(QUI / "out" / "feature_greedy.parquet")
    # E6 ha gia' una colonna ha_risposta: prendo da E9 solo cio' che serve, rinominato
    tieni = ["problem_id", "robust_ours", "ha_risposta"] + [c for c in ours.columns if c.startswith("stab_")]
    ours = ours[tieni].rename(columns={"ha_risposta": "ha_risposta_ours"})
    m = e6.merge(ours, on="problem_id", how="inner")
    m = m[m["ha_risposta_ours"].astype(bool)].reset_index(drop=True)   # etichetta nostra definita
    print("coppie con etichetta LORO e NOSTRA definita: %d (su %d E6)" % (len(m), len(e6)))
    loro = m["robust"].astype(bool).to_numpy(); nostra = m["robust_ours"].astype(bool).to_numpy()
    print("robust LORO %.1f%% | robust NOSTRA %.1f%%" % (100 * loro.mean(), 100 * nostra.mean()))

    # H19
    acc = (loro == nostra).mean(); magg = max(loro.mean(), 1 - loro.mean()); se = math.sqrt(0.25 / len(m))
    k = kappa(loro, nostra)
    print("\nH19  accordo %.4f | maggioranza %.4f | +%.1f punti (SE %.1f) | kappa %.3f"
          % (acc, magg, 100 * (acc - magg), 100 * se, k))
    h19 = (acc - magg) >= se and k >= 0.30
    print("     -> %s" % ("SOPRAVVIVE" if h19 else "MORTA"))

    # H20: soglia calibrata su NOSTRA (train) -> accuratezza su LORO (test), CV per problema
    s, t_eredit = punteggio_regressore(m)
    fd = fold_di(m.problem_id); f = m.problem_id.map(fd).to_numpy()
    g_cal = g_er = 0
    for kf in range(5):
        tr, te = f != kf, f == kf
        cands = sorted(set(s[tr].tolist()) | {t_eredit})
        mids = [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [cands[0] - 1e-6, cands[-1] + 1e-6, t_eredit]
        best = max(mids, key=lambda t: (((s[tr] < t) == nostra[tr]).sum(), -abs(t - t_eredit)))
        g_cal += ((s[te] < best) == loro[te]).sum(); g_er += ((s[te] < t_eredit) == loro[te]).sum()
    n = len(m); diff = (g_cal - g_er) / n
    print("\nH20  soglia calibrata su NOSTRA -> acc su LORO: %.4f | soglia ereditata 0.576: %.4f | %+.1f punti"
          % (g_cal / n, g_er / n, 100 * diff))
    h20 = diff > 0
    print("     -> %s" % ("SOPRAVVIVE" if h20 else "MORTA"))
    print("\nVERDETTO STADIO 1: %s" % ("PROXY VALIDO, si passa allo stadio 2" if (h19 and h20) else "PROXY NON VALIDO, stadio 2 NON si fa"))
    for fam in ["format", "latex_delims", "framing", "redundancy", "question_first"]:
        c = "stab_" + fam
        if c in m: print("  famiglia %-14s stabile nel %.0f%% (n=%d)" % (fam, 100 * m[c].dropna().astype(bool).mean(), m[c].notna().sum()))


if __name__ == "__main__":
    main(sys.argv[1])
