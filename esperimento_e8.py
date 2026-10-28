"""E8 — una calibrazione realizzabile cattura il tetto di E7?

Ipotesi e criteri di morte in PREREG_E8.md, scritti prima di questo run.
Controllo di onesta' obbligatorio: risultati CON e SENZA il caso avverso (Llama).
"""
import json, math
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260829
BASE = ["mean_logprob","min_logprob","max_logprob","std_logprob","mean_nll","ppl",
        "min_k_logprob","mean_entropy","mean_top1","mean_margin","frac_top1",
        "frac_alta","frac_bassa"]
NUOVE = ["n_token","satura_cap","ha_risposta","ent_prima","ent_dopo","ent_trend",
         "lp_prima","lp_dopo","lp_trend","top1_trend"]

df = pd.read_parquet("gpu/out/feature_greedy.parquet")
tab = json.load(open("dati/tabella_aggregata.json", encoding="utf-8"))
prior_pub = json.load(open("dati/prior_per_modello.json", encoding="utf-8"))
PRIOR_GLOBALE = 0.239

feat = {r.problem_id: r for r in df.itertuples()}
righe = []
for r in tab:
    m = r["model_id"]
    if m in ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"): continue
    if r["problem_id"] not in feat: continue
    f = feat[r["problem_id"]]
    d = {"bersaglio": m, "problem_id": r["problem_id"], "robust": bool(r["robust"])}
    for c in BASE + NUOVE: d[c] = getattr(f, c, np.nan)
    righe.append(d)
D = pd.DataFrame(righe)
FEATS = [c for c in BASE + NUOVE if c in D.columns and D[c].notna().any()]

def valuta(escludi_llama, modo):
    """modo: 'C1' prior pubblico del bersaglio, 'C2' prior globale, 'oracolo' tetto."""
    dati = D[D.bersaglio != "meta-llama__Llama-3.1-8B-Instruct_low"] if escludi_llama else D
    g = mg = n = 0
    for held in sorted(dati.bersaglio.unique()):
        tr, te = dati[dati.bersaglio != held], dati[dati.bersaglio == held]
        if len(te) < 5 or tr.robust.nunique() < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
        clf.fit(tr[FEATS].fillna(0.0), tr.robust.astype(int))
        p = clf.predict_proba(te[FEATS].fillna(0.0))[:, 1]
        y = te.robust.astype(int).to_numpy()
        if modo == "oracolo":
            k = int(y.sum())
        else:
            tasso = prior_pub.get(held, PRIOR_GLOBALE) if modo == "C1" else PRIOR_GLOBALE
            k = int(round(tasso * len(y)))
        k = max(0, min(len(y), k))
        pred = np.zeros(len(y), dtype=int)
        if k > 0: pred[np.argsort(-p)[:k]] = 1
        g += (pred == y).sum(); mg += max(y.sum(), len(y) - y.sum()); n += len(y)
    return g / n, mg / n, n

print("=" * 74); print("E8 — la calibrazione realizzabile"); print("=" * 74)
for escludi in (False, True):
    et = "SENZA Llama (controllo)" if escludi else "TUTTI i modelli (quello che conta)"
    print("\n" + "-" * 74); print(et); print("-" * 74)
    for modo, nome in (("C1", "H14  prior pubblico del bersaglio"),
                       ("C2", "H15  prior globale per tutti"),
                       ("oracolo", "     tetto (oracolo, non utilizzabile)")):
        acc, magg, n = valuta(escludi, modo)
        soglia = math.sqrt(0.25 / n)
        v = ""
        if modo != "oracolo":
            v = "  >>> %s" % ("SOPRAVVIVE" if (acc - magg) >= soglia else "MORTA")
        print("  %-40s %.4f vs %.4f  (%+.1f punti, soglia %.1f)%s"
              % (nome, acc, magg, 100 * (acc - magg), 100 * soglia, v))
