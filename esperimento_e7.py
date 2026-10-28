"""E7 — le feature di incertezza VERE del modello in cache predicono i modelli ciechi?

Ipotesi e criterio di morte in PREREG_E7.md, scritto prima di questo run.
Validazione leave-one-model-out: il modello bersaglio non e' mai in addestramento.
"""
import json, math
from collections import defaultdict
import numpy as np, pandas as pd

SEED = 20260829
BASE = ["mean_logprob","min_logprob","max_logprob","std_logprob","mean_nll","ppl",
        "min_k_logprob","mean_entropy","mean_top1","mean_margin","frac_top1",
        "frac_alta","frac_bassa"]
NUOVE = ["n_token","satura_cap","ha_risposta","ent_prima","ent_dopo","ent_trend",
         "lp_prima","lp_dopo","lp_trend","top1_trend"]

df = pd.read_parquet("gpu/out/feature_greedy.parquet")
tab = json.load(open("dati/tabella_aggregata.json", encoding="utf-8"))

feat = {r.problem_id: r for r in df.itertuples()}
righe = []
for r in tab:
    m = r["model_id"]
    if m in ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
        continue
    if r["problem_id"] not in feat:
        continue
    f = feat[r["problem_id"]]
    d = {"bersaglio": m, "problem_id": r["problem_id"], "robust": bool(r["robust"])}
    for c in BASE + NUOVE:
        d[c] = getattr(f, c, np.nan)
    righe.append(d)
D = pd.DataFrame(righe)
FEATS = [c for c in BASE + NUOVE if c in D.columns and D[c].notna().any()]

print("=" * 74)
print("E7 — il sensore con le feature vere")
print("=" * 74)
print("  %d coppie, %d modelli bersaglio, %d problemi, %d feature"
      % (len(D), D.bersaglio.nunique(), D.problem_id.nunique(), len(FEATS)))
print("  robust: %.1f%%" % (100 * D.robust.mean()))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def loco(oracolo=False):
    """leave-one-model-out. oracolo=True usa la soglia perfetta sul bersaglio (tetto)."""
    g = mg = n = 0
    per = {}
    for held in sorted(D.bersaglio.unique()):
        tr, te = D[D.bersaglio != held], D[D.bersaglio == held]
        if len(te) < 5 or tr.robust.nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
        clf.fit(tr[FEATS].fillna(0.0), tr.robust.astype(int))
        y = te.robust.astype(int).to_numpy()
        if oracolo:
            p = clf.predict_proba(te[FEATS].fillna(0.0))[:, 1]
            best = max(((p > t).astype(int) == y).sum()
                       for t in sorted(set(p.tolist() + [0.0, 1.0])))
            giuste = best
        else:
            giuste = (clf.predict(te[FEATS].fillna(0.0)) == y).sum()
        magg = max(y.sum(), len(y) - y.sum())
        g += giuste; mg += magg; n += len(y)
        per[held] = (giuste, magg, len(y))
    return g, mg, n, per

for nome, orac in (("H12  leave-one-model-out (utilizzabile)", False),
                   ("H13  con soglia ORACOLO sul bersaglio (tetto)", True)):
    g, mg, n, per = loco(orac)
    acc, magg = g / n, mg / n
    soglia = math.sqrt(0.25 / n)
    print("\n" + "-" * 74); print(nome); print("-" * 74)
    print("  n = %d predizioni" % n)
    print("  sensore      : %.4f" % acc)
    print("  maggioranza  : %.4f" % magg)
    print("  guadagno     : %+.4f (%.1f punti)   soglia di morte %.1f punti"
          % (acc - magg, 100 * (acc - magg), 100 * soglia))
    vivo = (acc - magg) >= soglia
    print("  >>> %s" % (("SOPRAVVIVE" if vivo else "MORTA") if not orac
                        else ("tetto positivo" if vivo else "tetto NEGATIVO")))
    if not orac:
        print("\n  per modello bersaglio:")
        for m, (gg, mm, nn) in sorted(per.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
            print("    %-44s %2d/%2d vs %2d  (%+.0f punti)"
                  % (m, gg, nn, mm, 100 * (gg - mm) / nn))
