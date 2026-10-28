"""Ricalcoli delle ipotesi del warmup chiesti dalla seconda verifica avversaria (23/9/2026).

Usa le funzioni degli script originali (esperimento_e1.py, esperimento_e4.py) e ripete per
controllo i numeri originali prima delle varianti. Varianti:
  A. H1 (E1): CV raggruppata per problema come nell'originale; stessa CV senza le 52 righe
     duplicate (le righe DeepSeek pubblicate anche come qwen3-8b:low); CV a 5 fold NON
     raggruppata, 3 semi, sui dati originali.
  B. H2 (E1, test originale: nullo a tasso globale, z > 2): originale e con l'alias unito.
  C. Regola T (E5, dati pubblici aggregati): originale e con l'alias unito.
  D. H7 e H8 (E4): originali e con l'alias unito.
  E. H13 (E7, oracolo con soglia scelta sul bersaglio): originale, e oracolo a cui sono vietate
     le soglie che danno una predizione costante; contributo per bersaglio.

Uso (dalla cartella AIMO): python vm/warmup_ricalcoli.py
"""
import json
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
import esperimento_e1 as E1  # noqa: E402
import esperimento_e4 as E4  # noqa: E402

ALIAS = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
DS = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
FILE6 = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math", "augmented-sample-math-full-filtered"]


# ------------------------------------------------------------------ A. H1
def h1(righe, gruppi_per_fold):
    nomi = sorted(E1.feature_testo(righe[0]["problema"]).keys())
    X = [[E1.feature_testo(r["problema"])[k] for k in nomi] for r in righe]
    y = [1 if r["robust"] else 0 for r in righe]
    n = len(righe)
    acc = magg = tot = 0
    for k in range(5):
        tr = [i for i in range(n) if gruppi_per_fold[i] != k]
        te = [i for i in range(n) if gruppi_per_fold[i] == k]
        mu, sd = E1.standardizza([X[i] for i in tr])
        Xtr = E1.applica([X[i] for i in tr], mu, sd)
        Xte = E1.applica([X[i] for i in te], mu, sd)
        ytr = [y[i] for i in tr]
        w, b = E1.addestra(Xtr, ytr)
        mg = 1 if sum(ytr) * 2 > len(ytr) else 0
        for j, i in enumerate(te):
            acc += (1 if E1.predici(w, b, Xte[j]) >= 0.5 else 0) == y[i]
            magg += mg == y[i]
            tot += 1
    return acc / tot, magg / tot, math.sqrt(0.25 / n), n


def fold_raggruppati(righe):
    unici = sorted({r["problem_id"] for r in righe})
    f = {p: i % 5 for i, p in enumerate(unici)}
    return [f[r["problem_id"]] for r in righe]


def fold_casuali(righe, seme):
    idx = list(range(len(righe)))
    random.Random(seme).shuffle(idx)
    f = [0] * len(righe)
    for pos, i in enumerate(idx):
        f[i] = pos % 5
    return f


# ------------------------------------------------------------------ B. H2 test di E1
def h2_e1(righe):
    per_prob = defaultdict(list)
    for r in righe:
        per_prob[r["problem_id"]].append(1 if r["robust"] else 0)
    multi = {k: v for k, v in per_prob.items() if len(v) >= 2}
    conc = sum(1 for v in multi.values() if len(set(v)) == 1)
    p = sum(1 for r in righe if r["robust"]) / len(righe)
    att = sum(p ** len(v) + (1 - p) ** len(v) for v in multi.values())
    se = math.sqrt(att * (1 - att / len(multi)))
    return len(multi), conc, att, (conc - att) / se


# ------------------------------------------------------------------ C/D. aggregazione con alias
def righe_grezze(alias):
    grezze = []
    for s in FILE6:
        grezze += json.load(open(os.path.join(BASE_DIR, "dati", s + ".json"), encoding="utf-8"))
    agg = defaultdict(list)
    for r in grezze:
        m = ALIAS.get(r["model_id"], r["model_id"]) if alias else r["model_id"]
        agg[(m, r["problem_id"])].append(r)
    out = []
    for (m, p), rr in agg.items():
        detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in rr)
        ba = [float(x["base_accuracy"]) for x in rr if x.get("base_accuracy") is not None]
        if ba:
            out.append({"model": m, "prob": p, "robust": detr == 0, "ba": sum(ba) / len(ba)})
    return out


def regola_t(righe):
    n = len(righe)
    g_reg = sum((r["ba"] < 0.05) == r["robust"] for r in righe)
    g_pri = sum((not r["robust"]) for r in righe)
    return n, g_reg / n, g_pri / n, math.sqrt(0.25 / n)


def h7_h8(righe):
    modelli = sorted({r["model"] for r in righe})
    g_mod = g_glob = n = 0
    for held in modelli:
        train = [r for r in righe if r["model"] != held]
        test = [r for r in righe if r["model"] == held]
        if len(test) < 5:
            continue
        t, _ = E4.soglia_migliore([(r["ba"], r["robust"]) for r in train])
        prior = sum(r["robust"] for r in train) / len(train) > 0.5
        for r in test:
            g_mod += ((r["ba"] > t) == r["robust"])
            g_glob += (prior == r["robust"])
            n += 1
    h7 = (g_mod - g_glob) / n
    g8 = n8 = 0
    g8_prior = 0
    for held in modelli:
        test = [r for r in righe if r["model"] == held]
        if len(test) < 5:
            continue
        _, a = E4.soglia_migliore([(r["ba"], r["robust"]) for r in test])
        g8 += a * len(test)
        n8 += len(test)
        g8_prior += max(sum(r["robust"] for r in test), len(test) - sum(r["robust"] for r in test))
    return n, h7, math.sqrt(0.25 / n), g8 / n8, g8_prior / n8


# ------------------------------------------------------------------ E. H13
SEED = 20260829
BASE = ["mean_logprob", "min_logprob", "max_logprob", "std_logprob", "mean_nll", "ppl",
        "min_k_logprob", "mean_entropy", "mean_top1", "mean_margin", "frac_top1", "frac_alta", "frac_bassa"]
NUOVE = ["n_token", "satura_cap", "ha_risposta", "ent_prima", "ent_dopo", "ent_trend",
         "lp_prima", "lp_dopo", "lp_trend", "top1_trend"]


def h13():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_parquet(os.path.join(BASE_DIR, "gpu", "out", "feature_greedy.parquet"))
    tab = json.load(open(os.path.join(BASE_DIR, "dati", "tabella_aggregata.json"), encoding="utf-8"))
    feat = {r.problem_id: r for r in df.itertuples()}
    righe = []
    for r in tab:
        m = r["model_id"]
        if m in ("qwen3-8b:low", DS) or r["problem_id"] not in feat:
            continue
        f = feat[r["problem_id"]]
        d = {"bersaglio": m, "problem_id": r["problem_id"], "robust": bool(r["robust"])}
        for c in BASE + NUOVE:
            d[c] = getattr(f, c, np.nan)
        righe.append(d)
    D = pd.DataFrame(righe)
    FEATS = [c for c in BASE + NUOVE if c in D.columns and D[c].notna().any()]
    ris = {"libero": [0, 0, 0], "senza_costanti": [0, 0, 0]}
    per = []
    for held in sorted(D.bersaglio.unique()):
        tr, te = D[D.bersaglio != held], D[D.bersaglio == held]
        if len(te) < 5 or tr.robust.nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
        clf.fit(tr[FEATS].fillna(0.0), tr.robust.astype(int))
        y = te.robust.astype(int).to_numpy()
        p = clf.predict_proba(te[FEATS].fillna(0.0))[:, 1]
        soglie = sorted(set(p.tolist() + [0.0, 1.0]))
        pred = [(p > t).astype(int) for t in soglie]
        libero = max((q == y).sum() for q in pred)
        nc = [q for q in pred if 0 < q.sum() < len(q)]
        senza = max((q == y).sum() for q in nc) if nc else 0
        magg = max(y.sum(), len(y) - y.sum())
        for k, v in (("libero", libero), ("senza_costanti", senza)):
            ris[k][0] += v; ris[k][1] += magg; ris[k][2] += len(y)
        per.append((held, len(y), magg, libero, senza))
    return ris, per


def main():
    righe = [r for r in json.load(open(os.path.join(BASE_DIR, "dati", "tabella_aggregata.json"), encoding="utf-8")) if r.get("problema")]
    ds_prob = {r["problem_id"] for r in righe if r["model_id"] == DS}
    q_prob = {r["problem_id"] for r in righe if r["model_id"] == "qwen3-8b:low"}
    dedup = [r for r in righe if not (r["model_id"] == DS and r["problem_id"] in q_prob)]
    print("tabella_aggregata: %d righe; righe DeepSeek duplicate di qwen3-8b:low: %d; dopo la deduplica %d" % (
        len(righe), len(ds_prob & q_prob), len(dedup)))

    print("\nA. H1 (testo del problema)")
    for nome, rr, fold in (("originale, CV raggruppata", righe, fold_raggruppati(righe)),
                           ("senza duplicati, CV raggruppata", dedup, fold_raggruppati(dedup))):
        a, m, s, n = h1(rr, fold)
        print("   %-34s n=%d | modello %.4f | maggioranza %.4f | guadagno %+.1f punti | soglia %.1f" % (nome, n, a, m, 100 * (a - m), 100 * s))
    for seme in (1, 2, 3):
        a, m, s, n = h1(righe, fold_casuali(righe, seme))
        print("   %-34s n=%d | modello %.4f | maggioranza %.4f | guadagno %+.1f punti | soglia %.1f" % ("originale, CV NON raggruppata, seme %d" % seme, n, a, m, 100 * (a - m), 100 * s))

    print("\nB. H2, test originale di E1 (nullo a tasso globale, vive se z > 2)")
    alias_righe = [dict(r, model_id=ALIAS.get(r["model_id"], r["model_id"])) for r in righe]
    for nome, rr in (("originale", righe), ("alias unito (una riga per coppia)", dedup)):
        k, c, att, z = h2_e1(rr)
        print("   %-34s problemi multi-modello %d | concordi %d | attesi %.1f | z = %.2f" % (nome, k, c, att, z))

    print("\nC. Regola T su tutti i dati pubblici aggregati")
    for nome, al in (("originale", False), ("alias unito", True)):
        n, a, p, s = regola_t(righe_grezze(al))
        print("   %-34s n=%d | regola %.4f | costante %.4f | guadagno %+.1f punti | soglia %.1f" % (nome, n, a, p, 100 * (a - p), 100 * s))

    print("\nD. H7 e H8 (base accuracy fra modelli)")
    for nome, al in (("originale", False), ("alias unito", True)):
        n, h7, s, h8, pri = h7_h8(righe_grezze(al))
        print("   %-34s n=%d | H7 guadagno %+.1f punti (soglia %.1f) | H8 tetto %.4f contro maggioranza per modello %.4f (%+.1f)" % (
            nome, n, 100 * h7, 100 * s, h8, pri, 100 * (h8 - pri)))

    print("\nE. H13 (oracolo con soglia scelta sugli item del bersaglio)")
    ris, per = h13()
    for k, (g, mg, n) in ris.items():
        print("   %-16s %d/%d contro maggioranza %d/%d | guadagno %+.1f punti (%+d item) | soglia %.1f" % (
            k, g, n, mg, n, 100 * (g - mg) / n, g - mg, 100 * math.sqrt(0.25 / n)))
    for held, n, magg, lib, senza in per:
        print("      %-44s n=%2d magg %2d | libero %2d (%+d) | senza costanti %2d (%+d)" % (held, n, magg, lib, lib - magg, senza, senza - magg))


if __name__ == "__main__":
    main()
