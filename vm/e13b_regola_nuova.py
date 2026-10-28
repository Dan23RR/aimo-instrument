"""E13b: l'etichetta di E13 riletta con la regola nuova degli organizzatori.

Protocollo in PREREG_E13b.md, scritto prima di qualunque calcolo di esito.
Regola nuova (README di aimo-interp/train-main-v2, 17/9/2026):
  robust se max_drop <= 0,10; non robust se max_drop >= 0,25; nessuna etichetta in mezzo.
Nessuna GPU: usa solo le risposte gia' salvate da E13 e i dataset pubblici.

Uso (dalla cartella AIMO):
  python vm/e13b_regola_nuova.py > vm_risultati/e13b_regola_nuova_2026-09-21.txt
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
E13 = os.path.join(ARCH, "e13")
DATI = os.path.join(ARCH, "dati")
FG = os.path.join(ARCH, "gpu", "out", "feature_greedy.parquet")
TRAIN = os.path.join(BASE, "dati", "train_main_v2", "train-00000-of-00001.parquet")
VECCHIO = os.path.join(BASE, "vm_risultati", "e13_analisi_2026-09-14.txt")

FAMIGLIE = ["rephrase", "rename", "domain", "distract", "typos"]
MODELLI = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
           "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SOGLIA_ROBUST, SOGLIA_NON = 0.10, 0.25
EPS = 1e-9
N_BOOT, SEME_NUOVO, SEME_VECCHIO, BASE_OK = 2000, 20260921, 20260908, 6


# ---------- identiche a vm/e13_analisi.py ----------
def kappa(a, b):
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_ic(a, b, n_boot, seed):
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    rng = np.random.RandomState(seed)
    k = kappa(a, b)
    if len(a) < 5:
        return k, float("nan"), float("nan")
    vals = [kappa(a[i], b[i]) for i in (rng.randint(0, len(a), len(a)) for _ in range(n_boot))]
    vals = [v for v in vals if not np.isnan(v)]
    if len(vals) < 20:
        return k, float("nan"), float("nan")
    return k, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def etichetta_vecchia(df, base_ok):
    """Regola vecchia di E13: base_ok E ogni variante accettata corretta."""
    out = []
    for pid, g in df.groupby("problem_id"):
        b = g[g.idx == -1]
        if len(b) == 0:
            continue
        b = b.iloc[0]
        v = g[g.idx >= 0]
        ok = int(b.n_corrette) >= base_ok
        out.append({"problem_id": pid, "robust": bool(ok and len(v) > 0 and bool((v.n_corrette > 0).all())),
                    "definito": bool(len(v) > 0)})
    return pd.DataFrame(out)


# ---------- regola nuova ----------
def regola(md):
    if md is None or (isinstance(md, float) and np.isnan(md)):
        return np.nan
    if md <= SOGLIA_ROBUST + EPS:
        return 1.0
    if md >= SOGLIA_NON - EPS:
        return 0.0
    return np.nan


def max_drop_nostro(df, min_var=1):
    out = []
    for pid, g in df.groupby("problem_id"):
        b = g[g.idx == -1]
        if len(b) == 0:
            continue
        b = b.iloc[0]
        base_acc = float(b.n_corrette) / float(b.n)
        v = g[g.idx >= 0]
        cali = []
        for f in FAMIGLIE:
            vf = v[v.famiglia == f]
            if len(vf) >= max(1, min_var):
                cali.append(base_acc - float(vf.n_corrette.sum()) / float(vf.n.sum()))
        md = max(cali) if cali else np.nan
        out.append({"problem_id": pid, "base_acc": base_acc, "max_drop": md, "n_fam": len(cali)})
    d = pd.DataFrame(out)
    d["label"] = d.max_drop.map(regola)
    return d


def ufficiali_nuove(fg_ids, esperti):
    """max_drop ufficiale di DeepSeek: massimo di absolute_accuracy_decay sulle righe del problema.
    Primaria: righe con almeno un tipo regolare (le righe aggregate robuste coprono i loro tipi,
    decadimento <= 0). Con esperti=True entrano anche le righe dei soli tipi expert_*."""
    righe, modelli_visti = [], set()
    for f in sorted(glob.glob(os.path.join(DATI, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for x in d:
            if not isinstance(x, dict) or "absolute_accuracy_decay" not in x:
                continue
            mid = str(x.get("model_id", "")).lower()
            if not (mid == "qwen3-8b:low" or mid.startswith("deepseek-ai/deepseek-r1-0528-qwen3-8b")):
                continue
            if x.get("problem_id") not in fg_ids:
                continue
            pt = x.get("permutation_type")
            tipi = json.loads(pt) if isinstance(pt, str) and pt.startswith("[") else (pt if isinstance(pt, list) else [pt])
            if not esperti and not any(t in FAMIGLIE for t in tipi):
                continue
            modelli_visti.add(x["model_id"])
            righe.append({"problem_id": x["problem_id"], "decay": float(x["absolute_accuracy_decay"]),
                          "base_acc": float(x["base_accuracy"]), "file": os.path.basename(f)})
    d = pd.DataFrame(righe)
    g = d.groupby("problem_id").agg(max_drop=("decay", "max"), base_acc=("base_acc", "max")).reset_index()
    g["label"] = g.max_drop.map(regola)
    return g, d, sorted(modelli_visti)


def confronta(a, b, filtro_base=False, seed=SEME_NUOVO):
    m = a[["problem_id", "label", "base_acc"]].merge(b[["problem_id", "label", "base_acc"]], on="problem_id", suffixes=("_1", "_2"))
    if filtro_base:
        m = m[(m.base_acc_1 >= 0.5) & (m.base_acc_2 >= 0.5)]
    n_tot = len(m)
    m = m.dropna(subset=["label_1", "label_2"])
    k, lo, hi = kappa_ic(m.label_1.astype(int), m.label_2.astype(int), N_BOOT, seed)
    return {"n": len(m), "n_tot": n_tot, "esclusi": n_tot - len(m), "k": k, "lo": lo, "hi": hi,
            "p1": float(m.label_1.mean()) if len(m) else float("nan"), "p2": float(m.label_2.mean()) if len(m) else float("nan"),
            "accordo": float((m.label_1 == m.label_2).mean()) if len(m) else float("nan")}


def riga(nome, r):
    return "   %-40s n=%3d (esclusi %3d su %3d)  kappa %+.4f  IC95 [%+.4f, %+.4f]  accordo %.3f  robust %.1f%% / %.1f%%" % (
        nome, r["n"], r["esclusi"], r["n_tot"], r["k"], r["lo"], r["hi"], r["accordo"], 100 * r["p1"], 100 * r["p2"])


def carica():
    serie = {}
    for mod in MODELLI:
        tag = mod.replace("/", "__")
        for c in ("A", "B", "Ap"):
            f = os.path.join(E13, "etich_%s_%s.parquet" % (tag, c))
            if os.path.exists(f):
                serie[(mod, c)] = pd.read_parquet(f)
    return serie


def controllo_integrita(serie, fg):
    """Ricalcola con la regola vecchia i kappa di e13_analisi_2026-09-14.txt; si ferma se non coincidono."""
    attesi, mod = {}, None
    for line in open(VECCHIO, encoding="utf-8"):
        mm = re.match(r"### (\S+)", line)
        if mm:
            mod = mm.group(1)
            continue
        mk = re.search(r"kappa\((A|Ap|B),(B|A'|loro)\)\s.*?n=\s*(\d+)\s+kappa ([+-]\d\.\d+)", line)
        if mod and mk:
            attesi[(mod, mk.group(1), mk.group(2))] = (int(mk.group(3)), float(mk.group(4)))
    vere = fg[["problem_id", "robust"]].rename(columns={"robust": "loro"})
    ok, righe = True, []
    for (mod, c1, c2), (n_att, k_att) in sorted(attesi.items()):
        e1 = etichetta_vecchia(serie[(mod, c1)], BASE_OK)
        e1 = e1[e1.definito]
        if c2 == "loro":
            m = e1.merge(vere, on="problem_id")
            a, b = m.robust.astype(int), m.loro.astype(int)
        else:
            e2 = etichetta_vecchia(serie[(mod, "Ap" if c2 == "A'" else c2)], BASE_OK)
            e2 = e2[e2.definito]
            m = e1.merge(e2, on="problem_id", suffixes=("_1", "_2"))
            a, b = m.robust_1.astype(int), m.robust_2.astype(int)
        k = kappa(a, b)
        coincide = (len(m) == n_att) and abs(round(k, 4) - k_att) < 5e-5
        ok = ok and coincide
        righe.append("   %-40s %-14s atteso n=%d %+.4f | ricalcolato n=%d %+.4f  %s" % (
            mod, "kappa(%s,%s)" % (c1, c2), n_att, k_att, len(m), k, "OK" if coincide else "DIVERSO"))
    return ok, righe, len(attesi)


def main():
    print("=" * 100)
    print("E13b: ETICHETTA DI E13 RILETTA CON LA REGOLA NUOVA (robust se max_drop <= 0,10; non robust se >= 0,25)")
    print("Protocollo: PREREG_E13b.md, scritto prima. Nessuna GPU, stessi dati di E13.")
    print("=" * 100)
    serie = carica()
    fg = pd.read_parquet(FG)
    print("serie caricate:", ", ".join("%s %s" % (m.split("/")[1], c) for (m, c) in serie))

    ok, righe, n_att = controllo_integrita(serie, fg)
    print("\n### CONTROLLO DI INTEGRITA' (regola vecchia, base >= %d su 10): %d kappa attesi da %s" % (BASE_OK, n_att, os.path.basename(VECCHIO)))
    for r in righe:
        print(r)
    if not ok or n_att == 0:
        print("\nINTEGRITA' FALLITA: l'analisi si ferma qui, come da protocollo.")
        sys.exit(1)
    print("   tutti coincidono: si procede.")

    nuove = {k: max_drop_nostro(v) for k, v in serie.items()}
    nuove_s2 = {k: max_drop_nostro(v, min_var=5) for k, v in serie.items()}
    vecchie = {}
    for k, v in serie.items():
        e = etichetta_vecchia(v, BASE_OK)
        vecchie[k] = e[e.definito]

    print("\n### QUOTE PER SERIE (regola nuova, primaria)")
    for (mod, c), d in nuove.items():
        n = len(d)
        print("   %-40s %-3s problemi %3d | robust %5.1f%% | non robust %5.1f%% | fascia %5.1f%% | senza varianti %d | base_acc < 0,5: %5.1f%%" % (
            mod, c, n, 100 * (d.label == 1).mean(), 100 * (d.label == 0).mean(),
            100 * (d.label.isna() & d.max_drop.notna()).mean(), int(d.max_drop.isna().sum()), 100 * (d.base_acc < 0.5).mean()))

    risultati = {}
    print("\n### AFFIDABILITA' TEST-RETEST")
    for mod in MODELLI:
        print("-- %s" % mod)
        r = confronta(nuove[(mod, "A")], nuove[(mod, "B")])
        risultati[(mod, "AB")] = r
        print(riga("kappa(A,B)  NUOVA, primaria", r))
        print(riga("kappa(A,B)  nuova, S1 base_acc >= 0,5", confronta(nuove[(mod, "A")], nuove[(mod, "B")], filtro_base=True)))
        print(riga("kappa(A,B)  nuova, S2 famiglie >= 5 varianti", confronta(nuove_s2[(mod, "A")], nuove_s2[(mod, "B")])))
        mv = vecchie[(mod, "A")].merge(vecchie[(mod, "B")], on="problem_id", suffixes=("_1", "_2"))
        kv = kappa(mv.robust_1.astype(int), mv.robust_2.astype(int))
        risultati[(mod, "AB_vecchia")] = kv
        print("   %-40s n=%3d  kappa %+.4f" % ("kappa(A,B)  VECCHIA (E13)", len(mv), kv))
        if (mod, "Ap") in nuove:
            r2 = confronta(nuove[(mod, "A")], nuove[(mod, "Ap")])
            print(riga("kappa(A,A') nuova, primaria (descrittivo)", r2))

    print("\n### VALIDITA': DEEPSEEK CONTRO LE ETICHETTE UFFICIALI RILETTE CON LA REGOLA NUOVA")
    fg_ids = set(fg.problem_id)
    uff, righe_uff, modelli_visti = ufficiali_nuove(fg_ids, esperti=False)
    uff_s3, _, _ = ufficiali_nuove(fg_ids, esperti=True)
    print("   righe ufficiali usate: %d (per file: %s) | model_id: %s" % (
        len(righe_uff), dict(righe_uff.groupby("file").size()), modelli_visti))
    print("   problemi con max_drop ufficiale: %d su %d | robust %.1f%% | non robust %.1f%% | fascia %.1f%%" % (
        len(uff), len(fg_ids), 100 * (uff.label == 1).mean(), 100 * (uff.label == 0).mean(), 100 * uff.label.isna().mean()))
    cambio = fg[["problem_id", "robust"]].merge(uff[["problem_id", "label"]], on="problem_id", how="left")
    tab = pd.crosstab(cambio.robust.map({True: "vecchia robust", False: "vecchia non robust"}),
                      cambio.label.map({1.0: "nuova robust", 0.0: "nuova non robust"}).fillna("nuova in fascia"))
    print("   etichette ufficiali, vecchia contro nuova:")
    for line in tab.to_string().splitlines():
        print("      " + line)
    for c in ("A", "B", "Ap"):
        r = confronta(nuove[(DEEPSEEK, c)], uff)
        risultati[("DS", c)] = r
        print(riga("kappa(%s, ufficiale nuova) primaria" % c, r))
    print(riga("kappa(A, ufficiale nuova) S1 base >= 0,5", confronta(nuove[(DEEPSEEK, "A")], uff, filtro_base=True)))
    print(riga("kappa(A, ufficiale nuova) S2 fam >= 5", confronta(nuove_s2[(DEEPSEEK, "A")], uff)))
    print(riga("kappa(A, ufficiale nuova) S3 con esperti", confronta(nuove[(DEEPSEEK, "A")], uff_s3)))

    print("\n### DESCRITTIVO: I 5 PROBLEMI IN COMUNE CON IL TRAIN SET UFFICIALE (regola nuova da entrambe le parti)")
    norm = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    testi = {}
    for f in glob.glob(os.path.join(DATI, "*.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, list):
            for x in d:
                if isinstance(x, dict) and x.get("problem_id") in fg_ids and x.get("original_problem"):
                    testi[norm(x["original_problem"])] = x["problem_id"]
    tr = pd.read_parquet(TRAIN)
    tr["problem_id"] = tr.problem.map(lambda s: testi.get(norm(s)))
    tr = tr[tr.problem_id.notna() & tr.model_id.isin(MODELLI) & (tr.reasoning_effort == "default")]
    print("   righe del train set sui nostri problemi e modelli: %d (con etichetta: %d)" % (len(tr), int(tr.is_robust.notna().sum())))
    for c in ("A", "B"):
        acc = dis = nostra_fascia = 0
        for _, x in tr[tr.is_robust.notna()].iterrows():
            d = nuove[(x.model_id, c)]
            lab = d.loc[d.problem_id == x.problem_id, "label"]
            if len(lab) == 0 or pd.isna(lab.iloc[0]):
                nostra_fascia += 1
            elif int(lab.iloc[0]) == int(bool(x.is_robust)):
                acc += 1
            else:
                dis += 1
        print("   serie %s: concordi %d, discordi %d, nostra etichetta in fascia o assente %d" % (c, acc, dis, nostra_fascia))

    print("\n" + "=" * 100)
    print("VERDETTI (criteri di PREREG_E13b.md)")
    k_ab = {m: risultati[(m, "AB")]["k"] for m in MODELLI}
    sotto = [m for m in MODELLI if not (k_ab[m] >= 0.60)]
    print("H-E13b-1 affidabilita': kappa(A,B) nuova %s -> modelli sotto 0,60: %d -> %s" % (
        ", ".join("%s %+.3f" % (m.split("/")[1][:12], k_ab[m]) for m in MODELLI), len(sotto), "MORTA" if len(sotto) >= 2 else "VIVE"))
    r = risultati[("DS", "A")]
    vive2 = (r["k"] >= 0.40) and (r["lo"] > 0)
    print("H-E13b-2 validita': kappa(A, ufficiale nuova) %+.3f IC95 [%+.3f, %+.3f] -> %s" % (r["k"], r["lo"], r["hi"], "VIVE" if vive2 else "MORTA"))
    delta = {m: k_ab[m] - risultati[(m, "AB_vecchia")] for m in MODELLI}
    su = [m for m in MODELLI if delta[m] >= 0.15]
    print("H-E13b-3 peso della regola: nuova meno vecchia %s -> modelli con +0,15 o piu': %d -> %s" % (
        ", ".join("%s %+.3f" % (m.split("/")[1][:12], delta[m]) for m in MODELLI), len(su), "VIVE" if len(su) >= 3 else "MORTA"))
    print("=" * 100)


if __name__ == "__main__":
    main()
