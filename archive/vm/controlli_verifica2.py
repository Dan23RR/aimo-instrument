"""Controlli chiesti dalla seconda verifica avversaria del report (23/9/2026).

Nessuna GPU: solo le risposte gia' salvate da E13 e i dataset pubblici. Tre controlli.

1. Simulazione plug-in sulle NOSTRE etichette (regola vecchia di E13, estrazione A): stessa
   procedura usata per gli organizzatori (vm/simulazione_affidabilita_organizzatori.py). Per ogni
   problema: base ~ Binomiale(10, quota osservata); per ogni famiglia ~ Binomiale(varianti della
   famiglia, quota osservata di varianti corrette); robust se base >= 6/10 e tutte corrette.
   Si confronta il kappa simulato con quelli misurati: kappa(A,A') (stesse varianti, seme nuovo)
   e kappa(A,B) (varianti nuove). Serve a capire quale dei due la simulazione approssima.
2. Regola nuova contro regola vecchia sugli STESSI problemi: kappa(A,B) con la regola vecchia
   ristretto ai problemi etichettati in entrambe le serie con la regola nuova.
3. DeepSeek, nostre etichette nuove contro ufficiali nuove: quota di robust nostra secondo
   l'etichetta ufficiale, e correlazione di Spearman fra i due max_drop.

Uso (dalla cartella AIMO): python vm/controlli_verifica2.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e13b_regola_nuova as E  # noqa: E402

N_REPLICHE, SEME = 2000, 20260923


def struttura_famiglie(df):
    out = {}
    for pid, g in df.groupby("problem_id"):
        b = g[g.idx == -1]
        v = g[g.idx >= 0]
        if len(b) == 0 or len(v) == 0:
            continue
        b = b.iloc[0]
        fam = [(int(len(vf)), float((vf.n_corrette > 0).mean())) for _, vf in v.groupby("famiglia")]
        out[pid] = (int(b.n), float(b.n_corrette) / float(b.n), fam)
    return out


def simula(s, rng):
    lab = []
    for pid in sorted(s):
        nb, pb, fam = s[pid]
        ok = rng.binomial(nb, pb) >= E.BASE_OK
        if ok:
            for n, p in fam:
                if rng.binomial(n, p) < n:
                    ok = False
                    break
        lab.append(ok)
    return np.array(lab)


def main():
    serie = E.carica()
    print("=" * 96)
    print("1. SIMULAZIONE PLUG-IN SULLE NOSTRE ETICHETTE (regola vecchia, estrazione A, %d coppie di repliche)" % N_REPLICHE)
    for mod in E.MODELLI:
        s = struttura_famiglie(serie[(mod, "A")])
        rng = np.random.RandomState(SEME)
        ks = []
        for _ in range(N_REPLICHE):
            k = E.kappa(simula(s, rng), simula(s, rng))
            if not np.isnan(k):
                ks.append(k)
        va = E.etichetta_vecchia(serie[(mod, "A")], E.BASE_OK); va = va[va.definito]
        vb = E.etichetta_vecchia(serie[(mod, "B")], E.BASE_OK); vb = vb[vb.definito]
        m = va.merge(vb, on="problem_id", suffixes=("_1", "_2"))
        kab = E.kappa(m.robust_1.astype(int), m.robust_2.astype(int))
        kap = float("nan")
        if (mod, "Ap") in serie:
            vp = E.etichetta_vecchia(serie[(mod, "Ap")], E.BASE_OK); vp = vp[vp.definito]
            m2 = va.merge(vp, on="problem_id", suffixes=("_1", "_2"))
            kap = E.kappa(m2.robust_1.astype(int), m2.robust_2.astype(int))
        print("   %-40s simulato media %.3f (IC90 %.3f-%.3f) | misurato kappa(A,A') %s | kappa(A,B) %.3f" % (
            mod, np.mean(ks), np.percentile(ks, 5), np.percentile(ks, 95),
            ("%.3f" % kap) if not np.isnan(kap) else "non c'e'", kab))

    print("=" * 96)
    print("2. REGOLA NUOVA CONTRO REGOLA VECCHIA SUGLI STESSI PROBLEMI (kappa(A,B))")
    for mod in E.MODELLI:
        na, nb = E.max_drop_nostro(serie[(mod, "A")]), E.max_drop_nostro(serie[(mod, "B")])
        mn = na[["problem_id", "label"]].merge(nb[["problem_id", "label"]], on="problem_id", suffixes=("_1", "_2")).dropna()
        stessi = set(mn.problem_id)
        va = E.etichetta_vecchia(serie[(mod, "A")], E.BASE_OK); va = va[va.definito]
        vb = E.etichetta_vecchia(serie[(mod, "B")], E.BASE_OK); vb = vb[vb.definito]
        mv = va.merge(vb, on="problem_id", suffixes=("_1", "_2"))
        mvs = mv[mv.problem_id.isin(stessi)]
        k_nuova = E.kappa(mn.label_1.astype(int), mn.label_2.astype(int))
        k_vecchia_tutti = E.kappa(mv.robust_1.astype(int), mv.robust_2.astype(int))
        k_vecchia_stessi = E.kappa(mvs.robust_1.astype(int), mvs.robust_2.astype(int))
        print("   %-40s n=%3d | nuova %.3f | vecchia sugli stessi %.3f (guadagno %+.3f) | vecchia su tutti %.3f (n=%d)" % (
            mod, len(mn), k_nuova, k_vecchia_stessi, k_nuova - k_vecchia_stessi, k_vecchia_tutti, len(mv)))

    print("=" * 96)
    print("3. DEEPSEEK: NOSTRE ETICHETTE NUOVE CONTRO UFFICIALI NUOVE")
    fg = pd.read_parquet(E.FG)
    uff, _, _ = E.ufficiali_nuove(set(fg.problem_id), esperti=False)
    for c in ("A", "B"):
        n = E.max_drop_nostro(serie[(E.DEEPSEEK, c)])
        m = n[["problem_id", "label", "max_drop"]].merge(uff[["problem_id", "label", "max_drop"]], on="problem_id", suffixes=("_n", "_u"))
        mm = m.dropna(subset=["label_n", "label_u"])
        r1 = mm[mm.label_u == 1].label_n.mean(); r0 = mm[mm.label_u == 0].label_n.mean()
        md = m.dropna(subset=["max_drop_n", "max_drop_u"])
        rho = spearmanr(md.max_drop_n, md.max_drop_u).correlation
        print("   serie %s: n=%d | nostra quota robust dove l'ufficiale e' robust %.3f (n=%d), dove non lo e' %.3f (n=%d) | Spearman max_drop %+.3f (n=%d)" % (
            c, len(mm), r1, int((mm.label_u == 1).sum()), r0, int((mm.label_u == 0).sum()), rho, len(md)))


if __name__ == "__main__":
    main()
