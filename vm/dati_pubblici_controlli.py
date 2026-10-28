"""Controlli sui dati pubblici degli organizzatori chiesti dalla seconda verifica (23/9/2026).

1. Le 164 etichette DeepSeek: quante sono dichiarate nei file e quante derivate da noi con la
   regola di congiunzione; cosa darebbe una regola di maggioranza sulle derivate.
2. Regime di campionamento per file: predizioni sulla base, predizioni perturbate per riga di
   famiglia, varianti e predizioni per variante dove i file le elencano.
3. val-sample contro aimo-interp-challenge-sample-full: righe ripetute.
4. Difetto 2.3(a): righe etichettate nei file aggregati, robust, permutation_type a lista.
5. Coppie robust con base_accuracy 0 (robust per banalita').
6. Regola V8 (n_char >= 98) sulle coppie pubbliche: con e senza le 52 misure duplicate.

Uso (dalla cartella AIMO): python vm/dati_pubblici_controlli.py
"""
import json
import os
from collections import Counter, defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati")
FG = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "gpu", "out", "feature_greedy.parquet")
FILE6 = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math", "augmented-sample-math-full-filtered"]
DS_NOMI = ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def carica():
    return {f: json.load(open(os.path.join(D, f + ".json"), encoding="utf-8")) for f in FILE6}


def e_lista(pt):
    return isinstance(pt, list) or (isinstance(pt, str) and pt.startswith("["))


def main():
    R = carica()
    fg = pd.read_parquet(FG)
    ids = set(fg.problem_id)

    print("1. LE 164 ETICHETTE DEEPSEEK")
    righe = defaultdict(list)
    for f, rows in R.items():
        for r in rows:
            if r["model_id"] in DS_NOMI and r["problem_id"] in ids:
                righe[r["problem_id"]].append((f, r))
    dich = {}
    for p, rr in righe.items():
        et = [r["model_is_robust"] for f, r in rr if r.get("model_is_robust") is not None]
        if et:
            dich[p] = all(x in (True, "True") for x in et)
    derivate = sorted(set(ids) - set(dich))
    print("   con etichetta dichiarata %d (robust %d) | derivate con la congiunzione %d" % (len(dich), sum(dich.values()), len(derivate)))
    congiunzione = dict(zip(fg.problem_id, fg.robust.astype(bool)))
    rob_der = [p for p in derivate if congiunzione[p]]
    base0 = [p for p in rob_der if all(float(r["base_accuracy"]) == 0 for f, r in righe[p])]
    fonti = Counter("+".join(sorted({f for f, r in righe[p]})) for p in derivate)
    print("   derivate: robust per la congiunzione %d (con base_accuracy 0: %d) | fonti %s" % (len(rob_der), len(base0), dict(fonti)))
    maggioranza_fam = {}
    for p in derivate:
        pa = [float(r["permuted_accuracy"]) for f, r in righe[p] if r.get("permuted_accuracy") is not None and not e_lista(r.get("permutation_type"))]
        maggioranza_fam[p] = bool(pa) and all(x > 0.5 for x in pa)
    n_mag = sum(maggioranza_fam.values())
    tasso = (sum(dich.values()) + n_mag) / len(ids)
    print("   una regola di maggioranza per famiglia (accuratezza perturbata > 0,5 in ogni famiglia) chiama robust %d delle %d derivate; tasso DeepSeek risultante %.3f (con la congiunzione %.3f)" % (
        n_mag, len(derivate), tasso, fg.robust.mean()))
    fam_mezzo = sum(1 for p in derivate if any(0.5 < float(r["permuted_accuracy"]) < 1 for f, r in righe[p] if r.get("permuted_accuracy") is not None))
    print("   derivate con almeno una famiglia ad accuratezza perturbata fra 0,5 e 1 (esclusi): %d" % fam_mezzo)
    dich_rows = [(f, r) for f, rows in R.items() for r in rows if r.get("model_is_robust") is not None]
    ok_m = sum((float(r["permuted_accuracy"]) > 0.5) == (r["model_is_robust"] in (True, "True")) for f, r in dich_rows)
    print("   sulle %d righe con etichetta dichiarata, 'accuratezza perturbata > 0,5' concorda con l'etichetta su %d" % (len(dich_rows), ok_m))

    print("\n2. REGIME DI CAMPIONAMENTO PER FILE")
    for f, rows in R.items():
        nb = Counter(int(r["n_base_predictions"]) for r in rows)
        fam = [r for r in rows if not e_lista(r.get("permutation_type"))]
        npp = Counter(int(r["n_permuted_predictions"]) for r in fam)
        modelli = Counter(r["model_id"] for r in rows)
        print("   %-38s righe %4d | modelli %d | predizioni base (piu' frequenti) %s | predizioni perturbate per riga di famiglia (piu' frequenti) %s" % (
            f, len(rows), len(modelli), nb.most_common(3), npp.most_common(4)))
        varianti, pred_var = [], []
        for r in fam:
            try:
                lst = json.loads(r.get("permutations_causing_decay") or "[]")
            except Exception:
                continue
            if isinstance(lst, list) and lst:
                varianti.append(len(lst))
                pred_var += [int(float(x.get("n_predictions"))) for x in lst if isinstance(x, dict) and x.get("n_predictions") is not None]
        if varianti:
            print("      varianti dannose elencate per riga (piu' frequenti): %s | campo n_predictions delle varianti elencate (piu' frequenti): %s, su %d varianti" % (
                Counter(varianti).most_common(5), Counter(pred_var).most_common(5), len(pred_var)))

    print("\n3. VAL-SAMPLE CONTRO SAMPLE-FULL")
    sf = {(r["model_id"], r["problem_id"], json.dumps(r["permutation_type"]) if isinstance(r["permutation_type"], list) else r["permutation_type"]): r for r in R["aimo-interp-challenge-sample-full"]}
    coppie_sf = {(r["model_id"], r["problem_id"]) for r in R["aimo-interp-challenge-sample-full"]}
    vs = R["val-sample"]
    in_sf = sum(1 for r in vs if (r["model_id"], r["problem_id"]) in coppie_sf)
    stesse_base = sum(1 for r in vs if any(abs(float(x["base_accuracy"]) - float(r["base_accuracy"])) < 1e-9 for x in R["aimo-interp-challenge-sample-full"] if (x["model_id"], x["problem_id"]) == (r["model_id"], r["problem_id"])))
    print("   righe di val-sample %d, coppie %d | righe la cui coppia e' anche in sample-full %d | con la stessa base_accuracy %d" % (
        len(vs), len({(r["model_id"], r["problem_id"]) for r in vs}), in_sf, stesse_base))

    print("\n4. DIFETTO 2.3(a): permutation_type a lista se e solo se robust")
    for f, rows in R.items():
        et = [r for r in rows if r.get("model_is_robust") is not None]
        if not et:
            continue
        rob = [r for r in et if r["model_is_robust"] in (True, "True")]
        lista = [r for r in et if e_lista(r.get("permutation_type"))]
        iff = sum((r["model_is_robust"] in (True, "True")) == e_lista(r.get("permutation_type")) for r in et)
        print("   %-38s etichettate %3d | robust %3d | a lista %3d | 'lista sse robust' vale su %d/%d" % (f, len(et), len(rob), len(lista), iff, len(et)))

    print("\n5. COPPIE ROBUST CON BASE_ACCURACY 0 (tabella aggregata dei 6 file, 335 coppie)")
    tab = json.load(open(os.path.join(BASE, "dati", "tabella_aggregata.json"), encoding="utf-8"))
    rob = [r for r in tab if r["robust"]]
    print("   coppie %d, robust %d, robust con base_accuracy 0: %d" % (len(tab), len(rob), sum(1 for r in rob if float(r["base_accuracy"]) == 0)))

    print("\n6. REGOLA V8 (n_char >= 98, lunghi = robust) SULLE COPPIE PUBBLICHE")
    dup = {r["problem_id"] for r in tab if r["model_id"] == "qwen3-8b:low"}
    for nome, rr in (("335 coppie pubblicate", tab), ("283 coppie distinte", [r for r in tab if not (r["model_id"] == DS_NOMI[1] and r["problem_id"] in dup)])):
        for s, verso in ((98, 1), (98, -1)):
            tp = sum(1 for r in rr if r["robust"] and (len(r["problema"]) >= s) == (verso == 1))
            fn = sum(1 for r in rr if r["robust"]) - tp
            tn = sum(1 for r in rr if not r["robust"] and (len(r["problema"]) >= s) != (verso == 1))
            fp = sum(1 for r in rr if not r["robust"]) - tn
            print("   %-22s soglia %d, verso %s: sensibilita' %.3f, specificita' %.3f, accuratezza bilanciata %.5f" % (
                nome, s, "lunghi robust" if verso == 1 else "corti robust", tp / (tp + fn), tn / (tn + fp), 0.5 * (tp / (tp + fn) + tn / (tn + fp))))


if __name__ == "__main__":
    main()
