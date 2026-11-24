"""Controlli per la v2.4 del report chiesti dalla terza verifica avversaria (23/9/2026).

1. Coppie DeepSeek per famiglia peggiore (accuratezza perturbata minima fra le righe per famiglia),
   dichiarate e derivate.
2. Meccanismo dell'aggregazione: augmented-sample-math-agg contro il file non aggregato.
3. H7 contro la maggioranza per modello pre-registrata (PREREG_E4), oltre che contro il prior globale.
4. E13, regola nuova: seme contro varianti sugli stessi problemi etichettati in A, B e A'.
5. v6: le dieci righe gpt-oss su cui si tara la soglia (calibrazione_gptoss.json nel bundle).
6. Regime di campionamento delle 164 coppie DeepSeek: base e varianti, file MATH contro file AIMO.
7. Composizione dei dati della simulazione degli organizzatori.
8. Intervalli di Wilson al 95% dei punteggi citati.
9. Qwen3.5-4B col prompt nudo: campioni di base chiusi in \\boxed{} e token per problema.
10. Varianti d'esempio delle cinque famiglie e pool few-shot.

Uso (dalla cartella AIMO): python vm/controlli_verifica4.py
"""
import json
import os
import sys
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vm"))
sys.path.insert(0, BASE)
import e13b_regola_nuova as E  # noqa: E402
import esperimento_e4 as E4  # noqa: E402
import simulazione_affidabilita_organizzatori as S  # noqa: E402
import warmup_ricalcoli as W  # noqa: E402

D = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati")
FG = os.path.join(BASE, "gpu", "out", "feature_greedy.parquet")
FILE6 = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math", "augmented-sample-math-full-filtered"]
DS = ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
FAM5 = ("rephrase", "rename", "domain", "distract", "typos")


def lista(pt):
    return isinstance(pt, list) or (isinstance(pt, str) and pt.startswith("["))


def carica_ds():
    R = {f: json.load(open(os.path.join(D, f + ".json"), encoding="utf-8")) for f in FILE6}
    ids = set(pd.read_parquet(FG).problem_id)
    righe = defaultdict(list)
    for f in FILE6:
        for r in R[f]:
            if r["model_id"] in DS and r["problem_id"] in ids:
                righe[r["problem_id"]].append((f, r))
    return R, ids, righe


def sez1(R, ids, righe):
    print("1. COPPIE DEEPSEEK PER FAMIGLIA PEGGIORE (righe per famiglia, non a lista)")
    dich = {}
    for p, rr in righe.items():
        et = [r["model_is_robust"] for f, r in rr if r.get("model_is_robust") is not None]
        if et:
            dich[p] = all(x in (True, "True") for x in et)

    def classe(p):
        pa = [float(r["permuted_accuracy"]) for f, r in righe[p] if not lista(r["permutation_type"])]
        if not pa:
            return "solo righe a lista"
        m = min(pa)
        return "1" if m == 1 else ("(0,5; 1)" if m > 0.5 else "<= 0,5")

    def fonte(p):
        return "MATH" if any(f.startswith("augmented") for f, r in righe[p]) else "AIMO"

    cd = Counter((classe(p), "robust" if dich[p] else "non robust") for p in dich)
    print("   dichiarate (%d): %s" % (len(dich), dict(sorted(cd.items()))))
    mid_dich = [p for p in dich if any(0.5 < float(r["permuted_accuracy"]) < 1 for f, r in righe[p] if not lista(r["permutation_type"]))]
    print("   dichiarate con almeno una famiglia in (0,5; 1): %d (non robuste %d; file di quelle famiglie %s)" % (
        len(mid_dich), sum(not dich[p] for p in mid_dich),
        dict(Counter(f for p in mid_dich for f, r in righe[p] if not lista(r["permutation_type"]) and 0.5 < float(r["permuted_accuracy"]) < 1))))
    der = sorted(ids - set(dich))
    cder = Counter((classe(p), fonte(p)) for p in der)
    print("   derivate (%d): %s" % (len(der), dict(sorted(cder.items()))))
    lab_mid = sum(1 for f in FILE6 for r in R[f] if r.get("model_is_robust") is not None and 0.5 < float(r["permuted_accuracy"]) < 1)
    print("   righe etichettate (tutti i modelli) con accuratezza perturbata in (0,5; 1): %d" % lab_mid)


def sez2(R):
    print("\n2. AGGREGAZIONE: augmented-sample-math-agg CONTRO augmented-sample-math-full-filtered")
    A, F = defaultdict(list), defaultdict(list)
    for r in R["augmented-sample-math-agg"]:
        A[r["problem_id"]].append(r)
    for r in R["augmented-sample-math-full-filtered"]:
        F[r["problem_id"]].append(r)
    n = tenuto_min = scartati = 0
    for p in A:
        if p not in F or any(r["model_is_robust"] in (True, "True") for r in A[p]):
            continue
        n += 1
        tenuti = {r["permutation_type"] for r in A[p]}
        mn = min(float(r["permuted_accuracy"]) for r in F[p])
        peggiori = {r["permutation_type"] for r in F[p] if float(r["permuted_accuracy"]) == mn}
        tenuto_min += tenuti == peggiori
        altri_dannosi = {r["permutation_type"] for r in F[p] if int(r["n_detrimental_permutations"]) > 0} - tenuti
        scartati += bool(altri_dannosi)
    print("   coppie non robuste dell'aggregato presenti anche nel file non aggregato: %d | tipo tenuto = tipo (o tipi) ad accuratezza minima: %d | con almeno un altro tipo dannoso scartato: %d" % (n, tenuto_min, scartati))


def sez3():
    print("\n3. H7 CONTRO LA MAGGIORANZA PER MODELLO (baseline di PREREG_E4) E CONTRO IL PRIOR GLOBALE (script originale)")
    for alias in (False, True):
        righe = W.righe_grezze(alias)
        g_mod = g_glob = g_maj = n = 0
        for held in sorted({r["model"] for r in righe}):
            tr = [r for r in righe if r["model"] != held]
            te = [r for r in righe if r["model"] == held]
            if len(te) < 5:
                continue
            t, _ = E4.soglia_migliore([(r["ba"], r["robust"]) for r in tr])
            prior = sum(r["robust"] for r in tr) / len(tr) > 0.5
            maj = sum(r["robust"] for r in te) > len(te) / 2
            for r in te:
                g_mod += ((r["ba"] > t) == r["robust"]); g_glob += (prior == r["robust"]); g_maj += (maj == r["robust"]); n += 1
        print("   %-24s n=%d | H7 %.4f | prior globale %.4f (%+.1f punti) | maggioranza per modello %.4f (%+.1f punti)" % (
            "alias unito" if alias else "335 coppie pubblicate", n, g_mod / n, g_glob / n, 100 * (g_mod - g_glob) / n, g_maj / n, 100 * (g_mod - g_maj) / n))


def sez4():
    print("\n4. E13, REGOLA NUOVA: SEME CONTRO VARIANTI SUGLI STESSI PROBLEMI (A, B, A')")
    serie = E.carica()
    for mod in ("Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think"):
        lab = {c: E.max_drop_nostro(serie[(mod, c)])[["problem_id", "label"]].dropna().set_index("problem_id").label for c in ("A", "B", "Ap")}
        ab = sorted(set(lab["A"].index) & set(lab["B"].index))
        ap = sorted(set(lab["A"].index) & set(lab["Ap"].index))
        com = sorted(set(ab) & set(ap))
        k = lambda x, y, s: E.kappa(lab[x][s].astype(int), lab[y][s].astype(int))  # noqa: E731
        d = lambda x, y, s: int((lab[x][s] != lab[y][s]).sum())  # noqa: E731
        print("   %-22s insiemi propri: kappa(A,B) %.3f (n=%d, %d disaccordi), kappa(A,A') %.3f (n=%d, %d disaccordi) | stessi %d problemi: kappa(A,B) %.3f (%d disaccordi), kappa(A,A') %.3f (%d disaccordi), quota del seme %.2f" % (
            mod.split("/")[-1], k("A", "B", ab), len(ab), d("A", "B", ab), k("A", "Ap", ap), len(ap), d("A", "Ap", ap),
            len(com), k("A", "B", com), d("A", "B", com), k("A", "Ap", com), d("A", "Ap", com),
            (1 - k("A", "Ap", com)) / (1 - k("A", "B", com))))


def sez5():
    print("\n5. v6: RIGHE GPT-OSS DELLA TARATURA (sonde/solution-v6-main.zip, calibrazione_gptoss.json)")
    z = zipfile.ZipFile(os.path.join(BASE, "sonde", "solution-v6-main.zip"))
    cal = json.loads(z.read("calibrazione_gptoss.json"))
    print("   righe %d | per identificativo %s | problemi distinti %d | robust %s" % (
        len(cal), dict(Counter(r["model_id"] for r in cal)), len({r["problema"] for r in cal}),
        [(r["model_id"], len(r["problema"])) for r in cal if r["robust"]]))
    per_testo = defaultdict(list)
    for r in cal:
        per_testo[r["problema"]].append((r["model_id"], r["robust"]))
    print("   etichette sul problema dell'unica riga robust: %s" % [v for v in per_testo.values() if any(x[1] for x in v)])


def sez6(R, ids, righe):
    print("\n6. REGIME DI CAMPIONAMENTO DELLE 164 COPPIE DEEPSEEK")
    math = [p for p in ids if any(f.startswith("augmented") for f, r in righe[p])]
    aimo = [p for p in ids if p not in set(math)]
    base = Counter(max(int(r["n_base_predictions"]) for f, r in righe[p] if f.startswith("augmented")) for p in math)
    tutte = sorted({int(r["n_base_predictions"]) for p in math for f, r in righe[p] if f.startswith("augmented")})
    print("   problemi MATH %d: predizioni sulla base per problema (massimo fra le righe) %s | valori da %d a %d | problemi con base 10: %d, con base 1: %d" % (
        len(math), sorted(base.items()), tutte[0], tutte[-1], base[10], base[1]))
    ab = sorted(max(int(r["n_base_predictions"]) for f, r in righe[p]) for p in aimo)
    npp = Counter(int(r["n_permuted_predictions"]) for p in aimo for f, r in righe[p] if f == "aimo-interp-challenge-sample-full")
    var = Counter()
    for p in aimo:
        for f, r in righe[p]:
            if f != "aimo-interp-challenge-sample-full":
                continue
            try:
                lst = json.loads(r.get("permutations_causing_decay") or "[]")
            except Exception:
                lst = []
            for v in lst if isinstance(lst, list) else []:
                if isinstance(v, dict) and v.get("n_predictions") is not None:
                    var[int(float(v["n_predictions"]))] += 1
    print("   problemi AIMO %d: predizioni sulla base %s | predizioni perturbate per riga di famiglia da %d a %d (%s) | varianti elencate per numero di risposte %s" % (
        len(aimo), ab, min(npp), max(npp), dict(sorted(npp.items())), dict(var)))


def sez7():
    print("\n7. DATI DELLA SIMULAZIONE DEGLI ORGANIZZATORI")
    S.FILE = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-full-filtered"]
    righe, coppie = S.carica()
    per_mod = Counter(k[0] for k in coppie)
    ds = [k for k in coppie if k[0] in DS]
    ds_math = sum(1 for k in ds if any(str(r.get("dataset_id", "")).startswith("hendrycks") for r in coppie[k]))
    esperte = Counter(r["permutation_type"] for r in righe if str(r["permutation_type"]).startswith("expert"))
    lista_esperte = sum(1 for r in righe if lista(r["permutation_type"]) and "expert" in str(r["permutation_type"]))
    print("   coppie %d: DeepSeek %d (MATH %d, AIMO %d), altri identificativi %s" % (
        len(coppie), len(ds), ds_math, len(ds) - ds_math, {m: n for m, n in per_mod.items() if m not in DS}))
    print("   righe delle famiglie esperte: %s, piu' %d righe a lista che le includono | righe con 50 predizioni perturbate: %d (di DeepSeek %d)" % (
        dict(esperte), lista_esperte, sum(1 for r in righe if int(r["n_permuted_predictions"]) == 50),
        sum(1 for r in righe if int(r["n_permuted_predictions"]) == 50 and r["model_id"] in DS)))


def wilson(k, n, z=1.959964):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def sez8():
    print("\n8. INTERVALLI DI WILSON AL 95%")
    for k, n in ((11, 21), (10, 19), (10, 21), (7, 14), (46, 71)):
        lo, hi = wilson(k, n)
        print("   %d/%d = %.3f: da %.3f a %.3f" % (k, n, k / n, lo, hi))


def sez9():
    print("\n9. QWEN3.5-4B COL PROMPT NUDO (esecuzioni scartate)")
    for c in ("A", "B"):
        df = pd.read_parquet(os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "e13", "scartati_prompt_nudo", "etich_Qwen__Qwen3.5-4B_%s.parquet" % c))
        b = df[df.idx == -1]
        print("   serie %s: problemi %d, campioni di base %d, con risposta estraibile (\\boxed{}) %d | token per problema: mediana %.0f, media %.0f" % (
            c, b.problem_id.nunique(), int(b.n.sum()), int(b.n_con_risposta.sum()), b.tok_medio.median(), b.tok_medio.mean()))


def sez10(R):
    print("\n10. VARIANTI D'ESEMPIO DELLE FAMIGLIE NEI DUE FILE PER FAMIGLIA, E POOL FEW-SHOT")
    rec = set()
    for f in ("aimo-interp-challenge-sample-full", "augmented-sample-math-full-filtered"):
        for r in R[f]:
            try:
                lst = json.loads(r.get("permutations_causing_decay") or "[]")
            except Exception:
                continue
            for v in lst if isinstance(lst, list) else []:
                if isinstance(v, dict) and v.get("question"):
                    rec.add((r["problem_id"], v.get("permutation_type") or r["permutation_type"], v["question"]))
    fam = Counter(t for _, t, _ in rec)
    print("   record distinti (problema, famiglia, testo): %d | per famiglia %s | nelle cinque famiglie %d" % (
        len(rec), dict(fam.most_common()), sum(v for k, v in fam.items() if k in FAM5)))
    pool = json.load(open(os.path.join(BASE, "gpu", "e12_fewshot.json"), encoding="utf-8"))
    print("   pool few-shot (gpu/e12_fewshot.json): %s, totale %d" % ({k: len(v) for k, v in pool.items()}, sum(len(v) for v in pool.values())))


def main():
    R, ids, righe = carica_ds()
    sez1(R, ids, righe)
    sez2(R)
    sez3()
    sez4()
    sez5()
    sez6(R, ids, righe)
    sez7()
    sez8()
    sez9()
    sez10(R)


if __name__ == "__main__":
    main()
