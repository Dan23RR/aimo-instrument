"""Controlli per la v2.4 del report (23/9/2026): cifre usate dal testo che nessuno script salvato ricalcolava.

1. E12, analisi esplorativa: AUROC della frazione continua di varianti corrette contro l'etichetta ufficiale.
2. Test fra modelli (H7 in E4; H12-H15 in E7/E8): righe di test il cui problema compare anche in addestramento.
3. Simulazione degli organizzatori: quante delle 116 coppie sono robust con la regola del par. 2.1
   (n_detrimental = 0 in ogni riga) e la simulazione senza le righe di val-sample.
4. E13, soglia di base 10/10 contro 6/10: kappa(A,B), kappa(A,A') e concordanza di A' con l'etichetta ufficiale.
5. E13, quota di robust dell'estrazione A su 164 problemi e sui soli problemi definiti.
6. Classifica pubblica: istantanee del 7, 21 e 23 settembre salvate in dati/.

Uso (dalla cartella AIMO): python vm/controlli_verifica3.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vm"))
sys.path.insert(0, BASE)
import e13b_regola_nuova as E  # noqa: E402
import simulazione_affidabilita_organizzatori as S  # noqa: E402
import esperimento_e4 as E4  # noqa: E402

FG = os.path.join(BASE, "gpu", "out", "feature_greedy.parquet")
DS_NOMI = ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")


def sez1():
    print("1. E12, FRAZIONE CONTINUA DI VARIANTI CORRETTE (esplorativo)")
    d = pd.read_parquet(os.path.join(BASE, "gpu", "out", "e12_s2_per_problema.parquet"))
    d = d[d.definito]
    fr = d.n_var_corrette / d.n_acc
    print("   problemi definiti %d | AUROC contro l'etichetta ufficiale %.3f" % (len(d), roc_auc_score(d.robust.astype(int), fr)))


def sez2():
    print("\n2. TEST FRA MODELLI: PROBLEMI DEL TEST PRESENTI IN ADDESTRAMENTO")
    righe = E4.dati()
    tot = dentro = ident = 0
    per = {}
    for held in sorted({r["model"] for r in righe}):
        tr = [r for r in righe if r["model"] != held]
        te = [r for r in righe if r["model"] == held]
        if len(te) < 5:
            continue
        chiavi = defaultdict(set)
        for r in tr:
            chiavi[r["prob"]].add((r["robust"], round(r["ba"], 9)))
        a = sum(1 for r in te if r["prob"] in chiavi)
        b = sum(1 for r in te if (r["robust"], round(r["ba"], 9)) in chiavi.get(r["prob"], set()))
        tot += len(te); dentro += a; ident += b; per[held] = (a, b, len(te))
    print("   H7 (E4, un modello tenuto fuori alla volta): righe di test %d | problema presente in addestramento %d | con stessa etichetta e stessa base_accuracy %d" % (tot, dentro, ident))
    print("   DeepSeek (nome esplicito): %s" % (per.get(DS_NOMI[1]),))
    tab = json.load(open(os.path.join(BASE, "dati", "tabella_aggregata.json"), encoding="utf-8"))
    ids = set(pd.read_parquet(FG).problem_id)
    D = pd.DataFrame([{"bersaglio": r["model_id"], "problem_id": r["problem_id"]} for r in tab
                      if r["model_id"] not in DS_NOMI and r["problem_id"] in ids])
    tot = dentro = 0
    righe_per = []
    for held in sorted(D.bersaglio.unique()):
        tr, te = D[D.bersaglio != held], D[D.bersaglio == held]
        if len(te) < 5:
            continue
        a = int(te.problem_id.isin(set(tr.problem_id)).sum())
        tot += len(te); dentro += a
        righe_per.append("%s %d/%d" % (held.split("/")[-1], a, len(te)))
    print("   H12-H15 (E7/E8, un bersaglio tenuto fuori alla volta): coppie di test %d | problema presente in addestramento %d" % (tot, dentro))
    print("      per bersaglio: %s" % "; ".join(righe_per))


def sez3():
    print("\n3. SIMULAZIONE DEGLI ORGANIZZATORI")
    righe, coppie = S.carica()
    r21 = [k for k, rs in coppie.items() if all(int(r.get("n_detrimental_permutations") or 0) == 0 for r in rs)]
    base0 = [k for k in r21 if float(coppie[k][0]["base_accuracy"]) == 0.0]
    sopra = [k for k in r21 if any(float(r["permuted_accuracy"]) > float(r["base_accuracy"]) + 1e-9 for r in coppie[k])]
    tutte1 = [k for k, rs in coppie.items() if float(rs[0]["base_accuracy"]) > 0 and all(abs(float(r["permuted_accuracy"]) - 1) < 1e-9 for r in rs)]
    print("   coppie %d | robust con la regola del par. 2.1 (nessuna perturbazione dannosa in ogni riga): %d, di cui base_accuracy 0: %d, con accuratezza perturbata sopra la base: %d" % (
        len(coppie), len(r21), len(base0), len(sopra)))
    print("   coppie con base > 0 e ogni risposta perturbata corretta (la lettura simulata): %d" % len(tutte1))
    S.FILE = ["aimo-interp-challenge-sample-full", "augmented-sample-math-full-filtered"]
    righe2, coppie2 = S.carica()
    dati = S.prepara(coppie2)
    rng = np.random.RandomState(S.SEME)
    ks, tassi = [], []
    for _ in range(S.N_REPLICHE):
        a = S.simula(dati, 1e-9, rng); b = S.simula(dati, 1e-9, rng)
        ks.append(S.kappa(a, b)); tassi.append(a.mean())
    ks = np.array([k for k in ks if not np.isnan(k)])
    print("   senza val-sample: righe %d | coppie %d | robust simulato %.1f%% | kappa media %.3f, mediana %.3f, IC90 [%.3f, %.3f]" % (
        len(righe2), len(dati), 100 * np.mean(tassi), ks.mean(), np.median(ks), np.percentile(ks, 5), np.percentile(ks, 95)))


def sez4_5():
    serie = E.carica()
    uff = pd.read_parquet(FG)[["problem_id", "robust"]].rename(columns={"robust": "uff"})
    print("\n4. E13, SOGLIA DI BASE 10/10 CONTRO 6/10 (regola vecchia)")
    for mod in ("Skywork/Skywork-OR1-Math-7B", E.DEEPSEEK):
        for soglia in (6, 10):
            lab = {c: E.etichetta_vecchia(serie[(mod, c)], soglia) for c in ("A", "B", "Ap") if (mod, c) in serie}
            lab = {c: v[v.definito] for c, v in lab.items()}
            def k(c1, c2):
                m = lab[c1].merge(lab[c2], on="problem_id", suffixes=("_1", "_2"))
                return E.kappa(m.robust_1.astype(int), m.robust_2.astype(int)), len(m)
            s = "   %-36s base >= %2d/10 | kappa(A,B) %.3f (n=%d) | kappa(A,A') %.3f (n=%d)" % ((mod.split("/")[-1], soglia) + k("A", "B") + k("A", "Ap"))
            if mod == E.DEEPSEEK:
                m = lab["Ap"].merge(uff, on="problem_id")
                s += " | kappa(A', ufficiale) %.3f (n=%d)" % (E.kappa(m.robust.astype(int), m.uff.astype(int)), len(m))
            print(s)
        for c in ("A", "B", "Ap"):
            df = serie[(mod, c)]
            n = 0
            for pid, g in df.groupby("problem_id"):
                b = g[g.idx == -1]; v = g[g.idx >= 0]
                if len(b) and len(v) and bool((v.n_corrette > 0).all()) and 6 <= int(b.iloc[0].n_corrette) < 10:
                    n += 1
            print("      %s: problemi con ogni variante corretta e base fra 6 e 9 su 10: %d" % (c, n))
    print("\n5. E13, QUOTA DI ROBUST DELL'ESTRAZIONE A (regola vecchia, base >= 6/10)")
    for mod in E.MODELLI:
        v = E.etichetta_vecchia(serie[(mod, "A")], E.BASE_OK)
        print("   %-36s su 164: %.1f%% | sui %d definiti: %.1f%%" % (mod.split("/")[-1], 100 * v.robust.sum() / 164, v.definito.sum(), 100 * v[v.definito].robust.mean()))


def punteggi(d, task):
    return sorted(((s["created_when"], float(s["scores"][0]["score"]), s["id"], s["owner"]) for s in d if s["task"] == task and s["scores"]))


def sez6():
    print("\n6. CLASSIFICA PUBBLICA (istantanee in dati/)")
    d7 = json.load(open(os.path.join(BASE, "dati", "classifica_17850_2026-09-07.json"), encoding="utf-8"))["submissions"]
    for task, n, nome in ((35766, 21, "Main vecchio"), (35767, 19, "Small vecchio")):
        p = punteggi(d7, task)
        org = [x for x in p if x[3] == "stefanik12" and x[1] > 0]
        rif = round(org[0][1] * n)
        primo = next((x for x in p if round(x[1] * n) > rif), None)
        print("   %s (%d item): voce degli organizzatori %s = %d/%d | prima voce sopra: %s, %d/%d" % (
            nome, n, org[0][2], rif, n, primo[0][:16], round(primo[1] * n), n))
        per_giorno = defaultdict(int)
        for t, sc, _, _ in p:
            per_giorno[t[:10]] = max(per_giorno[t[:10]], round(sc * n))
        print("      massimo per giorno: %s" % ", ".join("%s %d" % (g[5:], v) for g, v in sorted(per_giorno.items())))
    d23 = json.load(open(os.path.join(BASE, "dati", "classifica_17850_2026-09-23.json"), encoding="utf-8"))["submissions"]
    d21 = json.load(open(os.path.join(BASE, "dati", "classifica_17850_2026-09-21.json"), encoding="utf-8"))["submissions"]
    print("   istantanea del 21/9: invio piu' recente %s" % max(s["created_when"] for s in d21)[:19])
    for task, n, nome in ((36237, 21, "Main corrente"), (36241, 14, "Small corrente")):
        p = punteggi(d23, task)
        prima = max(round(x[1] * n) for x in p if x[0] < "2026-09-21T10:00")
        sopra = [x for x in p if round(x[1] * n) > prima]
        nostri = [("%s %d/%d" % (x[2], round(x[1] * n), n)) for x in p if x[3] == "dan23rr"]
        print("   %s (%d item): massimo prima delle 10:00 UTC del 21/9 %d/%d | primi invii sopra: %s | massimo al 23/9 %d/%d | nostri: %s" % (
            nome, n, prima, n, ", ".join("%s %s %d/%d" % (x[2], x[0][:16], round(x[1] * n), n) for x in sopra[:3]),
            max(round(x[1] * n) for x in p), n, ", ".join(nostri)))


def sez7():
    print("\n7. ALTRI CONTROLLI SUI DATI PUBBLICI, SUL REGRESSORE E SULLE ESECUZIONI DI E13")
    D = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati")
    for f in ("aimo-interp-challenge-sample-full", "augmented-sample-math-full-filtered", "augmented-sample-math"):
        rows = json.load(open(os.path.join(D, f + ".json"), encoding="utf-8"))
        n_lista = sum(1 for r in rows if isinstance(r.get("permutation_type"), list) or str(r.get("permutation_type", "")).startswith("["))
        print("   %-38s righe %d | permutation_type a lista: %d" % (f, len(rows), n_lista))
    tab = json.load(open(os.path.join(BASE, "dati", "tabella_aggregata.json"), encoding="utf-8"))
    fonti = defaultdict(set)
    for f in ("aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
              "augmented-sample-math", "augmented-sample-math-full-filtered"):
        for r in json.load(open(os.path.join(D, f + ".json"), encoding="utf-8")):
            fonti[(r["model_id"], r["problem_id"])].add(f)
    b0 = [r for r in tab if r["robust"] and float(r["base_accuracy"]) == 0]
    print("   coppie robust con base_accuracy 0: %d | file di provenienza: %s" % (
        len(b0), dict(pd.Series(["+".join(sorted(fonti[(r["model_id"], r["problem_id"])])) for r in b0]).value_counts())))
    vs = [r for r in tab if "val-sample" in fonti[(r["model_id"], r["problem_id"])]]
    print("   coppie etichettate presenti in val-sample: %d | con base_accuracy sotto 0,05: %d" % (len(vs), sum(1 for r in vs if float(r["base_accuracy"]) < 0.05)))
    import joblib
    art = joblib.load(os.path.join(BASE, "soluzione", "uncertainty_artifacts", "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib"))
    prov = art["training_provenance"]
    print("   regressore spedito: provenienza %s, modelli %s | righe del file di origine: %d" % (
        prov.get("source_dataset"), prov.get("source_model_ids"), len(json.load(open(os.path.join(D, "augmented-sample-math-agg.json"), encoding="utf-8")))))
    scart = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "e13", "scartati_prompt_nudo", "etich_Qwen__Qwen3.5-4B_A.parquet")
    serie = E.carica()
    for nome, df in (("Qwen3.5-4B A, prompt nudo (scartata)", pd.read_parquet(scart)),
                     ("Qwen3.5-4B A, con la riga di formato", serie[("Qwen/Qwen3.5-4B", "A")]),
                     ("DeepSeek A (prompt nudo)", serie[(E.DEEPSEEK, "A")]),
                     ("DeepSeek B (prompt nudo)", serie[(E.DEEPSEEK, "B")]),
                     ("DeepSeek A' (riga di formato)", serie[(E.DEEPSEEK, "Ap")])):
        b = df[df.idx == -1]
        print("   %-38s problemi %d | base ok (almeno 6 su 10) %d | risposte estraibili sulla base, media su 10: %.2f" % (
            nome, b.problem_id.nunique(), int((b.n_corrette >= 6).sum()), b.n_con_risposta.mean()))


def sez8():
    print("\n8. CONTROLLI PUNTUALI PER IL TESTO")
    import itertools
    import pickle
    import e14_tabella as T
    runs = T.carica()
    pids = sorted(next(iter(runs.values())))
    firme = {n: tuple(runs[n][p]["sha256_testo"] for p in pids) for n in runs}
    stati = defaultdict(list)
    for n, fi in firme.items():
        stati[fi].append(n)
    for (f1, n1), (f2, n2) in itertools.combinations(stati.items(), 2):
        c = sum(1 for x, y in zip(f1, f2) if x == y)
        if c > 2:
            print("   E14: stati %s e %s condividono %d generazioni" % (n1, n2, c))
    c5 = [fi for fi, ns in stati.items() if ns[0].startswith("C5")][0]
    print("   E14: generazioni che lo stato di C5 condivide con ciascun altro stato: massimo %d" % max(
        sum(1 for x, y in zip(c5, fi) if x == y) for fi in stati if fi != c5))
    R = os.path.join(BASE, "vm_risultati")
    righe = [json.loads(l) for l in open(os.path.join(R, "e16_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.jsonl"), encoding="utf-8") if l.strip()]
    diff = [abs(r["generation_mean_prob"] - r["generation_mean_top1_prob"]) for r in righe]
    print("   E16 DeepSeek, estrazione ufficiale: |generation_mean_prob - generation_mean_top1_prob| massimo %.1e su %d problemi" % (max(diff), len(righe)))
    tab = json.load(open(os.path.join(BASE, "dati", "tabella_aggregata.json"), encoding="utf-8"))
    lun = sorted({r["problem_id"]: len(r["problema"]) for r in tab}.values())
    print("   V8: problemi distinti %d | mediana dei caratteri %.1f | valori centrali %d e %d (mediana inferiore %d)" % (
        len(lun), float(np.median(lun)), lun[len(lun) // 2 - 1], lun[len(lun) // 2], lun[len(lun) // 2 - 1]))
    art = pickle.load(open(os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "e15", "probe_artifacts", "probe_artifact.pkl"), "rb"))
    solve = open(os.path.join(BASE, "dati", "baselines_prompts_solve_2026-09-23.txt"), encoding="utf-8").read()
    print("   artefatto-sonda: system_prompt uguale a baselines/prompts/solve.txt (spazi esterni esclusi): %s" % (str(art.get("system_prompt", "")).strip() == solve.strip()))
    csv = pd.read_csv(os.path.join(BASE, "dati", "baselines_math-robust-agg_2026-09-23.csv"))
    D = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati")
    agg = json.load(open(os.path.join(D, "augmented-sample-math-agg.json"), encoding="utf-8"))
    print("   baselines/data/math-robust-agg.csv: righe %d, problemi %d | augmented-sample-math-agg: righe %d, problemi %d | stesso insieme di problemi: %s" % (
        len(csv), csv.problem_id.nunique(), len(agg), len({r["problem_id"] for r in agg}), set(csv.problem_id) == {r["problem_id"] for r in agg}))
    fonti = defaultdict(set)
    for f in ("aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
              "augmented-sample-math", "augmented-sample-math-full-filtered"):
        for r in json.load(open(os.path.join(D, f + ".json"), encoding="utf-8")):
            if r["model_id"] in DS_NOMI:
                fonti[r["problem_id"]].add("MATH" if f.startswith("augmented") else "AIMO")
    ids = set(pd.read_parquet(FG).problem_id)
    print("   le 164 coppie DeepSeek per tipo di file: %s" % dict(pd.Series(["+".join(sorted(fonti[p])) for p in ids]).value_counts()))
    import joblib
    a = joblib.load(os.path.join(BASE, "soluzione", "uncertainty_artifacts", "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib"))
    rf = a["estimator"].steps[-1][1]
    imp = pd.Series(rf.feature_importances_, index=a["feature_names"]).sort_values(ascending=False)
    print("   regressore: rango di importanza di generation_frac_selected_is_top1: %d su %d (%.3f)" % (
        list(imp.index).index("generation_frac_selected_is_top1") + 1, len(imp), imp["generation_frac_selected_is_top1"]))
    serie = E.carica()
    print("   E13, regola vecchia al variare della soglia di base (quota robust A su 164 | kappa(A,B) | kappa(A,A')):")
    for mod in E.MODELLI:
        out = []
        for soglia in range(5, 11):
            lab = {c: E.etichetta_vecchia(serie[(mod, c)], soglia) for c in ("A", "B", "Ap") if (mod, c) in serie}
            q = 100 * lab["A"].robust.sum() / 164
            lab = {c: v[v.definito] for c, v in lab.items()}
            m = lab["A"].merge(lab["B"], on="problem_id", suffixes=("_1", "_2"))
            s = "%d: %.1f%% | %.3f" % (soglia, q, E.kappa(m.robust_1.astype(int), m.robust_2.astype(int)))
            if "Ap" in lab:
                m = lab["A"].merge(lab["Ap"], on="problem_id", suffixes=("_1", "_2"))
                s += " | %.3f" % E.kappa(m.robust_1.astype(int), m.robust_2.astype(int))
            out.append(s)
        print("      %-28s %s" % (mod.split("/")[-1], " ; ".join(out)))


def main():
    sez1()
    sez2()
    sez3()
    sez4_5()
    sez6()
    sez7()
    sez8()


if __name__ == "__main__":
    main()
