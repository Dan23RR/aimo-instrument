"""E13 — verdetto: affidabilità test-retest dell'etichetta di robustezza (PREREG_E13.md).

Uso: python e13_analisi.py [--dir /work/aimo/e13] [--base-ok 6]

Ricostruisce l'etichetta dal regime loro e calcola le tre statistiche primarie:
  kappa(A,B)   affidabilità totale: varianti diverse e campionamento diverso
  kappa(A,A')  quota dovuta al solo campionamento: stesse varianti, seme diverso
  kappa(A,loro) e kappa(B,loro)  solo su DeepSeek, che ha la verità a terra
con intervalli bootstrap sui problemi. Poi applica la griglia di lettura a quattro celle,
che è scritta nella pre-registrazione e non viene ritoccata qui.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="/work/aimo/e13")
ap.add_argument("--base-ok", type=int, default=6)
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=20260908)
ap.add_argument("--etichette", default="/work/aimo/gpu/out/feature_greedy.parquet")
args = ap.parse_args()
FAMIGLIE = ["rephrase", "rename", "domain", "distract", "typos"]


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
    if len(vals) < 20:   # kappa non definito (una delle due etichette e' costante): niente IC
        return k, float("nan"), float("nan")
    return k, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def etichetta(df, base_ok):
    """robust = base_ok E ogni variante accettata corretta. Restituisce anche la diagnostica."""
    out = []
    for pid, g in df.groupby("problem_id"):
        b = g[g.idx == -1]
        if len(b) == 0:
            continue
        b = b.iloc[0]
        v = g[g.idx >= 0]
        ok = int(b.n_corrette) >= base_ok
        r = {"problem_id": pid, "base_corrette": int(b.n_corrette), "base_ok": ok,
             "n_var": len(v), "n_var_corrette": int(v.n_corrette.sum()) if len(v) else 0,
             "robust": bool(ok and len(v) > 0 and bool((v.n_corrette > 0).all())),
             "definito": bool(len(v) > 0)}
        for f in FAMIGLIE:
            vf = v[v.famiglia == f]
            r["stab_" + f] = bool((vf.n_corrette > 0).all()) if len(vf) else None
        out.append(r)
    return pd.DataFrame(out)


def main():
    files = sorted(glob.glob(os.path.join(args.dir, "etich_*_[ABp]*.parquet")))
    files = [f for f in files if "parziale" not in f]
    if not files:
        print("nessun file di etichettatura in %s" % args.dir)
        return
    dati = {}
    for f in files:
        d = pd.read_parquet(f)
        dati.setdefault(d.model.iloc[0], {})[d["cond"].iloc[0]] = d
    vere = None
    if os.path.isfile(args.etichette):
        vere = pd.read_parquet(args.etichette)[["problem_id", "robust"]].rename(columns={"robust": "loro"})

    print("=" * 78)
    print("E13 — AFFIDABILITÀ TEST-RETEST DELL'ETICHETTA  (base_ok = %d/10 campioni corretti)" % args.base_ok)
    print("=" * 78)
    riga_finale = {}
    for model, conds in sorted(dati.items()):
        print()
        print("### %s   condizioni presenti: %s" % (model, ", ".join(sorted(conds))))
        et = {c: etichetta(d, args.base_ok) for c, d in conds.items()}
        for c, e in sorted(et.items()):
            print("   %-3s problemi %3d, definiti %3d | base_ok %5.1f%% | robust %5.1f%% | varianti per problema %.1f"
                  % (c, len(e), int(e.definito.sum()), 100 * e.base_ok.mean(), 100 * e.robust.mean(), e.n_var.mean()))
        res = {}
        for c1, c2, nome in (("A", "B", "kappa(A,B)  affidabilità totale"),
                             ("A", "Ap", "kappa(A,A') solo campionamento")):
            if c1 in et and c2 in et:
                m = et[c1].merge(et[c2], on="problem_id", suffixes=("_1", "_2"))
                m = m[m.definito_1 & m.definito_2]
                k, lo, hi = kappa_ic(m.robust_1.astype(int), m.robust_2.astype(int), args.n_boot, args.seed)
                acc = (m.robust_1 == m.robust_2).mean()
                nota = ""
                if m.robust_1.nunique() < 2 or m.robust_2.nunique() < 2:
                    nota = "  [ATTENZIONE: un'etichetta e' costante, kappa non informativo]"
                print("   %-34s n=%3d  kappa %+.4f  IC95 [%+.4f, %+.4f]  accordo grezzo %.3f%s"
                      % (nome, len(m), k, lo, hi, acc, nota))
                res[c1 + c2] = k
                res[c1 + c2 + "_accordo"] = float(acc)
                res[c1 + c2 + "_pos1"] = float(m.robust_1.mean())
                res[c1 + c2 + "_pos2"] = float(m.robust_2.mean())
        # Le etichette ufficiali che possediamo sono di DeepSeek-R1-0528-Qwen3-8B. Per quel
        # modello il confronto e' una CONCORDANZA; per ogni altro modello e' un CONTROLLO
        # CROSS-MODELLO (dice solo se esiste un effetto "problema" comune ai modelli) e non
        # va letto come riproduzione dell'etichetta.
        e_deepseek = "DeepSeek-R1-0528" in model
        if vere is not None:
            for c in sorted(et):
                m = et[c].merge(vere, on="problem_id")
                m = m[m.definito]
                if len(m) > 10:
                    k, lo, hi = kappa_ic(m.robust.astype(int), m.loro.astype(int), args.n_boot, args.seed)
                    nome = ("kappa(%s,loro) CONCORDANZA   " % c) if e_deepseek else ("kappa(%s,DS-loro) controllo x-modello" % c)
                    print("   %-38s n=%3d  kappa %+.4f  IC95 [%+.4f, %+.4f]  robust loro %.1f%%"
                          % (nome, len(m), k, lo, hi, 100 * m.loro.mean()))
                    if e_deepseek:
                        res["%s_loro" % c] = k
                    else:
                        res["%s_xmodello" % c] = k
        # secondarie: per famiglia, e curva di affidabilità
        if "A" in et and "B" in et:
            m = et["A"].merge(et["B"], on="problem_id", suffixes=("_1", "_2"))
            print("   per famiglia, kappa(A,B):", end=" ")
            for f in FAMIGLIE:
                s = m[m["stab_%s_1" % f].notna() & m["stab_%s_2" % f].notna()]
                if len(s) > 10:
                    print("%s %+.3f" % (f[:4], kappa(s["stab_%s_1" % f].astype(bool).astype(int), s["stab_%s_2" % f].astype(bool).astype(int))), end="  ")
            print()
        riga_finale[model] = res

    print()
    print("=" * 78)
    print("GRIGLIA DI LETTURA (PREREG_E13.md, scritta prima)")
    print("  nota: la colonna kappa(A,loro) esiste solo per DeepSeek, l'unico modello di cui")
    print("  possediamo le etichette ufficiali. Per gli altri la griglia usa il solo kappa(A,B).")
    for model, r in sorted(riga_finale.items()):
        kab = r.get("AB", float("nan"))
        kal = r.get("A_loro", float("nan"))
        if np.isnan(kab):
            cella = "incompleto"
        elif kab < 0.3:
            cella = "(i) etichetta INSTABILE PER COSTRUZIONE: nessun metodo la predice oltre il caso"
        elif kab > 0.6 and (np.isnan(kal) or abs(kal) < 0.15):
            cella = "(ii) riproducibile, ma le LORO non vengono dalla definizione dichiarata"
        elif kab > 0.6 and kal > 0.3:
            cella = "(iii) E12 era sottodimensionato: si riapre lo Stadio 3"
        else:
            cella = "zona grigia: si riporta la curva e non si conclude oltre"
        print("  %-45s kappa(A,B) %+.4f -> %s" % (model, kab, cella))
    print("=" * 78)
    (open(os.path.join(args.dir, "e13_verdetto.json"), "w", encoding="utf-8")
     .write(json.dumps({m: {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
                        for m, r in riga_finale.items()}, indent=1)))


if __name__ == "__main__":
    main()
