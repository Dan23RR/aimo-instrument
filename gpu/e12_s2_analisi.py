"""E12 / Stadio 2 — verdetto su H32, protocollo di PREREG_E12.md + PREREG_E12_DEVIAZIONE.md.

Uso: python e12_s2_analisi.py

  robust_E12 = originale corretto E ogni variante accettata corretta ("corretto" = uguale al
  riferimento gpt-oss), definito solo con riferimento e >= 1 variante accettata.
  Primaria: kappa di Cohen fra robust_E12 e l'etichetta loro. Morte se kappa < 0.30.
  Secondarie: per famiglia; sottoinsieme con riferimento confermato dalla modale di Stadio 1;
  accuratezza di robust_E12 contro la maggioranza.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
SOGLIA_MORTE = 0.30
FAMIGLIE = ["rephrase", "rename", "domain", "distract", "typos"]


def kappa(a, b):
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    po = (a == b).mean()
    pe = (a.mean() * b.mean()) + ((1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    var = pd.read_parquet(QUI / "out" / "e12_s2_varianti.parquet")
    ds = pd.read_parquet(QUI / "out" / "e12_s2_deepseek.parquet")
    lab = pd.read_parquet(QUI / "out" / "feature_greedy.parquet")[["problem_id", "robust"]]
    s1 = pd.read_parquet(QUI / "out" / "e12_s1_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet")[["problem_id", "modale"]]
    d = var.merge(ds, on=["problem_id", "famiglia", "idx"], how="left")
    d["corretto"] = (d.risp_ds.notna()) & (d.risp_ds == d.risp_ref)

    print("=" * 74)
    print("E12 / STADIO 2 — VERDETTO H32")
    print("=" * 74)
    v = d[d.idx >= 0]
    print("  varianti generate %d, accettate %d (%.1f%%); motivi di scarto: %s"
          % (len(v), int(v.accettata.sum()), 100 * v.accettata.mean(), v[~v.accettata].motivo.value_counts().to_dict()))
    tasso = v.groupby("famiglia").accettata.mean()
    print("  tasso di accettazione per famiglia: %s" % {f: round(float(tasso.get(f, 0)), 3) for f in FAMIGLIE})
    # regola fissata in PREREG_E12_DEVIAZIONE: famiglie sotto il 60% si riportano ma NON entrano nel primario
    escluse = [f for f in FAMIGLIE if tasso.get(f, 0) < 0.60]
    print("  famiglie ESCLUSE dal primario (accettazione < 60%%): %s" % (escluse or "nessuna"))
    orig = d[d.idx == -1].set_index("problem_id")
    acc_tutte = d[(d.idx >= 0) & d.accettata]
    acc = acc_tutte[~acc_tutte.famiglia.isin(escluse)]
    righe = []
    for pid, o in orig.iterrows():
        if o.risp_ref is None or pd.isna(o.risp_ref):
            righe.append({"problem_id": pid, "definito": False, "motivo": "senza_riferimento"})
            continue
        va = acc[acc.problem_id == pid]
        if len(va) == 0:
            righe.append({"problem_id": pid, "definito": False, "motivo": "nessuna_variante_accettata"})
            continue
        vt = acc_tutte[acc_tutte.problem_id == pid]
        r = {"problem_id": pid, "definito": True, "motivo": "", "orig_corretto": bool(o.corretto),
             "n_acc": len(va), "n_var_corrette": int(va.corretto.sum()),
             "robust_E12": bool(o.corretto and va.corretto.all()),
             "robust_E12_tutte": bool(o.corretto and vt.corretto.all()) if len(vt) else None}
        for f in FAMIGLIE:
            vf = vt[vt.famiglia == f]
            r["stab_" + f] = bool(vf.corretto.all()) if len(vf) else None
            r["n_" + f] = len(vf)
        righe.append(r)
    R = pd.DataFrame(righe).merge(lab, on="problem_id").merge(s1, on="problem_id", how="left")
    R["ref_confermato"] = R.problem_id.map(orig.risp_ref) == R.modale
    D = R[R.definito].copy()
    print()
    print("  problemi: %d totali, %d definiti; esclusi: %s" % (len(R), len(D), R[~R.definito].motivo.value_counts().to_dict()))
    print("  originale corretto (DeepSeek == riferimento gpt-oss): %d/%d" % (int(D.orig_corretto.sum()), len(D)))
    y = D.robust.astype(int).to_numpy()
    e = D.robust_E12.astype(int).to_numpy()
    k = kappa(e, y)
    accu = (e == y).mean()
    magg = max(y.mean(), 1 - y.mean())
    print()
    print("  robust LORO %.1f%% | robust E12 %.1f%%" % (100 * y.mean(), 100 * e.mean()))
    print("  H32  kappa = %.4f   (morte se < %.2f)   accuratezza %.4f contro maggioranza %.4f (%+.1f punti)"
          % (k, SOGLIA_MORTE, accu, magg, 100 * (accu - magg)))
    esito = "SOPRAVVIVE" if k >= SOGLIA_MORTE else "MORTA"
    print("  tavola: E12=1&loro=1 %d | E12=1&loro=0 %d | E12=0&loro=1 %d | E12=0&loro=0 %d"
          % (int(((e == 1) & (y == 1)).sum()), int(((e == 1) & (y == 0)).sum()), int(((e == 0) & (y == 1)).sum()), int(((e == 0) & (y == 0)).sum())))
    print()
    print("  secondarie:")
    T = D[D.robust_E12_tutte.notna()]
    if len(T) > 5 and escluse:
        print("    con TUTTE le famiglie (anche %s): n=%d, kappa %.4f, robust E12 %.1f%%"
              % (escluse, len(T), kappa(T.robust_E12_tutte.astype(bool).astype(int), T.robust.astype(int)), 100 * T.robust_E12_tutte.astype(bool).mean()))
    C = D[D.ref_confermato]
    if len(C) > 5:
        print("    sottoinsieme con riferimento confermato dalla modale S1: n=%d, kappa %.4f" % (len(C), kappa(C.robust_E12.astype(int), C.robust.astype(int))))
    for f in FAMIGLIE:
        S = D[D["stab_" + f].notna() & D.orig_corretto]
        if len(S) > 5:
            print("    famiglia %-9s n=%3d  stabile %.1f%%  kappa vs loro %.4f" % (f, len(S), 100 * S["stab_" + f].astype(bool).mean(), kappa(S["stab_" + f].astype(bool).astype(int), S.robust.astype(int))))
    # lettura per chi non guarda solo il numero: fra i robust loro, quante varianti rompono DeepSeek?
    print("    varianti corrette fra i problemi robust loro: %.1f%% ; fra gli spurious loro: %.1f%%"
          % (100 * D[D.robust].n_var_corrette.sum() / max(1, D[D.robust].n_acc.sum()),
             100 * D[~D.robust].n_var_corrette.sum() / max(1, D[~D.robust].n_acc.sum())))
    print()
    print("=" * 74)
    print("VERDETTO H32: %s  (kappa %.4f contro soglia %.2f, n=%d)" % (esito, k, SOGLIA_MORTE, len(D)))
    print("=" * 74)
    (QUI / "out" / "e12_s2_verdetto.json").write_text(json.dumps(
        {"n_definiti": int(len(D)), "kappa": float(k), "accuratezza": float(accu), "maggioranza": float(magg),
         "robust_loro": float(y.mean()), "robust_E12": float(e.mean()), "esito": esito}, indent=1), encoding="utf-8")
    D.to_parquet(QUI / "out" / "e12_s2_per_problema.parquet")


if __name__ == "__main__":
    main()
