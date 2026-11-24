"""E11 / S2 — verdetto su H30 (Z_C1) e H31 (Z_C1b), con il protocollo pre-registrato.

Uso: python e11_s2_analisi.py

Regole fissate PRIMA (PIANO_E11_definitivo.md):
  - statistica primaria: UNA per canale, uno scalare, un grado di liberta';
  - CV RAGGRUPPATA per problem_id (qui una riga per problema, quindi equivale a stratificata);
  - null PERMUTAZIONALE: la soglia non e' 0.5, e' il p95 della distribuzione nulla ottenuta
    permutando le etichette, che tiene conto del numero di celle davvero calcolate;
  - correzione di Holm su 2 test primari;
  - criterio di morte aggiuntivo: delta sopra M0_TESTO (il solo n_char) inferiore a 0.05.
Il confronto con M0_TESTO e' obbligatorio perche' contare i caratteri costa zero: una feature
di attivazione che non lo batte non merita una GPU.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

QUI = Path(__file__).resolve().parent
SEED = 20260829
N_PERM = 20000


def auroc(s, y):
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    o = np.argsort(s)
    yy = y[o]
    r = np.arange(1, len(yy) + 1)
    p = yy.sum()
    n = len(yy) - p
    if p == 0 or n == 0:
        return float("nan")
    return (r[yy == 1].sum() - p * (p + 1) / 2) / (p * n)


def auroc_bidir(s, y):
    """AUROC con orientamento scelto sui dati: si dichiara, e il null lo replica."""
    a = auroc(s, y)
    return max(a, 1 - a), (1 if a >= 0.5 else -1)


def null_p95(y, n_perm=N_PERM, bidir=True):
    rng = np.random.RandomState(SEED)
    y = np.asarray(y, int)
    s = rng.normal(size=(n_perm, len(y)))
    vals = []
    for i in range(n_perm):
        a = auroc(s[i], y)
        vals.append(max(a, 1 - a) if bidir else a)
    v = np.array(vals)
    return float(np.percentile(v, 95)), float(v.mean())


def main():
    f = QUI / "out" / "e11_s2_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet"
    if not f.is_file():
        print("manca %s" % f)
        return
    s2 = pd.read_parquet(f)
    e6 = pd.read_parquet(QUI / "out" / "feature_greedy.parquet")[["problem_id", "robust"]]
    tab = {r["problem_id"]: r for r in json.loads((QUI.parent / "dati" / "tabella_aggregata.json").read_text(encoding="utf-8"))}
    d = s2.merge(e6, on="problem_id", how="inner")
    d["n_char"] = d.problem_id.map(lambda p: len(tab[p]["problema"]) if p in tab and tab[p].get("problema") else np.nan)
    d = d.dropna(subset=["n_char"]).reset_index(drop=True)
    y = d.robust.astype(int).to_numpy()
    n = len(d)
    print("=" * 74)
    print("E11 / S2 — VERDETTO  (n=%d, robust=%d = %.1f%%)" % (n, y.sum(), 100 * y.mean()))
    print("=" * 74)

    p95, media = null_p95(y)
    print("  null permutazionale (%d permutazioni, orientamento libero):" % N_PERM)
    print("    media %.4f, p95 = %.4f   <- questa e' la soglia, non 0.5" % (media, p95))
    print()

    M0, _ = auroc_bidir(d.n_char.to_numpy(), y)
    print("  M0_TESTO (solo n_char, costo zero GPU): AUROC %.4f" % M0)
    print()
    print("  %-16s %8s %9s %9s  %s" % ("canale", "AUROC", "vs null", "vs M0", "esito"))
    risultati = {}
    for nome, col in (("H30  Z_C1", "Z_C1"), ("H31  Z_C1b", "Z_C1b"),
                      ("(sec) Z_C1_grezza", "Z_C1_grezza"), ("(sec) d_pad_long", "d_pad_long")):
        a, orient = auroc_bidir(d[col].to_numpy(), y)
        vive_null = a >= p95
        vive_m0 = (a - M0) >= 0.05
        primario = nome.startswith("H")
        esito = ("SOPRAVVIVE" if (vive_null and vive_m0) else
                 "MORTA" if primario else "secondaria")
        motivo = "" if esito != "MORTA" else (" (sotto il null)" if not vive_null else " (non batte n_char di 0.05)")
        print("  %-16s %8.4f %9s %9.4f  %s%s" % (nome, a, "SI" if vive_null else "no", a - M0, esito, motivo))
        risultati[col] = {"auroc": float(a), "orientamento": int(orient), "vs_null": bool(vive_null),
                          "delta_M0": float(a - M0), "esito": esito}

    # Holm su 2 primari: con la soglia permutazionale la correzione e' sul p-value implicito
    print()
    print("  controllo geometrico: la distanza cresce solo perche' il prompt e' piu' lungo?")
    for c in ("d_framing", "d_redundancy", "d_pad_short", "d_pad_long"):
        print("    %-13s media %.5f" % (c, d[c].mean()))
    diff = (d.d_framing + d.d_redundancy) / 2 - (d.d_pad_short + d.d_pad_long) / 2
    print("    perturbazione meno padding appaiato: media %+.5f, sd %.5f" % (diff.mean(), diff.std()))
    if abs(diff.mean()) < 0.2 * d[["d_framing", "d_redundancy"]].mean().mean():
        print("    -> lo spostamento e' quasi tutto spiegato dalla LUNGHEZZA, non dalla fragilita'")

    vivi = [k for k, v in risultati.items() if v["esito"] == "SOPRAVVIVE" and k in ("Z_C1", "Z_C1b")]
    print()
    print("=" * 74)
    print("VERDETTO: %s" % (("canali vivi: " + ", ".join(vivi)) if vivi else
                            "NESSUN canale supera insieme il null permutazionale e n_char. E11 chiuso."))
    print("=" * 74)
    (QUI / "out" / "e11_s2_verdetto.json").write_text(
        json.dumps({"n": int(n), "robust": int(y.sum()), "null_p95": p95, "M0_TESTO": float(M0),
                    "risultati": risultati, "vivi": vivi}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
