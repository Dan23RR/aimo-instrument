"""E5 — la scoperta di H9 e' sfruttabile o e' un artefatto filtrato via dagli organizzatori?

H9 ha mostrato: quando base_accuracy ~ 0 (il modello non risolve MAI l'originale), l'item e'
etichettato robust nel 100% dei casi. E' una conseguenza della definizione: robust =
n_detrimental == 0 = nessuna perturbazione peggiora l'accuratezza; se l'accuratezza e' gia' a
zero, non puo' peggiorare, quindi robust "banale".

Regola NON MONOTONA che ne segue: robust se ba ~ 0 (banale) OPPURE ba alta e stabile.
Una soglia semplice non la trova. Ma prima di costruirci sopra: gli organizzatori
distribuiscono set "filtered". Se il set di validazione FINALE e' filtrato per ba alta,
i casi banali non esistono nel test e il segnale e' morto per la gara.

val-sample "copre tutti i tipi di modelli del test set": e' il proxy migliore del test.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "dati"


def righe_di(nomi):
    grezze = []
    for s in nomi:
        grezze += json.loads((CACHE / (s + ".json")).read_text(encoding="utf-8"))
    agg = defaultdict(list)
    for r in grezze:
        agg[(r["model_id"], r["problem_id"])].append(r)
    out = []
    for (m, p), rr in agg.items():
        detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in rr)
        ba = [float(x["base_accuracy"]) for x in rr if x.get("base_accuracy") is not None]
        lab = None
        et = [x for x in rr if x.get("model_is_robust") is not None]
        if et:
            lab = all(x["model_is_robust"] in (True, "True") for x in et)
        if ba:
            out.append({"model": m, "prob": p, "robust": detr == 0,
                        "lab_dich": lab, "ba": sum(ba) / len(ba)})
    return out


def distr(righe, nome):
    n = len(righe)
    z = sum(1 for r in righe if r["ba"] < 0.05)
    uno = sum(1 for r in righe if r["ba"] >= 0.95)
    mid = n - z - uno
    rob = sum(r["robust"] for r in righe)
    print("  %-42s n=%3d  ba~0:%2d  ba~1:%3d  mid:%2d  robust:%.2f"
          % (nome, n, z, uno, mid, rob / n))
    return z


def main():
    print("=" * 78)
    print("E5 — il segnale 'trivial-robust' esiste nel proxy del test set?")
    print("=" * 78)

    print("\n[distribuzione di base_accuracy per dataset]")
    distr(righe_di(["val-sample"]), "val-sample (PROXY DEL TEST)")
    distr(righe_di(["augmented-sample-math-agg"]), "augmented-sample-math-agg")
    distr(righe_di(["augmented-sample-math-agg-filtered"]), "  ...-agg-FILTERED")
    distr(righe_di(["augmented-sample-math"]), "augmented-sample-math")
    distr(righe_di(["augmented-sample-math-full-filtered"]), "  ...-full-FILTERED")
    distr(righe_di(["aimo-interp-challenge-sample-full"]), "challenge-sample-full")

    # -------- la regola non monotona batte il prior su val-sample, per modello?
    print("\n" + "-" * 78)
    print("La regola 'robust se ba<0.05, altrimenti spurious' su val-sample")
    print("-" * 78)
    val = righe_di(["val-sample"])
    modelli = sorted({r["model"] for r in val})
    g_reg = g_prior = 0
    for r in val:
        pred_reg = (r["ba"] < 0.05)          # robust solo se banale
        g_reg += (pred_reg == r["robust"])
        g_prior += (False == r["robust"])    # prior: tutto spurious
    n = len(val)
    print("  n = %d (val-sample intero, %d modelli)" % (n, len(modelli)))
    print("  regola trivial-robust : %.4f" % (g_reg / n))
    print("  prior tutto-spurious  : %.4f" % (g_prior / n))
    print("  guadagno              : %+.1f punti" % (100 * (g_reg - g_prior) / n))

    # -------- e su TUTTI i dati pubblici, per modello (leave-one-model-out)
    print("\n" + "-" * 78)
    print("Stessa regola, leave-one-model-out su tutti i dati pubblici")
    print("-" * 78)
    tutte = righe_di(["aimo-interp-challenge-sample-full", "val-sample",
                      "augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
                      "augmented-sample-math", "augmented-sample-math-full-filtered"])
    g_reg = g_prior = n = 0
    for r in tutte:
        g_reg += ((r["ba"] < 0.05) == r["robust"])
        g_prior += (False == r["robust"])
        n += 1
    soglia = math.sqrt(0.25 / n)
    print("  n = %d" % n)
    print("  regola trivial-robust : %.4f" % (g_reg / n))
    print("  prior tutto-spurious  : %.4f" % (g_prior / n))
    print("  guadagno              : %+.1f punti,  soglia %.1f  -> %s"
          % (100 * (g_reg - g_prior) / n, 100 * soglia,
             "REALE" if (g_reg - g_prior) / n >= soglia else "rumore"))

    print("\n" + "=" * 78)
    print("VERDETTO")
    print("=" * 78)
    print("  Se val-sample ha casi ba~0: il segnale trivial-robust vive nel test e vale punti.")
    print("  Se val-sample non ne ha (o i filtered li rimuovono): il test e' filtrato per ba")
    print("  alta, il caso banale non c'e', e serve il vero discriminatore robust/spurious")
    print("  sui casi ad alta accuratezza, che richiede le feature del modello.")


if __name__ == "__main__":
    main()
