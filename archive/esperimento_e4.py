"""E4 — quanto vale base_accuracy come predittore di robustezza, offline sui dati pubblici.

Ipotesi e criteri di morte in AIMO/PREREG_E4.md.
CV raggruppata per modello: nessun modello in train e test insieme.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"


def dati():
    grezze = []
    for s in ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
              "augmented-sample-math-agg-filtered", "augmented-sample-math",
              "augmented-sample-math-full-filtered"]:
        grezze += json.loads((CACHE / (s + ".json")).read_text(encoding="utf-8"))
    agg = defaultdict(list)
    for r in grezze:
        agg[(r["model_id"], r["problem_id"])].append(r)
    righe = []
    for (m, p), rr in agg.items():
        detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in rr)
        ba = [float(x["base_accuracy"]) for x in rr if x.get("base_accuracy") is not None]
        if not ba:
            continue
        righe.append({"model": m, "prob": p, "robust": detr == 0,
                      "ba": sum(ba) / len(ba)})
    return righe


def soglia_migliore(coppie):
    """coppie = [(ba, robust)]. Trova la soglia t che massimizza l'accuratezza di
    'robust se ba > t'. Restituisce (soglia, accuratezza in-sample)."""
    if not coppie:
        return 0.5, 0.0
    cands = sorted({b for b, _ in coppie} | {0.0, 1.0})
    best_t, best_a = 0.5, 0.0
    for t in cands:
        a = sum(1 for b, y in coppie if (b > t) == y) / len(coppie)
        if a > best_a:
            best_a, best_t = a, t
    return best_t, best_a


def main():
    righe = dati()
    modelli = sorted({r["model"] for r in righe})
    print("=" * 78)
    print("E4 — base_accuracy come predittore di robustezza")
    print("=" * 78)
    print("  %d coppie, %d modelli" % (len(righe), len(modelli)))
    glob = sum(r["robust"] for r in righe) / len(righe)
    print("  tasso robust globale: %.3f" % glob)

    # correlazione grezza
    import statistics
    ba_r = [r["ba"] for r in righe if r["robust"]]
    ba_s = [r["ba"] for r in righe if not r["robust"]]
    print("\n  base_accuracy media:  robust %.3f   spurious %.3f   (differenza %.3f)"
          % (statistics.mean(ba_r), statistics.mean(ba_s),
             statistics.mean(ba_r) - statistics.mean(ba_s)))

    # -------------------------------------------------- H7: soglia per modello, CV per modello
    print("\n" + "-" * 78)
    print("H7 — soglia scelta su altri modelli, applicata al modello held-out")
    print("-" * 78)
    g_mod = g_prior = g_glob = n = 0
    for held in modelli:
        train = [r for r in righe if r["model"] != held]
        test = [r for r in righe if r["model"] == held]
        if len(test) < 5:
            continue
        t, _ = soglia_migliore([(r["ba"], r["robust"]) for r in train])
        # prior per modello held-out: lo conosciamo? nel warmup NO per i modelli nuovi.
        # uso il prior GLOBALE dei modelli di train (l'unica cosa nota per un modello mai visto)
        prior_glob = sum(r["robust"] for r in train) / len(train) > 0.5
        for r in test:
            g_mod += ((r["ba"] > t) == r["robust"])
            g_glob += (prior_glob == r["robust"])
            n += 1
        # prior del modello held-out stesso (oracolo, solo per confronto)
        maj = sum(x["robust"] for x in test) > len(test) / 2
        g_prior += sum(1 for x in test if (maj == x["robust"]))
    soglia = math.sqrt(0.25 / n)
    print("  n = %d predizioni (leave-one-model-out)" % n)
    print("  base_accuracy + soglia da altri modelli : %.4f" % (g_mod / n))
    print("  prior globale dei modelli di train      : %.4f  (cio' che sai di un modello nuovo)"
          % (g_glob / n))
    print("  prior del modello stesso (ORACOLO)      : %.4f  (non disponibile per modelli nuovi)"
          % (g_prior / n))
    print("  guadagno di base_accuracy sul prior globale: %+.1f punti, soglia %.1f  -> %s"
          % (100 * (g_mod - g_glob) / n, 100 * soglia,
             "VIVA" if (g_mod - g_glob) / n >= soglia else "MORTA"))

    # -------------------------------------------------- H8: tetto con soglia oracolo
    print("\n" + "-" * 78)
    print("H8 — tetto: soglia oracolo sul modello held-out stesso")
    print("-" * 78)
    g = n = 0
    for held in modelli:
        test = [r for r in righe if r["model"] == held]
        if len(test) < 5:
            continue
        _, a = soglia_migliore([(r["ba"], r["robust"]) for r in test])
        g += a * len(test)
        n += len(test)
    print("  base_accuracy con soglia perfetta per modello: %.4f" % (g / n))
    print("  (e' il massimo che questo SOLO segnale puo' dare; sopra ci vanno le")
    print("   feature di incertezza per avvicinarsi)")

    # -------------------------------------------------- H9: fascia estrema vs intermedia
    print("\n" + "-" * 78)
    print("H9 — base_accuracy e' piu' predittiva quando e' estrema?")
    print("-" * 78)
    for lo, hi, nome in [(0.0, 0.05, "ba ~ 0 (modello non risolve mai)"),
                         (0.95, 1.01, "ba ~ 1 (modello risolve sempre)"),
                         (0.05, 0.95, "ba intermedia")]:
        sub = [r for r in righe if lo <= r["ba"] < hi]
        if not sub:
            continue
        rob = sum(r["robust"] for r in sub) / len(sub)
        # quanto e' "decisa" la fascia: purezza = max(rob, 1-rob)
        print("  %-38s n=%3d  robust=%.2f  purezza=%.2f"
              % (nome, len(sub), rob, max(rob, 1 - rob)))

    print("\n" + "=" * 78)
    print("LETTURA")
    print("=" * 78)
    print("  Se H7 e' viva: base_accuracy batte il prior anche su modelli mai visti,")
    print("  ed e' il nucleo di un metodo che generalizza alla fase principale.")
    print("  Se e' morta: senza le feature di incertezza (che richiedono il modello),")
    print("  offline non si dimostra nulla, e il valore e' tutto nella carta di misura.")


if __name__ == "__main__":
    main()
