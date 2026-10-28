"""E3 — l'ordinamento di difficolta' trasferisce fra modelli, anche se il livello no?

Ipotesi e criterio di morte in AIMO/PREREG_E3.md, scritto prima di questo run.

Differenza da E2: li' il sensore veniva usato in valore ASSOLUTO (label, o base_accuracy
sopra/sotto la mediana) e moriva perche' si portava dietro il proprio tasso di base.
Qui si usa solo l'ORDINE dentro il lotto, che e' invariante al tasso di base.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"
IN_CACHE = "qwen3-8b:low"


def matrice():
    grezze = (json.loads((CACHE / "aimo-interp-challenge-sample-full.json").read_text(encoding="utf-8"))
              + json.loads((CACHE / "val-sample.json").read_text(encoding="utf-8")))
    g = defaultdict(list)
    for r in grezze:
        if r.get("dataset_id") == "aimo-2025-reference":
            g[(r["model_id"], r["problem_id"])].append(r)
    out = {}
    for k, righe in g.items():
        detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in righe)
        ba = [float(x["base_accuracy"]) for x in righe if x.get("base_accuracy") is not None]
        out[k] = (detr == 0, sum(ba) / len(ba) if ba else None)
    return out


def valuta(M, modelli, problemi, oracolo_k):
    """Per ogni coppia (A sensore, B bersaglio): ordina per base_accuracy di A,
    predice robust i primi k. k = meta' del lotto, oppure il vero numero se oracolo_k."""
    g = mg = n = 0
    per_sensore = defaultdict(lambda: [0, 0, 0])
    for a in modelli:
        ba = {p: M[(a, p)][1] for p in problemi if (a, p) in M and M[(a, p)][1] is not None}
        if len(ba) < 5:
            continue
        for b in modelli:
            if a == b:
                continue
            com = [p for p in problemi if p in ba and (b, p) in M]
            if len(com) < 5:
                continue
            vero = {p: M[(b, p)][0] for p in com}
            # ordine decrescente di facilita' secondo A; tie-break stabile sul problem_id
            ordine = sorted(com, key=lambda p: (-ba[p], p))
            k = sum(vero.values()) if oracolo_k else (len(com) + 1) // 2
            pred = {p: (i < k) for i, p in enumerate(ordine)}
            giuste = sum(1 for p in com if pred[p] == vero[p])
            magg = max(sum(vero.values()), len(com) - sum(vero.values()))
            g += giuste
            mg += magg
            n += len(com)
            s = per_sensore[a]
            s[0] += giuste
            s[1] += magg
            s[2] += len(com)
    return g, mg, n, per_sensore


def main():
    M = matrice()
    modelli = sorted({m for m, _ in M})
    problemi = sorted({p for _, p in M})
    print("=" * 78)
    print("E3 — trasferisce l'ORDINE di difficolta' fra modelli?")
    print("=" * 78)
    print("  %d modelli x %d problemi" % (len(modelli), len(problemi)))

    for etichetta, oracolo in (("H5  soglia = meta' del lotto (utilizzabile)", False),
                               ("H6  soglia = numero vero di positivi (TETTO, non utilizzabile)", True)):
        g, mg, n, per_sensore = valuta(M, modelli, problemi, oracolo)
        if not n:
            continue
        acc, magg = g / n, mg / n
        soglia = math.sqrt(0.25 / n)
        print("\n" + "-" * 78)
        print(etichetta)
        print("-" * 78)
        print("  predizioni valutate   : %d" % n)
        print("  ordinamento           : %.4f" % acc)
        print("  maggioranza           : %.4f" % magg)
        print("  guadagno              : %+.4f (%.1f punti)   soglia di morte %.1f punti"
              % (acc - magg, 100 * (acc - magg), 100 * soglia))
        vivo = (acc - magg) >= soglia
        print("  >>> %s" % (("SOPRAVVIVE" if vivo else "MORTA") if not oracolo
                            else ("tetto positivo" if vivo else "tetto NEGATIVO")))
        if not oracolo:
            print("\n  per sensore (chi ordina):")
            for a, (gg, mm, nn) in sorted(per_sensore.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
                marca = "  <-- IN CACHE" if a == IN_CACHE else ""
                print("    %-42s %.3f vs %.3f  (%+.1f punti)%s"
                      % (a, gg / nn, mm / nn, 100 * (gg / nn - mm / nn), marca))


if __name__ == "__main__":
    main()
