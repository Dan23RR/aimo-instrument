"""E2 — il modello in cache puo' fare da sensore per i modelli che non possiamo caricare?

Ipotesi e criteri di morte in AIMO/PREREG_E2.md, scritto prima di questo run.

Dati: pool `aimo-2025-reference`, matrice completa 8 modelli x 10 problemi = 80 coppie,
piu' le righe per-permutazione da cui viene la base_accuracy.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"

IN_CACHE = "qwen3-8b:low"        # alias pubblico di deepseek-ai/DeepSeek-R1-0528-Qwen3-8B


def carica(n):
    return json.loads((CACHE / ("%s.json" % n)).read_text(encoding="utf-8"))


def matrice():
    """(modello, problema) -> (robust, base_accuracy media) sul pool della gara."""
    grezze = carica("aimo-interp-challenge-sample-full") + carica("val-sample")
    g = defaultdict(list)
    for r in grezze:
        if r.get("dataset_id") != "aimo-2025-reference":
            continue
        g[(r["model_id"], r["problem_id"])].append(r)
    out = {}
    for k, righe in g.items():
        detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in righe)
        ba = [float(x["base_accuracy"]) for x in righe if x.get("base_accuracy") is not None]
        out[k] = (detr == 0, sum(ba) / len(ba) if ba else None)
    return out


def main():
    M = matrice()
    modelli = sorted({m for m, _ in M})
    problemi = sorted({p for _, p in M})
    print("=" * 78)
    print("E2 — trasferimento fra modelli sul pool della gara (aimo-2025-reference)")
    print("=" * 78)
    print("  %d modelli x %d problemi = %d coppie osservate" % (len(modelli), len(problemi), len(M)))

    rob = {m: [M[(m, p)][0] for p in problemi if (m, p) in M] for m in modelli}
    print("\n  tasso di robustezza per modello:")
    for m in modelli:
        v = rob[m]
        print("    %-42s %2d/%2d = %.0f%%" % (m, sum(v), len(v), 100 * sum(v) / len(v)))

    print("\n  robust per problema (su quanti modelli):")
    for p in problemi:
        v = [M[(m, p)][0] for m in modelli if (m, p) in M]
        print("    %-10s %d/%d robust" % (p, sum(v), len(v)))

    # ------------------------------------------------------------------ H3
    print("\n" + "-" * 78)
    print("H3 — la label di A predice la label di B sullo stesso problema?")
    print("-" * 78)
    tot_g = tot_m = tot_n = 0
    righe_out = []
    for a in modelli:
        for b in modelli:
            if a == b:
                continue
            com = [p for p in problemi if (a, p) in M and (b, p) in M]
            if len(com) < 5:
                continue
            g = sum(1 for p in com if M[(a, p)][0] == M[(b, p)][0])
            vb = [M[(b, p)][0] for p in com]
            mag = max(sum(vb), len(vb) - sum(vb))
            tot_g += g
            tot_m += mag
            tot_n += len(com)
            if a == IN_CACHE:
                righe_out.append((b, g, mag, len(com)))
    if tot_n:
        acc, mag = tot_g / tot_n, tot_m / tot_n
        soglia = math.sqrt(0.25 / tot_n)
        print("  su tutte le coppie ordinate: %d predizioni" % tot_n)
        print("    usando la label di A       : %.4f" % acc)
        print("    maggioranza di B           : %.4f" % mag)
        print("    guadagno                   : %+.4f (%.1f punti), soglia di morte %.1f punti"
              % (acc - mag, 100 * (acc - mag), 100 * soglia))
        print("    >>> H3 %s" % ("SOPRAVVIVE" if acc - mag >= soglia else "MORTA"))
    if righe_out:
        print("\n  in particolare, il modello IN CACHE (%s) come sensore:" % IN_CACHE)
        for b, g, mg, n in righe_out:
            print("    -> %-42s %2d/%2d = %.0f%%  (maggioranza %.0f%%)"
                  % (b, g, n, 100 * g / n, 100 * mg / n))

    # ------------------------------------------------------------------ H4
    print("\n" + "-" * 78)
    print("H4 — la base_accuracy di A predice la robustezza di B sullo stesso problema?")
    print("-" * 78)
    print("  (base_accuracy = quanto spesso A risolve il problema ORIGINALE.")
    print("   A runtime la self-consistency del modello in cache la approssima.)")
    tot_g = tot_m = tot_n = 0
    dettaglio = []
    for a in modelli:
        ba = {p: M[(a, p)][1] for p in problemi if (a, p) in M and M[(a, p)][1] is not None}
        if len(ba) < 5:
            continue
        mediana = sorted(ba.values())[len(ba) // 2]
        for b in modelli:
            if a == b:
                continue
            com = [p for p in problemi if p in ba and (b, p) in M]
            if len(com) < 5:
                continue
            # regola: A risolve facilmente il problema  ->  B e' robusto
            g = sum(1 for p in com if (ba[p] > mediana) == M[(b, p)][0])
            vb = [M[(b, p)][0] for p in com]
            mg = max(sum(vb), len(vb) - sum(vb))
            tot_g += g
            tot_m += mg
            tot_n += len(com)
            if a == IN_CACHE:
                dettaglio.append((b, g, mg, len(com)))
    if tot_n:
        acc, mag = tot_g / tot_n, tot_m / tot_n
        soglia = math.sqrt(0.25 / tot_n)
        print("\n  su tutte le coppie ordinate: %d predizioni" % tot_n)
        print("    usando base_accuracy di A  : %.4f" % acc)
        print("    maggioranza di B           : %.4f" % mag)
        print("    guadagno                   : %+.4f (%.1f punti), soglia di morte %.1f punti"
              % (acc - mag, 100 * (acc - mag), 100 * soglia))
        print("    >>> H4 %s" % ("SOPRAVVIVE" if acc - mag >= soglia else "MORTA"))
    if dettaglio:
        print("\n  in particolare, base_accuracy del modello IN CACHE come sensore:")
        for b, g, mg, n in dettaglio:
            print("    -> %-42s %2d/%2d = %.0f%%  (maggioranza %.0f%%)"
                  % (b, g, n, 100 * g / n, 100 * mg / n))

    # ------------------------------------------------------------------ tetto
    print("\n" + "-" * 78)
    print("IL TETTO DEL SENSORE: quanto varrebbe un sensore PERFETTO del problema")
    print("-" * 78)
    g = n = 0
    for p in problemi:
        v = [M[(m, p)][0] for m in modelli if (m, p) in M]
        if len(v) < 2:
            continue
        for i in range(len(v)):
            altri = v[:i] + v[i + 1:]
            voto = sum(altri) * 2 > len(altri)
            g += (voto == v[i])
            n += 1
    if n:
        tutte = [M[k][0] for k in M]
        mg = max(sum(tutte), len(tutte) - sum(tutte)) / len(tutte)
        print("  voto degli altri modelli sullo stesso problema: %d/%d = %.4f" % (g, n, g / n))
        print("  maggioranza globale                          : %.4f" % mg)
        print("  guadagno massimo teorico del canale-problema : %+.1f punti" % (100 * (g / n - mg)))


if __name__ == "__main__":
    main()
