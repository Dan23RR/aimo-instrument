"""Ricostruisce la regola che produce `model_is_robust` e la applica al set multi-modello.

Il set grande (558 righe, 8 modelli) NON porta la label. I due set che la portano
(val-sample, augmented-sample-math-agg) hanno le stesse colonne. Se la label e' una
funzione deterministica di quelle colonne, la si ricava e la si applica.

Regola d'oro del metodo: NON si accetta una regola che spiega "quasi tutto". O riproduce
la label al 100% su ogni riga etichettata disponibile, o non e' la regola.
"""
import json
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"


def carica(nome):
    return json.loads((CACHE / ("%s.json" % nome)).read_text(encoding="utf-8"))


def mostra_riga(r, prefisso="   "):
    for k, v in r.items():
        s = repr(v)
        if len(s) > 100:
            s = s[:97] + "..."
        print("%s%-30s %s" % (prefisso, k, s))


# --------------------------------------------------------------- regole candidate
REGOLE = {
    "n_detrimental == 0":
        lambda r: (r.get("n_detrimental_permutations") or 0) == 0,
    "permutations_causing_decay vuoto":
        lambda r: not (r.get("permutations_causing_decay") or []),
    "absolute_accuracy_decay <= 0":
        lambda r: (r.get("absolute_accuracy_decay") or 0) <= 0,
    "absolute_accuracy_decay < 1e-9":
        lambda r: (r.get("absolute_accuracy_decay") or 0) < 1e-9,
    "relative_accuracy_decay <= 0":
        lambda r: (r.get("relative_accuracy_decay") or 0) <= 0,
    "permuted >= base":
        lambda r: (r.get("permuted_accuracy") or 0) >= (r.get("base_accuracy") or 0),
    "permuted_accuracy == 1.0":
        lambda r: (r.get("permuted_accuracy") or 0) >= 1.0,
    "base_accuracy == 1.0":
        lambda r: (r.get("base_accuracy") or 0) >= 1.0,
}


def prova_regole(righe, nome):
    print("\n" + "-" * 74)
    print("REGOLE CANDIDATE su %s (%d righe etichettate)" % (nome, len(righe)))
    print("-" * 74)
    vere = sum(1 for r in righe if r["model_is_robust"] in (True, "True"))
    print("  label: %d robust, %d spurious" % (vere, len(righe) - vere))
    superstiti = []
    for etichetta, f in REGOLE.items():
        giuste = 0
        for r in righe:
            atteso = r["model_is_robust"] in (True, "True")
            try:
                ottenuto = bool(f(r))
            except Exception:
                ottenuto = None
            if ottenuto == atteso:
                giuste += 1
        stato = "ESATTA" if giuste == len(righe) else ""
        print("  %-38s %3d/%3d  %5.1f%%  %s"
              % (etichetta, giuste, len(righe), 100 * giuste / len(righe), stato))
        if giuste == len(righe):
            superstiti.append(etichetta)
    return superstiti


def main():
    val = carica("val-sample")
    agg = carica("augmented-sample-math-agg")
    aggf = carica("augmented-sample-math-agg-filtered")
    full = carica("aimo-interp-challenge-sample-full")

    print("=" * 74)
    print("STRUTTURA DEI DATI")
    print("=" * 74)
    print("\nUna riga di val-sample (CON label):")
    mostra_riga(val[0])
    print("\nUna riga del set grande (SENZA label):")
    mostra_riga(full[0])

    # granularita': il set grande e' per (modello, problema, permutazione)?
    print("\n" + "-" * 74)
    print("GRANULARITA'")
    print("-" * 74)
    for nome, righe in (("full", full), ("val-sample", val), ("agg", agg)):
        coppie = defaultdict(int)
        for r in righe:
            coppie[(r.get("model_id"), r.get("problem_id"))] += 1
        d = sorted(coppie.values())
        print("  %-12s %4d righe -> %4d coppie (modello,problema), righe per coppia: min %d max %d"
              % (nome, len(righe), len(coppie), d[0], d[-1]))

    superstiti = []
    for nome, righe in (("val-sample", val), ("augmented-sample-math-agg", agg),
                        ("augmented-sample-math-agg-filtered", aggf)):
        s = prova_regole(righe, nome)
        superstiti.append(set(s))

    comuni = set.intersection(*superstiti) if superstiti else set()
    print("\n" + "=" * 74)
    print("REGOLE ESATTE SU TUTTI E TRE I SET ETICHETTATI: %s"
          % (", ".join(sorted(comuni)) if comuni else "NESSUNA"))
    print("=" * 74)

    if not comuni:
        print("\nLa label NON e' una funzione semplice delle colonne disponibili.")
        print("Prima di inventare una regola composta, guardo i casi che le rompono.")
        for etichetta, f in REGOLE.items():
            rotti = [r for r in val if bool(f(r)) != (r["model_is_robust"] in (True, "True"))]
            if 0 < len(rotti) <= 3:
                print("\n  Regola '%s' rotta da %d righe di val-sample:" % (etichetta, len(rotti)))
                for r in rotti[:2]:
                    print("    label=%s  base=%s permuted=%s abs_decay=%s rel_decay=%s n_detr=%s"
                          % (r["model_is_robust"], r.get("base_accuracy"),
                             r.get("permuted_accuracy"), r.get("absolute_accuracy_decay"),
                             r.get("relative_accuracy_decay"),
                             r.get("n_detrimental_permutations")))


if __name__ == "__main__":
    main()
