"""VARIANTE CON ALIAS UNITO (21/9/2026). Controllo avversario su H2: il nullo che ho usato era sbagliato?

H2 dichiarava: i problemi sono concordi fra modelli nel 77% dei casi contro il 49% atteso
sotto indipendenza, z = 4,65. Ma quel 49% era calcolato con un UNICO tasso globale p.

I tassi di robustezza per modello vanno dal 20% al 70% (misurato in E2). Con tassi cosi'
diversi, il nullo corretto usa il tasso di CIASCUN modello, non uno globale, e la concordanza
attesa e' molto piu' alta: due modelli entrambi al 20% concordano spesso senza che il
problema c'entri nulla.

E3 ha appena mostrato che l'ordinamento NON trasferisce nemmeno con soglia oracolo. Se il
canale-problema fosse reale come dichiarato, quel risultato sarebbe strano. Quindi o H2 e'
gonfiata dal nullo sbagliato, o le due cose misurano canali diversi. Va deciso sui numeri.

Questo script non cerca di salvare H2. Cerca di ucciderla.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

CACHE = Path(__file__).resolve().parent / "dati"


def main():
    righe = json.loads((CACHE / "tabella_aggregata.json").read_text(encoding="utf-8"))
    # Variante del 21/9/2026 (verifica avversaria del report): le 52 righe pubblicate come
    # deepseek-ai/DeepSeek-R1-0528-Qwen3-8B sono copie delle righe qwen3-8b:low (etichetta e
    # max_drop identici 52/52). Si uniscono i due nomi, come fa gia' aggrega.py, cosi' la stessa
    # misura non conta come due modelli.
    ALIAS = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
    for r in righe:
        r["model_id"] = ALIAS.get(r["model_id"], r["model_id"])
    per_mod = defaultdict(dict)
    for r in righe:
        per_mod[r["model_id"]][r["problem_id"]] = r["robust"]

    tasso = {m: sum(v.values()) / len(v) for m, v in per_mod.items()}
    per_prob = defaultdict(list)          # problem_id -> [(modello, robust)]
    for m, d in per_mod.items():
        for p, rb in d.items():
            per_prob[p].append((m, rb))
    multi = {p: v for p, v in per_prob.items() if len(v) >= 2}

    print("=" * 78)
    print("CONTROLLO AVVERSARIO SU H2")
    print("=" * 78)
    print("  problemi visti da >=2 modelli: %d" % len(multi))

    concordi = sum(1 for v in multi.values() if len({rb for _, rb in v}) == 1)

    # nullo SBAGLIATO: un solo tasso globale
    tutti = [rb for v in multi.values() for _, rb in v]
    p_glob = sum(tutti) / len(tutti)
    att_sbagliato = sum(p_glob ** len(v) + (1 - p_glob) ** len(v) for v in multi.values())

    # nullo CORRETTO: il tasso di ciascun modello, indipendenti
    att_giusto = 0.0
    var = 0.0
    for v in multi.values():
        pr = 1.0
        ps = 1.0
        for m, _ in v:
            pr *= tasso[m]
            ps *= (1 - tasso[m])
        q = pr + ps
        att_giusto += q
        var += q * (1 - q)

    print("\n  concordi osservati                    : %d (%.0f%%)"
          % (concordi, 100 * concordi / len(multi)))
    print("  attesi, nullo con tasso GLOBALE       : %.1f (%.0f%%)   <- quello che avevo usato"
          % (att_sbagliato, 100 * att_sbagliato / len(multi)))
    print("  attesi, nullo con tasso PER MODELLO   : %.1f (%.0f%%)   <- quello corretto"
          % (att_giusto, 100 * att_giusto / len(multi)))
    z = (concordi - att_giusto) / math.sqrt(var) if var > 0 else 0.0
    print("\n  eccesso sul nullo corretto            : %+.1f problemi, z = %.2f" % (concordi - att_giusto, z))
    print("  >>> H2 %s sul nullo corretto" % ("SOPRAVVIVE" if z > 2 else "MORTA"))

    # -------------------------------------------------- il guadagno, contro la baseline giusta
    print("\n" + "-" * 78)
    print("IL GUADAGNO DEL CANALE-PROBLEMA, contro la baseline CORRETTA")
    print("-" * 78)
    print("  La baseline giusta non e' la maggioranza globale: e' la maggioranza")
    print("  DEL MODELLO BERSAGLIO, che a runtime si conosce dal solo model_id.")
    g = mg_glob = mg_mod = n = 0
    for p, v in multi.items():
        for i, (m, rb) in enumerate(v):
            altri = [x for j, x in enumerate(v) if j != i]
            voto = sum(1 for _, x in altri if x) * 2 > len(altri)
            g += (voto == rb)
            mg_mod += ((tasso[m] > 0.5) == rb)
            mg_glob += ((p_glob > 0.5) == rb)
            n += 1
    print("  voto degli altri modelli sul problema : %.4f" % (g / n))
    print("  maggioranza globale                   : %.4f   (differenza %+.1f punti)"
          % (mg_glob / n, 100 * (g - mg_glob) / n))
    print("  maggioranza PER MODELLO               : %.4f   (differenza %+.1f punti)"
          % (mg_mod / n, 100 * (g - mg_mod) / n))
    soglia = math.sqrt(0.25 / n)
    vero = (g - mg_mod) / n
    print("\n  soglia di significativita' su n=%d   : %.1f punti" % (n, 100 * soglia))
    print("  >>> il canale-problema %s una volta tolto cio' che il solo model_id gia' da'"
          % ("SOPRAVVIVE" if vero >= soglia else "NON SOPRAVVIVE"))


if __name__ == "__main__":
    main()
