"""E1 — il testo del problema, da solo, predice la robustezza?

Ipotesi, metrica e criterio di morte sono in AIMO/PREREG_E1.md, scritto prima di questo run.
Criterio di morte: guadagno sulla maggioranza < sqrt(0.25/n) = 2.7 punti su n=335 -> H1 morta.

Validazione RAGGRUPPATA PER PROBLEMA: nessun problem_id appare sia in train sia in test.
E' il punto che la maggior parte dei partecipanti sbaglia, perche' lo stesso problema
compare con modelli diversi e uno split casuale lo fa passare da entrambe le parti.
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
TAB = QUI / "dati" / "tabella_aggregata.json"

PAROLE = ["prove", "find", "compute", "how many", "probability", "triangle", "circle",
          "polynomial", "integer", "prime", "sequence", "function", "matrix", "sum",
          "maximum", "minimum", "least", "greatest", "remainder", "digits"]


def feature_testo(t: str) -> dict:
    """Solo dal testo del problema. Nessun accesso al modello, nessuna label."""
    b = t.lower()
    numeri = [int(x) for x in re.findall(r"\d+", t)[:200]]
    f = {
        "len_char": len(t),
        "len_parole": len(t.split()),
        "n_frasi": t.count(".") + t.count("?"),
        "dollari": t.count("$"),
        "backslash": t.count("\\"),
        "frac": b.count("frac"),
        "cifre": sum(c.isdigit() for c in t),
        "num_max": math.log1p(max(numeri) if numeri else 0),
        "n_numeri": len(numeri),
        "virgole": t.count(","),
        "maiuscole_isolate": len(re.findall(r"\b[A-Z]\b", t)),
        "parentesi": t.count("("),
    }
    for p in PAROLE:
        f["kw_" + p.replace(" ", "_")] = 1.0 if p in b else 0.0
    return f


# ------------------------------------------------------- regressione logistica a mano
def addestra(X, y, passi=800, lr=0.25, l2=1e-2):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(passi):
        gw = [0.0] * d
        gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            e = p - y[i]
            gb += e
            for j in range(d):
                gw[j] += e * X[i][j]
        b -= lr * gb / n
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
    return w, b


def predici(w, b, x):
    z = b + sum(w[j] * x[j] for j in range(len(x)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def standardizza(X):
    d = len(X[0])
    mu = [sum(r[j] for r in X) / len(X) for j in range(d)]
    sd = []
    for j in range(d):
        v = sum((r[j] - mu[j]) ** 2 for r in X) / max(1, len(X) - 1)
        sd.append(math.sqrt(v) or 1.0)
    return mu, sd


def applica(X, mu, sd):
    return [[(r[j] - mu[j]) / sd[j] for j in range(len(r))] for r in X]


def main():
    righe = json.loads(TAB.read_text(encoding="utf-8"))
    righe = [r for r in righe if r.get("problema")]
    n = len(righe)

    nomi = sorted(feature_testo(righe[0]["problema"]).keys())
    X = [[feature_testo(r["problema"])[k] for k in nomi] for r in righe]
    y = [1 if r["robust"] else 0 for r in righe]
    gruppi = [r["problem_id"] for r in righe]

    print("=" * 78)
    print("E1 — segnale nel solo testo del problema")
    print("=" * 78)
    print("  coppie: %d   problemi distinti: %d   feature: %d"
          % (n, len(set(gruppi)), len(nomi)))
    print("  robust: %d (%.1f%%)" % (sum(y), 100 * sum(y) / n))

    soglia_morte = math.sqrt(0.25 / n)
    print("  criterio di morte pre-registrato: guadagno < %.1f punti -> H1 morta"
          % (100 * soglia_morte))

    # ---------------------------------------------------- CV raggruppata per problema
    unici = sorted(set(gruppi))
    fold_di = {p: i % 5 for i, p in enumerate(unici)}
    acc_tot, magg_tot, tot = 0, 0, 0
    for k in range(5):
        tr = [i for i in range(n) if fold_di[gruppi[i]] != k]
        te = [i for i in range(n) if fold_di[gruppi[i]] == k]
        if not te or not tr:
            continue
        mu, sd = standardizza([X[i] for i in tr])
        Xtr = applica([X[i] for i in tr], mu, sd)
        Xte = applica([X[i] for i in te], mu, sd)
        ytr = [y[i] for i in tr]
        w, b = addestra(Xtr, ytr)
        magg = 1 if sum(ytr) * 2 > len(ytr) else 0
        for j, i in enumerate(te):
            if (1 if predici(w, b, Xte[j]) >= 0.5 else 0) == y[i]:
                acc_tot += 1
            if magg == y[i]:
                magg_tot += 1
            tot += 1

    acc, magg = acc_tot / tot, magg_tot / tot
    guadagno = acc - magg
    print("\n" + "-" * 78)
    print("RISULTATO — CV a 5 fold raggruppata per problem_id")
    print("-" * 78)
    print("  accuratezza del modello di testo : %.4f" % acc)
    print("  accuratezza della maggioranza    : %.4f" % magg)
    print("  guadagno                         : %+.4f (%.1f punti)" % (guadagno, 100 * guadagno))
    print()
    if guadagno < soglia_morte:
        print("  >>> H1 MORTA. Il guadagno (%.1f punti) e' sotto la soglia (%.1f punti)."
              % (100 * guadagno, 100 * soglia_morte))
        print("  >>> Non si cerca una variante che passi. Si dichiara e si va al prior.")
    else:
        print("  >>> H1 SOPRAVVIVE al criterio pre-registrato.")

    # ---------------------------------------------------- H2: trasferimento fra modelli
    print("\n" + "-" * 78)
    print("H2 — la label di un problema trasferisce fra modelli?")
    print("-" * 78)
    per_prob = defaultdict(list)
    for r in righe:
        per_prob[r["problem_id"]].append(1 if r["robust"] else 0)
    multi = {k: v for k, v in per_prob.items() if len(v) >= 2}
    concordi = sum(1 for v in multi.values() if len(set(v)) == 1)
    p = sum(y) / n
    attesi = 0.0
    for v in multi.values():
        m = len(v)
        attesi += p ** m + (1 - p) ** m       # tutti robust oppure tutti spurious
    print("  problemi visti da >=2 modelli : %d" % len(multi))
    print("  concordi su tutti i modelli   : %d (%.0f%%)" % (concordi, 100 * concordi / len(multi)))
    print("  attesi sotto indipendenza     : %.1f (%.0f%%)"
          % (attesi, 100 * attesi / len(multi)))
    se = math.sqrt(attesi * (1 - attesi / len(multi)))
    z = (concordi - attesi) / se if se else 0.0
    print("  eccesso                       : %+.1f problemi, z = %.2f" % (concordi - attesi, z))
    print()
    if z > 2:
        print("  >>> H2 SOPRAVVIVE: l'identita' del problema porta segnale trasferibile")
        print("      fra modelli, ed e' la sola leva disponibile sull'88%% cieco.")
    else:
        print("  >>> H2 non supera 2 sigma.")

    # ---------------------------------------------------- quanto vale sapere il problema
    print("\n" + "-" * 78)
    print("QUANTO VALEREBBE conoscere il comportamento degli altri modelli sul problema")
    print("-" * 78)
    giuste = tot_p = 0
    for prob, v in per_prob.items():
        for i in range(len(v)):
            altri = v[:i] + v[i + 1:]
            if not altri:
                continue
            voto = 1 if sum(altri) * 2 > len(altri) else 0
            giuste += (voto == v[i])
            tot_p += 1
    if tot_p:
        print("  voto di maggioranza degli ALTRI modelli sullo stesso problema (leave-one-out):")
        print("     %d/%d = %.4f" % (giuste, tot_p, giuste / tot_p))
        print("  contro la maggioranza globale %.4f" % max(p, 1 - p))


if __name__ == "__main__":
    main()
