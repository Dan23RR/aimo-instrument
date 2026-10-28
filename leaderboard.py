"""Forense della classifica: cosa si puo' misurare di una gara senza spedirle nulla.

Dati OSSERVATI il 29 agosto 2026 sulla scheda Results della competizione Codabench
16180 (AIMO Interpretability Challenge, NeurIPS 2026), fase Warmup, track "Main",
task "AIMO Robustness Validation v8". 26 righe totali, 19 con un punteggio numerico.

Il metodo e' quello di ARMA: non si misura il concorrente, si misura lo strumento.
Qui lo strumento e' la classifica stessa, e l'unico ingresso e' cio' che pubblica.

Livelli di evidenza, mai mescolati:
  OSSERVATO  i punteggi, letti dalla pagina pubblica.
  DERIVATO   la taglia del set, il pavimento, l'errore, il fabbisogno di item.
             Seguono dall'aritmetica: nessuna soglia arbitraria.
  IPOTESI    l'identita' dei predittori costanti. Falsificabile con una submission.
"""
import math
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ARMA"))
from carico import capacita_corretta, amplificazione  # noqa: E402

# ------------------------------------------------------------------ OSSERVATO

# (utente, data, accuratezza). None = "n/a" sulla pagina.
CLASSIFICA = [
    ("participant-002",         "2026-08-01", None),
    ("participant-003", "2026-08-01", None),
    ("participant-005",      "2026-08-05", None),
    ("participant-180",          "2026-08-07", None),
    ("participant-021",  "2026-08-27", 0.65),
    ("participant-014",   "2026-08-20", 0.65),
    ("participant-017",      "2026-08-24", 0.59),
    ("stefanik12",    "2026-08-25", 0.53),   # organizzatore della gara
    ("participant-019",    "2026-08-26", 0.53),
    ("participant-020",  "2026-08-26", 0.53),
    ("participant-024",   "2026-08-29", 0.53),
    ("participant-022",         "2026-08-28", 0.47),
    ("participant-025",         "2026-08-28", 0.47),
    ("participant-023",     "2026-08-28", 0.47),
    ("participant-015",       "2026-08-20", 0.47),
    ("participant-016", "2026-08-23", 0.47),
    ("participant-018",    "2026-08-25", 0.47),
    ("participant-008",         "2026-08-10", 0.41),
    ("participant-009",    "2026-08-12", 0.41),
    ("participant-010",         "2026-08-12", 0.41),
    ("participant-011",    "2026-08-13", 0.41),
    ("participant-179",      "2026-08-14", 0.41),
    ("participant-004",       "2026-08-04", None),
    ("participant-007",         "2026-08-08", None),
    ("participant-001", "2026-07-30", None),
    ("participant-093",         "2026-08-19", 0.0),
]

PUNTEGGI = [a for _, _, a in CLASSIFICA if a is not None]
DISTINTI = sorted(set(PUNTEGGI), reverse=True)


# ------------------------------------------------------------------ DERIVATO

def denominatori_compatibili(valori, dmax=400):
    """Ogni d per cui OGNI punteggio pubblicato e' k/d arrotondato a 2 decimali."""
    ok = []
    for d in range(2, dmax + 1):
        reticolo = {round(k / d, 2) for k in range(d + 1)}
        if all(v in reticolo for v in valori):
            ok.append(d)
    return ok


def coda_binomiale(k, n, p=0.5):
    """P(X >= k) con X ~ Binomiale(n, p). Esatta, senza librerie."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def item_necessari(eps, pavimento):
    """n >= 0.25 / (eps * (1 - f))^2 — errore binomiale nel caso peggiore,
    amplificato dal pavimento. Stesso risultato di ARMA/PORTATA.md."""
    return math.ceil(0.25 / ((eps * (1 - pavimento)) ** 2))


def main():
    print("=" * 78)
    print("CARTA DI MISURA DELLA CLASSIFICA — AIMO Interpretability Challenge")
    print("Codabench 16180, fase Warmup, track Main, task 'AIMO Robustness Validation v8'")
    print("letta il 29 agosto 2026")
    print("=" * 78)

    n_tot = len(CLASSIFICA)
    n_num = len(PUNTEGGI)
    print("\n[OSSERVATO] %d righe in classifica, %d con punteggio, %d con 'n/a'."
          % (n_tot, n_num, n_tot - n_num))
    print("[OSSERVATO] valori distinti pubblicati: %s"
          % ", ".join("%.2f" % v for v in DISTINTI))
    conteggi = Counter(PUNTEGGI)
    for v in DISTINTI:
        print("             %.2f  x%d" % (v, conteggi[v]))

    # --- 1. quanti item ha davvero il set di valutazione
    print("\n" + "-" * 78)
    print("1. LA TAGLIA DEL SET DI VALUTAZIONE  [DERIVATO]")
    print("-" * 78)
    comp = denominatori_compatibili(DISTINTI)
    n = comp[0]
    print("  Denominatori d per cui ogni punteggio pubblicato e' k/d:")
    print("    %s%s" % (", ".join(str(d) for d in comp[:8]),
                        " ..." if len(comp) > 8 else ""))
    print("  Il piu' piccolo, e l'unico parsimonioso, e' n = %d." % n)
    print()
    print("  Verifica, punteggio per punteggio:")
    for v in DISTINTI:
        k = round(v * n)
        print("    %.2f = %2d/%d = %.4f" % (v, k, n, k / n))
    print()
    reticolo = sorted({round(k / n, 2) for k in range(n + 1)})
    copertura = len(reticolo) / 101.0
    print("  Il reticolo 1/%d copre %d dei 101 valori possibili a due decimali (%.1f%%)."
          % (n, len(reticolo), 100 * copertura))
    print("  Se il set fosse grande, i %d valori distinti osservati cadrebbero tutti" % len(DISTINTI))
    print("  su quel reticolo con probabilita' circa %.1e. Non e' un caso: il set ha %d item."
          % (copertura ** len(DISTINTI), n))
    print()
    print("  RISOLUZIONE DELLA CLASSIFICA: 1 item = %.2f punti percentuali." % (100.0 / n))

    # --- 2. il pavimento, e chi ci sta sopra
    print("\n" + "-" * 78)
    print("2. IL PAVIMENTO  [DERIVATO da 1, piu' IPOTESI sui predittori costanti]")
    print("-" * 78)
    print("  Due gruppi complementari dominano la classifica:")
    print("    0.53 = %d/%d   (x%d, fra cui l'organizzatore stesso)" % (round(0.53 * n), n, conteggi[0.53]))
    print("    0.47 = %d/%d   (x%d)" % (round(0.47 * n), n, conteggi[0.47]))
    print("    somma esatta: %d/%d + %d/%d = %d/%d = 1.000"
          % (round(0.53 * n), n, round(0.47 * n), n, n, n))
    print()
    print("  IPOTESI: sono i due predittori costanti. Il repo getting-started")
    print("  distribuisce 'always-true' come soluzione di esempio, quindi e' la prima")
    print("  cosa che si spedisce per collaudare la pipeline. Se e' cosi', il set")
    print("  contiene %d robust e %d spurious, e il pavimento e' la classe di maggioranza."
          % (round(0.53 * n), round(0.47 * n)))
    f = round(0.53 * n) / n
    print()
    print("  PAVIMENTO f = %d/%d = %.4f" % (round(0.53 * n), n, f))
    print("  Amplificazione dell'errore 1/(1-f) = %.3fx" % amplificazione(f))
    print()
    print("  FALSIFICABILE IN UN COLPO: spedire 'always-spurious' deve restituire")
    print("  esattamente %.2f. Se restituisce altro, questa sezione e' morta." % (round(0.47 * n) / n))

    # --- 3. la classifica regge il proprio ordinamento?
    print("\n" + "-" * 78)
    print("3. LA CLASSIFICA DISTINGUE I CONCORRENTI?  [DERIVATO]")
    print("-" * 78)
    top = max(PUNTEGGI)
    k_top = round(top * n)
    p_caso = coda_binomiale(k_top, n, 0.5)
    attesi = n_num * p_caso
    almeno_uno = 1 - (1 - p_caso) ** n_num
    print("  Punteggio di testa: %.2f = %d/%d." % (top, k_top, n))
    print("  P(>= %d/%d lanciando una moneta) = %.4f" % (k_top, n, p_caso))
    print("  Su %d partecipanti che tirassero a caso, quanti arriverebbero a %d/%d?"
          % (n_num, k_top, n))
    print("     attesi %.1f     osservati %d" % (attesi, conteggi[top]))
    print("  P(almeno uno di %d ci arrivi per caso) = %.3f" % (n_num, almeno_uno))
    print()
    print("  Anche col pavimento come riferimento invece della moneta:")
    p_pav = coda_binomiale(k_top, n, f)
    print("     P(>= %d/%d | p = pavimento %.3f) = %.4f" % (k_top, n, f, p_pav))
    print()
    print("  L'intera distribuzione osservata (%.2f-%.2f) sta dentro la dispersione"
          % (min(v for v in PUNTEGGI if v > 0), max(PUNTEGGI)))
    se_caso = math.sqrt(0.25 / n)
    print("  binomiale attorno a 0.50, che su %d item vale +/- %.1f punti (1 sigma)."
          % (n, 100 * se_caso))

    # --- 4. capacita' corretta del primo, con la sua incertezza
    print("\n" + "-" * 78)
    print("4. QUANTA CAPACITA' HA DAVVERO IL PRIMO  [DERIVATO]")
    print("-" * 78)
    k = capacita_corretta(top, f)
    se_grezzo = math.sqrt(top * (1 - top) / n)
    se_k = se_grezzo * amplificazione(f)
    print("  grezzo            %.3f  (%d/%d)" % (top, k_top, n))
    print("  pavimento         %.3f  (%d/%d)" % (f, round(f * n), n))
    print("  capacita' k=(s-f)/(1-f)   %.3f" % k)
    print("  errore standard grezzo    +/- %.3f" % se_grezzo)
    print("  amplificato dal pavimento +/- %.3f" % se_k)
    print()
    print("  CAPACITA' DEL PRIMO IN CLASSIFICA: %.2f +/- %.2f (1 sigma)" % (k, se_k))
    print("  intervallo al 95%%: [%.2f, %.2f]" % (k - 1.96 * se_k, k + 1.96 * se_k))
    print("  -> contiene lo zero. Il primo non e' distinguibile dal pavimento.")

    # --- 5. cosa servirebbe
    print("\n" + "-" * 78)
    print("5. QUANTI ITEM SERVIREBBERO  [DERIVATO: solo binomiale + pavimento]")
    print("-" * 78)
    print("  %-22s %-16s %-16s" % ("precisione voluta", "senza pavimento", "con pav. %.2f" % f))
    for eps in (0.20, 0.10, 0.05, 0.02):
        n0 = math.ceil(0.25 / eps ** 2)
        n1 = item_necessari(eps, f)
        print("  +/- %-18.0f %-16d %-16d" % (100 * eps, n0, n1))
    print()
    print("  La classifica ha %d item." % n)
    print("  Per risolvere +/- 10 punti di capacita' ne servirebbero %d: %.0fx tanti."
          % (item_necessari(0.10, f), item_necessari(0.10, f) / n))
    print("  Per risolvere +/- 5 punti ne servirebbero %d: %.0fx tanti."
          % (item_necessari(0.05, f), item_necessari(0.05, f) / n))

    print("\n" + "=" * 78)
    print("SINTESI")
    print("=" * 78)
    print("  %d partecipanti iscritti, %d classificati, %d item di valutazione." % (89, n_tot, n))
    print("  Un item vale %.1f punti. Fra il primo e il diciottesimo ci sono %d item."
          % (100.0 / n, k_top - round(0.41 * n)))
    print("  Il punteggio di testa e' raggiunto a caso %.1f volte su 100." % (100 * p_caso))
    print("  La capacita' del primo, corretta per il pavimento, e' %.2f +/- %.2f." % (k, se_k))
    print("  Nessun ordinamento in questa classifica e' statisticamente sostenuto.")

    print("\n  Caveat dichiarati:")
    print("   - i punteggi sono arrotondati a 2 decimali dalla piattaforma;")
    print("   - il task e' etichettato 'v8': se il set fosse cambiato nel tempo,")
    print("     il reticolo sarebbe confuso. Le righe dal 10 al 29 agosto cadono")
    print("     tutte sullo stesso reticolo 1/%d, quindi in quella finestra e' stabile;" % n)
    print("   - l'identita' dei costanti e' IPOTESI finche' non la si spedisce;")
    print("   - questa e' la fase Warmup. Il set della fase principale puo' essere")
    print("     piu' grande. Ma la fase principale NON e' ancora configurata su")
    print("     Codabench, e il Warmup chiude il 1 settembre 2026 alle 01:55 CEST.")


if __name__ == "__main__":
    main()
