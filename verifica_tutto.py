"""Verifica avversaria di ogni numero prodotto finora. Cerca errori, non conferme.

Ricalcola da zero: taglia del set, regola della label, composizione, i cinque esiti
pre-registrati, e mette alla prova due affermazioni che potrebbero essere sbagliate:
  (A) "i 2 item del modello in cache sono 1 robust e 1 spurious"  (OSSERVATO_907677)
  (B) il guadagno del canale-problema, che avevo dato prima a +12,5 e poi a +3,4
"""
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"


def carica(n):
    return json.loads((CACHE / ("%s.json" % n)).read_text(encoding="utf-8"))


def ok(cond):
    return "OK " if cond else "!! ERRORE"


print("=" * 78)
print("VERIFICA AVVERSARIA — 29 agosto 2026")
print("=" * 78)

# ---------------------------------------------------------------- 1. taglia e punteggi
print("\n[1] TAGLIA DEL SET E PUNTEGGI OSSERVATI")
s_af = 0.4705882353     # always-false, submission 907310
s_bl = 0.5294117647     # baseline strumentata, 907677
print("  always-false  = %.10f  -> x*17 = %.6f  (intero? %s)"
      % (s_af, s_af * 17, ok(abs(s_af * 17 - round(s_af * 17)) < 1e-6)))
print("  baseline      = %.10f  -> x*17 = %.6f  (intero? %s)"
      % (s_bl, s_bl * 17, ok(abs(s_bl * 17 - round(s_bl * 17)) < 1e-6)))
print("  always-false = %d/17, baseline = %d/17" % (round(s_af * 17), round(s_bl * 17)))
print("  -> 8 spurious, 9 robust nel set. %s" % ok(round(s_af * 17) == 8))

# ---------------------------------------------------------------- 2. regola label
print("\n[2] REGOLA DELLA LABEL (deve essere 100%% su ogni set etichettato)")
etichettati = ["val-sample", "augmented-sample-math-agg",
               "augmented-sample-math-agg-filtered", "augmented-sample-math"]
tot = giuste = 0
for nome in etichettati:
    for r in carica(nome):
        lab = r.get("model_is_robust")
        if lab is None:
            continue
        atteso = lab in (True, "True")
        regola = (int(r.get("n_detrimental_permutations") or 0) == 0)
        tot += 1
        giuste += (regola == atteso)
print("  n_detrimental==0 riproduce la label: %d/%d  %s" % (giuste, tot, ok(giuste == tot)))

# tabella aggregata
grezze = []
for s in ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
          "augmented-sample-math-agg-filtered", "augmented-sample-math",
          "augmented-sample-math-full-filtered"]:
    grezze += carica(s)
agg = defaultdict(list)
for r in grezze:
    agg[(r["model_id"], r["problem_id"])].append(r)
tab = {}
for k, rr in agg.items():
    detr = sum(int(x.get("n_detrimental_permutations") or 0) for x in rr)
    tab[k] = {"robust": detr == 0, "dataset": rr[0].get("dataset_id"),
              "ba": [float(x["base_accuracy"]) for x in rr if x.get("base_accuracy") is not None]}
print("  coppie (modello,problema) totali: %d" % len(tab))

# ---------------------------------------------------------------- 3. AFFERMAZIONE (A)
print("\n[3] AFFERMAZIONE DA FALSIFICARE: i 2 item in cache sono 1 robust + 1 spurious")
print("  Metodo: enumero tutte le composizioni (cache_robust, blind_robust) compatibili")
print("  con i due punteggi osservati, senza assumere nulla.")
# cache = 2 item; blind = 15 item; totale robust = 9
# always-false: predice tutto spurious -> corretti = num spurious totali = 8  (vincolo gia' usato)
# baseline: predice 1 positivo tra i cache, 0 tra i blind. punteggio 9.
#   corretti = (blind spurious) + (cache: se il positivo e' su un robust: +1; e l'altro cache se spurious: +1)
compat = []
for cache_r in range(0, 3):                 # robust tra i 2 in cache
    blind_r = 9 - cache_r                    # robust tra i 15 blind
    if not (0 <= blind_r <= 15):
        continue
    cache_s = 2 - cache_r
    blind_s = 15 - blind_r
    # always-false: corretti = tutti gli spurious = cache_s + blind_s = 8?
    if cache_s + blind_s != 8:
        continue
    # baseline: 1 positivo tra i cache, resto negativo.
    # il positivo e' corretto sse c'e' almeno un robust in cache; assumo il modello lo metta li'.
    # corretti_baseline = blind_s + (1 se il positivo cade su un robust) + (cache negativi corretti)
    # con 1 predizione positiva su 2 cache: l'altro cache e' predetto negativo.
    #   se cache = (1R,1S): positivo su R (+1), negativo su S (+1) -> cache corretti 2
    #   se cache = (2R,0S): positivo su R (+1), negativo su R (0)  -> cache corretti 1
    #   se cache = (0R,2S): positivo su S (0),  negativo su S (+1) -> cache corretti 1
    if cache_r == 1:
        cache_corretti = 2
    elif cache_r == 2:
        cache_corretti = 1
    else:
        cache_corretti = 1
    baseline_corretti = blind_s + cache_corretti
    if baseline_corretti == 9:
        compat.append((cache_r, cache_s, blind_r, blind_s, baseline_corretti))
print("  composizioni compatibili con ENTRAMBI i punteggi:")
for cr, cs, br, bs, sc in compat:
    print("    cache = %dR+%dS, blind = %dR+%dS  -> baseline %d/17" % (cr, cs, br, bs, sc))
unico = len(compat) == 1
print("  -> la composizione dei 2 item in cache e' univocamente determinata? %s"
      % ("SI" if unico else "NO, ci sono %d possibilita'" % len(compat)))
if not unico:
    print("  !! CORREZIONE NECESSARIA in OSSERVATO_907677.md: NON e' determinata.")

# ---------------------------------------------------------------- 4. canale-problema
print("\n[4] GUADAGNO DEL CANALE-PROBLEMA, contro la baseline giusta (per modello)")
per_mod = defaultdict(dict)
for (m, p), v in tab.items():
    per_mod[m][p] = v["robust"]
tasso = {m: sum(d.values()) / len(d) for m, d in per_mod.items()}
per_prob = defaultdict(list)
for m, d in per_mod.items():
    for p, rb in d.items():
        per_prob[p].append((m, rb))
multi = {p: v for p, v in per_prob.items() if len(v) >= 2}
g = mg_mod = n = 0
for p, v in multi.items():
    for i, (m, rb) in enumerate(v):
        altri = [x for j, x in enumerate(v) if j != i]
        voto = sum(1 for _, x in altri if x) * 2 > len(altri)
        g += (voto == rb)
        mg_mod += ((tasso[m] > 0.5) == rb)
        n += 1
soglia = math.sqrt(0.25 / n)
print("  n predizioni leave-one-out: %d" % n)
print("  voto multi-modello sul problema : %.4f" % (g / n))
print("  maggioranza per modello (nota dal model_id) : %.4f" % (mg_mod / n))
print("  guadagno vero: %+.1f punti,  soglia %.1f punti  -> %s"
      % (100 * (g - mg_mod) / n, 100 * soglia, ok((g - mg_mod) / n >= soglia)))
print("  (il +12,5 iniziale era contro la maggioranza GLOBALE e su meno dati: era gonfiato)")

# ---------------------------------------------------------------- 5. tetto interpretabilita'
print("\n[5] TETTO DELL'INTERPRETABILITA' SUL WARMUP")
isp = 2
guad = isp / (2 * 17)
rumore = math.sqrt(0.25 / 17)
print("  item ispezionabili: %d/17. guadagno max %.3f, rumore %.3f, rapporto %.2f sigma  %s"
      % (isp, guad, rumore, guad / rumore, ok(abs(guad / rumore - 0.49) < 0.02)))

print("\n" + "=" * 78)
print("VERDETTO")
print("=" * 78)
print("  n=17: confermato a precisione piena da due punteggi indipendenti.")
print("  regola label: esatta su %d righe." % tot)
print("  composizione cache: %s" % ("determinata" if unico else "NON determinata (correggo il file)"))
print("  canale-problema: +%.1f punti, reale ma piccolo." % (100 * (g - mg_mod) / n))
