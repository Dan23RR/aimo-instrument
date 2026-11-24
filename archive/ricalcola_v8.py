"""Ricalcola i due parametri di V8 dai dataset pubblici (4/11/2026, per il README del rilascio aimo-instrument).

Uso (dalla cartella del repository, dopo `python dati.py` e `python aggrega.py`): python ricalcola_v8.py
Ripete i calcoli di vm/controlli_verifica3.py (sezione 8: la soglia) e di vm/dati_pubblici_controlli.py
(sezione 6: il verso), che pero' leggono anche file della copia della macchina virtuale e file degli
organizzatori non inclusi nel rilascio. Uscite del 23/9 da confrontare: vm_risultati/controlli_verifica3_2026-09-23.txt
(mediana inferiore 98) e vm_risultati/dati_pubblici_controlli_2026-09-23.txt (0.65245 contro 0.34755 sulle 335
coppie, 0.63057 contro 0.36943 sulle 283 distinte). Solo libreria standard.
"""
import json
import os

QUI = os.path.dirname(os.path.abspath(__file__))
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

tab = json.load(open(os.path.join(QUI, "dati", "tabella_aggregata.json"), encoding="utf-8"))
assert all("problema" in r for r in tab), \
    "dati/tabella_aggregata.json has no problem texts: run python dati.py, then python aggrega.py"

# soglia: mediana inferiore dei caratteri dei problemi distinti (vm/controlli_verifica3.py, sezione 8)
lun = sorted({r["problem_id"]: len(r["problema"]) for r in tab}.values())
soglia = lun[len(lun) // 2 - 1]
print("distinct problems %d | central character counts %d and %d | lower median, the threshold: %d"
      % (len(lun), lun[len(lun) // 2 - 1], lun[len(lun) // 2], soglia))

# verso: accuratezza bilanciata della regola nei due versi (vm/dati_pubblici_controlli.py, sezione 6);
# le 52 righe DeepSeek che ripetono i problemi di qwen3-8b:low (lo stesso modello) contano una volta sola
dup = {r["problem_id"] for r in tab if r["model_id"] == "qwen3-8b:low"}
distinte = [r for r in tab if not (r["model_id"] == DEEPSEEK and r["problem_id"] in dup)]
for nome, rr in (("%d published pairs" % len(tab), tab), ("%d distinct pairs" % len(distinte), distinte)):
    for verso in (1, -1):
        tp = sum(1 for r in rr if r["robust"] and (len(r["problema"]) >= soglia) == (verso == 1))
        fn = sum(1 for r in rr if r["robust"]) - tp
        tn = sum(1 for r in rr if not r["robust"] and (len(r["problema"]) >= soglia) != (verso == 1))
        fp = sum(1 for r in rr if not r["robust"]) - tn
        print("%-20s threshold %d, %-24s balanced accuracy %.5f"
              % (nome, soglia, "longer problems robust:" if verso == 1 else "shorter problems robust:",
                 0.5 * (tp / (tp + fn) + tn / (tn + fp))))
