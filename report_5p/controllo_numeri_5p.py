"""Controllo meccanico dei numeri del report da 5 pagine (28/10/2026).

Uso (dalla cartella report_5p): python controllo_numeri_5p.py
Prende ogni numero del testo principale da 5 pagine (dal riassunto alla fine della dichiarazione
sull'uso dell'IA) e controlla che compaia anche nella bozza da 9 pagine verificata. Stampa i numeri
che non vi compaiono: ognuno va giustificato da un file su disco (per esempio gli ID degli invii
del 28/10) o corretto. Controlla anche che nel testo non ci siano lineette lunghe o medie.
"""
import io
import os
import re

QUI = os.path.dirname(os.path.abspath(__file__))
cinque = io.open(os.path.join(QUI, "REPORT_5P_bozza_2026-10-28.md"), encoding="utf-8").read()
nove = io.open(os.path.join(QUI, "..", "report_9p", "REPORT_9P_bozza_2026-10-22.md"), encoding="utf-8").read()

testo = cinque[cinque.index("## Abstract"):cinque.index("## Acknowledgments")]
NUM = re.compile(r"(?<![\w.])[−+]?\d[\d,]*(?:\.\d+)?%?")


def numeri(t):
    return [m.group(0).rstrip(",") for m in NUM.finditer(t)]


in_nove = set(numeri(nove))
mancanti = {}
for n in numeri(testo):
    if n not in in_nove:
        mancanti[n] = mancanti.get(n, 0) + 1
print("numeri nel testo da 5 pagine:", len(numeri(testo)), "| non presenti nella bozza da 9 pagine:", len(mancanti))
for n, c in sorted(mancanti.items()):
    i = testo.index(n)
    print("  %-10s x%d  ...%s..." % (n, c, " ".join(testo[max(0, i - 60):i + 40].split())))
for ch, nome in (("—", "lineetta lunga"), ("–", "lineetta media")):
    k = cinque.count(ch)
    print(nome + ":", k)
