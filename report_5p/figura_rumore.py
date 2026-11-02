"""Figura 1 del report da 5 pagine (2/11/2026): le decisioni della pipeline ufficiale nelle 23 esecuzioni di E14.

Legge gli stessi file di ../vm/e14_tabella.py (23 esecuzioni x 40 problemi DeepSeek, tempi dai log della VM),
ma le etichette da ../gpu/out/feature_greedy.parquet, controlla che le cifre coincidano con quelle del report
(Section 4 e Appendice J) e scrive latex/fig_rumore.pdf (vettoriale) e latex/fig_rumore.png (anteprima).

Uso (dalla cartella report_5p): python figura_rumore.py
"""
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

QUI = os.path.dirname(os.path.abspath(__file__))
AIMO = os.path.dirname(QUI)
OUT = os.path.join(AIMO, "vm_risultati", "e14", "out")
VM = os.path.join(AIMO, "vm_backup", "estratto", "work", "aimo")
LOG = [os.path.join(VM, f) for f in ("e14_all.log", "e14_extra.log")]
FG = os.path.join(AIMO, "gpu", "out", "feature_greedy.parquet")  # stesse etichette della copia nella VM, ed e' nel repository

# Ordine e nomi come nella Section 4 del report; cifre attese: (identici, attraversano, secondi per item).
CONDIZIONI = [
    ("C0", "Default (batch 8)", 0, 24, 26.0),
    ("C1", "Batch size 1", 1, 20, 72.3),
    ("C3", "Two processes on one card", 0, 32, 54.7),
    ("C4", "Second physical card", 0, 19, 26.0),
    ("C2", "Deterministic algorithms", 40, 0, 27.4),
    ("C5", "Compilation disabled", 40, 0, 27.8),
]
ROBUST = "#256abf"      # blu, passo 500 della rampa sequenziale di riferimento
NON_ROBUST = "#e3e6ea"  # grigio chiaro neutro
INCHIOSTRO = "#1f1f1f"
SECONDARIO = "#52514e"

runs = {}
for f in sorted(glob.glob(os.path.join(OUT, "e14_C*_rep*_gpu*_*.jsonl"))):
    nome = os.path.basename(f)[4:-6]
    runs[nome] = {r["problem_id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8") if l.strip())}
assert len(runs) == 23, len(runs)
pids = sorted(next(iter(runs.values())))
assert len(pids) == 40 and all(sorted(r) == pids for r in runs.values())
lab = pd.read_parquet(FG).set_index("problem_id").robust
vero = {p: bool(lab[p]) for p in pids}
assert sum(vero.values()) == 7

firme = {n: tuple(runs[n][p]["sha256_testo"] for p in pids) for n in runs}
assert len(set(firme.values())) == 14, "stati numerici attesi: 14"

tempi = {}
for f in LOG:
    for l in open(f, encoding="utf-8", errors="replace"):
        m = re.search(r"fine\s+(C\d) rep(\d) ([ab]) -> gen (\d+)s = ([\d.]+) s/item", l)
        if m:
            tempi["%s_rep%s_%s" % (m.group(1), m.group(2), m.group(3))] = float(m.group(5))

righe, gruppi = [], []
for cnd, etichetta, ident_atteso, att_atteso, sec_attesi in CONDIZIONI:
    ns = sorted(n for n in runs if n.startswith(cnd + "_"))
    ident = sum(1 for i in range(len(pids)) if len({firme[n][i] for n in ns}) == 1)
    att = sum(1 for p in pids if len({bool(runs[n][p]["robust_pred"]) for n in ns}) > 1)
    t = [tempi[k] for k in tempi if k.startswith(cnd + "_")]
    sec = sum(t) / len(t)
    assert (ident, att, round(sec, 1)) == (ident_atteso, att_atteso, sec_attesi), (cnd, ident, att, sec)
    gruppi.append((etichetta, len(righe), len(ns), att, ident, sec))
    righe += ns

# Colonne: prima i problemi predetti robusti piu' spesso; a parita', prima i robusti veri.
quante = {p: sum(bool(runs[n][p]["robust_pred"]) for n in runs) for p in pids}
colonne = sorted(pids, key=lambda p: (-quante[p], not vero[p], p))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                     "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(5.5, 2.6))
ax = fig.add_axes([0.235, 0.015, 0.535, 0.87])
SALTO = 0.9      # spazio fra gruppi, in altezze di riga
GAP = 0.14       # spazio bianco fra celle
y = 0.0
centri = []
for etichetta, inizio, n, att, ident, sec in gruppi:
    y0 = y
    for k in range(n):
        nome = righe[inizio + k]
        for j, p in enumerate(colonne):
            colore = ROBUST if runs[nome][p]["robust_pred"] else NON_ROBUST
            ax.add_patch(Rectangle((j + GAP / 2, -(y + 1) + GAP / 2), 1 - GAP, 1 - GAP, color=colore, lw=0))
        y += 1
    centri.append((etichetta, -(y0 + y) / 2, att, ident, sec))
    y += SALTO
y += 0.35
for j, p in enumerate(colonne):
    colore = ROBUST if vero[p] else NON_ROBUST
    ax.add_patch(Rectangle((j + GAP / 2, -(y + 1.25) + GAP / 2), 1 - GAP, 1.25 - GAP, color=colore, lw=0))
y_lab = -(y + 0.625)
y += 1.25
ax.set_xlim(0, len(colonne))
ax.set_ylim(-y - 0.1, 0.1)
ax.axis("off")

COLONNE_DESTRA = [(len(colonne) + 4.6, "items\ncrossing"), (len(colonne) + 10.2, "byte-\nidentical"),
                  (len(colonne) + 15.6, "seconds\nper item")]
for x, titolo in COLONNE_DESTRA:
    ax.text(x, 0.35, titolo, ha="right", va="bottom", color=SECONDARIO, fontsize=6.5, linespacing=1.0)
for etichetta, yc, att, ident, sec in centri:
    ax.text(-0.8, yc, etichetta, ha="right", va="center", color=INCHIOSTRO)
    for (x, _), valore in zip(COLONNE_DESTRA, ("%d" % att, "%d/40" % ident, "%.1f" % sec)):
        ax.text(x, yc, valore, ha="right", va="center", color=INCHIOSTRO)
ax.text(-0.8, y_lab, "Label (7 robust)", ha="right", va="center", color=INCHIOSTRO, fontweight="bold")
ax.text(len(colonne) + 0.8, y_lab, "dark: robust", ha="left", va="center", color=SECONDARIO)

fig.savefig(os.path.join(QUI, "latex", "fig_rumore.pdf"))
fig.savefig(os.path.join(QUI, "latex", "fig_rumore.png"), dpi=200)
print("figura scritta: %d esecuzioni, %d problemi, %d stati; colonne ordinate per robust predetto" % (
    len(righe), len(colonne), len(set(firme.values()))))
