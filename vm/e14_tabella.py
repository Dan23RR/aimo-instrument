"""E14: tutte le cifre della sezione 8.1 del report, dagli output salvati (23/9/2026).

Input: vm_risultati/e14/out/e14_C*_rep*_gpu*_*.jsonl (23 esecuzioni x 40 problemi DeepSeek) e i
log di esecuzione nell'archivio della VM (tempi per item). Etichette: feature_greedy.parquet.
Condizioni: C0 default (batch 8), C1 batch 1, C2 flag deterministici, C3 due processi sulla stessa
scheda, C4 seconda scheda, C5 senza compilazione.

Uso (dalla cartella AIMO): python vm/e14_tabella.py
"""
import glob
import itertools
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "vm_risultati", "e14", "out")
LOG = [os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", f) for f in ("e14_all.log", "e14_extra.log")]
FG = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "gpu", "out", "feature_greedy.parquet")
NOMI = {"C0": "default (batch 8)", "C1": "batch 1", "C2": "flag deterministici", "C3": "contesa (2 processi)",
        "C4": "seconda scheda", "C5": "senza compilazione"}


def carica():
    runs = {}
    for f in sorted(glob.glob(os.path.join(OUT, "e14_C*_rep*_gpu*_*.jsonl"))):
        nome = os.path.basename(f)[4:-6]
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        runs[nome] = {r["problem_id"]: r for r in rows}
    return runs


def attraversamenti(runs, nomi):
    pids = sorted(runs[nomi[0]])
    return sum(1 for p in pids if len({bool(runs[n][p]["robust_pred"]) for n in nomi}) > 1)


def main():
    runs = carica()
    lab = pd.read_parquet(FG).set_index("problem_id").robust
    pids = sorted(next(iter(runs.values())))
    print("esecuzioni %d, problemi %d, robust veri %d" % (len(runs), len(pids), int(sum(bool(lab[p]) for p in pids))))
    firme = {n: tuple(runs[n][p]["sha256_testo"] for p in pids) for n in runs}
    stati = defaultdict(list)
    for n, fi in firme.items():
        stati[fi].append(n)
    print("stati numerici distinti: %d" % len(stati))
    for i, (fi, ns) in enumerate(sorted(stati.items(), key=lambda kv: kv[1])):
        print("   stato %2d: %s" % (i, ", ".join(ns)))
    lista = list(stati)
    condivise = []
    for a, b in itertools.combinations(range(len(lista)), 2):
        condivise.append(sum(1 for x, y in zip(lista[a], lista[b]) if x == y))
    c = np.array(condivise)
    print("coppie di stati %d: nessuna generazione comune %d, 1-2 comuni %d, fra 3 e 10 %d, oltre 10 %d | valori sopra 2: %s" % (
        len(c), int((c == 0).sum()), int(((c >= 1) & (c <= 2)).sum()), int(((c >= 3) & (c <= 10)).sum()), int((c > 10).sum()),
        sorted(int(x) for x in c if x > 2)))

    tempi = {}
    for f in LOG:
        for l in open(f, encoding="utf-8", errors="replace"):
            m = re.search(r"fine\s+(C\d) rep(\d) ([ab]) -> gen (\d+)s = ([\d.]+) s/item", l)
            if m:
                tempi["%s_rep%s_%s" % (m.group(1), m.group(2), m.group(3))] = float(m.group(5))
    print("\n%-22s %4s %12s %13s %-24s %-24s %6s" % ("condizione", "run", "identici/40", "attraversano", "robust per run", "corrette per run", "s/item"))
    per_cond = defaultdict(list)
    for n in sorted(runs):
        per_cond[n[:2]].append(n)
    medie = {}
    for cnd in sorted(per_cond):
        ns = per_cond[cnd]
        ident = sum(1 for i, p in enumerate(pids) if len({firme[n][i] for n in ns}) == 1)
        att = attraversamenti(runs, ns)
        rob = [sum(bool(runs[n][p]["robust_pred"]) for p in pids) for n in ns]
        corr = [sum(bool(runs[n][p]["robust_pred"]) == bool(lab[p]) for p in pids) for n in ns]
        t = [tempi[k] for k in tempi if k.startswith(cnd + "_")]
        medie[cnd] = np.mean(t)
        print("%-22s %4d %12s %13d %-24s %-24s %6.1f" % (NOMI[cnd], len(ns), "%d/40" % ident, att, ",".join(map(str, rob)), ",".join(map(str, corr)), np.mean(t)))
    print("costanti: sempre non robust %d/40, sempre robust %d/40" % (sum(not bool(lab[p]) for p in pids), sum(bool(lab[p]) for p in pids)))
    print("costo rispetto al default: batch 1 %+.1f%%, flag deterministici %+.1f%%, senza compilazione %+.1f%%, contesa %+.1f%%" % (
        100 * (medie["C1"] / medie["C0"] - 1), 100 * (medie["C2"] / medie["C0"] - 1), 100 * (medie["C5"] / medie["C0"] - 1), 100 * (medie["C3"] / medie["C0"] - 1)))

    print("\nattraversamenti a parita' di numero di esecuzioni:")
    for cnd, k in (("C3", 5), ("C0", 3), ("C3", 3), ("C3", 2), ("C0", 2), ("C1", 2), ("C4", 2)):
        vals = [attraversamenti(runs, list(s)) for s in itertools.combinations(per_cond[cnd], k)]
        print("   %-22s su %d esecuzioni: min %d, max %d, media %.1f (%d combinazioni)" % (NOMI[cnd], k, min(vals), max(vals), np.mean(vals), len(vals)))

    print("\ncoppie di contesa (stesso istante, stessa scheda), generazioni identiche:")
    for r in ("0", "1", "2"):
        a, b = "C3_rep%s_gpu0_a" % r, "C3_rep%s_gpu0_b" % r
        print("   rep%s: %d/40" % (r, sum(1 for i in range(len(pids)) if firme[a][i] == firme[b][i])))
    c2, c5 = "C2_rep0_gpu0_a", "C5_rep0_gpu1_a"
    stesse = sum(1 for p in pids if bool(runs[c2][p]["robust_pred"]) == bool(runs[c5][p]["robust_pred"]))
    comuni = sum(1 for i in range(len(pids)) if firme[c2][i] == firme[c5][i])
    print("punti fissi C2 e C5: robust %d e %d; generazioni comuni %d/40; stessa predizione %d/40" % (
        sum(bool(runs[c2][p]["robust_pred"]) for p in pids), sum(bool(runs[c5][p]["robust_pred"]) for p in pids), comuni, stesse))
    rif = firme["C0_rep0_gpu0_a"]
    for n in sorted(per_cond["C4"]):
        print("   %s nello stato di C0 rep0: %s" % (n, "si" if firme[n] == rif else "no"))
    sc = [abs(runs["C0_rep0_gpu0_a"][p]["score"] - runs["C0_rep3_gpu0_a"][p]["score"]) for p in pids]
    ecc = []
    for p in pids:
        s = [runs[n][p]["score"] for n in per_cond["C0"]]
        ecc.append(max(s) - min(s))
    print("escursione per item nel default (max - min sui 5 run): mediana %.3f | soglia %.3f" % (np.median(ecc), 0.576))


if __name__ == "__main__":
    main()
