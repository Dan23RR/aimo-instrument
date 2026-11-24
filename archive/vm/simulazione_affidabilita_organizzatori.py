"""Affidabilita' test-retest simulata del processo di etichettatura degli organizzatori.

Ricostruzione del 23/9/2026 della simulazione descritta in VERIFICA_classifica_2026-09-14.md
(script originale non salvato). Descrizione di allora: dai dati pubblici, che per ogni riga
riportano numero di predizioni e frequenza di successo, simulare il loro processo con la regola
«robust se base ok e tutte le varianti corrette», su 116 coppie con conteggi completi; esito
kappa 0,93 (IC90 0,87-1,00), tasso robust simulato 14,7% contro 13,8% osservato; 70,6% delle
righe con base_accuracy esattamente 0 o 1.

Insieme dei dati (ricostruito e verificato: 749 righe, 70,63% con base 0 o 1, 116 coppie):
righe con n_base_predictions >= 10 in aimo-interp-challenge-sample-full, val-sample e
augmented-sample-math-full-filtered.

Simulazione "plug-in": le frequenze osservate si trattano come probabilita' vere. Per ogni
coppia e replica: successi sulla base ~ Binomiale(n_base, base_accuracy); per ogni riga di
perturbazione, successi ~ Binomiale(n_permuted, permuted_accuracy); etichetta robust se la
base supera la soglia e tutte le predizioni perturbate sono corrette. Due repliche indipendenti
danno un kappa; 2000 coppie di repliche danno media e intervallo al 90%. Poiche' una coppia con
tutte le predizioni perturbate osservate corrette ha probabilita' 1 in simulazione, la
simulazione non la fa mai fallire: il valore e' ottimistico, non una misura.

Le definizioni di «base ok» provate sono elencate nell'output; quella che riproduce il tasso
osservato del 13,8% e' indicata.

Uso (dalla cartella AIMO): python vm/simulazione_affidabilita_organizzatori.py
"""
import json
import os
from collections import defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATI = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "dati")
FILE = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-full-filtered"]
N_REPLICHE, SEME = 2000, 20260914


def kappa(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def carica():
    righe = []
    for f in FILE:
        for r in json.load(open(os.path.join(DATI, f + ".json"), encoding="utf-8")):
            if int(r["n_base_predictions"]) >= 10:
                righe.append(r)
    coppie = defaultdict(list)
    for r in righe:
        coppie[(r["model_id"], r["problem_id"])].append(r)
    return righe, coppie


def prepara(coppie):
    out = []
    for k, rs in sorted(coppie.items()):
        nb = int(rs[0]["n_base_predictions"]); pb = float(rs[0]["base_accuracy"])
        pert = [(int(r["n_permuted_predictions"]), float(r["permuted_accuracy"])) for r in rs]
        out.append((k, nb, pb, pert))
    return out


def osservata(pb, pert, soglia):
    return (pb >= soglia) and all(abs(p - 1.0) < 1e-9 for _, p in pert)


def simula(dati, soglia, rng):
    lab = []
    for _, nb, pb, pert in dati:
        sb = rng.binomial(nb, min(max(pb, 0.0), 1.0)) / nb
        ok = sb >= soglia
        if ok:
            for n, p in pert:
                if rng.binomial(n, min(max(p, 0.0), 1.0)) < n:
                    ok = False
                    break
        lab.append(ok)
    return np.array(lab)


def main():
    righe, coppie = carica()
    dati = prepara(coppie)
    q01 = np.mean([float(r["base_accuracy"]) in (0.0, 1.0) for r in righe])
    print("righe %d | quota con base_accuracy 0 o 1: %.4f | coppie %d" % (len(righe), q01, len(dati)))
    incoerenti = sum(1 for rs in coppie.values() if len({float(r["base_accuracy"]) for r in rs}) > 1)
    print("coppie con base_accuracy diversa fra le loro righe: %d" % incoerenti)
    for nome, soglia in (("base > 0", 1e-9), ("base >= 0,5", 0.5), ("base >= 0,6", 0.6), ("nessuna condizione", -1.0)):
        oss = np.array([osservata(pb, pert, soglia) for _, _, pb, pert in dati])
        rng = np.random.RandomState(SEME)
        ks, tassi = [], []
        for _ in range(N_REPLICHE):
            a = simula(dati, soglia, rng); b = simula(dati, soglia, rng)
            ks.append(kappa(a, b)); tassi.append(a.mean())
        ks = np.array([k for k in ks if not np.isnan(k)])
        print("%-20s robust osservato %2d/%d = %.1f%% | simulato %.1f%% | kappa test-retest media %.3f, IC90 [%.3f, %.3f], mediana %.3f" % (
            nome, oss.sum(), len(oss), 100 * oss.mean(), 100 * np.mean(tassi), ks.mean(),
            np.percentile(ks, 5), np.percentile(ks, 95), np.median(ks)))


if __name__ == "__main__":
    main()
