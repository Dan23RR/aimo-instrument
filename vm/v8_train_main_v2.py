"""V8 sul train set pubblico della fase principale (aimo-interp/train-main-v2), 23/9/2026.

Gli organizzatori hanno indicato il 21/9 (getting-started, issue #10) che il report deve
riportare le valutazioni sul loro train set pubblico, sul set di validazione privato e sul test.
Questo script calcola la prima per la voce V8: robust se il testo del problema ha almeno 98
caratteri, nessun'altra informazione. Le righe senza etichetta (fascia fra 0,10 e 0,25 di
max_drop) sono escluse, come fanno gli organizzatori.

Uso (dalla cartella AIMO): python vm/v8_train_main_v2.py
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(BASE, "dati", "train_main_v2", "train-00000-of-00001.parquet")


def main():
    tr = pd.read_parquet(F)
    tr["n_char"] = tr.problem.str.len()
    tr["pred"] = tr.n_char >= 98
    lab = tr[tr.is_robust.notna()].copy()
    lab["y"] = lab.is_robust.astype(bool)
    giuste = int((lab.pred == lab.y).sum())
    sens = ((lab.pred) & (lab.y)).sum() / lab.y.sum()
    spec = ((~lab.pred) & (~lab.y)).sum() / (~lab.y).sum()
    print("train-main-v2: righe %d, etichettate %d (robust %d, non robust %d), testi di problema distinti %d" % (
        len(tr), len(lab), int(lab.y.sum()), int((~lab.y).sum()), tr.problem.nunique()))
    print("n_char: min %d, mediana %.1f, max %d; righe sotto 98 caratteri: %d su %d" % (
        tr.n_char.min(), tr.n_char.median(), tr.n_char.max(), int((tr.n_char < 98).sum()), len(tr)))
    print("V8 sulle righe etichettate: predice robust %d | corrette %d/%d = %.4f | accuratezza bilanciata %.4f" % (
        int(lab.pred.sum()), giuste, len(lab), giuste / len(lab), 0.5 * (sens + spec)))
    print("costante sempre robust: %d/%d = %.4f | costante mai robust: %.4f" % (
        int(lab.y.sum()), len(lab), lab.y.mean(), 1 - lab.y.mean()))
    for (m, e), g in lab.groupby(["model_id", "reasoning_effort"]):
        print("   %-40s %-7s n=%2d robust %2d | V8 corrette %2d" % (m, e, len(g), int(g.y.sum()), int((g.pred == g.y).sum())))


if __name__ == "__main__":
    main()
