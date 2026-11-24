"""E15-R: i due conti chiesti da [REDACTED: two reviewers] sulla sonda nostra (H36), 23/11/2026.

Procedura fissata qui PRIMA di eseguirla. Non cambia nessun conto del report: ne aggiunge due.

Dati
- Stati nascosti con system prompt: vm_backup/estratto/work/aimo/e15/hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz.
- Etichette: gpu/out/feature_greedy.parquet (164 problemi, 40 robust), come in vm/e15_valuta.py.
- Lunghezze n_char e ordine dei problemi: vm_risultati/e15_oof_sonda_2026-09-23.npz (stesse fonti del report).
- Etichette esplicite: le 138 con model_is_robust non nullo nei sei file pubblici in dati/ (vedi verifica_rilievi.py).

Parte 1. Sonda piu' lunghezza senza fuga ([REDACTED: reviewer and request])
- Per ogni seme s (164 problemi: 20 semi 20260915..20260934; 138 esplicite: 10 semi 20260915..20260924) si fanno
  5 fold esterni stratificati, costruiti come cv2 in vm/e15_controllo_lunghezza.py.
- Per ogni fold esterno k (train tr, test te):
  a) punteggi della sonda per tr: cv_nidificata(X[tr], y[tr], RandomState(10*s+k)) di vm/e15_valuta.py, cioe' OOF
     calcolati dentro tr;
  b) punteggio della sonda per te: layer e C scelti con la stessa CV interna a 4 fold su tutto tr
     (RandomState(10*s+k+5)), sonda adattata su tutto tr e applicata a te;
  c) secondo livello adattato su tr e valutato su te: regressione logistica (C=1, class_weight balanced, feature
     standardizzate) su (i) log n_char, (ii) sonda, (iii) entrambi.
- Si riportano, per seme, l'AUROC OOF dei tre modelli sugli stessi problemi e la differenza (iii) meno (i); poi
  media, minimo e massimo sui semi, e l'intervallo bootstrap al 95% della differenza (2000 ricampionamenti appaiati
  dei problemi sui punteggi di ogni seme; si riporta la mediana sui semi dei due estremi).
- Lettura fissata ora: se l'intervallo della differenza contiene lo 0, il testo dira' che la sonda non aggiunge
  niente di misurabile alla lunghezza; se sta tutto sopra lo 0, riportera' il guadagno con l'intervallo.
- In piu': la sonda riaddestrata sulle sole 138 esplicite (cv_nidificata, RandomState(20260914)), AUROC OOF accanto
  a quello di n_char sulle stesse 138.

Parte 2. Null nidificato di H36 con 1.000 permutazioni ([REDACTED: reviewer and request])
- Stessa procedura di vm/e15_null_nidificato.py, braccio B (griglia piena C in {0,001; 0,01; 0,1; 1}), stessi semi:
  permutazioni generate in fila da RandomState(20260915); la q-esima valutata con
  cv_nidificata(..., RandomState(20260914 + 100 + q)).
- Controllo: la statistica deve tornare 0,7097 e le prime 60 permutazioni devono ridare il braccio B del 23/9
  (media 0,4890, p95 0,5700).
- p = (1 + numero di null >= statistica) / (1 + N), con N le permutazioni finite.
- Salvataggio a blocchi di 25 con ripresa; il file STOP nella cartella d'uscita ferma la corsa a fine blocco.

Uso (dalla cartella AIMO): python revisione_condizionata_2026-11/e15r_ricalcolo.py CARTELLA_USCITA [N_LAVORI]
Con E15R_PROVA=1 fa una prova veloce (2 layer, 1 seme, 1 fold, 2 permutazioni) in CARTELLA_USCITA.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
NPZ_HS = os.path.join(ARCH, "e15", "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz")
FG = os.path.join(BASE, "gpu", "out", "feature_greedy.parquet")
OOF = os.path.join(BASE, "vm_risultati", "e15_oof_sonda_2026-09-23.npz")
DATI = os.path.join(BASE, "dati")
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math", "augmented-sample-math-full-filtered"]
NOMI_DS = {"qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
CS = (0.001, 0.01, 0.1, 1.0)
SEME = 20260914
PROVA = os.environ.get("E15R_PROVA") == "1"

# gli argomenti si leggono in main(): i processi figli di joblib possono rieseguire questo modulo
_ARGOMENTI = list(sys.argv[1:])
_USCITA, _LAVORI = None, 5
sys.argv = [sys.argv[0]]
sys.path.insert(0, os.path.join(BASE, "vm"))
import e15_valuta as V  # noqa: E402


def scrivi(nome, testo):
    with open(os.path.join(_USCITA, nome), "a", encoding="utf-8") as f:
        f.write(testo + "\n")


def log(testo):
    riga = time.strftime("%Y-%m-%d %H:%M:%S") + " " + testo
    print(riga, flush=True)
    scrivi("log.txt", riga)


def stop():
    return os.path.isfile(os.path.join(_USCITA, "STOP"))


def carica():
    d = np.load(NPZ_HS, allow_pickle=True)
    X, ids = d["X"].astype(np.float32), list(d["ids"])
    if PROVA:
        X = X[:, :3, :]
    lab = pd.read_parquet(FG)[["problem_id", "robust"]]
    X, y, keep = V.allinea(X, ids, lab)
    o = np.load(OOF)
    assert list(o["problem_id"]) == keep, "ordine dei problemi diverso dal file OOF del 23/9"
    esp = set()
    for nome in FONTI:
        for r in json.load(open(os.path.join(DATI, "%s.json" % nome), encoding="utf-8")):
            if r.get("model_id") in NOMI_DS and r.get("model_is_robust") is not None:
                esp.add(r["problem_id"])
    esplicite = np.array([p in esp for p in keep])
    return X, y, np.asarray(o["n_char"], float), esplicite, keep


def scegli_e_adatta(X, y, tr, te, rng):
    """Come l'interno di cv_nidificata: layer e C scelti con CV a 4 fold su tr, sonda adattata su tutto tr."""
    k_in = 4
    t1 = rng.permutation(tr[y[tr] == 1]); t0 = rng.permutation(tr[y[tr] == 0])
    inner = [np.concatenate([t1[j::k_in], t0[j::k_in]]) for j in range(k_in)]
    pos = {v: q for q, v in enumerate(tr)}
    best = (-1, None, None)
    for L in range(1, X.shape[1]):
        for C in CS:
            s = np.zeros(len(tr))
            for j in range(k_in):
                ite = inner[j]; itr = np.concatenate([inner[q] for q in range(k_in) if q != j])
                s[[pos[v] for v in ite]] = V._fit_logreg(X[itr, L, :], y[itr], C)(X[ite, L, :])
            a = V.auroc(s, y[tr])
            if a > best[0]:
                best = (a, L, C)
    _, L, C = best
    return V._fit_logreg(X[tr, L, :], y[tr], C)(X[te, L, :]), (L, C)


def secondo_livello(Ftr, ytr, Fte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(sc.transform(Ftr), ytr)
    return clf.decision_function(sc.transform(Fte))


def fold_esterni(y, seed):
    rng = np.random.RandomState(seed)
    idx1 = rng.permutation(np.where(y == 1)[0]); idx0 = rng.permutation(np.where(y == 0)[0])
    return [np.concatenate([idx1[i::5], idx0[i::5]]) for i in range(5)]


def unita(X, y, nchar, s, k):
    folds = fold_esterni(y, s)
    te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
    o_tr, _ = V.cv_nidificata(X[tr], y[tr], np.random.RandomState(10 * s + k))
    s_te, scelta = scegli_e_adatta(X, y, tr, te, np.random.RandomState(10 * s + k + 5))
    L = np.log(nchar)
    ris = {}
    for nome, Ftr, Fte in (("lunghezza", L[tr][:, None], L[te][:, None]),
                           ("sonda", o_tr[:, None], s_te[:, None]),
                           ("entrambi", np.column_stack([L[tr], o_tr]), np.column_stack([L[te], s_te]))):
        ris[nome] = secondo_livello(Ftr, y[tr], Fte)
    return s, k, te, ris, scelta


def boot_diff(a, b, y, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    d = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(V.auroc(a[i], y[i]) - V.auroc(b[i], y[i]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def parte1(X, y, nchar, nome, semi):
    log("Parte 1 su %s: n=%d robust %d, %d semi x 5 fold" % (nome, len(y), int(y.sum()), len(semi)))
    t0 = time.time()
    fold = [(s, k) for s in semi for k in range(1 if PROVA else 5)]
    out = Parallel(n_jobs=_LAVORI)(delayed(unita)(X, y, nchar, s, k) for s, k in fold)
    per_seme = {}
    for s, k, te, ris, scelta in out:
        d = per_seme.setdefault(s, {m: np.full(len(y), np.nan) for m in ("lunghezza", "sonda", "entrambi")})
        for m in d:
            d[m][te] = ris[m]
        scrivi("parte1_%s_scelte.txt" % nome, "seme %d fold %d: layer %d C %g" % (s, k, scelta[0], scelta[1]))
    righe = []
    for s, d in sorted(per_seme.items()):
        m = ~np.isnan(d["lunghezza"])
        a = {q: V.auroc(d[q][m], y[m]) for q in d}
        lo, hi = boot_diff(d["entrambi"][m], d["lunghezza"][m], y[m], seed=s)
        righe.append((s, a["lunghezza"], a["sonda"], a["entrambi"], a["entrambi"] - a["lunghezza"], lo, hi))
        scrivi("parte1_%s.txt" % nome, "seme %d | AUROC lunghezza %.4f sonda %.4f entrambi %.4f | differenza %+.4f IC95 [%+.4f, %+.4f]" % righe[-1])
    r = np.array([x[1:] for x in righe])
    sintesi = {"insieme": nome, "n": int(len(y)), "robust": int(y.sum()), "semi": len(righe),
               "media": dict(zip(["lunghezza", "sonda", "entrambi", "differenza"], map(float, r[:, :4].mean(0)))),
               "min": dict(zip(["lunghezza", "sonda", "entrambi", "differenza"], map(float, r[:, :4].min(0)))),
               "max": dict(zip(["lunghezza", "sonda", "entrambi", "differenza"], map(float, r[:, :4].max(0)))),
               "ic95_differenza_mediana_sui_semi": [float(np.median(r[:, 4])), float(np.median(r[:, 5]))],
               "secondi": round(time.time() - t0)}
    scrivi("sintesi.json", json.dumps({"parte1_" + nome: sintesi}, ensure_ascii=False))
    log("Parte 1 su %s finita: %s" % (nome, json.dumps(sintesi, ensure_ascii=False)))


def una_perm(X, yp, q):
    o, _ = V.cv_nidificata(X, yp, np.random.RandomState(SEME + 100 + q), Cs=CS)
    return q, V.auroc(o, yp)


def parte2(X, y, n_perm):
    t0 = time.time()
    oof, scelte = V.cv_nidificata(X, y, np.random.RandomState(SEME))
    stat = V.auroc(oof, y)
    log("Parte 2: statistica H36 AUROC OOF %.4f (atteso 0,7097) | scelte %s" % (stat, scelte))
    if not PROVA and abs(stat - 0.7097) > 5e-4:
        log("STOP: la statistica non si riproduce"); return
    rngp = np.random.RandomState(SEME + 1)
    perms = [rngp.permutation(y) for _ in range(n_perm)]
    fatti = os.path.join(_USCITA, "parte2_null.jsonl")
    fatte = {}
    if os.path.isfile(fatti):
        for riga in open(fatti, encoding="utf-8"):
            j = json.loads(riga); fatte[j["q"]] = j["auroc"]
    for inizio in range(0, n_perm, 25):
        if stop():
            log("STOP trovato: fermo la Parte 2 con %d permutazioni" % len(fatte)); break
        blocco = [q for q in range(inizio, min(n_perm, inizio + 25)) if q not in fatte]
        if not blocco:
            continue
        for q, a in Parallel(n_jobs=_LAVORI)(delayed(una_perm)(X, perms[q], q) for q in blocco):
            fatte[q] = a
            scrivi("parte2_null.jsonl", json.dumps({"q": q, "auroc": a}))
        nul = np.array([fatte[q] for q in sorted(fatte)])
        prime = np.array([fatte[q] for q in range(min(60, n_perm)) if q in fatte])
        log("Parte 2: %d permutazioni | media %.4f p95 %.4f | >= statistica %d | p %.4f | prime %d: media %.4f p95 %.4f | %.0f s" % (
            len(nul), nul.mean(), np.percentile(nul, 95), int((nul >= stat).sum()), (1 + (nul >= stat).sum()) / (1 + len(nul)),
            len(prime), prime.mean(), np.percentile(prime, 95), time.time() - t0))


def main():
    global _USCITA, _LAVORI
    _USCITA = _ARGOMENTI[0]
    _LAVORI = int(_ARGOMENTI[1]) if len(_ARGOMENTI) > 1 else 5
    os.makedirs(_USCITA, exist_ok=True)
    import sklearn
    log("E15-R | python %s | numpy %s | sklearn %s | lavori %d | prova %s" % (
        platform.python_version(), np.__version__, sklearn.__version__, _LAVORI, PROVA))
    X, y, nchar, esp, keep = carica()
    log("dati: X %s | robust %d su %d | esplicite %d (robust %d)" % (X.shape, int(y.sum()), len(y), int(esp.sum()), int(y[esp].sum())))
    semi164 = [20260915] if PROVA else list(range(20260915, 20260935))
    semi138 = [20260915] if PROVA else list(range(20260915, 20260925))
    if not stop():
        parte1(X, y, nchar, "164", semi164)
    if not stop():
        Xe, ye, ne = X[esp], y[esp], nchar[esp]
        oof, scelte = V.cv_nidificata(Xe, ye, np.random.RandomState(SEME))
        r = {"n": int(len(ye)), "robust": int(ye.sum()), "auroc_sonda_riaddestrata": float(V.auroc(oof, ye)),
             "auroc_n_char": float(V.auroc(ne, ye)), "scelte": [list(map(float, x)) for x in scelte]}
        scrivi("sintesi.json", json.dumps({"sonda_sulle_138": r}, ensure_ascii=False))
        log("sonda riaddestrata sulle esplicite: %s" % json.dumps(r, ensure_ascii=False))
        parte1(Xe, ye, ne, "138", semi138)
    if not stop():
        parte2(X, y, 2 if PROVA else 1000)
    log("fine")


if __name__ == "__main__":
    main()
