"""E15: il null nidificato di H36, rieseguito in locale e salvato (23/9/2026).

Il 14/9 il null nidificato fu eseguito sulla VM con il codice di vm/e15_valuta.py (60 permutazioni,
C in {0,01; 1}). Il suo output e' nel log work/aimo/e15_valuta.log dell'archivio
vm_backup/backup_aimo_2026-09-17.tar.gz (estratto in vm_backup/estratto il 23/9): con system prompt
media 0,4895 e p95 0,5744; senza system prompt media 0,4960 e p95 0,5888. (Correzione del 23/9:
una prima versione di questo docstring diceva che quell'output non era stato salvato.)
Qui si riesegue la stessa procedura, con gli stessi semi, sugli stati nascosti salvati:
  A. come il 14/9: 60 permutazioni, griglia ridotta C in {0,01; 1} (la deviazione non dichiarata);
  B. 60 permutazioni con la griglia della statistica vera, C in {0,001; 0,01; 0,1; 1}.
Le 500 permutazioni pre-registrate non sono eseguite: costerebbero decine di ore di CPU.

Uso (dalla cartella AIMO): python vm/e15_null_nidificato.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo")
sys.path.insert(0, os.path.join(BASE, "vm"))
sys.argv = [sys.argv[0],
            "--npz-sys", os.path.join(E, "e15", "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz"),
            "--npz-nosys", os.path.join(E, "e15", "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.npz"),
            "--artefatto", os.path.join(E, "e15", "probe_artifacts", "probe_artifact.pkl"),
            "--etichette", os.path.join(E, "gpu", "out", "feature_greedy.parquet")]
import e15_valuta as V  # noqa: E402

N_PERM = 60
OUT = os.path.join(BASE, "vm_risultati", "e15_null_nidificato_2026-09-23")


def una(X, yp, q, Cs):
    o, _ = V.cv_nidificata(X, yp, np.random.RandomState(V.args.seed + 100 + q), Cs=Cs)
    return V.auroc(o, yp)


def main():
    t0 = time.time()
    lab = pd.read_parquet(V.args.etichette)[["problem_id", "robust"]]
    X, ids = V.carica(V.args.npz_sys)
    X, y, _ = V.allinea(X, ids, lab)
    oof, scelte = V.cv_nidificata(X, y, np.random.RandomState(V.args.seed))
    stat = V.auroc(oof, y)
    righe = ["n=%d | robust %d | statistica H36: AUROC OOF %.4f | layer/C per fold %s | %.0f s" % (len(y), y.sum(), stat, scelte, time.time() - t0)]
    print(righe[-1], flush=True)
    rngp = np.random.RandomState(V.args.seed + 1)
    perms = [rngp.permutation(y) for _ in range(N_PERM)]
    ris = {}
    for nome, Cs in (("A_griglia_ridotta", (0.01, 1.0)), ("B_griglia_piena", (0.001, 0.01, 0.1, 1.0))):
        t1 = time.time()
        nul = np.array(Parallel(n_jobs=-1)(delayed(una)(X, perms[q], q, Cs) for q in range(N_PERM)))
        ris[nome] = nul
        righe.append("%s C=%s: %d permutazioni | media %.4f | p95 %.4f | quota >= statistica %.4f | %.0f s" % (
            nome, Cs, N_PERM, nul.mean(), np.percentile(nul, 95), (nul >= stat).mean(), time.time() - t1))
        print(righe[-1], flush=True)
    np.savez(OUT + ".npz", stat=stat, **ris)
    with open(OUT + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(righe) + "\n")


if __name__ == "__main__":
    main()
