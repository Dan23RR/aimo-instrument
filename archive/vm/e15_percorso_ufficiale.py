"""E15, ricalcolo del 21/9/2026 chiesto dalla verifica avversaria del report.

La sonda ufficiale di DeepSeek applicata alle nostre 164 etichette con la STESSA procedura
del suo codice di inferenza (probe_inference.py: layer della strategia raccomandata, media dei
margini di tutti i probe di tutti i gruppi, gruppi di controllo compresi), invece
dell'aggregazione pre-registrata in PREREG_E15 (gruppi NONE, ciascuno al proprio best layer).
Usa le funzioni ufficiali senza modificarle; torch e transformers servono solo a caricare il
modello e qui sono sostituiti da moduli vuoti, perche' gli stati nascosti sono gia' estratti.

Uso (dalla cartella AIMO): python vm/e15_percorso_ufficiale.py
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E15 = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "e15")
FG = os.path.join(BASE, "vm_backup", "estratto", "work", "aimo", "gpu", "out", "feature_greedy.parquet")

sys.modules.setdefault("torch", types.ModuleType("torch"))
_tf = types.ModuleType("transformers")
_tf.AutoModelForCausalLM = object
_tf.AutoTokenizer = object
sys.modules.setdefault("transformers", _tf)
sys.path.insert(0, E15)
import probe_inference as P  # noqa: E402


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); o = np.argsort(s, kind="mergesort"); yy = y[o]; ss = s[o]
    r = np.empty(len(ss)); i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0; i = j + 1
    p = yy.sum(); n = len(yy) - p
    return float((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    art = P.load_pickle_artifact(Path(E15) / "probe_artifacts" / "probe_artifact.pkl")
    layer = P._layer_index(art)
    d = np.load(os.path.join(E15, "hs_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B_sys.npz"), allow_pickle=True)
    X, ids = d["X"], list(d["ids"])
    lab = pd.read_parquet(FG).set_index("problem_id").robust
    keep = [i for i, p in enumerate(ids) if p in lab.index]
    y = np.array([int(lab[ids[i]]) for i in keep])
    m = np.array([P.mean_ensemble_margin(X[i, layer, :].astype(np.float32), art) for i in keep])
    n_probe = sum(np.asarray(g["probes"][layer]["weights"]).reshape(-1, X.shape[2]).shape[0]
                  for g in art.data["groups"] if layer in g.get("probes", {}))
    print("sonda ufficiale, percorso di probe_inference.py: layer %d, %d probe mediati, %d gruppi" % (layer, n_probe, len(art.data["groups"])))
    print("n=%d problemi etichettati | AUROC del margine %.4f | predetti robust (margine >= 0) %.1f%%" % (len(y), auroc(m, y), 100 * (m >= 0).mean()))


if __name__ == "__main__":
    main()
