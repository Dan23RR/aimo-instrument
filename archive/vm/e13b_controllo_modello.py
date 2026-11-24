"""E13b, controllo NON pre-registrato, dichiarato come tale nell'esito.

Domanda: la concordanza nulla con le etichette ufficiali dipende dal fatto che parte di esse
ha model_id 'qwen3-8b:low' invece di 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'?
(L'identificazione qwen3-8b = DeepSeek-R1-0528-Qwen3-8B e' ereditata da E11/E13/E15 ed e'
quella usata dall'artefatto ufficiale della sonda.) Stesse funzioni di e13b_regola_nuova.py.
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e13b_regola_nuova as E  # noqa: E402


def ufficiali_per_modello(fg_ids, quale):
    righe = []
    for f in sorted(glob.glob(os.path.join(E.DATI, "*.json"))):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, list):
            continue
        for x in d:
            if not isinstance(x, dict) or "absolute_accuracy_decay" not in x:
                continue
            mid = str(x.get("model_id", "")).lower()
            if quale == "deepseek" and not mid.startswith("deepseek-ai/deepseek-r1-0528-qwen3-8b"):
                continue
            if quale == "qwen3-8b" and mid != "qwen3-8b:low":
                continue
            if x.get("problem_id") not in fg_ids:
                continue
            pt = x.get("permutation_type")
            tipi = json.loads(pt) if isinstance(pt, str) and pt.startswith("[") else (pt if isinstance(pt, list) else [pt])
            if not any(t in E.FAMIGLIE for t in tipi):
                continue
            righe.append({"problem_id": x["problem_id"], "decay": float(x["absolute_accuracy_decay"]),
                          "base_acc": float(x["base_accuracy"])})
    d = pd.DataFrame(righe)
    g = d.groupby("problem_id").agg(max_drop=("decay", "max"), base_acc=("base_acc", "max")).reset_index()
    g["label"] = g.max_drop.map(E.regola)
    return g


def main():
    fg = pd.read_parquet(E.FG)
    fg_ids = set(fg.problem_id)
    serie = E.carica()
    nostre = {c: E.max_drop_nostro(serie[(E.DEEPSEEK, c)]) for c in ("A", "B", "Ap")}
    print("E13b, controllo NON pre-registrato: etichette ufficiali separate per model_id")
    for quale in ("deepseek", "qwen3-8b"):
        u = ufficiali_per_modello(fg_ids, quale)
        print("-- righe ufficiali con model_id %s: %d problemi | robust %.1f%% | non robust %.1f%% | fascia %.1f%%" % (
            "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" if quale == "deepseek" else "qwen3-8b:low",
            len(u), 100 * (u.label == 1).mean(), 100 * (u.label == 0).mean(), 100 * u.label.isna().mean()))
        for c in ("A", "B", "Ap"):
            print(E.riga("kappa(%s, ufficiale nuova %s)" % (c, quale), E.confronta(nostre[c], u)))

    # Esplorativo, anch'esso NON pre-registrato: le due fonti ufficiali sugli stessi problemi.
    # Se sono la stessa misura il kappa e' ~1; se sono due misure indipendenti dello stesso
    # modello e' un test-retest ufficiale; se sono modelli diversi e' una concordanza fra modelli.
    ud, uq = ufficiali_per_modello(fg_ids, "deepseek"), ufficiali_per_modello(fg_ids, "qwen3-8b")
    m = ud.merge(uq, on="problem_id", suffixes=("_ds", "_q"))
    print("-- ufficiale contro ufficiale sugli stessi problemi: %d in comune" % len(m))
    print("   max_drop identico in %d su %d | correlazione di Spearman dei max_drop %+.3f" % (
        int((m.max_drop_ds - m.max_drop_q).abs().lt(1e-9).sum()), len(m), m.max_drop_ds.corr(m.max_drop_q, method="spearman")))
    print(E.riga("kappa(ufficiale deepseek, ufficiale qwen3-8b)", E.confronta(ud, uq)))


if __name__ == "__main__":
    main()
