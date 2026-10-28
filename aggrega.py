"""Aggrega i dati pubblici in una tabella (modello, problema) -> robust, e la interroga.

REGOLA DELLA LABEL, ricostruita e verificata su 223 righe etichettate (val-sample 28,
augmented-sample-math-agg 141, augmented-sample-math-agg-filtered 54), 100% su tutte e tre:

    model_is_robust  <=>  n_detrimental_permutations == 0
                     <=>  absolute_accuracy_decay <= 0
                     <=>  permuted_accuracy == 1.0

Il set grande (558 righe) e' per (modello, problema, permutazione). Una coppia e' robust
se NESSUNA delle sue permutazioni ha causato decadimento, cioe' se la somma di
n_detrimental_permutations sulle sue righe e' zero. Questo estende la label a OTTO modelli.

Le due domande che decidono il metodo:
  1. c'e' un effetto MODELLO? (alcuni modelli sono piu' robusti di altri)
  2. c'e' un effetto PROBLEMA? (alcuni problemi sono spurious per tutti)
Se vale la 2, il testo del problema porta segnale anche per un modello mai visto, ed e'
l'unica cosa utilizzabile sui 15 item su 17 in cui il container nega l'accesso al modello.
"""
import json
from collections import defaultdict
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"

# i quattro modelli che la gara ci ha passato nella fase warmup (OSSERVATO, submission 907310)
MODELLI_GARA = {
    "allenai/Olmo-3-7B-Instruct": 4,
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": 2,
    "google/gemma-3-27b-it": 2,
    "openai/gpt-oss-120b": 9,
}
# alias dichiarato nel README ufficiale: qwen3-8b:low -> deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
ALIAS = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "gpt-oss-120b:low": "openai/gpt-oss-120b",
}


def carica(nome):
    return json.loads((CACHE / ("%s.json" % nome)).read_text(encoding="utf-8"))


def aggrega(righe):
    """(model_id, problem_id) -> dict con label derivata e testo del problema."""
    gruppi = defaultdict(list)
    for r in righe:
        gruppi[(r.get("model_id"), r.get("problem_id"))].append(r)

    fuori = {}
    for chiave, gr in gruppi.items():
        detr = sum(int(r.get("n_detrimental_permutations") or 0) for r in gr)
        etichettate = [r for r in gr if r.get("model_is_robust") is not None]
        label_dichiarata = None
        if etichettate:
            label_dichiarata = all(r["model_is_robust"] in (True, "True") for r in etichettate)
        fuori[chiave] = {
            "model_id": chiave[0],
            "problem_id": chiave[1],
            "dataset_id": gr[0].get("dataset_id"),
            "problema": gr[0].get("original_problem"),
            "robust": detr == 0,
            "robust_dichiarata": label_dichiarata,
            "n_righe": len(gr),
            "base_accuracy": gr[0].get("base_accuracy"),
        }
    return fuori


def main():
    sorgenti = ["aimo-interp-challenge-sample-full", "val-sample",
                "augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
                "augmented-sample-math", "augmented-sample-math-full-filtered"]
    tutte = []
    for s in sorgenti:
        tutte.extend(carica(s))

    tabella = aggrega(tutte)

    print("=" * 78)
    print("TABELLA AGGREGATA (modello, problema) -> robust")
    print("=" * 78)
    print("  %d righe grezze -> %d coppie (modello, problema)" % (len(tutte), len(tabella)))

    # controllo: la label derivata coincide con quella dichiarata dove esiste?
    conf = [(v["robust"], v["robust_dichiarata"]) for v in tabella.values()
            if v["robust_dichiarata"] is not None]
    accordo = sum(1 for a, b in conf if a == b)
    print("  verifica della regola: %d/%d coppie con label dichiarata, accordo %d (%.1f%%)"
          % (len(conf), len(tabella), accordo, 100 * accordo / max(1, len(conf))))
    if accordo != len(conf):
        print("  *** LA REGOLA NON REGGE SULL'AGGREGATO. Fermarsi qui. ***")
        for v in tabella.values():
            if v["robust_dichiarata"] is not None and v["robust"] != v["robust_dichiarata"]:
                print("    %s / %s: derivata %s, dichiarata %s"
                      % (v["model_id"], v["problem_id"], v["robust"], v["robust_dichiarata"]))
        return

    # ------------------------------------------------------------- effetto MODELLO
    print("\n" + "-" * 78)
    print("1. EFFETTO MODELLO: tasso di robustezza per modello")
    print("-" * 78)
    per_modello = defaultdict(list)
    for v in tabella.values():
        per_modello[v["model_id"]].append(v["robust"])
    print("  %-46s %6s %8s  %s" % ("model_id", "coppie", "robust", "nella gara?"))
    for m, vals in sorted(per_modello.items(), key=lambda kv: -len(kv[1])):
        tasso = sum(vals) / len(vals)
        gara = ALIAS.get(m)
        nota = ("SI -> %s" % gara.split("/")[-1]) if gara else ""
        print("  %-46s %6d %7.1f%%  %s" % (m, len(vals), 100 * tasso, nota))

    # ------------------------------------------------------------- effetto PROBLEMA
    print("\n" + "-" * 78)
    print("2. EFFETTO PROBLEMA: problemi visti da piu' modelli")
    print("-" * 78)
    per_problema = defaultdict(list)
    for v in tabella.values():
        per_problema[(v["dataset_id"], v["problem_id"])].append(v["robust"])
    multi = {k: v for k, v in per_problema.items() if len(v) >= 2}
    print("  problemi distinti: %d, di cui visti da >=2 modelli: %d"
          % (len(per_problema), len(multi)))
    if multi:
        concordi = sum(1 for v in multi.values() if len(set(v)) == 1)
        print("  fra i multi-modello, quanti hanno la STESSA label su tutti i modelli: %d/%d (%.0f%%)"
              % (concordi, len(multi), 100 * concordi / len(multi)))
        dist = defaultdict(int)
        for v in multi.values():
            dist[(len(v), sum(v))] += 1
        print("\n  distribuzione (modelli che l'hanno visto, quanti robust):")
        for (n, r), c in sorted(dist.items()):
            print("    visto da %2d modelli, robust su %2d  ->  %d problemi" % (n, r, c))

    # ------------------------------------------------------------- pool aimo-2025
    print("\n" + "-" * 78)
    print("3. IL POOL aimo-2025-reference (lo stesso della gara)")
    print("-" * 78)
    aimo = {k: v for k, v in tabella.items() if v["dataset_id"] == "aimo-2025-reference"}
    problemi = {v["problem_id"] for v in aimo.values()}
    modelli = {v["model_id"] for v in aimo.values()}
    rob = sum(1 for v in aimo.values() if v["robust"])
    print("  coppie: %d   problemi distinti: %d   modelli: %d" % (len(aimo), len(problemi), len(modelli)))
    print("  robust: %d (%.1f%%)   spurious: %d" % (rob, 100 * rob / len(aimo), len(aimo) - rob))
    print()
    print("  Il set di valutazione warmup ha 17 item su 4 modelli. Se i suoi problemi")
    print("  vengono da questo pool, il testo del problema e' una CHIAVE, non una feature.")

    # salva
    fuori = QUI / "dati" / "tabella_aggregata.json"
    fuori.write_text(json.dumps(list(tabella.values()), ensure_ascii=False), encoding="utf-8")
    print("\n  scritta %s" % fuori)


if __name__ == "__main__":
    main()
