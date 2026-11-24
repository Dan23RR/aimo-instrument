"""E16 — punteggi del regressore ufficiale sui 174 problemi PUBBLICI, per i modelli di gara.

Uso (dentro l'immagine ufficiale, vedi e16_all.sh):
  python e16_estrai.py --model Qwen/Qwen3.5-4B --gpu 0

Serve a calibrare OFFLINE, su dati pubblici, la soglia per-modello. Usa esattamente il
percorso di produzione (`uncertainty_profile` + artefatto DeepSeek + art.generation_config),
cioe' lo stesso codice che gira in gara, cosi' che gli score siano sulla stessa scala.
Scrive un parquet per modello con problem_id, score e le 14 feature.
Nessun testo di problema nei print.
"""
import argparse
import json
import os
import sys
import time

FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=174)
    ap.add_argument("--out", default="/work/aimo/e16")
    args = ap.parse_args()

    sys.path.insert(0, "/work/aimo/soluzione")
    from uncertainty_profile.artifact import load_artifact
    from uncertainty_profile.extraction import (extract_generation_features,
                                                load_model_and_tokenizer, release_model)
    import pandas as pd

    problemi = {}
    for nome in FONTI:
        for r in json.load(open("/work/aimo/dati/%s.json" % nome, encoding="utf-8")):
            pid, t = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(t, str) and pid not in problemi:
                problemi[pid] = t
    ids = sorted(problemi)[:args.n]
    art = load_artifact("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    assert art is not None
    os.makedirs(args.out, exist_ok=True)
    tag = args.model.replace("/", "__")
    # JSONL, non parquet: l'immagine ufficiale non ha pyarrow (verificato il 13/9)
    parz = os.path.join(args.out, "e16_%s.jsonl" % tag)
    righe, fatti = [], set()
    if os.path.isfile(parz):
        righe = [json.loads(l) for l in open(parz, encoding="utf-8")]
        fatti = set(r["problem_id"] for r in righe)
    da_fare = [p for p in ids if p not in fatti]
    print("E16 modello=%s problemi=%d da_fare=%d soglia_ereditata=%.4f"
          % (args.model, len(ids), len(da_fare), art.decision_threshold), flush=True)

    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(args.model, local_files_only=True)
    print("caricato in %.0fs" % (time.time() - t0), flush=True)
    LOTTO = 8
    try:
        for i in range(0, len(da_fare), LOTTO):
            sub = da_fare[i:i + LOTTO]
            t1 = time.time()
            feats = extract_generation_features(model=mdl, tokenizer=tok,
                                                problem_texts=[problemi[p] for p in sub],
                                                config=art.generation_config,
                                                include_generated_text=False)
            X = pd.DataFrame.from_records(feats, columns=list(art.feature_names))
            s = art.estimator.predict(X)
            for p, sc, f in zip(sub, s, feats):
                righe.append({"problem_id": p, "model": args.model, "score": float(sc),
                              "threshold_ereditata": float(art.decision_threshold),
                              **{k: (float(v) if v is not None else None) for k, v in f.items()}})
            with open(parz, "w", encoding="utf-8") as fh:
                for r in righe:
                    fh.write(json.dumps(r) + "\n")
            print("[%3d/%d] %.0fs" % (i + len(sub), len(da_fare), time.time() - t1), flush=True)
    finally:
        mdl = None
        tok = None
        release_model()
    sc = sorted(r["score"] for r in righe)
    n = len(sc)
    q = lambda f: sc[min(n - 1, int(f * n))]
    print("scritto %s: %d righe | score min %.4f q25 %.4f mediana %.4f q75 %.4f max %.4f | sotto la soglia ereditata %.1f%%"
          % (parz, n, sc[0], q(0.25), q(0.5), q(0.75), sc[-1],
             100 * sum(1 for x in sc if x < art.decision_threshold) / n), flush=True)


if __name__ == "__main__":
    main()
