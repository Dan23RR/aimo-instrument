"""E9 stadio 2 — punteggi del regressore ufficiale per un modello, sui problemi pubblici.

Uso:
  modal run e9_feature_modal.py --model Qwen/Qwen3.5-4B --tag stadio2 --limite 174

Usa ESATTAMENTE il pacchetto `uncertainty_profile` degli organizzatori (spedito nel bundle
di gara) e l'artefatto DeepSeek: stessa generazione greedy, stesse 14 feature, stesso
regressore. Il risultato e' il punteggio che il container calcola a runtime per quel
modello su quel problema: e' il numero su cui va calibrata la soglia per-modello.

Scrive /vol/e9feat_<tag>_<model>.parquet con: problem_id, score, threshold_ereditata.
Nessun testo di problema nei print.
"""
import modal

app = modal.App("aimo-e9-feat")
QUI = __file__.rsplit("e9_feature_modal.py", 1)[0]
SOL = QUI.replace("gpu" + ("\\" if "\\" in QUI else "/"), "soluzione" + ("\\" if "\\" in QUI else "/"))

IMG = (modal.Image.from_registry("pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime")
       .pip_install(  # PEP 668: l'immagine pytorch ha python externally managed
           "accelerate==1.13.0", "huggingface-hub==1.22.0", "joblib==1.5.3",
                    "pandas==3.0.3", "safetensors==0.8.0", "scikit-learn==1.8.0",
                    "tokenizers==0.22.2", "transformers==5.13.0", "pyarrow", extra_options="--break-system-packages")
       .add_local_dir(SOL + "uncertainty_profile", "/root/uncertainty_profile")
       .add_local_dir(SOL + "uncertainty_artifacts", "/root/uncertainty_artifacts")
       .add_local_dir(QUI.replace("gpu" + ("\\" if "\\" in QUI else "/"), "dati" + ("\\" if "\\" in QUI else "/")).rstrip("\\/"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]


def _problemi_pubblici():
    """Problemi pubblici dalla cache locale spedita nell'immagine; rete solo come ripiego."""
    import json
    import os
    import time
    import urllib.request
    visti = {}
    locali = [os.path.join("/root/dati", n + ".json") for n in FONTI]
    if all(os.path.isfile(p) for p in locali):
        for p in locali:
            for r in json.load(open(p, encoding="utf-8")):
                pid = r.get("problem_id")
                t = r.get("original_problem")
                if pid and isinstance(t, str) and pid not in visti:
                    visti[pid] = t
        print("problemi dalla cache locale: %d" % len(visti), flush=True)
        return visti
    for nome in FONTI:
        off = 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=aimo-interp%%2F%s"
                   "&config=default&split=validation&offset=%d&length=100" % (nome, off))
            for tentativo in range(6):
                try:
                    b = json.loads(urllib.request.urlopen(url, timeout=90).read())
                    break
                except Exception:
                    if tentativo == 5:
                        raise
                    time.sleep(3 * 2 ** tentativo)
            rows = [x["row"] for x in b.get("rows", [])]
            for r in rows:
                pid = r.get("problem_id")
                t = r.get("original_problem")
                if pid and isinstance(t, str) and pid not in visti:
                    visti[pid] = t
            off += 100
            if off >= b.get("num_rows_total", 0) or not rows:
                break
    return visti


@app.function(image=IMG, gpu="A100-80GB", volumes={"/vol": VOL}, timeout=6 * 3600)
def estrai(model: str, tag: str, limite: int = 0):
    import sys
    import time
    sys.path.insert(0, "/root")
    import numpy as np
    import pandas as pd
    from uncertainty_profile.artifact import load_artifact
    from uncertainty_profile.extraction import (extract_generation_features,
                                                load_model_and_tokenizer, release_model)

    art = load_artifact("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    assert art is not None, "artefatto DeepSeek non trovato nel bundle"
    problemi = _problemi_pubblici()
    ids = sorted(problemi)
    if limite:
        ids = ids[:limite]
    print("modello=%s problemi=%d" % (model, len(ids)), flush=True)

    t0 = time.time()
    mdl, tok = load_model_and_tokenizer(model, local_files_only=False)
    print("caricato in %.0fs" % (time.time() - t0), flush=True)
    righe = []
    try:
        lotto = 8
        for i in range(0, len(ids), lotto):
            sub = ids[i:i + lotto]
            t1 = time.time()
            feats = extract_generation_features(model=mdl, tokenizer=tok,
                                                problem_texts=[problemi[p] for p in sub],
                                                config=art.generation_config,
                                                include_generated_text=False)
            X = pd.DataFrame.from_records(feats, columns=list(art.feature_names))
            s = np.asarray(art.estimator.predict(X), dtype=float)
            for p, sc, f in zip(sub, s, feats):
                righe.append({"problem_id": p, "model": model, "score": float(sc), **f})
            print("[%3d/%d] %.0fs per lotto di %d" % (i + len(sub), len(ids), time.time() - t1, len(sub)),
                  flush=True)
            if (i // lotto) % 5 == 4:
                pd.DataFrame(righe).to_parquet("/vol/e9feat_%s_%s_parziale.parquet"
                                               % (tag, model.replace("/", "__")))
                VOL.commit()
    finally:
        mdl = None
        tok = None
        release_model()
    df = pd.DataFrame(righe)
    df["threshold_ereditata"] = float(art.decision_threshold)
    nome = "/vol/e9feat_%s_%s.parquet" % (tag, model.replace("/", "__"))
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe | score min %.3f max %.3f" % (nome, len(df), df.score.min(), df.score.max()),
          flush=True)
    return {"file": nome, "n": len(df)}


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-4B", tag: str = "stadio2", limite: int = 0):
    print(estrai.remote(model=model, tag=tag, limite=limite))
