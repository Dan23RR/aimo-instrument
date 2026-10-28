"""E9 — etichette di robustezza con perturbazioni NOSTRE, via vLLM su Modal.

Uso:
  modal run e9_modal.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --tag stadio1
  modal run e9_modal.py --model Qwen/Qwen3.5-4B --tag stadio2 --limite 174

Per ogni problema pubblico: generazione greedy dell'originale + di ogni famiglia nostra.
Etichetta: robust_ours = risposta originale presente E identica sotto tutte le famiglie.
Scrive un parquet sul volume `aimo-e9`; nessun testo di problema nei print (solo id/conteggi).
"""
import modal

app = modal.App("aimo-e9")
# Base CUDA "devel": include nvcc, che flashinfer richiede per il JIT del sampler.
# In piu' il sampler flashinfer viene disattivato: a temperatura 0 non serve.
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_file(__file__.replace("e9_modal.py", "perturbazioni.py"), "/root/perturbazioni.py")
       .add_local_dir(__file__.replace("gpu" + ("\\" if "\\" in __file__ else "/") + "e9_modal.py", "dati"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
MAX_TOK = 4096
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]


def _problemi_pubblici():
    """Tutti i problemi distinti dei dataset pubblici aimo-interp.

    Prima dalla copia locale spedita nell'immagine (/root/dati/*.json, la stessa cache
    usata per ogni analisi offline), cosi' il run non dipende dalla rete. Se manca,
    ripiego sull'API datasets-server con tentativi e attesa esponenziale (il 502 del
    2 settembre ha ucciso un run prima ancora di caricare il modello).
    """
    import json, os, time, urllib.request
    visti = {}
    locali = [os.path.join("/root/dati", n + ".json") for n in FONTI]
    if all(os.path.isfile(p) for p in locali):
        for p in locali:
            for r in json.load(open(p, encoding="utf-8")):
                pid = r.get("problem_id"); t = r.get("original_problem")
                if pid and isinstance(t, str) and pid not in visti: visti[pid] = t
        print("problemi dalla cache locale: %d" % len(visti), flush=True)
        return visti
    for nome in FONTI:
        off = 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=aimo-interp%%2F%s"
                   "&config=default&split=validation&offset=%d&length=100" % (nome, off))
            for tentativo in range(6):
                try:
                    b = json.loads(urllib.request.urlopen(url, timeout=90).read()); break
                except Exception as exc:
                    if tentativo == 5: raise
                    time.sleep(3 * 2 ** tentativo)
            rows = [x["row"] for x in b.get("rows", [])]
            for r in rows:
                pid = r.get("problem_id"); t = r.get("original_problem")
                if pid and isinstance(t, str) and pid not in visti: visti[pid] = t
            off += 100
            if off >= b.get("num_rows_total", 0) or not rows: break
    print("problemi dalla rete: %d" % len(visti), flush=True)
    return visti


@app.function(image=IMG, gpu="A100-40GB", volumes={"/vol": VOL}, timeout=4 * 3600)
def etichetta(model: str, tag: str, limite: int = 0, solo_ids: str = ""):
    import sys, time, json
    sys.path.insert(0, "/root")
    from perturbazioni import perturba, risposta, FAMIGLIE
    import pandas as pd
    from vllm import LLM, SamplingParams

    problemi = _problemi_pubblici()
    if solo_ids:
        keep = set(solo_ids.split(","))
        problemi = {k: v for k, v in problemi.items() if k in keep}
    ids = sorted(problemi)
    if limite: ids = ids[:limite]
    print("modello=%s problemi=%d" % (model, len(ids)), flush=True)

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92,
              trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOK)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)

    # batch unico: (problem_id, famiglia, prompt)
    lavori = []
    for pid in ids:
        testo = problemi[pid]
        lavori.append((pid, "original", testo))
        for fam, var in perturba(testo).items():
            lavori.append((pid, fam, var))
    convs = [[{"role": "user", "content": p}] for _, _, p in lavori]
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0
    print("generazioni=%d in %.0fs (%.1f s/gen)" % (len(lavori), dt, dt / max(1, len(lavori))), flush=True)

    per = {}
    for (pid, fam, _), o in zip(lavori, outs):
        txt = o.outputs[0].text
        per.setdefault(pid, {})[fam] = {"risposta": risposta(txt), "token": len(o.outputs[0].token_ids),
                                        "finito": o.outputs[0].finish_reason == "stop"}
    righe = []
    for pid in ids:
        d = per[pid]; orig = d["original"]["risposta"]
        fams = [f for f in FAMIGLIE if f in d]
        stabile = {f: (d[f]["risposta"] is not None and d[f]["risposta"] == orig) for f in fams}
        righe.append({"problem_id": pid, "model": model, "risposta_orig": orig,
                      "ha_risposta": orig is not None, "n_famiglie": len(fams),
                      "n_stabili": sum(stabile.values()),
                      "robust_ours": (orig is not None and all(stabile.values())),
                      **{"stab_" + f: stabile.get(f) for f in FAMIGLIE},
                      **{"tok_" + f: d[f]["token"] for f in ["original"] + fams},
                      "finito_orig": d["original"]["finito"]})
    df = pd.DataFrame(righe)
    nome = "/vol/e9_%s_%s.parquet" % (tag, model.replace("/", "__"))
    df.to_parquet(nome); VOL.commit()
    print("scritto %s: %d righe | con risposta %d | robust_ours %d" % (
        nome, len(df), int(df.ha_risposta.sum()), int(df.robust_ours.sum())), flush=True)
    return {"file": nome, "n": len(df), "con_risposta": int(df.ha_risposta.sum()),
            "robust_ours": int(df.robust_ours.sum()), "secondi": round(dt)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", tag: str = "stadio1", limite: int = 0, solo_ids: str = ""):
    print(etichetta.remote(model=model, tag=tag, limite=limite, solo_ids=solo_ids))
