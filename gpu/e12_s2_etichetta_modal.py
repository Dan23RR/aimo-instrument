"""E12 / Stadio 2, passo B — DeepSeek greedy su originali + varianti accettate. SOLO ESTRAZIONE.

Uso:
  modal run e12_s2_etichetta_modal.py

Legge /vol/e12_s2_varianti.parquet (passo A). Greedy, max_tokens=8192, come PREREG_E12.
Salvataggi parziali a blocchi (preemption). Nessuna etichetta nel container.
"""
import modal

app = modal.App("aimo-e12-s2-ds")
_SEP = "\\" if "\\" in __file__ else "/"
_GPU = __file__.rsplit(_SEP, 1)[0]
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_file(_GPU + _SEP + "perturbazioni.py", "/root/perturbazioni.py"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
MAX_TOK = 8192
BLOCCO = 320


@app.function(image=IMG, gpu="A100-80GB", volumes={"/vol": VOL}, timeout=4 * 3600)
def etichetta(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    var = pd.read_parquet("/vol/e12_s2_varianti.parquet")
    lav = var[var.accettata | (var.idx == -1)][["problem_id", "famiglia", "idx", "testo"]].reset_index(drop=True)
    parz = "/vol/e12_s2_ds_parziale.parquet"
    righe = []
    fatte = set()
    if os.path.isfile(parz):
        df0 = pd.read_parquet(parz)
        righe = df0.to_dict("records")
        fatte = set(zip(df0.problem_id, df0.famiglia, df0.idx))
    lavori = [r for r in lav.itertuples(index=False) if (r.problem_id, r.famiglia, r.idx) not in fatte]
    print("modello=%s generazioni da fare %d, gia' fatte %d (originali %d, varianti accettate %d)"
          % (model, len(lavori), len(fatte), int((lav.idx == -1).sum()), int((lav.idx >= 0).sum())), flush=True)

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=MAX_TOK + 2560,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOK)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)

    tot = time.time()
    for i in range(0, len(lavori), BLOCCO):
        blocco = lavori[i:i + BLOCCO]
        convs = [[{"role": "user", "content": r.testo}] for r in blocco]
        t1 = time.time()
        outs = llm.chat(convs, sp, use_tqdm=False)
        for r, o in zip(blocco, outs):
            righe.append({"problem_id": r.problem_id, "famiglia": r.famiglia, "idx": int(r.idx),
                          "risp_ds": risposta(o.outputs[0].text),
                          "tok_ds": len(o.outputs[0].token_ids),
                          "finito_ds": o.outputs[0].finish_reason == "stop"})
        pd.DataFrame(righe).to_parquet(parz)
        VOL.commit()
        print("blocco %d/%d in %.0fs | salvate %d" % (i // BLOCCO + 1, (len(lavori) + BLOCCO - 1) // BLOCCO, time.time() - t1, len(righe)), flush=True)
    df = pd.DataFrame(righe).drop_duplicates(["problem_id", "famiglia", "idx"])
    nome = "/vol/e12_s2_deepseek.parquet"
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe | con risposta %.3f | finiti %.3f | %.0fs"
          % (nome, len(df), df.risp_ds.notna().mean(), df.finito_ds.mean(), time.time() - tot), flush=True)
    return {"file": nome, "n": len(df)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    print(etichetta.remote(model=model))
