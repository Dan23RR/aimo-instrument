"""E12 / Stadio 1 — stessa estrazione di e12_s1_modal.py, ma A BLOCCHI con salvataggi parziali.

Uso:
  modal run e12_s1_blocchi_modal.py

Serve solo se Modal preempta il container (successo il 7/9 alle 09:16 UTC dopo 16 minuti di
generazione): ogni blocco di problemi viene scritto sul volume appena finito, e un riavvio
salta i blocchi gia' fatti. Protocollo, parametri e feature IDENTICI a e12_s1_modal.py e a
PREREG_E12.md: temperature=0.6, top_p=0.95, n=8, max_tokens=8192, seed=20260906.
Nessuna etichetta nel container, nessun testo di problema nei print.
"""
import modal

app = modal.App("aimo-e12-s1-blocchi")
_SEP = "\\" if "\\" in __file__ else "/"
_GPU = __file__.rsplit(_SEP, 1)[0]
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_file(_GPU + _SEP + "perturbazioni.py", "/root/perturbazioni.py")
       .add_local_file(_GPU + _SEP + "e12_ids.txt", "/root/e12_ids.txt")
       .add_local_dir(_GPU.rsplit(_SEP, 1)[0] + _SEP + "dati", "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
TEMPERATURA = 0.6
TOP_P = 0.95
N_CAMPIONI = 8
MAX_TOK = 8192
SEME = 20260906
BLOCCO = 41   # 4 blocchi da 41 problemi = 328 sequenze per blocco


def _problemi():
    import json
    import os
    v = {}
    for n in FONTI:
        for r in json.load(open(os.path.join("/root/dati", n + ".json"), encoding="utf-8")):
            pid = r.get("problem_id")
            t = r.get("original_problem")
            if pid and isinstance(t, str) and pid not in v:
                v[pid] = t
    return v


def _righe(ids, outs, model, risposta):
    from collections import Counter
    righe = []
    for pid, o in zip(ids, outs):
        risp = [risposta(c.text) for c in o.outputs]
        tok = [len(c.token_ids) for c in o.outputs]
        fin = [c.finish_reason == "stop" for c in o.outputs]
        chiavi = [r if r is not None else "__nessuna__" for r in risp]
        cnt = Counter(chiavi)
        modale, n_modale = cnt.most_common(1)[0]
        k = len(risp)
        accordo = n_modale / k
        risposte = sum(1 for r in risp if r is not None) / k
        righe.append({"problem_id": pid, "model": model, "k": k,
                      "accordo": accordo, "risposte": risposte,
                      "accordo_pesato": accordo * risposte,
                      "modale": modale, "n_modale": n_modale, "n_distinte": len(cnt),
                      "tok_medio": sum(tok) / k, "frac_finiti": sum(fin) / k,
                      "risposte_lista": "|".join(str(r) for r in chiavi),
                      "tok_lista": "|".join(str(t) for t in tok)})
    return righe


@app.function(image=IMG, gpu="A100-80GB", volumes={"/vol": VOL}, timeout=4 * 3600)
def campiona(model: str):
    import os
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    ids = [l.strip() for l in open("/root/e12_ids.txt", encoding="utf-8") if l.strip()]
    problemi = _problemi()
    ids = [p for p in ids if p in problemi]
    parziale = "/vol/e12_s1_parziale_%s.parquet" % model.replace("/", "__")
    fatti = set()
    if os.path.isfile(parziale):
        fatti = set(pd.read_parquet(parziale).problem_id.tolist())
    da_fare = [p for p in ids if p not in fatti]
    print("modello=%s problemi=%d gia_fatti=%d da_fare=%d T=%.1f top_p=%.2f n=%d max_tok=%d seed=%d"
          % (model, len(ids), len(fatti), len(da_fare), TEMPERATURA, TOP_P, N_CAMPIONI, MAX_TOK, SEME), flush=True)

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=MAX_TOK + 2560,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    sp = SamplingParams(temperature=TEMPERATURA, top_p=TOP_P, n=N_CAMPIONI,
                        max_tokens=MAX_TOK, seed=SEME)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)

    tot = time.time()
    for i in range(0, len(da_fare), BLOCCO):
        blocco = da_fare[i:i + BLOCCO]
        convs = [[{"role": "user", "content": problemi[p]}] for p in blocco]
        t1 = time.time()
        outs = llm.chat(convs, sp, use_tqdm=False)
        righe = _righe(blocco, outs, model, risposta)
        df = pd.DataFrame(righe)
        if os.path.isfile(parziale):
            df = pd.concat([pd.read_parquet(parziale), df], ignore_index=True)
        df.to_parquet(parziale)
        VOL.commit()
        print("blocco %d/%d: %d problemi in %.0fs | salvati finora %d"
              % (i // BLOCCO + 1, (len(da_fare) + BLOCCO - 1) // BLOCCO, len(blocco),
                 time.time() - t1, len(df)), flush=True)
    dt = time.time() - tot
    df = pd.read_parquet(parziale)
    df = df[df.problem_id.isin(ids)].drop_duplicates("problem_id").sort_values("problem_id")
    nome = "/vol/e12_s1_%s.parquet" % model.replace("/", "__")
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe | accordo medio %.3f | risposte medie %.3f | finiti %.3f | %.0fs"
          % (nome, len(df), df.accordo.mean(), df.risposte.mean(), df.frac_finiti.mean(), dt), flush=True)
    return {"file": nome, "n": len(df), "secondi": round(dt)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    print(campiona.remote(model=model))
