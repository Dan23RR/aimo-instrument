"""E12 / Stadio 1 — self-consistency su DeepSeek, via vLLM su Modal. SOLO ESTRAZIONE.

Uso:
  modal run e12_s1_modal.py

Protocollo fissato in PREREG_E12.md PRIMA di eseguire, nessuna sweep:
  temperature=0.6, top_p=0.95, n=8 campioni per problema, max_tokens=8192, seed=20260906,
  stesso chat template di E9, risposta estratta con perturbazioni.risposta (\\boxed{}).

Problemi: i 164 con etichetta vera (lista di soli id in e12_ids.txt: nessuna etichetta entra
nel container). Feature per problema, tre e sole tre:
  accordo        frazione degli 8 campioni la cui risposta coincide con la modale
                 ("nessuna" e' una risposta a se')
  risposte       frazione degli 8 campioni con una risposta estratta
  accordo_pesato accordo * risposte
Salva anche le 8 risposte per problema, per trasparenza. Nessun testo di problema nei print.
Il verdetto e' offline, in e12_s1_analisi.py.
"""
import modal

app = modal.App("aimo-e12-s1")
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
# parametri FISSATI in PREREG_E12.md
TEMPERATURA = 0.6
TOP_P = 0.95
N_CAMPIONI = 8
MAX_TOK = 8192
SEME = 20260906


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


@app.function(image=IMG, gpu="A100-80GB", volumes={"/vol": VOL}, timeout=4 * 3600)
def campiona(model: str):
    import sys
    import time
    from collections import Counter
    sys.path.insert(0, "/root")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    ids = [l.strip() for l in open("/root/e12_ids.txt", encoding="utf-8") if l.strip()]
    problemi = _problemi()
    manca = [p for p in ids if p not in problemi]
    ids = [p for p in ids if p in problemi]
    print("modello=%s problemi=%d mancanti=%d T=%.1f top_p=%.2f n=%d max_tok=%d seed=%d"
          % (model, len(ids), len(manca), TEMPERATURA, TOP_P, N_CAMPIONI, MAX_TOK, SEME), flush=True)

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=MAX_TOK + 2560,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    sp = SamplingParams(temperature=TEMPERATURA, top_p=TOP_P, n=N_CAMPIONI,
                        max_tokens=MAX_TOK, seed=SEME)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)

    convs = [[{"role": "user", "content": problemi[p]}] for p in ids]
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0
    n_seq = sum(len(o.outputs) for o in outs)
    print("sequenze=%d in %.0fs (%.1f s/seq)" % (n_seq, dt, dt / max(1, n_seq)), flush=True)

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
                      "modale": modale, "n_modale": n_modale,
                      "n_distinte": len(cnt),
                      "tok_medio": sum(tok) / k, "frac_finiti": sum(fin) / k,
                      "risposte_lista": "|".join(str(r) for r in chiavi),
                      "tok_lista": "|".join(str(t) for t in tok)})
    df = pd.DataFrame(righe)
    nome = "/vol/e12_s1_%s.parquet" % model.replace("/", "__")
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe | accordo medio %.3f | risposte medie %.3f | finiti %.3f"
          % (nome, len(df), df.accordo.mean(), df.risposte.mean(), df.frac_finiti.mean()), flush=True)
    return {"file": nome, "n": len(df), "secondi": round(dt),
            "accordo_medio": round(float(df.accordo.mean()), 4),
            "risposte_medie": round(float(df.risposte.mean()), 4)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    print(campiona.remote(model=model))
