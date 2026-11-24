"""E10 fase B — etichetta di robustezza ALLA TRADUZIONE per un modello valutato.

Uso:
  modal run e10_modal.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --tag stadio1 \
      --translator microsoft/Phi-4

Legge /vol/e10_traduzioni_<translator>.parquet (fase A), tiene solo le traduzioni che hanno
passato il cancello di qualita', e fa girare il modello valutato (greedy, 4096 token) su
originale + traduzioni. Etichetta: robust_trad = l'originale ha una risposta E ogni
traduzione accettata da' la stessa risposta finale. Traduzione senza risposta = instabile.
Nessun testo di problema nei print.
"""
import modal

app = modal.App("aimo-e10")
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_file(__file__.replace("e10_modal.py", "perturbazioni.py"), "/root/perturbazioni.py")
       .add_local_dir(__file__.replace("gpu" + ("\\" if "\\" in __file__ else "/") + "e10_modal.py", "dati"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
MAX_TOK = 4096


def _problemi():
    import json
    import os
    visti = {}
    for n in FONTI:
        p = os.path.join("/root/dati", n + ".json")
        for r in json.load(open(p, encoding="utf-8")):
            pid = r.get("problem_id")
            t = r.get("original_problem")
            if pid and isinstance(t, str) and pid not in visti:
                visti[pid] = t
    return visti


@app.function(image=IMG, gpu="A100-40GB", volumes={"/vol": VOL}, timeout=4 * 3600)
def etichetta_trad(model: str, tag: str, translator: str, limite: int = 0):
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    VOL.reload()
    trad = pd.read_parquet("/vol/e10_traduzioni_%s.parquet" % translator.replace("/", "__"))
    trad = trad[trad.ok_gate]
    problemi = _problemi()
    ids = sorted(set(trad.problem_id) & set(problemi))
    if limite:
        ids = ids[:limite]
    per_pid = {pid: dict(zip(g.lang, g.traduzione)) for pid, g in trad.groupby("problem_id")}
    print("modello=%s problemi=%d traduzioni accettate=%d" % (model, len(ids), len(trad)), flush=True)

    t0 = time.time()
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92,
              trust_remote_code=True)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOK)

    lavori = []
    for pid in ids:
        lavori.append((pid, "original", problemi[pid]))
        for lang, testo in per_pid.get(pid, {}).items():
            lavori.append((pid, lang, testo))
    convs = [[{"role": "user", "content": t}] for _, _, t in lavori]
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False)
    dt = time.time() - t0
    print("generazioni=%d in %.0fs (%.1f s/gen)" % (len(lavori), dt, dt / max(1, len(lavori))), flush=True)

    per = {}
    for (pid, k, _), o in zip(lavori, outs):
        per.setdefault(pid, {})[k] = {"risposta": risposta(o.outputs[0].text),
                                      "token": len(o.outputs[0].token_ids)}
    righe = []
    for pid in ids:
        d = per[pid]
        orig = d["original"]["risposta"]
        langs = [k for k in d if k != "original"]
        stab = {k: (d[k]["risposta"] is not None and d[k]["risposta"] == orig) for k in langs}
        righe.append({"problem_id": pid, "model": model, "risposta_orig": orig,
                      "ha_risposta": orig is not None, "n_lingue": len(langs),
                      "n_stabili": sum(stab.values()),
                      "robust_trad": bool(orig is not None and langs and all(stab.values())),
                      **{"stab_" + k: stab[k] for k in langs},
                      **{"tok_" + k: d[k]["token"] for k in ["original"] + langs}})
    df = pd.DataFrame(righe)
    nome = "/vol/e10_%s_%s.parquet" % (tag, model.replace("/", "__"))
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe | con risposta %d | robust_trad %d" % (
        nome, len(df), int(df.ha_risposta.sum()), int(df.robust_trad.sum())), flush=True)
    return {"file": nome, "n": len(df), "con_risposta": int(df.ha_risposta.sum()),
            "robust_trad": int(df.robust_trad.sum()), "secondi": round(dt)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", tag: str = "stadio1",
         translator: str = "microsoft/Phi-4", limite: int = 0):
    print(etichetta_trad.remote(model=model, tag=tag, translator=translator, limite=limite))
