"""E10 fase A — traduzione dei problemi pubblici con un modello ESTERNO ai cinque valutati.

Uso:
  modal run e10_traduci_modal.py --translator microsoft/Phi-4 --lingue it,fr,de

La traduzione e' una perturbazione SEMANTICA che non compare fra le famiglie degli
organizzatori (distract, domain, rename, rephrase, typos, expert_*, paraphrase).
Il traduttore NON e' uno dei modelli valutati e non e' della loro famiglia, per non
produrre testi "comodi" per il solutore.

Cancello di qualita' (deterministico): la traduzione e' accettata solo se conserva
ESATTAMENTE il multiinsieme dei segmenti LaTeX inline `$...$` e delle cifre dell'originale.
Una traduzione che tocca la matematica non e' una perturbazione logic-preserving e viene
scartata (ok_gate=False). Nessun testo di problema nei print: solo id e conteggi.
"""
import modal

app = modal.App("aimo-e10-trad")
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_dir(__file__.replace("gpu" + ("\\" if "\\" in __file__ else "/") + "e10_traduci_modal.py", "dati"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
LINGUE = {"it": "Italian", "fr": "French", "de": "German", "es": "Spanish"}


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


def _firma(t: str):
    """Multiinsieme di segmenti LaTeX inline e di cifre: cio' che la traduzione deve conservare."""
    import re
    from collections import Counter
    latex = Counter(re.sub(r"\s+", "", m) for m in re.findall(r"\$[^$]+\$", t))
    cifre = Counter(re.findall(r"\d+", re.sub(r"\$[^$]+\$", "", t)))
    return latex, cifre


def _prompt(testo: str, lingua: str) -> str:
    return ("Translate the following mathematics problem into %s.\n"
            "Rules: keep every mathematical expression, LaTeX command, dollar-delimited formula, "
            "number, symbol and variable name EXACTLY as written; translate only the natural-language "
            "words; do not solve, comment, or add anything. Output only the translated problem.\n\n"
            "%s" % (lingua, testo))


@app.function(image=IMG, gpu="A100-80GB", volumes={"/vol": VOL}, timeout=3 * 3600)
def traduci(translator: str, lingue: str = "it,fr,de", limite: int = 0):
    import time
    import pandas as pd
    from vllm import LLM, SamplingParams

    problemi = _problemi()
    ids = sorted(problemi)
    if limite:
        ids = ids[:limite]
    codici = [c for c in lingue.split(",") if c in LINGUE]
    print("traduttore=%s problemi=%d lingue=%s" % (translator, len(ids), codici), flush=True)

    t0 = time.time()
    llm = LLM(model=translator, dtype="bfloat16", max_model_len=6144, gpu_memory_utilization=0.90,
              trust_remote_code=True)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)
    sp = SamplingParams(temperature=0.0, max_tokens=1536)

    lavori = [(pid, c) for pid in ids for c in codici]
    convs = [[{"role": "user", "content": _prompt(problemi[pid], LINGUE[c])}] for pid, c in lavori]
    t0 = time.time()
    outs = llm.chat(convs, sp, use_tqdm=False)
    print("traduzioni=%d in %.0fs" % (len(lavori), time.time() - t0), flush=True)

    righe = []
    for (pid, c), o in zip(lavori, outs):
        tr = o.outputs[0].text.strip()
        # rimuove eventuali blocchi <think> di modelli "thinking"
        if "</think>" in tr:
            tr = tr.split("</think>")[-1].strip()
        fo, fc = _firma(problemi[pid])
        to, tc = _firma(tr)
        ok = (fo == to) and (fc == tc) and len(tr) > 20 and tr != problemi[pid]
        righe.append({"problem_id": pid, "lang": c, "traduzione": tr, "ok_gate": bool(ok),
                      "n_latex": sum(fo.values()), "finito": o.outputs[0].finish_reason == "stop"})
    df = pd.DataFrame(righe)
    nome = "/vol/e10_traduzioni_%s.parquet" % translator.replace("/", "__")
    df.to_parquet(nome)
    VOL.commit()
    per_lingua = df.groupby("lang").ok_gate.mean().round(3).to_dict()
    print("scritto %s: %d traduzioni | ok_gate per lingua %s" % (nome, len(df), per_lingua), flush=True)
    return {"file": nome, "n": len(df), "ok_gate_per_lingua": per_lingua}


@app.local_entrypoint()
def main(translator: str = "microsoft/Phi-4", lingue: str = "it,fr,de", limite: int = 0):
    print(traduci.remote(translator=translator, lingue=lingue, limite=limite))
