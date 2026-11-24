"""E12 / Stadio 2, passo A — varianti nelle famiglie LORO con gpt-oss-120b, e cancello di qualita'.

Uso:
  modal run e12_s2_genera_modal.py

Protocollo: PREREG_E12.md + PREREG_E12_DEVIAZIONE.md, fissati prima di eseguire.
  1. per ogni (problema, famiglia): k=3 varianti, reasoning low, 5 esempi few-shot loro;
  2. gpt-oss-120b risolve originale e ogni variante (reasoning medium);
  3. cancello: testo diverso, rapporto lunghezza in [0.7, 4.0], risposta == riferimento.
Salvataggi parziali a blocchi sul volume (preemption). Nessuna etichetta nel container.
Nessun testo di problema nei print.
"""
import modal

app = modal.App("aimo-e12-s2-gen")
_SEP = "\\" if "\\" in __file__ else "/"
_GPU = __file__.rsplit(_SEP, 1)[0]
IMG = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
       .pip_install("vllm", "pandas", "pyarrow", "huggingface-hub", "transformers", "openai-harmony")
       .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
       .add_local_file(_GPU + _SEP + "perturbazioni.py", "/root/perturbazioni.py")
       .add_local_file(_GPU + _SEP + "e12_ids.txt", "/root/e12_ids.txt")
       .add_local_file(_GPU + _SEP + "e12_fewshot.json", "/root/e12_fewshot.json")
       .add_local_dir(_GPU.rsplit(_SEP, 1)[0] + _SEP + "dati", "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
SEME = 20260906
K_VAR = 3
N_FEWSHOT = 5
EFFORT_GEN = "low"
EFFORT_SOLVE = "medium"
MAX_TOK_GEN = 2048
MAX_TOK_SOLVE = 4096
RATIO_MIN, RATIO_MAX = 0.7, 4.0
BLOCCO = 160
FAMIGLIE = {
    "rephrase": "Rewrite the problem in different words while preserving exactly its mathematical content and its final answer.",
    "rename": "Rename the variables, people, objects or entities in the problem while preserving its mathematical structure and its final answer.",
    "domain": "Transfer the problem to a different real-world surface domain (a different story, setting and objects) while preserving exactly the same mathematical structure and the same final answer.",
    "distract": "Insert irrelevant but plausible additional information (extra facts, numbers or context) that does not change the problem or its final answer.",
    "typos": "Introduce harmless typographical or formatting noise (misspellings, spacing, punctuation, capitalisation) without changing the meaning of the problem or its final answer.",
}


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


def _prompt_genera(fam, esempi, problema):
    righe = ["You create variants of mathematics competition problems for a robustness benchmark.",
             "Perturbation type: %s. %s" % (fam, FAMIGLIE[fam]),
             "The variant must be a complete, self-contained problem with exactly the same final answer as the original. Do not solve the problem. Do not add hints or solutions.",
             "", "Here are %d examples of this perturbation type:" % len(esempi), ""]
    for i, e in enumerate(esempi, 1):
        righe += ["Example %d" % i, "Original: %s" % e["originale"], "Variant: %s" % e["variante"], ""]
    righe += ["Now write %d different variants of the following problem. Output each variant between <variant> and </variant> tags, and nothing else." % K_VAR,
              "", "Original: %s" % problema]
    return "\n".join(righe)


def _prompt_risolvi(testo):
    return ("Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}.\n\n" + testo)


def _estrai_varianti(txt):
    import re
    vs = [v.strip() for v in re.findall(r"<variant>(.*?)</variant>", txt, flags=re.S)]
    vs = [v for v in vs if v]
    out = []
    for v in vs:
        if v not in out:
            out.append(v)
    return out[-K_VAR:]


@app.function(image=IMG, gpu="H100", volumes={"/vol": VOL}, timeout=4 * 3600)
def genera(model: str = "openai/gpt-oss-120b"):
    import json
    import os
    import random
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    ids = [l.strip() for l in open("/root/e12_ids.txt", encoding="utf-8") if l.strip()]
    problemi = _problemi()
    ids = [p for p in ids if p in problemi]
    pool = json.load(open("/root/e12_fewshot.json", encoding="utf-8"))
    rng = random.Random(SEME)
    print("modello=%s problemi=%d famiglie=%d k=%d effort_gen=%s effort_solve=%s"
          % (model, len(ids), len(FAMIGLIE), K_VAR, EFFORT_GEN, EFFORT_SOLVE), flush=True)

    t0 = time.time()
    llm = LLM(model=model, max_model_len=12288, gpu_memory_utilization=0.92, trust_remote_code=True)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)
    sp_gen = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_TOK_GEN, seed=SEME)
    sp_sol = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_TOK_SOLVE, seed=SEME)

    # ---- passo 1: varianti, a blocchi con checkpoint
    gen_parz = "/vol/e12_s2_gen_parziale.parquet"
    fatte = set()
    righe_gen = []
    if os.path.isfile(gen_parz):
        df0 = pd.read_parquet(gen_parz)
        righe_gen = df0.to_dict("records")
        fatte = set(zip(df0.problem_id, df0.famiglia))
    lavori = [(p, f) for p in ids for f in FAMIGLIE if (p, f) not in fatte]
    print("passo 1: coppie (problema,famiglia) da fare %d, gia' fatte %d" % (len(lavori), len(fatte)), flush=True)
    for i in range(0, len(lavori), BLOCCO):
        blocco = lavori[i:i + BLOCCO]
        convs = []
        for p, f in blocco:
            cand = [e for e in pool[f] if e["problem_id"] != p]
            esempi = rng.sample(cand, min(N_FEWSHOT, len(cand)))
            convs.append([{"role": "user", "content": _prompt_genera(f, esempi, problemi[p])}])
        t1 = time.time()
        outs = llm.chat(convs, sp_gen, chat_template_kwargs={"reasoning_effort": EFFORT_GEN}, use_tqdm=False)
        for (p, f), o in zip(blocco, outs):
            vs = _estrai_varianti(o.outputs[0].text)
            righe_gen.append({"problem_id": p, "famiglia": f, "n_varianti": len(vs),
                              "varianti": json.dumps(vs, ensure_ascii=False),
                              "tok": len(o.outputs[0].token_ids)})
        pd.DataFrame(righe_gen).to_parquet(gen_parz)
        VOL.commit()
        print("passo 1 blocco %d/%d in %.0fs | coppie salvate %d" % (i // BLOCCO + 1, (len(lavori) + BLOCCO - 1) // BLOCCO, time.time() - t1, len(righe_gen)), flush=True)
    gen = pd.DataFrame(righe_gen)
    n_var = int(gen.n_varianti.sum())
    print("passo 1 finito: varianti totali %d, coppie senza varianti %d" % (n_var, int((gen.n_varianti == 0).sum())), flush=True)

    # ---- passo 2: risoluzione di originali + varianti, a blocchi con checkpoint
    sol_parz = "/vol/e12_s2_sol_parziale.parquet"
    righe_sol = []
    fatte = set()
    if os.path.isfile(sol_parz):
        df0 = pd.read_parquet(sol_parz)
        righe_sol = df0.to_dict("records")
        fatte = set(zip(df0.problem_id, df0.famiglia, df0.idx))
    lavori = [(p, "original", -1, problemi[p]) for p in ids]
    for r in gen.itertuples():
        for j, v in enumerate(json.loads(r.varianti)):
            lavori.append((r.problem_id, r.famiglia, j, v))
    lavori = [l for l in lavori if (l[0], l[1], l[2]) not in fatte]
    print("passo 2: soluzioni da fare %d, gia' fatte %d" % (len(lavori), len(fatte)), flush=True)
    for i in range(0, len(lavori), BLOCCO):
        blocco = lavori[i:i + BLOCCO]
        convs = [[{"role": "user", "content": _prompt_risolvi(t)}] for _, _, _, t in blocco]
        t1 = time.time()
        outs = llm.chat(convs, sp_sol, chat_template_kwargs={"reasoning_effort": EFFORT_SOLVE}, use_tqdm=False)
        for (p, f, j, t), o in zip(blocco, outs):
            righe_sol.append({"problem_id": p, "famiglia": f, "idx": j, "testo": t,
                              "risp_gptoss": risposta(o.outputs[0].text),
                              "tok_sol": len(o.outputs[0].token_ids),
                              "finito": o.outputs[0].finish_reason == "stop"})
        pd.DataFrame(righe_sol).to_parquet(sol_parz)
        VOL.commit()
        print("passo 2 blocco %d/%d in %.0fs | soluzioni salvate %d" % (i // BLOCCO + 1, (len(lavori) + BLOCCO - 1) // BLOCCO, time.time() - t1, len(righe_sol)), flush=True)
    sol = pd.DataFrame(righe_sol)

    # ---- cancello di qualita'
    ref = {r.problem_id: r.risp_gptoss for r in sol[sol.idx == -1].itertuples()}
    righe = []
    for r in sol.itertuples():
        rif = ref.get(r.problem_id)
        if r.idx == -1:
            righe.append({"problem_id": r.problem_id, "famiglia": "original", "idx": -1, "testo": r.testo,
                          "risp_gptoss": r.risp_gptoss, "risp_ref": rif, "len_ratio": 1.0,
                          "accettata": rif is not None, "motivo": "" if rif is not None else "senza_riferimento"})
            continue
        orig = problemi[r.problem_id]
        ratio = len(r.testo) / max(1, len(orig))
        if rif is None:
            acc, mot = False, "senza_riferimento"
        elif r.testo.strip() == orig.strip():
            acc, mot = False, "identica"
        elif not (RATIO_MIN <= ratio <= RATIO_MAX):
            acc, mot = False, "lunghezza"
        elif r.risp_gptoss is None:
            acc, mot = False, "variante_senza_risposta"
        elif r.risp_gptoss != rif:
            acc, mot = False, "risposta_diversa"
        else:
            acc, mot = True, ""
        righe.append({"problem_id": r.problem_id, "famiglia": r.famiglia, "idx": r.idx, "testo": r.testo,
                      "risp_gptoss": r.risp_gptoss, "risp_ref": rif, "len_ratio": ratio,
                      "accettata": acc, "motivo": mot})
    df = pd.DataFrame(righe)
    nome = "/vol/e12_s2_varianti.parquet"
    df.to_parquet(nome)
    VOL.commit()
    v = df[df.idx >= 0]
    print("scritto %s: originali con riferimento %d/%d | varianti %d, accettate %d (%.1f%%) | motivi: %s"
          % (nome, sum(1 for x in ref.values() if x is not None), len(ids), len(v), int(v.accettata.sum()),
             100 * v.accettata.mean(), v[~v.accettata].motivo.value_counts().to_dict()), flush=True)
    print("accettate per famiglia: %s" % v[v.accettata].famiglia.value_counts().to_dict(), flush=True)
    return {"file": nome, "varianti": len(v), "accettate": int(v.accettata.sum())}


@app.local_entrypoint()
def main(model: str = "openai/gpt-oss-120b"):
    print(genera.remote(model=model))
