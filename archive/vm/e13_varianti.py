"""E13 — varianti nelle famiglie loro, due estrazioni indipendenti, con cancello. Sulla VM, vLLM locale.

Uso (nel venv, HF_HOME=/work/hf):
  python e13_varianti.py --draw D1 --seed 20260913 [--n 164] [--k 10]
  python e13_varianti.py --draw D2 --seed 20260914

Protocollo: PREREG_E13.md + PREREG_E12_DEVIAZIONE.md. Identico a e12_s2_genera_modal.py salvo:
k=10 varianti per famiglia, seme per estrazione, esecuzione locale, output /work/aimo/e13/.
Passo 1: varianti (gpt-oss-120b, reasoning low). Passo 2: risoluzione di originali + varianti
(reasoning medium). Cancello: testo diverso, rapporto lunghezza in [0.7, 4.0], risposta uguale
al riferimento. Checkpoint a blocchi. Nessuna etichetta. Nessun testo di problema nei print.
"""
import argparse
import json
import os
import random
import re
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--draw", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--n", type=int, default=164)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--model", default="openai/gpt-oss-120b")
ap.add_argument("--gpu", default="0")
ap.add_argument("--out", default="/work/aimo/e13")
args = ap.parse_args()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
os.environ.setdefault("HF_HOME", "/work/hf")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, "/work/aimo/gpu")
from perturbazioni import risposta  # noqa: E402
import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

N_FEWSHOT = 5
EFFORT_GEN, EFFORT_SOLVE = "low", "medium"
MAX_TOK_GEN, MAX_TOK_SOLVE = 4096, 8192   # 8192 per il risolutore: emendamento pre-esecuzione (prova: 5/30 varianti senza risposta a 4096)
RATIO_MIN, RATIO_MAX = 0.7, 4.0
BLOCCO = 200
FAMIGLIE = {
    "rephrase": "Rewrite the problem in different words while preserving exactly its mathematical content and its final answer.",
    "rename": "Rename the variables, people, objects or entities in the problem while preserving its mathematical structure and its final answer.",
    "domain": "Transfer the problem to a different real-world surface domain (a different story, setting and objects) while preserving exactly the same mathematical structure and the same final answer.",
    "distract": "Insert irrelevant but plausible additional information (extra facts, numbers or context) that does not change the problem or its final answer.",
    "typos": "Introduce harmless typographical or formatting noise (misspellings, spacing, punctuation, capitalisation) without changing the meaning of the problem or its final answer.",
}
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]


def problemi_pubblici():
    v = {}
    for nome in FONTI:
        for r in json.load(open("/work/aimo/dati/%s.json" % nome, encoding="utf-8")):
            pid, t = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(t, str) and pid not in v:
                v[pid] = t
    return v


def prompt_genera(fam, esempi, problema, k):
    righe = ["You create variants of mathematics competition problems for a robustness benchmark.",
             "Perturbation type: %s. %s" % (fam, FAMIGLIE[fam]),
             "The variant must be a complete, self-contained problem with exactly the same final answer as the original. Do not solve the problem. Do not add hints or solutions.",
             "", "Here are %d examples of this perturbation type:" % len(esempi), ""]
    for i, e in enumerate(esempi, 1):
        righe += ["Example %d" % i, "Original: %s" % e["originale"], "Variant: %s" % e["variante"], ""]
    righe += ["Now write %d different variants of the following problem. Make them genuinely different from each other. Output each variant between <variant> and </variant> tags, and nothing else." % k,
              "", "Original: %s" % problema]
    return "\n".join(righe)


def prompt_risolvi(testo):
    return "Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}.\n\n" + testo


def estrai_varianti(txt, k):
    vs = [v.strip() for v in re.findall(r"<variant>(.*?)</variant>", txt, flags=re.S)]
    out = []
    for v in vs:
        if v and v not in out:
            out.append(v)
    return out[-k:]


def main():
    ids = [l.strip() for l in open("/work/aimo/gpu/e12_ids.txt", encoding="utf-8") if l.strip()][:args.n]
    problemi = problemi_pubblici()
    ids = [p for p in ids if p in problemi]
    pool = json.load(open("/work/aimo/gpu/e12_fewshot.json", encoding="utf-8"))
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    gen_parz = "%s/varianti_%s_gen_parziale.parquet" % (args.out, args.draw)
    sol_parz = "%s/varianti_%s_sol_parziale.parquet" % (args.out, args.draw)
    finale = "%s/varianti_%s.parquet" % (args.out, args.draw)
    print("E13 varianti draw=%s seed=%d problemi=%d k=%d modello=%s" % (args.draw, args.seed, len(ids), args.k, args.model), flush=True)

    t0 = time.time()
    llm = LLM(model=args.model, max_model_len=16384, gpu_memory_utilization=0.92, trust_remote_code=True, seed=args.seed)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)
    sp_gen = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_TOK_GEN, seed=args.seed)
    sp_sol = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=MAX_TOK_SOLVE, seed=args.seed)

    # passo 1
    righe_gen, fatte = [], set()
    if os.path.isfile(gen_parz):
        df0 = pd.read_parquet(gen_parz)
        righe_gen = df0.to_dict("records")
        fatte = set(zip(df0.problem_id, df0.famiglia))
    lavori = [(p, f) for p in ids for f in FAMIGLIE if (p, f) not in fatte]
    # il campionamento few-shot deve essere riproducibile per (draw, problema, famiglia)
    for i in range(0, len(lavori), BLOCCO):
        blocco = lavori[i:i + BLOCCO]
        convs = []
        for p, f in blocco:
            r2 = random.Random("%d|%s|%s" % (args.seed, p, f))
            cand = [e for e in pool[f] if e["problem_id"] != p]
            esempi = r2.sample(cand, min(N_FEWSHOT, len(cand)))
            convs.append([{"role": "user", "content": prompt_genera(f, esempi, problemi[p], args.k)}])
        t1 = time.time()
        outs = llm.chat(convs, sp_gen, chat_template_kwargs={"reasoning_effort": EFFORT_GEN}, use_tqdm=False)
        for (p, f), o in zip(blocco, outs):
            vs = estrai_varianti(o.outputs[0].text, args.k)
            righe_gen.append({"problem_id": p, "famiglia": f, "n_varianti": len(vs),
                              "varianti": json.dumps(vs, ensure_ascii=False), "tok": len(o.outputs[0].token_ids)})
        pd.DataFrame(righe_gen).to_parquet(gen_parz)
        print("passo 1 blocco %d/%d in %.0fs | coppie %d" % (i // BLOCCO + 1, (len(lavori) + BLOCCO - 1) // BLOCCO, time.time() - t1, len(righe_gen)), flush=True)
    gen = pd.DataFrame(righe_gen)
    print("passo 1 finito: varianti %d, media per coppia %.1f, coppie vuote %d" % (int(gen.n_varianti.sum()), gen.n_varianti.mean(), int((gen.n_varianti == 0).sum())), flush=True)

    # passo 2
    righe_sol, fatte = [], set()
    if os.path.isfile(sol_parz):
        df0 = pd.read_parquet(sol_parz)
        righe_sol = df0.to_dict("records")
        fatte = set(zip(df0.problem_id, df0.famiglia, df0.idx))
    lavori = [(p, "original", -1, problemi[p]) for p in ids]
    for r in gen.itertuples():
        for j, v in enumerate(json.loads(r.varianti)):
            lavori.append((r.problem_id, r.famiglia, j, v))
    lavori = [l for l in lavori if (l[0], l[1], l[2]) not in fatte]
    print("passo 2: soluzioni da fare %d" % len(lavori), flush=True)
    for i in range(0, len(lavori), BLOCCO):
        blocco = lavori[i:i + BLOCCO]
        convs = [[{"role": "user", "content": prompt_risolvi(t)}] for _, _, _, t in blocco]
        t1 = time.time()
        outs = llm.chat(convs, sp_sol, chat_template_kwargs={"reasoning_effort": EFFORT_SOLVE}, use_tqdm=False)
        for (p, f, j, t), o in zip(blocco, outs):
            righe_sol.append({"problem_id": p, "famiglia": f, "idx": j, "testo": t,
                              "risp_gptoss": risposta(o.outputs[0].text),
                              "tok_sol": len(o.outputs[0].token_ids), "finito": o.outputs[0].finish_reason == "stop"})
        pd.DataFrame(righe_sol).to_parquet(sol_parz)
        print("passo 2 blocco %d/%d in %.0fs | soluzioni %d" % (i // BLOCCO + 1, (len(lavori) + BLOCCO - 1) // BLOCCO, time.time() - t1, len(righe_sol)), flush=True)
    sol = pd.DataFrame(righe_sol)

    # cancello
    ref = {r.problem_id: (r.risp_gptoss if isinstance(r.risp_gptoss, str) else None) for r in sol[sol.idx == -1].itertuples()}
    righe = []
    for r in sol.itertuples():
        rif = ref.get(r.problem_id)
        rg = r.risp_gptoss if isinstance(r.risp_gptoss, str) else None
        if r.idx == -1:
            righe.append({"problem_id": r.problem_id, "famiglia": "original", "idx": -1, "testo": r.testo, "risp_gptoss": rg,
                          "risp_ref": rif, "len_ratio": 1.0, "accettata": rif is not None, "motivo": "" if rif is not None else "senza_riferimento"})
            continue
        orig = problemi[r.problem_id]
        ratio = len(r.testo) / max(1, len(orig))
        if rif is None:
            acc, mot = False, "senza_riferimento"
        elif r.testo.strip() == orig.strip():
            acc, mot = False, "identica"
        elif not (RATIO_MIN <= ratio <= RATIO_MAX):
            acc, mot = False, "lunghezza"
        elif rg is None:
            acc, mot = False, "variante_senza_risposta"
        elif rg != rif:
            acc, mot = False, "risposta_diversa"
        else:
            acc, mot = True, ""
        righe.append({"problem_id": r.problem_id, "famiglia": r.famiglia, "idx": r.idx, "testo": r.testo, "risp_gptoss": rg,
                      "risp_ref": rif, "len_ratio": ratio, "accettata": acc, "motivo": mot})
    df = pd.DataFrame(righe)
    df.to_parquet(finale)
    v = df[df.idx >= 0]
    print("scritto %s: riferimenti %d/%d | varianti %d, accettate %d (%.1f%%) | motivi %s | per famiglia %s"
          % (finale, sum(1 for x in ref.values() if x is not None), len(ids), len(v), int(v.accettata.sum()), 100 * v.accettata.mean(),
             v[~v.accettata].motivo.value_counts().to_dict(),
             {f: round(float(v[v.famiglia == f].accettata.mean()), 3) for f in FAMIGLIE}), flush=True)


if __name__ == "__main__":
    main()
