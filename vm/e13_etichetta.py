"""E13 — etichettatura con il regime LORO: 10 campioni sull'originale, 1 per variante accettata.

Uso (nel venv, HF_HOME=/work/hf):
  python e13_etichetta.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --cond A --draw D1 --seed 20260921 --gpu 0

Condizioni (PREREG_E13.md): A = D1 + seme a; B = D2 + seme b; Ap = D1 + seme c.
Campionamento: T=0.6, top_p=0.95, max_tokens=8192, seme fissato per condizione.
"corretto" = risposta uguale al riferimento gpt-oss del file delle varianti.
Salva le risposte grezze; l'etichetta (base_ok, robust) si calcola offline in e13_analisi.py.
Checkpoint a blocchi. Nessun testo di problema nei print.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cond", required=True, choices=["A", "B", "Ap"])
ap.add_argument("--draw", required=True, choices=["D1", "D2"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--gpu", default="0")
ap.add_argument("--n", type=int, default=164)
ap.add_argument("--n-base", type=int, default=10)
ap.add_argument("--prompt", choices=["nudo", "esplicito"], default="esplicito",
                help="nudo = solo il problema (DeepSeek-R1 chiude da solo con \\boxed); "
                     "esplicito = con richiesta del formato (necessario per Qwen/Skywork/Olmo, "
                     "misurato il 9/9: col prompt nudo Qwen produce 0.03 risposte su 10)")
ap.add_argument("--out", default="/work/aimo/e13")
args = ap.parse_args()
ISTR = "Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}.\n\n"


def chiedi(testo):
    return testo if args.prompt == "nudo" else ISTR + testo
os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
os.environ.setdefault("HF_HOME", "/work/hf")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

sys.path.insert(0, "/work/aimo/gpu")
from perturbazioni import risposta  # noqa: E402
import pandas as pd  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

T, TOP_P, MAX_TOK = 0.6, 0.95, 16384   # 16384: emendamento pre-esecuzione, vedi PREREG_E13 (prova: 2/3 problemi saturano 8192)
BLOCCO_PROBLEMI = 24   # 24 problemi per blocco: 240 campioni base + ~1000 varianti


def main():
    var = pd.read_parquet("%s/varianti_%s.parquet" % (args.out, args.draw))
    ids = [l.strip() for l in open("/work/aimo/gpu/e12_ids.txt", encoding="utf-8") if l.strip()][:args.n]
    orig = var[(var.idx == -1) & var.problem_id.isin(ids)].set_index("problem_id")
    acc = var[(var.idx >= 0) & var.accettata & var.problem_id.isin(ids)]
    tag = args.model.replace("/", "__")
    parz = "%s/etich_%s_%s_parziale.parquet" % (args.out, tag, args.cond)
    finale = "%s/etich_%s_%s.parquet" % (args.out, tag, args.cond)
    righe, fatti = [], set()
    if os.path.isfile(parz):
        df0 = pd.read_parquet(parz)
        righe = df0.to_dict("records")
        fatti = set(df0.problem_id)
    da_fare = [p for p in ids if p in orig.index and p not in fatti]
    print("E13 etichetta modello=%s cond=%s draw=%s seed=%d problemi=%d da_fare=%d varianti_accettate=%d prompt=%s"
          % (args.model, args.cond, args.draw, args.seed, len(ids), len(da_fare), len(acc), args.prompt), flush=True)

    t0 = time.time()
    kw = {}
    if "gpt-oss" not in args.model:
        kw["dtype"] = "bfloat16"
    llm = LLM(model=args.model, max_model_len=MAX_TOK + 2560, gpu_memory_utilization=0.92,
              trust_remote_code=True, seed=args.seed, **kw)
    print("vLLM caricato in %.0fs" % (time.time() - t0), flush=True)
    ctk = {"reasoning_effort": "low"} if "gpt-oss" in args.model else {}
    sp_base = SamplingParams(temperature=T, top_p=TOP_P, max_tokens=MAX_TOK, n=args.n_base, seed=args.seed)
    sp_var = SamplingParams(temperature=T, top_p=TOP_P, max_tokens=MAX_TOK, n=1, seed=args.seed)

    tot = time.time()
    for i in range(0, len(da_fare), BLOCCO_PROBLEMI):
        blocco = da_fare[i:i + BLOCCO_PROBLEMI]
        t1 = time.time()
        convs_b = [[{"role": "user", "content": chiedi(orig.loc[p, "testo"])}] for p in blocco]
        outs_b = llm.chat(convs_b, sp_base, chat_template_kwargs=ctk, use_tqdm=False) if ctk else llm.chat(convs_b, sp_base, use_tqdm=False)
        va = acc[acc.problem_id.isin(blocco)].reset_index(drop=True)
        convs_v = [[{"role": "user", "content": chiedi(t)}] for t in va.testo]
        outs_v = (llm.chat(convs_v, sp_var, chat_template_kwargs=ctk, use_tqdm=False) if ctk else llm.chat(convs_v, sp_var, use_tqdm=False)) if len(convs_v) else []
        for p, o in zip(blocco, outs_b):
            rif = orig.loc[p, "risp_ref"]
            rs = [risposta(c.text) for c in o.outputs]
            righe.append({"problem_id": p, "famiglia": "original", "idx": -1, "n": len(rs),
                          "risposte": "|".join(str(r) for r in rs), "n_corrette": sum(1 for r in rs if r is not None and r == rif),
                          "n_con_risposta": sum(1 for r in rs if r is not None),
                          "tok_medio": sum(len(c.token_ids) for c in o.outputs) / max(1, len(rs))})
        for r, o in zip(va.itertuples(), outs_v):
            rs = [risposta(c.text) for c in o.outputs]
            righe.append({"problem_id": r.problem_id, "famiglia": r.famiglia, "idx": int(r.idx), "n": len(rs),
                          "risposte": "|".join(str(x) for x in rs), "n_corrette": sum(1 for x in rs if x is not None and x == r.risp_ref),
                          "n_con_risposta": sum(1 for x in rs if x is not None),
                          "tok_medio": sum(len(c.token_ids) for c in o.outputs) / max(1, len(rs))})
        pd.DataFrame(righe).to_parquet(parz)
        print("blocco %d/%d: %d problemi, %d varianti in %.0fs | righe %d"
              % (i // BLOCCO_PROBLEMI + 1, (len(da_fare) + BLOCCO_PROBLEMI - 1) // BLOCCO_PROBLEMI, len(blocco), len(va), time.time() - t1, len(righe)), flush=True)
    df = pd.DataFrame(righe).drop_duplicates(["problem_id", "famiglia", "idx"])
    df["model"], df["cond"], df["draw"], df["seed"] = args.model, args.cond, args.draw, args.seed
    df.to_parquet(finale)
    b = df[df.idx == -1]
    print("scritto %s: %d righe | base: corrette medie %.2f/%d, con risposta %.2f | varianti corrette %.3f | %.0fs"
          % (finale, len(df), b.n_corrette.mean(), args.n_base, b.n_con_risposta.mean(),
             (df[df.idx >= 0].n_corrette.mean() if (df.idx >= 0).any() else float("nan")), time.time() - tot), flush=True)


if __name__ == "__main__":
    main()
