"""E13 — cancello di fattibilita' per modello, PRIMA di spendere ore di etichettatura.

Uso: python e13_fattibilita.py --model Qwen/Qwen3.5-4B --gpu 1

Stima `base_ok` su un campione di problemi con pochi campioni ciascuno, usando il prompt
ESPLICITO. Se il modello non risolve abbastanza problemi originali, l'etichetta di robustezza
e' costante per costruzione e il test-retest non e' definito: il modello non entra in E13 e la
cosa viene dichiarata invece di essere scoperta dopo sei ore di GPU.

Criterio, fissato prima: il modello ENTRA se almeno il 15% dei problemi campionati ha
>= 60% di campioni con risposta uguale al riferimento gpt-oss.
"""
import argparse
import json
import os
import sys

SOGLIA_INGRESSO = 0.15
FRAZ_BASE_OK = 0.6
ISTR = "Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}.\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max-tok", type=int, default=16384)
    ap.add_argument("--out", default="/work/aimo/e13/fattibilita.json")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/work/hf")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    sys.path.insert(0, "/work/aimo/gpu")
    from perturbazioni import risposta
    import pandas as pd
    from vllm import LLM, SamplingParams

    var = pd.read_parquet("/work/aimo/e13/varianti_D1.parquet")
    orig = var[var.idx == -1].set_index("problem_id")
    ids = [p for p in orig.index if isinstance(orig.loc[p, "risp_ref"], str)][:args.n]
    print("modello=%s problemi campionati=%d k=%d prompt=esplicito" % (args.model, len(ids), args.k), flush=True)

    kw = {} if "gpt-oss" in args.model else {"dtype": "bfloat16"}
    llm = LLM(model=args.model, max_model_len=args.max_tok + 2560, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=20260909, **kw)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tok, n=args.k, seed=20260909)
    convs = [[{"role": "user", "content": ISTR + orig.loc[p, "testo"]}] for p in ids]
    outs = llm.chat(convs, sp, use_tqdm=False)

    righe = []
    for p, o in zip(ids, outs):
        rif = orig.loc[p, "risp_ref"]
        rs = [risposta(c.text) for c in o.outputs]
        n_ok = sum(1 for r in rs if r is not None and r == rif)
        righe.append({"problem_id": p, "n_con_risposta": sum(1 for r in rs if r is not None),
                      "n_corrette": n_ok, "frazione": n_ok / len(rs),
                      "satura": sum(1 for c in o.outputs if c.finish_reason == "length") / len(rs),
                      "tok_medio": sum(len(c.token_ids) for c in o.outputs) / len(rs)})
    d = pd.DataFrame(righe)
    quota = float((d.frazione >= FRAZ_BASE_OK).mean())
    esito = "ENTRA" if quota >= SOGLIA_INGRESSO else "ESCLUSO"
    print("\n%s: problemi con >=%.0f%% di campioni corretti: %.1f%% (soglia %.0f%%) -> %s"
          % (args.model, 100 * FRAZ_BASE_OK, 100 * quota, 100 * SOGLIA_INGRESSO, esito), flush=True)
    print("  campioni con risposta estratta: %.2f/%d | saturano il tetto: %.1f%% | token medi %.0f"
          % (d.n_con_risposta.mean(), args.k, 100 * d.satura.mean(), d.tok_medio.mean()), flush=True)

    tutti = {}
    if os.path.isfile(args.out):
        tutti = json.load(open(args.out, encoding="utf-8"))
    tutti[args.model] = {"quota_base_ok": quota, "esito": esito, "n": len(d),
                         "risposta_estratta_media": float(d.n_con_risposta.mean() / args.k),
                         "satura": float(d.satura.mean()), "tok_medio": float(d.tok_medio.mean())}
    json.dump(tutti, open(args.out, "w", encoding="utf-8"), indent=1)
    print("scritto %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
