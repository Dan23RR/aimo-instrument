"""Diagnosi: perche' Qwen3.5-4B non produce risposte estraibili nell'etichettatura di E13.

Confronta, sugli stessi problemi, il prompt NUDO (problema e basta, come in e13_etichetta.py)
e il prompt ESPLICITO (con richiesta di \\boxed{}, come nel cancello di E13). Stampa la coda del
testo generato, cosi' si vede COSA produce il modello, non solo che l'estrazione fallisce.
"""
import argparse
import json
import os
import sys

ISTR = "Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}.\n\n"
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tok", type=int, default=16384)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/work/hf")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    sys.path.insert(0, "/work/aimo/gpu")
    from perturbazioni import risposta
    from vllm import LLM, SamplingParams

    problemi = {}
    for nome in FONTI:
        for r in json.load(open("/work/aimo/dati/%s.json" % nome, encoding="utf-8")):
            pid, t = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(t, str) and pid not in problemi:
                problemi[pid] = t
    ids = [l.strip() for l in open("/work/aimo/gpu/e12_ids.txt", encoding="utf-8") if l.strip()][:args.n]

    llm = LLM(model=args.model, max_model_len=args.max_tok + 2560, gpu_memory_utilization=0.90,
              trust_remote_code=True, seed=1)
    tmpl = (llm.get_tokenizer().chat_template or "").lower()
    print("\n=== modello %s | chat template menziona 'think': %s ===" % (args.model, "think" in tmpl))
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tok, n=1, seed=1)

    for stile in ("nudo", "esplicito"):
        convs = [[{"role": "user", "content": (problemi[p] if stile == "nudo" else ISTR + problemi[p])}] for p in ids]
        outs = llm.chat(convs, sp, use_tqdm=False)
        print("\n" + "=" * 70)
        print("STILE: %s" % stile)
        for p, o in zip(ids, outs):
            c = o.outputs[0]
            rc = getattr(c, "reasoning_content", None)
            print("  %s | token %5d | stop=%s | risposta: %s | reasoning_content: %s"
                  % (p, len(c.token_ids), c.finish_reason, risposta(c.text),
                     ("%d char" % len(rc)) if rc else "assente"))
            print("     coda: %s" % repr(c.text[-200:]))


if __name__ == "__main__":
    main()
