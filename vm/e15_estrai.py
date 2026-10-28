"""E15 — estrazione degli stati nascosti dell'ULTIMO token di input, a TUTTI i layer.

Uso (nel venv, HF_HOME=/work/hf):
  python e15_estrai.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --gpu 0 [--system-prompt "..."] [--n 174]

Replica esattamente la rappresentazione usata dalle baseline ufficiali (baselines repo e
solutions/trained-probe): un solo forward sul prompt formattato con il chat template, e per
ogni layer il vettore hidden_states[layer][0, -1, :] in float32. Nessuna generazione.
Salva un .npz per modello: X[n_problemi, n_layer+1, hidden], ids, n_token.
Nessuna etichetta entra qui. Nessun testo di problema nei print.
"""
import argparse
import json
import os
import sys
import time

FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--n", type=int, default=174)
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", default="/work/aimo/e15")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", "/work/hf")

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problemi = {}
    for nome in FONTI:
        for r in json.load(open("/work/aimo/dati/%s.json" % nome, encoding="utf-8")):
            pid, t = r.get("problem_id"), r.get("original_problem")
            if pid and isinstance(t, str) and pid not in problemi:
                problemi[pid] = t
    ids = sorted(problemi)[:args.n]
    print("E15 estrazione modello=%s problemi=%d system_prompt=%s" % (args.model, len(ids), "si" if args.system_prompt else "no"), flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda", local_files_only=True)
    mdl.eval()
    print("caricato in %.0fs | layer=%d | hidden=%d" % (time.time() - t0, mdl.config.num_hidden_layers, mdl.config.hidden_size), flush=True)

    X = []
    ntok = []
    t1 = time.time()
    with torch.no_grad():
        for i, pid in enumerate(ids):
            msgs = ([{"role": "system", "content": args.system_prompt}] if args.system_prompt else []) + [{"role": "user", "content": problemi[pid]}]
            try:
                testo = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                testo = (args.system_prompt + "\n\n" if args.system_prompt else "") + problemi[pid]
            enc = tok(testo, return_tensors="pt", truncation=True, max_length=args.max_len).to("cuda")
            out = mdl(**enc, output_hidden_states=True)
            hs = torch.stack([h[0, -1, :] for h in out.hidden_states], 0).float().cpu().numpy()  # [L+1, hidden]
            X.append(hs)
            ntok.append(int(enc["input_ids"].shape[1]))
            if (i + 1) % 25 == 0:
                print("  %d/%d in %.0fs" % (i + 1, len(ids), time.time() - t1), flush=True)
    X = np.stack(X, 0)
    os.makedirs(args.out, exist_ok=True)
    nome = os.path.join(args.out, "hs_%s%s.npz" % (args.model.replace("/", "__"), "_sys" if args.system_prompt else ""))
    np.savez_compressed(nome, X=X, ids=np.array(ids), n_token=np.array(ntok))
    print("scritto %s: X%s | token medi %.0f | %.0fs" % (nome, X.shape, np.mean(ntok), time.time() - t1), flush=True)


if __name__ == "__main__":
    main()
