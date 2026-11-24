"""E14 — una esecuzione del pipeline ufficiale su N problemi, dentro l'immagine ufficiale.

Uso (dentro il container, vedi e14_all.sh):
  python e14_run.py --cond C0 --rep 0 --gpu 0 --tag a --n 40

Riproduce ESATTAMENTE il percorso _predici_con_modello di solution.py: uncertainty_profile
degli organizzatori, art.generation_config (greedy, 4096, cache statica, batch 8) e il
regressore ufficiale. In piu' salva l'hash SHA-256 del testo generato per ogni item.
Condizioni (PREREG_E14.md):
  C0 default | C1 batch 1 | C2 flag deterministici (env E14_DETERMINISTIC=1 +
  CUBLAS_WORKSPACE_CONFIG dal docker run) | C3 due processi sulla stessa scheda | C4 scheda 1.
Nessun testo di problema nei print. Output: JSONL, una riga per item.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--cond", required=True)
ap.add_argument("--rep", type=int, required=True)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--tag", default="a")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
ap.add_argument("--out", default="/work/aimo/e14/out")
args = ap.parse_args()

DET = os.environ.get("E14_DETERMINISTIC") == "1"
import torch  # noqa: E402
if DET:
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

sys.path.insert(0, "/work/aimo/soluzione")
from uncertainty_profile.artifact import load_artifact  # noqa: E402
from uncertainty_profile.config import GenerationConfidenceConfig  # noqa: E402
from uncertainty_profile.extraction import (extract_generation_features,  # noqa: E402
                                            load_model_and_tokenizer, release_model)
import pandas as pd  # noqa: E402

ids = [l.strip() for l in open("/work/aimo/gpu/e12_ids.txt", encoding="utf-8") if l.strip()][:args.n]
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered", "augmented-sample-math"]
problemi = {}
for nome in FONTI:
    for r in json.load(open("/work/aimo/dati/%s.json" % nome, encoding="utf-8")):
        pid, t = r.get("problem_id"), r.get("original_problem")
        if pid and isinstance(t, str) and pid not in problemi:
            problemi[pid] = t
testi = [problemi[p] for p in ids]

art = load_artifact(args.model)
assert art is not None, "artefatto non trovato"
cfg_d = art.generation_config.to_dict()
if args.cond == "C1":
    cfg_d["batch_size"] = 1
cfg = GenerationConfidenceConfig.from_dict(cfg_d)
print("E14 cond=%s rep=%d gpu=%d tag=%s n=%d batch=%d det=%s torch=%s cuda=%s dev=%s"
      % (args.cond, args.rep, args.gpu, args.tag, len(ids), cfg.batch_size, DET, torch.__version__,
         torch.version.cuda, torch.cuda.get_device_name(0)), flush=True)

t0 = time.time()
mdl, tok = load_model_and_tokenizer(args.model, local_files_only=True)
t_load = time.time() - t0
t1 = time.time()
righe = extract_generation_features(model=mdl, tokenizer=tok, problem_texts=testi, config=cfg,
                                    include_generated_text=True)
t_gen = time.time() - t1
X = pd.DataFrame.from_records(righe, columns=list(art.feature_names))
score = art.estimator.predict(X)
release_model()

os.makedirs(args.out, exist_ok=True)
nome = os.path.join(args.out, "e14_%s_rep%d_gpu%d_%s.jsonl" % (args.cond, args.rep, args.gpu, args.tag))
with open(nome, "w", encoding="utf-8") as f:
    for pid, r, s in zip(ids, righe, score):
        txt = r.get("generated_text", "") or ""
        rec = {"problem_id": pid, "cond": args.cond, "rep": args.rep, "gpu": args.gpu, "tag": args.tag,
               "score": float(s), "robust_pred": bool(s < art.decision_threshold),
               "sha256_testo": hashlib.sha256(txt.encode("utf-8")).hexdigest(),
               "n_char_testo": len(txt),
               **{k: (float(r[k]) if r.get(k) is not None else None) for k in art.feature_names}}
        f.write(json.dumps(rec) + "\n")
print("scritto %s | load %.0fs | gen %.0fs = %.1f s/item | soglia %.4f | robust_pred %d/%d"
      % (nome, t_load, t_gen, t_gen / max(1, len(ids)), art.decision_threshold,
         int((score < art.decision_threshold).sum()), len(ids)), flush=True)
