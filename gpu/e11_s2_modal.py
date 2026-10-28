"""E11 / S2 — estrazione delle feature di distanza fra attivazioni sui 164 con etichetta vera.

Uso:
  modal run e11_s2_modal.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B

Eseguito SOLO perche' H29 (ripetibilita') e' passato: ICC 0.997-0.999 su tutte le candidate.

Protocollo fissato in PIANO_E11_definitivo.md PRIMA di guardare le etichette:
  - profondita' relativa 0.80 (nessuna sweep sui layer: con n=164 una sweep su 36 layer
    fabbrica +9 punti di puro rumore, misurato).
  - famiglie: framing e redundancy, le uniche sempre applicabili (no-op su 0/164).
  - CONTROLLI DI PADDING appaiati in token per item: pad_short aggiunge lo stesso numero di
    token che aggiunge framing, pad_long lo stesso di redundancy, ma con testo NEUTRO che non
    riformula il problema. Servono a separare "la rappresentazione si sposta perche' il
    prompt e' piu' lungo" da "si sposta perche' il modello e' fragile su questo problema".
  - Z_C1 = mean(d_framing, d_redundancy) - mean(d_pad_short, d_pad_long)  [residualizzata]
  - Z_C1b = cos(differenza_framing, differenza_redundancy)                [adimensionale]

Questo script ESTRAE soltanto. Non guarda le etichette e non calcola AUROC: l'analisi e'
offline, in e11_s2_analisi.py, con CV raggruppata per problema.
Nessun testo di problema nei log.
"""
import modal

app = modal.App("aimo-e11-s2")
IMG = (modal.Image.from_registry("pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime")
       .pip_install("accelerate==1.13.0", "huggingface-hub==1.22.0", "pandas==3.0.3",
                    "safetensors==0.8.0", "tokenizers==0.22.2", "transformers==5.13.0", "pyarrow",
                    extra_options="--break-system-packages")
       .add_local_file(__file__.replace("e11_s2_modal.py", "perturbazioni.py"), "/root/perturbazioni.py")
       .add_local_dir(__file__.replace("gpu" + ("\\" if "\\" in __file__ else "/") + "e11_s2_modal.py", "dati"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
PROFONDITA = 0.80
# testo neutro per il padding: non riformula, non istruisce, non riguarda il problema
NEUTRO = ("Note for the archive: this record was transcribed in the reading room and the "
          "cataloguer verified the page numbering before filing it on the shelf. ")


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


@app.function(image=IMG, gpu="A100-40GB", volumes={"/vol": VOL}, timeout=2 * 3600)
def estrai(model: str):
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import framing, redundancy
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problemi = _problemi()
    ids = sorted(problemi)
    print("modello=%s problemi=%d profondita=%.2f" % (model, len(ids), PROFONDITA), flush=True)

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
    mdl.eval()
    n_layer = mdl.config.num_hidden_layers
    layer = max(1, min(n_layer, int(round(PROFONDITA * n_layer))))
    print("layer %d di %d" % (layer, n_layer), flush=True)

    def prompt(t):
        return tok.apply_chat_template([{"role": "user", "content": t}],
                                       tokenize=False, add_generation_prompt=True)

    def n_tok(t):
        return len(tok(prompt(t)).input_ids)

    def pad_a(testo, delta_token):
        """Aggiunge testo NEUTRO fino ad avvicinarsi a delta_token token in piu'."""
        if delta_token <= 0:
            return testo
        unita = max(1, len(tok(NEUTRO).input_ids))
        rip = max(1, int(round(delta_token / unita)))
        return testo + "\n\n" + (NEUTRO * rip).strip()

    @torch.no_grad()
    def attiv(testi, batch=8):
        out = []
        for i in range(0, len(testi), batch):
            enc = tok([prompt(t) for t in testi[i:i + batch]], return_tensors="pt",
                      padding=True, truncation=True, max_length=2048)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            h = mdl(**enc, output_hidden_states=True).hidden_states[layer]
            out.append(h[:, -1, :].float().cpu().numpy())
        return np.concatenate(out, 0)

    def cos(a, b):
        return (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-9)

    t0 = time.time()
    base = [problemi[p] for p in ids]
    n_base = np.array([n_tok(t) for t in base])
    v_fram = [framing(t) for t in base]
    v_redu = [redundancy(t) for t in base]
    d_fram_tok = np.array([n_tok(t) for t in v_fram]) - n_base
    d_redu_tok = np.array([n_tok(t) for t in v_redu]) - n_base
    v_pads = [pad_a(t, int(d)) for t, d in zip(base, d_fram_tok)]
    v_padl = [pad_a(t, int(d)) for t, d in zip(base, d_redu_tok)]
    print("delta token mediani: framing %+d, redundancy %+d" % (int(np.median(d_fram_tok)), int(np.median(d_redu_tok))), flush=True)

    H = {}
    for nome, testi in (("orig", base), ("framing", v_fram), ("redundancy", v_redu),
                        ("pad_short", v_pads), ("pad_long", v_padl)):
        t1 = time.time()
        H[nome] = attiv(testi)
        print("  %-11s %d attivazioni in %.0fs" % (nome, len(testi), time.time() - t1), flush=True)

    d = {k: 1.0 - cos(H["orig"], H[k]) for k in ("framing", "redundancy", "pad_short", "pad_long")}
    dif_f = H["framing"] - H["orig"]
    dif_r = H["redundancy"] - H["orig"]
    df = pd.DataFrame({
        "problem_id": ids,
        "d_framing": d["framing"], "d_redundancy": d["redundancy"],
        "d_pad_short": d["pad_short"], "d_pad_long": d["pad_long"],
        "Z_C1": (d["framing"] + d["redundancy"]) / 2.0 - (d["pad_short"] + d["pad_long"]) / 2.0,
        "Z_C1_grezza": (d["framing"] + d["redundancy"]) / 2.0,
        "Z_C1b": cos(dif_f, dif_r),
        "norm_orig": np.linalg.norm(H["orig"], axis=-1),
        "n_tok_orig": n_base, "d_tok_framing": d_fram_tok, "d_tok_redundancy": d_redu_tok,
    })
    nome = "/vol/e11_s2_%s.parquet" % model.replace("/", "__")
    df.to_parquet(nome)
    VOL.commit()
    print("scritto %s: %d righe in %.0fs totali" % (nome, len(df), time.time() - t0), flush=True)
    print("Z_C1 residualizzata: media %.5f, sd %.5f" % (df.Z_C1.mean(), df.Z_C1.std()), flush=True)
    return {"file": nome, "n": len(df), "layer": layer, "secondi": round(time.time() - t0)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"):
    print(estrai.remote(model=model))
