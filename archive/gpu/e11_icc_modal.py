"""E11 / H29 — cancello di ripetibilita' delle feature di distanza fra attivazioni.

Uso:
  modal run e11_icc_modal.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --n 40

Misura QUANTO SONO RIPETIBILI le feature di distanza fra attivazioni, PRIMA di guardare
qualunque etichetta. Due passate sugli stessi item, variando esattamente cio' che varia in
produzione:
  passata A: batch 1, nessun padding
  passata B: batch 8, left padding, vicini diversi (ordine invertito)

Se una feature non e' ripetibile, non puo' predire nulla, e va scartata prima di misurarne
il potere predittivo. Criterio di morte fissato prima: ICC(2,1) < 0.78.

Solo forward sul prompt (niente generazione): le distanze di rappresentazione non richiedono
di generare. Costo ~1 euro.

Feature candidate, FISSATE PRIMA (nessuna sweep sui layer: profondita' relativa 0.80, che il
piano ha scelto prima di vedere i dati):
  d_framing, d_redundancy   distanza coseno h(originale) vs h(perturbato)
  Z_C1                      media delle due
  Z_C1b                     coseno fra i due vettori di differenza
  norm_orig                 norma dell'attivazione originale (controllo: deve essere ripetibile)
Nessun testo di problema nei log.
"""
import modal

app = modal.App("aimo-e11-icc")
IMG = (modal.Image.from_registry("pytorch/pytorch:2.12.1-cuda13.2-cudnn9-runtime")
       .pip_install(  # PEP 668: l'immagine pytorch ha python externally managed
           "accelerate==1.13.0", "huggingface-hub==1.22.0", "pandas==3.0.3",
                    "safetensors==0.8.0", "tokenizers==0.22.2", "transformers==5.13.0", "pyarrow", extra_options="--break-system-packages")
       .add_local_file(__file__.replace("e11_icc_modal.py", "perturbazioni.py"), "/root/perturbazioni.py")
       .add_local_dir(__file__.replace("gpu" + ("\\" if "\\" in __file__ else "/") + "e11_icc_modal.py", "dati"),
                      "/root/dati"))
VOL = modal.Volume.from_name("aimo-e9", create_if_missing=True)
FONTI = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
         "augmented-sample-math-agg-filtered", "augmented-sample-math-full-filtered",
         "augmented-sample-math"]
PROFONDITA = 0.80          # fissata prima, nessuna sweep
FAMIGLIE = ["framing", "redundancy"]   # le uniche sempre applicabili (no-op su 0/164)


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
def icc(model: str, n: int = 40):
    import sys
    import time
    sys.path.insert(0, "/root")
    from perturbazioni import perturba
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    problemi = _problemi()
    ids = sorted(problemi)[:n]
    print("modello=%s item=%d profondita=%.2f famiglie=%s" % (model, len(ids), PROFONDITA, FAMIGLIE), flush=True)

    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="auto")
    mdl.eval()
    n_layer = mdl.config.num_hidden_layers
    layer = max(1, min(n_layer, int(round(PROFONDITA * n_layer))))
    print("layer scelto: %d di %d" % (layer, n_layer), flush=True)

    def prompt(t):
        return tok.apply_chat_template([{"role": "user", "content": t}],
                                       tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def attiv(testi, batch):
        """Attivazione dell'ULTIMO token del prompt, al layer scelto. Left padding -> -1 e' valido."""
        out = []
        for i in range(0, len(testi), batch):
            sub = [prompt(t) for t in testi[i:i + batch]]
            enc = tok(sub, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            enc = {k: v.to(mdl.device) for k, v in enc.items()}
            h = mdl(**enc, output_hidden_states=True).hidden_states[layer]
            out.append(h[:, -1, :].float().cpu().numpy())
        return np.concatenate(out, 0)

    def cos(a, b):
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        return (a * b).sum(-1) / np.maximum(na * nb, 1e-9)

    def passata(ordine, batch, etichetta):
        t0 = time.time()
        testi_o = [problemi[p] for p in ordine]
        H = {"original": attiv(testi_o, batch)}
        for f in FAMIGLIE:
            H[f] = attiv([perturba(problemi[p])[f] for p in ordine], batch)
        d = {f: 1.0 - cos(H["original"], H[f]) for f in FAMIGLIE}
        dif = {f: H[f] - H["original"] for f in FAMIGLIE}
        r = pd.DataFrame({"problem_id": ordine,
                          "d_framing": d["framing"], "d_redundancy": d["redundancy"],
                          "Z_C1": (d["framing"] + d["redundancy"]) / 2.0,
                          "Z_C1b": cos(dif["framing"], dif["redundancy"]),
                          "norm_orig": np.linalg.norm(H["original"], axis=-1)})
        print("passata %s (batch=%d): %.0fs" % (etichetta, batch, time.time() - t0), flush=True)
        return r.set_index("problem_id")

    A = passata(ids, 1, "A")                      # batch 1, nessun padding
    B = passata(list(reversed(ids)), 8, "B")      # batch 8, left padding, vicini diversi
    B = B.loc[ids]

    FEAT = ["d_framing", "d_redundancy", "Z_C1", "Z_C1b", "norm_orig"]

    def icc21(x, y):
        """ICC(2,1), two-way random, single measure, accordo assoluto."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        k, nn = 2, len(x)
        M = np.column_stack([x, y])
        gm = M.mean()
        ms_r = k * ((M.mean(1) - gm) ** 2).sum() / (nn - 1)
        ms_c = nn * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
        ms_e = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum() / ((nn - 1) * (k - 1))
        den = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / nn
        return (ms_r - ms_e) / den if den > 0 else float("nan")

    print()
    print("=" * 70)
    print("H29 — RIPETIBILITA'. Criterio di morte fissato prima: ICC(2,1) < 0.78")
    print("=" * 70)
    righe = []
    for f in FEAT:
        a, b = A[f].to_numpy(), B[f].to_numpy()
        i = icc21(a, b)
        rel = float(np.mean(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-9)) * 2)
        esito = "PASSA" if i >= 0.78 else "SCARTATA"
        righe.append({"feature": f, "icc": float(i), "media_A": float(a.mean()),
                      "media_B": float(b.mean()), "err_rel": rel, "esito": esito})
        print("  %-14s ICC=%.4f  media A=%.5f B=%.5f  errore relativo medio=%.1f%%  -> %s"
              % (f, i, a.mean(), b.mean(), 100 * rel, esito))
    res = pd.DataFrame(righe)
    A.join(B, rsuffix="_B").to_parquet("/vol/e11_icc_grezzo.parquet")
    res.to_parquet("/vol/e11_icc.parquet")
    VOL.commit()
    vive = res[res.esito == "PASSA"].feature.tolist()
    vive = [v for v in vive if v != "norm_orig"]
    print()
    print("VERDETTO: %s" % ("candidate sopravvissute: " + ", ".join(vive) if vive
                            else "NESSUNA feature di distanza supera 0.78 -> S2 NON si esegue, E11 chiuso"))
    return {"risultati": righe, "vive": vive, "layer": layer, "n": len(ids)}


@app.local_entrypoint()
def main(model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", n: int = 40):
    print(icc.remote(model=model, n=n))
