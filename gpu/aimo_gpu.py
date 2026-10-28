"""Kit GPU per E6 su Modal — estrazione feature fedele al container di gara.

Tre comandi, in ordine di costo crescente. Non saltare i primi due.

  modal run aimo_gpu.py::verifica          # ~1 min, pochi centesimi: versioni, GPU, dati
  modal run aimo_gpu.py::pilota            # ~20 min: throughput bf16 reale su 6 problemi
  modal run aimo_gpu.py::estrai --extra 1  # il run vero

Fedelta' al container di gara (Dockerfile.competition): bf16, greedy deterministico,
transformers 5.13.0. Le versioni effettive vengono stampate da `verifica` e vanno
dichiarate nel writeup: se differiscono dal container, va detto.

I pesi del modello sono messi in cache su un Volume Modal: si scaricano una volta sola.
"""

import modal

app = modal.App("aimo-e6")

IMMAGINE = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "accelerate", "safetensors", "pandas", "pyarrow",
        "transformers==5.13.0", "tokenizers==0.22.2", "huggingface-hub",
        "scikit-learn==1.8.0", "joblib",
    )
    .env({"HF_HOME": "/cache/hf"})
)

VOL_CACHE = modal.Volume.from_name("aimo-hf-cache", create_if_missing=True)
VOL_OUT = modal.Volume.from_name("aimo-e6-out", create_if_missing=True)
MODELLO = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SEED = 20260829
VOLUMI = {"/cache": VOL_CACHE, "/out": VOL_OUT}


# --------------------------------------------------------------------------- dati
def _coppie_etichettate():
    """Coppie (problema, robust) del modello in cache, dai dataset pubblici degli
    organizzatori. Label = n_detrimental_permutations == 0, regola verificata al 100%
    su 223 righe etichettate (AIMO/etichetta.py)."""
    import json
    import urllib.request

    fonti = ["augmented-sample-math-agg", "augmented-sample-math-agg-filtered",
             "val-sample", "aimo-interp-challenge-sample-full",
             "augmented-sample-math-full-filtered"]
    per = {}
    for nome in fonti:
        off = 0
        while True:
            url = ("https://datasets-server.huggingface.co/rows?dataset=aimo-interp%%2F%s"
                   "&config=default&split=validation&offset=%d&length=100" % (nome, off))
            with urllib.request.urlopen(url, timeout=90) as r:
                b = json.loads(r.read().decode())
            righe = [x["row"] for x in b.get("rows", [])]
            for x in righe:
                # alias pubblico del checkpoint nella cache del container
                if x.get("model_id") != "qwen3-8b:low":
                    continue
                v = per.setdefault(x["problem_id"],
                                   {"problema": x["original_problem"], "detr": 0})
                v["detr"] += int(x.get("n_detrimental_permutations") or 0)
            off += 100
            if off >= b.get("num_rows_total", 0) or not righe:
                break
    return [{"problem_id": k, "problema": v["problema"], "robust": v["detr"] == 0}
            for k, v in sorted(per.items())]


def _perturba_format(t):
    """Famiglia NOSTRA 'format': stesso contenuto, layout diverso."""
    return "Problem statement:\n\n" + "\n".join(p.strip() for p in t.split(". ") if p.strip())


def _perturba_ridondanza(t):
    """Famiglia NOSTRA 'ridondanza': istruzione neutra premessa."""
    return "Read the problem carefully, then solve it.\n\n" + t


# --------------------------------------------------------------------- 1. verifica
@app.function(image=IMMAGINE, gpu="A10G", volumes=VOLUMI, timeout=900)
def verifica():
    """Costo trascurabile. Stampa l'ambiente effettivo e prova la sorgente dati."""
    import torch
    import transformers

    p = torch.cuda.get_device_properties(0)
    info = {
        "gpu": p.name, "vram_MB": p.total_memory // (1 << 20),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "bf16": torch.cuda.is_bf16_supported(),
        "transformers": transformers.__version__,
    }
    print("AMBIENTE MODAL:", info)
    print("CONTAINER GARA: torch 2.12.1+cu132, transformers 5.13.0, bf16")

    dati = _coppie_etichettate()
    rob = sum(d["robust"] for d in dati)
    print("coppie etichettate: %d  (robust %d / spurious %d)" % (len(dati), rob, len(dati) - rob))
    print("esempio problem_id:", [d["problem_id"] for d in dati[:5]])
    info["coppie"] = len(dati)
    info["robust"] = rob
    return info


# ----------------------------------------------------------------------- motore
def _carica(torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELLO, cache_dir="/cache/hf")
    model = AutoModelForCausalLM.from_pretrained(
        MODELLO, dtype=torch.bfloat16, device_map="auto", cache_dir="/cache/hf")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _genera(model, tok, torch, testo, max_new, sample=False, temp=0.6, seed=None):
    """Ritorna (testo_generato, feature, secondi, n_token). Feature solo se non sample."""
    import statistics as st
    import time
    import torch.nn.functional as F

    prompt = tok.apply_chat_template([{"role": "user", "content": testo}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)[:, -2048:]
    if seed is not None:
        torch.manual_seed(seed)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new, do_sample=sample,
            temperature=(temp if sample else None), top_p=(0.95 if sample else None),
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True, output_scores=not sample)
    dt = time.time() - t0
    seq = out.sequences[0][ids.shape[-1]:]
    testo_out = tok.decode(seq, skip_special_tokens=True)

    feat = {}
    if not sample and out.scores:
        lps, ents, top1, marg, is_top1 = [], [], [], [], []
        for i, lg in enumerate(out.scores):
            logp = F.log_softmax(lg[0].float(), dim=-1)
            p = logp.exp()
            lps.append(logp[seq[i]].item())
            ents.append(-(p * logp).sum().item())
            due = torch.topk(p, 2).values
            top1.append(due[0].item())
            marg.append((due[0] - due[1]).item())
            is_top1.append(1.0 if int(lg[0].argmax()) == int(seq[i]) else 0.0)
        k = max(1, int(0.20 * len(lps)))
        feat = {
            # --- le 14 della baseline ufficiale (uncertainty_profile/config.py)
            "mean_logprob": st.mean(lps), "min_logprob": min(lps),
            "max_logprob": max(lps),
            "std_logprob": st.pstdev(lps) if len(lps) > 1 else 0.0,
            "mean_nll": -st.mean(lps),
            "ppl": float(pow(2.718281828, -st.mean(lps))),
            "min_k_logprob": st.mean(sorted(lps)[:k]),
            "mean_entropy": st.mean(ents), "mean_top1": st.mean(top1),
            "mean_margin": st.mean(marg), "frac_top1": st.mean(is_top1),
            "frac_alta": sum(1 for x in top1 if x >= 0.9) / len(top1),
            "frac_bassa": sum(1 for x in top1 if x <= 0.1) / len(top1),
        }
        # --- H11-bis: feature che la baseline NON usa, a costo zero di GPU.
        # Le 14 sopra sono tutte medie per token, invarianti alla lunghezza: cancellano
        # sia quanto ragionamento serve, sia come l'incertezza evolve mentre ragiona.
        meta = len(ents) // 2
        feat.update({
            "n_token": len(seq),
            "satura_cap": 1.0 if len(seq) >= max_new else 0.0,
            "ent_prima": st.mean(ents[:meta]) if meta else 0.0,
            "ent_dopo": st.mean(ents[meta:]) if meta else 0.0,
            "ent_trend": (st.mean(ents[meta:]) - st.mean(ents[:meta])) if meta else 0.0,
            "lp_prima": st.mean(lps[:meta]) if meta else 0.0,
            "lp_dopo": st.mean(lps[meta:]) if meta else 0.0,
            "lp_trend": (st.mean(lps[meta:]) - st.mean(lps[:meta])) if meta else 0.0,
            "top1_trend": (st.mean(top1[meta:]) - st.mean(top1[:meta])) if meta else 0.0,
        })
    return testo_out, feat, dt, len(seq)


def _risposta(testo_out):
    import re
    m = re.findall(r"boxed\{([^{}]*)\}", testo_out)
    if m:
        return m[-1].strip()
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", testo_out[-200:])
    return m[-1].strip() if m else ""


# ----------------------------------------------------------------------- 2. pilota
@app.function(image=IMMAGINE, gpu="A100-40GB", volumes=VOLUMI, timeout=3600)
def pilota(n: int = 6, max_new: int = 4096):
    """Misura il throughput bf16 REALE e verifica che le risposte escano al cap di gara."""
    import time
    import torch

    dati = _coppie_etichettate()
    rob = [d for d in dati if d["robust"]][:n // 2]
    spu = [d for d in dati if not d["robust"]][:n // 2]
    campione = rob + spu

    t0 = time.time()
    model, tok = _carica(torch)
    t_load = time.time() - t0
    VOL_CACHE.commit()
    print("load %.0fs | gpu=%s | bf16=%s"
          % (t_load, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported()))

    righe = []
    for i, d in enumerate(campione):
        _t, feat, dt, ntok = _genera(model, tok, torch, d["problema"], max_new)
        r = _risposta(_t)
        righe.append({"problem_id": d["problem_id"], "robust": d["robust"],
                      "secondi": dt, "n_token": ntok, "toks": ntok / dt,
                      "ha_risposta": bool(r), **feat})
        print("[%d/%d] %s robust=%s %.0fs %.1f tok/s token=%d risposta=%r"
              % (i + 1, len(campione), d["problem_id"], d["robust"], dt,
                 ntok / dt, ntok, r[:14]), flush=True)

    tps = sum(x["toks"] for x in righe) / len(righe)
    sec = sum(x["secondi"] for x in righe) / len(righe)
    con_risp = sum(x["ha_risposta"] for x in righe)
    print("\n=== PILOTA bf16 ===")
    print("throughput: %.1f tok/s | %.0f s per problema a cap %d" % (tps, sec, max_new))
    print("con risposta finale: %d/%d" % (con_risp, len(righe)))
    print("proiezione run completo su %d coppie: %.1f ore di sola passata greedy"
          % (len(dati), len(dati) * sec / 3600))
    return {"tok_s": round(tps, 1), "sec_per_problema": round(sec), "con_risposta": con_risp,
            "n": len(righe), "coppie_totali": len(dati), "load_s": round(t_load)}


# ----------------------------------------------------------------------- 3. estrai
@app.function(image=IMMAGINE, gpu="A100-40GB", volumes=VOLUMI, timeout=11 * 3600)
def estrai(max_new: int = 4096, limite: int = 0, da: int = 0):
    """Run vero: una sola passata greedy per coppia, con le feature della baseline
    PIU' quelle di H11-bis (lunghezza, saturazione, trend), che costano zero GPU.

    Vedi PREREG_E6_DEVIAZIONE.md: gli extra self-consistency/perturbazione sono stati
    ritirati perche' costerebbero 4,5x il budget e perche' confrontano risposte finali
    che meta' delle generazioni non produce (3/6 nel pilota).

    `da` permette di riprendere da un indice se il run viene interrotto.
    """
    import time

    import pandas as pd
    import torch

    dati = _coppie_etichettate()
    if limite:
        dati = dati[:limite]
    dati = dati[da:]
    model, tok = _carica(torch)
    VOL_CACHE.commit()
    print("coppie da processare: %d (da indice %d) | cap=%d" % (len(dati), da, max_new),
          flush=True)

    righe, t_avvio = [], time.time()
    for i, d in enumerate(dati):
        testo_out, feat, dt, ntok = _genera(model, tok, torch, d["problema"], max_new)
        risp = _risposta(testo_out)
        r = {"problem_id": d["problem_id"], "robust": d["robust"],
             "risposta": risp, "ha_risposta": 1.0 if risp else 0.0,
             "secondi": dt, "coda_testo": testo_out[-200:], **feat}
        righe.append(r)
        trascorso = time.time() - t_avvio
        print("[%3d/%3d] %s robust=%s %.0fs tok=%d risp=%r | %.2f h, stima totale %.1f h"
              % (i + 1, len(dati), d["problem_id"], d["robust"], dt, ntok, risp[:12],
                 trascorso / 3600, trascorso / (i + 1) * len(dati) / 3600), flush=True)
        if i % 5 == 4:
            pd.DataFrame(righe).to_parquet("/out/feature_parziale.parquet")
            VOL_OUT.commit()

    df = pd.DataFrame(righe)
    df.to_parquet("/out/feature_greedy.parquet")
    VOL_OUT.commit()
    print("scritto /out/feature_greedy.parquet: %d righe, %d colonne"
          % (len(df), len(df.columns)))
    print("robust: %d / %d | con risposta: %d | saturano il cap: %d"
          % (df.robust.sum(), len(df), df.ha_risposta.sum(), df.satura_cap.sum()))
    return {"righe": len(df), "ore": round((time.time() - t_avvio) / 3600, 2),
            "con_risposta": int(df.ha_risposta.sum()), "satura": int(df.satura_cap.sum())}


@app.function(image=IMMAGINE, volumes=VOLUMI, timeout=900)
def scarica():
    """Restituisce il parquet dei risultati come bytes, per salvarlo in locale."""
    from pathlib import Path
    for nome in ("feature_greedy.parquet", "feature_parziale.parquet"):
        p = Path("/out") / nome
        if p.is_file():
            b = p.read_bytes()
            print("trovato %s: %d byte" % (nome, len(b)))
            return {"nome": nome, "bytes": b}
    return {"nome": None, "bytes": b""}


@app.local_entrypoint()
def main():
    print(verifica.remote())
