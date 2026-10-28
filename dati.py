"""Scarica e mette in cache i dataset pubblici della gara AIMO.

Sei dataset pubblici su Hugging Face sotto aimo-interp. Due portano la label
`model_is_robust`, gli altri no ma portano le colonne da cui la label si deriva.
Il pezzo che conta e' `aimo-interp-challenge-sample-full`: 558 righe su OTTO modelli,
stesso dataset_id (`aimo-2025-reference`) del campione di validazione ufficiale.

Nessuna autenticazione: si usa l'API pubblica datasets-server.
"""
import json
import os
import ssl
import urllib.request
from pathlib import Path

QUI = Path(__file__).resolve().parent
CACHE = QUI / "dati"
CACHE.mkdir(exist_ok=True)

DATASET = [
    "aimo-interp-challenge-sample-full",   # 558 righe, 8 modelli, SENZA label
    "val-sample",                          # 28 righe, 7 modelli, CON label
    "augmented-sample-math-agg",           # 141 righe, 1 modello, CON label
    "augmented-sample-math-full-filtered",  # 259 righe, 1 modello, senza label
    "augmented-sample-math",               # 156 righe, 1 modello, CON label
    "augmented-sample-math-agg-filtered",  # non interrogato prima
]

BASE = "https://datasets-server.huggingface.co/rows"
CTX = ssl.create_default_context()


def scarica(nome: str) -> list[dict]:
    """Tutte le righe di un dataset, paginate a 100."""
    fuori = CACHE / ("%s.json" % nome)
    if fuori.exists():
        return json.loads(fuori.read_text(encoding="utf-8"))

    righe, offset = [], 0
    while True:
        url = ("%s?dataset=aimo-interp%%2F%s&config=default&split=validation"
               "&offset=%d&length=100" % (BASE, nome, offset))
        try:
            with urllib.request.urlopen(url, timeout=60, context=CTX) as r:
                blocco = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            print("  %-40s ERRORE offset %d: %s" % (nome, offset, str(exc)[:70]))
            break
        if "rows" not in blocco:
            print("  %-40s risposta senza righe: %s" % (nome, str(blocco)[:100]))
            break
        nuove = [x["row"] for x in blocco["rows"]]
        righe.extend(nuove)
        totale = blocco.get("num_rows_total", len(righe))
        if len(righe) >= totale or not nuove:
            break
        offset += 100

    if righe:
        fuori.write_text(json.dumps(righe, ensure_ascii=False), encoding="utf-8")
    return righe


def main():
    print("Scarico i dataset pubblici aimo-interp\n")
    tutto = {}
    for nome in DATASET:
        righe = scarica(nome)
        tutto[nome] = righe
        if not righe:
            print("  %-40s VUOTO" % nome)
            continue
        modelli = sorted({r.get("model_id") for r in righe})
        con_label = sum(1 for r in righe if r.get("model_is_robust") is not None)
        print("  %-40s %4d righe  %d modelli  label su %d"
              % (nome, len(righe), len(modelli), con_label))

    print("\nColonne del set principale:")
    p = tutto.get("aimo-interp-challenge-sample-full") or []
    if p:
        for k in p[0]:
            print("   ", k)
    print("\nCache in %s" % CACHE)


if __name__ == "__main__":
    main()
