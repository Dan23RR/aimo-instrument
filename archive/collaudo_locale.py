"""Collaudo locale della sonda contro i programmi VERI di ingestion e scoring.

Non usa i dati privati della gara (non li abbiamo). Costruisce un set sintetico
con la struttura esatta dichiarata dagli organizzatori e ci fa girare sopra
`components/ingestion_program/ingestion.py` e `components/scoring_program/scoring.py`
del repo getting-started, senza modificarli.

Scopo: verificare PRIMA di spedire che
  1. il bundle sia caricabile e la firma accettata;
  2. coverage = 1.0 e invalid_predictions = 0;
  3. l'accuratezza della sonda sia esattamente (numero di spurious)/(totale),
     cioe' che la previsione pre-registrata sia una conseguenza dello scorer
     e non una speranza.

Il caso costruito replica l'ipotesi da falsificare: 17 item, 9 robust e 8 spurious,
distribuiti su due model_id per esercitare anche il raggruppamento per modello.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

QUI = Path(__file__).resolve().parent
GS = Path(os.environ.get("AIMO_GS", "")) if os.environ.get("AIMO_GS") else None
SONDA = Path(os.environ.get("AIMO_BUNDLE") or (QUI / "sonde" / "always-spurious"))

N_ITEM = 17
N_ROBUST = 9          # ipotesi: 0.53 = 9/17
N_SPURIOUS = 8        # ipotesi: 0.47 = 8/17


def costruisci_dati(base: Path) -> tuple[Path, Path]:
    """cases.jsonl (quello che vede la submission) e labels.jsonl (quello che vede lo scorer)."""
    inp, ref = base / "input", base / "reference"
    inp.mkdir(parents=True), ref.mkdir(parents=True)

    etichette = [True] * N_ROBUST + [False] * N_SPURIOUS
    casi, righe_etichette = [], []
    for i, robusto in enumerate(etichette):
        case_id = "case-%03d" % i
        # due model_id, per verificare che il raggruppamento per modello funzioni
        model_id = "qwen3-8b:low" if i % 2 == 0 else "modello-non-in-cache:alto"
        casi.append({"id": case_id, "model_id": model_id,
                     "problem": "Problema sintetico numero %d. Quanto fa %d + %d?" % (i, i, i)})
        righe_etichette.append({"id": case_id, "is_robust": robusto})

    (inp / "cases.jsonl").write_text(
        "".join(json.dumps(c, sort_keys=True) + "\n" for c in casi), encoding="utf-8")
    (ref / "labels.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in righe_etichette), encoding="utf-8")
    return inp, ref


def main():
    if GS is None or not (GS / "components").is_dir():
        raise SystemExit("Serve la variabile AIMO_GS che punta al clone di aimo-interp/getting-started")
    if not (SONDA / "solution.py").is_file():
        raise SystemExit("Manca %s" % (SONDA / "solution.py"))

    print("=" * 74)
    print("COLLAUDO LOCALE DELLA SONDA — programmi ufficiali, dati sintetici")
    print("=" * 74)
    print("  bundle:    %s" % SONDA)
    print("  ingestion: %s" % (GS / "components" / "ingestion_program" / "ingestion.py"))
    print("  scoring:   %s" % (GS / "components" / "scoring_program" / "scoring.py"))
    print("  set finto: %d item, %d robust, %d spurious, 2 model_id" % (N_ITEM, N_ROBUST, N_SPURIOUS))
    print()

    with tempfile.TemporaryDirectory(prefix="aimo-collaudo-") as tmp:
        base = Path(tmp)
        inp, ref = costruisci_dati(base)

        res = base / "arena" / "res"
        arena = base / "arena"
        (arena / "ref").mkdir(parents=True)
        for f in ref.iterdir():
            (arena / "ref" / f.name).write_bytes(f.read_bytes())
        out = base / "out"

        print("-" * 74)
        print("USCITA DELLA SONDA (quello che finira' nei log di Codabench)")
        print("-" * 74)
        r1 = subprocess.run(
            [sys.executable, str(GS / "components" / "ingestion_program" / "ingestion.py"),
             str(inp), str(res), str(SONDA)],
            capture_output=True, text=True)
        print(r1.stdout.strip() or "(nessuna)")
        if r1.returncode != 0:
            print("INGESTION FALLITA:\n" + r1.stderr)
            raise SystemExit(1)

        r2 = subprocess.run(
            [sys.executable, str(GS / "components" / "scoring_program" / "scoring.py"),
             str(arena), str(out)],
            capture_output=True, text=True)
        if r2.returncode != 0:
            print("SCORING FALLITO:\n" + r2.stderr)
            raise SystemExit(1)

        riassunto = json.loads((res / "ingestion_summary.json").read_text(encoding="utf-8"))
        punteggi = json.loads((out / "scores.json").read_text(encoding="utf-8"))

        print()
        print("-" * 74)
        print("ESITO")
        print("-" * 74)
        print("  casi processati:      %d" % riassunto["cases"])
        print("  predizioni invalide:  %d" % riassunto["invalid_predictions"])
        print("  accuracy:             %.4f" % punteggi["accuracy"])
        print("  coverage:             %.4f" % punteggi["coverage"])
        print("  invalid_predictions:  %d" % punteggi["invalid_predictions"])
        print()

        atteso = N_SPURIOUS / N_ITEM
        controlli = [
            ("accuracy == n_spurious/n_totale", abs(punteggi["accuracy"] - atteso) < 1e-9,
             "%.4f atteso %.4f (arrotondato dalla piattaforma: %.2f)" % (punteggi["accuracy"], atteso, atteso)),
            ("coverage == 1.0", punteggi["coverage"] == 1.0, "%.4f" % punteggi["coverage"]),
            ("invalid == 0", punteggi["invalid_predictions"] == 0, str(punteggi["invalid_predictions"])),
            ("casi == %d" % N_ITEM, riassunto["cases"] == N_ITEM, str(riassunto["cases"])),
        ]
        tutto_ok = True
        for nome, ok, dettaglio in controlli:
            print("  [%s] %-34s %s" % ("OK " if ok else "NO ", nome, dettaglio))
            tutto_ok = tutto_ok and ok

        print()
        if tutto_ok:
            print("  Il bundle e' valido e lo scorer si comporta come derivato.")
            print("  La previsione 0.47 NON e' una speranza: e' cio' che questo scorer")
            print("  restituisce per costruzione se il set nascosto ha 8 spurious su 17.")
        else:
            print("  COLLAUDO FALLITO: non spedire.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
