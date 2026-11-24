"""V8 — la submission deterministica. AIMO Interpretability Challenge.

Regola congelata in PREREG_V8.md prima di qualunque invio:

    robust  <=>  len(problema) >= 98 caratteri

98 e' la mediana di n_char sui 174 problemi pubblici degli organizzatori. La direzione (i
problemi lunghi sono i robusti) e' misurata sulle 335 coppie etichettate dei dataset pubblici,
dove la regola ha accuratezza bilanciata 0,658.

Perche' una regola che non tocca il modello. Su questa stessa infrastruttura abbiamo misurato
che lo score del regressore ufficiale cambia SEGNO fra esecuzioni identiche (AUROC 0,513 e
0,398 in due ripetizioni dello stesso codice sullo stesso hardware). Non esiste un verso
stabile da calibrare. Questa regola invece produce lo stesso output byte per byte a ogni
esecuzione, su qualunque GPU, in qualunque stato numerico, e costa zero secondi di inferenza.

Proprieta' garantite per costruzione:
  - la predizione dell'item i dipende SOLO dall'item i (nessun quantile sul lotto, nessuna
    dipendenza da model_id, dall'ordine o dalla dimensione del batch);
  - copertura 1.0 e zero predizioni non valide: la funzione non puo' fallire ne' andare in
    timeout;
  - perfettamente riproducibile.

Nei log finiscono solo conteggi aggregati e il model_id che l'harness ci passa. Mai il testo
dei problemi.
"""

SOGLIA_CARATTERI = 98


def _testo(problema) -> str:
    """L'harness passa stringhe; accetto anche il dizionario delle versioni precedenti."""
    if isinstance(problema, str):
        return problema
    if isinstance(problema, dict):
        for chiave in ("original_problem", "problem", "text", "question"):
            valore = problema.get(chiave)
            if isinstance(valore, str):
                return valore
    return str(problema)


def are_robust(model_id: str, problems: list) -> list:
    lunghezze = [len(_testo(p)) for p in problems]
    predizioni = [n >= SOGLIA_CARATTERI for n in lunghezze]
    if lunghezze:
        ordinate = sorted(lunghezze)
        mediana = ordinate[len(ordinate) // 2]
        print("[v8] model_id=%r problemi=%d soglia=%d robust=%d (%.0f%%) "
              "n_char: min=%d mediana=%d max=%d"
              % (model_id, len(problems), SOGLIA_CARATTERI, sum(predizioni),
                 100.0 * sum(predizioni) / len(predizioni), ordinate[0], mediana, ordinate[-1]),
              flush=True)
    else:
        print("[v8] model_id=%r lotto vuoto" % (model_id,), flush=True)
    return predizioni
