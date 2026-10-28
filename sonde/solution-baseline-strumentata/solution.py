"""Riproduzione strumentata della baseline ufficiale `uncertainty-profiling`.

SCOPO. Non e' un tentativo di vincere: e' l'unico esperimento che una submission puo'
risolvere e nessun calcolo offline puo'. Tre domande, tutte decisive per la fase principale:

  1. il percorso GPU funziona davvero nel container? (il modello in cache si carica?)
  2. quanto ci mette? Il tie-break della gara e' il runtime medio su tre prove, e il
     limite e' 3600 s per run. Con 2 item su 17 non lo sappiamo; con 240 item conta tutto.
  3. quanto vale la baseline di riferimento su QUESTO set?

METODO. Il codice di predizione e' quello degli organizzatori, `uncertainty_profile`,
non modificato di una riga. L'unica aggiunta e' la cronometria e un log aggregato.

CIO' CHE LA BASELINE FA, e che va detto perche' decide il punteggio del campo intero.
Dal README ufficiale: "If no supported alias or matching artifact exists, the solution
returns False for every problem so the batch remains valid." Sui 4 modelli del warmup
l'artefatto esiste per uno solo, quindi 15 item su 17 ricevono un `False` costante.
NON lo correggiamo: correggerlo verso `True` significherebbe tarare sulla classe di
maggioranza osservata nel set di valutazione, cioe' fare overfitting sul test, che e'
esattamente cio' che il regolamento dice di voler prevenire. Il prior dei dati pubblici
degli organizzatori dice comunque `spurious` (dal 24% al 39% di robust a seconda del pool).

DIAGNOSTICA. Solo aggregati: model_id, numero di problemi, secondi, quante predizioni
positive. Nessun testo di problema viene registrato, conservato o trasmesso.
"""

import time

_LOG: list[tuple[str, int, float, int, str]] = []


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    avvio = time.perf_counter()
    esito = "ok"
    try:
        from uncertainty_profile.inference import predict_robustness

        predizioni = predict_robustness(model_id=model_id, problem_texts=list(problems))
        if not (isinstance(predizioni, list) and len(predizioni) == len(problems)
                and all(type(p) is bool for p in predizioni)):
            esito = "forma-non-valida"
            predizioni = [False for _ in problems]
    except Exception as exc:                      # noqa: BLE001
        # Mai far fallire il lotto: l'ingestion marcherebbe invalid tutte le predizioni.
        esito = "eccezione:%s" % type(exc).__name__
        predizioni = [False for _ in problems]

    secondi = time.perf_counter() - avvio
    _LOG.append((model_id, len(problems), secondi, sum(predizioni), esito))

    print("[baseline] model_id=%r problemi=%d secondi=%.1f positive=%d esito=%s"
          % (model_id, len(problems), secondi, sum(predizioni), esito), flush=True)

    tot_s = sum(r[2] for r in _LOG)
    tot_p = sum(r[1] for r in _LOG)
    print("[baseline] cumulato: %d chiamate, %d problemi, %.1f s totali, %.2f s per problema"
          % (len(_LOG), tot_p, tot_s, tot_s / max(1, tot_p)), flush=True)

    return predizioni
