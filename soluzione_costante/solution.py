"""Controllo: la costante «sempre robust». AIMO Interpretability Challenge.

Preparata il 22/10/2026 insieme alla V8 invariata, dopo il cambio di valutazione del 15/10
(una chiamata per problema). Serve a due cose:
  - dare la quota di robust del set valutato (la validazione ora, il test se nominata), cioe'
    il punteggio di riferimento per ogni altra submission;
  - separare un eventuale errore della V8 (firma legacy senza reasoning_effort) da un errore
    dell'harness.

Due fonti ufficiali danno la firma con gli argomenti in ordine diverso: il README di
getting-started `are_robust(model_id, reasoning_effort, problems)`, l'annuncio Discord del
17/9 `are_robust(model_id, problems, reasoning_effort)`. Questa versione (rifatta il 22/10
alle 18, prima di qualunque caricamento) usa i nomi del README e prende come lotto
l'argomento che e' una lista, cosi' risponde giusto con entrambi gli ordini, con gli argomenti
per nome e anche se chiamata come una submission legacy, (model_id, problems).

Nei log finiscono solo conteggi aggregati, mai il testo dei problemi.
"""


def _lotto(reasoning_effort, problems):
    for x in (problems, reasoning_effort):
        if isinstance(x, (list, tuple)):
            return x
    return problems if problems is not None else []


def are_robust(model_id: str, reasoning_effort: str = None, problems: list = None) -> list:
    lotto = _lotto(reasoning_effort, problems)
    print("[costante] model_id=%r problemi=%d -> tutti robust" % (model_id, len(lotto)), flush=True)
    return [True] * len(lotto)
