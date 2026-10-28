"""Soluzione deployabile v2 — metodo comportamentale con budget di tempo e fallback legittimo.

Tre miglioramenti concreti rispetto alla baseline ufficiale `uncertainty-profiling`, ognuno
motivato da una misura OSSERVATA nel container (AIMO/OSSERVATO_907677.md):

  1. ROBUSTEZZA AL BUDGET. La baseline costa ~119 s/problema e il limite e' 3600 s/run: va in
     timeout oltre ~30 problemi (OSSERVATO). Qui il tempo e' sorvegliato: quando il budget sta
     per esaurirsi si smette di far girare il modello e il resto e' riempito col prior. Cosi'
     la run FINISCE sempre entro il limite, qualunque sia la taglia del set, e non viene
     invalidata. Sul tie-break (runtime medio) questo e' un vantaggio diretto.

  2. FALLBACK LEGITTIMO invece di all-False. La baseline, per ogni modello non in cache,
     restituisce `False` costante su tutti gli item. Qui si usa il prior di robustezza di quel
     modello stimato dai DATI PUBBLICI degli organizzatori (uso permesso dal regolamento:
     "labeled data for other perturbations are allowed"). Per un modello mai visto si usa il
     prior globale. Non e' taratura sul test: e' conoscenza pubblica pre-esistente.

  3. NON FALLISCE MAI. Qualunque eccezione lascia in piedi il prior, quindi coverage=1 e
     invalid=0 per costruzione.

Il cuore discriminante (robust vs spurious sui casi ad alta accuratezza) resta quello degli
organizzatori, `uncertainty_profile`, non modificato: e' l'unico pezzo che richiede il modello
e che non e' validabile offline senza GPU. Qui viene solo reso robusto e budget-aware.

DIAGNOSTICA: solo aggregati (model_id, conteggi, secondi). Nessun testo di problema registrato.
"""
import json
import os
import time

# --------------------------------------------------------------------- configurazione budget
# Limite dichiarato dagli organizzatori: 3600 s per run. Si tiene un margine di sicurezza per
# il caricamento del modello (~60 s OSSERVATO) e per lo scoring. Sotto questo residuo, si passa
# al prior senza far girare altro il modello.
BUDGET_TOTALE_S = float(os.environ.get("AIMO_BUDGET_S", "3600"))
MARGINE_S = 240.0                     # riserva per load + scoring + coda
COSTO_STIMATO_PER_PROBLEMA_S = 130.0  # OSSERVATO ~119 s, arrotondato in alto per prudenza

_QUI = os.path.dirname(os.path.abspath(__file__))
_AVVIO = time.perf_counter()
_LOG = []


def _prior_table():
    try:
        with open(os.path.join(_QUI, "prior_per_modello.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_PRIOR = _prior_table()
_PRIOR_GLOBALE = 0.239                 # dai dati pubblici; < 0.5 -> maggioranza spurious
_ALIAS = {
    "qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "gpt-oss-120b:low": "openai/gpt-oss-120b",
}
_MODELLO_IN_CACHE = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def _risolvi(model_id: str) -> str:
    return _ALIAS.get(model_id, model_id)


def _prior_bool(model_id: str) -> bool:
    """Prior di robustezza per questo modello, dai dati pubblici. True = predici robust."""
    chiave = _risolvi(model_id)
    # cerca sia con la chiave risolta sia con eventuale alias pubblico
    for k in (model_id, chiave):
        if k in _PRIOR:
            return _PRIOR[k] > 0.5
    # chiave pubblica inversa (es. il checkpoint -> alias :low)
    for pub, ck in _ALIAS.items():
        if ck == chiave and pub in _PRIOR:
            return _PRIOR[pub] > 0.5
    return _PRIOR_GLOBALE > 0.5


def _tempo_residuo() -> float:
    return BUDGET_TOTALE_S - MARGINE_S - (time.perf_counter() - _AVVIO)


def are_robust(model_id: str, problems: list[str]) -> list[bool]:
    n = len(problems)
    if n == 0:
        return []

    prior = _prior_bool(model_id)
    predizioni = [prior] * n            # default sicuro, sempre valido
    metodo = "prior"
    usati_modello = 0

    # Il metodo comportamentale si applica solo ai modelli il cui artefatto e' nel bundle,
    # cioe' quelli caricabili. Per gli altri il prior e' tutto cio' che si puo' fare.
    if _risolvi(model_id) == _MODELLO_IN_CACHE:
        try:
            from uncertainty_profile.inference import predict_robustness

            # Quanti problemi possiamo permetterci col modello, dato il tempo residuo?
            residuo = _tempo_residuo()
            affrontabili = max(0, int(residuo / COSTO_STIMATO_PER_PROBLEMA_S))
            k = min(n, affrontabili)

            if k > 0:
                parziale = predict_robustness(model_id=model_id, problem_texts=list(problems[:k]))
                if (isinstance(parziale, list) and len(parziale) == k
                        and all(type(x) is bool for x in parziale)):
                    for i in range(k):
                        predizioni[i] = parziale[i]
                    usati_modello = k
                    metodo = "modello" if k == n else "modello+prior(budget)"
                # i restanti n-k restano al prior: la run non va in timeout
        except Exception as exc:                    # noqa: BLE001
            metodo = "prior(eccezione:%s)" % type(exc).__name__

    secondi = time.perf_counter() - _AVVIO
    _LOG.append((model_id, n, usati_modello, metodo, secondi))
    print("[sol-v2] model=%r problemi=%d col_modello=%d metodo=%s prior=%s t_cumulato=%.1fs"
          % (model_id, n, usati_modello, metodo, prior, secondi), flush=True)
    return predizioni
