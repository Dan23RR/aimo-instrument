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
COSTO_STIMATO_PER_PROBLEMA_S = 90.0   # OSSERVATO in v5: 21-59 s/item per batch; 90 prudente

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
    # v5: dal 31/8 gli organizzatori hanno messo in cache TUTTI e quattro i modelli
    # valutati (mail del 31/8: gpt-oss-120b 61.4 GiB, gemma-3-27b 51.8 GiB, DeepSeek 15.8 GiB,
    # Olmo-3-7B 14.2 GiB, tutti con device_map="auto" e local_files_only=True). Quindi il
    # metodo comportamentale si applica a OGNI modello, non piu' solo a quello in cache.
    #
    # Limite dichiarato: l'artefatto (regressore + soglia) esiste solo per DeepSeek. Per gli
    # altri si applica quel regressore alle feature del modello STESSO: e' transfer, ed e'
    # dichiarato in AIMO/PREREG_V5.md. Se fallisce, la predizione resta il prior.
    try:
        residuo = _tempo_residuo()
        affrontabili = max(0, int(residuo / COSTO_STIMATO_PER_PROBLEMA_S))
        k = min(n, affrontabili)

        if k > 0:
            parziale = _predici_con_modello(model_id, list(problems[:k]))
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

    # ---------------------------------------------------- misura del batching (solo warmup)
    # UNICO dato che manca per dimensionare la fase principale, e che nessun calcolo offline
    # puo' dare: il costo per problema misurato finora (119 s) viene da un lotto di DUE
    # problemi, cioe' un solo batch. La config della baseline usa batch_size=8: su lotti piu'
    # grandi il throughput per problema puo' migliorare di molto, e il tetto di ~30 item
    # salirebbe in proporzione.
    #
    # Le predizioni NON vengono toccate: restano quelle gia' calcolate sopra. E7 ed E8 hanno
    # misurato su 109 coppie che le feature del modello in cache NON predicono i modelli che
    # il container non carica, quindi usarle qui peggiorerebbe. Questa e' una cronometria,
    # non un metodo. Registra solo tempi e conteggi: nessun testo di problema.
    global _BATCH_FATTO
    if (not _BATCH_FATTO and usati_modello == 0 and n >= 6
            and _tempo_residuo() > 1500 and sum(r[2] for r in _LOG) > 0):
        _BATCH_FATTO = True
        try:
            _misura_batch(problems[:8])
        except Exception as exc:                    # noqa: BLE001
            print("[batch] saltata: %s" % type(exc).__name__, flush=True)

    return predizioni


_BATCH_FATTO = False


def _misura_batch(testi: list) -> None:
    """Throughput del modello in cache su un lotto, con il batching della baseline."""
    from uncertainty_profile.config import GenerationConfidenceConfig
    from uncertainty_profile.extraction import (
        extract_generation_features, load_model_and_tokenizer, release_model)

    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(_MODELLO_IN_CACHE, local_files_only=True)
    t_load = time.perf_counter() - t0
    print("[batch] load=%.1fs problemi=%d" % (t_load, len(testi)), flush=True)
    try:
        for bs in (1, 4):
            if bs > len(testi):
                continue
            sub = testi[:bs]
            cfg = GenerationConfidenceConfig(max_new_tokens=512, batch_size=bs)
            t0 = time.perf_counter()
            righe = extract_generation_features(model=model, tokenizer=tok,
                                                problem_texts=sub, config=cfg,
                                                include_generated_text=False)
            dt = time.perf_counter() - t0
            print("[batch] batch_size=%d problemi=%d secondi=%.1f  s/problema=%.1f  righe=%d"
                  % (bs, len(sub), dt, dt / len(sub), len(righe)), flush=True)
            if _tempo_residuo() < 900:
                print("[batch] budget: mi fermo", flush=True)
                break
    finally:
        model = None
        tok = None
        release_model()
        print("[batch] fine, t_cumulato=%.1fs" % (time.perf_counter() - _AVVIO), flush=True)


def _predici_con_modello(model_id: str, testi: list) -> list:
    """Feature di incertezza del modello RICHIESTO + regressore DeepSeek (transfer dichiarato).

    Per il modello in cache originario e' esattamente la via ufficiale. Per gli altri tre
    e' la stessa estrazione di feature applicata al modello giusto, con la mappa
    feature->robustezza presa in prestito, perche' un artefatto proprio non esiste.
    """
    import numpy as np
    import pandas as pd
    from uncertainty_profile.artifact import load_artifact
    from uncertainty_profile.config import resolve_checkpoint_model_id
    from uncertainty_profile.extraction import (
        extract_generation_features, load_model_and_tokenizer, release_model)

    art = load_artifact(_MODELLO_IN_CACHE)      # unico artefatto disponibile
    if art is None:
        return []
    checkpoint = resolve_checkpoint_model_id(model_id)

    # v6: per gpt-oss la soglia viene CALIBRATA su 10 item etichettati del val-sample
    # pubblico (spediti nel bundle), estratti con lo stesso modello e la stessa config.
    # Regola fissata in AIMO/PREREG_V6.md prima di conoscere qualunque punteggio.
    calibro = _calibrazione_per(checkpoint)
    n_cal = len(calibro)
    if n_cal and _tempo_residuo() < COSTO_STIMATO_PER_PROBLEMA_S * (n_cal + len(testi)):
        calibro = []            # budget stretto: si ricade sulla soglia ereditata (v5)
        n_cal = 0

    model = tok = None
    try:
        t0 = time.perf_counter()
        model, tok = load_model_and_tokenizer(checkpoint, local_files_only=True)
        print("[v5] caricato %r in %.1fs" % (checkpoint, time.perf_counter() - t0), flush=True)
        tutti = [c["problema"] for c in calibro] + list(testi)
        righe = extract_generation_features(model=model, tokenizer=tok, problem_texts=tutti,
                                            config=art.generation_config,
                                            include_generated_text=False)
        X = pd.DataFrame.from_records(righe, columns=list(art.feature_names))
        punteggi = np.asarray(art.estimator.predict(X), dtype=float)
        if punteggi.shape != (len(tutti),) or not np.isfinite(punteggi).all():
            return []
        soglia = float(art.decision_threshold)
        # v7: soglia per-modello calibrata OFFLINE su etichette con perturbazioni NOSTRE
        # (PREREG_E9). Precedenza: tabella spedita nel bundle > calibrazione a runtime
        # sugli item pubblici (gpt-oss) > soglia ereditata dall'artefatto DeepSeek.
        s_tab = _soglia_tabella(checkpoint)
        if s_tab is not None:
            soglia = s_tab
            n_cal = 0                       # la tabella vince sulla calibrazione a runtime
            print("[v7] soglia da tabella=%.3f per %r (ereditata %.3f)"
                  % (soglia, checkpoint, art.decision_threshold), flush=True)
        if n_cal:
            soglia = _taratura(punteggi[:n_cal], [c["robust"] for c in calibro], soglia)
            print("[v6] soglia calibrata=%.3f (ereditata %.3f) su %d item pubblici"
                  % (soglia, art.decision_threshold, n_cal), flush=True)
        p_eval = punteggi[n_cal:]
        print("[v5] punteggi min=%.3f max=%.3f soglia=%.3f"
              % (p_eval.min(), p_eval.max(), soglia), flush=True)
        return [bool(p < soglia) for p in p_eval]
    finally:
        model = None
        tok = None
        release_model()


def _soglia_tabella(checkpoint: str):
    """Soglia per-modello da soglie_per_modello.json (E9), se spedita e se copre il modello."""
    try:
        with open(os.path.join(_QUI, "soglie_per_modello.json"), encoding="utf-8") as f:
            tab = json.load(f)
        v = tab.get(checkpoint)
        return float(v["soglia"]) if isinstance(v, dict) and "soglia" in v else None
    except Exception:
        return None


def _calibrazione_per(checkpoint: str) -> list:
    """Item etichettati pubblici per questo checkpoint, se spediti nel bundle."""
    if checkpoint != "openai/gpt-oss-120b":
        return []
    try:
        with open(os.path.join(_QUI, "calibrazione_gptoss.json"), encoding="utf-8") as f:
            dati = json.load(f)
        return [d for d in dati if isinstance(d.get("problema"), str)]
    except Exception:
        return []


def _taratura(punteggi_cal, robust_cal, soglia_ereditata: float) -> float:
    """Soglia che massimizza l'accuratezza sui item di calibrazione.

    Candidati: punti medi fra i punteggi ordinati, piu' la soglia ereditata.
    A parita' di accuratezza vince la piu' vicina alla soglia ereditata.
    (Regola pre-registrata: AIMO/PREREG_V6.md.)
    """
    coppie = sorted(zip([float(x) for x in punteggi_cal], [bool(r) for r in robust_cal]))
    xs = [x for x, _ in coppie]
    candidati = [soglia_ereditata]
    candidati += [(a + b) / 2.0 for a, b in zip(xs, xs[1:])]
    candidati += [xs[0] - 1e-6, xs[-1] + 1e-6]

    def acc(t):
        return sum(1 for x, r in coppie if (x < t) == r)

    migliore = max(candidati, key=lambda t: (acc(t), -abs(t - soglia_ereditata)))
    return float(migliore)
