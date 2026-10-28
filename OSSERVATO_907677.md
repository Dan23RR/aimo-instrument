# OSSERVATO — la baseline ufficiale misurata dentro il container

Ricevuta della submission **907677**, utente `dan23rr`, Codabench 16180, fase Warmup,
task "AIMO Robustness Validation v8". Caricata il **29 agosto 2026 alle 14:38:48 UTC**,
stato **Finished**, host `llmc-gpu-2`.

Bundle: `AIMO/sonde/solution-baseline-strumentata.zip` (991 KB). Il codice di predizione è
quello degli organizzatori (`uncertainty_profile`, dal repo getting-started), **non
modificato di una riga**. L'unica aggiunta è la cronometria.

## Punteggio, a precisione piena dall'API

```
accuracy            0.5294117647   =  9/17   esatto
coverage            1.0000000000
invalid_predictions 0.0000000000
on_leaderboard      false
```

Il 9/17 esatto **riconferma n = 17** a precisione piena, indipendentemente dal reticolo dei
punteggi arrotondati da cui era stato inferito.

## Il log, integrale

```
[baseline] model_id='allenai/Olmo-3-7B-Instruct'            problemi=4  secondi=2.9    positive=0  ok
[baseline] model_id='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' problemi=2  secondi=238.1  positive=1  ok
[baseline] model_id='google/gemma-3-27b-it'                 problemi=2  secondi=0.0    positive=0  ok
[baseline] model_id='openai/gpt-oss-120b'                   problemi=9  secondi=0.0    positive=0  ok
                                             TOTALE        17 problemi  241.0 s
```

## I quattro fatti

**1. Il percorso GPU funziona.** I pesi del modello in cache si caricano (399 shard, meno di
un secondo) e la generazione gira. Il container fa quello che promette.

**2. Il costo è 119 secondi per problema.** 238,1 secondi per 2 problemi sul modello in cache.
Il limite dichiarato è **3600 secondi per run**. Quindi:

> **La baseline ufficiale può processare al massimo 30 problemi prima di andare in timeout.**

Se la fase principale valuta più di ~30 item su modelli caricabili, la baseline degli
organizzatori **non finisce**. Non è una questione di ottimizzazione: è il metodo di
riferimento che non entra nel proprio budget.

**3. Il default costante è confermato dal vivo.** Per i tre modelli non in cache la funzione
ritorna in **2,9 s, 0,0 s, 0,0 s** con **zero predizioni positive**: quindici item su
diciassette ricevono un `False` costante senza che nulla venga misurato. È il comportamento
documentato nel README ufficiale, e spiega la forma della classifica.

**4. E il punteggio che ne esce.** La baseline predice 1 positivo sui 2 item del modello in
cache, e ottiene **9/17**. Il predittore costante `always-true` ottiene **9/17**. Sono lo
stesso numero, e quattro partecipanti in classifica (**incluso l'organizzatore**) stanno
esattamente lì.

> Il metodo di riferimento della gara, che fa girare un modello di ragionamento da 8 miliardi
> di parametri per quattro minuti su GPU, ottiene lo stesso punteggio di una costante che non
> calcola nulla. E il tie-break della gara è il runtime, quindi **perde contro la costante**.

## Cosa se ne deduce sulla composizione del set, senza toccare nulla

**CORRETTO il 29/8 sera, dopo verifica avversaria (`verifica_tutto.py`).** La prima stesura
di questo file affermava che i 2 item del modello in cache fossero "1 robust e 1 spurious".
**Era una sovradichiarazione**: enumerando tutte le composizioni compatibili con i due
punteggi osservati, ne restano **due**, non una:

| cache (2 item) | blind (15 item) | baseline |
|---|---|---|
| 1 robust + 1 spurious | 8 robust + 7 spurious | 9/17 |
| 2 robust + 0 spurious | 7 robust + 8 spurious | 9/17 |

I due punteggi non le separano. Distinguerle richiederebbe un'altra submission mirata, cioè
risolvere etichette per differenza, che è la linea chiusa dal regolamento e dalla nostra
pre-registrazione. **Resta indeterminato, e va bene così.** In entrambi i casi la baseline
coincide col predittore costante `always-true` (9/17) e nulla dell'analisi cambia.

Il resto vale invariato: **non si tara alcuna costante su questi numeri**, sarebbe
overfitting sul set di valutazione e non trasferirebbe alla fase principale.

## Righe non passate

Nessun testo di problema registrato, conservato o trasmesso. Nessuna etichetta toccata.
Le due submission **non sono state pubblicate**: `on_leaderboard: false`, e i tre controlli
disponibili sulla riga ("Add to Leaderboard", "Make Public", "Delete Submission") sono stati
verificati e **non premuti**. Nessun atto pubblico è avvenuto a nome di Daniel.
