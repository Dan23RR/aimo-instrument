# Esito E6 — 30 agosto 2026, ore 23:20

Run Modal `ap-pYpRC7OsVoUDKQBJiIOdYH`, terminato. **164 coppie su 164**, zero NaN, zero
errori, 164 problemi distinti. **9,2 ore di A100-40GB bf16, circa 19,3 $.**

Protocollo eseguito senza modifiche: fold e seed (20260829) fissati il mattino del 30 agosto,
criterio di morte appaiato scritto prima, script provato a vuoto sui parziali per verificare
che non si rompesse, non per scegliere cosa guardare.

## H10 — la baseline si riproduce

| | |
|---|---|
| accuratezza in CV raggruppata per problema | **0,7744** |
| classe di maggioranza | 0,7561 |
| dichiarato dagli organizzatori | 0,692 ± 0,101 |
| compatibile | **sì** |

La pipeline su Modal produce feature coerenti con quelle degli organizzatori, nonostante GPU
diversa (A100 contro RTX PRO 6000) e torch 2.13.0 contro 2.12.1. È la premessa che rendeva
sensato tutto il resto, ed è verificata.

Nota che va detta: la baseline batte la maggioranza di **1,8 punti** su questi dati, con un
errore standard di circa 3,4 punti. Sul nostro campione il suo vantaggio sul costante non è
distinguibile dal rumore.

## H11-bis — MORTA, e su entrambi i fronti

| | baseline (14 feat) | con H11-bis (23 feat) | guadagno | SE |
|---|---|---|---|---|
| set completo, n=164 | 0,7744 | 0,7561 | **−1,8 punti** | 2,0 |
| solo base_accuracy ≥ 0,05, n=162 | 0,7531 | 0,7284 | **−2,5 punti** | 1,7 |

Le dieci feature che la baseline non guarda (lunghezza della generazione, saturazione del
budget di token, presenza di una risposta, e i trend di entropia e logprob fra prima e seconda
metà del ragionamento) **peggiorano** la predizione su problemi mai visti. Non "non aiutano":
peggiorano, su entrambi i sottoinsiemi.

Nessuna variante è stata cercata dopo aver visto i numeri.

## Il valore di questa morte

Sul training set la differenza c'era, ed era forte: i casi `robust` saturano il budget di
token il **70%** delle volte contro il **46%** degli `spurious`. Chi si fosse fermato lì
avrebbe spedito una feature che in validazione su problemi nuovi costa **2,5 punti**.

È esattamente il modo in cui è morta H1 il 29 agosto: una differenza vera nel campione, che
non generalizza. Il fatto che sia successo due volte, su due canali completamente diversi,
dice qualcosa sul dominio, non sulla sfortuna.

## Bilancio delle ipotesi

**Otto pre-registrate, otto morte.**

| # | Ipotesi | Guadagno |
|---|---|---|
| H1 | il testo del problema predice la robustezza | −8,7 |
| H3 | la label di un modello predice quella di un altro | −7,3 |
| H4 | la base_accuracy di un modello predice la robustezza altrui | −11,8 |
| H5 | l'ordine di difficoltà trasferisce fra modelli | −25,4 |
| H6 | lo stesso con soglia oracolo (**il tetto**) | −13,0 |
| H7 | base_accuracy batte il prior su modelli mai visti | +0,0 |
| — | regola "trivial-robust" (+6,0 sui dati pubblici) | +0,0 sul proxy del test |
| H11-bis | ciò che le medie per token cancellano | −1,8 / −2,5 |

Sopravvive solo **H2**, ridimensionata da me stesso da +12,5 a **+3,4 punti** con il nullo
corretto, e richiede l'aggregazione su più modelli che il container non concede.

## Conclusione operativa

**Sul discriminatore non abbiamo un vantaggio, e nemmeno lo hanno gli altri.** La baseline
ufficiale non batte in modo distinguibile la classe di maggioranza sul set di sviluppo, e le
otto direzioni che ho testato sono tutte negative.

Il margine competitivo reale è **l'ingegneria**, ed è misurato: la baseline costa 119 s per
problema nel container e muore per timeout oltre i ~30 item, mentre `solution v2` sorveglia il
budget, degrada al prior e finisce sempre, con coverage 1 e invalid 0. Sul tie-break di
runtime, che è il criterio di spareggio della gara, questo è un vantaggio diretto.

Il secondo margine è il writeup: otto falsificazioni pre-registrate su un benchmark NeurIPS
non le ha nessuno.

## Materiale prodotto

- `gpu/out/feature_greedy.parquet` — 164 coppie, 28 colonne, il dataset di feature reali del
  modello in cache. **Riutilizzabile**: se la fase principale cambia le regole, l'estrazione
  non va rifatta.
- `gpu/aimo_gpu.py`, `gpu/addestra.py` — pipeline riproducibile end-to-end.
- Nessun artefatto di classificazione esportato: il verdetto non lo autorizza.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **H10 non e' stata eseguita come pre-registrata.** PREREG_E6 fissava «14 feature di incertezza da una generazione greedy, regressore, soglia». `gpu/addestra.py` addestra invece una regressione logistica standardizzata (C = 1,0) su 13 feature (manca generation_mean_prob), tre delle quali costanti nella nostra estrazione (max_logprob, frac_top1, frac_bassa). «Protocollo eseguito senza modifiche» e «baseline (14 feat)» sono quindi inesatti. Il numero (0,7744 contro una maggioranza di 0,7561) e il verdetto restano.
2. **Fonte del termine di confronto.** 0,692 +/- 0,101 e' il 69,23% (+/- 10,13) che la proposta degli organizzatori (arXiv 2607.13899) riporta per il miglior classificatore di incertezza, in CV stratificata a 10 fold sul loro sample validation set: un protocollo diverso dalla nostra CV raggruppata per problema.
