# Esito E5 — il segnale trivial-robust è una trappola, e la verifica l'ha fermato

29 agosto 2026. `esperimento_e5.py`.

## Il numero seducente

Sui dati pubblici completi, la regola "robust se base_accuracy < 0,05, altrimenti spurious"
batte il prior tutto-spurious di **+6,0 punti** (0,821 contro 0,761), z ben sopra soglia.
Sei punti gratis sarebbero enormi in una gara vinta da chi supera il 69,2%.

## Perché è morto, verificato prima di costruirci sopra

Il +6,0 viene **tutto** da un solo dataset, `challenge-sample-full`, che ha 20 casi ba~0 su
80. Ma quel dataset non è il test.

Il proxy del test è **val-sample**, che gli organizzatori dichiarano coprire "tutti i tipi di
modelli del test set". Distribuzione di base_accuracy in val-sample:

| dataset | ba~0 | ba~1 | intermedi | robust |
|---|---|---|---|---|
| **val-sample (proxy del test)** | **0** | 17 | 7 | 0,38 |
| augmented-agg | 0 | 134 | 3 | 0,26 |
| augmented-agg-FILTERED | 0 | 50 | 2 | 0,21 |
| challenge-sample-full | **20** | 33 | 27 | 0,39 |

**val-sample ha zero casi ba~0.** Il test è filtrato per accuratezza di base alta: contiene
solo problemi che il modello sa risolvere, dove la robustezza è la domanda vera e non banale.
Sulla stessa regola applicata a val-sample: **+0,0 punti**.

Se avessi costruito il metodo sul +6,0 dei dati pubblici, avrei fatto overfitting su una
distribuzione che il test non ha. La regola "verifica sul proxy del test, non sul pool
pubblico più comodo" l'ha intercettato.

## Cosa resta, senza illusioni

Nel test (proxy val-sample) i casi sono tutti ad accuratezza di base alta: il modello risolve
l'originale, e la domanda è se regge le perturbazioni. Distinguere robust da spurious lì
**richiede le feature interne del modello** (incertezza, stabilità sotto perturbazioni),
che non sono calcolabili offline senza far girare il modello.

**Conclusione onesta:** offline, senza GPU, non posso dimostrare un metodo che batte il 69,2%,
perché il segnale discriminante vive in feature che richiedono il modello. Il valore
costruibile ora è (a) il metodo deployabile scritto e pronto, (b) la robustezza al budget di
tempo che la baseline non ha, (c) la carta di misura per il writeup.

## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Regola T e righe duplicate.** Il +6,0 e' calcolato sulle 335 coppie, con le 52 misure DeepSeek contate due volte. Contate una volta (283 coppie) la regola vale 0,8269 contro 0,7562, +7,1 punti con soglia 3,0; su val-sample resta +0,0, perche' val-sample non contiene duplicati. La conclusione (trappola) non cambia. `vm/warmup_ricalcoli.py`.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Descrizione di val-sample.** «val-sample, che gli organizzatori dichiarano coprire tutti i tipi di modelli del test set» non ha una fonte primaria nei nostri archivi: nessuna versione salvata del README degli organizzatori ne' la scheda del dataset la contiene. E val-sample non contiene Olmo-3-7B-Instruct ne' gemma-3-27b-it, due dei quattro modelli del warmup (report, 2.4 e Appendice D, voce 54).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Inferenze sul set di test.** Ritirate anche «Il proxy del test è val-sample», «Il test è filtrato per accuratezza di base alta: contiene solo problemi che il modello sa risolvere, dove la robustezza è la domanda vera e non banale» e «Nel test (proxy val-sample) i casi sono tutti ad accuratezza di base alta». val-sample non ha coppie etichettate con base sotto 0,05 (`vm/controlli_verifica3.py`), ma che rappresenti il set di test non ha fonte (report, 2.4). Vale anche per il commento di esperimento_e5.py, riga 13.
