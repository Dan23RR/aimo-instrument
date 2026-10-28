# Pre-registrazione V8: la submission deterministica

Scritta il 13 settembre 2026, PRIMA di costruire il bundle e prima di qualunque invio.

## Perché questa e non un'altra

E16 (13 settembre, sulla VM degli organizzatori, dentro la loro immagine) ha misurato che lo
score del regressore ufficiale **cambia segno fra esecuzioni identiche**: AUROC 0,513 in una
ripetizione e 0,398 in un'altra, stesso codice, stesso hardware, stessa immagine. Non esiste
un verso stabile da calibrare. Ogni metodo che poggia su quel canale eredita l'instabilità.

Resta una sola cosa che in questo progetto ha dato lo stesso numero due volte di fila: il
**conteggio dei caratteri del problema**. Non tocca il modello, non usa la GPU, non dipende
dallo stato numerico del processo. È l'unica submission che possiamo difendere come
riproducibile, ed è anche la più economica possibile per il criterio di spareggio sul runtime.

## La regola, congelata adesso

```
are_robust(model_id, problems)  ->  [len(p) >= 98 for p in problems]
```

**Soglia 98 caratteri**, cioè la mediana di `n_char` sui 174 problemi pubblici degli
organizzatori. Nessuna dipendenza da `model_id`, dal lotto, dall'ordine o da quantili calcolati
sul batch: la predizione dell'item i dipende solo dall'item i. Rispetta il vincolo
anti-contaminazione fissato in `PIANO_E11_definitivo.md`.

**Direzione: i problemi lunghi sono i robusti.** Verificata sui dati pubblici e, per
trasparenza, invertita rispetto alla mia prima stesura: il calcolo ha mostrato accuratezza
bilanciata 0,342 con il verso sbagliato e 0,658 con quello giusto. L'errore è stato trovato
prima di spedire perché la regola è stata valutata, non assunta.

## Numeri di riferimento, dai dati pubblici (335 coppie etichettate, 23,9% robust)

| | |
|---|---|
| accuratezza bilanciata | **0,658** |
| sensibilità / specificità | 0,850 / 0,467 |
| predice robust | 60,9% degli item |
| su DeepSeek e qwen3-8b (n=216) | 0,688 |
| su tutti gli altri modelli (n=119) | 0,600 |

## Previsioni falsificabili, scritte prima dell'invio

Gli organizzatori hanno dichiarato in Discord il 2 settembre che il set di valutazione è
«rebalanced» e che «it's not possible to improve solely by model_id or idiosyncratic features
of the problem». **La lunghezza del testo è esattamente una feature idiosincratica del
problema.** Quindi la previsione principale è pessimistica.

- **P-V8a.** Se hanno neutralizzato la lunghezza, V8 otterrà fra 0,45 e 0,55 su entrambi i
  track, cioè **peggio o uguale a v6** (0,52 main, 0,53 small).
- **P-V8b.** Se non l'hanno neutralizzata e il set è bilanciato, V8 otterrà **circa 0,66**,
  cioè 13 punti sopra v6.
- **P-V8c.** In ogni caso V8 avrà `coverage 1.0`, `invalid_predictions 0` e un runtime
  **inferiore a 60 secondi**, contro i 619-822 secondi di v6.

**Il risultato più probabile è P-V8a, e lo dico prima.** Spedisco perché il caso in cui una
regola a costo zero batte un'ora di GPU è informativo quanto quello in cui fallisce, e perché
il tempo di esecuzione è un dato che serve al report indipendentemente dal punteggio.

## Costo dichiarato

`Force_Latest_Multiple`: questo invio **sostituisce** v6 in classifica su entrambi i track.
Se P-V8a si avvera perdiamo uno o due punti percentuali di posizione. È accettato
consapevolmente: la posizione attuale è comunque dentro il rumore di riesecuzione misurato in
E14, e il bersaglio dichiarato del progetto è il report, non il rango.

## Cosa NON si fa

- Nessuna soglia tarata sul punteggio di gara o su range osservati nei log.
- Nessuna regola che dipenda dalla composizione del lotto.
- Nessun secondo invio per "provare un'altra soglia": sarebbe tuning sul canale del punteggio.

## Esito

Da riempire dopo l'invio, senza riscrivere nulla di quanto sopra.

## ESITO, 13 settembre 2026, ore 17:25. P-V8a confermata.

Inviate due submission: **924968** `solution-v8-main.zip` e **924972** `solution-v8-small.zip`.

**Main: 0,5238 = 11/21. Small: 0,5263 = 10/19.** Identici a v6 su entrambi i track, che
costava 838 e 509 secondi di GPU. (Letto dall'API il 14/9: ogni invio è stato valutato dalla
piattaforma su entrambi i task perché le due caselle non erano deselezionabili; il task il cui
vincolo su `small.txt` il bundle viola lo rifiuta prima dell'import, coverage 0 e 21 o 19
predizioni non valide. Il figlio valido di 924968 è il Main, quello di 924972 è lo Small.)

**Correzione del 14 settembre, ore 11:05.** Con la regola «conta l'ultimo invio per task», il
figlio fallito 924973 (Main, 0,00) era diventato la nostra riga Main in classifica. Rimediato
con un terzo invio dello stesso identico bundle `solution-v8-main.zip`, questa volta con il solo
task Main selezionato (casella Small deselezionata via `document.getElementById('task-35767').click()`,
stato verificato `{main:true, small:false}` prima del caricamento; il primo tentativo è fallito
per un 502 del proxy di storage di Codabench, il secondo è passato). Esito, letto dall'API:
**925822, task 35766 unico, zero figli, accuracy 0,5238, coverage 1,0, invalid 0**. Nota sulla
visualizzazione, verificata sulla pagina dei risultati: la regola dichiarata dall'API è
`Force_Latest_Multiple`, ma la tabella pubblica elenca **ogni invio come riga a sé** (614 righe
al 14/9, 50 per pagina; un solo account occupa 21 delle 45 righe lette nella prima pagina),
quindi il rango stampato è per invio, non per partecipante. Tenendo per ogni partecipante solo
l'ultimo invio per task, come dice la regola, siamo 8° su 43 nel Main (7 sopra, 4 a pari merito)
e 8° su 46 nello Small (4 sopra, 15 a pari merito). Non è un nuovo esperimento e non cambia
nessuna previsione: è lo stesso codice, valutato una terza volta con lo stesso risultato, il
che è a sua volta un dato (tre esecuzioni, tre volte 11/21).

**P-V8a era la previsione principale ed è confermata:** V8 cade dentro la banda 0,45-0,55 e
non batte v6. La lunghezza del testo non aiuta sul set di valutazione, coerentemente con la
dichiarazione degli organizzatori di aver neutralizzato le feature idiosincratiche del
problema. P-V8b (circa 0,66) è falsificata.

**Il dato che resta, e che era il vero motivo dell'invio:** una regola di una riga, senza
alcun accesso al modello, ottiene **lo stesso punteggio** di una pipeline che carica cinque
modelli da miliardi di parametri e genera per oltre dieci minuti. Sul canale del punteggio le
due cose sono indistinguibili.

## Osservazione non pre-registrata, fatta leggendo la classifica subito dopo l'invio

[REDACTED: an observation, made after our submission and not pre-registered, about other participants' leaderboard scores of 13 September (their usernames and figures, an argument that a score lay outside a reliability ceiling, and a remark about another participant's GitHub issue). The argument was wrong and is retracted below, on 14 September; nothing was done on it]

## RITRATTAZIONE, 14 settembre 2026. L'osservazione qui sopra è sbagliata e viene ritirata.

Il testo resta com'era, per il registro. L'argomento del «tetto» usava l'affidabilità del
**nostro** processo di rigenerazione (kappa 0,41, E13) dove serviva quella del processo degli
**organizzatori**, che simulato con il loro regime (dieci varianti per famiglia, una predizione
per variante, etichetta «robusta» solo con zero fallimenti) dà 0,93 (IC90 0,87-1,00); e
l'aritmetica era sbagliata di un fattore due (0,70 invece di 0,82 anche con il numero sbagliato).
Con l'affidabilità giusta, un 19/19 non sta fuori da nessun tetto. La verifica completa è in
`VERIFICA_classifica_2026-09-14.md` (revisione strutturale del codice di valutazione: nessuna
via di fuga per le etichette; sequenze dei punteggi; nessun rilascio di dati fra l'8 e il 13);
la ritrattazione pubblica è nel report, §11.4. **Nessuna segnalazione agli organizzatori su
terzi è stata fatta né va fatta**: non c'è alcuna prova, e la sola cosa che l'osservazione ha
prodotto è una correzione a nostro carico.

## CORREZIONE, 21 settembre 2026 (verifica avversaria del report)

«Tre esecuzioni, tre volte 11/21» è sbagliato. Sui task vecchi le valutazioni valide di V8 sono
tre, ma su task diversi: il bundle Main ha dato 11/21 due volte sul task Main (924969, figlio di
924968, e 925822); il bundle Small ha dato 10/19 una volta sul task Small (924974, figlio di
924972). I due figli rifiutati (924970, 924973) non contano. Il fatto che resta è più piccolo:
due valutazioni dello stesso bundle sullo stesso task, due punteggi identici.


## CORREZIONE, 23 settembre 2026 (verifica avversaria del report)

I numeri di riferimento sopra (accuratezza bilanciata 0,658, specificita' 0,467, predice robust
60,9%) sono quelli della soglia 99, non della 98 spedita. Per la regola spedita (n_char >= 98)
sui dati pubblici: accuratezza bilanciata 0,653, sensibilita' 0,850, specificita' 0,455, predice
robust 61,8%. La mediana dei 174 problemi e' 98,5: 98 e' la mediana inferiore. Sul train set
pubblico della fase principale (`vm/v8_train_main_v2.py`) la regola ottiene 46/71 = 0,648,
sotto la costante «sempre robust» (52/71 = 0,732), accuratezza bilanciata 0,442.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Ritirato** (report, Appendice D, voce 16): «La lunghezza del testo non aiuta sul set di valutazione, coerentemente con la dichiarazione degli organizzatori di aver neutralizzato le feature idiosincratiche del problema». Una valutazione su 21 e 19 item non lo mostra; 0,66 sta dentro gli intervalli al 95% di entrambi i punteggi, quindi i dati non escludono nemmeno P-V8b.
2. **Force_Latest_Multiple.** La regola dichiarata non tiene solo l'ultimo invio per task: Codabench aggiunge in classifica ogni invio valutato come riga a se' e non ne toglie nessuno. 924969 (11/21) e' rimasto in classifica accanto al figlio rifiutato a 0,00; il reinvio 925822 resta un dato valido, ma la motivazione scritta qui era sbagliata.
3. **Accuratezza bilanciata della regola.** La nota del 23/9 sopra corregge 0,658 in 0,653; il valore esatto a soglia 98 e' 0,65245, cioe' 0,652, sulle 335 coppie, e 0,631 sulle 283 coppie distinte (le 52 misure DeepSeek pubblicate con due nomi contate una volta). `vm/dati_pubblici_controlli.py`.
4. **Dati di addestramento.** La regola non usa modelli, ma i suoi due parametri (soglia 98 e verso) sono fissati sui sei dataset pubblici, che sono quindi i suoi dati di addestramento.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Regime nella ritrattazione del 14/9.** «Il loro regime (dieci varianti per famiglia, una predizione per variante)» descrive il file dei problemi AIMO, non i file MATH dietro le etichette DeepSeek, dove la maggior parte delle varianti ha dieci risposte e la base da 1 a 60 (report, 2.1 e Appendice D, voce 34; `vm/dati_pubblici_controlli.py`, `vm/controlli_verifica4.py`). E il simulato 0,93 non e' «l'affidabilita' giusta»: applicata alle nostre etichette, la stessa simulazione sovrastima l'affidabilita' misurata di circa 0,4, quindi l'affidabilita' dell'etichetta ufficiale resta non misurata (report, 7.3 e voce 41). L'argomento del tetto resta ritirato per le altre due ragioni scritte sopra. La ritrattazione pubblica e' ora la voce 4 dell'Appendice D.
2. **E16 non ha mostrato un cambio di segno** (motivazione di V8 sopra; report, 8.2 e voci 8 e 45): nel verso della regola i due stati danno AUROC 0,487 e 0,602, uno al caso e l'altro debolmente sopra, e la differenza non e' significativa (bootstrap appaiato al 95% da -0,015 a 0,246); cio' che non si riproduce fra esecuzioni identiche e' l'ordinamento (Spearman 0,24).
3. **Il reinvio 925822 del 14/9** fu caricato dall'assistente dal browser dell'autore senza un'istruzione specifica per quell'invio, per ripristinare una riga Main che credeva persa per la lettura sbagliata di Force_Latest_Multiple (nota precedente, punto 2). Il report lo dichiara (par. 13).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Runtime di v6 in P-V8c.** «contro i 619-822 secondi di v6» non e' esatto. 619 s e' il totale di v5 nel warmup (PREREG_V5), v6 nel warmup impiego' 785 s (PREREG_V6), e 619-822 s e' l'intervallo delle esecuzioni dei metodi nel warmup, registrato solo nella bozza v1.0 del report (Sezioni 1 e 8). Nella fase principale v6 ha impiegato 838 s nel Main e 509 s nello Small (esito sopra; OSSERVATO_MAIN_2026-09-02.md; report, 6.2).
2. **Fonti e citazioni della nota precedente.** I numeri del punto 2 (0,487, 0,602, l'intervallo da -0,015 a 0,246 e lo Spearman 0,24) vengono da `vm/e16_analisi.py`. Le citazioni del punto 1 vanno lette alla lettera: «simulato con il loro regime (dieci varianti per famiglia, una predizione per variante, etichetta «robusta» solo con zero fallimenti) dà 0,93» e «Con l'affidabilità giusta».
