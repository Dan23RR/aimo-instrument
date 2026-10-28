# E16: il canale ufficiale non ha un potere predittivo stabile in segno

13 settembre 2026. Eseguito sulla VM degli organizzatori, dentro la loro immagine ufficiale,
sulla stessa GPU dei worker di valutazione. Tutto ricalcolabile da
`vm_risultati/e16*/*.jsonl` e `gpu/out/feature_greedy.parquet`.

## Cosa è stato misurato

Le 14 feature ufficiali e lo score del regressore ufficiale, estratti sui 174 problemi
pubblici per i modelli di gara, dentro il container ufficiale. Sui 164 problemi con verità a
terra per DeepSeek si può confrontare lo score con l'etichetta.

## Risultato 1: una nostra affermazione precedente era sbagliata, e va ritirata

`RISULTATI_E11_benchmark.md`, difetto 3, dichiara: «la soglia spedita costa 43 punti di
accuratezza», «il tasso vero di robust è 0,244; la soglia spedita ne predice 0,665».

**Sull'hardware di gara quel difetto non esiste.** La soglia ereditata 0,576 predice il
**25,6%** di robust contro un tasso vero del 24,4%. Il 66,5% di E11 veniva da feature
estratte su A100 noleggiate, con un'altra versione dello stack. **Il difetto 3 di E11 è
ritirato:** era un artefatto dell'ambiente di estrazione, non una proprietà dell'artefatto.

## Risultato 2: il segnale cambia direzione fra esecuzioni identiche

Tre estrazioni complete sui 174 problemi, stesso codice, stesso hardware, stessa immagine:

| esecuzione | mediana score | AUROC contro l'etichetta vera | predice robust |
|---|---|---|---|
| con flag deterministici | 0,6625 | **0,5131** | 25,6% |
| senza flag, ripetizione 0 | 0,6625 | **0,5131** | 25,6% |
| senza flag, ripetizione 1 | 0,6492 | **0,3980** | 33,5% |

Due fatti.

**I flag deterministici non c'entrano.** L'esecuzione senza flag, ripetizione 0, produce score
**identici su 174 problemi su 174** a quella con i flag. Il processo è semplicemente caduto
nello stesso stato numerico. La raccomandazione di E14 non è quindi la causa di nulla: va
corretta la parte della nota a Michal che lasciava aperto quel dubbio.

**Il potere predittivo dipende dallo stato numerico, e ne cambia il segno.** Fra le due
ripetizioni senza flag, gli score coincidono su 6 problemi su 174 e differiscono di 0,126 in
mediana. L'AUROC passa da **0,513** a **0,398**: la prima ordina (debolmente) nel verso atteso,
la seconda nel verso opposto. Non è una differenza di intensità, è una differenza di **segno**.

## Cosa unifica

E14 aveva misurato che gli stati spostano gli score di 0,13-0,23 in mediana, e l'aveva
presentato come un problema di riproducibilità del punteggio. E16 mostra che è peggio: quello
spostamento è abbastanza grande da **distruggere e invertire il segnale predittivo** del canale
su cui è costruita l'intera baseline ufficiale.

Non è che il canale sia debole. È che non ha una direzione stabile. Su A100 dava 0,649, qui dà
0,513 oppure 0,398 a seconda di come è partito il processo.

**Questo chiude la domanda aperta dall'inizio del progetto**, cioè perché nessuno batte la
classe di maggioranza. Non perché il segnale sia piccolo: perché non è ripetibile nemmeno nel
verso. Qualunque soglia calibrata su una esecuzione è anti-calibrata sull'esecuzione
successiva.

## Il termine di paragone che resta in piedi

| canale | AUROC sui 164 | costo GPU | ripetibile? |
|---|---|---|---|
| score ufficiale, stato A | 0,513 | ~75 min | no |
| score ufficiale, stato B | 0,602 (orientato) | ~75 min | no |
| **numero di caratteri del problema** | **0,705** | **zero** | **sì, esattamente** |

Contare i caratteri del testo ordina meglio delle quattordici feature che costano un'ora di
GPU, ed è l'unica cosa in tutto il progetto che dà lo stesso numero due volte di fila.

**Avvertenza, la stessa di sempre:** AUROC non è accuratezza. Una regola su `n_char` in CV a
5 fold raggruppata dà 0,7622 contro una maggioranza di 0,7561: **+0,6 punti**, dentro il
rumore. Il conteggio dei caratteri ordina meglio ma non classifica meglio, e questo va detto
ogni volta che il numero 0,705 compare.

## Conseguenze pratiche

1. Nessuna ricalibrazione della soglia del canale ufficiale è difendibile: non c'è un verso
   stabile da calibrare.
2. Una submission che dipende dagli score ufficiali eredita questa instabilità. Una che non
   li usa, no.
3. Per gli organizzatori: fissare il determinismo (E14) resta giusto, ma non basta. Due
   configurazioni deterministiche diverse danno due canali che ordinano in versi opposti.
   Il problema non è la varianza del punteggio, è che la feature su cui poggia la baseline
   non è una misura stabile di nulla.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Verso.** Il regressore predice il calo di accuratezza e la regola chiama robust gli score sotto la soglia: nel verso della regola le tre estrazioni danno 0,487, 0,487 e 0,602, non 0,513 «nel verso atteso» e 0,398 contro. `vm/e16_analisi.py`.
2. **Segno.** Uno stato non si distingue dal caso, l'altro e' debolmente sopra (p permutazionale 0,026 unilaterale, 0,051 bilaterale); la differenza fra i due non e' significativa (bootstrap appaiato al 95% da -0,015 a 0,246). I dati mostrano un ordinamento che non si riproduce fra esecuzioni identiche (Spearman 0,24), non un cambio di segno dimostrato.
3. **Flag.** La prima estrazione aveva solo CUBLAS_WORKSPACE_CONFIG; lo script non attiva gli algoritmi deterministici. Non e' il rimedio di E14, e «due configurazioni deterministiche» va letto «due stati numerici».
4. **Il confronto con A100** (0,649) usava la nostra estrazione, che non registra generation_mean_prob, riempita con exp(mean_logprob); riempita con la media della probabilita' top-1, come restituisce l'estrazione ufficiale in greedy, da' 0,640 (0,360 nel verso della regola): anti-predittiva.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Risultato 1.** La soglia spedita predice il 25,6% di robust solo nello stato raggiunto da due estrazioni su tre; nell'altro predice il 33,5%, e l'accuratezza della regola e' 0,598 e 0,640 contro 0,756 della costante di maggioranza, un costo da 12 a 16 punti invece di 43 (report, 8.2 e Appendice D, voci 3 e 8).
2. **Il 66,5% di E11** veniva dalle feature della nostra reimplementazione dell'estrazione, che dava generation_frac_selected_is_top1 = 1,0 su ogni item e non registrava generation_mean_prob (report, voci 3 e 5): «un'altra versione dello stack» non e' cio' che i dati identificano.
3. **Ritirato** «questo chiude la domanda ... perche' nessuno batte la classe di maggioranza»: dal 3 settembre la classifica pubblica aveva gia' voci sopra quelle degli organizzatori su entrambi i track.


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Precisazione al punto 3 della nota precedente.** Il ritiro di «Questo chiude la domanda aperta dall'inizio del progetto, cioè perché nessuno batte la classe di maggioranza» si regge sulla classifica del 13 settembre: quel giorno singoli invii avevano gia' raggiunto 0,95 nel Main e 1,00 nello Small (PREREG_V8; report, Appendice D, voce 1). Voci sopra quelle degli organizzatori c'erano dal 3 settembre (12/21 e 13/19; `vm/controlli_verifica3.py` su `dati/classifica_17850_2026-09-07.json`), ma da sole non mostrano di battere la costante di maggioranza, che vale almeno 11/21 e 10/19 e non fu misurata (report, 6.3).
2. **Fonti del punto 1 della nota precedente.** 25,6%, 33,5%, 0,598, 0,640 e 0,756: `vm/e16_analisi.py`.
