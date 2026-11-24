# Tre difetti verificati del benchmark AIMO, e il tetto del canale ufficiale

3 settembre 2026. Verificato personalmente, riga per riga, dopo che un workflow avversario a
nove agenti li ha proposti. Ogni numero qui sotto l'ho ricalcolato io dai file su disco.

## Nota di metodo, che viene prima dei risultati

**Questi non sono risultati pre-registrati e non vengono presentati come tali.** Sono stati
trovati cercandoli, quindi la disciplina delle PREREG non si applica. Va detto perché in
questo progetto la differenza conta.

Ma va detta anche la ragione per cui reggono lo stesso: **i primi due non sono inferenze
statistiche, sono ispezioni**. "379 righe su 379" e "4915 split su 28738" sono conteggi
deterministici su file pubblici, riproducibili da chiunque in un minuto, senza stime,
intervalli o soglie. Non c'è nulla da pre-registrare in un conteggio. Il terzo è in parte
statistico e viene riportato con la sua validazione incrociata.

---

## Difetto 1. L'etichetta è leggibile dai dati pubblici senza alcun modello

Nei dataset pubblici degli organizzatori, `permutation_type` è una **lista** se e solo se
l'item è `robust`:

| dataset | accordo |
|---|---|
| val-sample | 28/28 |
| augmented-sample-math-agg | 141/141 |
| augmented-sample-math-agg-filtered | 54/54 |
| augmented-sample-math | 156/156 |
| **totale** | **379/379 = 100,00%** |

**Il meccanismo è identificato**, non congetturato: nel file non aggregato
`aimo-interp-challenge-sample-full.json` (558 righe) `permutation_type` **non è mai una
lista** (0 su 558). L'aggregazione accorpa tutti i tipi di perturbazione per gli item robusti
e conserva il solo tipo dannoso per gli altri. La forma sintattica del campo tradisce
l'etichetta.

**Perché conta.** Chi addestra sui dati pubblici includendo quel campo, o una feature che lo
riflette, ottiene un classificatore quasi perfetto in sviluppo che a inferenza non ha quel
campo e crolla. È una trappola silenziosa piazzata sul percorso naturale di ogni
partecipante. Riproducibile in un minuto: 379 righe, quattro file.

---

## Difetto 2. L'artefatto ufficiale poggia per un quinto su una feature fuori supporto

L'artefatto spedito nel bundle di gara
(`uncertainty_artifacts/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib`) è una
`Pipeline(SimpleImputer, RandomForestRegressor)`, addestrata su **141 righe / 137 gruppi**
di un solo modello (`training_provenance`).

| | |
|---|---|
| split totali nella foresta | 28.738 |
| split su `generation_frac_selected_is_top1` | **4.915 (17,1%)** |
| importanza della feature | **0,2100**, seconda su quattordici |
| soglie usate su quella feature | tutte in **[0,96836, 0,99348]** |
| valore nella nostra estrazione | **costante 1,00000 su 164/164** |
| esito | **fuori supporto**: 1,0 sta sopra ogni soglia |

**La causa è nella loro stessa configurazione.** L'artefatto dichiara `do_sample: false`, cioè
decoding greedy. **Con decoding greedy il token emesso è il top-1 per definizione**, quindi
quella feature vale identicamente 1 a inferenza. Le soglie tutte sotto 1 dimostrano che
l'addestramento è avvenuto in un regime diverso da quello di produzione.

**Conseguenza.** Il 17% degli split della foresta instrada ogni campione di produzione
deterministicamente dalla stessa parte: un quinto della capacità del modello di riferimento
non discrimina nulla. Vale per chiunque usi l'artefatto ufficiale, cioè per buona parte del
campo.

Altre due delle quattordici feature sono costanti nella nostra estrazione
(`max_logprob` ≡ 0, `frac_low_conf_tokens` ≡ 0) ma non compaiono in nessuno split, quindi
sono inerti e non dannose. Le feature realmente informative sono **dieci**, non quattordici.

---

## Difetto 3. La soglia spedita costa 43 punti di accuratezza

Applicando l'artefatto ufficiale alle nostre 164 estrazioni, di cui **137 stanno nel suo
stesso insieme di addestramento** (quindi per lui è un test in-sample, dove dovrebbe
eccellere):

| | accuratezza |
|---|---|
| **regola spedita** (`score < 0,576` → robust) | **0,3232** |
| classe di maggioranza | 0,7561 |
| soglia ricalibrata, **CV raggruppata per problema** | **0,7500** (SE 0,034) |
| soglia ottima in-sample (limite superiore non onesto) | 0,7622 |

Il tasso vero di `robust` è **0,244**; la soglia spedita ne predice **0,665**. È scalibrata di
un fattore quasi tre, e questo costa **42,7 punti, settanta item su 164**.

Lo score in sé **non è rotto**: ordina nella direzione giusta, con AUROC **0,6486** e Spearman
+0,221. È la soglia a essere sbagliata, non il modello.

---

## Il risultato che chiude il canale, ed è il più importante

**Anche con la soglia perfettamente ricalibrata, il canale ufficiale non batte una costante:
0,7500 contro 0,7561 della classe di maggioranza.** Meno un item su 164.

Messo accanto a quello che avevamo già misurato:

| canale | AUROC OOF | batte la maggioranza in accuratezza? |
|---|---|---|
| le 10 feature ufficiali informative | 0,656 | no |
| **il solo numero di caratteri del problema** | **0,698** | no |
| entrambe | 0,702 | no |

Contare i caratteri del testo, a costo zero di GPU e in microsecondi, **ordina meglio delle
quattordici feature che costano quaranta minuti di A100**. Nessuno dei due batte la costante
in accuratezza, che è la metrica della gara.

**Avvertenza necessaria:** AUROC non è accuratezza. Con il 24,4% di classe positiva e una
maggioranza a 0,7561, un AUROC di 0,70 non implica battere la maggioranza. H1 (il testo)
resta morta sull'accuratezza; sale solo sull'ordinamento. Le due cose non vanno confuse, ed è
esattamente l'errore che il campo sta facendo.

---

## Cosa ne segue

**Per la gara.** Nessuno dei tre difetti è una leva di classifica. Il primo riguarda i dati di
sviluppo, non il test. Il secondo e il terzo, corretti, portano il canale ufficiale a 0,75,
sotto la costante. Questo spiega, dall'interno e con verità a terra, perché al 3 settembre
l'intera classifica è ferma alla classe di maggioranza: **non è che i partecipanti non
provino, è che lo strumento fornito non arriva alla costante nemmeno usato al meglio.**

**Per il report.** Sono tre difetti verificabili, non noti, riproducibili a costo zero da
chiunque scarichi i file pubblici, e insieme costituiscono una diagnosi completa di perché la
competizione non discrimina. È materiale da sezione propria.

**Per la nostra soluzione.** v6 usa la soglia 0,576 su quattro modelli su cinque. Il difetto 3
dice che quella soglia è scalibrata sui dati pubblici. **Ma non la cambiamo su questa base**:
la ricalibrazione arriva a 0,7500, sotto la maggioranza, quindi non c'è guadagno atteso, e i
range di score osservati in gara sono diversi da quelli dei 164 (la scalibratura potrebbe non
trasferire). Cambiare la soglia sarebbe un'azione senza guadagno dimostrato e con rischio non
misurato. Resta v6.

## Riproducibilità

Ogni numero è ricalcolabile da: `dati/*.json` (sei dataset pubblici),
`gpu/out/feature_greedy.parquet` (164 estrazioni con le feature ufficiali e le etichette
vere), `soluzione/uncertainty_artifacts/*.joblib` (l'artefatto ufficiale). Nessun dato
privato, nessuna informazione dal canale del punteggio.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **«Lo score in se' non e' rotto: ordina nella direzione giusta».** E' sbagliato. Il regressore predice il calo di accuratezza e la regola chiama robust gli score bassi: un'AUROC grezza di 0,6486 con robust come classe positiva e' anti-predittiva nel verso della regola (0,351). Inoltre la nostra estrazione non registra generation_mean_prob: 0,6486 viene dal riempirla con exp(mean_logprob); con la media della probabilita' top-1, come l'estrazione ufficiale in greedy, e' 0,640 (0,360 nel verso della regola). `vm/e16_analisi.py`.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Difetto 2 ritirato** (report, 2.3b e Appendice D, voce 5): il valore costante 1,0 di generation_frac_selected_is_top1 veniva dalla nostra reimplementazione dell'estrazione, che calcolava la feature con un argmax; l'estrazione ufficiale sull'hardware degli organizzatori non restituisce mai 1,0 (da 0,958 a 0,999 su 1.964 esecuzioni-item, il 92,6% dentro l'intervallo delle soglie; `vm/e16_analisi.py`). Sono ritirate quindi le affermazioni che un quinto della foresta instradi ogni campione allo stesso modo e che il difetto valga per chiunque usi l'artefatto ufficiale.
2. **Difetto 3** fu ritirato in RISULTATI_E16_segnale.md (13 settembre): sull'hardware di gara la soglia spedita predice il 25,6% o il 33,5% di robust secondo lo stato numerico e costa da 12 a 16 punti contro la costante di maggioranza, non 43 (report, Appendice D, voce 3). Dei tre difetti del titolo resta solo il primo.
3. **Ritirato** «al 3 settembre l'intera classifica e' ferma alla classe di maggioranza» e la spiegazione dei punteggi altrui che segue: quel giorno c'erano gia' voci sopra quelle degli organizzatori su entrambi i track (12/21 alle 03:35 UTC e 13/19 alle 09:19 UTC; `vm/controlli_verifica3.py` su `dati/classifica_17850_2026-09-07.json`).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Difetto 1, meccanismo.** Il difetto resta, ma con il meccanismo corretto: per gli item non robusti l'aggregazione conserva il tipo o i tipi ad accuratezza perturbata minima e scarta i tipi dannosi piu' lievi, non il solo tipo dannoso (16 delle 41 coppie non robuste di augmented-sample-math-agg coperte dal file non aggregato; `vm/controlli_verifica4.py`, sez. 2; report, 2.3a e Appendice D, voce 54).
2. **Lo 0,7500 del canale ufficiale.** «Anche con la soglia perfettamente ricalibrata, il canale ufficiale non batte una costante: 0,7500 contro 0,7561 della classe di maggioranza» e le conclusioni che ne seguono sono calcolati applicando il regressore spedito alle feature della nostra reimplementazione dell'estrazione su A100 (gpu/out/feature_greedy.parquet: generation_frac_selected_is_top1 = 1,0 su 164/164, generation_mean_prob non registrata), non sul percorso ufficiale. Sul canale ufficiale un'accuratezza con soglia ricalibrata non e' stata misurata (report, 2.3b e 8.2; `vm/e16_analisi.py`).
3. **Ritirato** anche «ed è esattamente l'errore che il campo sta facendo» (Avvertenza necessaria): non abbiamo osservato il metodo di nessun partecipante (report, par. 11 e 13).
4. **Fonti del punto 2 della nota precedente.** 25,6%, 33,5%, 0,598, 0,640 e 0,756: `vm/e16_analisi.py`.
