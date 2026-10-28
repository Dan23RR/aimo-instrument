# PIANO DEFINITIVO E11 (v2) — pronto per PREREG_E11.md

Scritto il 3 settembre 2026 dopo tre attacchi. Ogni numero marcato **[verificato ora]** l'ho misurato io su disco durante questa revisione, ricontrollando anche gli attaccanti. Script: `...\afb9b121-13de-4913-8c95-f4be3c1f36db\scratchpad\v1.py` ... `v14.py`.

---

## 0. Verdetto, prima di tutto

**Il piano E11 come programma per battere una baseline di AUROC è morto. Non si esegue.** Non per la ragione data dall'attacco statistico, che ho verificato essere in parte sbagliata, ma per una ragione più semplice e più dura che l'attacco ingegneristico ha trovato e che ho confermato e rafforzato.

Sopravvivono tre cose, in quest'ordine di valore:

1. **Una modifica di produzione a costo zero GPU** che elimina il rischio di timeout, vince il tie-break sul runtime, e non peggiora l'accuratezza in modo dimostrabile.
2. **Il writeup**, il cui contenuto cambia: non più "l'instabilità di rappresentazione predice la robustezza" ma tre difetti verificabili del benchmark ufficiale, tutti riproducibili da chiunque scarichi i file pubblici, tutti a costo zero.
3. **Un esperimento sulle attivazioni molto più piccolo (~1,0 euro)**, il cui deliverable primario non è un guadagno in classifica ma un numero che oggi non esiste da nessuna parte: la ripetibilità test-retest di una feature di distanza fra attivazioni. Si esegue solo dopo i cancelli offline, e si esegue sapendo in anticipo che l'esito atteso è negativo.

---

## 1. Cosa ho verificato io, incluse le correzioni AGLI ATTACCHI

Gli attaccanti si contraddicono su due punti decisivi. Li ho risolti.

### 1.1 Confermato senza riserve

| claim | fonte | esito |
|---|---|---|
| `permutation_type` che inizia con `[` ⇔ `model_is_robust` | attacco statistico F1 | **223/223** su val-sample (28), agg (141), agg-filtered (54). **[verificato ora]** Nel file `aimo-interp-challenge-sample-full.json` (558 righe, non aggregato) `permutation_type` non è **mai** una lista, il che identifica il meccanismo: le righe robuste aggregano su tutti i tipi, le non robuste solo sul tipo che ha fatto danno |
| I contatori di bookkeeping predicono l'etichetta | F2 | AUROC(robust, `n_permuted_predictions`) = **0,7798**; OOF dei due contatori = **0,8619** sul merge corretto a 138 righe **[verificato ora]** |
| Il difetto dell'artefatto ufficiale è `frac_selected_is_top1` fuori supporto | ingegneria #4, statistica L7 | **4915 split su 28738 (17,1%)**, importanza **0,2100** (seconda su 14), tutte le soglie in **[0,96836, 0,99348]**, nostro valore **1,0000 su 164/164**, loro mediana di training **0,98254**. **[verificato ora]** Diagnosi completa a zero euro |
| Le famiglie di perturbazione sono degeneri | ingegneria #2 | `format` no-op **58,5%**, `latex_delims` no-op **21,3%** con AUROC(no-op) **0,623** e Fisher **p=0,0017**, `question_first` inapplicabile **66,5%**; `framing` e `redundancy` sempre attive **[verificato ora]** |
| La soglia di morte 0,62 è sotto il null di selezione | statistica M1, ingegneria #5 | max-of-k su rumore, n=164, 40 positivi: k=37 mediana **0,610** p95 **0,657**; k=185 mediana **0,640** p95 **0,676**. La soglia 0,62 sta **sotto la mediana** del null a 37 celle **[verificato ora]** |

### 1.2 Refutato: H27 non è un cancello (l'attacco statistico sbaglia, l'ingegneria ha ragione)

L'attacco statistico chiede di rendere H27 un cancello duro perché il tetto da rumore di etichetta starebbe a 0,744 con flip 0,20. **È falso.** `base_accuracy == 1` nel **96,6%** delle righe, quindi l'etichetta è letteralmente "tutte le m predizioni perturbate corrette", ma i tassi osservati sono quasi ovunque degeneri: ricampionando due repliche indipendenti dai tassi osservati con i loro n reali, **l'accordo è 0,9998** **[verificato ora]**. Il tasso di flip non è 0,20, è ~0,0002. Il tetto da rumore di etichetta è ~1,0 e **non vincola nulla**. H27 si chiude adesso, con quel numero, e non consuma uno stadio.

### 1.3 Refutato: F3 non è stabilito, e il suo contrario è più interessante

L'attacco statistico dichiara FATALE sulla base di "nello strato non confonduto M1 crolla a 0,5554". Sul merge corretto (138 righe DeepSeek, accordo etichette **100%**; il suo merge a 145 includeva 5 problemi con i metadati di **altri modelli**) **[verificato ora]**:

| | strato m<=50, n=112, 10 positivi |
|---|---|
| M1 (10 uff + `n_token` + `satura_cap`) OOF | **0,6028**, IC95 bootstrap **[0,362, 0,803]** |
| `n_char` (lunghezza del problema) OOF | **0,8676**, IC95 **[0,780, 0,939]** |

L'IC di M1 **contiene** il valore a campione pieno (0,696): il crollo non è dimostrato, è un risultato senza potenza, e l'attaccante cita lui stesso SE 0,158 e poi ne trae una conclusione che quella SE non sostiene. Ma nello stesso strato la feature di solo testo **sale a 0,868**. Lo strato non dice "è tutto bookkeeping": dice che il canale derivato dal modello è debole e il canale di testo è forte.

Test permutazionale stratificato su m (l'analisi corretta, non la stratificazione su un collider) **[verificato ora]**, 138 righe:

| canale | AUROC grezza | AUROC stratificata | p |
|---|---|---|---|
| `n_char` | 0,7331 | **0,8295** | <0,0001 |
| `mean_entropy` | 0,6392 | **0,7318** | 0,009 |
| `ppl` | 0,6405 | **0,7185** | 0,015 |
| `frac_alta` | 0,3762 | **0,2815** | 0,017 |
| `n_token` | 0,6604 | 0,5381 | 0,343 |
| `secondi` | 0,7018 | 0,5695 | 0,248 |

Lunghezza del testo e incertezza a livello di token **sopravvivono e si rafforzano**; lunghezza della generazione e tempo di parete **crollano a caso**. Conseguenza diretta: **M1 come definito nel piano originale (11 ufficiali + `n_token` + `satura_cap`) è la specificazione più contaminata possibile**, perché aggiunge esattamente i due canali che il controllo uccide.

### 1.4 Il numero che riordina tutto

**[verificato ora]**, sui 164 completi, CV 5-fold x20 stratificata (una riga per problema, quindi grouped ≡ stratified):

| modello | OOF AUROC | costo |
|---|---|---|
| `n_char` da sola (lunghezza in caratteri) | **0,6868** (diretta 0,7039) | zero GPU, microsecondi |
| 10 feature ufficiali non degeneri | 0,6462 | ~40 min A100 |
| 10 uff + `n_token` + `satura_cap` (il vecchio M1) | 0,6960 | ~40 min A100 |
| `n_char` + 10 uff + `n_token` + `satura_cap` | 0,7049 | ~40 min A100 |

Delta appaiato bootstrap (M0_TESTO meno M1): **-0,0062, IC95 [-0,114, +0,104]**. Indistinguibili. E la lunghezza **non** è un artefatto di dataset: dentro `hendrycks-math-test-reference-100` vale 0,818, dentro `hendrycks-math-train-reference-500` vale 0,718, e `dataset_id` da solo vale 0,512 **[verificato ora]**.

Correzione minore ma vincolante: le feature ufficiali degeneri sono **tre**, non due. `frac_top1` ≡ 1, `frac_bassa` ≡ 0, e **`max_logprob` ≡ 0,0 con `nunique = 1`** **[verificato ora]**. Il blocco ufficiale informativo ha **10 dimensioni**, non 11 e non 14. Entrambi gli attaccanti hanno scritto 11.

---

## 2. Cosa muore, in chiaro

| voce del piano originale | esito | motivo |
|---|---|---|
| **S0.5, submission `always False`** | **CANCELLATA** | Violazione tripla e non recuperabile: STRATEGIA.md:98-102 la definisce sonda, PREREG_E1.md:28-31 chiude la linea, WRITEUP.md:155 la dichiara già ritirata, e `Force_Latest_Multiple` la renderebbe pubblica sovrascrivendo v6. Legge il prior con IC ±0,21 su 21 item, cioè non lo legge |
| **H26 con `pi_test = 0,45`** | **CANCELLATA nella forma proposta** | 0,45 non esiste su disco. Le uniche fonti pubbliche danno 0,2115 / 0,2439 (DeepSeek), 0,40 (gpt-oss), 0,38 (proxy val-sample). L'unico modo di arrivare a 0,45 passa dal pavimento warmup 9/17 = 0,529, che è canale punteggio |
| **S1.a, 24 problemi su Modal, 0,5 euro** | **CANCELLATA** | La diagnosi è completa offline e l'ho conclusa sopra. Costa zero |
| **C3b come disegnata** | **CANCELLATA e sostituita** | Positivo fabbricato: un sottospazio 5-dim da differenze reali dà rho >> 1 per anisotropia locale a prescindere dalla semantica, e il braccio V-perp a 4091 dimensioni è numericamente identico al braccio isotropo. Tre bracci ne erano due |
| **`d_max` come statistica primaria** | **SOSTITUITA** | Le famiglie applicabili variano fra 2 e 5 per item (media 3,54) **[verificato ora]**. Il massimo su k variabile introduce un offset di +0,133 sd fra k=4 e k=5. Si usa la **media su un insieme a cardinalità fissa** |
| **`format`, `latex_delims`, `question_first` come sorgenti di C1** | **RIMOSSE** | No-op su 58,5% / 21,3% / 66,5%. `d_f = 0` da perturbazione nulla non è `d_f = 0` da modello stabile, e l'indicatore di no-op di `latex_delims` predice già l'etichetta a 0,623 con p=0,0017 |
| **Controllo RANDOMIZATION a coseno** | **RIMOSSO** | A d=13 il coseno ha sd ~0,28 sotto il nullo e P(cos>0,9) ~ 1e-6. Non scatta mai, e verrebbe riportato come "controllo superato" senza contenuto |
| **Controllo negativo di profondità come criterio di morte** | **DECLASSATO a descrittivo** | Sotto il nullo P(0,25 batte 0,80) = 0,5 e per un segnale reale non monotono è ancora 0,5. Dimezza la potenza senza aggiungere specificità |
| **M1 come baseline** | **SOSTITUITA da M0_TESTO** | §1.4 |
| **Soglia di morte 0,62** | **SOSTITUITA** | Sta sotto la mediana del null a 37 celle |
| **Soglia di successo +0,08** | **SOSTITUITA da +0,107** | SE di Hanley a AUROC 0,73 con 40/124 = **0,0493**; delta appaiato minimo per 80% di potenza a rho 0,7 = **0,1070** **[verificato ora]** |
| **`mu_l`, `sigma_l` nell'artefatto** | **SOSTITUITE** | Contraddicono l'adimensionalità richiesta da S3, sollevano eccezione su Skywork (d=3584) e Qwen (d=2560), e non esistono per gpt-oss |
| **C2 (allineamento token), CKA** | restano tagliati | erano già tagliati bene |
| **H27 come stadio** | **CHIUSA con un numero** | §1.2 |

---

## 3. Le ipotesi, con criteri di morte quantitativi

Numerazione da H25. Le probabilità a priori sono dichiarate prima dell'esecuzione, come richiesto.

### Blocco A. Zero GPU, esiti già in mano o ottenibili in mezza giornata

**H25 (CONFERMA, non ipotesi prospettica).** *Etichetta: anomalia misurata il 3 settembre 2026 prima dell'esistenza di questa pre-registrazione. Non è pre-registrata e non viene presentata come tale.*
L'artefatto ufficiale applicato alle nostre feature è anti-predittivo sui suoi stessi 137 problemi di addestramento (AUROC 0,361, Spearman in-sample -0,125, accuratezza della regola spedita 0,348 contro maggioranza 0,756). **Causa identificata:** `generation_frac_selected_is_top1` è degenere a 1,0000 su 164/164 nella nostra estrazione, mentre l'artefatto ha 4915 split su quella feature con soglie tutte in [0,96836, 0,99348] e importanza 0,2100. Ogni albero instrada il nostro campione deterministicamente dalla stessa parte.
**Parte prospettica genuina:** far girare i due percorsi di calcolo di `is_top1` (`argmax` sugli score grezzi contro `topk` su `log_softmax(scores.float())`) sullo stesso tensore sintetico e misurare il disaccordo sui quasi-pareggi bf16. **Criterio di morte:** se il disaccordo simulato è < 0,5%, la spiegazione bf16 non basta e va cercata altrove prima di correggere `aimo_gpu.py`. **Costo: 0.**

**H26 (BENCHMARK, sostituisce la vecchia H26 sulla prior).** Nel benchmark ufficiale l'etichetta `model_is_robust` è esattamente recuperabile dal tag di aggregazione `permutation_type`, e un contatore di bookkeeping la predice.
**Predizioni pre-registrate:** (a) la regola tiene su 223/223 righe etichettate in tre file [già verificata]; (b) `AUROC(robust, n_permuted_predictions) >= 0,70` [misurata 0,7798]; (c) il meccanismo è l'aggregazione, verificabile perché nel file non aggregato `permutation_type` non è mai una lista [verificato].
**Criterio di morte:** se gli organizzatori documentano che il tag è generato indipendentemente dall'esito, il finding decade. **Costo: 0.** **Prior di sopravvivenza: 0,95.**

**H27 (CHIUSA).** Tetto da rumore di etichetta. **Esito: accordo fra repliche 0,9998, tetto ~1,0. Non vincola. Nessun cancello.** Si riporta il numero e si va avanti. La stima 0,744 dell'attacco statistico è respinta con la misura.

**H28 (LA BASELINE VERA, e il cancello di spedizione).** Un predittore di **solo testo**, senza generazione e senza GPU, non è inferiore al canale ufficiale.
**Feature fissate adesso, scelte a priori e non selezionate dopo aver guardato:** `n_char`, `n_frasi`, `n_dollar`, `ha_display`, `n_backslash`, `l_ultima_frase_e_domanda`. Sei, congelate.
**Predizioni:** (a) M0_TESTO OOF >= 0,65 [misurata 0,687 con la sola `n_char`]; (b) il limite superiore dell'IC95 della differenza (M1 meno M0_TESTO) su predizioni out-of-fold, con DeLong appaiato, sta **sopra -0,03** (non inferiorità a margine 0,03).
**Criterio di morte:** se M0_TESTO OOF < 0,62, non si spedisce niente di forward-free e si resta su v6. **Prior: 0,85.**

### Blocco B. Il cancello che decide se vale la pena accendere una GPU

**H29 (CANCELLO DI RIPETIBILITÀ, precede ogni misura di AUROC).** Una feature di distanza fra attivazioni è abbastanza ripetibile da poter raggiungere la soglia di successo.
**Misura:** dentro la stessa run S1, 40 item estratti **due volte** variando esattamente ciò che varia in produzione: passata A a batch 1 senza padding, passata B con gli stessi item in un batch di 8 con left padding e vicini diversi. ICC(2,1) per ogni feature candidata.
**Criterio di morte, dichiarato prima:** una feature con **ICC < 0,78** non può raggiungere AUROC osservata 0,84 nemmeno con AUROC vera 1,0 dopo attenuazione, quindi **si scarta prima di guardare l'etichetta**. Se **tutte** le feature candidate hanno ICC < 0,78, S2 non si esegue, si scrive il risultato negativo e si chiude E11.
**Costo marginale: +5% su S1.** **Prior che almeno una feature superi 0,78: 0,55.**

### Blocco C. I canali, se e solo se H29 passa

Ogni canale ha **una** statistica primaria pre-registrata, uno scalare, un grado di libertà. Holm su 5 test primari (alpha 0,010).

**H30 (C1, spostamento di rappresentazione).** `Z_C1 = media su {framing, redundancy, pad_short, pad_long} residualizzata` = `mean(d_framing, d_redundancy) - mean(d_pad_short, d_pad_long)`, alla profondità relativa **0,80**.
Solo `framing` e `redundancy` perché sono le uniche due famiglie **sempre applicabili** (no-op su 0/164 **[verificato ora]**). Cardinalità fissa a 2, quindi nessun artefatto di max-su-k variabile.
`pad_short` e `pad_long` sono appaiati **in token per item**, non a lunghezza fissa, perché `redundancy` aggiunge una quantità proporzionale all'item (+106,5 caratteri mediani, +100%).
**Criterio di morte:** AUROC OOF < **p95 del null permutazionale calcolato sul numero esatto di celle prodotte dal codice** (§4.3), oppure delta sopra M0_TESTO < 0,05. **Prior: 0,12.**

**H31 (C1b, coerenza direzionale).** `Z_C1b = cos(h(x_framing) - h(x), h(x_redundancy) - h(x))` a profondità 0,80. Adimensionale, di second'ordine, costo zero sugli stessi tensori. **Stessi criteri di morte.** **Prior: 0,10.**

**H32 (C3, iniezione di rumore).** `Z_C3 = KL(p_orig || p_noise)` medio a **eps = 0,2** fisso, non `eps_to_flip`. Motivo: con R=8 e griglia a 4 punti la sd della probabilità di flip è 0,177 a p=0,5, cioè un passo di griglia intero, e la statistica primaria di un canale non può avere quella risoluzione. La griglia diventa **8 punti logaritmici** e `eps_to_flip` scende a secondaria, stimata per interpolazione sulla curva.
**Degenerazione dichiarata prima:** si misura in due posizioni (ultimo token di contenuto, ultimo token di template). **Se l'entropia mediana in entrambe è sotto 0,1 nat, C3 si dichiara non applicabile e si passa oltre.** **Prior: 0,15.**

**H33 (C3b CORRETTA, il controllo geometrico).** Il controllo isotropo è cancellato. Il controllo corretto è: sottospazio `V` generato dalle differenze di attivazione **dello stesso item** contro sottospazio `W` generato da differenze di attivazione **di altri problemi**, stessa dimensione, stessa norma di iniezione.
`rho = KL_V / KL_W` a eps 0,2. **Criterio:** se l'IC95 bootstrap appaiato di `rho` **contiene 1**, le direzioni testuali specifiche dell'item non sono speciali, e C1 sta misurando la perturbazione, non la fragilità. **Questo risultato è informativo in entrambi i versi e non è predeterminato**, a differenza della versione originale. **Prior che rho sia distinguibile da 1: 0,40.**

**H34 (C4b, il controllo che può azzerare tutto).** Statistica di angolo della traiettoria fra layer sul **solo problema originale**. **Criterio:** se M0_TESTO + C4b eguaglia M0_TESTO + C1 + C3 entro 0,05, **l'instabilità non esiste come canale** e quello che abbiamo trovato è una sonda di stato statico. Va potuto dichiarare. **Prior che C4b eguagli i canali di instabilità: 0,55.**

### Blocco D. Trasferimento

**H35 (STABILITÀ DISTRIBUZIONALE, non accuratezza).** Sui tre modelli senza etichette le distribuzioni delle distanze adimensionali non degenerano e restano nel range osservato su DeepSeek.
**Criterio di morte:** se su un modello la sd della feature primaria è sotto il 20% di quella su DeepSeek, o se oltre il 10% dei valori è non finito, il canale non è applicabile a quel modello e quel modello cade sul prior congelato. **Prior: 0,50.**

---

## 4. Protocollo di validazione

### 4.1 Il bersaglio, dichiarato prima
Analisi primaria sui **164 completi**. La stratificazione su `n_permuted_predictions` **non** è l'analisi primaria: m è simultaneamente un discendente dell'etichetta (aggregazione) e un modulatore del rumore (congiunzione su m prove), i due effetti hanno segno opposto, e condizionare su un collider parziale non è riparazione, è un'altra distorsione. Prova: aggiungendo `log m` al modello il coefficiente di `log n_char` **sale** da +1,02 a +1,93 **[verificato ora]**, che è amplificazione da collider, non controllo di confondimento.
**Obbligo:** ogni AUROC riportata porta accanto, obbligatoriamente, (a) la stessa AUROC stratificata su quartili di m come analisi di sensibilità, e (b) l'AUROC dei soli contatori di bookkeeping (**0,8619 OOF**) come riferimento superiore, con la frase che quel riferimento **non è disponibile a inferenza**.

### 4.2 CV
Una riga per problema, le famiglie sono colonne. Mai espandere a 164x5. CV raggruppata per `problem_id`, 5 fold, 20 ripetizioni. Sui nostri dati grouped ≡ stratified perché i 164 `problem_id` sono unici **[verificato]**, ma il raggruppamento resta scritto perché diventa vincolante se il set cresce.

### 4.3 Iperparametri senza contaminazione, e il null sulla griglia REALE
1. **Nessuna scansione di layer.** Tre bin di profondità relativa fissati adesso: **0,25** (descrittivo), **0,55** (descrittivo), **0,80** (primario). La profondità primaria è fissata a priori su SEP e sul paper della correttezza del codice (layer 30/36 = 0,83).
2. **Nessun tuning.** Logistica L2 con **C = 0,05 fissato adesso**. Nessuna griglia, nessuna CV interna, nessuna selezione.
3. **Tutto ciò che è stimato dai dati sta dentro il fold.** Nessuna statistica calcolata sui 164 completi entra in una feature valutata in CV, nemmeno se non usa le etichette. Questo include la normalizzazione e qualunque rimozione di dimensioni outlier. La versione stimata su tutto il training si salva nell'artefatto **solo per l'inferenza in gara**, mai per la validazione offline.
4. **Le celle si contano PRIMA di scrivere il codice di analisi**, e la soglia di morte è il **p95 empirico del null permutazionale su quel numero esatto di celle**. Riferimenti misurati **[verificato ora]**, n=164, 40 positivi: k=1 p95 0,584; k=12 p95 **0,640**; k=37 p95 **0,657**; k=120 p95 **0,670**; k=185 p95 **0,676**. Con la griglia ridotta di questo piano (5 statistiche primarie x 3 profondità x 2 varianti = **30 celle**) la soglia di morte onesta è **0,655**, non 0,62.
5. **Esplorativo:** tutto il resto. Mai in un claim, mai in una submission.

### 4.4 Soglie di vita e morte, congelate
- **Potenza dichiarata prima di vedere i numeri:** SE di Hanley a AUROC 0,73 con 40 positivi e 124 negativi = **0,0493**. Delta appaiato minimo per **80% di potenza** a rho 0,7 = **0,1070**. Potenza al vecchio bersaglio +0,08: **~50% senza correzione, ~32% sotto Holm**. Potenza per un guadagno realistico di 0,02-0,03: **3-4%**. n necessario per 80% di potenza a delta 0,03: **circa 3100 problemi**. Ne abbiamo 164.
- **Morto:** AUROC OOF < 0,655 (p95 del null a 30 celle).
- **Vivo ma inutile:** delta sopra M0_TESTO < 0,05. Non entra nella submission. Nota che l'attacco ha misurato che aggiungendo **una** feature di puro rumore e prendendo il migliore di 4, il p95 del delta è +0,026 e il massimo su 300 prove +0,090: la banda sotto 0,05 è interamente rumore.
- **Vivo e utile:** delta >= **0,107** con DeLong appaiato sulle predizioni out-of-fold, p < 0,010 dopo Holm su 5 canali. **La barra è 0,107, non 0,08**, e se rho osservato fra i punteggi è più basso di 0,7 la barra sale (0,139 a rho 0,4). La barra si ricalcola sul rho **osservato** e si dichiara.
- **Winner's curse:** ogni delta che supera la soglia va riportato con la correzione per troncamento. E[delta osservato | superato] = 0,099 quando il vero è 0,03, e 0,103 quando il vero è 0,05: il numero pubblicato sarebbe per costruzione circa il doppio del vero. **Mitigazione scelta:** si congelano **40 dei 164** item come set di conferma mai toccato, accettando che l'analisi primaria giri su 124 con 30 positivi. Il numero del writeup è quello del set di conferma.

---

## 5. Gli stadi, con le condizioni di passaggio

### S0 — zero GPU, blocca tutto, mezza giornata. Si esegue sempre.
**PREREG_E11.md va congelato PRIMA di S0, non prima di Modal.** S0 contiene ipotesi e una decisione sulla prevalenza: scriverlo dopo aver visto i numeri di S0 sarebbe pre-registrazione retrodatata.

- **S0.1** H25 offline: simulazione del disaccordo `argmax` contro `topk(log_softmax(float))`, poi correzione di `gpu/aimo_gpu.py` per chiamare direttamente `uncertainty_profile.extraction`. Nessuna nostra reimplementazione.
- **S0.2** H26: consolidamento del finding di benchmark, tre file, 223 righe, più il file non aggregato come prova del meccanismo.
- **S0.3** H28: M0_TESTO con le sei feature congelate, DeLong appaiato di non inferiorità contro M1.
- **S0.4** Prova a secco dell'intero apparato statistico (CV raggruppata, null permutazionale sulla griglia contata, DeLong, Holm, correzione per troncamento) su feature di rumore iniettate. Deve riprodurre il p95 = 0,584 a k=1 e 0,655 a k=30. Se non lo riproduce, il bug è nella pipeline e si scopre prima di spendere.
- **S0.5** **CANCELLATA.** Nessuna submission diagnostica, per nessun motivo, con nessuna etichetta di trasparenza.
- **S0.6** Smoke test dentro `Dockerfile.competition`, su CPU, senza pesi: `np.load` dell'npz con `allow_pickle=False`, import di tutti i moduli spediti, `are_robust` su 3 stringhe finte, assert che il ritorno sia `list` e `all(type(x) is bool)`. **Oggi non esiste ed è l'unica prova reale che l'artefatto è caricabile.**
- **S0.7** Verifica che nessun chat template inietti la data: rendering di `apply_chat_template` per i cinque tokenizer con due date diverse, e diff. Olmo-3 e il formato harmony di gpt-oss lo fanno. Se lo fanno, si congela o si normalizza la data.

**Cancello per S1:** S0.4 riproduce il null **e** H29 è implementata nell'harness. Altrimenti non si accende niente.

### S1 — una sola run Modal su DeepSeek. ~25 min A100-80GB, **~1,0 euro**.
Tutto in un singolo caricamento del modello.
- **S1.a CANCELLATA** (diagnosi già chiusa a zero euro).
- **S1.b** Estrazione attivazioni: ultimo token di prompt, tutti i layer, per originale + `framing` + `redundancy` + `pad_short` + `pad_long` = **5 x 164 = 820 forward**. Batch 1, `use_cache=False`, `attn_implementation='eager'` congelato, `dtype=torch.bfloat16` congelato, `device_map` su **una sola scheda**, `model.model(...)` invece di `model(...)`. Salvataggio grezzo in fp16 (~360 MB) per rifare ogni analisi senza ripagare la GPU.
- **S1.c** H29: ri-estrazione di 40 item a batch 8 con left padding e vicini diversi. **+5%.**
- **S1.d** H32/H33: 8 estrazioni x 8 eps x 2 condizioni (V dell'item, W di altri problemi) x 164, rumore nella dimensione di batch.

**Cancello per S2:** H29 passa (almeno una feature con ICC >= 0,78). Altrimenti si scrive il negativo e si chiude.

### S2 — analisi, zero GPU.
Ordine dei test primari: **H32, H30, H34, H31, H33**, ciascuno con le soglie di §4.4, Holm su 5.

**Cancello per S3:** almeno un canale con delta >= 0,107 sopra M0_TESTO, confermato sul set di conferma da 40 item mai toccato. **Se nessuno passa, S3 e S4 non si eseguono.**

### S3 — tre modelli senza etichette. ~25 min, **~0,9 euro**. Solo se S2 passa.
Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think. Solo H35 (stabilità distribuzionale). `gpt-oss-120b` non è estraibile offline (223,5 GiB in bf16), quindi il canale non gli si applica e cade sul prior congelato: sono ~4 item su 21 in Main, e va detto.

### S4 — artefatto e **una sola** submission. Solo se S3 conferma H35.
**Gate di non inferiorità dichiarato prima:** la variante forward-only si spedisce solo se il suo AUROC out-of-fold non è inferiore a M0_TESTO di più di 0,03 (DeLong appaiato, limite superiore dell'IC95 della differenza sopra -0,03). **Il guadagno di runtime non è mai da solo una ragione sufficiente per cambiare ciò che si spedisce.** La frase "indipendentemente dall'AUROC" del piano originale è cancellata.

### Cosa NON si esegue mai
- Generazione perturbata (6 passate x 838 s sfondano i 3600).
- Espansione non deduplicata con `augmented-sample-math*.json` (il numero di tipi elencati per riga predice l'etichetta a AUROC 0,92, è artefatto di costruzione, non disponibile a inferenza, ed è informazione sulle loro famiglie).
- Selezione del layer sul test set.
- Qualunque submission diagnostica.

---

## 6. Costo GPU per stadio

| stadio | costo | condizione |
|---|---|---|
| S0 (tutto, incluse H25/H26/H27/H28 e lo smoke test) | **0 euro** | sempre |
| S1 (DeepSeek: attivazioni + ICC + rumore) | **~1,0 euro** | S0.4 riprodotto |
| S2 | 0 euro | H29 passa |
| S3 (tre modelli) | **~0,9 euro** | almeno un canale a delta >= 0,107 |
| margine per una ripetizione | ~1,0 euro | |
| **impegnato realisticamente** | **~1,0 euro**, tetto 2,9 | |

Restano oltre 7 euro dei 10. Il vincolo che morde non è la GPU, è n=164 con 40 positivi.

---

## 7. Validato, surrogato, euristico

**VALIDATO con le 164 etichette vere (solo DeepSeek):**
- che la feature sia calcolabile, non degenere e **ripetibile** (H29, ICC, il numero che oggi non esiste);
- la mappa feature → `robust` in CV raggruppata, con null permutazionale sulla griglia contata, controllo outlier dentro il fold, e set di conferma da 40 item;
- l'incremento sopra **M0_TESTO** con DeLong appaiato, con la barra a 0,107 e la correzione per troncamento;
- **leave-one-family-out declassato.** L'attacco ha ragione: con due sole famiglie sempre applicabili che in E9 concordano nel 92-93% dei casi, il LOFO passa per costruzione e non ha potenza. Si riporta come descrittivo, non come "un test vero".

**CONTROLLO NEGATIVO CHE DEVE FALLIRE (segno invertito rispetto al piano originale):**
- `robust_ours`, l'etichetta E9 generabile su tutti e cinque i modelli, ha kappa **0,02** con l'etichetta vera. Quindi **un trasferimento riuscito del coefficiente su `robust_ours` è evidenza CONTRO il canale**, non a favore: significherebbe che il coefficiente cattura la perturbazione e non la robustezza. Il paragrafo 7 del piano originale lo presentava come "valida il trasferimento della mappa": riclassificato. **Non è mai criterio di spedizione.**

**EURISTICO, e resterà tale:**
- che il coefficiente stimato su DeepSeek valga per Qwen3.5-4B, Skywork, Olmo-Think. **Zero item etichettati pubblici per tutti e tre.** La dimensione campionaria efficace dell'ipotesi di trasferimento è **1 modello**.
- la normalizzazione auto-contenuta nell'item come allineamento del confine di decisione fra modelli. Ragionevole, non verificabile.
- il prior per modello e quindi la soglia. Congelati da `prior_per_modello.json`, mai stimati sul lotto.
- `gpt-oss` ha 10 item etichettati. SE di un'accuratezza su n=10 è ~0,16. **Potenza nulla.** È aneddoto, non validazione.

**La frase per il writeup:** il canale è self-supervised nel calcolo e supervisionato nella calibrazione. Applicarlo ai tre modelli senza etichette è una **estensione dichiarata**, non un risultato validato.

---

## 8. Conformità (sezione da trascrivere verbatim)

**8.1 Cosa è lecito, con le parole del regolamento.**
(a) "teams can use weights, activations, token probabilities, or any other model internal signals as inputs for their methods". E11 calcola attivazioni e distanze sugli item di valutazione **come input al metodo, per item, dentro il container, e le scarta**. Non ricostruisce, non registra, non trasmette alcun dato di valutazione.
(b) Le quattro famiglie usate a runtime (`framing`, `redundancy`, `pad_short`, `pad_long`) sono meccaniche e deterministiche, nessun LLM le genera, e **nessuna coincide** con le famiglie del validation set finale (`distract`, `domain`, `rename`, `rephrase`, `typos`, `expert_no_solution`, `expert_perturbations`, `paraphrase`).
(c) Le uniche etichette usate per l'addestramento sono quelle pubblicate dagli organizzatori, le stesse su cui è addestrato il loro artefatto: non sono extra labeled data, sono le loro. Le etichette generate da noi usano solo famiglie diverse dalle loro, ciò che il regolamento permette verbatim.
(d) Resta vietata l'espansione non deduplicata con `augmented-sample-math*.json`.

**8.2 Clausola di logging (estesa alle categorie nuove).**
"Nei log del container si stampano soltanto: `model_id`, conteggi di problemi, tempi, e statistiche aggregate per lotto (minimo, massimo, mediana). Non vengono mai stampati, salvati su disco, o restituiti: il testo dei problemi di valutazione, il testo perturbato degli stessi, vettori di attivazione o loro proiezioni, distanze o punteggi per singolo item, e qualunque quantità da cui l'ordinamento degli item del lotto sia ricostruibile. Le attivazioni grezze salvate in fp16 riguardano esclusivamente i 164 problemi PUBBLICI e non toccano mai il set di valutazione."

**8.3 Divieto di transduzione, con controllo eseguibile.**
"Ogni costante di decisione (normalizzazione, bin di profondità relativa, prior per modello, soglia) è stimata offline su dati pubblici, congelata nell'artefatto spedito, e non viene mai ricalcolata, aggiornata o riscalata sugli item di valutazione. La predizione dell'item i dipende solo dall'item i: sono vietati quantili calcolati sul lotto, regole top-k o top-p sul lotto, e stimatori di prevalenza (Saerens/EM) applicati ai punteggi del lotto. **Controllo eseguibile e obbligatorio prima della spedizione: permutare o dimezzare la composizione del lotto non deve cambiare nemmeno una predizione.**" Motivazione: PREREG_E3.md:49 (regola metà-lotto, -25,4) e PREREG_E8.md:56-58 (top-p nel lotto, -2,8 e -10,1) hanno già ucciso questa famiglia sui nostri dati.

**8.4 Prevalenza congelata.**
"`pi_test` per modello è preso da `soluzione/prior_per_modello.json`, file pubblico derivato dai dati pubblicati dagli organizzatori. Nessun numero derivato da un punteggio di classifica, del warmup o della fase principale, entra in questo calcolo. **Il valore 0,45 non compare in nessun file pubblico su disco e non viene usato.** Se questa prevalenza si rivelasse sbagliata dopo la chiusura, non viene corretta: si riporta l'errore." Controllo decisivo di PREREG_V4.md:64-70: togliere ogni numero di origine punteggio e verificare che le predizioni non cambino di un solo item.

**8.5 Assenza di stato.**
"Il predittore è senza stato. Nulla di calcolato sugli item di valutazione viene persistito su disco, accumulato fra lotti, o riportato da una submission alla successiva. L'unico output verso l'esterno sono le predizioni booleane e i log aggregati sopra definiti. Le attivazioni sono usate come input per item e scartate a fine batch."

**8.6 Regola di submission nel nuovo regime.**
"Fase principale, `submission_rule` `Force_Latest_Multiple` (OSSERVATO_MAIN_2026-09-02.md:46-51): non esistono più submission private, l'ultima spedita è quella pubblica. Quindi: **una sola submission per traccia in tutta E11**, spedita per ultima, solo se il gate di §4.4 e quello di non inferiorità di S4 sono passati, e il suo punteggio viene riportato, non ottimizzato. Nessun secondo invio per vedere. Se il punteggio è peggiore di quello di v6, si riporta comunque."

**8.7 Separazione da E9-Stadio 2.**
"E11 non riapre H19/H20. Quelle riguardavano l'etichetta binaria costruita dalla risposta finale sotto le nostre cinque famiglie, e sono morte (PREREG_E9.md:90). E11 misura una quantità diversa calcolata dallo stesso stimolo: lo spostamento continuo delle rappresentazioni interne, che esiste anche quando la risposta finale coincide. Il prior dichiarato prima dell'esecuzione è che C1 superi il criterio di morte con probabilità **0,12**, proprio perché eredita la scommessa sulle famiglie che E9 ha perso. **Se muore, non si cerca una sesta famiglia che passi.**"

**8.8 Divergenza d'ambiente (nello stile di PREREG_E6_DEVIAZIONE.md:68-78).**

| | Modal (estrazione) | container (gara) |
|---|---|---|
| GPU | A100-80GB, sm_80 | 8x RTX PRO 6000 Blackwell, sm_120 |
| torch | 2.13.0+cu130 | 2.12.1+cu130 |
| transformers | 5.13.0 | 5.13.0 |
| dtype | bf16 congelato | bf16 congelato |
| attn | `eager` congelato | `eager` congelato |
| device_map | scheda singola | scheda singola |

Questa è la classe di errore che ha già prodotto AUROC 0,361 in H25: una singola divergenza strutturale nell'estrazione non degrada un canale, lo rende **anti-predittivo**. Non è rumore centrato, è un bias sui coefficienti spediti.

---

## 9. Ingegneria di produzione (si fa comunque, indipendentemente da E11)

Queste modifiche proteggono `coverage = 1` e `invalid = 0` e vanno fatte anche se E11 non parte.

1. **Sottoprocesso con deadline.** Tutto il lavoro sul modello dentro `multiprocessing` con contesto `spawn`, `p.join(deadline)` seguito da `p.kill()`. Il padre tiene la lista prior pronta e la restituisce alla scadenza. Questo, e solo questo, garantisce insieme: coverage 1 anche in timeout, immunità all'avvelenamento del contesto CUDA che oggi propaga un crash del secondo modello su tutti i successivi, e recupero della memoria senza dipendere da `release_model()`. **È la modifica con il rapporto valore/costo più alto dell'intera revisione.**
2. **Tabella dei costi per modello, e budget del caricamento.** `COSTO_STIMATO_PER_PROBLEMA_S = 90.0` è una costante unica contro 202 s/item misurati su DeepSeek. Caricamento ~0,27 GB/s dal mount: Qwen3.5-4B ~30 s, Skywork ~55 s, Olmo ~54 s, DeepSeek ~60 s, gpt-oss 218 GiB → 218-810 s, **oggi non budgetato affatto**. Quota riservata per modello (3600 / n_modelli_attesi), non consumo in ordine di arrivo, che non controlliamo. `StoppingCriteria` a scadenza dentro `generate()`, perché oggi il controllo avviene prima e `generate` non è interrompibile.
3. **Artefatto come `.npz` di soli array numerici**, `np.load(allow_pickle=False)`, punteggio ricalcolato a mano come `sigmoid(x @ w + b)`. Nessun oggetto sklearn, nessun DataFrame, nessun tensore torch. Elimina in blocco i rischi numpy 1.x → 2.5.1, joblib 1.5.0 → 1.5.3, pandas 2.3.3 → **3.0.3 (salto MAJOR)**, python 3.11 → 3.12. Rimuovere anche `pd.DataFrame.from_records` da `solution.py`. Cinque righe di codice.
4. **Mappa profondità → indice congelata per iscritto.** `hidden_states` ha L+1 elementi, indice 0 = embedding. Formula: `idx = 1 + int(floor(r * (L-1)))`, niente `round()`. Valori risultanti scritti nel prereg per i cinque modelli. In transformers 5.13.0 `hidden_states[L]` è post-norm finale, scala diversa: il livello 1,00 non entra mai nei test primari.
5. **Normalizzazione auto-contenuta nell'item**, non `mu_l`/`sigma_l` da artefatto: si centra sul vettore originale e si scala per la mediana delle norme delle differenze dell'item stesso. Nessun vettore di dimensione d nell'artefatto, niente da trasferire, e il canale funziona anche dove non c'è estrazione offline.
6. **Asserzioni difensive.** `assert np.isfinite` su tutte le attivazioni e feature derivate prima del predittore, con fallback esplicito al prior e log del conteggio. E assert che il numero di predizioni diverse dal prior sia `> 0` e `< n`: una submission in cui il metodo ha prodotto tutte le stesse predizioni del prior è indistinguibile dal prior e va segnalata a log, non spedita in silenzio. Senza questa, un NaN produce 21 `False` perfettamente validi e nessuno se ne accorge.
7. **Stessa composizione di batch** nell'estrazione offline e in gara. Altrimenti si ricrea esattamente il difetto di H25.

---

## 10. La mossa che vale più di tutto E11, e costa zero

Dalla §1.4: `n_char` da sola vale **0,687 OOF / 0,704 diretta**, contro **0,696** del canale ufficiale completo, con delta indistinguibile da zero. Il canale ufficiale richiede 202 s/item di generazione; `n_char` richiede microsecondi.

**Conseguenza operativa:** un predittore di solo testo, senza generazione, elimina l'intero rischio di budget. Il metodo attuale in Main è stimato a ~2600 s su 3600 nominali (margine 28%), e il README dichiara esplicitamente che l'ambiente di calcolo è condiviso: un fattore 1,5 di contesa porta a ~3900 s, e il fallimento non è graduale perché `predictions.jsonl` viene scritto solo alla fine del ciclo su tutti i modelli. Un timeout sul quinto modello distrugge anche i primi quattro.

Il predittore di solo testo gira in **millisecondi per l'intera run**, ha margine 3600/3600, e **il tie-break della gara è il runtime medio su tre prove**. È un premio concreto, ottenuto con un test di non inferiorità dichiarato prima (H28), non con taratura sul canale del punteggio. È conforme, è difendibile, e non costa un centesimo di GPU.

**Avvertenza obbligatoria nel writeup:** il segno è "problemi più lunghi sono più robusti", plausibile meccanicamente (più ridondanza informativa, quindi una perturbazione locale ha meno probabilità di cambiare la risposta), ma misurato su **un solo modello** e contro le famiglie degli organizzatori fra cui c'è `expert_no_solution`, dove l'effetto potrebbe invertirsi. Non è un risultato di robustezza, è una regolarità di superficie con una spiegazione plausibile.

---

## 11. Probabilità di successo, onesta

**Definizione di successo per il rango:** un canale di attivazioni che batte M0_TESTO di delta >= 0,107 confermato sul set da 40 item.

| passo | probabilità | motivo |
|---|---|---|
| H29 passa (ICC >= 0,78 su almeno una feature) | 0,55 | un forward singolo in bf16 ha errore relativo ~4e-3 per elemento che composto su 36 layer dà 1-cos di ordine 1e-5..1e-4 contro un segnale atteso 0,05-0,5, rapporto 1000:1. La minaccia non è il rumore, è lo scostamento sistematico Modal/container, che l'ICC a batch 8 contro batch 1 misura solo in parte |
| Almeno un canale supera 0,655 (non morto) | 0,40 | con la griglia ridotta a 30 celle e le famiglie ripulite |
| Almeno un canale supera M0_TESTO di >= 0,107 | **0,06** | richiede AUROC osservata ~0,84 contro una baseline di solo testo a 0,70-0,73. La potenza a un delta vero di 0,03 è 3-4%; il piano ha 80% di potenza solo se il delta vero è 0,131 |
| **Successo per il rango, congiunto** | **~0,04** | |

**Perché così basso, in una frase:** stiamo proponendo che la distanza coseno fra attivazioni sotto perturbazioni meccaniche di forma predica "tutte le m perturbazioni semantiche degli organizzatori hanno preservato la risposta" meglio di quanto lo faccia la lunghezza del testo, e le stesse cinque famiglie hanno già dato kappa 0,02 (E9) e kappa -0,10 (E10) contro quella stessa etichetta. Un segnale continuo estratto dallo stesso stimolo non è evidenza indipendente, è la stessa sonda con un lettore più fine.

**Definizione di successo per il writeup:** un contributo verificabile e riproducibile. **Probabilità: ~0,90.** Il materiale è già in mano e non richiede GPU:
1. `model_is_robust` è esattamente recuperabile dal tag di aggregazione `permutation_type` in **223 righe su 223** in tre file indipendenti, e il meccanismo è provato dal file non aggregato dove il tag non è mai una lista;
2. un contatore di bookkeeping predice l'etichetta a **0,78-0,86 OOF**, e non è disponibile a inferenza;
3. l'artefatto ufficiale spedito è **anti-predittivo sui suoi stessi 137 problemi di addestramento**, e la causa è una singola feature (`frac_selected_is_top1`, 17,1% degli split, importanza 0,21) fuori dal supporto di addestramento per una differenza di implementazione nel calcolo di `is_top1`;
4. la baseline dichiarata degli organizzatori (probe meccanicistico 58,4%) sta **sotto la classe di maggioranza** (75,6%), e il canale ufficiale completo non è distinguibile dalla lunghezza del testo.

**Definizione di successo per l'integrità:** zero submission diagnostiche, zero numeri derivati dal canale del punteggio, la frase di WRITEUP.md:155 ("the floor probe to remain our only information-yielding submission") che resta vera al report di compliance del 15 novembre. **Probabilità: 1,0, se e solo se S0.5 resta cancellata.** Se S0.5 viene eseguita, quella frase va ritrattata e il valore del writeup da 5.000 dollari, che si costruisce esattamente sul rifiuto di questa mossa, crolla.

---

## 12. Le tre cose da guardare in faccia

**Prima.** La baseline da battere non è 0,50, non è il 58,4% del probe ufficiale, e non è 0,691. È **0,70 di AUROC ottenuta contando i caratteri del problema, senza modello, senza GPU, senza tokenizer** **[verificato ora]**. Ogni euro speso su Modal per battere 0,691 stava inseguendo un numero già pareggiato gratis.

**Seconda.** Con 40 positivi il minimo miglioramento dimostrabile è **+0,107 di AUROC**, che vale circa **1,3 item su 21**, contro un pavimento di rumore dello strumento misurato a **sd 1,0 item su 17** (11, 9, 11, 11 con bundle byte-identico) e un IC95 binomiale su 21 item di semiampiezza **4,5 item**. Il criterio di successo sta tre volte e mezzo sotto il pavimento di rumore. **Nella Main il capolista sta 1 item sopra di noi: quella differenza è interamente dentro il rumore del nostro stesso strumento. Il rango in Main non è un bersaglio raggiungibile in modo misurabile, ed è meglio dirlo prima che dopo.**

**Terza, e la più scomoda per chi ha scritto il piano originale.** La cosa più preziosa emersa non è l'idea sulle attivazioni, e non è nemmeno il presunto mismatch di spazio delle feature, che ho verificato non esistere (13 mediane su 14 coincidono entro 1-6% con quelle dell'imputer dentro il loro pickle). È che **il benchmark ufficiale ha un difetto di costruzione documentabile in mezza giornata senza GPU**, che l'artefatto che gli organizzatori distribuiscono è rotto in modo dimostrabile, e che la loro nota 4 dichiara già per iscritto che il segnale sul solo problema originale è un'assunzione non verificata. Se il bersaglio è il writeup da 5k, **quello è il writeup**, e il programma sulle attivazioni è al massimo la sua sezione negativa.

---

**Prossimo artefatto da produrre:** `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\PREREG_E11.md`, con H25-H35, le soglie di §4.4, la potenza dichiarata, e le clausole di conformità §8 congelate, **prima di S0 e non solo prima di Modal**.

**File di riferimento (percorsi assoluti):**
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\gpu\out\feature_greedy.parquet` (164 righe, 40 robust; **tre** colonne degeneri: `frac_top1` ≡ 1, `frac_bassa` ≡ 0, `max_logprob` ≡ 0)
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\gpu\out\_base164.parquet` (testo dei problemi, base di M0_TESTO)
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\gpu\out\_diag_perturb.parquet` (applicabilità e no-op delle 5 famiglie)
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\soluzione\uncertainty_artifacts\deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib` (chiavi: `estimator` Pipeline imputer+regressor, `decision_threshold` 0,5759658816, `training_provenance` 141 righe / 137 gruppi / target `absolute_accuracy_decay`, `library_versions` python 3.12.0 numpy 2.5.1 pandas 3.0.3 sklearn 1.8.0 torch 2.12.1+cu130 transformers 5.13.0)
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\gpu\aimo_gpu.py` (da correggere in S0.1: sostituire l'estrazione locale con `uncertainty_profile.extraction`)
- `C:\Users\Utente\MODELLIPERSONALIZZATI\DC2026\AIMO\soluzione\solution.py` (da correggere: sottoprocesso con deadline, tabella costi per modello, rimozione di `pd.DataFrame.from_records`)
- Script di verifica di questa revisione: `C:\Users\Utente\AppData\Local\Temp\claude\C--Users-Utente-MODELLIPERSONALIZZATI-DC2026\afb9b121-13de-4913-8c95-f4be3c1f36db\scratchpad\v1.py` ... `v14.py`. I decisivi: `v6.py` (null di procedura), `v11.py` (test permutazionale stratificato con merge corretto per modello), `v9.py` (split e mediane dentro l'artefatto ufficiale), `v14.py` (IC nello strato, null max-of-k).

## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Anomalia A100 (H25 e difetto della feature).** La feature generation_frac_selected_is_top1 costante a 1,0 veniva dalla nostra reimplementazione dell'estrazione, non dall'artefatto ufficiale: l'estrazione ufficiale non restituisce mai 1,0 (report, 2.3b e Appendice D, voce 5). L'anomalia dell'AUROC resta riportata (report, 8.2, con generation_mean_prob riempita con la media della probabilita' top-1: 0,36 nel verso della regola), ma non e' un difetto dell'artefatto. La parte prospettica di H25 (simulazione argmax contro top-k, con criterio di morte) non ha esito registrato.
2. **Deviazioni dell'analisi** dichiarate il 23/9 in PREREG_E11_ESITO.md e nel report (par. 4 e 13).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. L'intestazione della nota precedente («comprese le note precedenti») e' sbagliata: era la prima nota di questo file.
2. **Par. 1.1, meccanismo dell'aggregazione.** «le non robuste solo sul tipo che ha fatto danno» va letto: le non robuste sul tipo o sui tipi ad accuratezza perturbata minima, con i tipi dannosi piu' lievi scartati (16 delle 41 coppie non robuste verificabili; `vm/controlli_verifica4.py`, sez. 2; report, 2.3a).
3. **Fonte del punto 1 della nota precedente.** 0,36 nel verso della regola: `vm/e16_analisi.py`.
