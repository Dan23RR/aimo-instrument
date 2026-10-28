# Pre-registrazione E14: il nondeterminismo del pipeline ufficiale, sul metallo della gara

Scritta l'8 settembre 2026, PRIMA di eseguire. Hardware: la VM concessa dagli organizzatori,
2 × RTX PRO 6000 Blackwell 96 GB, **la stessa GPU del container di valutazione** (8 × RTX PRO
6000 condivise). Ambiente: l'immagine ufficiale del runtime via Docker, così che il pipeline
misurato sia quello che calcola la classifica, non una nostra ricostruzione.

## Perché

Quattro invii byte-identici hanno dato 11, 9, 11, 11 su 17 (PREREG_V7). Erano greedy, quindi
non è campionamento. Le feature di attivazione sul prompt si ripetono all'1% (E11, H29). Non
sappiamo dove nasce la deriva: kernel non deterministici, forma del batch, contesa fra
submission sulla stessa scheda. Michal (4/9) vuole capirlo "on the infra side" e non ha lo
strumento. Noi ora abbiamo la sua GPU per quattro giorni. È una misura, non un'ipotesi di
metodo: serve a loro, e al report.

## Disegno, fissato ora

**Pipeline.** Esattamente `uncertainty_profile.extraction.extract_generation_features` con
`art.generation_config` dell'artefatto ufficiale (greedy, 4.096 token, cache statica, batch
8) e il regressore ufficiale, cioè il percorso `_predici_con_modello` di `solution.py`. Modello:
`deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`. Problemi: i **primi 40** in ordine di `problem_id` dei
164 etichettati (lo stesso insieme di H29), fissi per tutte le condizioni. Si registra anche
l'hash del testo generato per ogni item, con `include_generated_text=True`.

**Condizioni.**

| | descrizione | ripetizioni |
|---|---|---|
| C0 | default del container: batch 8, una scheda, nessun flag | 5 |
| C1 | batch 1 | 3 |
| C2 | batch 8 + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `torch.use_deterministic_algorithms(True, warn_only=True)` | 3 |
| C3 | contesa: due processi C0 identici **sulla stessa scheda**, contemporanei | 3 coppie |
| C4 | C0 sulla seconda scheda (stesso modello di GPU, altro esemplare) | 3 |

**Misure, per condizione.** (m1) frazione di item con testo generato **byte-identico** in
tutte le ripetizioni; (m2) escursione massima dello score del regressore per item;
(m3) numero di item che attraversano la soglia 0,576 fra ripetizioni; (m4) ICC(2,1) di
ciascuna delle 14 feature; (m5) per C3, se (m1)-(m3) peggiorano rispetto a C0.

**Previsioni falsificabili, scritte ora.**
- **P-E14a.** In C0, almeno un item su 40 attraversa la soglia in 5 ripetizioni (cioè il
  container riproduce la deriva vista in gara). Se falsa: la deriva in gara dipende dalla
  condivisione, non dalla scheda, e C3 diventa la condizione chiave.
- **P-E14b.** In C2 tutti i 40 item sono byte-identici in 3 ripetizioni. Se vera, i flag
  chiudono il problema a costo di throughput (che si misura). Se falsa, il nondeterminismo non
  è nei kernel cuBLAS e va cercato altrove (attention, cache statica).
- **P-E14c.** C3 ha (m2) maggiore di C0. Se vera, la contesa nel container condiviso è una
  causa e l'isolamento è un rimedio; è l'ipotesi "grandi stabili, piccoli instabili" della
  mail del 4/9, ora testata.
- **P-E14d.** C1 (batch 1) ha (m1) più alta di C0: la forma del batch conta.

**Throughput.** Per ogni condizione si registra il tempo per item, perché la raccomandazione
agli organizzatori deve dire quanto costa il rimedio.

## Aggiunta dichiarata durante l'esecuzione, 8 settembre 2026 ore 14:00, PRIMA di eseguirla

**Osservato in C0 dopo 4 ripetizioni** (le prime 3 completate prima di questa aggiunta): rep 0,
1, 2 **byte-identiche** su 40/40 item; rep 3 **diversa su 40/40 item**, con 17 item che
cambiano lato della soglia e salti di score fino a 0,60. Non è rumore per item: è una
divergenza **per processo, tutto-o-niente**, cioè una scelta fatta all'avvio che cambia
tutte le sequenze insieme. È coerente con gli "stati discreti" visti in gara (PREREG_V7:
DeepSeek 3 stati in 4 run, range identici al millesimo dentro uno stato).

**Ipotesi di meccanismo, scritta ora:** il pipeline ufficiale usa `cache_implementation=
"static"`, che in transformers 5 fa compilare automaticamente il forward con torch.compile;
la selezione dei kernel all'avvio (autotuning di inductor e scelta dell'algoritmo cuBLAS) può
dipendere dallo stato della macchina in quel momento, e ogni processo "cade" in uno di pochi
stati numerici, ciascuno deterministico.

**Condizione aggiunta C5:** C0 con `TORCH_COMPILE_DISABLE=1` (nessuna compilazione), 3
ripetizioni, stessa scheda. **P-E14e:** in C5 le ripetizioni sono byte-identiche fra loro
**e** identiche a rep 0-2 di C0 oppure a rep 3, non a un terzo stato. Se P-E14e è vera, il
meccanismo è la compilazione e il rimedio per gli organizzatori è una riga. Se è falsa, il
meccanismo sta in cuBLAS/attention e C2 decide.

Nessuna altra modifica: C0-C4 restano come sopra.

## Costo

40 item × (5 + 3 + 3 + 6 + 3) = 800 item-esecuzioni. Nel container, 4.096 token greedy a
batch 8 costano circa 30-60 s per item amortizzati: **7-13 ore-scheda**. Si esegue per primo:
è la misura a più alto valore per ora spesa.

## Cosa si consegna

Una tabella per condizione con (m1)-(m4) e il tempo per item; una nota a Michal entro il 22
settembre, come servizio, con la raccomandazione che ne segue e il costo in throughput. Nessuna
richiesta in cambio.

## Esito

Da riempire senza riscrivere nulla di quanto sopra.

## ESITO — 8 settembre 2026. COMPLETO alle 19:17 (la tabella sotto include C3).

Eseguito sulla VM degli organizzatori, dentro la loro immagine ufficiale
(`torch 2.12.1+cu132`, `transformers 5.13.0`, `sklearn 1.8.0`), su RTX PRO 6000 Blackwell,
40 problemi, DeepSeek-R1-0528-Qwen3-8B, percorso `_predici_con_modello` invariato.

| cond | descrizione | run | item byte-identici in tutte | item che attraversano la soglia | robust_pred per run | s/item |
|---|---|---|---|---|---|---|
| **C0** | default del container | 5 | **0/40** | **24/40** | 19, 19, 19, 14, 17 | 26,0 |
| **C1** | batch 1 | 3 | 1/40 | 20/40 | 19, 18, 11 | **73,3** |
| **C2** | flag deterministici | 2 | **40/40** | **0** | 15, 15 | 27,4 |
| **C3** | contesa: due processi sulla stessa scheda | 6 | 0/40 | **32/40** | 11, 12, 19, 13, 18, 16 | 54,7 |
| **C4** | seconda scheda fisica | 3 | 0/40 | 19/40 | 19, 19, 12 | 26,0 |
| **C5** | senza compilazione | 3 | **40/40** | **0** | 12, 12, 12 | 27,8 |

**Quattordici stati distinti in ventitré esecuzioni.** C2 (3 run) e C5 (3 run) occupano un
solo stato ciascuno; tutte le altre condizioni si sparpagliano.

### 1. Il nondeterminismo è reale, ed è più grave di quanto misurato in gara

**Ventiquattro item su quaranta, il 60%, cambiano lato della soglia** fra esecuzioni
byte-identiche del pipeline ufficiale. La predizione `robust` per lo stesso identico bundle
oscilla fra 14 e 19 su 40. Escursione mediana dello score per item: **0,229** su una soglia di
0,576. In gara avevamo misurato 2 item su 17; qui, con cinque ripetizioni invece di quattro e
un insieme più grande, la portata è molto maggiore.

### 2. Il meccanismo: stati numerici discreti, scelti all'avvio del processo

Le esecuzioni non differiscono item per item: **cadono in stati**. Dentro uno stato i 40 testi
sono identici byte a byte; fra due stati diversi **zero item su quaranta coincidono**. Nove
stati distinti osservati in 16 esecuzioni. La firma è la stessa vista in gara (PREREG_V7:
DeepSeek in 3 stati, range identici al millesimo dentro ciascuno).

**La scheda fisica non c'entra:** C4, sulla seconda GPU, riproduce esattamente lo stato di
C0 rep 0-2 in due esecuzioni su tre. Lo stato non è una proprietà del silicio.

### 3. Due rimedi indipendenti lo azzerano, e costano quasi nulla

`CUBLAS_WORKSPACE_CONFIG=:4096:8` con `torch.use_deterministic_algorithms(True, warn_only=True)`
**oppure** `TORCH_COMPILE_DISABLE=1`: entrambi danno 40 item su 40 byte-identici in **tutte e
tre** le ripetizioni e zero attraversamenti della soglia. Costo: **+5,4%** e **+6,9%** di
tempo. Per confronto, batch 1 costa **+182%** e non risolve nulla, e la contesa costa **+110%**
e peggiora tutto.

### 4. Le previsioni, giudicate onestamente

- **P-E14a** (almeno un item attraversa la soglia in C0): **VERA**, 24 su 40. *Nota: dopo le
  prime tre ripetizioni, tutte identiche, avevo scritto che era falsificata. Era una
  conclusione affrettata su un campione parziale; la quarta e la quinta l'hanno smentita.
  Registrato qui perché l'errore vale quanto il risultato.*
- **P-E14b** (C2 tutta byte-identica): **VERA** su 2 ripetizioni, la terza è in coda.
- **P-E14c** (la contesa peggiora): **VERA, ed è il caso peggiore misurato.** C3 porta gli
  attraversamenti da 24 a **32 su 40** e l'escursione mediana da 0,229 a **0,317**, al doppio
  del costo per item. Il dettaglio che chiude la diagnosi: **due processi lanciati nello stesso
  istante sulla stessa scheda finiscono in stati diversi**, con 1, 9 e 0 item identici su 40
  nelle tre coppie. Lo stato non è una proprietà della macchina, del momento o della scheda:
  è una proprietà del singolo processo, decisa al suo avvio. Questo spiega anche il dato di
  gara più strano, cioè perché due modelli restavano fermi e due derivavano: dipende da quale
  processo, in quale container condiviso, ha vinto quale corsa all'avvio.
- **P-E14d** (batch 1 più stabile di batch 8): **FALSA**. C1 produce tre stati distinti in tre
  esecuzioni, il caso peggiore osservato, e costa quasi tre volte tanto.
- **P-E14e** (C5 identica a uno stato già visto di C0): **PARZIALMENTE FALSA**. La prima metà
  regge, le tre ripetizioni sono identiche fra loro. La seconda no: C5 produce uno stato
  **nuovo**, disgiunto da tutti gli altri. Anche C2 produce uno stato proprio. Quindi non
  esiste "lo stato giusto": ogni configurazione ha un suo punto fisso deterministico, e i punti
  fissi sono diversi fra loro (differenza mediana di score fra stati 0,09-0,17, e i conteggi
  robust vanno da 12 a 19). **La riproducibilità si può ottenere; l'unicità della risposta no.**
  Questo è più forte, non più debole: significa che il numero pubblicato in classifica dipende
  da una scelta di configurazione arbitraria, non solo dal caso.

### 5. Cosa se ne fa

Nota a Michal con questa tabella: una riga di configurazione, il 5% di tempo, e il rumore di
riesecuzione sparisce. Va detto anche il punto 4, che è scomodo: fissare il determinismo non
rende il punteggio "vero", lo rende ripetibile. Per il report è la sezione actionable, con la
diagnosi del meccanismo che nessuno ha pubblicato.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Nota a Michal.** La nota prevista «entro il 22 settembre» non e' stata inviata: al 23/9 e' in bozza (MAIL_michal_e14_2026-09-08.md) e non copre E16.
2. **Lettura di P-E14e.** Con la regola scritta prima («Se e' falsa, il meccanismo sta in cuBLAS/attention e C2 decide») l'esito non isola la compilazione, e C2 rende le generazioni identiche senza disattivarla. La causa degli stati non e' identificata.
3. **Misura (m4).** L'ICC(2,1) delle 14 feature non e' stata calcolata.
4. **Punti fissi.** I due punti fissi deterministici sono stati misurati su schede diverse (C2 su gpu0, C5 su gpu1): configurazione e scheda non sono separate, anche se lo stato di default si riproduce fra le schede. C2 e C5 non hanno generazioni in comune e danno la stessa predizione su 21 item su 40. `vm/e14_tabella.py`.
5. **«Il caso peggiore osservato» per C1.** Gli attraversamenti crescono con il numero di esecuzioni; in media sulle coppie di esecuzioni attraversano 15,3 item in contesa, 13,3 con batch 1, 12,7 sulla seconda scheda e 11,4 nel default. Il caso peggiore e' la contesa. `vm/e14_tabella.py`.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **«Fra due stati diversi zero item su quaranta coincidono»** e' sbagliato, gia' sulle 16 esecuzioni di allora: su 23 esecuzioni 61 delle 91 coppie di stati non condividono generazioni, 28 ne condividono una o due, una coppia di contesa ne condivide 9 e le prime due esecuzioni con batch 1 ne condividono 38 (`vm/e14_tabella.py`; report, Appendice D, voce 7).
2. **Costo di batch 1.** 72,3 secondi per item in media sulle tre esecuzioni, +178% sul default; 73,3 secondi e +182% sono la sola prima esecuzione. C2 ha tre esecuzioni, non le due della tabella.
3. **Rimedi.** Le due configurazioni hanno reso le generazioni identiche in tre esecuzioni sequenziali, ciascuna su una sola scheda, e non sono state provate in contesa ne' fra schede: «due rimedi indipendenti lo azzerano» e «il rumore di riesecuzione sparisce» non sono stabiliti (report, 8.1).
4. **Warmup.** Che i modelli del warmup restassero fermi o derivassero secondo quale processo avesse vinto una corsa all'avvio in un container condiviso e' un'inferenza: se sui worker di valutazione gli invii condividano una scheda non e' qualcosa che potessimo osservare (report, Appendice D, voci 7 e 14).
5. **Bozza della nota a Michal** (MAIL_michal_e14_2026-09-08.md, non inviata): prima dell'invio vanno corrette le stesse affermazioni («zero of forty», «remove it entirely», 73.0 s, «reproduced the first card's state exactly», «each configuration has its own fixed point»).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Avvio del processo e rumore per item.** Che lo stato sia deciso all'avvio del processo (titolo del par. 2 dell'esito e P-E14c) e' un'inferenza, tratta dalle coppie lanciate nello stesso secondo e dalla seconda scheda, non una misura (report, abstract e par. 12). «Le esecuzioni non differiscono item per item» e «Non è rumore per item: è una divergenza per processo, tutto-o-niente» (aggiunta dell'8/9) vanno letti con il punto 1 della nota precedente: stati diversi condividono alcune generazioni, e due esecuzioni con batch 1 ne condividono 38 su 40 (`vm/e14_tabella.py`; report, Appendice D, voce 7).
2. **Bozza della nota a Michal, precisazione al punto 5 della nota precedente.** Prima dell'invio vanno corrette anche le frasi elencate nella nota della quarta verifica in MAIL_michal_e14_2026-09-08.md: il meccanismo dato come compilazione, i robust predetti chiamati punteggi, la VM data ancora disponibile, l'oggetto «with a one-line fix» e «It is not per-item noise».
3. **Fonti del punto 2 della nota precedente.** 72,3 s e +178%: `vm/e14_tabella.py`; i 73,3 s della prima esecuzione di batch 1 e le tre esecuzioni di C2: log e14_all.log nell'archivio della VM.
