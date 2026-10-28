# Pre-registrazione E15: la sonda sulle attivazioni, loro e nostra, contro la verità a terra

Scritta il 14 settembre 2026, PRIMA di guardare qualunque numero che leghi attivazioni ed
etichette. L'estrazione degli stati nascosti è in corso sulla VM; nessuna analisi è stata
eseguita.

## Perché è l'ultimo canale, e perché ora

Sette canali pre-registrati sono morti con verità a terra: testo, trasferimento fra modelli,
feature ufficiali, perturbazioni nostre, traduzione, self-consistency, distanza fra attivazioni.
La sonda **supervisionata** sulle attivazioni non è mai stata eseguita: E11 misurava distanze,
non addestrava. È il metodo delle baseline ufficiali («linear probe on the final input-token
hidden state, every layer») [REDACTED: a conjecture about the methods of the leading participants].

Due fatti nuovi, verificati oggi, la rendono decidibile:
- il repository pubblico contiene **l'artefatto della sonda addestrata dagli organizzatori**
  su DeepSeek (`probe_artifact.pkl`, 54 MB, schema 2, 4 fold × 5 semi × 37 layer, strategia
  `best_layer_mean_margin`), con i punteggi di validazione salvati dentro;
- abbiamo 164 etichette vere per lo stesso modello.

## Cosa dice già l'artefatto, prima di ogni nostro calcolo

Punteggi di validazione salvati dagli organizzatori (accuratezza bilanciata, fold di controllo
`NONE`): miglior sonda per fold **0,500 / 0,738 / 0,600 / 0,500**; media per layer mai sopra
0,547 in nessun fold; miglior layer globale il 31 con **0,506**. I quattro fold di controllo
`RANDOMIZATION` stanno tutti a 0,500, come devono. Lo 0,738 è un seme in un fold su 740
candidati (37 layer × 5 semi × 4 fold): in PIANO_E11 avevamo calcolato che una selezione su
36 layer con n=164 fabbrica circa +9 punti di puro rumore. **Sui loro dati, con il loro
metodo, il segnale lineare nelle attivazioni di DeepSeek è indistinguibile dal caso.** Questo
è scritto prima dei nostri risultati e ne fissa l'aspettativa.

## Disegno

**Dati.** Stati nascosti dell'ultimo token di input, tutti i 37 layer, DeepSeek-R1-0528-Qwen3-8B,
174 problemi pubblici, due versioni: con il system prompt esatto dell'artefatto e senza.
Etichette: le 164 di `feature_greedy.parquet` (24,4% robust).

**H35, la sonda loro.** Applico l'ensemble ufficiale come fa `probe_inference.py`: per ogni
gruppo `NONE`, per ogni seme, margine = w·x + b − soglia al layer scelto dal gruppo; media dei
margini; robust ⟺ media ≥ 0. Statistica: AUROC del margine medio contro le 164 etichette,
e accuratezza bilanciata della regola. **Morta se AUROC < p95 del null permutazionale**
(0,604 con n=164, misurato in E11). Nessuna scelta libera: l'artefatto è congelato da loro.

**H36, la sonda nostra.** Regressione logistica con regolarizzazione L2 su ogni layer, CV
esterna a 5 fold stratificata (una riga per problema = raggruppata), **selezione del layer e
della regolarizzazione dentro la CV interna**, mai sul fold di test. Statistica: AUROC
out-of-fold. Null permutazionale con la stessa procedura nidificata dentro ogni permutazione
(500 permutazioni: il costo è la CV nidificata, non le attivazioni). **Morta se AUROC OOF <
p95 di quel null.** Secondarie: accuratezza bilanciata OOF, curva per layer, effetto del
system prompt.

**Correzione:** Holm su 2 primarie.

## Previsione, scritta ora

Dato l'artefatto, **la previsione è che entrambe muoiano.** Se H36 vive, la conseguenza è
limitata per costruzione: DeepSeek è 4 item su 21 nel Main e 4 su 19 nello Small, quindi anche
una sonda perfetta su DeepSeek vale al più due item in classifica. Una submission che usi
questa sonda ha senso solo se H36 vive con AUROC > 0,75; sotto, il guadagno atteso è dentro
il rumore di riesecuzione di E14.

## Cosa NON si fa

Nessuna sweep sui layer riportata come risultato: il layer si sceglie solo dentro la CV
interna. Nessun uso delle etichette generate in E13 come bersaglio: non concordano con le vere.
Nessuna sonda sugli altri quattro modelli: non hanno etichette vere, e addestrare su etichette
che sappiamo scorrelate produrrebbe un artefatto.

## Esito

Da riempire senza riscrivere nulla di quanto sopra.

## ESITO, 14 settembre 2026. Contro la previsione: ENTRAMBE SOPRAVVIVONO. Con due riserve.

Estrazione: 174 problemi, 37 layer, hidden 4096, un forward ciascuno, 47 token medi, 16
secondi in tutto. Null permutazionale su n=164: p95 = 0,6022.

| | AUROC | esito |
|---|---|---|
| **H35** sonda loro, ensemble ufficiale | **0,7391** | sopravvive |
| **H36** sonda nostra, CV nidificata, con system prompt | **0,7097** OOF | sopravvive (null nidificato p95 0,5744) |
| H36 senza system prompt | 0,6183 OOF | sopravvive al margine (p95 0,5888) |

H35: accuratezza bilanciata della regola `margine ≥ 0` **0,500**, predice robust **0%**: la
soglia dell'ensemble è tutta negativa sui nostri dati, quindi la loro sonda ordina ma non
classifica. H36: accuratezza bilanciata OOF **0,6456**; layer scelti dalla CV interna per
fold: 4, 4, 20, 20, 7, tutti con la regolarizzazione più forte.

**Riserva 1, su H35.** L'artefatto ufficiale è stato addestrato su 141 righe dei dataset
pubblici; 137 dei nostri 164 problemi sono quelli. Lo 0,739 è quindi in gran parte
**in-sample** per la loro sonda. Il suo vero fuori-campione è nei punteggi di validazione
salvati nell'artefatto stesso: 0,50/0,74/0,60/0,50 per fold. H35 sopravvive al null ma non
è un test pulito, e non va citata come tale.

**Riserva 2, su H36, decisiva.** 0,71 è **lo stesso numero del conteggio dei caratteri**
(0,70), e i layer scelti sono precoci, dove la rappresentazione è ancora lessicale. Se la
sonda legge la lunghezza, sul set di gara fallirà come V8, perché gli organizzatori hanno
neutralizzato le feature idiosincratiche del problema. Il controllo è eseguito qui sotto,
prima di qualunque decisione.

### Il controllo sulla lunghezza: la sonda legge quasi solo la lunghezza

| | AUROC |
|---|---|
| sonda nostra, OOF | 0,7097 |
| correlazione di Spearman fra score della sonda e n_char | **+0,551** |
| sonda al netto di log(n_token), residuo | 0,667 |
| dentro il terzile dei problemi lunghi (n=56, 21 robust) | 0,537 |
| **n_char da solo**, logistica in CV | **0,7012** |
| **n_char + sonda** | **0,7093** |
| sonda da sola | 0,6946 |

**Aggiungere la sonda al conteggio dei caratteri vale +0,008 di AUROC.** Le due cose portano
la stessa informazione. Dentro il terzile dei problemi lunghi, dove la lunghezza non
discrimina più, la sonda scende a 0,537, cioè al caso. Lo 0,917 nel terzile corto poggia su
4 robust e non conta.

**Verdetto.** H36 sopravvive al proprio criterio statistico e **non è utilizzabile**: il suo
segnale è la lunghezza del problema riletta dalle attivazioni dei layer lessicali. V8 ha già
misurato cosa vale la lunghezza sul set di gara, dove è stata neutralizzata: 0,52. Una sonda
che porta +0,008 oltre la lunghezza porterà lo stesso 0,52 più rumore, sui 4 item di
DeepSeek. **Nessuna submission da questo canale.** La previsione «entrambe muoiono» era
sbagliata nella forma e giusta nella sostanza: sopravvivono come statistiche, muoiono come
metodo, e il criterio che le ha uccise, il confronto con n_char, era nel protocollo di E11 e
andava incluso anche qui fin dall'inizio. Registro l'omissione.

**Bilancio del progetto:** ventisei ipotesi pre-registrate; ventuno morte al proprio criterio;
due sopravvissute e poi spiegate da una feature che il set di gara annulla; tre ancora
aperte nel report (previsioni P1, P2 di E12 e la nota E14). Nessun canale porta oltre la
classe di maggioranza sul set di valutazione, con verità a terra, nemmeno quello degli
organizzatori.


## CORREZIONI E DEVIAZIONI, dichiarate il 21 e il 23 settembre 2026 (verifica avversaria del report)

Il testo sopra resta com'era. Quello che segue lo corregge.

1. **Null nidificato di H36.** Sono state usate 60 permutazioni invece delle 500 scritte nel
   disegno, e una griglia di regolarizzazione ridotta (C in {0,01; 1}) mentre la statistica vera
   sceglie fra {0,001; 0,01; 0,1; 1} e ha scelto 0,001 in tutti e cinque i fold. La deviazione
   non fu dichiarata al momento. La correzione di Holm sulle due primarie non fu calcolata.
2. **H35.** L'aggregazione del disegno (gruppi NONE, ciascuno al proprio layer migliore) non e'
   quella del codice di inferenza dell'artefatto (layer 31, media dei margini di tutti i 40
   probe, gruppi di controllo compresi). Con il percorso ufficiale l'AUROC e' 0,709
   (`vm/e15_percorso_ufficiale.py`), con quella del disegno 0,739; in entrambi i casi la regola
   predice robust per nessun problema. L'artefatto non registra con quali dati fu addestrato:
   che 137 dei 164 problemi siano nel suo addestramento e' un'inferenza, non un fatto.
3. **Controllo di lunghezza.** Lo script non era stato salvato. Ricostruito il 23/9
   (`vm/e15_controllo_lunghezza.py`, output in `vm_risultati/e15_controllo_lunghezza_2026-09-23.txt`),
   riproduce esattamente AUROC OOF 0,7097, Spearman +0,551, residuo 0,667 e terzile lungo 0,537
   (56 problemi, 21 robust, terzili per soglia di quantile). La CV di secondo livello dipende
   dalla divisione in fold: su 20 divisioni n_char vale 0,696 in media, la sonda 0,697, entrambi
   0,714. **Il guadagno della sonda sulla lunghezza e' +0,018 in media (da +0,006 a +0,032), non
   +0,008**: quello era una sola divisione. La conclusione (la sonda legge soprattutto la
   lunghezza) regge; «+0,008» va letto «circa due punti».
4. **«Layer lessicali».** Nessuna analisi lo mostra, e due fold su cinque scelgono il layer 20.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Ritirato** (report, Appendice D, voce 16): che V8 sia fallito sul set di valutazione perche' gli organizzatori avevano neutralizzato le feature idiosincratiche del problema (Riserva 2), che V8 avesse gia' misurato quanto vale la lunghezza sul set di gara, dove sarebbe stata neutralizzata, a 0,52 (Verdetto), e che le due ipotesi sopravvissute siano spiegate da una feature che il set di gara annulla (Bilancio). Una valutazione su 21 e 19 item non mostra che una feature non porta nulla, e 0,66 sta dentro gli intervalli di Wilson al 95% di entrambi i punteggi (`vm/controlli_verifica4.py`).
2. **Riserva 1 corretta** (report, 8.3 e Appendice D, voce 46). I punteggi salvati nell'artefatto sono accuratezze bilanciate di validazione, sulla partizione che il codice pubblico degli organizzatori (`scripts/probe.py`, copia in `dati/`) usa anche per l'early stopping, di una regola che predice quasi sempre la stessa classe: 734 dei 740 probe non di controllo valgono esattamente 0,500, e la media per layer piu' alta e' 0,5475 (fold 1, layer 31), non «mai sopra 0,547». Non sono il suo rendimento fuori campione. L'addestramento probabilmente in-sample non e' cio' che produce la sua AUROC sulle nostre etichette: il margine segue il numero di caratteri (Spearman +0,77 con l'aggregazione del codice di inferenza, +0,85 con quella del disegno), residualizzato su log(n_char) vale 0,57 e 0,62, nel terzile dei prompt piu' lunghi 0,46 e 0,48, e i quattro gruppi di controllo, addestrati su etichette rimescolate, danno da soli 0,68; per ogni seme il probe ha la stessa direzione con etichette vere o rimescolate (coseno da 0,93 a 0,998, contro al piu' 0,08 fra semi diversi). Il probabile file di addestramento coincide con `data/math-robust-agg.csv` del repository pubblico aimo-interp/baselines (141 righe, 137 problemi). `vm/e15_artefatto_controlli.py`, `vm/controlli_verifica3.py`.
3. **Ritirata** la frase della nota 3 precedente secondo cui «la conclusione (la sonda legge soprattutto la lunghezza) regge» (report, 8.3 e Appendice D, voce 46). Come predittori dell'etichetta sonda e numero di caratteri sono in gran parte ridondanti: insieme guadagnano circa due punti su ciascuno da solo (0,714 contro 0,696 e 0,697, medie su 20 divisioni). Che cosa legga la sonda non e' stabilito: residualizzata su log(n_char) conserva AUROC 0,62, e log(n_char) residualizzato sulla sonda conserva 0,62.
4. **Null nidificato rieseguito** in locale il 23/9 dagli stati nascosti salvati, con gli stessi semi (`vm/e15_null_nidificato.py`): con la griglia ridotta di allora il p95 e' di nuovo 0,5744 (media 0,4885); con la griglia piena dei quattro valori di C il p95 e' 0,5700 (media 0,4890). In nessuno dei due casi una delle 60 permutazioni raggiunge 0,7097. Le 500 permutazioni del disegno non sono state eseguite.
5. **Estrazione.** H35 e H36 usano l'estrazione con il system prompt dell'artefatto: 91 token di prompt in media e 17 secondi per i 174 problemi (log `e15_estrai.log` nell'archivio della VM). I 47 token e 16 secondi dell'esito sopra sono quelli dell'estrazione senza system prompt.


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Null nidificato, precisazione al punto 4 della nota precedente.** Il log del 14/9 e' salvato: work/aimo/e15_valuta.log nell'archivio vm_backup/backup_aimo_2026-09-17.tar.gz, estratto il 23/9 in vm_backup/estratto (con system prompt: media 0,4895, p95 0,5744; senza system prompt: media 0,4960, p95 0,5888). La riesecuzione locale, con lo stesso codice e gli stessi semi, riproduce il p95 (0,5744) ma non la media (0,4885): non tutte le 60 permutazioni danno lo stesso valore, per una causa non documentata (gli ambienti differiscono: sulla VM scikit-learn 1.9.1 e numpy 2.3.5, in locale 1.8.0 e 2.4.4). Il docstring di `vm/e15_null_nidificato.py`, che diceva quell'output non salvato, e' stato corretto.
2. **Layer 31, precisazione al punto 2 della nota precedente.** Lo 0,68 dei quattro gruppi di controllo e i coseni fra probe (da 0,93 a 0,998 per lo stesso seme con etichette vere o rimescolate, al piu' 0,08 fra semi diversi) sono calcolati al layer 31 (`vm/e15_artefatto_controlli.py`); agli altri layer non valgono.
3. **Ritirata** anche la frase «Sui loro dati, con il loro metodo, il segnale lineare nelle attivazioni di DeepSeek è indistinguibile dal caso» (sezione «Cosa dice già l'artefatto»): i punteggi salvati sono accuratezze bilanciate di una regola quasi costante e non dicono come l'artefatto ordina i problemi (report, 8.3).
