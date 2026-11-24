# Pre-registrazione E13: affidabilità test-retest dell'etichetta di robustezza

Scritta l'8 settembre 2026, PRIMA di eseguire. Hardware: la VM degli organizzatori.

## La domanda che E12 ha aperto e non può chiudere

E12 (7/9): le etichette rigenerate con le famiglie loro, il loro generatore e la loro regola
di congiunzione **non concordano** con le loro (kappa −0,001, n=157). Due spiegazioni opposte
producono lo stesso risultato:

- **(i) L'etichetta è instabile per costruzione.** Il loro regime, letto dai dati pubblici, è:
  `n_base_predictions` = 10 (707 righe su 990), `n_permuted_predictions` = 10 per famiglia
  (528 righe), `n_predictions` = 1 per variante (1.553 varianti su 1.787). Cioè dieci varianti
  per famiglia, **una sola predizione campionata per variante**, dieci sulla base, robust se
  tutte corrette: una congiunzione su ~50 estrazioni singole a temperatura. Con la
  self-consistency del 93,5% misurata in H33, un pair perfettamente robusto le supera tutte
  per caso il 3-4% delle volte. Se è così, due estrazioni indipendenti darebbero etichette
  diverse anche a loro, e nessun metodo può predire l'etichetta oltre il caso.
- **(ii) L'etichetta è stabile, ma la loro è stata prodotta con un processo diverso** da quello
  dichiarato (varianti diverse, cancello diverso, correzione manuale). Allora il difetto è
  nell'esecuzione, non nella definizione, e si può indicare dove.

Solo il **test-retest** le separa: rigenerare l'etichetta due volte, indipendentemente, e
misurare quanto le due copie concordano fra loro. **Non richiede etichette vere**, quindi si
misura su tutti e cinque i modelli di gara, anche i quattro che non ne hanno.

## Disegno, fissato ora

**Problemi.** I 164 con etichetta vera per DeepSeek (`gpu/out/feature_greedy.parquet`), usati
per tutti i modelli.

**Varianti (una volta sola, sono del problema e non del modello).** Generatore
`openai/gpt-oss-120b`, reasoning low, T=1,0, top_p=1,0, famiglie loro (`rephrase`, `rename`,
`domain`, `distract`, `typos`), **10 varianti per famiglia**, 5 esempi few-shot dai loro 638
testi mai dello stesso problema, come in E12. **Due estrazioni indipendenti**: D1 con seme
20260913, D2 con seme 20260914. Cancello di qualità come in `PREREG_E12_DEVIAZIONE.md`:
riferimento = gpt-oss medium sull'originale; variante accettata se testo diverso, rapporto di
lunghezza in [0,7, 4,0], risposta uguale al riferimento. Regola del 60% per famiglia
invariata (in E12 ha escluso `domain`).

**Regime di etichettatura, il loro.** Per ogni modello: T=0,6, top_p=0,95, max_tokens 8.192
(chat template del modello, reasoning dove esiste); **10 campioni sull'originale, 1 campione
per variante**; "corretto" = uguale al riferimento; **`base_ok` = almeno 6 campioni su 10
corretti** (il loro set di valutazione esclude i problemi con base_accuracy vicina a zero:
sensibilità riportata a 5/10 e 8/10); **`robust` = `base_ok` E ogni variante accettata
corretta.** Problemi senza riferimento o senza `base_ok` esclusi e contati.

**Tre condizioni per modello.**

| | varianti | seme di campionamento | cosa isola |
|---|---|---|---|
| A | D1 | a | riferimento |
| B | D2 | b | affidabilità **totale**: kappa(A,B) |
| A′ | D1 | c | quota dovuta al **solo campionamento**: kappa(A,A′) |

**Statistiche primarie, per modello: kappa(A,B) e kappa(A,A′)**, con intervallo di
confidenza bootstrap sui problemi (2.000 ricampionamenti). Su DeepSeek anche **kappa(A,loro)
e kappa(B,loro)**. Nessuna soglia di "morte" nel senso di E12: qui la domanda è quale cella
della griglia, e la griglia è scritta ora.

**Griglia di lettura, scritta ora.**

| kappa(A,B) | kappa(A,loro) su DeepSeek | lettura | conseguenza |
|---|---|---|---|
| < 0,3 | ~0 | l'etichetta è instabile per costruzione (i) | nessun metodo la predice oltre il caso; P1/P2 di E12 confermate in anticipo; va detto agli organizzatori con i numeri e con l'offerta di rifarlo sul loro pipeline |
| > 0,6 | ~0 | riproducibile, ma le loro non vengono dalla definizione (ii) | si indica dove diverge; si chiede il loro pipeline |
| > 0,6 | > 0,3 | E12 era sottodimensionato (3 varianti contro 10) | si riapre lo Stadio 3 di E12, pre-registrato a parte |
| 0,3-0,6 | qualunque | zona grigia | si riporta la curva di affidabilità e non si conclude oltre |

**Secondarie.** (s1) Curva di affidabilità: kappa(A,B) ricalcolato sottocampionando 3, 5 e 10
varianti per famiglia, per calibrare retroattivamente E12. (s2) kappa(A,B) per famiglia.
(s3) Affidabilità della sola base: accordo di `base_ok` fra A e A′. (s4) Tasso di robust per
modello e condizione.

## Ordine di esecuzione e costo

1. Varianti D1 e D2 + cancello: 164 × 5 × 10 × 2 = 16.400 varianti, 16.600 risoluzioni su
   gpt-oss-120b. In E12 gpt-oss ha risolto 2.613 problemi in 17 minuti su H100: **3-4
   ore-scheda in tutto**.
2. Etichettatura, un modello per volta, nell'ordine: **DeepSeek** (ha la verità a terra),
   Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think, gpt-oss-120b. Per modello: 3 condizioni ×
   164 × (10 + ~40 varianti accettate) ≈ 25.000 generazioni; a 1,8 s per generazione (E12,
   A100-80GB, 8.192 token): **~12-15 ore-scheda per modello piccolo**, ~20-25 per gpt-oss.
   Totale ≈ 75-85 ore-scheda.
3. Se il budget stringe, si taglia prima A′ sui modelli senza etichette vere, poi Skywork.

Tutto a blocchi con salvataggio parziale, in tmux, con copia serale dei parquet fuori dalla VM.

## Emendamento pre-esecuzione, 8 settembre 2026 ore 12:55, dopo la prova su 3 problemi

La prova a scala ridotta sulla VM (3 problemi, k=2, DeepSeek con 3 campioni base) ha
mostrato che **due problemi su tre saturano gli 8.192 token in tutti i campioni senza dare
risposta**, fra cui `02c9e0` che loro etichettano robust. Con quel tetto, `base_ok` fallirebbe
per il budget di token e non per il comportamento del modello, e la riproduzione sarebbe
distorta verso i problemi corti. La loro base_accuracy di 0,97 dice che il loro budget bastava.
Quindi, **prima di qualunque esecuzione a scala**: `max_tokens` dell'etichettatura da 8.192 a
**16.384**; `max_tokens` del risolutore del cancello (gpt-oss medium) da 4.096 a **8.192**
(nella prova 5 varianti su 30 erano scartate per risposta assente del risolutore). Tutto il
resto invariato. Costo stimato per modello: da ~15 a ~20 ore-scheda; il taglio di A′ sui
modelli senza etichette vere resta la prima riserva.

## Secondo emendamento pre-esecuzione, 9 settembre 2026 ore 08:15: il formato del prompt

**Difetto trovato nei nostri script, non nel disegno.** L'etichettatura passava al modello il
solo testo del problema. DeepSeek-R1 chiude spontaneamente in `\boxed{}` e infatti produce
8,99 risposte estraibili su 10; **Qwen3.5-4B no**: entra in un loop e satura il tetto di token
senza mai chiudere, producendo 0,03 risposte su 10. Due esecuzioni complete di Qwen (13
ore-scheda) sono state invalidate e messe in `e13/scartati_prompt_nudo/` con la spiegazione.

**Correzione, prima di qualunque nuova esecuzione:** i modelli diversi da DeepSeek ricevono
l'istruzione esplicita di formato, la stessa già usata dal risolutore del cancello ("Solve the
following problem. Reason step by step, then give only the final answer inside \boxed{}").
DeepSeek resta col prompt nudo, perché la sua condizione A è già completa e la coerenza che
conta per il kappa è **dentro** il modello, non fra modelli. La scelta è codificata in
`e13_coda.sh` e dichiarata in ogni log.

**Cancello di fattibilità aggiunto,** per non ripetere l'errore al buio: prima di spendere sei
ore su un modello, 30 problemi × 5 campioni (mezz'ora) stimano la quota di problemi con almeno
il 60% di campioni corretti. Il modello entra in E13 solo se quella quota è ≥ 15%; altrimenti
l'etichetta sarebbe costante per costruzione, il kappa non definito, e la cosa va dichiarata
invece di essere scoperta dopo.

**Risultato collaterale, che vale oltre la diagnosi.** Sugli stessi 30 problemi, con lo stesso
modello e la stessa temperatura, la quota di problemi risolti passa da circa **0% con il
prompt nudo a 80% con l'istruzione di formato**. La misura di robustezza, che è definita come
congiunzione di risposte corrette, **dipende quindi in modo drastico da una scelta di prompt
che il benchmark non specifica**. Se gli organizzatori hanno etichettato con un formato e un
partecipante ne usa un altro, le due etichette non sono confrontabili. Va nel report accanto a
E12 e a E14: è la terza forma dello stesso problema, cioè che il numero misurato dipende da
convenzioni non dichiarate.

## Cosa NON si fa

Nessuna sweep su temperatura, numero di varianti, soglia di `base_ok`. Nessun addestramento
su queste etichette per una submission. Nessuna lettura del punteggio di gara.

## Esito

Da riempire per modello, senza riscrivere nulla di quanto sopra.

## ESITO FINALE, quattro modelli, undici etichettature. 13 settembre 2026.

Eseguito interamente sulla VM degli organizzatori fra l'8 e il 10 settembre. 164 problemi,
due estrazioni indipendenti di varianti (10 per famiglia, 5 famiglie loro, ~40 varianti
accettate per problema), regime loro (10 campioni sulla base, 1 per variante, congiunzione).
Circa 100 ore-scheda. Tutti i dati copiati fuori dalla VM in `AIMO/vm_risultati/`.

| modello | robust nostro | **kappa(A,B)** | IC 95% | kappa(A,A′) | quota di instabilità dal solo campionamento |
|---|---|---|---|---|---|
| Qwen3.5-4B | 9,1% | **0,178** | [−0,057, +0,393] | non eseguita | n.d. |
| Olmo-3-7B-Think | 65,9% | **0,378** | [+0,215, +0,537] | 0,653 | 56% |
| DeepSeek-R1-Qwen3-8B | 37,8% | **0,413** | [+0,262, +0,555] | 0,464 | 91% |
| Skywork-OR1-Math-7B | 72,6% | **0,439** | [+0,276, +0,587] | 0,789 | 38% |

**Media 0,352, intervallo osservato 0,18-0,44. Nessun modello raggiunge quello che si
chiamerebbe accordo sostanziale.**

### Risultato 1: l'etichetta non riproduce se stessa

Applicando due volte la definizione dichiarata, con varianti generate indipendentemente e
campionamento indipendente, l'accordo fra le due copie sta fra 0,18 e 0,44. Su Qwen l'intervallo
di confidenza include lo zero: per quel modello la griglia assegna la cella **(i), etichetta
instabile per costruzione**. Per gli altri tre la griglia assegna **zona grigia**, e la
pre-registrazione impone di non concludere oltre. Lo rispetto.

### Risultato 2: la sorgente dell'instabilità cambia da modello a modello

La terza condizione, che tiene ferme le varianti e cambia solo il seme, decompone il rumore.
Il risultato è che **non esiste una sorgente unica**: su DeepSeek il 91% dell'instabilità viene
dal solo campionamento, su Olmo il 56%, su Skywork il 38%. DeepSeek satura spesso il tetto di
token e la sua risposta oscilla; Skywork risponde nettamente ma è sensibile a *quali*
riformulazioni gli capitano. Chi volesse stabilizzare l'etichetta deve quindi intervenire su
entrambi i fronti, e in proporzioni diverse per ogni modello.

### Risultato 3: la concordanza con le etichette ufficiali resta zero, e non è rumore

Su DeepSeek, l'unico modello di cui possediamo le etichette vere: **+0,025, −0,043, +0,025**
nelle tre condizioni, con intervalli che includono sempre lo zero. La curva di affidabilità
(sotto) mostra che non è un problema di scala. La correzione per attenuazione mostra che non è
un problema di rumore: anche assumendo il **loro** processo perfettamente affidabile, la
concordanza vera sarebbe 0,025/√0,413 = **0,039**.

### Risultato 4: la robustezza non è una proprietà del problema

Controllo cross-modello, cioè le nostre etichette per Qwen, Skywork e Olmo confrontate con le
etichette ufficiali **di DeepSeek**: valori fra −0,114 e −0,003. Se esistesse un effetto
"problema" comune ai modelli, cioè problemi intrinsecamente robusti per tutti, vedremmo
correlazione positiva. Non c'è. Su Qwen è anzi significativamente negativa.

### Il tetto, e perché spiega la classifica

Se il processo che genera l'etichetta concorda con se stesso a 0,35 in media, **nessun metodo
può concordare con quell'etichetta oltre quella soglia**: si sta predicendo una misura che non
concorda con se stessa più di così. Il primo in classifica, al 13 settembre, sta a 16 item su
21, che sono 2,2 errori standard binomiali sopra la classe di maggioranza. Su un test set di
quella dimensione, e con un'etichetta di quella affidabilità, quel margine non distingue un
metodo che funziona da un'estrazione fortunata. È il contenuto quantitativo delle previsioni
P1 e P2 registrate in E12.

### Cosa NON dico

Non dichiaro che il benchmark sia sbagliato. Dichiaro che, con la definizione pubblicata e le
famiglie pubblicate, la misura ha un'affidabilità test-retest fra 0,18 e 0,44 e che le nostre
repliche non concordano con le etichette ufficiali. Se gli organizzatori hanno usato un
processo diverso da quello dichiarato, il numero che conta è il loro test-retest, non il
nostro: ed è la misura che chiediamo loro di fare, offrendo il nostro codice.

## ESITO PARZIALE, DeepSeek, condizioni A e B. 9 settembre 2026, ore 09:15.

Prime due condizioni complete sul solo modello con verità a terra. Ap in esecuzione, gli altri
tre modelli in coda. 164 problemi, ~40 varianti accettate per problema in ciascuna estrazione,
regime loro (10 campioni sulla base, 1 per variante, congiunzione).

| | A | B |
|---|---|---|
| problemi definiti | 160 | 159 |
| base_ok | 89,6% | 89,6% |
| robust | 37,8% | 31,1% |
| varianti corrette | 90,8% | 89,6% |

| statistica | valore | IC 95% |
|---|---|---|
| **kappa(A,B)**, affidabilità del nostro processo | **+0,413** | [+0,262, +0,555] |
| **kappa(A,loro)** | **+0,025** | [−0,119, +0,175] |
| **kappa(B,loro)** | **+0,025** | [−0,121, +0,182] |

Per famiglia, kappa(A,B): rephrase +0,602, distract +0,545, typos +0,491, domain +0,394,
rename +0,343.

### La domanda che E12 aveva lasciato aperta è chiusa

E12 era sospettato di essere sottodimensionato: tre varianti per famiglia invece delle dieci
loro. La curva di affidabilità, ricalcolata sottocampionando le varianti, dice di no.

| varianti per problema | robust A | kappa(A,B) | kappa(A,loro) |
|---|---|---|---|
| 2 | 82,5% | +0,417 | −0,011 |
| 5 | 72,6% | +0,355 | −0,008 |
| 10 | 64,0% | +0,356 | −0,012 |
| 20 | 50,3% | +0,344 | +0,012 |
| 40 | 41,1% | +0,389 | +0,016 |

**Due fatti, entrambi stabili su venti volte di scala.** Primo: l'affidabilità del processo non
migliora aggiungendo varianti, resta fra 0,34 e 0,42. Non è un problema di potenza, è una
proprietà della definizione. Secondo: **la concordanza con le loro etichette resta nulla a
ogni scala.** Quindi il kappa −0,001 di E12 non era un artefatto del sottodimensionamento:
era il valore vero, e adesso è confermato con dieci volte le varianti e un'estrazione
indipendente.

### La correzione per attenuazione, che è il punto

Si potrebbe obiettare che la concordanza è nulla perché il nostro processo è rumoroso. Il conto
lo esclude. Con la formula classica, la concordanza vera è quella osservata divisa per la
radice del prodotto delle affidabilità. Anche assumendo che il **loro** processo sia
perfettamente affidabile, la concordanza vera sarebbe 0,025 / √0,413 = **0,039**. Ancora nulla.
**La mancanza di accordo non è attribuibile al rumore: è una differenza sistematica.** Il nostro
processo e il loro non misurano la stessa cosa, pur applicando la stessa definizione dichiarata
alle stesse famiglie sullo stesso modello.

### Il tetto che ne segue, e che vale per tutta la classifica

Se il processo che genera le etichette ha affidabilità test-retest attorno a 0,4, allora
**nessun metodo, per quanto buono, può concordare con l'etichetta oltre quella soglia**: si
sta predicendo una misura che non concorda con se stessa più di così. È il tetto che spiega,
senza bisogno di altre ipotesi, perché tutto il campo è rimasto incollato alla classe di
maggioranza per due mesi.

### Insensibilità alla soglia, che è una buona notizia metodologica

Cambiare `base_ok` da 5/10 a 10/10 non muove nulla: robust resta 39,0% e kappa(A,B) 0,413. La
ragione è strutturale, e la si vede nei dati: il 38,8% dei problemi ha **tutte** le varianti
corrette e il 3,1% nessuna, quindi la congiunzione sulle varianti domina completamente la
condizione sulla base. La scelta della soglia, che era la nostra deviazione più discrezionale,
non influenza nessun risultato.

### Rispetto della griglia

kappa(A,B) = 0,413 cade nella fascia 0,3-0,6, che la pre-registrazione chiama **zona grigia**:
si riporta la curva e non si conclude oltre. Lo rispetto: non dichiaro né (i) né (ii). Quello
che i dati aggiungono, e che la griglia non prevedeva, è che **kappa(A,loro) è zero a ogni
scala e resta zero dopo la correzione per attenuazione**. Questo era il ramo che nessuna delle
quattro celle contemplava, ed è il risultato da riportare.


## CORREZIONI E DEVIAZIONI, dichiarate il 21 e il 23 settembre 2026 (verifica avversaria del report)

Il testo sopra resta com'era. Quello che segue lo corregge.

1. **Prompt di A' su DeepSeek.** La condizione A' di DeepSeek e' stata etichettata con il prompt
   esplicito sul formato (il suo log dichiara `prompt=esplicito`), mentre A e B con il testo nudo:
   la coda gpu0 era partita l'8/9 con la vecchia `e13_coda.sh`, e quando A' parti' lo script di
   etichettatura aveva gia' il prompt esplicito come default. Il secondo emendamento sopra, che
   dice DeepSeek col prompt nudo in tutte e tre le condizioni, e' quindi sbagliato per A'. Il
   kappa(A,A') di DeepSeek mescola seme e prompt e non misura il solo campionamento.
2. **gpt-oss-120b** era nel disegno e non e' stato etichettato; la decisione non fu registrata.
3. **Griglia di lettura.** Assegna Qwen3.5-4B alla cella (i), etichetta instabile per
   costruzione. Il report se ne discosta, e lo dichiara (§7.3 e §11): la griglia presupponeva che
   la nostra rigenerazione riproducesse il processo degli organizzatori, e la concordanza 0,025
   mostra che non lo fa.
4. **Curva di sottocampionamento.** Quella del 9/9 usava una sola estrazione per taglia e nessuno
   script salvato; la griglia per problema sostitui' quella del disegno (per famiglia) senza
   dichiararlo. Ricostruita il 23/9 con 20 estrazioni per taglia (`vm/e13_sottocampionamento.py`,
   output in `vm_risultati/e13_sottocampionamento_2026-09-23.txt`): kappa(A,B) medio fra 0,36 e
   0,47 (estrazioni singole da 0,16 a 0,58), kappa(A,ufficiale) medio fra -0,02 e +0,03. La
   conclusione (piu' varianti non migliorano ne' l'affidabilita' ne' la concordanza) regge.
5. **Soglia della base.** «Da 5/10 a 10/10 non cambia nulla» vale per DeepSeek e Skywork; su
   Qwen3.5-4B a 10/10 il kappa(A,B) sale da 0,18 a 0,30, su Olmo da 0,378 a 0,384. Fra 5/10 e
   8/10, l'intervallo del disegno, non cambia nulla su nessun modello.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Ritirati i due paragrafi sul tetto** («Il tetto, e perché spiega la classifica» e «Il tetto che ne segue, e che vale per tutta la classifica»). Usavano l'affidabilita' della nostra rigenerazione dove serviva quella degli organizzatori, e un'affidabilita' limita la concordanza attraverso la sua radice quadrata, non al proprio valore: sulla classifica non ne segue nulla (report, Appendice D, voce 4).
2. **Regime di campionamento.** «n_predictions = 1 per variante (1.553 varianti su 1.787)» mescola file e modelli. Vale per il file dei problemi AIMO (784 varianti su 821); nei tre file MATH da cui vengono 154 delle 164 coppie DeepSeek la maggior parte delle varianti elencate come dannose ha 10 predizioni (290 su 345, 101 su 121, 146 su 190), e la base ne ha per lo piu' 10. E13 non ha quindi usato i conteggi del processo che ha prodotto le etichette DeepSeek confrontate. `vm/dati_pubblici_controlli.py`.
3. **Soglia della base, precisazione alla nota 5.** A 10/10 kappa(A,B) non cambia su DeepSeek e Skywork, ma nella condizione A' due problemi per modello hanno tutte le varianti corrette e base fra 6 e 9 su 10: kappa(A,A') passa da 0,789 a 0,760 su Skywork e da 0,464 a 0,486 su DeepSeek, e la concordanza di A' con le ufficiali da -0,043 a -0,082. `vm/controlli_verifica3.py`.
4. **Riferimento della correttezza.** «Corretto» significa uguale alla risposta di gpt-oss-120b (reasoning medium) sull'originale; il cancello e' lo stesso modello che genera le varianti, indipendente dai modelli etichettati ma non dal generatore.
5. **Seme e varianti.** Con la regola di questo file la quota di instabilita' dal solo campionamento e' 38% su Skywork e 56% su Olmo (tabella dell'esito): «conta quali varianti si estraggono» vale per Skywork, non per Olmo. Con la regola nuova (E13b) vale per entrambi (9% e 40%).


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Olmo con la regola nuova.** «Con la regola nuova vale per entrambi (9% e 40%)» (nota precedente, punto 5) confronta kappa calcolati su insiemi di problemi diversi (137 e 133). Sui 126 problemi etichettati in tutte e tre le serie, Olmo ha kappa(A,A') 0,715 e kappa(A,B) 0,434, con 5 e 9 disaccordi: il solo seme vale circa meta' dell'instabilita', come con la regola vecchia. Su Skywork la lettura regge (127 problemi comuni: 0,937 contro 0,467, 1 e 8 disaccordi). `vm/controlli_verifica4.py`; report, 7.2, 7.5 e Appendice D, voci 39 e 43.
2. **Curva, precisazione alla nota 4 del primo blocco.** La concordanza media con l'etichetta ufficiale sale da -0,02 (2 varianti) a +0,03 (40): resta lontana da un accordo utile, ma «non migliorano ... la concordanza» non e' esatto (report, 7.2; Appendice D, voce 39).


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Citazione.** Il punto 1 della nota precedente cita «Con la regola nuova vale per entrambi (9% e 40%)»; il testo (nota della seconda verifica, punto 5) e': «Con la regola nuova (E13b) vale per entrambi (9% e 40%)».
