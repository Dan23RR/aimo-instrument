# Pre-registrazione V8b: lo stesso bundle V8 sui task sostituiti

Scritta il 21 settembre 2026, PRIMA di qualunque invio sui task nuovi.
**Stato al momento della stesura: in attesa del via di Daniel.** Esito in fondo.

## Perché

Il 16 settembre gli organizzatori hanno sostituito entrambi i task della fase di gara
(verificato dalla definizione della fase via API pubblica di Codabench, 21/9):

| | task vecchio | task nuovo |
|---|---|---|
| Main | 35766, 21 item | **36237**, "AIMO Robustness Validation v8 - Main Track", 21 item |
| Small | 35767, 19 item | **36241**, "AIMO Robustness Validation v8 - Small Model Track", 14 item |

I nostri invii (v6 912096/912097, V8 924969/924974/925822) stanno sui task vecchi e **non
compaiono nella classifica corrente**. Il sito, alla voce Transparency, chiede che per i
premi del report gli organizzatori possano riprodurre «their score» dalle sole istruzioni del
report. Serve quindi un invio valido sui task correnti. V8 è l'unico nostro invio
riproducibile per costruzione: nessuna GPU, nessuno stato numerico, tre valutazioni su tre
identiche (11/21) sul vecchio Main. v6 invece eredita la deriva misurata in E14: il suo
punteggio non è riproducibile esattamente, per nostra stessa misura.

## Cosa si invia

Gli stessi identici file del 13 settembre, senza modifiche:

- `sonde/solution-v8-main.zip`, sha256 `7b70f15c51752dcf4285e477d58c541d0ecf052d833c325c0702dbbd054ffb5f`, **solo** task 36237 (Main)
- `sonde/solution-v8-small.zip`, sha256 `0b83b55f94dc16b9852a7c2b35d71ac5787e43d81afd6396dee00fb5e011e500`, **solo** task 36241 (Small)

Interfaccia: la vecchia `are_robust(model_id, problems)`. Il README aggiornato il 16/9
(PR #9, «reasoning effort») dice che «remains fully supported; existing submissions require
no changes». L'ingestione controlla la firma con `inspect` e usa la chiamata vecchia quando
manca il parametro `reasoning_effort`.

**Prima di caricare:** deve essere spuntato un solo task. Lezione del 13-14/9: con due caselle
spuntate nasce un figlio fallito sull'altro task, e con la regola «conta l'ultimo» diventa la
nostra riga in quel task.

## Previsioni, scritte prima

- **P-V8b-1.** coverage 1,0 e invalid 0 su entrambi i task: la chiamata vecchia funziona con
  la nuova ingestione. Se falsa, si corregge la firma prima di qualunque altra cosa.
- **P-V8b-2.** Nessun segnale: accuratezza entro due errori standard da 0,5, cioè **fra 6/21 e
  15/21 sul Main** e **fra 4/14 e 10/14 sullo Small**. Fuori banda verso l'alto vorrebbe dire
  che la lunghezza porta segnale sul set nuovo, contro quanto misurato sul vecchio (0,52).
- **P-V8b-3.** Tempo di esecuzione sotto i 60 secondi per invio.

## Cosa NON si fa

Nessun secondo invio per provare altre soglie o altre regole. Il punteggio ottenuto non entra
in nessuna decisione di metodo: serve solo come voce valida e riproducibile per la valutazione
finale e per il report. La decisione su cosa lasciare come ultimo invio a fine ottobre resta
aperta e separata.

## Esito

Da riempire dopo l'invio, senza riscrivere nulla di quanto sopra.

## ESITO, 21 settembre 2026, ore 10:35. Via di Daniel: «chrome collegato, procedi».

Inviati gli stessi identici file, un task per invio. Prima di ogni caricamento la casella
dell'altro task è stata tolta e lo stato verificato; dopo ogni invio la pagina rimette
entrambe le caselle spuntate, quindi il controllo va rifatto ogni volta.

- **936802** `solution-v8-main.zip`, task 36237 (Main), nessun figlio: **10/21 = 0,476**, coverage 1,0, invalid 0.
- **936806** `solution-v8-small.zip`, task 36241 (Small), nessun figlio: **7/14 = 0,500**, coverage 1,0, invalid 0.

**P-V8b-1 confermata:** la chiamata vecchia funziona con la nuova ingestione su entrambi i task.

**P-V8b-2 confermata, ma in modo degenere.** I conteggi che il bundle stesso scrive nel log
dicono che ha predetto «robust» per **tutti** gli item: 21 su 21 nel Main, 14 su 14 nello
Small. Sul set nuovo V8 coincide con la costante «sempre robust», cioè con la baseline
`always-true` degli organizzatori, e il suo punteggio è la quota di item robust: 47,6% nel
Main, 50% nello Small. Il set nuovo è davvero bilanciato. Una costante non può portare
segnale, quindi «nessun segnale» è vera per costruzione e non dice nulla sulla lunghezza.

**P-V8b-3 confermata come limite superiore.** 936802 risultava Finished prima dell'invio
successivo, creato 55 secondi dopo di lui, coda e punteggio compresi; 936806 entro circa 40
secondi. I log non riportano tempi.

**Il dato utile per leggere la classifica.** La costante vale 10/21 e 7/14 sui task nuovi. I
migliori ultimi invii per partecipante, letti il 21/9 mattina prima dei nostri, erano 13/21
e 9/14: +3 item (1,3 errori standard) e +2 item (1,1 errori standard) sopra la costante; il
massimo su singolo invio nel Main, 14/21, è +4 item (1,8 errori standard), prima di
correggere per la selezione del massimo su 68 e 126 invii. Nessuna voce della classifica
corrente si distingue dalla costante. È un'affermazione sulla classifica aggregata, non sul
metodo di nessuno.

**Nota di trasparenza.** Il bundle V8 scrive nel log, per ogni modello, il numero di problemi,
quanti ne predice robust e min/mediana/max dei caratteri. Le statistiche di lunghezza
riguardano input del set di valutazione: sono state lette solo per verificare il
comportamento della regola e non vengono usate per tarare nulla, come stabilito in PREREG_V8
(«nessuna soglia tarata su range osservati nei log»). Il report lo dichiara al §13.

**Conseguenza per la voce finale.** Se anche il set finale è fatto di problemi lunghi, V8
resterà la costante: la voce più onesta e riproducibile, ma senza informazione. Cosa lasciare
come ultimo invio a fine ottobre resta una decisione separata.

## CORREZIONE, 21 settembre 2026 (verifica avversaria del report)

«Tre esecuzioni, tre volte 11/21» è sbagliato. Sui task vecchi le valutazioni valide di V8 sono
tre, ma su task diversi: il bundle Main ha dato 11/21 due volte sul task Main (924969, figlio di
924968, e 925822); il bundle Small ha dato 10/19 una volta sul task Small (924974, figlio di
924972). I due figli rifiutati (924970, 924973) non contano. Il fatto che resta è più piccolo:
due valutazioni dello stesso bundle sullo stesso task, due punteggi identici.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Force_Latest_Multiple.** La lettura della regola come «solo l'ultimo invio per task» era sbagliata (vedi PREREG_V8, correzione del 23/9): Codabench aggiunge ogni invio valutato come riga a se'.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Ritirato** (report, Appendice D, voce 16): nella motivazione di P-V8b-2, «contro quanto misurato sul vecchio (0,52)» presuppone che V8 abbia misurato che la lunghezza non porta segnale sul set vecchio. Una valutazione su 21 e 19 item non lo mostra, e 0,66 stava dentro gli intervalli al 95% di entrambi i punteggi.


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Fonte.** Gli intervalli al 95% della nota precedente, che contengono 0,66, sono intervalli di Wilson (`vm/controlli_verifica4.py`, sez. 8: 11/21 da 0,324 a 0,717; 10/19 da 0,317 a 0,727).
