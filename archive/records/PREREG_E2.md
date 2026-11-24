# Pre-registrazione E2 — il modello in cache può fare da sensore per i modelli ciechi?

Scritta il 29 agosto 2026 **prima** di far girare `esperimento_e2.py`.

## Da dove nasce

Tre fatti già stabiliti, in ordine:

1. **OSSERVATO** (submission 907310): 17 item, 4 modelli, **2 item su 17 su un modello caricabile**.
2. **E1, H1 MORTA**: il testo del problema non predice la robustezza, fa 8,7 punti peggio
   della maggioranza in CV raggruppata per problema.
3. **E1, H2 VIVA a z = 4,65**: la robustezza è una proprietà **del problema**, concorde fra
   modelli nel 77% dei casi contro il 49% atteso sotto indipendenza.

Se la robustezza appartiene al problema e non al modello, allora **non serve ispezionare il
modello che si deve giudicare**. Basta ispezionare *un* modello sullo stesso problema. E un
modello ispezionabile ce l'abbiamo: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`, l'unico in cache,
che possiamo far girare su tutti e 17 i problemi anche quando l'item appartiene a un altro modello.

Questo trasforma il vincolo del container da tetto invalicabile in problema risolvibile,
e non è una speranza: è ciò che H2 implica se il trasferimento regge.

## Difetto del campo che rende la domanda urgente

Dal README ufficiale della baseline `uncertainty-profiling`, verbatim:

> "If no supported alias or matching artifact exists, the solution returns `False` for every
> problem so the batch remains valid."

La baseline distribuita dagli organizzatori, quella da 69,2%, **restituisce `False` su ogni
item di ogni modello che non sia quello in cache**. Sui 17 item del warmup significa 15 item
buttati su un default costante. Chi la spedisce così non sta misurando niente su 15 item su 17.

## Ipotesi e criteri di morte

**H3 (trasferimento fra modelli).** L'etichetta di robustezza di un modello A su un problema
predice quella di un modello B sullo stesso problema meglio della classe di maggioranza di B.
Misurata su tutte le coppie ordinate (A, B) del pool `aimo-2025-reference`, matrice completa
8 modelli × 10 problemi.

**H4 (il sensore vero).** La `base_accuracy` di un modello A sul problema, cioè quanto spesso
A risolve il problema originale, predice la robustezza di un altro modello B sullo stesso
problema. È la versione misurabile a runtime: la self-consistency del modello in cache
approssima la sua base_accuracy senza bisogno della risposta corretta.

**CRITERIO DI MORTE comune, dichiarato prima:** un'ipotesi è morta se il guadagno medio sulla
maggioranza è inferiore a una deviazione standard, `sqrt(0.25/n)` sul numero di predizioni
valutate. Se entrambe muoiono, i 15 item ciechi sono irriducibili e il metodo onesto è una
costante, dichiarata come tale.

## Confini che restano

Nessuna sonda ulteriore sulla leaderboard. Distinguo, e lo metto per iscritto prima:
**sonda** = submission costruita per risolvere le etichette nascoste per differenza fra
punteggi; **sviluppo** = submission che esegue un metodo vero e ne registra le diagnostiche
aggregate. La prima è chiusa dal regolamento e non la facciamo. La seconda è partecipare.
In nessun caso viene registrato, conservato o trasmesso il testo dei problemi.

## Esito, 29 agosto 2026

Eseguito una volta. Matrice completa 8 modelli × 10 problemi, 560 predizioni per ipotesi.

### H3 MORTA e H4 MORTA

| ipotesi | usando il sensore | maggioranza | guadagno | soglia |
|---|---|---|---|---|
| H3, label di A → label di B | 0,5893 | 0,6625 | **−7,3 punti** | +2,1 |
| H4, base_accuracy di A → label di B | 0,5446 | 0,6625 | **−11,8 punti** | +2,1 |

Entrambe sotto il criterio dichiarato prima, entrambe **negative**, non semplicemente deboli.
Nessuna variante è stata cercata dopo aver visto i numeri.

**Perché falliscono, ed è istruttivo.** I tassi di robustezza per modello vanno dal 20% al 70%
su questo pool. Un singolo modello usato come sensore trasporta con sé il proprio tasso di
base, quindi predice bene solo i modelli che gli somigliano. Nel dettaglio: `qwen3-8b:low`
predice `gemini-3.1-pro` al 100% (entrambi al 20%) e `gpt-5.2` al 50% (che sta al 70%).
Non è un sensore del problema: è uno specchio di se stesso.

### Ciò che invece è misurato, e non era un'ipotesi

Il **tetto del canale-problema**: il voto di maggioranza degli *altri* modelli sullo stesso
problema predice a **0,7375** contro **0,6125** della maggioranza globale, cioè **+12,5 punti**.

Quindi il segnale del problema esiste ed è grande, come già diceva H2 a z = 4,65. Ma serve
un'**aggregazione su più modelli** per estrarlo, e nel container di valutazione ne esiste
esattamente **uno** caricabile. Un solo modello è una misura a un bit di una quantità che
richiede otto misure per emergere.

### Conseguenza per il metodo, dichiarata

Con l'istrumentazione attuale della gara, i 15 item su 17 che stanno su modelli non in cache
sono **irriducibili**. Non per difficoltà, ma perché il canale che porta il segnale
(l'accordo fra modelli sullo stesso problema) richiede accesso a più modelli, e il container
ne concede uno.

Ne segue la richiesta concreta agli organizzatori, che vale più di qualunque metodo:
**mettere in cache i modelli valutati**. Con quattro modelli caricabili invece di uno, il
canale da +12,5 punti si apre e il compito che la gara premia diventa eseguibile.
