# Pre-registrazione E8 — la calibrazione realizzabile cattura il tetto?

Scritta il 31 agosto 2026 **prima** di eseguire `esperimento_e8.py`.

## Da dove nasce, e perché non è variant-hunting

E7 ha prodotto due risultati opposti e entrambi pre-registrati:
- **H12 morta** (−16,5): una soglia fissa applicata a un modello bersaglio mai visto fallisce.
- **H13, il tetto, positivo** (+7,3, soglia 4,8): con la soglia giusta per il bersaglio, le
  feature di incertezza del modello in cache **predicono modelli che non possiamo caricare**.

H12 resta morta e non la si rianima. La domanda di E8 è un'altra e nasce dal tetto: **esiste
una regola di calibrazione, calcolabile a runtime senza conoscere le etichette, che cattura
parte di quei 7,3 punti?**

La diagnosi di E7 dice esattamente cosa serve: il fallimento è il **tasso di base** del
bersaglio, non il ranking. `Llama-3.1-8B` è robust nello 0% dei casi e da solo produce
54 punti di danno.

## Le due calibrazioni candidate, entrambe realizzabili nella gara

**C1, prior pubblico.** Il `model_id` è noto a runtime. Per i modelli presenti nei dataset
pubblici degli organizzatori si conosce il tasso di robustezza; si predice `robust` la frazione
top-p del lotto, con p = quel prior. Per i modelli mai visti si usa il prior globale (0,239).

**C2, prior globale per tutti.** Nessuna informazione sul bersaglio: si predice `robust` il
top 23,9% di ogni lotto. È la versione che funziona anche su modelli completamente nuovi.

## Ipotesi

**H14.** C1 batte la classe di maggioranza per modello, in leave-one-model-out.
**H15.** C2 batte la classe di maggioranza per modello, in leave-one-model-out.

**CRITERIO DI MORTE, dichiarato prima:** guadagno < `sqrt(0.25/109)` = **4,8 punti** -> morta.

**Controllo di onestà obbligatorio:** entrambe vengono valutate **anche escludendo
`Llama-3.1-8B`**, e il risultato con e senza viene riportato. Se una calibrazione vive solo
grazie all'esclusione del caso peggiore, **non conta** e va dichiarata morta. Il senso non è
trovare un sottoinsieme che passa: è capire se il metodo regge sul caso avverso.

## Cosa decide

- Se H14 o H15 vivono **includendo Llama**: si costruisce la soluzione che carica il modello
  in cache e lo fa girare su **tutti** i problemi, anche quelli dei modelli ciechi, e la si
  spedisce. Costo stimato 17 x 119 s = 2023 s, dentro il limite di 3600 s.
- Se muoiono entrambe: il tetto resta teorico, i 15 item ciechi restano irriducibili, e la
  conclusione è definitiva perché misurata con le feature vere.

## Esito

Da riempire dopo l'esecuzione.

## Esito, 31 agosto 2026

| | con Llama (quello che conta) | senza Llama (controllo) |
|---|---|---|
| **H14** prior pubblico del bersaglio | −2,8 punti → **MORTA** | +1,4 → **MORTA** |
| **H15** prior globale per tutti | −10,1 punti → **MORTA** | +0,0 → **MORTA** |
| tetto realizzabile (k oracolo, top-k) | −2,8 | +1,4 |

Entrambe morte in tutte e quattro le celle. Il controllo di onestà non serve nemmeno a
scagionarle: non vivono neanche escludendo il caso avverso.

## E una correzione a me stesso, importante

Il tetto di E7 (H13) dava **+7,3 punti**; il tetto di E8, che è la stessa cosa ma vincolata a
scegliere i top-k, dà **−2,8**. La differenza non è rumore: è che **l'oracolo di E7 poteva
degenerare alla costante**. Per `Llama-3.1-8B`, robust nello 0% dei casi, la soglia ottimale è
"non predire mai robust", che recupera 39 su 39 senza usare una sola feature.

Quindi il "+7,3 punti di tetto" che ieri ho presentato come prova che *"il segnale esiste ed è
cross-model"* era in buona parte la libertà di non usare il segnale. Il tetto onesto, quello
che obbliga a ordinare gli item e sceglierne k, è negativo.

**Va corretto anche nell'esito di E7**, dove quella frase resta scritta come l'ho pensata
allora. Il canale non è "esistente ma scalibrato": è chiuso.

## Bilancio

**Dieci ipotesi pre-registrate, dieci morte.** Le ultime tre (H12, H14, H15) usavano le
feature di incertezza reali estratte con 9,2 ore di GPU, non più proxy. Il che rende la
conclusione definitiva invece che provvisoria:

> Le feature interne del modello in cache non predicono la robustezza di modelli che il
> container non consente di caricare. I 15 item su 17 sono irriducibili, ed è misurato con
> lo strumento vero.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Meccanismo del calo di H13.** Su Llama anche l'oracolo con k vero (k = 0) fa 39 su 39, come la maggioranza: il calo da +7,3 a -2,8 viene tutto dagli altri sette bersagli, cioe' dal fissare k e dover ordinare gli item, non dal caso Llama. La spiegazione data sopra («in buona parte» Llama) va ritirata. `vm/warmup_ricalcoli.py`.
2. **H14 coincide con il tetto.** Il prior pubblico di ogni bersaglio in `dati/prior_per_modello.json` e' il suo tasso di robust sulle stesse righe su cui e' valutato; per tutti gli otto bersagli round(prior x n) e' il numero vero di robust, quindi H14 sceglie il k dell'oracolo: 0,7431 in entrambi i casi (-2,8), +1,4 senza Llama in entrambi i casi. Il verdetto (morta) regge. `vm_risultati/esperimento_e8_2026-09-23.txt`.
