# Pre-registrazione V7 — varianza di riesecuzione della STESSA bundle

Scritta il 31 agosto 2026 dopo v6 (11/17), PRIMA di rispedire.

## Perché, e cosa NON è

Confrontando i log di v5 e v6 a parità di codice e input, i punteggi del regressore
cambiano fra run: DeepSeek max 0,722 → 0,751; gpt-oss max 0,870 → 0,902. L'estrazione di
feature **non è deterministica** (kernel GPU, batching, bf16), quindi gli item con punteggio
vicino alla soglia possono cambiare classificazione da una riesecuzione all'altra.

**Dichiarazione di onestà, scritta prima di guardare qualunque numero:** questo NON è un
miglioramento del metodo. È campionare il rumore dello strumento. Se il massimo dei campioni
supera 11/17, il guadagno è **rumore, non merito**, e va detto ovunque il numero compaia.

## H18 e la misura

**H18.** La stessa bundle, rieseguita, produce punteggi diversi.
**Misura:** 3 riesecuzioni di `solution-v6-calibrata.zip`, byte per byte identica (stesso
file, non ricostruito). Con la v6 originale fanno **4 campioni**.

Si riportano: tutti e 4 i punteggi, min, max, media, deviazione standard, e i punteggi
interni del regressore per modello estratti dai log, che misurano la varianza a monte
della soglia.

## Regola di pubblicazione, vincolante

- Si pubblica il massimo dei campioni **solo se > 11/17**.
- Se pubblicato, **ogni menzione del numero deve riportare che è il massimo di k campioni
  della stessa bundle**, non un metodo migliore. Vale per il writeup, per il report tecnico
  e per qualunque comunicazione agli organizzatori.
- **Nessuna modifica al codice fra le ripetizioni.** Se venisse la tentazione di ritoccare
  la soglia dopo aver visto i punteggi, è tuning sul canale del punteggio: vietato.
- Si riportano i risultati **anche se il massimo è più basso di 11**, cioè anche se la
  misura ci fa apparire peggiori.

## Cosa vale davvero questa misura

Il valore non è il rango: è quantificare che **il benchmark ha rumore interno alla singola
submission**. Se la deviazione standard su 4 campioni è dell'ordine di un item, allora i
distacchi in classifica fino a ±1 item non sono distinguibili dal rumore di riesecuzione, e
questo è un finding sullo strumento che vale per tutti i partecipanti, non solo per noi.

## Esito

Da riempire con tutti e 4 i campioni.

## ESITO — 4 campioni della stessa bundle (sha256 c9b03b36..., mai ricostruita)

| run | submission | punteggio | olmo | deepseek | gemma | gptoss |
|---|---|---|---|---|---|---|
| 1 | 909976 | **11/17** | [0.213,0.564] | [0.454,0.751] | [0.425,0.789] | [0.340,0.902] |
| 2 | 909994 | **9/17** | [0.213,0.564] | [0.371,0.625] | [0.425,0.789] | [0.340,0.902] |
| 3 | 910002 | **11/17** | [0.213,0.564] | [0.454,0.751] | [0.425,0.789] | [0.340,0.902] |
| 4 | 910013 | **11/17** | [0.402,0.557] | [0.532,0.787] | [0.425,0.789] | [0.340,0.902] |

min 9, max 11, media 10,5, escursione **2 item = 11,8 punti percentuali**.

## H18 confermata, ma il mio modello a due stati è FALSIFICATO

Dopo tre campioni avevo affermato che si trattava di uno **switch bimodale**: due stati
riproducibili, il range DeepSeek che tornava identico al millesimo. **Il quarto campione lo
smentisce.** Ha prodotto un terzo stato per DeepSeek, mai visto ([0.532,0.787]), e per la
prima volta ha mosso anche **Olmo** ([0.402,0.557] invece di [0.213,0.564]).

Conteggio degli stati distinti su 4 run: DeepSeek **3**, Olmo **2**, gemma **1**, gpt-oss **1**.

La lettura corretta, che sostituisce quella che avevo dato:
- la deriva numerica è **pervasiva e non bimodale**: due modelli su quattro l'hanno mostrata,
  con più di due stati;
- ma il **punteggio** cambia solo quando la deriva porta un item **oltre la soglia**. Nel run 4
  i valori sono diversi da tutti gli altri, eppure gli split coincidono con il run 1, quindi
  lo stesso 11/17;
- gemma e gpt-oss non si sono mai mossi in 4 run: la deriva non è uniforme fra modelli.

Avevo affermato la bimodalità su tre osservazioni. Era una generalizzazione prematura da
campione piccolo, esattamente l'errore che questo progetto esiste per non commettere, e
va corretta anche in tutti i punti dove l'ho scritta.

## Cosa resta, ed è solido

Su 4 riesecuzioni **byte-identiche**, il punteggio ha assunto valori 9 e 11 su 17.
**Un'escursione di 2 item a codice e input invariati.** In classifica la distanza fra il
primo (0,71) e il quarto (0,65) è **un item**. Quindi il vertice della classifica warmup non
è distinguibile dal rumore di riesecuzione dello strumento, e questo è misurato, non dedotto.

Nota per gli organizzatori: il regolamento spezza le parità di punteggio con il **runtime
medio su tre prove**, cioè riconosce che il tempo va mediato. Il **punteggio**, invece, viene
preso da **una prova sola**.

## Regola di pubblicazione, rispettata

Nessuno dei campioni supera 11/17, quindi non è stato pubblicato nulla di nuovo. In classifica
resta 909976 (0,65), che è **il massimo di 4 campioni della stessa bundle**, non un metodo
migliore, e va dichiarato così ovunque compaia.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **H18 non ha un criterio di morte scritto.** Il file fissa la misura e le regole di pubblicazione; il report la conta fra le voci senza criterio.
2. **Descrizione per modello del 1/9.** I log registravano solo minimo e massimo dello score per modello: per gemma e DeepSeek, con due item, sono tutti gli item; per gpt-oss-120b sono 2 su 9 e per Olmo-3-7B-Instruct 2 su 4. «Punteggi identici per item» non vale per gpt-oss, e i «due stati distinti» di Olmo sono un minimo (report, Appendice D, voce 14).
