# Pre-registrazione E3 — trasferisce l'ORDINE anche se non trasferisce il LIVELLO?

Scritta il 29 agosto 2026 **prima** di far girare `esperimento_e3.py`.

## Perché H3 e H4 sono morte, e perché questa è una domanda diversa

H3 e H4 usavano un modello A come sensore per predire il modello B **in valore assoluto**:
la label di A, oppure la sua `base_accuracy` sopra o sotto la mediana. Sono morte, e il
motivo è nei dati: i tassi di robustezza per modello vanno dal 20% al 70%, quindi un sensore
usato in valore assoluto **trasporta il proprio tasso di base** e sbaglia sistematicamente
sui modelli che non gli somigliano.

Ma la difficoltà di un problema potrebbe trasferire **come ordinamento** anche quando non
trasferisce come livello. Se il problema P è più difficile del problema Q per il modello A,
è plausibile che lo sia anche per B, pur avendo B una soglia completamente diversa.

Questa è una domanda che H3 e H4 **non hanno testato**, e non è una variante cercata dopo
aver visto i numeri: è la conseguenza diretta della spiegazione del loro fallimento.

## Perché è realizzabile a runtime

La gara chiama `are_robust(model_id, problems)` un lotto per modello. Dentro un lotto
possiamo **ordinare** i problemi con il modello in cache (l'unico caricabile) e predire
`robust` la metà più facile. L'ordinamento è immune al disallineamento dei tassi di base,
perché non usa mai una soglia assoluta.

Costo misurato: **119 secondi per problema** (OSSERVATO, submission 907677). Su 17 item
sono 2023 secondi, dentro il limite di 3600. Sulla fase principale sarebbe il vincolo che
lega tutto.

## Ipotesi

**H5.** Ordinando i problemi con la `base_accuracy` del modello A e predicendo `robust` la
metà superiore, si predice il modello B meglio della sua classe di maggioranza.

**H6 (tetto).** Lo stesso, ma con il numero corretto di positivi noto (oracolo su k). Misura
quanto varrebbe l'ordinamento se la calibrazione fosse perfetta. Non è utilizzabile: serve
solo a separare l'errore di ordinamento dall'errore di soglia.

**CRITERIO DI MORTE, dichiarato prima:** H5 è morta se il guadagno medio sulla maggioranza
è inferiore a `sqrt(0.25/n)` sul numero di predizioni valutate. Se muore, la conclusione è
che su questa istrumentazione i 15 item ciechi sono irriducibili, e lo si dichiara invece di
cercare una settima variante.

## Esito, 29 agosto 2026

| ipotesi | ordinamento | maggioranza | guadagno | soglia |
|---|---|---|---|---|
| H5, soglia = metà del lotto (utilizzabile) | 0,4089 | 0,6625 | **−25,4 punti** | +2,1 |
| H6, soglia = numero vero di positivi (**tetto**) | 0,5321 | 0,6625 | **−13,0 punti** | +2,1 |

**H5 morta, e il tetto H6 è negativo.** Quest'ultimo è il risultato che chiude la porta:
anche con la soglia perfetta, ordinare i problemi con la `base_accuracy` di un modello
predice un altro modello **peggio** della sua classe di maggioranza. Non è un problema di
calibrazione: l'ordinamento stesso non porta informazione utilizzabile.

Il modello in cache è il quinto sensore su otto, a −24,3 punti. Nessun sensore è positivo.

## Correzione a H2, dal controllo avversario

E3 era in tensione con H2, quindi ho riesaminato H2 cercando di ucciderla
(`controllo_h2.py`). Il nullo che avevo usato era **sbagliato**: usava un unico tasso globale,
mentre i tassi per modello vanno dal 20% al 70%, e due modelli entrambi al 20% concordano
spesso senza che il problema c'entri.

| | valore |
|---|---|
| concordanza osservata | 56/73 = 77% |
| attesa, nullo con tasso globale (**quello che avevo usato**) | 50% |
| attesa, nullo con tasso per modello (**corretto**) | 56% |
| eccesso sul nullo corretto | +15,3 problemi, **z = 4,04** |

H2 sopravvive, ma più debole di quanto dichiarato. E il guadagno va ricalcolato contro la
baseline giusta, che non è la maggioranza globale ma **la maggioranza del modello bersaglio**,
nota a runtime dal solo `model_id`:

| | |
|---|---|
| voto degli altri modelli sul problema | 0,8162 |
| maggioranza per modello | 0,7821 |
| **guadagno vero** | **+3,4 punti**, soglia di significatività 3,3 |

**Il canale-problema vale +3,4 punti, non +12,5 come avevo scritto in E2.** Quel numero era
misurato su un sottoinsieme più piccolo e contro una baseline che ignora il `model_id`.
Sopravvive di un decimo di punto sopra la soglia.

## Conclusione, chiusa

Cinque ipotesi pre-registrate, cinque morti. Sopravvive solo l'aggregazione multi-modello,
che vale **+3,4 punti** e richiede accesso a più modelli sullo stesso problema.
Il container ne concede uno.

> Su questa istrumentazione i 15 item su 17 che stanno su modelli non caricabili sono
> **irriducibili**. Non perché il compito sia difficile, ma perché l'unico canale che porta
> segnale richiede ciò che il container nega.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **H2 con le righe duplicate contate una volta.** Il nullo per modello e la soglia sulla maggioranza per modello, introdotti qui dopo la prima esecuzione di E1, danno z = 0,88 e -1,3 punti (soglia 4,0) quando le 52 coppie DeepSeek pubblicate con due nomi si contano una volta sola (`controllo_h2_alias_2026-09-21.py`). H2 muore per il proprio criterio.
