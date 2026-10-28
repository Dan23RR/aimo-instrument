# Pre-registrazione E1 — esiste segnale nel solo testo del problema?

Scritta il 29 agosto 2026 **prima** di far girare `esperimento_e1.py`. Nessun numero di
questo esperimento è stato ancora osservato.

## Perché questa è la domanda che decide tutto

Dalla submission 907310 (OSSERVATO): il set di valutazione warmup ha 17 item su 4 modelli,
e **solo 2 item su 17 stanno su un modello caricabile nel container**. Per i restanti 15 non
esistono attivazioni, logit o probabilità dei token. Restano due cose sole: la stringa
`model_id` e il testo del problema.

Quindi la domanda non è "quale probe di interpretabilità costruiamo". È: **il testo del
problema, da solo, predice se un modello sarà robusto su di esso?** Se sì, si può competere
sull'88% cieco del compito. Se no, l'88% è irriducibilmente casuale e va detto.

## Il vincolo di regolamento, verbatim

> "Data: teams may not use extra labeled data for the same perturbation types as in our
> final validation set; however, unlabeled data and labeled data for other perturbations
> are allowed... Any attempts for leakage of evaluation data will result in the disqualification."

Ne seguono due confini che accetto in anticipo:

1. **Non generiamo dati etichettati nostri** con le stesse famiglie di perturbazione. Usiamo
   solo i dataset che gli organizzatori hanno pubblicato loro stessi su Hugging Face, sui
   quali le loro stesse baseline si addestrano.
2. **Nessun'altra sonda sulla leaderboard.** Recuperare le etichette del set nascosto per
   differenza fra submission ricadrebbe plausibilmente sotto "attempts for leakage of
   evaluation data". La sonda del pavimento già spedita ha registrato solo conteggi e
   `model_id`, mai testo, e resta l'unica. La linea è chiusa qui, per scelta.

## I dati

335 coppie (modello, problema) su 174 problemi distinti e 10 identificativi di modello,
ottenute aggregando i sei dataset pubblici `aimo-interp` e derivando la label con la regola
`n_detrimental_permutations == 0`, verificata al 100% su 262 coppie con label dichiarata.

## Ipotesi e criterio di morte

**H1.** Un classificatore che usa **solo feature del testo del problema** (nessun accesso al
modello) predice `robust` meglio della classe di maggioranza, in validazione incrociata
raggruppata per problema (nessun problema appare sia in train sia in test).

**Metrica:** accuratezza in CV raggruppata a 5 fold, gruppi = `problem_id`.

**Baseline da battere:** la classe di maggioranza sullo stesso split.

**CRITERIO DI MORTE, dichiarato prima:** H1 è **morta** se il guadagno sulla maggioranza è
inferiore a **una deviazione standard** dell'accuratezza stimata, cioè se
`acc − maggioranza < sqrt(0.25/n)`. Con n = 335 la soglia è **2,7 punti percentuali**.
Se H1 muore, non si cerca una variante che passi: si dichiara che il testo non porta segnale
e il metodo si riduce al prior per modello, dichiarandolo.

**H2 (secondaria).** L'identità del **problema** trasferisce fra modelli: sui problemi visti
da almeno due modelli, la label concorda su tutti i modelli più spesso di quanto farebbe il
caso dato il tasso marginale. Osservato in aggregazione: 56/73 = 77% di concordanza. H2 verrà
testata formalmente contro il modello nullo di indipendenza.

## Cosa NON viene fatto in questo esperimento

- Nessuna taratura di soglia sul set di valutazione della gara.
- Nessuna selezione di feature guidata dal punteggio di leaderboard.
- Nessun modello caricato: E1 è per costruzione un esperimento a costo zero di GPU.

## Esito, 29 agosto 2026

Eseguito `esperimento_e1.py` una volta sola, senza modificare feature o soglie dopo aver
visto i numeri. 335 coppie, 174 problemi, 32 feature di testo.

### H1 è MORTA

| | |
|---|---|
| accuratezza del modello di testo, CV raggruppata per problema | **0,6746** |
| accuratezza della classe di maggioranza sullo stesso split | **0,7612** |
| guadagno | **−8,7 punti** |
| soglia di morte pre-registrata | +2,7 punti |

Il testo del problema non solo non aiuta: **fa peggio della maggioranza di quasi nove punti.**
Il classificatore impara struttura superficiale che non trasferisce a problemi mai visti.
Come dichiarato prima dell'esecuzione, non si cerca una variante che passi. H1 resta morta.

Questo è un risultato, non un fallimento: dice che sull'88% cieco del set di valutazione non
esiste una scorciatoia testuale, e chiunque dichiari il contrario ha probabilmente validato
con uno split casuale invece che raggruppato per problema.

### H2 SOPRAVVIVE, e forte

| | |
|---|---|
| problemi visti da ≥2 modelli | 73 |
| con label concorde su tutti i modelli | **56 (77%)** |
| attesi sotto indipendenza | 36,1 (49%) |
| eccesso | **+19,9 problemi, z = 4,65** |

E la misura di quanto varrebbe: il voto di maggioranza degli **altri** modelli sullo stesso
problema, in leave-one-out, predice al **0,8162** contro lo 0,7612 della maggioranza globale.
**+5,5 punti**, ed è l'unico segnale trasferibile trovato.

### Cosa ne segue per il metodo

La robustezza è una proprietà **del problema** molto più che del modello, e non è leggibile
dal testo. Quindi l'unica leva reale è l'identità del problema, cioè: riconoscere un problema
già visto nei dati pubblici e usare il comportamento degli altri modelli su di esso.

Il pool pubblico `aimo-2025-reference`, lo stesso su cui la gara è costruita, contiene però
solo **10 problemi distinti** su 8 modelli. Se i 17 item nascosti non intersecano quei 10, il
segnale è inutilizzabile e resta solo il prior per modello.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **H2 e le righe duplicate.** Le 52 coppie DeepSeek sono pubblicate due volte, come `qwen3-8b:low` e con il nome DeepSeek, con etichetta e base_accuracy identiche. Contate una volta sola, H2 muore anche con il test originale di questo file (nullo a tasso globale, vive se z > 2): 49 problemi multi-modello, 32 concordi contro 25,7 attesi, z = 1,81 (con le righe contate due volte, z = 4,65). Inoltre la concordanza 56 su 73 era gia' osservata quando H2 fu scritta qui. `vm/warmup_ricalcoli.py`.
2. **H1 e le righe duplicate.** Sulle 283 coppie distinte H1 fa 0,682 contro 0,756, -7,4 punti con soglia 3,0 (sulle 335: -8,7). Il verdetto non cambia.
3. **H1 senza raggruppamento.** Con uno split casuale a 5 fold, stesso codice e stessi dati, H1 resta sotto la maggioranza (-0,9, -0,3 e -2,1 punti su tre semi, soglia 2,7): il raggruppamento ha approfondito la perdita ma non l'ha causata. `vm/warmup_ricalcoli.py`.
