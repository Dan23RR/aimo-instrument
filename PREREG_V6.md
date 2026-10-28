# Pre-registrazione v6 — soglia calibrata per gpt-oss sui dati pubblici

Scritta il 31 agosto 2026, dopo il risultato v5 (10/17) e PRIMA di costruire v6.

## L'ipotesi H17

La soglia di v5 (0.576) e' presa in prestito dall'artefatto DeepSeek e applicata a tutti.
Per gpt-oss-120b (9 item su 17, punteggi osservati 0.340-0.870 a cavallo della soglia) una
soglia calibrata sul comportamento di gpt-oss stesso dovrebbe classificare meglio.

**H17:** v6 (soglia gpt-oss calibrata su dati pubblici etichettati) > v5 (10/17).
**Criterio di morte:** v6 <= 10/17 -> H17 morta; si tiene v5 pubblicata e NON si provano
altre varianti di soglia (diventerebbe tuning iterativo sul canale del punteggio).

## I dati di calibrazione, e i loro limiti dichiarati

10 item etichettati dal val-sample PUBBLICO degli organizzatori, spediti dentro il bundle:
- gpt-oss-120b:low - 7 item, 0 robust
- huikang-gpt-oss-120b-aimo3:low (finetune dello stesso base) - 3 item, 1 robust

Limiti, tutti dichiarati prima: (a) 10 item sono pochi e sbilanciati (1R/9S); (b) le
etichette sono del profilo :low e di un finetune, applicate al checkpoint base: transfer
approssimato; (c) il val-sample usa le famiglie di perturbazione expert, che sono quelle del
test finale: e' il miglior proxy disponibile del regime del test nascosto.

Nota onesta: sotto le famiglie expert gpt-oss risulta quasi mai robust (1/10), quindi la
calibrazione spingera' verso "spurious" sul batch gpt-oss. Se il test nascosto e' ricco di
robust, v6 puo' peggiorare v5. Non uso quell'informazione (verrebbe dal punteggio del test):
la regola di pubblicazione protegge il ribasso.

## La regola di taratura, fissata ora

A runtime, col modello gpt-oss caricato: estrai le feature per i 10 item di calibrazione
CON LO STESSO modello e la stessa configurazione degli item di valutazione. Candidati di
soglia = punti medi fra i punteggi ordinati dei 10, piu' la soglia ereditata 0.576. Scegli
quella che massimizza l'accuratezza sui 10; a parita', la piu' vicina a 0.576. Applicala
solo al batch gpt-oss. Gli altri modelli restano come in v5 (nessun dato per calibrarli:
Olmo e gemma hanno zero item etichettati pubblici).

## Budget

Costo misurato in v5: 21-59 s/item per batch, 619 s totali. v6 aggiunge 10 generazioni
gpt-oss (~33 s/item misurato) ~ +330 s -> ~950 s su 3600. La guardia di budget salta la
calibrazione se il tempo residuo non copre (10 + n) item, ricadendo su v5 esatta.

## Regola di pubblicazione

Privata; pubblicata SOLO se > 10/17. Altrimenti resta v5.

## ESITO — 909976: **0.6471 = 11/17. H17 SOPRAVVIVE.**

La soglia calibrata sui 10 item pubblici e' uscita **0.563** (ereditata: 0.576) e sul batch
gpt-oss ha guadagnato +1 rispetto a v5. Pubblicata secondo la regola (>10/17).

**Classifica: quarti fra i punteggi reali**, nel gruppo a 0.65 con [REDACTED: two other participants], dietro [REDACTED: two other participants and their scores]. Runtime 785 s su 3600, coverage 1, invalid 0.

## Scoperta collaterale, che spiega anche l'anomalia v4

Confrontando i log v5/v6 a parita' di codice e input: i punteggi del regressore CAMBIANO
fra run (DeepSeek max 0.722 -> 0.751; gpt-oss max 0.870 -> 0.902). **L'estrazione di feature
non e' deterministica fra run** (kernel GPU / batching in bf16), quindi gli item con
punteggio vicino alla soglia possono flippare da una run all'altra.

Questo (a) spiega la falsificazione della previsione v4 senza bisogno di ipotesi esotiche:
il "+1 deterministico" di DeepSeek non era deterministico; (b) e' un finding da writeup:
il benchmark ha rumore di misura INTERNO alla singola submission, non solo statistico;
(c) implica che parte dei distacchi in classifica sta dentro il rumore di riesecuzione.

Due punte pre-registrate rimangono aperte per la fase principale: quantificare la varianza
di riesecuzione (stessa bundle piu' volte) e calibrare anche Olmo/gemma quando gli
organizzatori pubblicheranno le attivazioni o nuovi dati etichettati.
