# Pre-registrazione E7 — il sensore, con le feature VERE

Scritta il 31 agosto 2026 **prima** di eseguire `esperimento_e7.py`.

## Perché rifare una domanda già morta

H3, H4 e H5 (il modello in cache come sensore per i modelli ciechi) sono morte, ma tutte e
tre usavano **proxy grossolani**: la label binaria del modello, o la sua `base_accuracy`
tagliata alla mediana. Nessuna usava le feature di incertezza vere, perché non le avevo.

Adesso ce le ho: E6 ha prodotto le 23 feature reali per 164 problemi, e **49 di quei problemi
sono visti anche da altri modelli**, per un totale di **109 coppie (modello bersaglio,
problema)** su 8 modelli.

Questa è l'unica versione della domanda che conta, ed è l'ultima carta che il container
consente: se le feature di incertezza del modello in cache predicono la robustezza di un
modello che NON possiamo caricare, allora i 15 item ciechi su 17 diventano attaccabili.

## Ipotesi

**H12.** Un classificatore addestrato sulle feature di incertezza del modello in cache
predice la robustezza di un **altro** modello sullo stesso problema, meglio della classe di
maggioranza di quel modello bersaglio.

**Validazione:** leave-one-model-out. Il modello bersaglio in test non compare mai in
addestramento, perché nella fase principale i modelli saranno diversi da questi.

**CRITERIO DI MORTE, dichiarato prima:** guadagno medio sulla maggioranza per modello
inferiore a `sqrt(0.25/109)` = **4,8 punti** -> H12 morta, e con lei l'intera idea del
sensore. Nessuna variante verrà cercata dopo aver visto i numeri.

**H13 (tetto).** Lo stesso con soglia oracolo per modello bersaglio: misura se il problema è
la calibrazione o l'assenza di segnale. Se anche il tetto è negativo, il canale non esiste.

## Cosa decide

- Se H12 vive: si spedisce una submission che carica il modello in cache e lo fa girare su
  **tutti** i problemi, anche quelli dei modelli ciechi. Costo stimato 17 x 119 s = 2023 s,
  dentro il limite di 3600 s.
- Se H12 muore: nona ipotesi caduta, e la conclusione che i 15 item ciechi sono irriducibili
  passa da "misurata con proxy" a "misurata con le feature vere". Il che la rende definitiva.

## Esito

Da riempire dopo l'esecuzione.

## Esito, 31 agosto 2026

**H12 MORTA: −16,5 punti** (soglia di morte 4,8). **H13, il tetto, POSITIVO: +7,3 punti.**

Questa combinazione è diversa da tutte le precedenti e va letta bene. In E3 anche il tetto era
negativo, il che significava assenza di segnale. **Qui il tetto è positivo e sopra soglia: il
segnale nelle feature di incertezza vere ESISTE ed è cross-model.** Ciò che manca è la
calibrazione.

Il dettaglio per modello bersaglio lo dimostra:

| bersaglio | sensore | maggioranza | |
|---|---|---|---|
| Qwen3.5-397B | 9/10 | 8 | **+10** |
| gemini-3.1-pro | 9/10 | 8 | **+10** |
| gpt-5.2:low | 6/10 | 5 | **+10** |
| gpt-oss-120b | 7/10 | 6 | **+10** |
| huikang-gpt-oss | 7/10 | 6 | **+10** |
| GLM-5.1 | 6/10 | 5 | **+10** |
| gpt-5.2 (full) | 4/10 | 7 | −30 |
| **Llama-3.1-8B** | **18/39** | **39** | **−54** |

**Sei modelli su otto guadagnano dieci punti.** Il disastro viene da due casi, e uno solo
spiega quasi tutto: `Llama-3.1-8B` ha un tasso di robustezza dello **0%** su 39 coppie, quindi
la sua classe di maggioranza è perfetta e imbattibile, e il sensore, addestrato su modelli al
27% di robust, predice `robust` ventuno volte sbagliando sempre.

È il **problema del tasso di base**, lo stesso che uccise H3, ma stavolta con una differenza
decisiva: non è il segnale a mancare, è la soglia a essere sbagliata per il bersaglio.

**AVVERTENZA aggiunta il 31/8 dopo E8:** la frase qui sopra sul "segnale che esiste" e' SBAGLIATA.
Il tetto di H13 e' gonfiato perche' la soglia oracolo puo' degenerare alla costante (per Llama,
robust 0%, la soglia ottimale e' "non predire mai robust", che prende 39/39 senza usare una
sola feature). Il tetto onesto, vincolato a ordinare e scegliere k, e' **negativo**: vedi E8.

**Conseguenza:** H12 resta morta come dichiarato, e nessuna variante viene cercata per farla
passare. Ma il tetto positivo apre una domanda nuova e legittima, che va pre-registrata a
parte: **una calibrazione realizzabile senza oracolo cattura quei +7,3 punti?** Vedi E8.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Meccanismo di H13.** Il +7,3 non viene dalla possibilita' di degenerare nella costante. Escludendo le soglie che danno predizioni costanti l'oracolo guadagna ancora +5,5 punti (90 contro 84 su 109), sopra la soglia di 4,8; tutti gli 8 item di guadagno vengono da soglie non costanti scelte sugli stessi item su cui sono valutate. L'opzione costante evita solo due perdite (gpt-5.2 e Llama). `vm/warmup_ricalcoli.py`.
2. **Split per bersaglio, non per problema.** 70 delle 109 coppie di test hanno il loro problema, con lo stesso vettore di feature DeepSeek, anche in addestramento (Llama 0 su 39, gli altri sette bersagli 10 su 10). `vm/controlli_verifica3.py`.
