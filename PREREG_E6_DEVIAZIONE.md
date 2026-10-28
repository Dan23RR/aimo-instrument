# Deviazione dichiarata da PREREG_E6 — 30 agosto 2026, prima del run vero

Il pilota bf16 su A100 (Modal, app `ap-x7jHRty89aOHrp3ul5OK4A`, 6 problemi) ha prodotto due
misure che rendono H11 **non eseguibile come pre-registrata**. Dichiaro la deviazione qui,
prima di spendere credito, invece di riscrivere la pre-registrazione.

## Le misure che forzano il cambio

**1. Throughput reale: 23,0 token/s su A100-40GB bf16, 169 s per problema al cap di 4096.**
Avevo stimato 55 tok/s. La stima era sbagliata di un fattore 2,4.

| ambiente | tok/s | fonte |
|---|---|---|
| Colab free T4 4-bit | 10,3 | OSSERVATO 30/8 |
| **A100-40GB bf16** | **23,0** | **OSSERVATO 30/8** |
| container di gara (RTX PRO 6000) | 34,4 | OSSERVATO submission 907677 |

Costo della sola passata greedy su 164 coppie: **7,7 ore ≈ 16 $**.
Costo di H11 come pre-registrata (4 campioni self-consistency + 2 perturbazioni, ognuna
una generazione intera): **circa 4,5 volte tanto, 35 ore ≈ 73 $**. Fuori dai 30 € disponibili.

**2. Il modello spesso non chiude il ragionamento entro 4096 token: 3 risposte su 6.**

| problem_id | robust | token generati | risposta |
|---|---|---|---|
| 02c9e0 | True | 4096 (cap) | vuota |
| 057f8a | True | 4096 (cap) | vuota |
| 0ef9e9 | True | 4096 (cap) | `1` |
| 04fd05 | False | **2812** | `3` |
| 0507bc | False | 4096 (cap) | vuota |
| 0d430e | False | 4096 (cap) | `100` |

Questo colpisce H11 al cuore: **self-consistency e stabilità sotto perturbazione confrontano
risposte finali, e per metà dei casi la risposta finale non esiste.** Non è un problema di
implementazione: DeepSeek-R1 è un reasoning model e satura il budget di token.

Nota di cautela, contro me stesso: nel pilota i tre `robust` saturano tutti il cap, e verrebbe
voglia di leggerci il segnale "satura → robust". Ma **anche due spurious su tre saturano**.
Con sei punti non si conclude nulla e non ci costruisco sopra.

## H11 riformulata, a costo zero

Sostituisco le feature che richiedono una risposta finale con feature estraibili **dalla stessa
identica generazione greedy**, quindi senza un secondo di GPU in più:

- `n_token` e `satura_cap`: quanto ragionamento serve, e se il budget si esaurisce.
  **La baseline ufficiale non le usa**: le sue 14 feature sono tutte medie per token,
  invarianti alla lunghezza (verificato in `uncertainty_profile/config.py`).
- `ha_risposta`: se il modello arriva a un risultato.
- **trend dell'incertezza**: entropia e logprob medi nella prima metà contro la seconda metà
  della generazione, e la loro differenza. Cattura se il modello si stabilizza o si smarrisce
  mentre ragiona, che una media globale cancella per costruzione.
- `coda_testo`: gli ultimi 200 caratteri generati, salvati **solo** per diagnosi. Sono
  generazioni su problemi del set di sviluppo **pubblico** degli organizzatori, non sul set
  di valutazione: nessun dato di valutazione viene registrato.

**H11-bis:** aggiungere queste feature alle 14 della baseline migliora l'accuratezza in CV
raggruppata per problema, confronto appaiato sugli stessi fold.

**Criterio di morte invariato:** guadagno appaiato medio < 1 errore standard appaiato → morta.
Fold e seed restano quelli pre-registrati (5 fold per `problem_id`, seed 20260829).

## Cosa resta invariato

bf16 obbligatorio, greedy deterministico, cap 4096 come la baseline, CV raggruppata per
problema, nessuna taratura sul set di valutazione, nessuna submission per lo sviluppo.

## Differenza d'ambiente da dichiarare nel writeup

| | Modal | container di gara |
|---|---|---|
| transformers | 5.13.0 | 5.13.0 (identico) |
| torch | 2.13.0+cu130 | 2.12.1+cu132 |
| GPU | A100-40GB | RTX PRO 6000 Blackwell |
| precisione | bf16 | bf16 (identica) |

La differenza di versione torch e di GPU può spostare i valori delle feature ai decimali
bassi. Va detto, e va tenuto presente se l'artefatto addestrato qui venisse spedito là.
