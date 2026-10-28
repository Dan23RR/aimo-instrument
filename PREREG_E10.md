# Pre-registrazione E10 — robustezza alla TRADUZIONE come proxy semantico

Scritta il 2 settembre 2026, prima di qualunque run. Segue la morte di E9 (perturbazioni
di sola forma: kappa 0,02 con le etichette della gara).

## Perché la traduzione

La lezione di E9 è che la robustezza premiata è a perturbazioni **semantiche**: distrattori,
riformulazioni, cambio di dominio, errori tipografici, tutte famiglie che il regolamento
vieta di usare per etichettare. La traduzione in un'altra lingua è una perturbazione
semantica **che non compare** nella loro lista: la matematica resta identica, cambia tutto
il linguaggio naturale che la veicola. Se un modello risolve un problema per comprensione
della struttura, la traduzione non lo tocca; se lo risolve per appigli superficiali del
testo inglese, la traduzione li rimuove. È esattamente la distinzione robust/spurious.

## Il traduttore, scelto prima

Deve essere (a) esterno ai cinque modelli valutati, (b) di famiglia diversa (niente Qwen,
niente gpt-oss, niente Olmo, niente Skywork), (c) non gated su Hugging Face, perché il
container Modal scarica senza token. Ordine di preferenza fissato ora, si usa il primo
che soddisfa (c): `microsoft/phi-4` (14B, MIT, testuale), poi
`mistralai/Mistral-Small-3.2-24B-Instruct-2506`. Il traduttore scelto viene registrato in
`gpu/out/e10_traduttore.txt` prima del run.

Lingue: **italiano, francese, tedesco**. Prompt che impone di conservare ogni formula,
comando LaTeX, numero e nome di variabile, e di non risolvere.

## Cancello di qualità, deterministico

Una traduzione è accettata solo se conserva **esattamente** il multiinsieme dei segmenti
LaTeX inline `$...$` e delle cifre dell'originale. Se la traduzione tocca la matematica non
è logic-preserving e viene scartata. Il tasso di accettazione per lingua viene riportato.

## Etichetta

`robust_trad` = l'originale ha una risposta finale (`\boxed{}`) **e** ogni traduzione
accettata produce la **stessa** risposta. Traduzione senza risposta = instabile.
Originale senza risposta = etichetta indefinita, escluso (come in E5 ed E9).

## Stadio 1 — il proxy vale? (DeepSeek, dove abbiamo la verità)

**H22.** L'accordo fra `robust_trad` e l'etichetta loro supera la classe di maggioranza di
almeno un errore standard, e kappa di Cohen ≥ 0,30.
**H23.** Una soglia tarata su `robust_trad` (punteggi del regressore E6, CV raggruppata per
problema) predice l'etichetta loro **meglio** della soglia ereditata 0,576.
**Morte:** se H22 o H23 muore, lo stadio 2 **non si fa** e lo si dichiara.

## Stadio 2 — solo se H22 e H23 vivono

Per Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think, DeepSeek: etichette `robust_trad`
via vLLM, punteggi del regressore via estrazione HF identica al container
(`e9_feature_modal.py`), soglia per modello con la regola di v6/E9. gpt-oss-120b resta
calibrato sui 10 item val-sample. Bundle v8 = v7 con la tabella `soglie_per_modello.json`.

**H24.** v8 batte v6. **Si misura offline** (H23 su DeepSeek), non in classifica: 21 e 19
item, errore standard 11 punti, e la regola `Force_Latest` sovrascrive l'entry a ogni invio.
Una sola submission per traccia, solo se lo stadio 2 è completo e validato, e il punteggio
si riporta, non si ottimizza.

## Cosa non si fa

Nessuna costante, nessuna sonda, nessuna taratura sui punteggi di classifica, nessun testo
di problema nei log del container, nessun traduttore della famiglia dei modelli valutati.

## Budget

Fase A (traduzione, 174 × 3 lingue) ≈ 10 min A100-80GB ≈ 1 $. Stadio 1 (DeepSeek,
originale + ≤3 traduzioni) ≈ 20 min A100-40GB ≈ 1 $. Stadio 2 ≈ 4 modelli × 20 min +
estrazione HF ≈ 10-15 $. Totale sotto i 20 $ di crediti Modal.

## Esito

Da riempire per stadio, senza riscrivere nulla di quanto sopra.

## Esito fase A — 2 settembre 2026

Traduttore: `microsoft/phi-4` (primo non gated nell'ordine fissato). 174 problemi × 3 lingue
= 522 traduzioni in 14 s di generazione (vLLM, A100-80GB, ~1 $).
Cancello di qualità (formule LaTeX e cifre conservate esattamente): **it 75,3%, fr 52,9%,
de 40,8%**. La diagnosi delle cause di scarto è riportata nel log della sessione; il
cancello NON viene allentato a posteriori. Stadio 1 lanciato sulle sole traduzioni accettate.

## ESITO STADIO 1 — 2 settembre 2026. H22 MORTA, H23 MORTA. Stadio 2 NON si fa.

DeepSeek su originale + traduzioni accettate: 428 generazioni in 365 s (vLLM, ~1 $).
134 problemi con almeno una traduzione pulita, 82 con risposta all'originale; incrocio con
le etichette loro (E6): **77 coppie** con entrambe definite.

| | |
|---|---|
| robust LORO | 26,0% |
| robust TRADUZIONE | 39,0% |
| H22 accordo | 0,507 contro maggioranza 0,740 → **−23,4 punti**, kappa **−0,10** |
| H23 soglia tarata su traduzione → accuratezza su LORO | 0,740 = ereditata 0,740 → **+0,0** |

Stabilità per lingua: it 51%, fr 43%, de 40%. Il dato che chiude la questione: fra gli item
che la traduzione **rompe**, il 30% è robust per loro; fra quelli che la traduzione **non
rompe**, il 20%. Cioè la stabilità alla traduzione è, se mai, leggermente **anti**-correlata
con la robustezza che la gara misura. Sono assi diversi: un modello può reggere il cambio
di lingua e cadere su un distrattore, o viceversa.

**Conclusione delle due prove (E9 forma, E10 traduzione):** la porta "labeled data for
other perturbations" del regolamento è aperta di diritto e chiusa di fatto. Le uniche
perturbazioni che predicono l'etichetta della gara sono quelle della gara, e quelle non si
possono usare. Non è più un'ipotesi: sono due esperimenti pre-registrati con verità a terra.

Bilancio complessivo: **venti ipotesi pre-registrate, quindici morte.** In classifica
restano le v6. Nessun invio. Spesa E9+E10 complessiva: circa 4 $.
