# Esito del pilota Colab free — 30 agosto 2026

Notebook: `AIMO_pilota_E6.ipynb` sul Drive di Daniel, eseguito nel suo Chrome su runtime
Colab free (Tesla T4, DeepSeek-R1-0528-Qwen3-8B quantizzato 4-bit nf4, compute fp16).

## Cosa ha funzionato

- **La pipeline regge end-to-end.** Scarica i dati pubblici, trova **138 coppie etichettate**
  del modello in cache, carica il modello, genera, estrae le feature di incertezza.
- **Il fix del chat template è confermato.** La via `apply_chat_template(tokenize=False)` +
  tokenizer funziona su transformers 5.x, dove `return_tensors="pt"` diretto falliva con
  l'AttributeError visto nel container.
- Load del modello: 241 s (download 16,4 GB incluso, una tantum per sessione).

## I due numeri che decidono

**1. Throughput T4 4-bit: 10,3 token/s** (stabile su 4 problemi: 10,1 / 10,3 / 10,4).

Confronto con il container di gara, entrambi OSSERVATI:

| ambiente | tok/s | rapporto |
|---|---|---|
| Colab free, T4, 4-bit | **10,3** | 1x |
| container gara, RTX PRO 6000, bf16 | **34,4** | 3,3x più veloce |

Passata greedy completa su 138 coppie a 4096 token:

| ambiente | ore |
|---|---|
| Colab free T4 | **15,2** |
| A100 40GB bf16 | 2,9 |
| container gara | 4,6 |

**Colab free si scollega intorno alle 4 ore. La passata greedy non ci sta, per un fattore 4.**

**2. A 768 token il modello non chiude il ragionamento: 4 risposte su 4 vuote.**
DeepSeek-R1 è un reasoning model e satura il cap senza mai emettere `\boxed{}`. Questo
conferma dall'esterno perché la baseline ufficiale usa 4096 token, e quindi perché costa
119 s per problema nel container, e quindi il tetto dei ~30 problemi prima del timeout.

## Verdetto, dichiarato

**Colab free non è sufficiente per E6**, e non per un dettaglio: manca un fattore 4 sul
tempo di sessione, e in più la quantizzazione 4-bit renderebbe le feature non deployabili
(il container gira in bf16). Il pilota è servito esattamente a stabilirlo con una misura
invece che con una stima, al costo di zero euro e zero submission.

## Costo del run vero, dai throughput misurati

Run completo H10 + H11 (greedy 4096 + 4 campioni self-consistency + 2 perturbazioni nostre,
cap 1024): **7,1 ore su A100 bf16**.

| prezzo orario | costo |
|---|---|
| 1,10 $/h | 7,9 $ |
| 2,10 $/h | **15,0 $** |
| 3,20 $/h | 22,8 $ |

Restano validi i confini di PREREG_E6: bf16 obbligatorio per fedeltà al container, CV
raggruppata per problema, seed 20260829 fissato prima, criterio di morte appaiato.
