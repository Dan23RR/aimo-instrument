# Pre-registrazione E9 — calibrazione per-modello con perturbazioni NOSTRE

Scritta il 2 settembre 2026, prima di qualunque run. Fase principale, 21 item (Main) e
19 (Small), modelli: Qwen/Qwen3.5-4B, Skywork/Skywork-OR1-Math-7B, allenai/Olmo-3-7B-Think,
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B, openai/gpt-oss-120b (solo Main).

## Il problema esatto

La soglia di v6 (0,576) è ereditata dall'artefatto DeepSeek e applicata a tre modelli mai
visti, senza un solo item etichettato pubblico. Effetto visibile nei log: Olmo-Think in Main
ha punteggi tutti in [0,622, 0,709] e viene predetto **tutto spurious** in blocco. Non è una
predizione, è la soglia sbagliata. In classifica siamo tutti, organizzatore incluso, alla
classe di maggioranza (11/21, 10/19).

## La regola che lo permette, verbatim

> "teams may not use extra labeled data for the same perturbation types as in our final
> validation set; however, unlabeled data and **labeled data for other perturbations are
> allowed**, within the bulk evaluation limit of 20min/sample."

Le famiglie LORO: distract, domain, rename, rephrase, typos, expert_no_solution,
expert_perturbations, paraphrase. Le famiglie NOSTRE, tutte meccaniche, deterministiche,
che preservano la risposta e non toccano il contenuto matematico:

| famiglia | cosa cambia | cosa non cambia |
|---|---|---|
| `format` | una frase per riga, spazi normalizzati | ogni parola e simbolo |
| `latex_delims` | `$...$` → `\(...\)` | il LaTeX interno |
| `framing` | istruzione neutra prima e dopo il testo | il testo |
| `redundancy` | il testo ripetuto dopo "Restated:" | il testo |
| `question_first` | la frase-domanda spostata in testa (solo se separabile) | le frasi |

Nessuna coincide con una famiglia loro: non riscriviamo, non rinominiamo, non distraiamo,
non introduciamo errori, non cambiamo dominio. Nessun LLM genera le perturbazioni: zero
contaminazione, riproducibilità totale.

## La definizione di etichetta nostra, e perché è confrontabile

Loro: robust ⇔ nessuna perturazione fa calare l'accuratezza. Sul test, filtrato a problemi
che il modello risolve (E5), questo equivale a: **la risposta resta quella giusta sotto ogni
perturbazione**. La nostra: robust_ours ⇔ la risposta finale (`\boxed{}`) è identica
all'originale sotto **tutte** e cinque le famiglie, e l'originale ha una risposta. Problemi
senza risposta all'originale: esclusi (etichetta indefinita, come i "banali" di E5).
Risposta persa sotto una perturbazione: spurious.

## Due stadi, il secondo condizionato al primo

### Stadio 1 — il proxy è valido? (solo DeepSeek, dove abbiamo la verità)

164 coppie con etichetta loro (E6) e con i punteggi del regressore già estratti (parquet E6).
Generiamo l'etichetta nostra sulle stesse 164 con vLLM (5 famiglie + originale, greedy,
4096 token).

**H19.** L'accordo fra etichetta nostra e loro supera la classe di maggioranza di almeno
un errore standard, e kappa di Cohen ≥ 0,30.
**H20.** Una soglia calibrata sull'etichetta NOSTRA (sui punteggi E6, CV raggruppata per
problema) predice l'etichetta LORO meglio della soglia ereditata 0,576.
**Morte:** se H19 o H20 muore, il proxy non vale e lo stadio 2 **non si fa**. Si dichiara.

### Stadio 2 — calibrazione dei modelli nuovi (solo se H19 e H20 vivono)

Per Qwen3.5-4B, Skywork, Olmo-Think, DeepSeek: etichette nostre via vLLM sui problemi
pubblici (fino a 174), feature del regressore via estrazione HF identica al container.
Soglia per modello = massima accuratezza sull'etichetta nostra, ties verso 0,576, come in v6.
gpt-oss-120b resta calibrato sui 10 item val-sample (218 GiB, fuori budget per ora).

**H21.** v7 (soglie per-modello) batte v6 sui set della gara.
**Come si misura, dichiarato prima:** NON dalla classifica. Con 21 e 19 item l'errore
standard è di 11 punti e la regola `Force_Latest_Multiple` sovrascrive l'entry a ogni invio.
Il test di H21 è **offline**: su DeepSeek, la soglia calibrata sull'etichetta nostra deve
battere 0,576 contro l'etichetta loro in CV. La submission finale è **una sola per traccia**,
spedita solo se il test offline passa, e il suo punteggio viene **riportato, non ottimizzato**.
Nessun secondo invio "per vedere".

## Cosa non si fa

Nessuna costante, nessuna sonda, nessuna taratura su ciò che la classifica restituisce.
Nessun testo di problema nei log del container. Nessuna famiglia che assomigli alle loro.

## Budget

vLLM è 10-30× più veloce di HF generate. Stadio 1: 164 × 6 generazioni ≈ 1,5 M token
≈ 15-25 min di A100 ≈ **1-2 $**. Stadio 2: 4 modelli × ~174 × 6 ≈ 1 h vLLM + ~2,5 h di
estrazione HF ≈ **8-12 $**. Tutto dentro i crediti Modal residui.

## Esito

Da riempire per stadio, senza riscrivere nulla di quanto sopra.

## ESITO STADIO 1 — 2 settembre 2026, ore 15:50. H19 MORTA, H20 MORTA. Stadio 2 NON si fa.

Run Modal ap-UEkRezYjFHZka67edaxrmK: DeepSeek-R1-0528-Qwen3-8B, 174 problemi × 6 varianti,
vLLM greedy 4096 token, 1197 s, ~1 $. Con risposta: 105/174 (il resto satura il cap senza
`\boxed{}`). Incrocio con le 164 etichette loro (E6): **99 coppie** con entrambe definite.

| | |
|---|---|
| robust LORO (famiglie semantiche) | **21,2%** |
| robust NOSTRA (famiglie meccaniche) | **77,8%** |
| H19 accordo | 0,354 contro maggioranza 0,788 → **−43,4 punti**, kappa **0,02** |
| H20 soglia tarata su NOSTRA → accuratezza su LORO | **0,212** contro 0,788 ereditata → **−57,6 punti** |

Stabilità per famiglia: format 92%, latex_delims 93%, framing 92%, redundancy 93%,
question_first 86% (n=29). Le nostre perturbazioni sono troppo deboli: il modello le
attraversa quasi sempre, l'etichetta nostra è quasi costante (78% True) e porta zero
informazione su quella loro (kappa 0,02 ≈ indipendenza). Peggio: tararci sopra una soglia
inverte la decisione (0,21) perché le poche instabilità nostre non coincidono con le loro.

**Lettura che vale per il writeup:** la robustezza che la gara premia è robustezza a
perturbazioni *semantiche* (distrattori, riformulazioni, cambio di dominio, errori
tipografici). Le perturbazioni di sola *forma* non la misurano affatto. Il regolamento vieta
proprio le famiglie che misurerebbero la cosa giusta: la porta "labeled data for other
perturbations" è aperta di diritto e chiusa di fatto, almeno per famiglie meccaniche.
Un E10 con famiglie semantiche *diverse* dalle loro (es. traduzione in altra lingua) resta
concepibile ma richiede un LLM che generi le varianti, cioè contaminazione e costo, e va
pre-registrato a parte. Non si lancia senza deciderlo esplicitamente.

Bilancio complessivo: **diciotto ipotesi pre-registrate, tredici morte.** Le entry in
classifica restano v6 (912096 Main 11/21, 912097 Small 10/19). Nessun nuovo invio.
