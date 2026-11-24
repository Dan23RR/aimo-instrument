# OSSERVATO — fase principale, prime due submission (2 settembre 2026)

Submission 912096 (Main Track, `solution-v6-main.zip`) e 912097 (Small Model Track,
`solution-v6-small.zip`), stesso codice del warmup v6 con/senza marker `small.txt`.
Log: solo conteggi, tempi e `model_id`; nessun testo di problema.

## Composizione dei set (dai lotti passati a `are_robust`)

| modello | Main (21) | Small (19) | nel README ufficiale? |
|---|---|---|---|
| Qwen/Qwen3.5-4B | 3 | 3 | **no** |
| Skywork/Skywork-OR1-Math-7B | 5 | 6 | **no** |
| allenai/Olmo-3-7B-**Think** | 4 | 6 | no (README cita l'Instruct) |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 4 | 4 | sì |
| openai/gpt-oss-120b | 5 | 0 | sì |

gemma-3-27b, presente nel warmup e nel README, **non compare** in nessuna traccia.
La Small e' la Main meno gpt-oss, con piu' item su Skywork e Olmo. Set piccoli come nel
warmup: SE binomiale **10,9 punti** (Main) e **11,5 punti** (Small). Le dimensioni coincidono
con i conteggi di invalid (21 e 19) delle due entry incrociate dell'organizzatore.

## Esecuzione

Tutti i modelli caricati dalla cache in 1,5-5,8 s (anche i tre non documentati), **tutti gli
item misurati col modello** (col_modello = n ovunque), nessun fallback a prior, nessun OOM
nonostante le 8 GPU condivise. Main 838 s (40 s/item), Small 509 s (27 s/item), limite 3600.
stderr: solo warning transformers (rope yarn, fast path, kernel hub). Coverage 1, invalid 0.

## Punteggi

Main **0,5238 = 11/21**, Small **0,5263 = 10/19**. Identici al centesimo alle due entry
dell'organizzatore (0,52 / 0,53). Con set di 21 e 19 la coincidenza e' plausibile per caso;
non e' distinguibile se siamo alla classe di maggioranza. Il pavimento NON e' misurato in
questa fase e per pre-registrazione non lo misuriamo con una costante.

## Soglie

Presa in prestito 0,576 (artefatto DeepSeek) su Qwen3.5-4B, Skywork, Olmo-Think e DeepSeek.
Per gpt-oss la taratura sui 10 item pubblici ha dato **0,521** (warmup: 0,563): stesso
codice, stessi item, soglia diversa, coerente con la non-riproducibilita' misurata (V7).
Range dei punteggi del regressore, Main: Qwen [0.536,0.747], Skywork [0.554,0.750],
Olmo-Think [0.622,0.709] (tutti > soglia: predetti tutti spurious), DeepSeek [0.523,0.683],
gpt-oss [0.423,0.633]. Small: Qwen [0.479,0.605], Skywork [0.600,0.817], Olmo-Think
[0.320,0.763], DeepSeek [0.455,0.829].

## Regola di classifica

`submission_rule: Force_Latest_Multiple`: in classifica appare **l'ultima** submission per
partecipante e task, non la migliore. Ogni nuovo invio sovrascrive il precedente. La
disciplina "pubblico solo se batte" del warmup non esiste piu': l'ultima cosa spedita e'
quella che conta, e va scelta come tale.

## Dati pubblici per i nuovi modelli

Nessun item etichettato pubblico per Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think
(i model_id pubblici sono altri: Qwen3.5-397B, Llama-3.1-8B, gpt-5.2, gemini, GLM).
Calibrazione per-modello legittima possibile solo per DeepSeek (164) e gpt-oss (10).


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **«Ogni nuovo invio sovrascrive il precedente».** E' sbagliato: con la regola dichiarata, Force_Latest_Multiple, Codabench aggiunge in classifica ogni invio valutato come riga a se' e non ne toglie nessuno (report, Appendice D, voce 36).
