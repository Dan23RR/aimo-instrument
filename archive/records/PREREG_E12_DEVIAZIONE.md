# Deviazione dichiarata da PREREG_E12, Stadio 2 — 7 settembre 2026, ore 12:20, PRIMA di eseguire

## L'errore che la impone

PREREG_E12 scrive che il cancello di qualità dello Stadio 2 accetta una variante solo se
"conserva la risposta attesa (che per i 164 abbiamo dai dati pubblici)". **È falso.** Verificato
il 7/9 alle 12:05: nessuno dei sei dataset pubblici contiene un campo con la risposta. L'ho
scritto senza controllare. La deviazione va dichiarata qui, prima di spendere, invece di
riscrivere la pre-registrazione.

## Il nuovo cancello di qualità, fissato ora

Senza risposta d'oro, la validità di una variante si verifica con un **risolutore
indipendente dal modello sotto esame**, altrimenti "la variante rompe DeepSeek" e "la variante
è sbagliata" diventano indistinguibili.

- **Riferimento** per ogni problema: la risposta di `openai/gpt-oss-120b` sull'originale,
  estratta con `\boxed{}` (stessa funzione `perturbazioni.risposta` di tutto il progetto).
  Problemi senza riferimento (nessun `\boxed{}`) sono **esclusi** da H32 e contati.
- Una variante è **accettata** se e solo se, tutte e tre: (1) il testo differisce
  dall'originale; (2) il rapporto di lunghezza variante/originale sta in **[0,7, 4,0]**
  (le varianti loro stanno fra 0,97 al 5° percentile e 2,67 al 95°; il cancello è più largo
  di proposito); (3) gpt-oss-120b sulla variante dà la stessa risposta del riferimento.
- **Controllo secondario di fedeltà del riferimento**: il riferimento è "confermato" se
  coincide con la risposta modale campionata di DeepSeek nello Stadio 1 (disponibile per 134
  problemi su 164). Il kappa primario si calcola su tutti i problemi con riferimento; il kappa
  sul solo sottoinsieme confermato si riporta come secondario, senza decidere nulla.
- `robust_E12` è definito solo per problemi con riferimento e **almeno una** variante
  accettata; gli altri si escludono e si contano. Definizione invariata: originale corretto E
  ogni variante accettata corretta, dove "corretto" = uguale al riferimento.

## Parametri non fissati in PREREG_E12, fissati ora

| | generazione varianti | risoluzione (cancello) |
|---|---|---|
| modello | gpt-oss-120b | gpt-oss-120b |
| reasoning effort | low (come pre-registrato) | **medium** |
| temperature / top_p | 1,0 / 1,0 (i default del modello) | 1,0 / 1,0 |
| seed | 20260906 | 20260906 |
| max_tokens | 2.048 | 4.096 |
| few-shot | 5 esempi per famiglia, seme fisso, mai dello stesso problema | |
| GPU | H100 (kernel MXFP4) | H100 |

Pozze few-shot dai 638 testi loro: domain 201, rename 74, typos 100, rephrase 113,
distract 133 (`gpu/e12_fewshot.json`). Etichettatura DeepSeek: greedy, 8.192 token,
A100-80GB, come pre-registrato.

Tutte le generazioni vanno **a blocchi con salvataggio parziale sul volume**: Modal ha
preemptato un container al minuto 16 nello Stadio 1.

## Cosa NON cambia

Statistica primaria (kappa di Cohen fra `robust_E12` e l'etichetta loro), criterio di morte
(**kappa < 0,30 → H32 MORTA**), famiglie (le cinque loro), k=3 varianti per famiglia,
secondarie (accordo per famiglia, accuratezza contro la maggioranza 0,756), Holm su 2.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Varianti d'esempio.** Nei due file per famiglia i testi delle cinque famiglie sono 624, non 638: gli altri 14 sono 11 varianti scritte da esperti (expert_no_solution, expert_perturbations) e 3 di una sesta famiglia, 'paraphrase'. I pool few-shot ne usano 621.
2. **«Risolutore indipendente».** E' gpt-oss-120b a reasoning medium, lo stesso modello che genera le varianti a reasoning low: indipendente dal modello sotto esame, non dal generatore. Poiche' i dataset pubblici non contengono le risposte, la sua risposta sull'originale e' anche il riferimento della correttezza, mentre la proposta degli organizzatori la definisce rispetto alla risposta vera.
