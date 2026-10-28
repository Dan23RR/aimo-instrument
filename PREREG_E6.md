# Pre-registrazione E6 — il discriminatore che deve battere il 69,2%, su GPU esterna

Scritta il 29 agosto 2026, prima di qualunque run GPU.

## Perché serve la GPU e cosa cambia

Sette ipotesi offline sono morte perché il segnale discriminante (robust vs spurious sui
casi ad alta accuratezza, che sono l'intero test) vive nelle feature del modello in
esecuzione. Con Modal/Colab possiamo finalmente far girare `DeepSeek-R1-0528-Qwen3-8B`,
lo stesso checkpoint della cache del container, e sviluppare il discriminatore vero.

## Dati di addestramento, e perché sono leciti

Le **164 coppie etichettate** del modello in cache nei dataset pubblici degli organizzatori
(`qwen3-8b:low` in augmented-agg, agg-filtered, challenge-full, val-sample). È lo stesso
materiale su cui la baseline ufficiale dichiara di essersi addestrata: non è "extra labeled
data", è il loro. Le famiglie di perturbazione NOSTRE usate come feature a runtime
(formattazione, ridondanza, self-consistency) sono diverse dalle loro famiglie
(distract, domain, rename, rephrase, typos, expert_*), come il regolamento richiede.

## Ipotesi

**H10 (riproduzione).** Il protocollo della baseline (14 feature di incertezza da una
generazione greedy, regressore, soglia) riprodotto sulle 164 coppie in CV raggruppata per
problema dà un'accuratezza compatibile col 69,2% dichiarato (entro il suo intervallo).
Se non si riproduce, prima si capisce perché, poi si va avanti.

**H11 (il colpo).** Aggiungere alle 14 feature: (a) self-consistency su k=4 campioni a
temperatura 0,6 (accordo di maggioranza sulla risposta finale, entropia delle risposte);
(b) stabilità della risposta sotto 2-3 perturbazioni NOSTRE answer-preserving (formattazione,
ridondanza) batte la sola baseline sugli stessi fold.

**Metrica e confronto appaiato.** Stessi fold per baseline e variante (CV raggruppata per
`problem_id`, 5 fold, seed fissato PRIMA: 20260829). Confronto appaiato per item.

**CRITERIO DI MORTE:** H11 è morta se il guadagno appaiato medio è < 1 errore standard
appaiato. Se muore, si spedisce la baseline riprodotta con il wrapper budget e si dichiara
che il margine competitivo è l'ingegneria (robustezza al timeout), non il classificatore.

## Budget e costi, stimati prima

- 1 passata greedy completa su 164 problemi ≈ 164 × ~120 s ≈ 5,5 ore GPU.
- k=4 campioni + 2 perturbazioni ≈ ×5 con cap a 2048 token ≈ 12-15 ore GPU totali.
- Su Modal A100 40GB (~2-2,5 $/h): **25-40 $**. Su Colab Pro A100: ~10-15 $ di compute units.
- **Vincolo di fedeltà: bf16, come il container.** Niente quantizzazione 4-bit (Colab T4
  gratuito): cambierebbe la distribuzione delle feature fra training e valutazione.

## Cosa NON si fa

Nessuna nuova submission per sviluppare il classificatore: lo sviluppo è tutto offline su
dati pubblici. Le submission restano per: (già spesa) la curva di costo del container, e poi
il deploy finale validato. Nessuna generazione di dati etichettati con le famiglie loro.

## Esito

Da riempire dopo i run.
