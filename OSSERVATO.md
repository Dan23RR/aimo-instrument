# OSSERVATO — la struttura del set di valutazione AIMO

Ricevuta della submission **907310**, utente `dan23rr`, competizione Codabench 16180,
fase Warmup, task "AIMO Robustness Validation v8".
Caricata il **29 agosto 2026 alle 12:01 CEST**, stato **Finished**.
Bundle: `AIMO/sonde/solution-always-spurious.zip` (1021 byte, un solo file).

## Cosa è stato spedito

Un predittore costante che restituisce `False` per ogni coppia. Nessun modello caricato,
nessuna inferenza. La previsione era **pre-registrata prima della spedizione**, dentro il
docstring del file spedito e in `AIMO/leaderboard.py`: punteggio atteso 0.47, coverage 1.0,
invalid 0. Il bundle era stato collaudato in locale contro i programmi ufficiali di
ingestion e scoring (`AIMO/collaudo_locale.py`): accuracy 0.4706, coverage 1.0, invalid 0
su un set sintetico di 17 item con 8 spurious.

## Cosa è tornato

La sonda registrava solo conteggi aggregati e l'identificativo del modello, che sono
argomenti che la gara passa alla funzione per proprio disegno. Nessun testo dei problemi
è stato registrato, conservato o trasmesso. Uscita integrale:

```
[sonda-pavimento] chiamata=1 model_id='allenai/Olmo-3-7B-Instruct'             problemi_in_questa=4 problemi_finora=4
[sonda-pavimento] chiamata=2 model_id='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'  problemi_in_questa=2 problemi_finora=6
[sonda-pavimento] chiamata=3 model_id='google/gemma-3-27b-it'                  problemi_in_questa=2 problemi_finora=8
[sonda-pavimento] chiamata=4 model_id='openai/gpt-oss-120b'                    problemi_in_questa=9 problemi_finora=17
```

## I tre fatti

**1. Il set ha 17 item.** L'inferenza fatta il mattino del 29 agosto dal solo spaziamento
dei punteggi pubblici (tutti multipli esatti di 1/17) è confermata per osservazione diretta.
Metodo e conferma sono in file separati e datati: la previsione non è stata riscritta dopo.

**2. Il set ha 4 modelli, e uno solo è caricabile.**

| model_id | item | parametri | nella cache del container |
|---|---|---|---|
| `allenai/Olmo-3-7B-Instruct` | 4 | 7B | no |
| `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | 2 | 8B | **sì** |
| `google/gemma-3-27b-it` | 2 | 27B | no |
| `openai/gpt-oss-120b` | 9 | 120B | no |

Il README ufficiale dice: *"Models currently available in the cache:
deepseek-ai/DeepSeek-R1-0528-Qwen3-8B. Models outside this list cannot be loaded during
evaluation."* Il `Dockerfile.competition` imposta `HF_HUB_OFFLINE=1` e
`TRANSFORMERS_OFFLINE=1`, con la rete disabilitata.

**3. Quindi 15 item su 17 sono ciechi.** Per l'88,2% del set nessun partecipante può
ottenere attivazioni, logit, probabilità dei token o campionamento comportamentale del
modello che deve giudicare. Restano il testo del problema e una stringa col nome del modello.

Nella traccia principale, quella da 3.000 dollari, i modelli sono `gemma-3-27b-it` e
`gpt-oss-120b`: **11 item su 11 non ispezionabili, zero accesso al modello giudicato.**

## Cosa ne segue, in aritmetica

Da `AIMO/tetto.py`:

- Un metodo **perfetto** sui 2 item ispezionabili e ottimale-costante sugli altri 15
  arriva al massimo a **11/17 = 0.65**. È esattamente il punteggio del primo in classifica.
  Lo stesso 11/17 si ottiene tirando a caso **16,6 volte su 100**. Le due spiegazioni non
  sono separabili da questo strumento.
- Il guadagno massimo ottenibile dall'interpretabilità è **5,9 punti percentuali**.
  L'errore standard di un punteggio su 17 item è **12,1 punti**.
  Il guadagno vale **0,49 deviazioni standard**.

> Su questo strumento, un metodo di interpretabilità perfetto non produce un segnale
> distinguibile dal rumore.

## La previsione pre-registrata è confermata

Il punteggio è uscito poco dopo, sulla stessa riga della submission:

```
907310   solution-always-spurious.zip   2026-08-29 12:01   Finished   0.47
```

**0.47 = 8/17.** Era la previsione scritta prima della spedizione, dentro il file spedito
e in `AIMO/leaderboard.py`. Il criterio di morte dichiarato era: *"se il punteggio non è
0.47, l'ipotesi è falsa e va dichiarata tale"*. Non lo è.

Da questo momento le seguenti righe passano da **IPOTESI** a **OSSERVATO**:

- il set contiene **8 item spurious e 9 robust** su 17;
- il pavimento (classe di maggioranza) è **f = 9/17 = 0,529**, e l'amplificazione
  dell'errore che ne discende è **2,125x**;
- il gruppo a 0.53 in classifica è il predittore costante `always-true`, cioè quattro
  partecipanti (**incluso l'organizzatore**) sono al pavimento esatto;
- il gruppo a 0.47 è `always-false`, la stessa sonda spedita qui.

Restano valide tutte le conseguenze aritmetiche già derivate, che ora poggiano su una
misura e non su una congettura: capacità del primo in classifica **0,26 ± 0,25**, tetto
strutturale **11/17**, guadagno massimo dell'interpretabilità **0,49 sigma**.

[REDACTED: an observation about another participant's leaderboard entry and a conjecture about its method] Il metodo non è esclusivo e non va raccontato come se lo fosse.

## Righe che non abbiamo passato

- Non è stato registrato né esfiltrato alcun contenuto del set privato.
- Non è stata toccata alcuna etichetta: la sonda non ne ha accesso.
- Non sono state fatte submission ripetute per ricostruire etichette per differenza.
  Una sola sonda, una sola previsione, dichiarata prima.
