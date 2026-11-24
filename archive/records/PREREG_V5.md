# Pre-registrazione v5 — il metodo vero, ora che i modelli esistono

Scritta il 31 agosto 2026 dopo la mail degli organizzatori, PRIMA di costruire.

## Il vincolo che uccideva la gara e' stato rimosso

Gli organizzatori, rispondendo alla nostra misurazione, hanno messo in cache **tutti e
quattro** i modelli valutati, gia' nel warmup:

    openai/gpt-oss-120b               61.4 GiB
    google/gemma-3-27b-it             51.8 GiB
    deepseek-ai/DeepSeek-R1-...-8B    15.8 GiB
    allenai/Olmo-3-7B-Instruct        14.2 GiB

Tutti caricabili con `device_map="auto"` e `local_files_only=True`, senza offload su CPU.

**Questo ribalta la conclusione centrale di undici esperimenti.** L'irriducibilita' dei 15
item su 17 non era una proprieta' del compito: era una proprieta' dell'istrumentazione, e
l'istrumentazione e' cambiata. Le ipotesi H3, H4, H5, H12 erano morte perche' usavano UN
modello come sensore per gli altri. Adesso ogni modello puo' misurare se stesso.

## Il metodo

Per ogni lotto (model_id, problemi): carica QUEL modello, estrai le 14 feature di incertezza
della baseline ufficiale con la sua stessa configurazione, applica il regressore.

**Problema noto e dichiarato:** l'artefatto (regressore + soglia) esiste solo per DeepSeek,
distribuito dagli organizzatori. Per gli altri tre non esiste e non abbiamo dati etichettati
con le loro feature per addestrarne uno. Quindi v5 applica il **regressore DeepSeek alle
feature di ciascun modello**: e' transfer, dichiarato come tale.

E' diverso da cio' che E7 ha ucciso. E7 usava le feature DEL MODELLO IN CACHE per predire
ALTRI modelli (sensore esterno, morto a −16,5). Qui si usano le feature del modello STESSO,
con una mappa feature→robustezza presa in prestito. E' l'unica opzione disponibile e la sua
riuscita non e' garantita.

## Ipotesi e criterio di morte

**H16.** v5 batte 9/17 sul set di valutazione warmup.
**Criterio:** se v5 <= 9/17, il transfer del regressore non funziona e va dichiarato; non si
itera cambiando soglie sul punteggio (sarebbe probing).

## Rischio operativo e mitigazione

Budget 3600 s. Costo stimato: 17 item x ~120 s = ~2040 s, piu' il caricamento di quattro
modelli di cui uno da 61 GiB. Puo' non entrare. La guardia di budget gia' collaudata degrada
al prior quando il tempo residuo non basta, quindi **la run finisce sempre**, con coverage 1
e invalid 0: nel caso peggiore v5 degenera a v3 e pareggia, non perde.

Ordine dei lotti deciso dalla gara, non da noi. Se il tempo finisce, i lotti tardivi vanno al
prior: e' una perdita di informazione, non un fallimento.

## Spedita

Submission caricata il 31 agosto. Attesa: lunga, perche' carica quattro modelli.
Previsione pre-registrata: se il transfer del regressore funziona, > 9/17. Se non funziona,
<= 9/17 e si dichiara morto. Nel caso peggiore la guardia di budget degrada al prior e il
risultato pareggia v3.

## ESITO — 909853: **0.5882 = 10/17. H16 SOPRAVVIVE.**

Prima ipotesi sopravvissuta su tredici. Il log mostra il metodo pieno: 4 modelli caricati
(1.2s / 1.4s / 4.3s / 4.4s: la cache nuova e' calda), 17 item su 17 misurati col modello,
619 s totali su 3600. Coverage 1, invalid 0. PUBBLICATA secondo la regola pre-registrata
(>9/17): classifica 0.59, quinta fra i punteggi reali (0.71, 0.71, 0.65, 0.65, noi).

Le distribuzioni dei punteggi del regressore per modello (dai log, soglia presa in prestito 0.576):
  Olmo     0.213-0.564  -> TUTTI sotto soglia = tutti predetti robust
  DeepSeek 0.454-0.722  -> 1 sopra, 1 sotto
  gemma    0.425-0.789  -> 1 sopra, 1 sotto
  gpt-oss  0.340-0.870  -> a cavallo della soglia, split ignoto
