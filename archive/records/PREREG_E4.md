# Pre-registrazione E4 — quale segnale calcolabile a runtime batte la baseline?

Scritta il 29 agosto 2026 prima di `esperimento_e4.py`.

## Il bersaglio vero non è il warmup

Il warmup è capato a 9/17 (verificato cinque volte). Il premio è nella **fase principale**,
dove i modelli saranno presumibilmente caricabili (anche perché è ciò che la mail agli
organizzatori chiede), e dove il metodo da battere è la baseline uncertainty a **69,2%**.

Per vincere lì serve un classificatore di robustezza migliore del 69,2%, calcolabile dentro
il budget di 3600 s con un modello caricato. Prima di scrivere codice che gira su GPU (che
non abbiamo), stabilisco offline **quanto vale il segnale più economico** sui dati pubblici.

## Cosa è calcolabile a runtime, e cosa no

A runtime, dato un modello caricabile e un problema, possiamo misurare:
- **base_accuracy** = quanto spesso il modello risolve il problema originale (self-consistency
  su K campioni). Forte ma costoso: K forward pass.
- **stabilità sotto perturbazioni nostre** = generiamo perturbazioni (rename, riordino,
  distrattori) e vediamo se la risposta cambia. È misura diretta di robustezza, permessa dal
  regolamento (perturbazioni di famiglie diverse da quelle del val finale), ma costosa.

Le feature di incertezza della baseline (entropia, logprob) NON sono ricostruibili offline
senza il modello, quindi E4 non le tocca: misura solo il segnale `base_accuracy`, che è il
predittore candidato più semplice e il più caro da giustificare.

## Ipotesi e criteri di morte

**H7.** `base_accuracy` come singola feature, con soglia scelta per modello in CV raggruppata
per modello, predice la robustezza meglio della classe di maggioranza per modello.
Criterio di morte: guadagno < `sqrt(0.25/n)`.

**H8.** Il tetto: `base_accuracy` con soglia oracolo per fold. Misura il massimo che questo
singolo segnale può dare, per sapere se vale la pena aggiungerci le feature di incertezza.

**H9.** `base_accuracy` è più forte come predittore quando è **estrema** (vicino a 0 o a 1):
se il modello risolve sempre o mai il problema, la robustezza è più prevedibile che nella
zona intermedia. Se vero, un classificatore che astiene sulla fascia intermedia e lì usa il
prior batte quello che decide ovunque.

## Confini

Nessuna submission per questo esperimento: è tutto offline sui dati pubblici. Nessuna
taratura sul set di valutazione. La CV è raggruppata per modello: nessun modello sta sia in
train sia in test, perché la fase principale conterrà modelli che non abbiamo mai visto.

## Esito, 29 agosto 2026

**H7 MORTA.** base_accuracy con soglia scelta su altri modelli predice il modello held-out a
0,7612, **identico** al prior globale tutto-spurious. Guadagno 0,0 punti.

**H8 (tetto):** con soglia oracolo per modello, base_accuracy arriva a 0,776, cioè +1,5 punti
sul prior. Il singolo segnale, anche perfettamente calibrato, dà pochissimo.

**H9 ha rivelato la struttura vera, e va a E5.** base_accuracy ~ 0 (modello che non risolve
mai l'originale) è robust nel 100% dei casi (20/20). È un artefatto della definizione: robust
= nessuna perturbazione peggiora, e se l'accuratezza è già a zero non può peggiorare. Il segnale
è non monotono, quindi una soglia non lo cattura, ed è per questo che H7 è morta pur esistendo
un segnale. Se sfruttabile, andrebbe verificato sul proxy del test. Vedi E5.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Righe duplicate.** H7 e H8 sono calcolate sulle 335 coppie, con le 52 misure DeepSeek contate due volte. Sulle 283 coppie distinte H7 resta +0,0 e il tetto H8 vale 0,7739 (+1,8 punti sulla costante globale, +0,4 sulla maggioranza per modello). `vm/warmup_ricalcoli.py`.
2. **Split per modello, non per problema.** In H7 234 delle 335 righe di test hanno il loro problema anche in addestramento, 198 con la stessa etichetta e la stessa base_accuracy, fra cui tutte le 52 righe DeepSeek, copie esatte di righe di addestramento pubblicate come `qwen3-8b:low`. La sovrapposizione favorisce l'ipotesi; il verdetto non cambia. `vm/controlli_verifica3.py`.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Baseline di H7.** Questo file fissa come baseline di H7 «la classe di maggioranza per modello»; esperimento_e4.py confronta invece con il prior globale dei modelli di addestramento. Contro la baseline pre-registrata H7 fa -1,2 punti sulle 335 coppie (-1,4 sulle 283 con le righe duplicate contate una volta); contro il prior globale +0,0. Il verdetto (morta) si rafforza. La deviazione non fu dichiarata al momento. `vm/controlli_verifica4.py`; report, par. 4 e 13.
