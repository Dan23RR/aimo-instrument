# Esito E11 — il canale delle attivazioni, chiuso con misura

3 settembre 2026. Protocollo in PIANO_E11_definitivo.md, fissato prima di ogni run.
Costo totale: circa 2 euro di Modal.

## H29, cancello di ripetibilita': PASSATO, e in modo schiacciante

Due passate sugli stessi 40 item variando cio' che varia in produzione (batch 1 senza
padding contro batch 8 con left padding e vicini diversi). Soglia di morte fissata prima:
ICC(2,1) < 0,78.

| feature | ICC(2,1) | errore relativo medio |
|---|---|---|
| d_framing | **0,9987** | 1,1% |
| d_redundancy | **0,9984** | 1,5% |
| Z_C1 | **0,9984** | 1,1% |
| Z_C1b | **0,9972** | 1,2% |

**Risultato collaterale, e vale piu' del cancello.** Il punteggio della gara oscilla di
**11,8 punti** fra riesecuzioni byte-identiche (misurato nel warmup: 11, 9, 11, 11 su 17).
Le attivazioni del prompt oscillano dell'**1%**. Il nondeterminismo del benchmark **non e'
nel modello**: e' nella generazione, cioe' nel percorso di decoding su 4096 token, dove una
scelta vicina al pareggio cambia tutto il seguito. Diagnosi precisa dell'origine del rumore
che rende la classifica indistinguibile.

## H30 e H31: entrambe MORTE

174 problemi estratti in 16 secondi (forward-only, layer 29 di 36, profondita' 0,80 fissata
prima). 164 con etichetta vera, 24,4% robust.

Soglia data dal **null permutazionale** a 20.000 permutazioni con orientamento libero:
**p95 = 0,6036**, non 0,5. Obbligo aggiuntivo: battere di almeno 0,05 il solo conteggio dei
caratteri (M0_TESTO, AUROC **0,7046**, costo zero GPU).

| canale | AUROC | supera il null? | contro n_char | esito |
|---|---|---|---|---|
| **H30** Z_C1 (spostamento residualizzato) | 0,6147 | si | **−0,090** | **MORTA** |
| **H31** Z_C1b (coerenza direzionale) | 0,5552 | **no** | −0,149 | **MORTA** |
| Z_C1 grezza (secondaria) | 0,5770 | no | −0,128 | |
| d_pad_long (controllo) | 0,5165 | no | −0,188 | |

H30 supera di poco il null ma **perde di nove punti contro il conteggio dei caratteri**.
Nessun canale supera entrambe le condizioni.

## Il controllo geometrico, che spiega perche'

Il padding neutro appaiato in token era li' per separare "la rappresentazione si sposta
perche' il prompt e' piu' lungo" da "si sposta perche' il modello e' fragile".

| | distanza coseno media |
|---|---|
| d_framing (perturbazione, +24 token) | 0,0486 |
| d_redundancy (perturbazione, +37 token) | 0,0585 |
| **d_pad_short** (testo neutro, +24 token) | **0,0685** |
| **d_pad_long** (testo neutro, +37 token) | **0,0711** |

**Il testo neutro sposta la rappresentazione PIU' delle perturbazioni vere**, a parita' di
token aggiunti. La differenza perturbazione meno padding e' **negativa**: −0,0163.

Quindi lo spostamento di rappresentazione misura la **lunghezza del prompt**, non la
fragilita' del modello sul problema. Senza il controllo di padding avremmo misurato 0,577 di
AUROC sulla distanza grezza e l'avremmo chiamato segnale di fragilita': sarebbe stato falso.
Il controllo, deciso prima, ha impedito l'errore.

## Bilancio

**Ventidue ipotesi pre-registrate, diciassette morte.** Il canale delle attivazioni era
l'ultimo non falsificato, ed e' chiuso con misura, non per rinuncia: le feature sono
ripetibili al 99,8%, quindi il fallimento non e' rumore, e' assenza di segnale.

Resta in piedi il risultato di RISULTATI_E11_benchmark.md: anche il canale ufficiale, con la
soglia ricalibrata al meglio, fa 0,7500 contro 0,7561 della classe di maggioranza. E il
conteggio dei caratteri, a costo zero, ordina meglio (AUROC 0,7046) di qualunque cosa
estratta dal modello.

Nessuna submission. In classifica resta v6.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Holm non applicato.** `gpu/e11_s2_analisi.py` annuncia nel docstring una correzione di Holm su 2 test primari, ma alla riga 97 c'e' solo un commento: il verdetto confronta ogni AUROC con il p95 non corretto di una singola statistica (0,6036). Il piano la voleva su cinque test. Con Holm sui due test eseguiti H30 non supera: p permutazionale 0,0299 contro 0,025 (`vm/e11_holm.py`). «H30 supera di poco il null» vale solo per il null non corretto; il verdetto (morta, sotto 0,655) non cambia.
2. **Altre deviazioni dal piano, non dichiarate al momento.** Tutti i 164 problemi invece dei 124 con 40 tenuti come set di conferma (par. 4.4 del piano); AUROC calcolate sullo scalare grezzo con orientamento scelto sui dati, invece di AUROC fuori campione da una logistica L2 con C = 0,05 in CV 5x20; M0_TESTO ridotto all'AUROC del solo n_char (0,7046) invece delle sei feature di testo di H28; H32-H34 mai eseguite; H25-H28 non riportate con quegli identificativi.
3. **ICC di H29.** I valori 0,9987, 0,9984, 0,9984 e 0,9972 esistono solo in questo file: `gpu/e11_icc_modal.py` scrisse l'output per item sul volume Modal, che non e' stato scaricato.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Alfa di Holm.** Il piano voleva Holm su cinque test primari ad alfa 0,010, applicato al criterio di utilita'; il criterio di morte era la soglia 0,655. Il confronto 0,030 contro 0,025 della nota precedente usa l'alfa 0,05 implicito nella soglia p95 dello script: H30 non supera Holm sui due test nemmeno con quell'alfa.
2. **Altre deviazioni non dichiarate.** Il piano (par. 4.1) rendeva obbligatorie, accanto a ogni AUROC riportata, la stessa AUROC stratificata sui quartili del numero di predizioni perturbate e l'AUROC di riferimento dei contatori di bookkeeping (0,8619 fuori campione, non disponibile a inferenza): nessuna delle due e' stata calcolata. Il piano fissava tre profondita' (0,25 e 0,55 descrittive, 0,80 primaria): e' stata estratta solo la 0,80. Nessuna delle due deviazioni cambia il verdetto.


## CORREZIONI, 23 settembre 2026 (quarta verifica, sulle note e sulle correzioni della terza)

Il testo sopra resta com'era. Quello che segue lo corregge o lo precisa; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Lo 0,7500 del Bilancio.** «Resta in piedi il risultato di RISULTATI_E11_benchmark.md: anche il canale ufficiale, con la soglia ricalibrata al meglio, fa 0,7500 contro 0,7561 della classe di maggioranza» riguarda il regressore spedito applicato alle feature della nostra reimplementazione dell'estrazione su A100 (generation_frac_selected_is_top1 = 1,0 su 164/164, generation_mean_prob non registrata), non il percorso ufficiale, su cui un'accuratezza con soglia ricalibrata non e' stata misurata (report, 2.3b e 8.2; `vm/e16_analisi.py`).
