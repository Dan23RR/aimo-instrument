# Pre-registrazione E13b: l'etichetta di E13 riletta con la regola nuova

Scritta il 21 settembre 2026, PRIMA di calcolare qualunque etichetta con la regola nuova e
qualunque concordanza. Fino a qui ho guardato solo la struttura dei file: righe, colonne,
famiglie, campioni per variante, sovrapposizione dei testi. Nessun numero di esito.

## Perché

Il 17 settembre gli organizzatori hanno pubblicato, con il train set `aimo-interp/train-main-v2`,
la regola delle etichette della fase principale: **robust se max_drop <= 0,10, non robust se
max_drop >= 0,25, nessuna etichetta nella fascia intermedia**, anche in validazione e test.
`max_drop` è il calo di accuratezza più grande fra le famiglie di perturbazioni rispetto al
problema originale.

E13 (14 settembre) aveva rigenerato l'etichetta con la regola **vecchia** dei dataset pubblici:
robust se la base è risolta (almeno 6 su 10) e nessuna variante accettata è sbagliata. Esiti:
kappa(A,B) fra 0,18 e 0,44 sui quattro modelli, kappa(A,A') fra 0,46 e 0,79, concordanza con
le etichette ufficiali di DeepSeek 0,025. Con la regola vecchia un solo errore campionato su
circa 50 varianti cambia l'etichetta: è plausibile che buona parte dell'instabilità misurata
fosse della regola e non del processo. E13b lo decide sugli stessi dati, senza nuove
generazioni.

## Dati, tutti già su disco, nessuna GPU

- `vm_backup/estratto/work/aimo/e13/etich_<modello>_<cond>.parquet`: 4 modelli, condizioni A,
  B e A' (A' manca per Qwen). Per ogni problema: base con 10 campioni (`idx == -1`); varianti
  accettate dal cancello, 1 campione ciascuna, 5 famiglie (rephrase, rename, domain, distract,
  typos), da 1 a 50 varianti per problema. 164 problemi per file.
- Etichette ufficiali di DeepSeek: le righe `qwen3-8b:low` e `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
  nei dataset pubblici in `vm_backup/estratto/work/aimo/dati/`. `augmented-sample-math-full-filtered`
  ha una riga per tipo su 69 problemi; `augmented-sample-math-agg` e `-agg-filtered` hanno righe
  aggregate, dove per i problemi non robusti restano solo i tipi che hanno causato un calo;
  `aimo-interp-challenge-sample-full` ha 10 problemi con 5 tipi regolari più 2 tipi esperti.
  Identificazione qwen3-8b = DeepSeek-R1-0528-Qwen3-8B, la stessa usata in E11, E13 ed E15.
  L'insieme dei 164 problemi è quello di `feature_greedy.parquet`, con l'etichetta ufficiale
  vecchia.
- Train set ufficiale `dati/train_main_v2/train-00000-of-00001.parquet` (scaricato oggi da
  Hugging Face, 9,5 KB): 82 righe. Controllo di struttura: **5 dei nostri 164 problemi vi
  compaiono con testo identico.**

## Definizioni, congelate adesso

**max_drop nostro**, per (modello, condizione, problema): base_acc = corrette / 10 sulla base;
per ogni famiglia con almeno una variante accettata, acc_f = corrette / varianti della
famiglia; max_drop = massimo su f di (base_acc - acc_f).

**max_drop ufficiale**, per (DeepSeek, problema): massimo di `absolute_accuracy_decay` su tutte
le righe ufficiali del problema, in tutti i dataset, nelle 5 famiglie regolari. Una riga
aggregata robusta conta con il suo tipo o i suoi tipi (il suo decadimento è <= 0 per
costruzione).

**Etichetta nuova**: robust se max_drop <= 0,10; non robust se max_drop >= 0,25; nessuna
etichetta altrimenti. Ogni confronto usa solo i problemi etichettati in entrambe le serie.

**Analisi primaria**: regola letterale, senza filtro sulla base, come scritta nel README.
**Secondaria S1**: solo problemi con base_acc >= 0,5 in entrambe le serie. Motivo: la regola
letterale chiama robust anche un problema che il modello non risolve mai, perché il calo non
può superare base_acc. **Secondaria S2**: nel max_drop nostro solo le famiglie con almeno 5
varianti accettate. **Secondaria S3**: max_drop ufficiale con i tipi `expert_*` inclusi.
Nessun'altra secondaria.

**Statistica**: kappa di Cohen con IC95 bootstrap, 2000 ricampionamenti, seme 20260921,
con le stesse funzioni `kappa` e `kappa_ic` di `vm/e13_analisi.py`. Per ogni confronto si
riportano anche n, la quota nella fascia e la prevalenza di robust.

**Controllo di integrità, prima di tutto**: lo stesso script ricalcola con la regola vecchia
(base >= 6 su 10, seme 20260908) i kappa di `vm_risultati/e13_analisi_2026-09-14.txt`. Se non
ritrova quei numeri, l'analisi si ferma e non si guarda nient'altro.

## Ipotesi e previsioni, scritte prima

**H-E13b-1, affidabilità.** Con la regola nuova kappa(A,B) >= 0,60 per almeno 3 modelli su 4.
**Morta** se kappa(A,B) < 0,60 per 2 o più modelli.
Previsione: **vive**. Con dieci varianti per famiglia un errore singolo vale un calo di 0,10,
sotto la soglia, e la fascia toglie i casi al limite.

**H-E13b-2, validità (solo DeepSeek).** kappa(A, ufficiale nuova) >= 0,40 con estremo
inferiore dell'IC95 > 0. **Morta** altrimenti.
Previsione: **morta**, con kappa fra 0,10 e 0,40. Le differenze che restano non dipendono
dalla regola: varianti nostre contro le loro, decodifica, basi ufficiali con 1-15 campioni.

**H-E13b-3, quanto pesa la regola.** Per almeno 3 modelli su 4, kappa(A,B) nuova meno kappa(A,B)
vecchia >= 0,15. **Morta** altrimenti.
Previsione: **vive**.

Nessuna correzione per molteplicità: i criteri sono soglie su stime con intervallo, non test
di ipotesi nulla, e le tre ipotesi si giudicano separatamente.

**Descrittivi, senza criterio:** quota nella fascia per modello e condizione; per DeepSeek,
quante etichette ufficiali cambiano passando dalla regola vecchia alla nuova; kappa(A,A')
nuova per Skywork, Olmo e DeepSeek; sui 5 problemi in comune con il train set ufficiale, la
concordanza fra la nostra etichetta nuova e la loro per ogni modello disponibile, riportata
come conteggio (troppo pochi casi per un kappa).

## Cosa NON si fa

- Nessuna soglia diversa da 0,10 e 0,25 nella primaria; nessuna scelta di famiglie, di minimi di
  varianti o di sottoinsiemi dopo aver visto i numeri. Le secondarie sono queste tre.
- Nessuna generazione nuova, nessuna GPU.
- Il train set ufficiale non serve a tarare nulla: solo il confronto descrittivo sui 5 problemi.

## Esito

Da riempire dopo l'esecuzione, senza riscrivere nulla di quanto sopra.

## ESITO, 21 settembre 2026. Tutte e tre le ipotesi morte. Due previsioni su tre erano sbagliate.

Output completo in `vm_risultati/e13b_regola_nuova_2026-09-21.txt`, script `vm/e13b_regola_nuova.py`.

**Controllo di integrità superato:** con la regola vecchia lo script ritrova identici i 10
kappa di `e13_analisi_2026-09-14.txt`, stessi n.

| modello | kappa(A,B) vecchia | kappa(A,B) nuova, primaria | S1 base >= 0,5 | S2 fam >= 5 | kappa(A,A') vecchia | kappa(A,A') nuova |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | 0,178 | **0,342** [0,127, 0,534], n=86 | 0,308 | 0,429 | non c'è | non c'è |
| Skywork-OR1-Math-7B | 0,439 | **0,491** [0,168, 0,757], n=132 | 0,524 | 0,482 | 0,789 | 0,956 |
| Olmo-3-7B-Think | 0,378 | **0,406** [0,090, 0,687], n=133 | 0,405 | 0,277 | 0,653 | 0,763 |
| DeepSeek-R1-0528-Qwen3-8B | 0,413 | **0,564** [0,361, 0,734], n=100 | 0,550 | 0,524 | 0,464 | 0,754 |

**H-E13b-1, affidabilità: MORTA.** kappa(A,B) nuova sotto 0,60 per 4 modelli su 4. La mia
previsione era «vive»: sbagliata.

**H-E13b-3, peso della regola: MORTA.** Miglioramento di almeno 0,15 solo per 2 modelli su 4
(Qwen +0,164, DeepSeek +0,151; Skywork +0,052, Olmo +0,029). Previsione «vive»: sbagliata.

**H-E13b-2, validità: MORTA.** DeepSeek contro le etichette ufficiali rilette con la regola
nuova: kappa(A) **+0,023** [-0,088, +0,135], n=133; B +0,026; A' -0,006; S1 +0,006; S2 -0,027;
S3 (con i tipi esperti) +0,038. Previsione «morta» giusta, ma la fascia prevista (0,10-0,40)
era sbagliata: la concordanza è zero, come con la regola vecchia (0,025).

**Descrittivi.**
- Quota nella fascia senza etichetta: Qwen 25-28%, DeepSeek 12-26%, Skywork 6-11%, Olmo 8-9%
  (nel train set ufficiale 13%). La fascia toglie molti casi a Qwen (78 su 164) e a DeepSeek
  (64 su 164), e gli intervalli si allargano.
- **Il dato che spiega:** con le stesse varianti e un seme diverso (A contro A') la concordanza
  sale a 0,76-0,96. Con varianti diverse (A contro B) resta a 0,34-0,56. Con la regola nuova il
  rumore di campionamento è piccolo: l'etichetta dipende soprattutto da **quali varianti
  vengono estratte**.
- **Le nostre perturbazioni sono molto più deboli delle loro.** Sui problemi confrontati
  DeepSeek è robust nel 71-83% dei casi con le nostre varianti e nel 28-32% con le loro.
  Lo scarto non viene dalla regola.
- Etichette ufficiali di DeepSeek, vecchia contro nuova: i 40 robust restano tutti robust; dei
  124 non robust, 106 restano tali, 8 diventano robust, 10 finiscono nella fascia. Con la regola
  nuova le ufficiali sono robust al 29,3%.
- I 5 problemi in comune con il train set ufficiale (5 righe etichettate): serie A 2 concordi,
  2 discordi, 1 nella fascia; serie B 3 concordi, 2 discordi. Troppo pochi per dire qualcosa.
- Dettaglio d'implementazione: nella primaria il max_drop ufficiale usa le righe con almeno un
  tipo regolare, comprese le righe aggregate a lista; le righe dei soli tipi esperti entrano
  solo in S3. È la lettura della definizione scritta sopra.

**Controlli NON pre-registrati, dichiarati come tali** (`vm/e13b_controllo_modello.py`, output
in `vm_risultati/e13b_controllo_modello_2026-09-21.txt`).
1. Una parte delle etichette ufficiali ha model_id `qwen3-8b:low`. Sulle sole righe con
   model_id `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` (52 problemi) la concordanza resta zero:
   kappa(A) +0,087 [-0,043, +0,230], B +0,015, A' +0,005. Lo scarto di prevalenza resta
   (nostre 77-88% robust, loro 21-24%).
2. Sui 52 problemi in comune le righe ufficiali con nome DeepSeek hanno **max_drop identico**
   alle righe `qwen3-8b:low`, 52 su 52: è la stessa misura pubblicata con due nomi.
   L'identificazione qwen3-8b = DeepSeek-R1-0528-Qwen3-8B è quindi degli organizzatori. Non è
   un test-retest ufficiale e non va citato come tale.

**Cosa cambia per il report.** La regola nuova non salva la rigenerazione. Con varianti
estratte di nuovo, l'affidabilità resta fra 0,34 e 0,56. Con le etichette ufficiali la
concordanza è zero con entrambe le regole. La conclusione del §7, «the label is stable; our
regeneration of it is not», regge e ora ha un meccanismo: la dipendenza è da quali varianti
vengono estratte e da quanto sono difficili, non dal campionamento né dalla regola.

## CORREZIONE ALL'ESITO, 21 settembre 2026, dopo la verifica avversaria del report

Il descrittivo «con le stesse varianti e un seme diverso (A contro A') la concordanza sale a
0,76-0,96» è vero per Skywork e Olmo, **non per DeepSeek**. Il log di E13 mostra che la
condizione A' di DeepSeek è stata etichettata con il prompt esplicito sul formato
(`prompt=esplicito`), mentre A e B hanno usato il testo nudo del problema: la coda gpu0 era
partita l'8/9 con la vecchia `e13_coda.sh`, e quando A' è partita lo script di etichettatura
aveva già il default esplicito. Per DeepSeek kappa(A,A') = 0,754 mescola seme e prompt.
La lettura «conta quali varianti si estraggono» resta sostenuta da Skywork (0,956) e Olmo
(0,763), che hanno lo stesso prompt in tutte le condizioni. Le ipotesi e i loro verdetti non
cambiano: nessuna usava A'.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Data della regola.** La regola del max_drop era nella scheda pubblica di train-main-v2 dal 16 settembre (commit 41d83b6 delle 21:39 UTC, con la fascia senza etichetta estesa a val/test; il commit del 17/9 cambia solo la descrizione di due colonne), non dal 17 come dicono la riga 9 e il docstring di `vm/e13b_regola_nuova.py`.
2. **Ritirata** la lettura, in «Cosa cambia per il report», che la conclusione del par. 7 «the label is stable; our regeneration of it is not» regga. Il report non conclude piu' che l'etichetta ufficiale sia stabile (par. 7.3; Appendice D, voce 10).
3. **Previsioni.** Nessuna delle tre ha retto per intero: H-E13b-2 prevedeva la morte con kappa fra 0,10 e 0,40, ed e' 0,02.
4. **Regola vecchia e nuova sugli stessi problemi.** Ristretta ai problemi etichettati in entrambe le serie con la regola nuova, la regola vecchia da' gia' 0,233, 0,400, 0,398 e 0,460 (Qwen, Skywork, Olmo, DeepSeek): il cambio di regola vale +0,109, +0,090, +0,008 e +0,104, e nessun modello arriva a +0,15. Il verdetto di H-E13b-3 (morta) si rafforza. `vm/controlli_verifica2.py`.
5. **Severita' e ordinamento.** La nostra quota di robust (serie A) e' 0,775 dove l'etichetta ufficiale e' robust e 0,742 dove non lo e'; Spearman fra i due max_drop -0,094 (A) e -0,006 (B). Lo scarto non e' solo di severita'. `vm/controlli_verifica2.py`.
6. **Arrotondamento.** L'estremo superiore dell'IC95 di kappa(A, ufficiale nuova) e' 0,134955: a due decimali 0,13.
7. **Righe con il nome DeepSeek.** Sono 54 su 52 problemi; il controllo 2 confronta il max_drop per problema.


## CORREZIONI, 23 settembre 2026 (terza verifica avversaria del report, bozza v2.4)

Il testo sopra resta com'era, comprese le note precedenti. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **Olmo.** La lettura «conta quali varianti si estraggono», sostenuta nell'esito e nella correzione del 21/9 anche con Olmo (0,763), confronta kappa calcolati su insiemi di problemi diversi. Sui 126 problemi etichettati in tutte e tre le serie Olmo ha kappa(A,A') 0,715 e kappa(A,B) 0,434, con 5 e 9 disaccordi: il seme vale circa meta' dell'instabilita', e cosi' pochi disaccordi non separano le due fonti. La lettura regge per Skywork (0,937 contro 0,467 sui 127 problemi comuni). `vm/controlli_verifica4.py`; report, 7.5 e Appendice D, voce 43.
