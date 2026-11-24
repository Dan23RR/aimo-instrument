# Pre-registrazione E12: le famiglie giuste, e la self-consistency mai misurata

Scritta il 6 settembre 2026, ore 21, PRIMA di qualunque esecuzione. Nessun dato di E12 è
stato guardato. Costo previsto totale sotto i 10 $ di crediti Modal.

## Perché adesso, e cosa è cambiato (tutto verificato oggi)

1. **La regola sui dati di addestramento è caduta.** Michal Štefánik, Discord #qa, 5/9/2026
   ore 11:20: "we have removed the rule on the training data from the official rules listed on
   the website... you can use those [self-generated perturbations and labels] for anything."
   Il sito il 6/9 non elenca più la regola. **Ma i Terms su Codabench dicono ancora**: "The
   submitted methods shall not use extra labeled data for mathematical reasoning". Le due
   fonti sono in contrasto, e il paper arXiv 2607.13899 contiene ancora la regola vecchia.
2. **Il punteggio finale sarà su problemi nuovi.** Stesso messaggio: "the final scoring will
   be done on the new problems". Il paper: test split di 30 problemi AIMO 3 mai pubblicati.
   I 21 e 19 item della classifica attuale NON sono il bersaglio.
3. **La classifica si è rotta il 6/9.** [REDACTED: name, personal data and leaderboard figures of another participant, compared with the majority constant (11/21 and 10/19) and our v6 (0,52 / 0,53)] Non è rumore. Non sappiamo che cosa ha fatto e non lo indoviniamo.
4. **Le famiglie di perturbazione degli organizzatori sono semantiche**, verificato sui
   dataset pubblici (campo `permutations_causing_decay`, 638 testi su 65 problemi):
   `domain`, `rename`, `typos`, `rephrase`, `distract`, più due famiglie di esperti non
   replicabili. Generate da LLM: gpt-5.2:low (350 righe), gpt-oss-120b (259), gemini-3.1-pro
   (100). Circa 15 varianti per problema per famiglia.
   **Le nostre famiglie in E9 erano sintattiche** (format, latex_delims, framing, redundancy,
   question_first): il kappa 0,02 di E9 era la risposta giusta alla domanda sbagliata. E10
   (traduzione) idem. Nessuno dei nostri esperimenti ha mai testato le famiglie loro.
5. **La self-consistency non è mai stata misurata.** Era H11 in PREREG_E6, tagliata in
   PREREG_E6_DEVIAZIONE per costo (73 $ stimati con transformers a 23 tok/s) e sostituita da
   H11-bis a costo zero, morta. Con vLLM (E9: 174 generazioni greedy in 1197 s, ~1 $) costa
   un decimo. Michal stesso, 4/9: i modelli "work better in sampled generation". E
   l'interfaccia per le attivazioni annunciata darà "10 rounds of CoT generation" per item:
   la self-consistency diventerà gratis per tutti. Meglio sapere prima se predice qualcosa.
6. **Regola dell'etichetta ricontrollata oggi**: congiunzione (`n_detrimental == 0`, cioè
   ogni variante risolta) 379/379; maggioranza 221/379. Il paper dice "majority", i dati
   dicono congiunzione. Un altro partecipante ([REDACTED: username], issue #6) ha trovato lo stesso.

## Cancello di conformità, prima di qualunque spedizione

**G0.** Nessun artefatto addestrato su etichette generate da noi viene spedito in gara
finché Michal non conferma **per iscritto, via mail**, che le perturbazioni e le etichette
auto-generate sono ammesse, dato il testo dei Terms. La bozza della domanda è in
`MAIL_michal_regola_2026-09-06.md`. Gli esperimenti su dati pubblici qui sotto sono sviluppo,
non spedizione, e possono procedere.

## Stadio 1. H33: la self-consistency predice l'etichetta loro (DeepSeek, verità a terra)

**Dati.** I 164 problemi con etichetta vera per `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`
(`gpu/out/feature_greedy.parquet`, 24,4% robust). Una riga per problema: la CV raggruppata
coincide con quella stratificata.

**Generazione, parametri FISSATI ORA, nessuna sweep.** vLLM, `temperature=0.6`, `top_p=0.95`
(i valori raccomandati dalla scheda del modello), `n=8` campioni per problema, `max_tokens=8192`,
`seed=20260906`, stesso chat template di E9, estrazione della risposta con la funzione
`\boxed{}` di `perturbazioni.py`.

**Feature, tre e sole tre.**
- `accordo`: frazione degli 8 campioni la cui risposta coincide con la risposta modale
  (i campioni senza risposta contano come una risposta a sé, "nessuna");
- `risposte`: frazione degli 8 campioni con una risposta estratta;
- `accordo_pesato`: `accordo * risposte`.

**Statistica primaria, una sola: AUROC di `accordo` contro l'etichetta loro**, orientamento
libero dichiarato, soglia dal null permutazionale (20.000 permutazioni; con n=164 e 24% di
positivi il p95 è 0,604, misurato in E11). Le altre due feature sono secondarie e si
riportano senza pretendere nulla.

**Criterio di morte, scritto ora.** H33 è MORTA se `AUROC(accordo) < 0,65`. Motivazione:
0,65 sta sopra il p95 del null di quasi 5 punti e sotto il 0,70 del solo `n_char` di 5,
e `n_char` è esattamente il tipo di feature "idiosincratica del problema" che gli
organizzatori dichiarano di avere neutralizzato nel set nuovo: batterla non è più il
criterio giusto, sopravvivere al null con margine lo è. In più, per la sola informazione:
accuratezza bilanciata di una soglia su `accordo` in CV a 5 fold raggruppata, riportata.

**Cosa cambia se vive.** `accordo` è una feature black-box, ammessa da entrambe le versioni
delle regole, calcolabile nel container a ~8 campioni per item (senza vLLM nel container,
con transformers a batch 8: costo da misurare, ma i modelli Small sono 4-8B). Andrebbe
in v8 con una soglia calibrata sui 164, e poi sui campioni loro quando l'interfaccia uscirà.

**Costo.** 164 × 8 × fino a 8192 token ≈ 5 M token generati, A100-80GB, ~40 min, ~3 $.

## Stadio 2. H32: le famiglie loro, generate da noi, riproducono le etichette loro

**Perché.** Se le nostre varianti nelle famiglie giuste, con la regola di congiunzione,
riproducono l'etichetta loro su DeepSeek dove abbiamo la verità a terra, allora possiamo
etichettare i modelli nuovi (Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think) sui 174
problemi pubblici, che oggi non hanno nessuna etichetta, e addestrare per-modello. È la porta
che E9 credeva chiusa avendo bussato a quella sbagliata.

**Generatore.** `openai/gpt-oss-120b` via vLLM su Modal (lo stesso che hanno usato per 259
righe), reasoning basso, un prompt per famiglia con 5 esempi few-shot presi dai 638 testi
pubblici della stessa famiglia, `k=3` varianti per famiglia per problema, cinque famiglie:
`domain`, `rename`, `typos`, `rephrase`, `distract`. Le famiglie di esperti non si replicano.
Cancello di qualità fissato prima, come in E10: una variante è accettata solo se conserva
la risposta attesa (che per i 164 abbiamo dai dati pubblici) e non è identica all'originale.
Le famiglie con meno del 60% di varianti accettate si riportano ma non entrano nel calcolo.

**Etichetta E12.** DeepSeek greedy, `max_tokens=8192`, su originale e su ogni variante
accettata; risposta assente = sbagliata, come nel loro pipeline; `robust_E12 = 1` se e solo
se l'originale e OGNI variante accettata sono corretti (congiunzione, 379/379).

**Statistica primaria: kappa di Cohen fra `robust_E12` e l'etichetta loro sui 164.**
Secondarie: accordo per famiglia (quale famiglia trasferisce), e accuratezza di `robust_E12`
come predittore contro la maggioranza 0,756.

**Criterio di morte, scritto ora.** H32 è MORTA se `kappa < 0,30`. Sotto 0,30 le etichette
generate sono un bersaglio di addestramento peggiore del rumore che aggiungono. E9 ha dato
0,02 con le famiglie sbagliate; se le famiglie giuste danno lo stesso, la porta è chiusa
per davvero e va scritto nel report come risultato negativo definitivo, che vale più di un
tentativo in più.

**Costo.** Varianti: 164 × 5 × 3 = 2.460 generazioni brevi, ~10 min, ~2 $. Etichettatura:
2.624 generazioni DeepSeek fino a 8192 token, ~1 h, ~5 $.

## Stadio 3, solo se H32 vive E G0 è superato

Etichettare i tre modelli Small sui 174 problemi con lo stesso pipeline; estrarre per ciascuno
`accordo` (se H33 vive) e le 10 feature ufficiali informative; classificatore per-modello a
una o due feature con soglia in CV raggruppata; null permutazionale; Holm. Da pre-registrare a
parte, con numeri, prima di eseguire.

## Correzione di Holm e ordine

Due ipotesi primarie (H33, H32), Holm su 2. Si esegue prima lo Stadio 1 perché costa meno e
non ha cancelli di conformità. Lo Stadio 2 è indipendente e si esegue comunque, perché il
suo esito negativo vale quanto quello positivo per il report.

## Cosa NON si fa

- Nessun invio in classifica prima degli ultimi due giorni di ottobre, come deciso.
- Nessuna sweep su temperatura, k, max_tokens, layer, famiglie: tutto fissato sopra.
- Nessuna lettura del punteggio di gara come feedback: i 21/19 item non sono il bersaglio.
- Nessuna congettura sul metodo di [REDACTED: participant] nel report: non lo conosciamo.

## Esito

Da riempire per stadio, senza riscrivere nulla di quanto sopra.

## ESITO STADIO 1 — 7 settembre 2026, ore 12:01. H33 MORTA.

Eseguito come pre-registrato: vLLM su A100-80GB, T=0,6, top_p=0,95, n=8, max_tokens=8192,
seed=20260906, 164 problemi, 1.312 sequenze in 2.357 s (1,8 s/seq). Una preemption di Modal
al minuto 16 ha fatto ripartire il container da zero; costo totale circa 3 $.
File: `gpu/out/e12_s1_deepseek-ai__DeepSeek-R1-0528-Qwen3-8B.parquet`, verdetto in
`gpu/out/e12_s1_verdetto.json`, script `gpu/e12_s1_modal.py` e `gpu/e12_s1_analisi.py`.

| canale | AUROC | sopra il null (p95 0,604)? | acc. bilanciata CV | esito |
|---|---|---|---|---|
| **H33 `accordo`** | **0,5017** | no | 0,494 | **MORTA** (soglia 0,65) |
| `risposte` (sec.) | 0,538 (orient. −) | no | 0,516 | |
| `accordo_pesato` (sec.) | 0,533 | no | 0,512 | |
| `n_distinte` (sec.) | 0,500 | no | 0,484 | |
| `tok_medio` (sec.) | 0,652 | sì | 0,618 | proxy di n_char |
| `n_char` (info) | 0,704 | sì | 0,651 | |

**Il dato che chiude la questione.** Il modello è auto-consistente al 93,5% in media, e lo è
allo stesso modo nelle due classi: accordo 0,938 fra i robust, 0,934 fra gli spurious. In
123 problemi su 164 tutti e otto i campioni danno la stessa risposta, e fra questi 123 i
robust sono 30, cioè il 24,4%: **esattamente il tasso di base**. Sapere che il modello è
d'accordo con se stesso non dice nulla su come si comporta quando il problema viene
riformulato. L'etichetta misura la sensibilità alla perturbazione semantica del testo, non la
varianza di campionamento sull'originale. Sono assi ortogonali, come lo erano E9 (forma) ed
E10 (traduzione), ma questa volta l'ortogonalità è misurata sul canale che gli organizzatori
stessi stanno per rendere gratuito a tutti.

**Conseguenza per l'interfaccia delle attivazioni.** I "10 rounds of CoT generation" per item
che gli organizzatori annunciano daranno a ogni partecipante la self-consistency gratis. Su
DeepSeek, con verità a terra, quella feature vale 0,50. Va nel report.

**Secondarie, per onestà.** `risposte` ha orientamento negativo: i problemi robust hanno meno
campioni con risposta estratta (0,750 contro 0,816), cioè il modello satura più spesso gli
8.192 token proprio dove è robusto; AUROC 0,538, sotto il null, non si conclude nulla.
`tok_medio` a 0,652 sta sopra il null ma sotto `n_char` (0,704) e ne è il proxy ovvio: i
problemi lunghi fanno ragionare a lungo. È la feature "idiosincratica del problema" che il
set nuovo neutralizza per costruzione. Non è un canale.

**Bilancio.** Ventitré ipotesi pre-registrate, diciotto morte. Lo Stadio 2 (H32, le famiglie
loro generate da noi) è indipendente e resta da eseguire come pre-registrato.

## ESITO STADIO 2 — 7 settembre 2026, ore 15:40. H32 MORTA. Stadio 3 NON si esegue.

Eseguito come pre-registrato più la deviazione dichiarata in `PREREG_E12_DEVIAZIONE.md`.
Passo A, gpt-oss-120b su H100: 820 coppie problema-famiglia in 2 minuti, 2.449 varianti
(una coppia su 820 senza output), poi 2.613 risoluzioni a reasoning medium in 17 minuti.
Passo B, DeepSeek greedy a 8.192 token su A100-80GB: 2.016 generazioni in 3.710 s.
Costo dello Stadio 2 circa 6 $; E12 intero circa 9 $. File: `gpu/out/e12_s2_varianti.parquet`,
`gpu/out/e12_s2_deepseek.parquet`, `gpu/out/e12_s2_per_problema.parquet`,
`gpu/out/e12_s2_verdetto.json`; script `gpu/e12_s2_*.py`.

**Cancello di qualità.** 1.852 varianti accettate su 2.449 (75,6%). Scarti: 326 per
lunghezza, 250 per risposta diversa dal riferimento, 21 identiche. Per famiglia: typos 89,9%,
rephrase 89,8%, rename 78,5%, distract 69,7%, **domain 50,4%: sotto il 60%, esclusa dal
primario per la regola fissata prima** (181 scarti per lunghezza: le storie in un altro dominio
escono lunghe fino a 25 volte l'originale). Riferimento gpt-oss per 158 problemi su 164 (il
log del passo A stampava 164: un NaN contato come valore, bug di stampa ininfluente). Problemi
definiti: **157** (6 senza riferimento, 1 senza varianti accettate); mediana 11 varianti
accettate per problema.

| | |
|---|---|
| robust LORO | 23,6% |
| robust E12 (congiunzione su ~11 varianti) | 54,1% |
| **H32 kappa** | **−0,0008** (morte se < 0,30) |
| accuratezza di robust_E12 contro l'etichetta loro | 0,478 contro maggioranza 0,764, **−28,7 punti** |
| tavola E12×loro | 1&1: 20, 1&0: 65, 0&1: 17, 0&0: 55 |

**Secondarie.** Con tutte le famiglie, domain compresa: kappa −0,030. Sul sottoinsieme in cui
il riferimento gpt-oss coincide con la modale campionata di DeepSeek (n=134): kappa 0,025. Per
famiglia, stabilità di DeepSeek e kappa con l'etichetta loro: rephrase 85% / −0,012; rename
86% / −0,070; domain 77% / −0,044; distract 81% / −0,010; typos 89% / −0,014. **Le varianti
risolte correttamente sono l'84,4% fra i problemi che loro chiamano robust e l'83,3% fra
quelli che chiamano spurious.**

**Esplorativo, non pre-registrato, dichiarato come tale.** AUROC della frazione continua di
varianti corrette contro l'etichetta loro: **0,503** (null p95 0,607). Originale risolto da
DeepSeek: 83,8% fra i robust loro, 82,5% fra gli spurious. La loro base_accuracy: 0,970 e
0,981. Non c'è nemmeno un ordinamento.

**Cosa dice il dato, e perché è il risultato più importante del progetto.** La definizione
loro dell'etichetta è: robust se e solo se il modello resiste a ogni perturbazione delle
famiglie dichiarate. Abbiamo rigenerato quelle famiglie con lo stesso generatore che hanno
usato loro per 259 righe, con i loro stessi esempi come few-shot, filtrate con un risolutore
indipendente, e applicato la loro regola di congiunzione al loro stesso modello. **La
concordanza con l'etichetta è zero.** Non bassa: zero, in ogni famiglia, in ogni sottoinsieme,
anche come ordinamento continuo. Tre esperimenti pre-registrati con tre famiglie diverse
(E9 forma, E10 traduzione, E12 le loro) danno kappa 0,02, −0,10, −0,001. Messi accanto a
H33 (self-consistency 0,50), alle 14 feature ufficiali (0,656 ricalibrate, sotto la costante)
e alle attivazioni (E11, morte), la lettura più economica è una sola: **l'etichetta non è una
proprietà stabile della coppia modello-problema riproducibile dalla sua definizione. È
dominata dall'estrazione specifica delle varianti e dal pipeline con cui è stata calcolata.**
Questo spiega dall'interno perché nulla la predice, e perché il solo `n_char` la ordina meglio
di tutto: la lunghezza del problema è correlata a quante varianti si possono estrarre e a
quante possono andare storte.

**Avvertenze, prima che le trovi qualcun altro.** (1) Il riferimento è gpt-oss medium, non
una risposta d'oro: ma sul sottoinsieme confermato da un secondo modello il kappa è uguale.
(2) Il nostro cancello tiene solo varianti che un modello forte risolve come l'originale; il
loro dataset pubblico contiene proprio le varianti che hanno *rotto* il modello, e non
possiamo escludere che alcune di quelle cambino la risposta: sarebbe un problema loro, non
nostro, ma va detto. (3) DeepSeek greedy singolo contro il loro campionamento con n
predizioni. (4) Tre varianti per famiglia contro le loro circa quindici: la nostra
congiunzione è su ~11 varianti, la loro su ~75, e questo spiega il 54% contro il 24% di
robust, **ma non spiega l'assenza di ordinamento** (AUROC 0,503).

**Previsione pre-registrata sul test finale, scritta ora, 7 settembre 2026.** Se la lettura
sopra è giusta, nessun metodo generalizza ai problemi nuovi su cui sarà calcolato il
punteggio finale. **P1:** l'accuratezza della prima classificata sul test finale starà entro
due errori standard binomiali dalla costante di maggioranza di quel set. **P2:** l'accuratezza
finale di [REDACTED: the participant whose entries led both tracks on 7 September] sarà inferiore ai suoi [REDACTED: leaderboard figures of 7 September] attuali di più di due errori standard.
Se P1 o P2 falliscono, l'etichetta ha una struttura apprendibile che noi non abbiamo trovato,
e va scritto così.

**Stadio 3 non si esegue.** Il cancello G0 è superfluo. **Bilancio: ventiquattro ipotesi
pre-registrate, diciannove morte.** Nessun invio in classifica.


## CORREZIONI, 23 settembre 2026 (seconda verifica avversaria del report, bozza v2.3)

Il testo sopra resta com'era. Quello che segue lo corregge; i numeri vengono dagli script citati, con output in `vm_risultati/`.

1. **P1 e P2.** P2 riguarda un partecipante, le cui voci in testa il 7/9 sono due invii distinti (uno per track). Nella copia che sara' rilasciata con il report, il nome, i dati personali del partecipante e le sue cifre saranno oscurati, con la redazione segnata sul posto (report, par. 13 e Appendice C).
2. **Data della rottura della classifica.** Le voci stavano sopra quelle degli organizzatori dal 3 settembre ([REDACTED: that participant's figures]); il 6/9 [REDACTED: that participant's figures]. `vm/controlli_verifica3.py`.
3. **Esplorativo.** L'AUROC 0,503 della frazione continua di varianti corrette e' ricalcolata da `vm/controlli_verifica3.py`; la concordanza 0,025 sui 134 problemi con riferimento confermato e' in `vm_risultati/e12_s2_analisi_2026-09-23.txt`.
