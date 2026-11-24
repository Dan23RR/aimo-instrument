"""Il tetto: quanto puo' valere al massimo un metodo di interpretabilita' su questo set.

Non e' una stima e non e' una simulazione. E' un vincolo strutturale che discende da
due fatti osservati e da un fatto dichiarato dagli organizzatori.

OSSERVATO (submission 907310, dan23rr, 29 agosto 2026 ore 12:01 CEST, stato Finished).
La sonda ha registrato gli argomenti che la gara le ha passato:

    chiamata=1  model_id='allenai/Olmo-3-7B-Instruct'                problemi=4
    chiamata=2  model_id='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'     problemi=2
    chiamata=3  model_id='google/gemma-3-27b-it'                     problemi=2
    chiamata=4  model_id='openai/gpt-oss-120b'                       problemi=9
                                                             TOTALE  17

DICHIARATO dagli organizzatori, README di aimo-interp/getting-started:
    "Models currently available in the cache: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B.
     Models outside this list cannot be loaded during evaluation."
e nel Dockerfile.competition:
    ENV HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1        (rete disabilitata)

Ne segue che per 15 dei 17 item nessun partecipante puo' ottenere alcun segnale dal
modello: ne' attivazioni, ne' logit, ne' probabilita' dei token, ne' campionamento
comportamentale. Restano il testo del problema e una stringa col nome del modello.
"""
import math

# ------------------------------------------------------------------ OSSERVATO
LOTTI = [
    ("allenai/Olmo-3-7B-Instruct",             4,  7.0,  False),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",  2,  8.0,  True),   # unico in cache
    ("google/gemma-3-27b-it",                  2, 27.0,  False),
    ("openai/gpt-oss-120b",                    9, 120.0, False),
]
N = sum(q for _, q, _, _ in LOTTI)
ISPEZIONABILI = sum(q for _, q, _, cache in LOTTI if cache)
CIECHI = N - ISPEZIONABILI

# IPOTESI dalla classifica: i cluster complementari 0.53 x4 e 0.47 x6 sono i costanti.
ROBUST, SPURIOUS = 9, 8


def tetto(inspezionabili: int, n: int, robust: int) -> tuple[int, int]:
    """Massimo numero di item corretti per un metodo che sia PERFETTO sugli item
    ispezionabili e ottimale-costante sugli altri. Peggior caso e miglior caso sulla
    ripartizione delle etichette fra ispezionabili e ciechi."""
    spurious = n - robust
    esiti = []
    for r_isp in range(max(0, inspezionabili - spurious), min(inspezionabili, robust) + 1):
        r_cieco = robust - r_isp
        s_cieco = (n - inspezionabili) - r_cieco
        esiti.append(inspezionabili + max(r_cieco, s_cieco))
    return min(esiti), max(esiti)


def main():
    print("=" * 78)
    print("IL TETTO STRUTTURALE — AIMO Interpretability Challenge, fase Warmup")
    print("dalla submission 907310 del 29 agosto 2026")
    print("=" * 78)

    print("\n[OSSERVATO] struttura del set di valutazione")
    print("  %-42s %-6s %-9s %s" % ("model_id", "item", "miliardi", "nella cache?"))
    for nome, q, par, cache in LOTTI:
        print("  %-42s %-6d %-9.0f %s" % (nome, q, par, "SI" if cache else "no"))
    print("  %-42s %-6d" % ("TOTALE", N))

    print("\n" + "-" * 78)
    print("1. QUANTO DEL SET E' EFFETTIVAMENTE ISPEZIONABILE  [DERIVATO]")
    print("-" * 78)
    print("  item su modelli caricabili nel container:  %2d / %d  = %.1f%%"
          % (ISPEZIONABILI, N, 100 * ISPEZIONABILI / N))
    print("  item su modelli NON caricabili:            %2d / %d  = %.1f%%"
          % (CIECHI, N, 100 * CIECHI / N))
    print()
    print("  Per l'%.1f%% del set, la gara chiede di giudicare la robustezza di un" % (100 * CIECHI / N))
    print("  modello a cui il container nega ogni accesso. Non e' una difficolta':")
    print("  e' l'assenza dell'oggetto da misurare.")

    print("\n" + "-" * 78)
    print("2. IL TETTO DI UN METODO PERFETTO  [DERIVATO, dato lo split %d/%d]" % (ROBUST, SPURIOUS))
    print("-" * 78)
    lo, hi = tetto(ISPEZIONABILI, N, ROBUST)
    print("  Un metodo PERFETTO su tutti gli item ispezionabili, e per il resto")
    print("  ottimale fra le due costanti, ottiene al massimo:")
    print("      %d/%d = %.4f   (arrotondato: %.2f)" % (hi, N, hi / N, hi / N))
    print("  e come minimo %d/%d = %.4f a seconda di come cadono le etichette." % (lo, N, lo / N))
    print()
    print("  In classifica il primo ha %d/%d = %.2f." % (11, N, 11 / N))
    print()
    if 11 > hi:
        print("  *** E' SOPRA IL TETTO DI UN METODO PERFETTO. Quel punteggio non puo'")
        print("  venire dall'ispezione dei modelli: viene dal testo, o dalla fortuna. ***")
    elif 11 == hi:
        print("  *** IL PRIMO STA ESATTAMENTE SUL TETTO. ***")
        print("  Cioe' il vertice della classifica coincide con cio' che si ottiene")
        print("  essendo perfetti sui %d item ispezionabili e fortunati sui %d ciechi."
              % (ISPEZIONABILI, CIECHI))
        print("  Ma lo stesso %d/%d si ottiene anche tirando a caso, %.1f volte su 100"
              % (11, N, 100 * sum(math.comb(N, i) for i in range(11, N + 1)) / 2 ** N))
        print("  (vedi AIMO/leaderboard.py). Le due spiegazioni non sono distinguibili")
        print("  da questo strumento, e questo e' il punto.")

    print("\n" + "-" * 78)
    print("3. QUANTA INFORMAZIONE PORTA L'UNICO MODELLO ISPEZIONABILE  [DERIVATO]")
    print("-" * 78)
    print("  Passare da tirare a caso a essere perfetti sui %d item ispezionabili" % ISPEZIONABILI)
    print("  sposta il punteggio atteso di %d/%d = %.1f punti percentuali."
          % (ISPEZIONABILI, 2 * N, 100 * ISPEZIONABILI / (2 * N)))
    se = math.sqrt(0.25 / N)
    print("  L'errore standard di un punteggio su %d item e' %.1f punti." % (N, 100 * se))
    print("  Il guadagno massimo dell'interpretabilita' vale %.2f errori standard."
          % ((ISPEZIONABILI / (2 * N)) / se))
    print()
    print("  Cioe': anche risolvendo perfettamente la parte ispezionabile del compito,")
    print("  lo spostamento e' piu' piccolo del rumore dello strumento che lo misura.")

    print("\n" + "-" * 78)
    print("4. LE DUE TRACCE  [DERIVATO dalla taglia dei modelli]")
    print("-" * 78)
    piccoli = [(n_, q, c) for n_, q, p, c in LOTTI if p < 10]
    grandi = [(n_, q, c) for n_, q, p, c in LOTTI if p >= 10]
    q_piccoli = sum(q for _, q, _ in piccoli)
    q_grandi = sum(q for _, q, _ in grandi)
    isp_piccoli = sum(q for _, q, c in piccoli if c)
    print("  Small Models Track (<10B): %2d item, di cui ispezionabili %d" % (q_piccoli, isp_piccoli))
    print("  Main Track       (>=10B): %2d item, di cui ispezionabili %d" % (q_grandi, 0))
    print()
    print("  Nella traccia dei modelli grandi NESSUN item e' ispezionabile.")
    print("  Il primo premio da 3.000 dollari si assegna su %d decisioni binarie" % q_grandi)
    print("  prese senza alcun accesso ai modelli giudicati.")

    print("\n" + "=" * 78)
    print("SINTESI")
    print("=" * 78)
    print("  17 item. 4 modelli. 1 caricabile. 2 item su 17 ispezionabili.")
    print("  Tetto di un metodo di interpretabilita' perfetto: %d/%d = %.2f." % (hi, N, hi / N))
    print("  Il primo in classifica sta esattamente su quel tetto, e lo stesso")
    print("  punteggio si ottiene a caso 16,6 volte su 100.")
    print("  Guadagno massimo dell'interpretabilita': %.1f punti. Rumore: %.1f punti."
          % (100 * ISPEZIONABILI / (2 * N), 100 * se))
    print("  Su questo strumento, un metodo di interpretabilita' perfetto non produce")
    print("  un segnale distinguibile dal rumore. Non e' una gara difficile da vincere:")
    print("  e' una gara in cui %.0f%% del compito e' cieco per costruzione." % (100 * CIECHI / N))


if __name__ == "__main__":
    main()
