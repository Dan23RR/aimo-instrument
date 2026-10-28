"""Varianza di riesecuzione della stessa bundle (PREREG_V7_RIPRODUCIBILITA.md).

Uso: python analisi_varianza.py
Legge dati/campioni_v7.json, che raccoglie a mano i punteggi e i range del regressore
letti dai log di ogni riesecuzione.
"""
import json, math, statistics as st
from pathlib import Path

F = Path(__file__).resolve().parent / "dati" / "campioni_v7.json"
N = 17

def main():
    if not F.is_file():
        print("manca %s" % F); return
    c = json.loads(F.read_text(encoding="utf-8"))
    punti = [x["corretti"] for x in c]
    print("=" * 74)
    print("VARIANZA DI RIESECUZIONE — stessa bundle, %d campioni" % len(c))
    print("=" * 74)
    print("  %-12s %-10s %s" % ("submission", "punteggio", "range regressore per modello"))
    for x in c:
        r = x.get("range", {})
        s = "  ".join("%s[%.3f-%.3f]" % (k, v[0], v[1]) for k, v in r.items())
        print("  %-12s %d/%d=%.4f  %s" % (x["id"], x["corretti"], N, x["corretti"]/N, s))
    print()
    print("  min %d  max %d  media %.2f  su %d item" % (min(punti), max(punti), st.mean(punti), N))
    if len(punti) > 1:
        sd = st.stdev(punti)
        print("  deviazione standard: %.2f item = %.1f punti percentuali" % (sd, 100*sd/N))
        print("  escursione: %d item = %.1f punti percentuali" % (max(punti)-min(punti), 100*(max(punti)-min(punti))/N))
    print()
    print("-" * 74)
    print("LETTURA")
    print("-" * 74)
    esc = max(punti) - min(punti)
    print("  La stessa identica bundle produce punteggi che variano di %d item." % esc)
    if esc >= 1:
        print("  Quindi in classifica un distacco di %d item o meno NON e' distinguibile" % esc)
        print("  dal rumore di riesecuzione dello strumento. Vale per tutti, non solo per noi.")
    print()
    print("  ONESTA' (pre-registrata): se il massimo e' stato pubblicato, il numero in")
    print("  classifica e' il MASSIMO DI %d CAMPIONI della stessa bundle, non un metodo" % len(c))
    print("  migliore. Va dichiarato ovunque il numero compaia.")

if __name__ == "__main__":
    main()
