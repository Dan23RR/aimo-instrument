"""E15-R, ripresa della Parte 2 dopo l'interruzione (24/11/2026).

La corsa di e15r_ricalcolo.py partita il 23/11 alle 10:18 si e' fermata a 925 permutazioni su 1.000: il computer si
e' riavviato il 23/11 alle 23:48 (ultima riga del log alle 23:46:15). Rilanciare e15r_ricalcolo.py rifarebbe la
Parte 1 e ne aggiungerebbe una seconda copia ai file della cartella. Questo script importa e15r_ricalcolo.py senza
cambiarlo e chiama solo parte2():
- la statistica di H36 si ricalcola e deve tornare 0,7097 (altrimenti parte2 si ferma da sola);
- le permutazioni sono le stesse, generate in fila da RandomState(20260915), e la q-esima si valuta con lo stesso
  seme RandomState(20260914 + 100 + q): le 925 gia' scritte in parte2_null.jsonl si saltano, le altre 75 si
  calcolano, e il risultato e' quello di una corsa senza interruzioni;
- nel log una riga dice dove comincia la ripresa, cosi' le ore di calcolo si contano senza la pausa.

Uso (dalla cartella AIMO): python revisione_condizionata_2026-11/e15r_riprendi.py CARTELLA_USCITA [N_LAVORI]
"""
import json
import os
import sys

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
CARTELLA = sys.argv[1]
LAVORI = int(sys.argv[2]) if len(sys.argv) > 2 else 5

import e15r_ricalcolo as R  # noqa: E402  (legge e svuota sys.argv: gli argomenti sono gia' presi sopra)


def main():
    R._USCITA, R._LAVORI = CARTELLA, LAVORI
    fatte = 0
    percorso = os.path.join(CARTELLA, "parte2_null.jsonl")
    if os.path.isfile(percorso):
        fatte = len({json.loads(r)["q"] for r in open(percorso, encoding="utf-8") if r.strip()})
    R.log("RIPRESA della Parte 2 con e15r_riprendi.py: %d permutazioni gia' scritte, lavori %d" % (fatte, LAVORI))
    X, y, nchar, esp, keep = R.carica()
    R.parte2(X, y, 2 if R.PROVA else 1000)
    R.log("fine")


if __name__ == "__main__":
    main()
