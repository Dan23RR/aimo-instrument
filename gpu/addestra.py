"""Verdetto E6: H10 (riproduzione della baseline) e H11-bis (le feature che lei non vede).

Protocollo pre-registrato in PREREG_E6.md e PREREG_E6_DEVIAZIONE.md. Fissati PRIMA di
vedere i dati e non modificabili dopo:
  - CV a 5 fold RAGGRUPPATA per problem_id, seed 20260829;
  - modello: logistica standardizzata, C=1.0;
  - confronto APPAIATO per item fra le 14 feature della baseline e le 14 + H11-bis;
  - criterio di morte: guadagno appaiato medio < 1 errore standard appaiato -> H11-bis morta.

Piu' un controllo che ho promesso prima di vedere i numeri: il TEST ANTI-ARTEFATTO.
In E4 avevo trovato che un modello che non risolve mai il problema e' `robust` per
definizione (un'accuratezza gia' a zero non puo' peggiorare), e in E5 che il set di
valutazione della gara e' FILTRATO per accuratezza di base alta, quindi quei casi li' non
ci sono. Se il vantaggio di H11-bis venisse da quei casi, non trasferirebbe alla gara.
Percio' il test viene rifatto sul solo sottoinsieme che assomiglia al test set.

Uso:  python addestra.py out/feature_greedy.parquet
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260829
QUI = Path(__file__).resolve().parent

# le 14 della baseline ufficiale (uncertainty_profile/config.py), tutte medie per token
BASE = ["mean_logprob", "min_logprob", "max_logprob", "std_logprob", "mean_nll", "ppl",
        "min_k_logprob", "mean_entropy", "mean_top1", "mean_margin", "frac_top1",
        "frac_alta", "frac_bassa"]
# H11-bis: cio' che una media per token cancella per costruzione
NUOVE = ["n_token", "satura_cap", "ha_risposta", "ent_prima", "ent_dopo", "ent_trend",
         "lp_prima", "lp_dopo", "lp_trend", "top1_trend"]


def cv_appaiata(df, feats, fold_di):
    """Ritorna un vettore 0/1 per item: predizione corretta in CV raggruppata."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = df["robust"].astype(int).to_numpy()
    fold = df["problem_id"].map(fold_di).to_numpy()
    giuste = np.zeros(len(df), dtype=int)
    for k in sorted(set(fold)):
        tr, te = fold != k, fold == k
        if te.sum() == 0 or len(set(y[tr])) < 2:
            giuste[te] = (y[te] == int(round(y[tr].mean()))).astype(int)
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
        clf.fit(df.loc[tr, feats].fillna(0.0), y[tr])
        giuste[te] = (clf.predict(df.loc[te, feats].fillna(0.0)) == y[te]).astype(int)
    return giuste


def fold_stabili(problemi):
    rng = np.random.RandomState(SEED)
    ordinati = sorted(problemi)
    perm = rng.permutation(len(ordinati))
    return {ordinati[idx]: (pos % 5) for pos, idx in enumerate(perm)}


def confronto(df, etichetta):
    feats_b = [c for c in BASE if c in df.columns]
    feats_e = feats_b + [c for c in NUOVE if c in df.columns]
    fold_di = fold_stabili(df["problem_id"].unique())

    g_b = cv_appaiata(df, feats_b, fold_di)
    g_e = cv_appaiata(df, feats_e, fold_di)
    magg = max(df.robust.mean(), 1 - df.robust.mean())

    diff = (g_e - g_b).astype(float)
    media = diff.mean()
    se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("inf")

    print("\n" + "=" * 74)
    print(etichetta)
    print("=" * 74)
    print("  n = %d coppie, %d problemi, robust %.1f%%"
          % (len(df), df.problem_id.nunique(), 100 * df.robust.mean()))
    print("  classe di maggioranza          : %.4f" % magg)
    print("  baseline riprodotta (14 feat)  : %.4f" % g_b.mean())
    print("  con H11-bis (%2d feat)          : %.4f" % (len(feats_e), g_e.mean()))
    print("  guadagno appaiato              : %+.4f (%.1f punti), SE appaiato %.4f"
          % (media, 100 * media, se))
    vivo = media >= se
    print("  criterio pre-registrato        : %s" % ("SOPRAVVIVE" if vivo else "MORTA"))
    return {"n": len(df), "magg": magg, "base": g_b.mean(), "est": g_e.mean(),
            "guadagno": media, "se": se, "viva": bool(vivo)}


def main(percorso):
    df = pd.read_parquet(percorso)
    df = df.reset_index(drop=True)
    print("caricato %s: %d righe, %d colonne" % (percorso, len(df), len(df.columns)))
    print("robust %d / spurious %d" % (df.robust.sum(), (~df.robust).sum()))

    # ---- H10: la baseline si riproduce?
    feats_b = [c for c in BASE if c in df.columns]
    fold_di = fold_stabili(df["problem_id"].unique())
    acc_b = cv_appaiata(df, feats_b, fold_di).mean()
    magg = max(df.robust.mean(), 1 - df.robust.mean())
    print("\n" + "=" * 74)
    print("H10 — riproduzione della baseline ufficiale")
    print("=" * 74)
    print("  accuratezza in CV per problema : %.4f" % acc_b)
    print("  classe di maggioranza          : %.4f" % magg)
    print("  dichiarato dagli organizzatori : 0.692 (+/- 0.101)")
    dentro = abs(acc_b - 0.692) <= 0.101
    print("  compatibile con il dichiarato  : %s" % ("SI" if dentro else "NO, da indagare"))

    # ---- H11-bis sul set completo
    pieno = confronto(df, "H11-bis — set completo")

    # ---- TEST ANTI-ARTEFATTO: solo i casi che assomigliano al test set della gara
    tab = QUI.parent / "dati" / "tabella_aggregata.json"
    filtrato = None
    if tab.is_file():
        righe = json.loads(tab.read_text(encoding="utf-8"))
        ba = {r["problem_id"]: r.get("base_accuracy") for r in righe
              if r.get("model_id") in ("qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
              and r.get("base_accuracy") is not None}
        df2 = df.copy()
        df2["base_accuracy"] = df2["problem_id"].map(ba)
        noti = df2.base_accuracy.notna().sum()
        print("\n  base_accuracy recuperata per %d/%d coppie" % (noti, len(df2)))
        sub = df2[df2.base_accuracy >= 0.05]
        if len(sub) >= 30 and sub.robust.nunique() > 1:
            filtrato = confronto(sub.reset_index(drop=True),
                                 "H11-bis — SOLO casi con base_accuracy >= 0.05\n"
                                 "(il test della gara e' filtrato cosi': vedi E5)")
        else:
            print("\n  sottoinsieme troppo piccolo o degenere per il test anti-artefatto")

    # ---- verdetto
    print("\n" + "=" * 74)
    print("VERDETTO")
    print("=" * 74)
    if pieno["viva"] and filtrato and filtrato["viva"]:
        print("  H11-bis sopravvive SIA sul set completo SIA sul sottoinsieme che")
        print("  assomiglia al test della gara. Il vantaggio non e' l'artefatto di E4/E5.")
        print("  -> costruire il classificatore e spedirlo.")
    elif pieno["viva"] and filtrato and not filtrato["viva"]:
        print("  H11-bis sopravvive sul set completo ma MUORE sul sottoinsieme filtrato.")
        print("  Il vantaggio viene dai casi che il test della gara NON contiene:")
        print("  e' l'artefatto di E4/E5, e non trasferisce. NON spedirlo.")
    elif pieno["viva"]:
        print("  H11-bis sopravvive sul set completo; test anti-artefatto non eseguibile.")
        print("  Trattare come non confermata finche' non lo e'.")
    else:
        print("  H11-bis MORTA per il criterio pre-registrato.")
        print("  Ottava ipotesi caduta. Il margine competitivo resta l'ingegneria:")
        print("  robustezza al budget di tempo, che la baseline non ha.")

    # artefatto solo se il verdetto e' pulito
    if pieno["viva"] and filtrato and filtrato["viva"]:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        feats = [c for c in BASE + NUOVE if c in df.columns]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=5000, C=1.0, random_state=SEED))
        clf.fit(df[feats].fillna(0.0), df["robust"].astype(int))
        fuori = Path(percorso).parent / "discriminatore_e6.joblib"
        joblib.dump({"clf": clf, "feats": feats, "seed": SEED}, fuori)
        print("\n  artefatto scritto: %s" % fuori)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/feature_greedy.parquet")
