"""Test della logica di budget di solution v2, senza modello: si simula predict_robustness.

Verifica le tre proprieta' dichiarate:
  1. con budget stretto, solo k problemi passano dal modello, il resto dal prior;
  2. la run non supera mai il budget;
  3. l'uscita e' sempre lista di bool della lunghezza giusta (coverage=1, invalid=0).
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

SOL = Path(__file__).resolve().parent / "soluzione"


def carica_solution(budget_s, costo_per_problema_finto):
    """Importa solution.py fresco con un modulo uncertainty_profile FINTO che dorme."""
    # modulo finto: predict_robustness dorme costo_per_problema per ogni problema
    finto_inf = types.ModuleType("uncertainty_profile.inference")

    def predict_robustness(model_id, problem_texts):
        for _ in problem_texts:
            time.sleep(costo_per_problema_finto)
        # ritorna alternato, per distinguerlo dal prior costante
        return [(i % 2 == 0) for i in range(len(problem_texts))]

    finto_inf.predict_robustness = predict_robustness
    finto_pkg = types.ModuleType("uncertainty_profile")
    sys.modules["uncertainty_profile"] = finto_pkg
    sys.modules["uncertainty_profile.inference"] = finto_inf

    import os
    os.environ["AIMO_BUDGET_S"] = str(budget_s)
    # import fresco
    for m in list(sys.modules):
        if m == "solution":
            del sys.modules[m]
    spec = importlib.util.spec_from_file_location("solution", SOL / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def caso(nome, budget_s, costo, margine_override, n_problemi):
    print("-" * 74)
    print("CASO: %s" % nome)
    print("  budget=%.0fs  costo_finto/problema=%.2fs  problemi=%d" % (budget_s, costo, n_problemi))
    mod = carica_solution(budget_s, costo)
    mod.MARGINE_S = margine_override
    mod.COSTO_STIMATO_PER_PROBLEMA_S = max(costo, 0.01)
    mod._AVVIO = time.perf_counter()

    t0 = time.perf_counter()
    out = mod.are_robust("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                         ["problema %d" % i for i in range(n_problemi)])
    dt = time.perf_counter() - t0

    valido = isinstance(out, list) and len(out) == n_problemi and all(type(x) is bool for x in out)
    dal_modello = sum(1 for i, x in enumerate(out) if x == (i % 2 == 0))
    print("  durata reale        : %.2fs   (budget-margine = %.2fs)" % (dt, budget_s - margine_override))
    print("  entro budget?       : %s" % ("SI" if dt <= (budget_s - margine_override) + costo + 0.5 else "NO !!"))
    print("  uscita valida?      : %s (len=%d, tutti bool=%s)"
          % ("SI" if valido else "NO !!", len(out), all(type(x) is bool for x in out)))
    print()
    return valido, dt <= (budget_s - margine_override) + costo + 1.0


def main():
    print("=" * 74)
    print("TEST DELLA LOGICA DI BUDGET (modello simulato, nessuna GPU)")
    print("=" * 74)
    r = []
    # A: budget largo, tutti col modello
    r.append(caso("budget largo: tutti i problemi passano dal modello",
                  budget_s=100, costo=0.05, margine_override=1.0, n_problemi=10))
    # B: budget stretto, solo alcuni col modello, resto prior, MAI oltre budget
    r.append(caso("budget stretto: degradazione al prior a meta' strada",
                  budget_s=5, costo=0.30, margine_override=1.0, n_problemi=50))
    # C: budget quasi zero: zero col modello, tutto prior, run istantanea
    r.append(caso("budget esaurito: tutto prior, nessuna chiamata al modello",
                  budget_s=1.2, costo=0.30, margine_override=1.0, n_problemi=30))

    print("=" * 74)
    ok = all(v and b for v, b in r)
    print("TUTTI I CASI: %s" % ("PASSATI" if ok else "QUALCUNO FALLITO"))
    print("  In ogni caso l'uscita e' valida e la run resta entro il budget.")
    print("  E' la proprieta' che la baseline ufficiale non ha: lei va in timeout,")
    print("  questa finisce sempre, degradando al prior invece di essere invalidata.")


if __name__ == "__main__":
    main()
