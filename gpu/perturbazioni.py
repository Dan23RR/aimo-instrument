"""Famiglie di perturbazione NOSTRE per E9: meccaniche, deterministiche, answer-preserving.

Nessuna coincide con le famiglie degli organizzatori (distract, domain, rename, rephrase,
typos, expert_*, paraphrase): non riscriviamo parole, non rinominiamo variabili, non
aggiungiamo distrattori, non introduciamo errori. Tocchiamo solo layout, delimitatori,
cornice e ordine delle frasi. Nessun LLM: zero contaminazione, riproducibilita' totale.

NOTA: questo file va scritto con uno strumento che non interpreta i backslash (non via
heredoc di shell): contiene regex e stringhe LaTeX con backslash letterali.
"""
import re

FAMIGLIE = ["format", "latex_delims", "framing", "redundancy", "question_first"]

_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(])")   # confine di frase conservativo


def _frasi(t: str) -> list:
    t = re.sub(r"[ \t]+", " ", t.strip())
    return [s for s in _SPLIT.split(t) if s]


def format_(t: str) -> str:
    """Una frase per riga, spazi normalizzati. Ogni carattere non-spazio e' invariato."""
    return "\n".join(_frasi(t))


def latex_delims(t: str) -> str:
    r"""$...$ -> \(...\) per il LaTeX inline. Non tocca $$...$$ ne' il contenuto."""
    out, i, n = [], 0, len(t)
    while i < n:
        if t.startswith("$$", i):
            j = t.find("$$", i + 2)
            if j < 0:
                out.append(t[i:])
                break
            out.append(t[i:j + 2])
            i = j + 2
            continue
        if t[i] == "$":
            j = t.find("$", i + 1)
            if j < 0:
                out.append(t[i:])
                break
            out.append("\\(" + t[i + 1:j] + "\\)")
            i = j + 1
            continue
        out.append(t[i])
        i += 1
    return "".join(out)


def framing(t: str) -> str:
    """Istruzione neutra prima e dopo. Il testo del problema e' incluso identico."""
    return ("Solve the following problem. Work carefully and give the final answer "
            "in \\boxed{}.\n\n" + t + "\n\nFinal answer in \\boxed{}.")


def redundancy(t: str) -> str:
    """Il testo, poi lo stesso testo ripetuto dopo 'Restated:'. Nulla di nuovo."""
    return t + "\n\nRestated: " + t


_DOMANDA = re.compile(
    r"\?$|^(find|determine|compute|calculate|what|how many|how much|evaluate|prove|show)\b",
    re.I)


def question_first(t: str):
    """Sposta in testa la frase-domanda finale, se separabile. Altrimenti None (salta)."""
    fr = _frasi(t)
    if len(fr) < 2:
        return None
    ultima = fr[-1]
    if not _DOMANDA.search(ultima.strip()):
        return None
    return ultima + "\n\nGiven: " + " ".join(fr[:-1])


def perturba(t: str) -> dict:
    """Tutte le famiglie applicabili a t. question_first puo' mancare."""
    out = {"format": format_(t), "latex_delims": latex_delims(t),
           "framing": framing(t), "redundancy": redundancy(t)}
    q = question_first(t)
    if q is not None:
        out["question_first"] = q
    return out


# ---------------------------------------------------------------- estrazione risposta
_BOXED = re.compile(r"\\boxed\s*\{")


def _boxed_ultimo(s: str):
    """Contenuto dell'ultimo \\boxed{...} con parentesi bilanciate."""
    pos = [m.end() for m in _BOXED.finditer(s)]
    if not pos:
        return None
    i = pos[-1]
    depth = 1
    j = i
    while j < len(s) and depth:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return s[i:j - 1] if depth == 0 else None


def normalizza(a):
    if a is None:
        return None
    a = a.strip().rstrip(".")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(left|right)", "", a)
    a = a.replace(" ", "").replace("$", "")
    a = re.sub(r"^\\?\(|\\?\)$", "", a)
    return a or None


def risposta(testo_generato: str):
    return normalizza(_boxed_ultimo(testo_generato))


if __name__ == "__main__":
    import json
    import pathlib
    dati = json.loads((pathlib.Path(__file__).resolve().parent.parent / "dati" / "val-sample.json")
                      .read_text(encoding="utf-8"))
    visti = set()
    n_q = 0
    for r in dati:
        p = r["original_problem"]
        if p in visti:
            continue
        visti.add(p)
        v = perturba(p)
        if "question_first" in v:
            n_q += 1
        assert re.sub(r"\s", "", v["format"]) == re.sub(r"\s", "", p)
        assert v["latex_delims"].replace("\\(", "$").replace("\\)", "$") == p or "$$" in p
        assert p in v["framing"] and v["redundancy"].count(p) == 2
        assert "\x08" not in v["framing"] and "\\boxed{}" in v["framing"]
    print("problemi distinti val-sample: %d | question_first applicabile: %d | invarianti OK"
          % (len(visti), n_q))
    casi = [(r"... so \boxed{\dfrac{3}{4}}.", r"\frac{3}{4}"),
            (r"\boxed{12} then \boxed{ 15 }", "15"),
            ("no box here", None),
            (r"\boxed{\text{ABC}}", "ABC"),
            (r"\boxed{\left(\frac{1}{2}\right)}", r"\frac{1}{2}"),
            (r"x = \boxed{ 3\sqrt{2} }.", r"3\sqrt{2}")]
    for s, atteso in casi:
        assert risposta(s) == atteso, (s, risposta(s), atteso)
    print("estrazione risposta: %d/%d OK" % (len(casi), len(casi)))
