"""Appendici del report da 5 pagine (28/10/2026), copiato da report_9p/appendici_v24.py del 23/10.

Uso (dalla cartella report_5p): python appendici_5p.py
Legge ../versioni_writeup/WRITEUP_v2.4_2026-09-23.md e scrive in latex/:
  app_A.tex  la Section 4 della v2.4 (campagna di falsificazione e tabella delle ipotesi)
  app_C.tex  la Section 11 e l'Appendix D della v2.4 (le 54 correzioni)
  app_D.tex  le Sections 12 e 13 della v2.4 (limiti, deviazioni, provenienza, calcolo)
  app_EFGH.tex  E: mappa delle sezioni; F e G: Appendix A e B della v2.4;
                H: risorse e licenze (controllate il 23/10/2026); I: prompt
Il testo riportato non viene cambiato: si traduce solo la marcatura. Rispetto alla versione da
9 pagine cambia solo la mappa E: Limitations e AI usage statement stanno dopo la Section 7, e le
Sections 2-7 sono date per intero nell'Appendice J.
"""
import hashlib
import io
import os
import re

QUI = os.path.dirname(os.path.abspath(__file__))
V24 = os.path.join(QUI, "..", "versioni_writeup", "WRITEUP_v2.4_2026-09-23.md")
USCITA = os.path.join(QUI, "latex")
SHA_V24 = "5a06e4fbb88a6ddb"

dati = open(V24, "rb").read()
assert hashlib.sha256(dati).hexdigest().startswith(SHA_V24), "la v2.4 congelata non corrisponde"
s = dati.decode("utf-8")

SPECIALI = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
            "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
UNICODE = {"\u2212": "$-$", "\u2032": "$'$", "\u2265": "$\\geq$", "\u2264": "$\\leq$", "\u00b1": "$\\pm$",
           "\u03ba": "$\\kappa$", "\u03b5": "$\\varepsilon$", "\u00b2": "$^2$", "\u00a7": "\\S{}",
           "\u2026": "\\ldots{}", "\u00d7": "$\\times$", "\u2192": "$\\rightarrow$"}


def escape(t):
    return "".join(SPECIALI.get(ch, ch) for ch in t)


def inline(t):
    pezzi = re.split(r"(`[^`]*`)", t)
    out = []
    for p in pezzi:
        if p.startswith("`") and p.endswith("`") and len(p) >= 2:
            out.append("\\texttt{" + escape(p[1:-1]) + "}")
            continue
        p = escape(p)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r'"([^"]*)"', r"``\1''", p)
        out.append(p)
    t = "".join(out)
    for a, b in UNICODE.items():
        t = t.replace(a, b)
    return t


def parte(inizio, fine):
    i = s.index(inizio)
    j = s.index(fine, i + 1) if fine else len(s)
    return s[i:j]


def tabella_lunga(b):
    righe = [r for r in b.split("\n") if r.strip()]
    celle = [[c.strip() for c in r.strip().strip("|").split("|")] for r in righe]
    intest, corpo = celle[0], celle[2:]
    assert all(len(r) == len(intest) for r in corpo), "riga della tabella con colonne sbagliate"
    h = " & ".join("\\textbf{" + inline(c) + "}" for c in intest) + " \\\\"
    out = ["{\\footnotesize\\setlength{\\tabcolsep}{4pt}",
           "\\begin{longtable}{L{1.1cm}L{3.7cm}L{5.7cm}L{2.2cm}}",
           "\\toprule", h, "\\midrule", "\\endfirsthead",
           "\\toprule", h, "\\midrule", "\\endhead",
           "\\midrule", "\\endfoot", "\\bottomrule", "\\endlastfoot"]
    for r in corpo:
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
    out += ["\\end{longtable}", "}"]
    return "\n".join(out)


def converti(testo):
    out = []
    for b in [x.strip() for x in re.split(r"\n\s*\n", testo) if x.strip()]:
        if b.startswith("## ") or b == "---" or b.startswith("*End of draft"):
            continue
        if b.startswith("| "):
            out.append(tabella_lunga(b))
            continue
        if re.match(r"^\d+\. ", b):
            voci = []
            for riga in b.split("\n"):
                m = re.match(r"^(\d+)\. (.*)$", riga)
                if m:
                    voci.append([m.group(1), m.group(2)])
                else:
                    voci[-1][1] += " " + riga.strip()
            for n, t in voci:
                out.append("\\noindent\\textbf{" + n + ".} " + inline(t) + "\\par")
            continue
        m = re.match(r"^\*\*([^*]+?)\*\*\s*(.*)$", b, flags=re.S)
        if m:
            out.append("\\paragraph{" + inline(m.group(1).strip()) + "}\n" + inline(m.group(2).strip()))
            continue
        out.append(inline(b))
    return "\n\n".join(out)


NOTA = ("\\emph{Carried over verbatim from the full report (draft v2.4, 23 September 2026), which is released "
        "with this report. Section and appendix numbers in this appendix refer to that report; Appendix~E maps "
        "them to this one.}\n\n")


def scrivi(nome, testo):
    io.open(os.path.join(USCITA, nome), "w", encoding="utf-8", newline="\n").write(testo + "\n")


scrivi("app_A.tex", NOTA + converti(parte("## 4. The falsification campaign", "## 5. ")))

c11 = converti(parte("## 11. Corrections since draft v1.0", "## 12. "))
cD = converti(parte("## Appendix D. Corrections to our earlier statements", None))
scrivi("app_C.tex", NOTA + "\\subsection*{Full report, Section 11: Corrections since draft v1.0}\n\n" + c11
       + "\n\n\\subsection*{Full report, Appendix D: Corrections to our earlier statements}\n\n" + cD)

c12 = converti(parte("## 12. Limitations", "## 13. "))
c13 = converti(parte("## 13. Reproducibility and disclosure", "## Appendix A"))
scrivi("app_D.tex", NOTA + "\\emph{Its paragraph on the use of AI assistants is superseded by the AI usage statement "
       "of this report, which records a second exception (submission 1016734).}\n\n"
       + "\\subsection*{Full report, Section 12: Limitations}\n\n" + c12
       + "\n\n\\subsection*{Full report, Section 13: Reproducibility and disclosure}\n\n" + c13)

MAPPA = [
    ("Abstract; 1. Stance and contributions", "Abstract; Section 1"),
    ("2. The instrument from the outside: public data", "Section 2 (condensed; 2.3(b), the withdrawn regressor defect, is not reproduced)"),
    ("3. The instrument from the inside: warmup measurements", "Not reproduced; H16 to H18 in Appendix A"),
    ("4. The falsification campaign", "Section 3 (condensed) and Appendix A (verbatim)"),
    ("5. GPU-scale replication (warmup)", "Section 3 and Appendix A (H10, H13)"),
    ("6. The main phase: the entry that measures, and the entry that does nothing", "Section 6; predictions in Appendix B"),
    ("7. Can the label be regenerated from its definition? (introduction)", "Section 5, first paragraph; the rule on training data in Appendix H"),
    ("7.1 Our families, translation, self-consistency: three dead ends with ground truth", "Section 3 (Table 1)"),
    ("7.2 The organizers' own families, regenerated; 7.3 What this does and does not show", "Section 5 (Table 2), 5.1 and 5.2; the first run of 7.2 (E12, H32) in Section 3 (Table 1)"),
    ("7.4 The answer-format dependence", "Section 5.3"),
    ("7.5 The same data under the main-phase rule (E13b)", "Section 5.1 and Table J.3"),
    ("8. The instrument on its own hardware (introduction)", "Section 4, first paragraph; compute in Appendix D"),
    ("8.1 The structure of re-execution noise", "Section 4 (Figure 1) and Table J.2"),
    ("8.2 The official score channel does not reproduce its ordering between identical runs", "Section 4, last paragraph (the comparison with our A100 features is not reproduced)"),
    ("8.3 Linear probes on the residual stream, ours and theirs", "Section 3, Length and probes"),
    ("8.4 Per-model score distributions on this hardware", "Not reproduced"),
    ("9. Correspondence with the organizers, and the compute grant", "Not reproduced, except the thanks (Acknowledgments), the question about measurements on the organizers' machine (Limitations and Appendix K) and the rule on training data (Appendix H)"),
    ("10. Recommendations", "Section 7, renumbered: R1 to R3 keep their numbers, R5 is R7 and R6 is R5; R4 (enlarge the labeled training split) is not reproduced; R4 and R6 of this report are new"),
    ("11. Corrections since draft v1.0", "Appendix C"),
    ("12. Limitations", "Limitations (after Section 7) and Appendix D"),
    ("13. Reproducibility and disclosure", "Appendix D and Appendix K; the use of AI assistants also in the AI usage statement"),
    ("Appendix A. Floor algebra", "Appendix F"),
    ("Appendix B. The allocation episode", "Appendix G"),
    ("Appendix C. Pre-registered predictions to be scored against the final evaluation", "Appendix B"),
    ("Appendix D. Corrections to our earlier statements", "Appendix C"),
]
mappa = ["\\section{Section map between the full report and this report}\\label{app:E}",
         "The full report (draft v2.4, 23 September 2026) is released with this report. Its sections correspond "
         "to this report as follows; the sections marked as not reproduced are only in the full report. "
         "Sections 2 to 7 of this report are condensed from the nine-page text given in Appendix J under the same numbers.",
         "", "\\begin{center}\\small\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{L{7.4cm}L{5.8cm}}", "\\toprule",
         "\\textbf{Full report (draft v2.4)} & \\textbf{This report} \\\\", "\\midrule"]
mappa += [inline(a) + " & " + inline(b) + " \\\\" for a, b in MAPPA]
mappa += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]

F = converti(parte("## Appendix A. Floor algebra", "## Appendix B"))
G = converti(parte("## Appendix B. The allocation episode", "## Appendix C"))

H = """\\section{Assets and licenses}\\label{app:H}
\\emph{New in this report; licenses as stated on the Hugging Face and GitHub pages on 23 October 2026.}

\\paragraph{Models.} gpt-oss-120b (OpenAI, Apache-2.0); Olmo-3-7B-Think (Allen Institute for AI, Apache-2.0); DeepSeek-R1-0528-Qwen3-8B (DeepSeek, MIT); Qwen3.5-4B (Qwen, Apache-2.0); Skywork-OR1-Math-7B (Skywork; no license is stated on its model card); phi-4 (Microsoft, MIT), used only for the translations of H22; and gemma-3-27b-it (Google, Gemma Terms of Use) and Olmo-3-7B-Instruct (Allen Institute for AI, Apache-2.0), which our warmup entries loaded in the organizers' container for H16 to H18. The model revisions on the organizers' virtual machine were Qwen3.5-4B 851bf6e, Skywork-OR1-Math-7B fcef970, Olmo-3-7B-Think d97e442, DeepSeek-R1-0528-Qwen3-8B 6e8885a and gpt-oss-120b b5c939d (as recorded on 17 September).

\\paragraph{Data.} The organizers' datasets published under \\texttt{aimo-interp} on Hugging Face: \\texttt{val-sample}, \\texttt{augmented-sample-math}, \\texttt{augmented-sample-math-full-filtered}, \\texttt{augmented-sample-math-agg-filtered}, \\texttt{aimo-interp-challenge-sample-full} and \\texttt{augmented-sample-math-agg}, whose cards state no license, and the main-phase train set \\texttt{train-main-v2}, whose card states the license as ``other'' without further detail; we used its revision 23791d9 of 17 September 2026 (82 rows, 71 labeled), which the organizers extended to 94 rows on 15 October. The problems of the six earlier datasets come from the MATH dataset [7] (MIT license) and from the AIMO 2025 reference problems, as their \\texttt{dataset\\_id} field records (\\texttt{hendrycks-math-train-reference-500}, \\texttt{hendrycks-math-test-reference-100}, \\texttt{aimo-2025-reference}). The website's rule against extra labeled data was commented out on 25 August (the Codabench Terms kept it until at least 16 September) and retired explicitly on 19 September [4]; on 5 September the organizers wrote in Discord \\#qa that participants can use the perturbations and labels they generate themselves ``for anything'' [6]. On 2 October, answering a question in repository issue \\#10, they allowed the publication of self-generated prompts, labels and output-level metrics, and of per-example rows derived from \\texttt{augmented-sample-math-agg} without the problem text or the models' raw outputs; the perturbed problem texts we would release go beyond that exchange, and we will release them only with the organizers' agreement [5].

\\paragraph{Code.} The organizers' repositories on GitHub. \\texttt{aimo-interp/getting-started} (no license stated) contains the evaluation harness, the official uncertainty-feature extraction (\\texttt{uncertainty\\_profile}) with the shipped DeepSeek regressor, and the DeepSeek probe artifact with its inference code; we used the last three [5]. Our model-running bundles include the extraction code and the regressor unchanged, so we will release them without these files, with instructions to fetch them, unless the organizers allow their redistribution. \\texttt{aimo-interp/baselines} (Apache-2.0) contains the organizers' encoding and probe training code; we used three of its files, \\texttt{scripts/probe.py}, \\texttt{prompts/solve.txt} and \\texttt{data/math-robust-agg.csv}. vLLM [8] (Apache-2.0) ran the generation, gating and labeling of Section 5 (version 0.28.0 on the organizers' virtual machine, as recorded on 17 September).
"""

ISTRUZIONI = [
    ("rephrase", "Rewrite the problem in different words while preserving exactly its mathematical content and its final answer."),
    ("rename", "Rename the variables, people, objects or entities in the problem while preserving its mathematical structure and its final answer."),
    ("domain", "Transfer the problem to a different real-world surface domain (a different story, setting and objects) while preserving exactly the same mathematical structure and the same final answer."),
    ("distract", "Insert irrelevant but plausible additional information (extra facts, numbers or context) that does not change the problem or its final answer."),
    ("typos", "Introduce harmless typographical or formatting noise (misspellings, spacing, punctuation, capitalisation) without changing the meaning of the problem or its final answer."),
]
GEN = [
    "You create variants of mathematics competition problems for a robustness benchmark.",
    "Perturbation type: <family>. <family instruction>",
    "The variant must be a complete, self-contained problem with exactly the same final answer as the original. Do not solve the problem. Do not add hints or solutions.",
    "",
    "Here are 5 examples of this perturbation type:",
    "",
    "Example 1",
    "Original: <public original>",
    "Variant: <public variant>",
    "",
    "(Examples 2 to 5 in the same form)",
    "",
    "Now write 10 different variants of the following problem. Make them genuinely different from each other. Output each variant between <variant> and </variant> tags, and nothing else.",
    "",
    "Original: <problem>",
]


def blocco_tt(righe):
    out = []
    for r in righe:
        out.append(inline(r) if r else "~")
    return "\\begin{quote}\\footnotesize\\ttfamily\n" + " \\\\\n".join(out) + "\n\\end{quote}"


I = ("\\section{Prompts}\\label{app:I}\n"
     "\\emph{New in this report; taken verbatim from our scripts \\texttt{vm/e13\\_varianti.py} and \\texttt{vm/e13\\_etichetta.py}, "
     "with the placeholders in angle brackets.}\n\n"
     "\\paragraph{Generator.} gpt-oss-120b at low reasoning effort, temperature 1.0, top\\_p 1.0, at most 4,096 tokens, one request "
     "per problem and family. For each draw, problem and family, five examples of that family are drawn reproducibly from the public "
     "examples (Section 5), excluding the problem itself. Lines in parentheses describe the prompt rather than quote it.\n\n"
     + blocco_tt(GEN) + "\n\nThe family instructions:\n\\begin{itemize}\\setlength{\\itemsep}{0pt}\n"
     + "\n".join("\\item \\texttt{" + f + "}: " + inline(t) for f, t in ISTRUZIONI) + "\n\\end{itemize}\n\n"
     "\\paragraph{Gate and reference answers.} gpt-oss-120b at medium reasoning effort, temperature 1.0, top\\_p 1.0, at most 8,192 "
     "tokens, with this instruction line followed by a blank line and the problem or the variant:\n\n"
     + blocco_tt(["Solve the following problem. Reason step by step, then give only the final answer inside \\boxed{}."]) + "\n\n"
     "\\paragraph{Labeling.} The same instruction line, followed by a blank line and the problem or the variant, was given to "
     "Qwen3.5-4B, Skywork-OR1-Math-7B and Olmo-3-7B-Think in all their series and to DeepSeek in A$'$; DeepSeek received the bare "
     "text in A and B. Sampling: temperature 0.6, top\\_p 0.95, at most 16,384 tokens; ten samples on the base problem and one on each variant.\n")

scrivi("app_EFGH.tex", "\n".join(mappa) + "\n\n"
       + "\\section{Floor algebra}\\label{app:F}\n" + NOTA + F + "\n\n"
       + "\\section{The allocation episode}\\label{app:G}\n" + NOTA + G + "\n\n" + H + "\n\n" + I)
print("scritte app_A, app_C, app_D, app_EFGH (report da 5 pagine)")
