"""Costruisce il report da 5 pagine nel modello NeurIPS 2026 (28/10/2026).

Uso (dalla cartella report_5p): python costruisci_5p.py
1. copia in latex/ lo stile e la checklist ufficiali scaricati in ../report_9p/neurips_2026/;
2. converte REPORT_5P_bozza_2026-10-28.md in latex/main.tex (traduce solo la marcatura);
3. scrive latex/app_J.tex: le Sections 2-7 della bozza da 9 pagine verificata
   (../report_9p/REPORT_9P_bozza_2026-10-22.md), senza le tabelle, che stanno nel testo
   principale identiche; l'unica modifica e' nel par. 6, dove il ricaricamento del 28/10
   sostituisce il segnaposto;
4. scrive latex/app_K.tex: gli invii Codabench (../dati/invii_dan23rr_2026-10-28.json),
   gli hash dei bundle ricalcolati da ../sonde/ e il rilascio.
Le appendici A, C, D, E-I le scrive appendici_5p.py, la checklist checklist_compila_5p.py.
"""
import hashlib
import io
import json
import os
import re
import shutil

QUI = os.path.dirname(os.path.abspath(__file__))
SORGENTE = os.path.join(QUI, "REPORT_5P_bozza_2026-10-28.md")
NOVE = os.path.join(QUI, "..", "report_9p", "REPORT_9P_bozza_2026-10-22.md")
INVII = os.path.join(QUI, "..", "dati", "invii_dan23rr_2026-10-28_sera.json")
SONDE = os.path.join(QUI, "..", "sonde")
USCITA = os.path.join(QUI, "latex")
os.makedirs(USCITA, exist_ok=True)
for f in ("neurips_2026.sty", "checklist.tex"):
    shutil.copyfile(os.path.join(QUI, "..", "report_9p", "neurips_2026", f), os.path.join(USCITA, f))

SPECIALI = {"\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", "_": "\\_",
            "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
UNICODE = {"\u2212": "$-$", "\u2032": "$'$", "\u2265": "$\\geq$", "\u2264": "$\\leq$", "\u00d7": "$\\times$",
           "\u2026": "\\ldots{}", "\u00b1": "$\\pm$", "\u2192": "$\\rightarrow$", "\u03ba": "$\\kappa$",
           "\u03b5": "$\\varepsilon$", "\u00b2": "$^2$", "\u00a7": "\\S{}"}


def escape(t):
    return "".join(SPECIALI.get(ch, ch) for ch in t)


def inline(t):
    """Marcatura in linea: `codice`, **grassetto**, virgolette, caratteri unicode."""
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


def tabella(righe, didascalia, etichetta, colonne, piccolo="\\small"):
    celle = [[c.strip() for c in r.strip().strip("|").split("|")] for r in righe]
    intest, corpo = celle[0], celle[2:]
    assert all(len(r) == len(intest) for r in corpo), "riga della tabella con colonne sbagliate"
    s = ["\\begin{table}[t]", "\\caption{" + inline(didascalia) + "}", "\\label{" + etichetta + "}", "\\centering",
         piccolo, "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{" + colonne + "}", "\\toprule",
         " & ".join("\\textbf{" + inline(c) + "}" for c in intest) + " \\\\", "\\midrule"]
    for r in corpo:
        s.append(" & ".join(inline(c) for c in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(s)


COLONNE = {
    "tab:campaign": "L{3.7cm}L{1.3cm}L{5.5cm}L{2.3cm}",
    "tab:noise": "L{4.6cm}L{0.9cm}L{2.2cm}L{2.0cm}L{1.6cm}",
    "tab:label": "L{1.5cm}L{1.1cm}L{2.1cm}L{2.1cm}L{2.2cm}L{3.0cm}",
}
ETICHETTE = {"1": "tab:campaign", "2": "tab:noise", "3": "tab:label"}


def elenco(linee):
    voci, cur = [], None
    for l in linee:
        if l.startswith("- "):
            if cur is not None:
                voci.append(cur)
            cur = l[2:]
        else:
            cur += " " + l.strip()
    voci.append(cur)
    return "\\begin{itemize}\n" + "\n".join("\\item " + inline(v) for v in voci) + "\n\\end{itemize}"


def e_elenco(b):
    linee = b.splitlines()
    return linee[0].startswith("- ") and all(l.startswith("- ") or l.startswith("  ") for l in linee)


def paragrafo(b):
    m = re.match(r"^\*\*([^*]+?)\*\*\s*(.*)$", b, flags=re.S)
    if m:
        return "\\paragraph{" + inline(m.group(1).strip()) + "}\n" + inline(m.group(2).strip())
    return inline(b)


# ---------------------------------------------------------------- testo principale
testo = io.open(SORGENTE, encoding="utf-8").read()
blocchi = re.split(r"\n\s*\n", testo.strip())
titolo = blocchi[0].lstrip("# ").strip()
corpo = []
riassunto = None
in_appendice = False
lettera = None
i = 1
while i < len(blocchi):
    b = blocchi[i].strip()
    i += 1
    if b.startswith("**Daniel Culotta**") or (b.startswith("*Five-page version") and b.endswith("*")):
        continue
    if b == "---":
        continue
    if b.startswith("## Abstract"):
        riassunto = inline(blocchi[i].strip())
        i += 1
        continue
    m = re.match(r"^## (\d+) (.+)$", b)
    if m:
        corpo.append("\\section{" + inline(m.group(2)) + "}\\label{sec:" + m.group(1) + "}")
        continue
    if b == "## Limitations":
        corpo.append("%%%%%% FINE DEL TESTO PRINCIPALE: le 5 pagine finiscono qui %%%%%%\n\\label{fine-testo}\n"
                     "\\section*{Limitations}")
        continue
    if b == "## AI usage statement":
        corpo.append("\\section*{AI usage statement}")
        continue
    if b.startswith("## Acknowledgments"):
        ack = inline(blocchi[i].strip())
        i += 1
        corpo.append("\\begin{ack}\n" + ack + "\n\\end{ack}")
        continue
    if b.startswith("## References"):
        voci = []
        while i < len(blocchi) and blocchi[i].strip().startswith("["):
            voci.append(inline(blocchi[i].strip()))
            i += 1
        corpo.append("\\section*{References}\n{\\small\n" + "\n\n".join(voci) + "\n}")
        continue
    m = re.match(r"^## Appendix ([A-Z])\. (.+)$", b)
    if m:
        if not in_appendice:
            corpo.append("%%%%%% FINE DEI RIFERIMENTI %%%%%%\n\\clearpage\n\\appendix\n\\raggedright")
            in_appendice = True
        lettera = m.group(1)
        corpo.append("\\section{" + inline(m.group(2)) + "}\\label{app:" + lettera + "}")
        continue
    if b.startswith("[Generated from the full report"):
        corpo.append("\\input{app_" + lettera + "}")
        continue
    if b.startswith("[Appendices E to K are generated"):
        corpo.append("\\input{app_EFGH}\n\n\\input{app_J}\n\n\\input{app_K}")
        continue
    m = re.match(r"^\*\*Table (\d)\.\*\* (.+)$", b, flags=re.S)
    if m:
        n = m.group(1)
        righe = blocchi[i].strip().splitlines()
        i += 1
        didasc = m.group(2).strip()
        if i < len(blocchi) and blocchi[i].strip().startswith("\\*"):
            didasc += " (*) " + blocchi[i].strip()[2:].strip()
            i += 1
        corpo.append(tabella(righe, didasc, ETICHETTE[n], COLONNE[ETICHETTE[n]],
                             "\\footnotesize" if n in ("1", "3") else "\\small"))
        continue
    if e_elenco(b):
        corpo.append(elenco(b.splitlines()))
        continue
    corpo.append(paragrafo(b))
assert riassunto and in_appendice

PREAMBOLO = r"""\documentclass{article}

%% Report da 5 pagine del 28/10/2026, convertito da REPORT_5P_bozza_2026-10-28.md con costruisci_5p.py.
%% Le 5 pagine sono il testo dal titolo alla fine della Section 7 (etichetta fine-testo);
%% limitazioni, dichiarazione sull'uso dell'IA, riferimenti e appendici non contano (regole del 28/10).
\usepackage[preprint]{neurips_2026}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\usepackage{amsfonts}
\usepackage{microtype}
\usepackage{xcolor}
\raggedbottom

\title{TITOLO}

\author{%
  Daniel Culotta \\
  Independent researcher (Codabench: \texttt{dan23rr})
}

\begin{document}

\maketitle

\begin{abstract}
RIASSUNTO
\end{abstract}

"""
PREAMBOLO = PREAMBOLO.replace("TITOLO", inline(titolo)).replace("RIASSUNTO", riassunto)
FINE = r"""

\newpage
\input{checklist_compilata.tex}

\end{document}
"""
io.open(os.path.join(USCITA, "main.tex"), "w", encoding="utf-8", newline="\n").write(PREAMBOLO + "\n\n".join(corpo) + FINE)

# ---------------------------------------------------------------- Appendice J
nove = io.open(NOVE, encoding="utf-8").read()
a = nove.index("## 2 The task, the labels and the public data")
z = nove.index("## 8 Limitations")
parte9 = nove[a:z]
VECCHIO = re.compile(r"we resubmitted V8 unchanged, with an all-robust constant as control, on \[TBD: date and scores\..*?"
                     r"report that failure and the constant's score instead\]\.", flags=re.S)
assert len(VECCHIO.findall(parte9)) == 1, "segnaposto del par. 6 non trovato una volta sola"
NUOVO = ("on 28 October we resubmitted V8 unchanged (Codabench submissions 1016713, Main, and 1016730, Small), "
         "with an all-robust constant as control (1016732 and 1016735). Called once per problem through its legacy "
         "signature `are_robust(model_id, problems)`, V8 again predicted robust for every item, as its logged counts "
         "show, and scored 10/21 and 7/14, the robust base rates; the constant scored the same.")
parte9 = VECCHIO.sub(lambda _: NUOVO, parte9)
FINALE = "On the final test set: [TBD after 1 November: nominated submissions and scores]."
assert parte9.count(FINALE) == 1, "segnaposto del test finale non trovato una volta sola"
parte9 = parte9.replace(FINALE, "For the final test we nominated V8 (1016713) and the all-robust constant (1016732), both on "
                        "the Main track; their scores were not yet available when we submitted this report.")
PRE ="and the change of rule adds only 0.01 to 0.11 when both rules are read on the same problems"
assert parte9.count(PRE) == 1, "frase del par. 5.1 non trovata una volta sola"
parte9 = parte9.replace(PRE, "and, post hoc, the change of rule adds only 0.01 to 0.11 when both rules are read on the same problems")
DOVE = {"1": "Section 3", "2": "Section 4", "3": "Section 5"}
J = ["\\section{Extended version of Sections 2 to 7}\\label{app:J}",
     "\\emph{This appendix gives Sections 2 to 7 in the longer form of our nine-page draft of 22 October 2026, which the "
     "main text condenses; section numbers and references are those of the main text. Tables 1 to 3 are not "
     "repeated: they are in Sections 3, 4 and 5, slightly condensed (Appendix A gives every entry of Table 1). Two "
     "changes from the nine-page draft: in Section 6 the resubmission of 28 October and our nominations for the final "
     "test replace two placeholders, and in Section 5.1 a comparison that was not pre-registered is marked post hoc.}"]
bl = [x.strip() for x in re.split(r"\n\s*\n", parte9.strip()) if x.strip()]
k = 0
tabelle_saltate = 0
while k < len(bl):
    b = bl[k]
    k += 1
    m = re.match(r"^## (\d+) (.+)$", b)
    if m:
        J.append("\\subsection*{J." + m.group(1) + "\\quad " + inline(m.group(2)) + "}")
        continue
    m = re.match(r"^\*\*Table (\d)\.\*\* ", b)
    if m:
        assert bl[k].startswith("| "), "manca la tabella dopo la didascalia"
        k += 1
        if k < len(bl) and bl[k].startswith("\\*"):
            k += 1
        J.append("\\emph{(Table " + m.group(1) + " is in " + DOVE[m.group(1)] + ".)}")
        tabelle_saltate += 1
        continue
    assert not b.startswith("| "), "tabella senza didascalia"
    if e_elenco(b):
        J.append(elenco(b.splitlines()))
        continue
    J.append(paragrafo(b))
assert tabelle_saltate == 3
io.open(os.path.join(USCITA, "app_J.tex"), "w", encoding="utf-8", newline="\n").write("\n\n".join(J) + "\n")

# ---------------------------------------------------------------- Appendice K
HASH = {
    "solution-v8-main.zip": "7b70f15c51752dcf4285e477d58c541d0ecf052d833c325c0702dbbd054ffb5f",
    "solution-v8-small.zip": "0b83b55f94dc16b9852a7c2b35d71ac5787e43d81afd6396dee00fb5e011e500",
    "solution-costante-main.zip": "35c47bbef717a5d4c5b6c7786bddbab9d87a02483e4f417cd34fd4ecf669c67d",
    "solution-costante-small.zip": "f798c2979b136a84c22e6d994bf0228bbfeffa044ac68f7868e13859b4c6075a",
}
for nome, h in HASH.items():
    assert hashlib.sha256(open(os.path.join(SONDE, nome), "rb").read()).hexdigest() == h, nome
BUNDLE = [
    ("solution-v8-main.zip", "V8: a single `solution.py` defining `are_robust(model_id, problems)`, which returns `[len(p) >= 98 for p in problems]`. Select only the Main task."),
    ("solution-v8-small.zip", "The same `solution.py`, with the empty marker file `small.txt` that the Small task requires. Select only the Small task."),
    ("solution-costante-main.zip", "The all-robust constant: a single `solution.py` whose `are_robust` returns `True` for every problem, written to accept the argument orders of both official descriptions of the interface and the legacy two-argument call. Select only the Main task."),
    ("solution-costante-small.zip", "The same `solution.py`, with `small.txt`. Select only the Small task."),
]
d = json.load(io.open(INVII, encoding="utf-8"))
invii = {x["id"]: x for x in d["invii"]}
assert len(invii) == 33 and all(v["fase"] in (26647, 30038) for v in invii.values())
ITEM = {35430: 17, 35766: 21, 35767: 19, 36237: 21, 36241: 14}
TASK = {35430: "warmup", 35766: "Main (original)", 35767: "Small (original)", 36237: "Main", 36241: "Small", None: "both tasks (parent)"}
DESCR = [
    (907310, "Null probe, every item non-robust; floor pre-registered at 8/17 (full report, Section 3.1)."),
    (907677, "Official baseline, unchanged, instrumented to log its runtime (Appendix D)."),
    (907967, "Early version of the measuring entry, with a time budget, as its released bundle documents; its logs give the container's GPU and library versions (Appendix D)."),
    (909567, "Later version of the same entry, as its released bundle documents."),
    (909843, "V4, the allocation entry, submitted once as a measurement against its pre-registered rule (Appendix G)."),
    (909853, "v5: every model measured by itself (H16, Appendix A)."),
    (909976, "v6: H16 with a gpt-oss threshold calibrated on `val-sample` (H17); with the next three, the four executions of H18."),
    (909994, "v6, re-executed (H17, H18)."),
    (910002, "v6, re-executed (H17, H18)."),
    (910013, "v6, re-executed (H17, H18)."),
    (912096, "v6, the measuring entry, on the original Main task (Section 6)."),
    (912097, "v6 on the original Small task (Section 6)."),
    (924968, "V8 Main bundle, sent to both original tasks: parent of 924969 and 924970."),
    (924969, "V8 on the original Main task (Section 6)."),
    (924970, "V8 Main bundle on the Small task, rejected before import because it lacks `small.txt` (coverage 0)."),
    (924972, "V8 Small bundle, sent to both original tasks: parent of 924973 and 924974."),
    (924973, "V8 Small bundle on the Main task, rejected before import because of `small.txt` (coverage 0)."),
    (924974, "V8 on the original Small task (Section 6)."),
    (925822, "V8 Main bundle again, Main task only, uploaded by the AI assistant without a specific instruction (AI usage statement)."),
    (936802, "V8 on the replacement Main task (Section 6, Appendix B)."),
    (936806, "V8 on the replacement Small task (Section 6, Appendix B)."),
    (1016713, "V8, unchanged, on the current validation task, after the change of 15 October (Section 6); nominated for the final test, Main track."),
    (1016730, "V8, unchanged, on the current validation task, after the change of 15 October (Section 6)."),
    (1016732, "All-robust constant, control (Section 6); nominated for the final test, Main track."),
    (1016733, "Constant Small bundle: parent of 1016734 and 1016735."),
    (1016734, "Constant Small bundle sent to the Main task in error (AI usage statement); rejected before import (\"small.txt is not allowed for a main-track submission\"), as 924973 was on the original Main task."),
    (1016735, "All-robust constant, control (Section 6)."),
]
NON_DESCRITTI = [928172, 928173, 928174, 928175, 929013, 929014]
assert sorted([x for x, _ in DESCR] + NON_DESCRITTI) == sorted(invii), "elenco degli invii incompleto"
for x in NON_DESCRITTI:
    assert invii[x]["stato"] == "Cancelled" and invii[x]["padre"] in (924968, 924972)


def punteggio(v):
    if v["task"] is None:
        return "parent"
    if v["punteggio"] is None:
        return "[TBD]" if v["stato"] in ("Submitted", "Submitting", "Running") else "none"
    p = [float(x) for x in v["punteggio"].split("/")]
    if p[1] == 0.0:
        return "coverage 0"
    n = ITEM[v["task"]]
    kk = round(p[0] * n)
    assert abs(kk / n - p[0]) < 0.0006, (v["id"], p[0], kk, n)
    return "%d/%d" % (kk, n)


righe = []
for x, testo_k in DESCR:
    v = invii[x]
    righe.append(" & ".join([str(x), TASK[v["task"]], v["creato"][:10], punteggio(v), inline(testo_k)]) + " \\\\")
K = ["\\section{Codabench submissions, bundles and release}\\label{app:K}",
     "\\paragraph{Submissions.} Every submission under our Codabench account (user \\texttt{dan23rr}) in both phases, "
     "read from the Codabench API on 28 October 2026. Accuracy is given as items correct over items; a parent "
     "submission sends one bundle to both tasks, with one child per task it was evaluated on (the parents of 13 "
     "September also carry the cancelled children listed below the table).",
     "{\\footnotesize\\setlength{\\tabcolsep}{4pt}",
     "\\begin{longtable}{L{1.3cm}L{1.9cm}L{1.7cm}L{1.5cm}L{5.9cm}}",
     "\\toprule",
     "\\textbf{ID} & \\textbf{Task} & \\textbf{Date (UTC)} & \\textbf{Accuracy} & \\textbf{What it is} \\\\",
     "\\midrule", "\\endfirsthead",
     "\\toprule",
     "\\textbf{ID} & \\textbf{Task} & \\textbf{Date (UTC)} & \\textbf{Accuracy} & \\textbf{What it is} \\\\",
     "\\midrule", "\\endhead",
     "\\bottomrule", "\\endlastfoot"] + righe + ["\\end{longtable}", "}",
     "Also under our account, and not described in this report: " + ", ".join(str(x) for x in NON_DESCRITTI[:-1])
     + " and " + str(NON_DESCRITTI[-1]) + ", children of 924968 and 924972 created on 16 September and cancelled, "
     "without scores.",
     "\\paragraph{Bundles.} Our entry on the current tasks is V8, with the all-robust constant as control; we "
     "nominated 1016713 and 1016732 for the final test, on the Main track. "
     "Neither loads a model or reads data at run time. V8's two parameters were fixed on the six public datasets, "
     "which are its training data; Appendix D (full report, Section 13) gives them, the scripts that recompute them "
     "and the instructions to reproduce its score. SHA-256 of each bundle:",
     "\\begin{itemize}"]
for nome, testo_b in BUNDLE:
    K.append("\\item \\texttt{" + escape(nome) + "}: " + inline(testo_b) + "\\\\\n{\\footnotesize\\texttt{" + HASH[nome] + "}}")
K += ["\\end{itemize}",
      "The V8 docstring, frozen with the bundle on 13 September, contains three statements that this report corrects: "
      "it calls 98 the median of the character counts (it is the lower median) and gives the rule's balanced accuracy "
      "as 0.658, the figure for a threshold of 99 (Appendix C, item 37); and it says that the official score changes "
      "sign between identical runs, quoting AUROC 0.513 and 0.398 with robust as the positive class on the raw score, "
      "where Section 4 gives the same runs as 0.487 and 0.602 in the regressor's own orientation and finds the "
      "difference not significant.",
      "\\paragraph{Reproducing our score.} Upload each bundle with only its own task selected: the Main bundle on "
      "task 36237, the Small bundle on task 36241. No model and no data are needed. On the current validation tasks the "
      "expected results are those of 1016713 and 1016730, 10/21 and 7/14; Appendix D gives the instructions and expected results "
      "of 23 September, before the change of 15 October.",
      "\\paragraph{Release.} Every pre-registration, outcome file and analysis script, the extraction outputs, the "
      "submission bundles, our snapshots of the Codabench leaderboard and the full report (draft v2.4) will be released "
      "with this report at https://github.com/Dan23RR/aimo-instrument, which keeps the layout of our working folder so "
      "that the file paths cited here are valid there, with other participants' names, usernames and passages about "
      "their entries redacted; the "
      "bundles without the organizers' files that state no license, and the perturbed problem texts only with the "
      "organizers' agreement (Appendix H). On 7 September we asked the organizers whether measurements taken on their "
      "virtual machine may appear in this report and undertook to share our results with them, and we repeated the "
      "question on 2 November; we have not yet received an answer, and the results have not yet been shared. Deviations from the pre-registrations are listed "
      "in Appendix D."]
io.open(os.path.join(USCITA, "app_K.tex"), "w", encoding="utf-8", newline="\n").write("\n\n".join(K) + "\n")
print("scritti latex/main.tex (%d blocchi), app_J.tex (%d blocchi), app_K.tex (%d invii descritti)"
      % (len(corpo), len(J), len(DESCR)))
