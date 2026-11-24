"""Shared helpers for the scripts in reproduce/. Inputs are read from the working folder: archive/ in the public
repository, the root itself in the author's workspace."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "archive") if os.path.isdir(os.path.join(REPO, "archive")) else REPO
E13 = os.path.join(ROOT, "vm_risultati", "e13")
E14 = os.path.join(ROOT, "vm_risultati", "e14")
OFFICIAL_DS = os.path.join(ROOT, "gpu", "out", "feature_greedy.parquet")
PROBE_OOF = os.path.join(ROOT, "vm_risultati", "e15_oof_sonda_2026-09-23.npz")
DATA = os.path.join(ROOT, "dati")
PRELIMINARY = ["aimo-interp-challenge-sample-full", "val-sample", "augmented-sample-math-agg",
               "augmented-sample-math-agg-filtered", "augmented-sample-math", "augmented-sample-math-full-filtered"]
DEEPSEEK_NAMES = {"qwen3-8b:low", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
CAP = 16384      # max new tokens in our label regeneration (E13)
BASE_OK = 6      # base condition: at least 6 of 10 base samples correct
SHORT = {"Qwen/Qwen3.5-4B": "Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B": "Skywork",
         "allenai/Olmo-3-7B-Think": "Olmo", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "DeepSeek"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def need(path, hint):
    if not os.path.exists(path):
        sys.exit("missing input: %s\n%s" % (os.path.relpath(path, REPO), hint))


def kappa(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def kappa_ci(a, b, n_boot=2000, seed=20260908):
    """Percentile bootstrap over problems, as in the original analysis (vm/e13_analisi.py)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    rng = np.random.RandomState(seed)
    vals = [kappa(a[i], b[i]) for i in (rng.randint(0, len(a), len(a)) for _ in range(n_boot))]
    vals = [v for v in vals if not np.isnan(v)]
    return kappa(a, b), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, int)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def label_runs():
    """{model: {condition: DataFrame}} for the final E13 label files (A, B, Ap)."""
    need(E13, "the E13 outputs are part of the release (vm_risultati/e13)")
    runs = {}
    for f in sorted(glob.glob(os.path.join(E13, "etich_*.parquet"))):
        if "parziale" in f:
            continue
        d = pd.read_parquet(f)
        runs.setdefault(d.model.iloc[0], {})[d["cond"].iloc[0]] = d
    return runs


def regenerated_labels(df, truncated="scored"):
    """Our label per problem: base condition and every accepted variant answered correctly.

    truncated: what to do with a variant whose chain reached the token cap: 'scored' (the protocol: the chain is
    scored by the last boxed answer it contains, and counts as wrong without one), 'dropped', or 'correct' (every
    truncated variant counts as correct)."""
    out = {}
    for pid, g in df.groupby("problem_id"):
        base = g[g.idx == -1]
        if len(base) == 0:
            continue
        v = g[g.idx >= 0]
        at_cap = v.tok_medio >= CAP
        ok = v.n_corrette > 0
        if truncated == "correct":
            ok = ok | at_cap
        elif truncated == "dropped":
            ok = ok[~at_cap]
        if len(ok) == 0:
            continue
        out[pid] = bool(int(base.iloc[0].n_corrette) >= BASE_OK and bool(ok.all()))
    return pd.Series(out, dtype=bool)


def official_deepseek():
    """The 164 DeepSeek labels under the earlier rule, with a flag for the 138 that the data publish."""
    need(OFFICIAL_DS, "gpu/out/feature_greedy.parquet is part of the release")
    for name in PRELIMINARY:
        need(os.path.join(DATA, name + ".json"), "run first: python archive/dati.py   (downloads the six preliminary datasets)")
    lab = pd.read_parquet(OFFICIAL_DS)[["problem_id", "robust"]].copy()
    published = set()
    for name in PRELIMINARY:
        for r in json.load(open(os.path.join(DATA, name + ".json"), encoding="utf-8")):
            if r.get("model_id") in DEEPSEEK_NAMES and r.get("model_is_robust") is not None:
                published.add(r["problem_id"])
    lab["published"] = lab.problem_id.isin(published)
    return lab.set_index("problem_id")
