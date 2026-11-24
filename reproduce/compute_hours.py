"""Appendix C: accelerator time of the main runs, itemized from the saved logs.

Usage, from the repository root:  python reproduce/compute_hours.py
All runs were on the organizers' machine: two NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (96 GB each).
Each run used one GPU; the two processes of the contention condition shared one GPU, so their pair counts once.
Times come from the logs and include model loading, except for the E13 labels (generation only, under 1% less).
Not in the run logs above, and listed at the end: two complete Qwen3.5-4B runs with the bare prompt, discarded (about
13 GPU-hours, no log; PREREG_E13, second amendment), and the run of Olmo-3-7B-Think A' on card 1 (11.0 GPU-hours,
from its queue log vm_backup/estratto/work/aimo/e13_coda_gpu1c.log, copied from the virtual machine backup): card 0 resumed from its partial file at 96 of 164 problems and
finished the rest, so the A' labels come from both cards.
"""
import glob
import os
import re
from datetime import datetime

from common import E13, E14, ROOT

LOGS = os.path.join(ROOT, "vm_risultati", "vm_logs")


def span_vllm(path):
    t = re.findall(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d)", open(path, encoding="utf-8", errors="replace").read())
    a, b = (datetime.strptime("2026-" + x, "%Y-%m-%d %H:%M:%S") for x in (t[0], t[-1]))
    return (b - a).total_seconds()


def span_utc(path):
    t = re.findall(r"(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d) UTC (\d{4})", open(path, encoding="utf-8", errors="replace").read())
    a, b = (datetime.strptime(x + " " + y, "%a %b %d %H:%M:%S %Y") for x, y in (t[0], t[-1]))
    return (b - a).total_seconds()


def main():
    rows = []
    for d in ("D1", "D2"):
        rows.append(("E13 variants, draw %s (gpt-oss-120b generates, solves and gates)" % d, span_vllm(os.path.join(E13, "log", "var_%s.log" % d))))
    for f in sorted(glob.glob(os.path.join(E13, "log", "etich_*.log"))):
        t = open(f, encoding="utf-8", errors="replace").read()
        m = re.search(r"scritto .*\| (\d+)s\s*$", t, flags=re.M)
        sec = float(m.group(1)) if m else sum(float(x) for x in re.findall(r"varianti in (\d+)s", t))
        bare = "prompt=" not in t.split("\n", 1)[0] and "DeepSeek" not in f   # Qwen runs without the format line were discarded
        note = " (discarded: bare prompt, stopped)" if bare else ("" if m else " (stopped, partial)")
        rows.append(("E13 labels, %s%s" % (os.path.basename(f)[6:-4], note), sec))
    pairs = {}
    for f in sorted(glob.glob(os.path.join(E14, "log", "*.log"))):
        m = re.search(r"load (\d+)s \| gen (\d+)s", open(f, encoding="utf-8", errors="replace").read())
        if not m:
            continue
        k = os.path.basename(f)[:-6]          # without _a / _b: the two processes of one contention pair
        pairs[k] = max(pairs.get(k, 0), int(m.group(1)) + int(m.group(2)))
    rows.append(("E14 re-execution, %d runs on DeepSeek (contention pairs counted once)" % len(pairs), float(sum(pairs.values()))))
    for name, f in (("E16 official score extraction with the cuBLAS workspace variable, card 0", "e16_gpu0.log"),
                    ("E16 official score extraction with the cuBLAS workspace variable, card 1", "e16_gpu1.log"),
                    ("E16 repeat without the cuBLAS workspace variable, card 1", "e16nodet.log")):
        rows.append((name, span_utc(os.path.join(LOGS, f))))
    tot = 0.0
    for name, sec in rows:
        tot += sec
        print("%-72s %8.0f s  %6.2f GPU-hours" % (name, sec, sec / 3600))
    print("%-72s %8.0f s  %6.2f GPU-hours" % ("total in the run logs above", tot, tot / 3600))
    print("not in the run logs above: two discarded bare-prompt Qwen3.5-4B runs, about 13 GPU-hours (no log); the run of")
    print("Olmo-3-7B-Think A' on card 1, 11.0 GPU-hours by its queue log, whose partial file card 0 completed.")
    print("Total on the virtual machine: about %.0f GPU-hours." % (tot / 3600 + 13 + 11.0))
    print("E15 extraction of hidden states (174 problems, two prompts): seconds, see vm_risultati/vm_logs/e15_estrai.log")


if __name__ == "__main__":
    main()
