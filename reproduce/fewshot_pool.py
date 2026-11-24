"""Appendix C: rebuild the few-shot pool used to generate our perturbation variants (experiments E12 and E13).

The pool holds 621 public examples (original problem, perturbed problem) over five families. Its texts are the
organizers' and are not redistributed here: vm_risultati/e13/fewshot_pool_ids.json gives, for each family and in pool
order, the problem identifier and the SHA-256 of the original and of the variant. This script finds the texts in the six
preliminary datasets (field original_problem, and the questions listed in permutations_causing_decay), checks every
hash and writes gpu/e12_fewshot.json, the file that vm/e13_varianti.py reads.

Usage, from the repository root:  python archive/dati.py   (once)   then   python reproduce/fewshot_pool.py
For each draw (seed s), problem p and family f, vm/e13_varianti.py draws its five examples with
random.Random("s|p|f").sample([e for e in pool[f] if e["problem_id"] != p], 5).
"""
import hashlib
import json
import os

from common import DATA, PRELIMINARY, ROOT, need

IDS = os.path.join(ROOT, "vm_risultati", "e13", "fewshot_pool_ids.json")
OUT = os.path.join(ROOT, "gpu", "e12_fewshot.json")


def h(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    need(IDS, "vm_risultati/e13/fewshot_pool_ids.json is part of the release")
    for name in PRELIMINARY:
        need(os.path.join(DATA, name + ".json"), "run first: python archive/dati.py")
    texts = {}
    for name in PRELIMINARY:
        for r in json.load(open(os.path.join(DATA, name + ".json"), encoding="utf-8")):
            if isinstance(r.get("original_problem"), str):
                texts[h(r["original_problem"])] = r["original_problem"]
            perms = r.get("permutations_causing_decay")
            if isinstance(perms, str):
                try:
                    perms = json.loads(perms)
                except ValueError:
                    perms = None
            for d in perms or []:
                if isinstance(d, dict) and isinstance(d.get("question"), str):
                    texts[h(d["question"])] = d["question"]
    ids = json.load(open(IDS, encoding="utf-8"))
    pool, missing = {}, 0
    for fam, rows in ids.items():
        pool[fam] = []
        for r in rows:
            o, v = texts.get(r["sha256_original"]), texts.get(r["sha256_variant"])
            if o is None or v is None:
                missing += 1
                continue
            pool[fam].append({"problem_id": r["problem_id"], "originale": o, "variante": v})
    total = sum(len(v) for v in ids.values())
    print("few-shot examples rebuilt: %d of %d (%s)" % (total - missing, total, ", ".join("%s %d" % (f, len(v)) for f, v in pool.items())))
    if missing:
        raise SystemExit("some texts were not found: the downloaded datasets differ from those of September 2026")
    text = json.dumps(pool, ensure_ascii=False)
    if os.path.isfile(OUT):   # never overwrite an existing pool: compare it instead
        same = open(OUT, encoding="utf-8").read() == text
        print("%s already exists: %s" % (os.path.relpath(OUT, ROOT), "identical to the rebuilt pool" if same else "DIFFERENT from the rebuilt pool"))
        return
    open(OUT, "w", encoding="utf-8").write(text)
    print("written %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
