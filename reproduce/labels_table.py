"""Section 4 and Table 2: agreement of the labels we regenerated with our own protocol (experiment E13).

Usage, from the repository root:  python archive/dati.py   (once)   then   python reproduce/labels_table.py
Inputs: vm_risultati/e13/etich_*.parquet (our saved answers), gpu/out/feature_greedy.parquet (the 164 DeepSeek
labels under the earlier rule), dati/*.json (the six preliminary datasets, to know which labels are published).
"""
import numpy as np

from common import CAP, SHORT, kappa, kappa_ci, label_runs, official_deepseek, regenerated_labels


def pair(r1, r2, truncated):
    a, b = regenerated_labels(r1, truncated), regenerated_labels(r2, truncated)
    common = a.index.intersection(b.index)
    return a[common].astype(int), b[common].astype(int)


def main():
    runs = label_runs()
    off = official_deepseek()
    print("Table 2. Cohen's kappa with 95% percentile bootstrap intervals over problems (2000 resamples).")
    print("A and B: two independent draws of variants. A': draw A's variants re-sampled with a new seed.")
    for model in ["Skywork/Skywork-OR1-Math-7B", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "allenai/Olmo-3-7B-Think", "Qwen/Qwen3.5-4B"]:
        r = runs[model]
        cap = ", ".join("%.1f%%" % (100 * (r[c][r[c].idx >= 0].tok_medio >= CAP).mean()) for c in ("A", "B"))
        robust_a = regenerated_labels(r["A"], "scored")
        print("\n%s | variant chains at the %d-token cap in A, B: %s | robust in A: %.1f%% of %d" % (
            SHORT[model], CAP, cap, 100 * robust_a.mean(), len(robust_a)))
        shares = []
        for t in ("scored", "dropped", "correct"):
            s = regenerated_labels(r["A"], t)
            shares.append("%s %.1f%% of %d" % (t, 100 * s.mean(), len(s)))
        print("   robust share in A, by treatment of truncated variant chains: " + ", ".join(shares))
        print("   problems with no accepted variant, hence no label: " + ", ".join(
            "%s %d" % (c, sum(1 for _, g in r[c].groupby("problem_id") if (g.idx >= 0).sum() == 0)) for c in ("A", "B")))
        for t in ("scored", "dropped", "correct"):
            a, b = pair(r["A"], r["B"], t)
            k, lo, hi = kappa_ci(a, b)
            print("   A and B, truncated variant chains %-7s n=%3d  kappa %.3f [%.2f, %.2f]  robust %.0f%% and %.0f%%" % (
                t, len(a), k, lo, hi, 100 * a.mean(), 100 * b.mean()))
        if "Ap" in r:
            a, b = pair(r["A"], r["Ap"], "scored")
            k, lo, hi = kappa_ci(a, b)
            print("   A and A', new seed                    n=%3d  kappa %.3f [%.2f, %.2f]" % (len(a), k, lo, hi))
        if "DeepSeek" in model:
            for c in ("A", "B", "Ap"):
                ours = regenerated_labels(r[c], "scored")
                m = off.join(ours.rename("ours"), how="inner")
                pub = m[m.published]
                k_all, lo_all, hi_all = kappa_ci(m.ours.astype(int), m.robust.astype(int))
                k_pub, lo_pub, hi_pub = kappa_ci(pub.ours.astype(int), pub.robust.astype(int))
                print("   %-2s and the organizers' labels: published only n=%d kappa %.3f [%.2f, %.2f] | with our derived labels n=%d kappa %.3f [%.2f, %.2f]" % (
                    c, len(pub), k_pub, lo_pub, hi_pub, len(m), k_all, lo_all, hi_all))
            ours, upper = regenerated_labels(r["A"], "scored"), regenerated_labels(r["A"], "correct")
            m = off.join(ours.rename("ours"), how="inner")
            for name, mm in (("published", m[m.published]), ("published and derived", m)):
                pos = mm[mm.robust == 1]
                flipped = pos[~pos.ours]
                only_cap = [p for p in flipped.index if upper.get(p, False)]
                print("   robust in the organizers' labels (%s): %d; draw A calls %d of them non-robust; %d of these change when truncated variants all count as correct" % (
                    name, len(pos), len(flipped), len(only_cap)))
                neg = mm[mm.robust == 0]
                print("   non-robust in the organizers' labels (%s): %d; draw A calls %d of them robust" % (
                    name, len(neg), int(neg.ours.sum())))


if __name__ == "__main__":
    main()
