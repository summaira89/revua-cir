"""Pre-registered analysis of the seed study (Seed_Study_Plan_2026-10-24.md).

Primary: difference of arm means of base R@1 (FC+AEP+RV - FC) over seeds
42, 43, 44, two-sided Welch t-test at alpha 0.05 with a 95% CI.
Also reported: per-seed matched differences, arm SDs, and the paired McNemar
test of each new seed pair on the same queries. Secondary (descriptive): fused
R@1 of each FC+AEP+RV seed and its fused-minus-base gain.

Seed 42 comes from the published per-query comparison (ablation_significance.json);
its fused R@1 39.369 is the published deployed-fusion value (Policy_Comparison_CIRR_Results_2026-10-22.md).

Usage:  python scripts/seed_study_analysis.py --dir . --out seed_study_analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from revua_cir.significance import mcnemar_exact  # noqa: E402

NEW_SEEDS = (43, 44)
SEED42_FUSED = 39.369
N_QUERIES = 4181


def load_run(directory: str, seed: int, arm: str) -> dict:
    d = json.load(open(os.path.join(directory, "seed_study_s%d_%s.json" % (seed, arm)), encoding="utf-8"))
    combo = arm.replace("_", "+")
    if d["schema"] != "seed-study-run/1" or d["seed"] != seed or d["combo"] != combo:
        raise ValueError("seed %d %s: wrong file (seed %r, combo %r)" % (seed, arm, d["seed"], d["combo"]))
    if d["n_queries"] != N_QUERIES or sum(d["valid"]) != N_QUERIES:
        raise ValueError("seed %d %s: expected %d valid queries" % (seed, arm, N_QUERIES))
    r1 = 100.0 * sum(d["hits_r1"]) / N_QUERIES
    if abs(r1 - d["metrics"]["recall_at_1"]) > 1e-6:
        raise ValueError("seed %d %s: hits give R@1 %.6f, metrics say %.6f" % (seed, arm, r1, d["metrics"]["recall_at_1"]))
    d["r1"] = r1
    return d


def welch(a, b, alpha=0.05) -> dict:
    """Welch two-sample test of mean(b) - mean(a), with its Welch-Satterthwaite CI."""
    va, vb = statistics.variance(a), statistics.variance(b)
    se2 = va / len(a) + vb / len(b)
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    diff = statistics.mean(b) - statistics.mean(a)
    half = stats.t.ppf(1 - alpha / 2, df) * math.sqrt(se2)
    p = stats.ttest_ind(b, a, equal_var=False).pvalue
    return {"diff_pp": diff, "ci95_pp": [diff - half, diff + half], "t": diff / math.sqrt(se2), "df": df, "p": float(p)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="folder holding the four seed_study_s*.json and ablation_significance.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    pub = json.load(open(os.path.join(args.dir, "ablation_significance.json"), encoding="utf-8"))["comparison"]
    if pub["n_paired_valid"] != N_QUERIES:
        raise ValueError("published seed-42 comparison is not on %d queries" % N_QUERIES)
    fc = {42: pub["r1_a_pct"]}
    full = {42: pub["r1_b_pct"]}
    fused = {42: SEED42_FUSED}
    mcnemar = {}
    for s in NEW_SEEDS:
        a, b = load_run(args.dir, s, "fc"), load_run(args.dir, s, "fc_aep_rv")
        if a["query_ids"] != b["query_ids"]:
            raise ValueError("seed %d: the two arms are not on the same query order" % s)
        fc[s], full[s], fused[s] = a["r1"], b["r1"], 100.0 * sum(b["hits_fused_r1"]) / N_QUERIES
        help_ = sum(1 for x, y in zip(a["hits_r1"], b["hits_r1"]) if x == 0 and y == 1)
        harm = sum(1 for x, y in zip(a["hits_r1"], b["hits_r1"]) if x == 1 and y == 0)
        mcnemar[s] = {"fc_wrong_full_right": help_, "fc_right_full_wrong": harm,
                      "diff_pp": full[s] - fc[s], "p_exact": mcnemar_exact(help_, harm)}

    seeds = sorted(fc)
    a, b = [fc[s] for s in seeds], [full[s] for s in seeds]
    res = {
        "seeds": seeds,
        "base_r1": {"fc": {str(s): fc[s] for s in seeds}, "fc+aep+rv": {str(s): full[s] for s in seeds}},
        "arm_mean": {"fc": statistics.mean(a), "fc+aep+rv": statistics.mean(b)},
        "arm_sd": {"fc": statistics.stdev(a), "fc+aep+rv": statistics.stdev(b)},
        "primary_welch": welch(a, b),
        "matched_diff_pp": {str(s): full[s] - fc[s] for s in seeds},
        "mcnemar_new_seeds": {str(s): v for s, v in mcnemar.items()},
        "secondary_fused": {str(s): {"fused_r1": fused[s], "fused_minus_base_pp": fused[s] - full[s]} for s in seeds},
    }
    w = res["primary_welch"]
    print("seed   FC       FC+AEP+RV  diff    fused   fused-base")
    for s in seeds:
        print("%-5d  %.3f   %.3f     %+.3f  %.3f  %+.3f" % (s, fc[s], full[s], full[s] - fc[s], fused[s], fused[s] - full[s]))
    print("arm means %.3f vs %.3f (SD %.3f vs %.3f)" % (statistics.mean(a), statistics.mean(b), statistics.stdev(a), statistics.stdev(b)))
    print("PRIMARY Welch: diff %+.3f pp, 95%% CI [%+.3f, %+.3f], t %.2f, df %.2f, p %.4f"
          % (w["diff_pp"], w["ci95_pp"][0], w["ci95_pp"][1], w["t"], w["df"], w["p"]))
    for s, m in mcnemar.items():
        print("McNemar seed %d: %+.3f pp, %d vs %d discordant, exact p %.4g"
              % (s, m["diff_pp"], m["fc_wrong_full_right"], m["fc_right_full_wrong"], m["p_exact"]))
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
