"""Which RV component carries the fusion gain? A novelty-claim test.

**The question this answers.** The paper's central novelty claim is
*inference-time candidate-level inverse-consistency verification*: score a
retrieved candidate by whether applying the REVERSE edit to it reconstructs
the reference's invariant content. A reviewer's natural challenge is that RV
might be an expensive re-scorer whose benefit comes from anything BUT the
reverse pass -- in which case the novelty claim collapses even though the
numbers survive.

That is testable offline, at zero GPU, because the cache stores all three
cosine terms separately.

**The decomposition** (`reverse_verifier.py`, verified against the cache):

    cos_sim = (1-w)*(0.5*recon_cos + 0.5*invariant_cos) + w*edit_cos
    verification_score = sigmoid(scale*cos_sim + bias)

with `w = sigmoid(edit_weight)` a single learned scalar, recovered from the
cache as **0.4888** (sd 8e-05 across 2200 candidate rows).

* `recon_cos`   -- cos(reverse-edit reconstruction of the candidate,
                   reference invariant embedding). **THIS is the
                   inverse-consistency mechanism.**
* `invariant_cos` -- cos(candidate invariant, reference invariant). A
                   preservation check; no reverse pass involved.
* `edit_cos`    -- cos(candidate editable part, forward text). A forward
                   relevance check; no reverse pass involved.

Only `recon_cos` requires running the edit backwards. If fusion gains
survive with `recon_cos` removed, the novelty claim is in trouble. If they
collapse, the claim is directly evidenced.

**Monotonicity note.** Platt calibration is monotonic, so RANK-based fusion
sees `cos_sim` and `verification_score` identically. Bounded fusion is
score-based and does differ slightly; both are reported.

Zero GPU: reads a cache, writes numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multipositive_analysis import (  # noqa: E402
    N_BOOTSTRAP,
    base_order,
    hit_at_1,
    load_cache,
    relevant_positions,
    _zscore,
)
from revua_cir.significance import (  # noqa: E402
    holm_bonferroni,
    mcnemar_exact,
    paired_bootstrap_ci,
)


def recover_edit_weight(recs):
    """Recover the learned `w = sigmoid(edit_weight)` from the cache.

    Solving cos_sim = (1-w)*base + w*edit for w at every candidate row.
    Returns (mean, sd, n). A large sd would mean the decomposition does not
    hold and every result below would be meaningless, so the caller checks.
    """
    ws = []
    for r in recs:
        for i in range(len(r["cos_sim"])):
            base = 0.5 * r["recon_cos"][i] + 0.5 * r["invariant_cos"][i]
            edit = r["edit_cos"][i]
            if abs(edit - base) > 1e-6:
                ws.append((r["cos_sim"][i] - base) / (edit - base))
    if not ws:
        return None, None, 0
    m = sum(ws) / len(ws)
    sd = math.sqrt(sum((x - m) ** 2 for x in ws) / len(ws))
    return m, sd, len(ws)


def component_score(rec, w, use_recon=True, use_invariant=True, use_edit=True):
    """Rebuild cos_sim with components switched off.

    A disabled component is DROPPED and the remaining weights renormalised,
    rather than zero-filled. Zero-filling would inject a constant that
    shifts every candidate equally under z-scoring but distorts the relative
    weighting of what remains; renormalising keeps the surviving terms in
    their trained proportion to each other, which is the fair ablation.
    """
    parts, weights = [], []
    if use_recon:
        parts.append(rec["recon_cos"])
        weights.append(0.5 * (1.0 - w))
    if use_invariant:
        parts.append(rec["invariant_cos"])
        weights.append(0.5 * (1.0 - w))
    if use_edit:
        parts.append(rec["edit_cos"])
        weights.append(w)
    if not parts:
        return None
    total = sum(weights)
    weights = [x / total for x in weights]
    n = len(parts[0])
    return [sum(wt * p[i] for wt, p in zip(weights, parts)) for i in range(n)]


def bounded_order(base_scores, rv_scores, lam):
    zb = _zscore(base_scores)
    zr = _zscore(rv_scores)
    s = [zb[i] + lam * math.tanh(zr[i]) for i in range(len(zb))]
    return sorted(range(len(s)), key=lambda i: (-s[i], i))


def rank_fuse_order_from(base_scores, rv_scores):
    n = len(base_scores)
    b = sorted(range(n), key=lambda i: (-base_scores[i], i))
    r = sorted(range(n), key=lambda i: (-rv_scores[i], i))
    br = {p: k for k, p in enumerate(b)}
    rr = {p: k for k, p in enumerate(r)}
    return sorted(range(n), key=lambda i: (2.0 * br[i] + rr[i], i))


def evaluate_variant(recs, w, lam, use_recon, use_invariant, use_edit,
                     fusion="bounded"):
    """Paired base-vs-fused comparison for one component configuration."""
    base_hits, new_hits = [], []
    for rec in recs:
        rel = relevant_positions(rec)
        rv = component_score(rec, w, use_recon, use_invariant, use_edit)
        if rv is None:
            return None
        if fusion == "bounded":
            order = bounded_order(rec["base_score"], rv, lam)
        elif fusion == "rank":
            order = rank_fuse_order_from(rec["base_score"], rv)
        else:
            raise ValueError("unknown fusion %r" % fusion)
        base_hits.append(hit_at_1(base_order(rec), rel))
        new_hits.append(hit_at_1(order, rel))

    n = len(base_hits)
    help_ = sum(1 for b, x in zip(base_hits, new_hits) if not b and x)
    harm = sum(1 for b, x in zip(base_hits, new_hits) if b and not x)
    boot = paired_bootstrap_ci(base_hits, new_hits, n_replicates=N_BOOTSTRAP,
                               seed=0)
    return {
        "n": n,
        "base_pct": 100.0 * sum(base_hits) / n,
        "new_pct": 100.0 * sum(new_hits) / n,
        "delta_pp": 100.0 * (sum(new_hits) - sum(base_hits)) / n,
        "help": help_,
        "harm": harm,
        "p": mcnemar_exact(help_, harm),
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
    }


VARIANTS = [
    ("full RV (recon+invariant+edit)", True, True, True),
    ("NO recon  (invariant+edit)", False, True, True),
    ("NO invariant (recon+edit)", True, False, True),
    ("NO edit   (recon+invariant)", True, True, False),
    ("recon ONLY", True, False, False),
    ("invariant ONLY", False, True, False),
    ("edit ONLY", False, False, True),
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Which RV component carries the fusion gain?")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--fusion", default="bounded", choices=("bounded", "rank"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    recs = load_cache(args.cache)
    w, sd, n_rows = recover_edit_weight(recs)
    if w is None:
        raise SystemExit("ABORT: could not recover edit_weight from this cache")
    if sd > 1e-3:
        raise SystemExit(
            "ABORT: edit_weight is not constant (sd=%.2e over %d rows). The "
            "documented decomposition does not hold for this cache, so every "
            "ablation below would be meaningless." % (sd, n_rows))

    print("=" * 78)
    print("RV component ablation: %s" % args.cache)
    print("records: %d   fusion: %s   lambda: %s" % (len(recs), args.fusion, args.lam))
    print("recovered edit_weight w = %.6f  (sd %.2e over %d candidate rows)"
          % (w, sd, n_rows))
    print("  reconstruction branch weight = %.4f  (recon %.4f + invariant %.4f)"
          % (1 - w, 0.5 * (1 - w), 0.5 * (1 - w)))
    print("  edit branch weight           = %.4f" % w)
    print("=" * 78)
    print()
    print("  %-32s %8s %8s %7s %6s %6s %11s"
          % ("variant", "base", "fused", "delta", "help", "harm", "p"))

    results, pvals, labels = {}, [], []
    for label, ur, ui, ue in VARIANTS:
        res = evaluate_variant(recs, w, args.lam, ur, ui, ue, args.fusion)
        results[label] = res
        pvals.append(res["p"])
        labels.append(label)
        print("  %-32s %8.2f %8.2f %+7.2f %6d %6d %11.4g"
              % (label, res["base_pct"], res["new_pct"], res["delta_pp"],
                 res["help"], res["harm"], res["p"]))

    keep = holm_bonferroni(pvals, alpha=0.05)
    print()
    print("  Holm-Bonferroni across the %d variants (alpha=0.05):" % len(pvals))
    for label, k, p in zip(labels, keep, pvals):
        print("    %-32s p=%-11.4g %s" % (label, p, "SURVIVES" if k else "-"))

    full = results["full RV (recon+invariant+edit)"]["delta_pp"]
    norecon = results["NO recon  (invariant+edit)"]["delta_pp"]
    print()
    print("  " + "-" * 74)
    print("  NOVELTY-CLAIM READING")
    print("  " + "-" * 74)
    print("  full RV delta          : %+.2f pp" % full)
    print("  without recon_cos      : %+.2f pp" % norecon)
    print("  attributable to recon  : %+.2f pp" % (full - norecon))
    if abs(full) > 1e-9:
        print("  share of the gain      : %.1f%%" % (100.0 * (full - norecon) / full))
    print()
    print("  recon_cos is the ONLY term requiring the reverse pass. If removing")
    print("  it preserves the gain, the inverse-consistency novelty claim is not")
    print("  supported by these data, whatever the headline numbers say.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"cache": str(args.cache), "fusion": args.fusion,
                       "lam": args.lam, "edit_weight": w, "edit_weight_sd": sd,
                       "variants": results,
                       "holm_survives": dict(zip(labels, keep))}, fh, indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
