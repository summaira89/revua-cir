"""Reviewer-requested follow-up analyses, amendment v3.1 §20 (IVC referee report). CPU only.

Each section implements one subsection of §20, which fixed every choice before this
file existed; nothing here may be tuned after seeing a result.

  §20.1 M3a  the reconstruction increment: two paired contrasts at fixed weights, Holm
  §20.2 M2a  training-free second signals on the deployed features (T, R, TR)
  §20.3 M8   lambda retuned per window width; latency shares recomputed from the records
  §20.4 M6.1 numerical spread diagnostics and fp16 / relative-noise perturbations
  §20.5 m7   AUROC of base_entropy + rv_margin against base_entropy
  §20.7 X6   EDIR annotated-negative counts (candidate lists only)

Every helper that defines a published number is imported, not re-implemented: the
split, z-score, bounded rule and hit@1 from fusion_controls; the component rebuild
from rv_component_ablation; the truncated-window fusion from window_width_analysis;
the confidence proxies and AUROC bootstrap from selective_retrieval_analysis; McNemar
and the paired bootstrap from revua_cir.significance.

The inputs are checked against the sha256 values recorded in §20.0, and the deployed
report-half figures (base 36.633, fusion 38.977) must reproduce before any contrast.

  (from revua_cir/) python scripts/revision_followups.py --root .. --out ../revision_work/followups_2026-11-03
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import fusion_controls as fc  # noqa: E402
import rv_component_ablation as rca  # noqa: E402
import selective_retrieval_analysis as sra  # noqa: E402
import window_width_analysis as wwa  # noqa: E402
from revua_cir.significance import mcnemar_exact, paired_bootstrap_ci  # noqa: E402

AMENDMENT = "v3.1 §20 (2026-11-03)"
INPUTS = {   # §20.0 and §20.7, paths relative to the project root
    "cache": ("bbnv_caches/rv_score_cache_fc_aep_rv.jsonl",
              "26d7fc552b4ae902d9ecfd0bd4ec2bad329e627536887ae4b241e5d9a85c0308"),
    "features": ("bbnv_feats/v1/cirr_val/features.npz",
                 "3a91e82caa69479ba6fe18ce93adbcca9345a454cc53812c3105d4927495033f"),
    "latency": ("latency_per_k_gpu.json",
                "19c00cce289230fed58fc8f342d85afc8f2ac5a37a69ca08baf2d6535287512d"),
    "latency_k12": ("revua_cir/scripts/latency_k12_gpu.json", None),
    "edir_annotated": ("edir_work/edir_annotated.jsonl",
                       "449ddb9553c405b24a5fbd58de33cd27d59b60cd519f8511d1fc29b28d6ec0d0"),
}
PUBLISHED_REPORT = {"base": 36.633, "deployed": 38.977}   # tab:fusionrisk, 3 d.p.
PUBLISHED_WINDOW = {1: 37.07, 2: 37.07, 3: 38.17, 5: 38.75, 7: 38.91, 10: 39.37}   # tab:window, full set
N_REP, BOOT_SEED = 10000, 0
ALPHA = 0.05
DEPLOYED_LAMBDA = 1.0
BETA_GRID = tuple(round(0.1 * i, 1) for i in range(11))
WIDTHS = (1, 2, 3, 5, 7, 10)
WIDTH_LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
NOISE_SEEDS = tuple(range(50))
NOISE_REL = 2.0 ** -11


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# shared statistics
# ---------------------------------------------------------------------------
def top1(scores: Sequence[float]) -> int:
    """Index ranked first; ties to the lower index (fc.hit_at_1's convention)."""
    return min(range(len(scores)), key=lambda i: (-scores[i], i))


def holm_adjusted(ps: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def paired(new: Sequence[int], ref: Sequence[int]) -> Dict[str, float]:
    """new - ref on paired 0/1 hits: delta (pp), discordant counts, exact McNemar, CI."""
    if len(new) != len(ref):
        raise ValueError("unpaired inputs")
    fixed = sum(1 for a, b in zip(new, ref) if a and not b)
    broken = sum(1 for a, b in zip(new, ref) if b and not a)
    ci = paired_bootstrap_ci(list(ref), list(new), n_replicates=N_REP, seed=BOOT_SEED)
    return {"delta_pp": 100.0 * (sum(new) - sum(ref)) / len(new), "fixed": fixed, "broken": broken,
            "p_mcnemar": mcnemar_exact(fixed, broken), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}


def outcome(hits: Sequence[int], base: Sequence[int]) -> Dict[str, float]:
    n = len(hits)
    resc = sum(1 for h, b in zip(hits, base) if h and not b)
    brk = sum(1 for h, b in zip(hits, base) if b and not h)
    return {"r1": 100.0 * sum(hits) / n, "rescues": resc, "breaks": brk, "utility": (resc - 2 * brk) / n}


def fused_hits(recs, rel, second: Sequence[Sequence[float]], lam: float, idx: Sequence[int]) -> List[int]:
    return [fc.hit_at_1(fc.rule_bounded(recs[i]["base_score"], second[i], lam), rel[i]) for i in idx]


def tune(recs, rel, second, idx, grid) -> float:
    """Highest tune-half hits; ties to the smaller lambda (fc.tune_lambda's rule)."""
    best, best_h = grid[0], -1
    for lam in grid:
        h = sum(fused_hits(recs, rel, second, lam, idx))
        if h > best_h:
            best, best_h = lam, h
    return best


# ---------------------------------------------------------------------------
# §20.1 M3a
# ---------------------------------------------------------------------------
VARIANTS = {"full": (True, True, True), "no_recon": (False, True, True),
            "edit_recon": (True, False, True), "edit_only": (False, False, True)}
CONTRASTS = (("P1", "full", "no_recon"), ("P2", "edit_recon", "edit_only"))


def m3a(recs, rel, tune_idx, report_idx, full_idx, base_hits) -> dict:
    w, sd, n_rows = rca.recover_edit_weight(recs)
    if w is None or sd > 1e-3:
        raise SystemExit("ABORT: mixture weight not recoverable (w=%s, sd=%s)" % (w, sd))
    scores = {v: [rca.component_score(r, w, *flags) for r in recs] for v, flags in VARIANTS.items()}
    out = {"edit_weight": w, "edit_weight_sd": sd, "lambda": DEPLOYED_LAMBDA, "halves": {}}
    for half, idx in (("report", report_idx), ("full", full_idx)):
        hits = {v: fused_hits(recs, rel, s, DEPLOYED_LAMBDA, idx) for v, s in scores.items()}
        bh = [base_hits[i] for i in idx]
        res = {"variants": {v: outcome(h, bh) for v, h in hits.items()}, "contrasts": {}}
        for name, a, b in CONTRASTS:
            res["contrasts"][name] = {"new": a, "ref": b, **paired(hits[a], hits[b])}
        adj = holm_adjusted([res["contrasts"][c]["p_mcnemar"] for c, _, _ in CONTRASTS])
        for (name, _, _), p in zip(CONTRASTS, adj):
            res["contrasts"][name]["p_holm"] = p
            res["contrasts"][name]["holm_significant"] = p <= ALPHA
        out["halves"][half] = res
    # secondary, descriptive: lambda retuned per variant on the tune half
    out["retuned_descriptive"] = {}
    for v, s in scores.items():
        lam = tune(recs, rel, s, tune_idx, fc.LAMBDA_GRID)
        h = fused_hits(recs, rel, s, lam, report_idx)
        out["retuned_descriptive"][v] = {"lambda": lam, **outcome(h, [base_hits[i] for i in report_idx])}
    return out


# ---------------------------------------------------------------------------
# §20.2 M2a.1
# ---------------------------------------------------------------------------
def training_free_signals(recs, feats) -> Dict[str, List[List[float]]]:
    img = feats["image_emb"].astype(np.float64)
    txt = feats["text_emb"].astype(np.float64)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    ref_row, tgt_row = feats["ref_row"], feats["tgt_row"]
    if len(txt) != len(recs) or len(img) != 2297:
        raise SystemExit("ABORT: feature shapes %s/%s do not match the cache" % (txt.shape, img.shape))
    T, R = [], []
    for q, r in enumerate(recs):
        if r["target_gallery_index"] != int(tgt_row[q]):
            raise SystemExit("ABORT: query %d target %s != feature tgt_row %s"
                             % (q, r["target_gallery_index"], tgt_row[q]))
        cand = r["candidate_gallery_ids"]
        if int(ref_row[q]) in cand:
            raise SystemExit("ABORT: query %d has its reference in the window" % q)
        C = img[cand]
        T.append((C @ txt[q]).tolist())
        R.append((C @ img[int(ref_row[q])]).tolist())
    return {"T": T, "R": R}


def tr_scores(T, R, beta: float) -> List[List[float]]:
    out = []
    for t, r in zip(T, R):
        zt, zr = fc._zscore(t), fc._zscore(r)
        out.append([beta * a + (1.0 - beta) * b for a, b in zip(zt, zr)])
    return out


def m2a(recs, rel, tune_idx, report_idx, base_hits, feats) -> dict:
    sig = training_free_signals(recs, feats)
    chosen = {}
    for name in ("T", "R"):
        chosen[name] = {"lambda": tune(recs, rel, sig[name], tune_idx, fc.LAMBDA_GRID), "scores": sig[name]}
    best = None   # (hits, -lambda, -|beta-0.5|) maximised: ties to smaller lambda, then beta near 0.5
    for beta in BETA_GRID:
        s = tr_scores(sig["T"], sig["R"], beta)
        for lam in fc.LAMBDA_GRID:
            key = (sum(fused_hits(recs, rel, s, lam, tune_idx)), -lam, -abs(beta - 0.5))
            if best is None or key > best[0]:
                best = (key, beta, lam, s)
    chosen["TR"] = {"lambda": best[2], "beta": best[1], "scores": best[3]}
    bh = [base_hits[i] for i in report_idx]
    deployed = fused_hits(recs, rel, [r["verification_score"] for r in recs], DEPLOYED_LAMBDA, report_idx)
    out = {"label": "no-additional-training controls using the deployed features", "signals": {}}
    for name, c in chosen.items():
        h = fused_hits(recs, rel, c["scores"], c["lambda"], report_idx)
        out["signals"][name] = {"lambda": c["lambda"], "beta": c.get("beta"), **outcome(h, bh),
                                "vs_base": paired(h, bh), "vs_revua": paired(h, deployed)}
    for fam in ("vs_base", "vs_revua"):
        names = list(out["signals"])
        adj = holm_adjusted([out["signals"][n][fam]["p_mcnemar"] for n in names])
        for n, p in zip(names, adj):
            out["signals"][n][fam]["p_holm"] = p
            out["signals"][n][fam]["holm_significant"] = p <= ALPHA
    out["revua_deployed"] = outcome(deployed, bh)
    return out


# ---------------------------------------------------------------------------
# §20.3 M8
# ---------------------------------------------------------------------------
def latency_table(lat: dict, lat12: dict):
    """Per-k timing with sessions never mixed (§20.3): k = 3, 5, 7, 10 from the main
    session; k = 1, 2 from the second, whose own k = 10 re-timing is its anchor and the
    denominator of its shares. Overhead is the median over repeats of the per-repeat
    paired difference (total - base)."""
    main = {r["k"]: r for r in lat["rows"]}
    second = {r["k"]: r for r in lat12["rows"]}
    if sorted(main) != [3, 5, 7, 10] or sorted(second) != [1, 2, 10]:
        raise SystemExit("ABORT: unexpected timing rows %s / %s" % (sorted(main), sorted(second)))
    latency = {}
    for session, rows, ks in (("main", main, (3, 5, 7, 10)), ("second", second, (1, 2))):
        ov10 = rows[10]["overhead_paired"]["p50"]
        for k in ks:
            ov = rows[k]["overhead_paired"]["p50"]
            latency[k] = {"session": session, "total_p50_ms": rows[k]["full"]["p50"],
                          "base_p50_ms": rows[k]["base"]["p50"], "overhead_paired_p50_ms": ov,
                          "overhead_share_of_k10": ov / ov10, "k10_anchor_overhead_ms": ov10}
    anchor = {"main_k10_total_ms": main[10]["full"]["p50"], "second_k10_total_ms": second[10]["full"]["p50"],
              "second_vs_main": second[10]["full"]["p50"] / main[10]["full"]["p50"] - 1.0}
    return latency, anchor


def m8(recs, rel, tune_idx, report_idx, base_hits, lat, lat12) -> dict:
    def hits_k(k, lam, idx):
        return [1 if wwa.fused_order(recs[i], k, lam)[0] in rel[i] else 0 for i in idx]

    bh = [base_hits[i] for i in report_idx]
    widths = {}
    for k in WIDTHS:
        best, best_h = WIDTH_LAMBDA_GRID[0], -1
        for lam in WIDTH_LAMBDA_GRID:
            h = sum(hits_k(k, lam, tune_idx))
            if h > best_h:
                best, best_h = lam, h
        fixed = hits_k(k, DEPLOYED_LAMBDA, report_idx)
        retuned = hits_k(k, best, report_idx)
        cover = sum(1 for i in report_idx if any(p < k for p in rel[i])) / len(report_idx)
        widths[k] = {"coverage": cover, "fixed": {**outcome(fixed, bh), "p_vs_base": paired(fixed, bh)["p_mcnemar"]},
                     "retuned": {"lambda": best, **outcome(retuned, bh), "p_vs_base": paired(retuned, bh)["p_mcnemar"]}}
    full_gain = widths[10]["fixed"]["r1"] - 100.0 * sum(bh) / len(bh)
    latency, anchor = latency_table(lat, lat12)
    # prose figures recomputed on the full set, as tab:window reports them
    full_idx = list(range(len(recs)))
    fb = [base_hits[i] for i in full_idx]
    fullset = {k: outcome(hits_k(k, DEPLOYED_LAMBDA, full_idx), fb) for k in WIDTHS}
    got = {k: round(fullset[k]["r1"], 2) for k in WIDTHS}
    if got != PUBLISHED_WINDOW:
        raise SystemExit("ABORT: fixed-policy widths %s do not reproduce tab:window %s" % (got, PUBLISHED_WINDOW))
    g10 = fullset[10]["r1"] - 100.0 * sum(fb) / len(fb)
    prose = {}
    for k in (3, 5, 7):
        g = fullset[k]["r1"] - 100.0 * sum(fb) / len(fb)
        share_gain, share_cost = g / g10, latency[k]["overhead_share_of_k10"]
        prose[k] = {"gain_share": share_gain, "overhead_share": share_cost, "efficiency_vs_k10": share_gain / share_cost}
    return {"widths_report": widths, "report_full_gain_pp": full_gain, "latency": latency, "session_anchor": anchor,
            "fullset_fixed": fullset, "prose_fullset": prose,
            "note": "overhead = median over repeats of the per-repeat paired difference (total - base)"}


# ---------------------------------------------------------------------------
# §20.4 M6.1
# ---------------------------------------------------------------------------
def m61(recs, rel, base_hits) -> dict:
    def q(xs):
        a = np.asarray(xs)
        return {p: float(np.quantile(a, p / 100.0)) for p in (0, 1, 5, 25, 50, 75, 95, 100)}

    out = {"window_sd": {}}
    for f in ("base_score", "verification_score", "cos_sim"):
        sds = [float(np.std(r[f])) for r in recs]
        out["window_sd"][f] = {"quantiles": q(sds), "below_1e-3": sum(s < 1e-3 for s in sds),
                               "below_1e-4": sum(s < 1e-4 for s in sds)}
    fused = [fc.rule_bounded(r["base_score"], r["verification_score"], DEPLOYED_LAMBDA) for r in recs]
    gaps, ties = [], 0
    for s in fused:
        a = sorted(s, reverse=True)
        gaps.append(a[0] - a[1])
        ties += a[0] == a[1]
    out["fused_top_gap"] = {"quantiles": q(gaps), "exact_ties": ties}
    ref_top = [top1(s) for s in fused]
    n = len(recs)

    def perturbed(fb, fv):
        tops = [top1(fc.rule_bounded(fb(r["base_score"]), fv(r["verification_score"]), DEPLOYED_LAMBDA)) for r in recs]
        flips = sum(1 for a, b in zip(tops, ref_top) if a != b)
        return flips, 100.0 * sum(1 for t, rr in zip(tops, rel) if t in rr) / n

    f16 = lambda xs: np.asarray(xs, dtype=np.float32).astype(np.float16).astype(np.float64).tolist()  # noqa: E731
    flips, r1 = perturbed(f16, f16)
    out["fp16_round"] = {"top1_changes": flips, "r1": r1, "r1_fp32": 100.0 * sum(1 for t, rr in zip(ref_top, rel) if t in rr) / n}
    draws = []
    for seed in NOISE_SEEDS:
        rng = np.random.default_rng(seed)

        def noisy(xs, rng=rng):
            a = np.asarray(xs, dtype=np.float64)
            return (a * (1.0 + rng.uniform(-NOISE_REL, NOISE_REL, a.shape))).tolist()
        draws.append(perturbed(noisy, noisy))
    fl, rs = [d[0] for d in draws], [d[1] for d in draws]
    out["relative_noise"] = {"rel": NOISE_REL, "seeds": len(NOISE_SEEDS), "changes_mean": float(np.mean(fl)),
                             "changes_min": min(fl), "changes_max": max(fl), "r1_min": min(rs), "r1_max": max(rs),
                             "base_r1": 100.0 * sum(base_hits) / n}
    return out


# ---------------------------------------------------------------------------
# §20.5 m7 and §20.7 X6
# ---------------------------------------------------------------------------
def m7(cache_path) -> dict:
    recs, mp, _ = sra.load_records(cache_path)
    correct = [sra.is_correct(r, 1, mp) for r in recs]
    ent = [sra.confidence(r, "base_entropy") for r in recs]
    rvm = [sra.confidence(r, "rv_margin") for r in recs]
    comb = sra.combine_by_rank(ent, rvm)
    return {"auroc": {"base_entropy": sra.auroc(ent, correct), "rv_margin": sra.auroc(rvm, correct),
                      "base_entropy+rv_margin": sra.auroc(comb, correct)},
            "combined_vs_entropy": sra.paired_bootstrap_auroc_delta(comb, ent, correct)}


def x6(path) -> dict:
    counts = Counter()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                counts[len(json.loads(line)["candidate_gallery_ids"]) - 1] += 1
    return {"negatives_per_query": dict(sorted(counts.items())), "queries": sum(counts.values())}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="project root (holds bbnv_caches/, edir_work/, ...)")
    ap.add_argument("--out", required=True, help="new directory for results.json")
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    paths = {k: os.path.join(args.root, p) for k, (p, _) in INPUTS.items()}
    hashes = {k: sha256(p) for k, p in paths.items()}
    for k, (_, want) in INPUTS.items():
        if want is not None and hashes[k] != want:
            raise SystemExit("ABORT: %s sha256 %s != %s recorded in §20" % (k, hashes[k], want))

    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181:
        raise SystemExit("ABORT: expected the 4181-query single-target CIRR cache")
    rel = [fc.relevant_set(r, mp) for r in recs]
    tune_idx, report_idx = fc.split_indices(len(recs))
    full_idx = list(range(len(recs)))
    base_hits = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    dep = fused_hits(recs, rel, [r["verification_score"] for r in recs], DEPLOYED_LAMBDA, report_idx)
    pre = {"base": round(100.0 * sum(base_hits[i] for i in report_idx) / len(report_idx), 3),
           "deployed": round(100.0 * sum(dep) / len(dep), 3)}
    if pre != PUBLISHED_REPORT or (len(tune_idx), len(report_idx)) != (2090, 2091):
        raise SystemExit("ABORT: pre-flight %s != published %s" % (pre, PUBLISHED_REPORT))
    print("[pre-flight] report half reproduces base %.3f and deployed fusion %.3f" % (pre["base"], pre["deployed"]), flush=True)

    with np.load(paths["features"], allow_pickle=False) as z:
        feats = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row")}
    with open(paths["latency"], encoding="utf-8") as fh:
        lat = json.load(fh)
    with open(paths["latency_k12"], encoding="utf-8") as fh:
        lat12 = json.load(fh)

    results = {"amendment": AMENDMENT, "inputs_sha256": hashes, "preflight": pre,
               "split": {"tune": len(tune_idx), "report": len(report_idx)}}
    for name, fn in (("m3a", lambda: m3a(recs, rel, tune_idx, report_idx, full_idx, base_hits)),
                     ("m2a", lambda: m2a(recs, rel, tune_idx, report_idx, base_hits, feats)),
                     ("m8", lambda: m8(recs, rel, tune_idx, report_idx, base_hits, lat, lat12)),
                     ("m61", lambda: m61(recs, rel, base_hits)),
                     ("m7", lambda: m7(paths["cache"])),
                     ("x6", lambda: x6(paths["edir_annotated"]))):
        results[name] = fn()
        print("[%s] done, %.0fs" % (name, time.time() - t0), flush=True)
    results["source_sha256"] = {f: sha256(os.path.join(_HERE, f)) for f in
                                ("revision_followups.py", "fusion_controls.py", "rv_component_ablation.py",
                                 "window_width_analysis.py", "selective_retrieval_analysis.py")}
    results["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
