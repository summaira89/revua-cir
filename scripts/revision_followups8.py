"""Reference-cluster sensitivity of the main CIRR R@1 contrasts: amendment v3.1 §28. CPU.

The locked §20-§26 intervals resample queries. Queries sharing a reference image are not
independent, so this re-estimates each primary R@1 contrast with a bootstrap over reference
images (bbnv_policy.clustered_bootstrap, 10,000 replicates, seed 54321), on the CIRR
held-out half. Per-query hits are recomputed with the same code paths and weights as the
locked analyses, and each contrast's fixed/broken counts must reproduce the locked values
exactly before any interval is computed. Descriptive sensitivity analysis; no claim depends
on it being significant.

  (from revua_cir/) python scripts/revision_followups8.py --root .. --out ../revision_work/followups28_2026-11-05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_policy as pol  # noqa: E402
import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import revision_followups5 as r5  # noqa: E402
import revision_followups6 as r6  # noqa: E402

AMENDMENT = "v3.1 §28 (2026-11-05)"
LOCKED = {
    "s20": ("revision_work/followups_2026-11-03/results.json", "a8ef5f658dad278ef66843c79b59365db20cabcb3e9041b025c57a53a357dd30"),
    "s21": ("revision_work/followups21_2026-11-03/results.json", "e738bac8bd560a380fbe206551e367efafa92123cd161477a65a2adccb0067f6"),
    "s25a": ("revision_work/followups25a_2026-11-04/results.json", "469628ab4bc609a42e7fef34ce6644567c6f8059131ee1e96801b79f151f8bcf"),
    "s26c": ("revision_work/followups26c_fc_2026-11-04/results.json", "7012dd9aef2608e34a58bf1a3265156721dc3c92e67d3aef3b2e46c25295e436"),
}
TOWERS = {"pretrained": "revision_work/towers_pretrained_2026-11-04", "fc_only": "revision_work/towers_fc_2026-11-04"}
N_REP = 10000


def load_locked(root: str) -> Dict[str, dict]:
    out = {}
    for k, (p, want) in LOCKED.items():
        path = os.path.join(root, p)
        if rf.sha256(path) != want:
            raise SystemExit("ABORT: %s differs from its lock" % p)
        with open(path, encoding="utf-8") as fh:
            out[k] = json.load(fh)
    return out


def counts(new: Sequence[int], ref: Sequence[int]) -> Dict[str, int]:
    return {"fixed": sum(1 for a, b in zip(new, ref) if a and not b),
            "broken": sum(1 for a, b in zip(new, ref) if b and not a)}


def clustered(new: Sequence[int], ref: Sequence[int], cluster: Sequence[int]) -> Dict[str, float]:
    """Mean paired R@1 difference in pp with a bootstrap over clusters (reference images)."""
    return pol.clustered_bootstrap([100.0 * (a - b) for a, b in zip(new, ref)], cluster, N_REP)


def contrast(name: str, new, ref, cluster, want: Dict[str, object]) -> Dict[str, object]:
    c = counts(new, ref)
    if (c["fixed"], c["broken"]) != (want["fixed"], want["broken"]):
        raise SystemExit("ABORT: %s recomputes %s, locked %s/%s" % (name, c, want["fixed"], want["broken"]))
    cb = clustered(new, ref, cluster)
    return {"delta_pp": cb["diff"], **c,
            "query_bootstrap_ci": [want["ci_low"], want["ci_high"]],
            "cluster_bootstrap_ci": [cb["ci_low"], cb["ci_high"]],
            "cluster_p_two_sided": cb["p_two_sided"], "n_clusters": cb["n_clusters"],
            "ci_width_ratio": (cb["ci_high"] - cb["ci_low"]) / (want["ci_high"] - want["ci_low"])}


def own_windows(L, recs, ours, report, cluster) -> Dict[str, dict]:
    rel = [fc.relevant_set(r, False) for r in recs]
    pos = [next(iter(s)) if s else None for s in rel]
    B = [r["base_score"] for r in recs]
    V = [r["verification_score"] for r in recs]
    T = r2.text_cosines(recs, ours, 2297)
    lt, lx = L["s21"]["A"]["contrasts"]["Q1"]["lambda_T"], L["s21"]["A"]["contrasts"]["Q1"]["lambda_X"]
    h = {"base": r5.hits(lambda i: B[i], pos, report),
         "BT": r5.hits(lambda i: fc.rule_bounded(B[i], T[i], r2.LAMBDA_T_S20), pos, report),
         "BV": r5.hits(lambda i: fc.rule_bounded(B[i], V[i], rf.DEPLOYED_LAMBDA), pos, report),
         "BTV": r5.hits(lambda i: r2.three_term(B[i], T[i], V[i], lt, lx), pos, report)}
    s20 = L["s20"]["m2a"]["signals"]["T"]
    return {"T_vs_base": contrast("T_vs_base", h["BT"], h["base"], cluster, s20["vs_base"]),
            "T_vs_verifier_fusion": contrast("T_vs_verifier_fusion", h["BT"], h["BV"], cluster, s20["vs_revua"]),
            "T+V_vs_T": contrast("T+V_vs_T", h["BTV"], h["BT"], cluster, L["s21"]["A"]["contrasts"]["Q1"]["vs_BT"])}


def clip4cir_windows(root, L, recs, ours, sig, V, s25c, gallery_sha, cluster) -> Dict[str, dict]:
    B, TO, pos, report = sig["B"], sig["T_O"], sig["pos"], sig["report"]
    lam = s25c["lambda_tuned"]
    enc = {}
    for name, d in TOWERS.items():
        _, arr, _ = r6.load_towers("%s=%s" % (name, os.path.join(root, d)), len(recs), 2297, gallery_sha)
        enc[name] = r5.window_cosines(arr["text_emb"], arr["image_emb"], sig["cand"]).tolist()
    le = L["s26c"]["E"]["lambda_tuned"]
    if le["ours"] != lam["A3"]:
        raise SystemExit("ABORT: §26c and §25c disagree on our text weight")
    h = {"A1": r5.hits(lambda i: B[i], pos, report),
         "A3": r5.hits(lambda i: fc.rule_bounded(B[i], TO[i], lam["A3"]), pos, report),
         "A4": r5.hits(lambda i: fc.rule_bounded(B[i], V[i], lam["A4"]), pos, report),
         "A5": r5.hits(lambda i: r2.three_term(B[i], TO[i], V[i], lam["A5"]["lambda_T"], lam["A5"]["lambda_V"]), pos, report),
         "pre": r5.hits(lambda i: fc.rule_bounded(B[i], enc["pretrained"][i], le["pretrained"]), pos, report),
         "fc": r5.hits(lambda i: fc.rule_bounded(B[i], enc["fc_only"][i], le["fc_only"]), pos, report)}
    c25 = s25c["report"]["contrasts"]
    e26 = L["s26c"]["E"]["report"]["fused_contrasts"]
    return {"C2_T_vs_base": contrast("C2", h["A3"], h["A1"], cluster, L["s25a"]["A"]["report"]["contrasts"]["C2"]),
            "C4_V_vs_base": contrast("C4", h["A4"], h["A1"], cluster, c25["C4"]),
            "C5_T+V_vs_T": contrast("C5", h["A5"], h["A3"], cluster, c25["C5"]),
            "deployed_vs_pretrained_fused": contrast("ours_vs_pretrained", h["A3"], h["pre"], cluster,
                                                     e26["ours_vs_pretrained"]),
            "deployed_vs_fc_fused": contrast("ours_vs_fc", h["A3"], h["fc"], cluster, e26["ours_vs_fc_only"]),
            "fc_vs_pretrained_fused": contrast("fc_vs_pretrained", h["fc"], h["pre"], cluster,
                                               e26["fc_only_vs_pretrained"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    L = load_locked(args.root)
    hashes, recs, ours, sig, V, s25c, gallery_sha = r6.load_s25(args.root)
    report = sig["report"]
    ref_row = [int(x) for x in ours["ref_row"]]
    cluster = [ref_row[i] for i in report]
    res = {"amendment": AMENDMENT, "status": "descriptive sensitivity; specified before execution",
           "inputs_sha256": {**hashes, **{k: v[1] for k, v in LOCKED.items()}},
           "population": "CIRR held-out half (2091 queries); clusters = reference images",
           "own_windows": own_windows(L, recs, ours, report, cluster),
           "clip4cir_windows": clip4cir_windows(args.root, L, recs, ours, sig, V, s25c, gallery_sha, cluster)}
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups8.py", "revision_followups6.py", "revision_followups5.py",
                             "revision_followups2.py", "revision_followups.py", "fusion_controls.py", "bbnv_policy.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for part in ("own_windows", "clip4cir_windows"):
        for k, v in res[part].items():
            print("[28] %-30s %+6.2f pp  query CI [%+.2f,%+.2f]  cluster CI [%+.2f,%+.2f]  width x%.2f"
                  % (k, v["delta_pp"], *v["query_bootstrap_ci"], *v["cluster_bootstrap_ci"], v["ci_width_ratio"]))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
