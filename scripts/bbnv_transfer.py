"""BBNV exploratory transfer of the FROZEN CIRR gates (v3.1 §16). CPU.

Applies the stored `forward_only` and `base_only` gates of the frozen CIRR
analysis (seed 0: their beta, feature means and SDs, accept iff p > 0.5;
nothing is refitted) to scored validation windows of another dataset, and
reports, per window run and pooled, everything §16 declares: each gate's mean
utility relative to base, their paired difference with a query bootstrap,
rescues retained, breaks accepted, acceptance rate, R@1 (base, ungated
fusion, both gates) and the oracle ceiling against the actual comparator
(oracle = accept exactly the rescues). All results are exploratory.

Usage
-----
  python scripts/bbnv_transfer.py --analysis ../bbnv_runs/analysis_cirr_v1 \\
      --windows ../bbnv_runs/windows_fiq_dress_v1 ../bbnv_runs/windows_fiq_shirt_v1 ../bbnv_runs/windows_fiq_toptee_v1 \\
      --out-dir ../bbnv_runs/transfer_fiq_v1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_analyze as ana  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import gate_suite as gs  # noqa: E402

FROZEN_ANALYSIS_SHA256 = "47fa313fa3e26fa982d9740b31fa1b8ae8223225949d2b835b1f9850dd8072cb"
GATE_SEED = 0
METHOD, CONTROL = "forward_only", "base_only"


class TransferError(ValueError):
    pass


def frozen_gate(params: Mapping[str, object], names: Sequence[str]):
    """accept(views) -> bools from STORED parameters: standardise with the
    stored means/SDs, logistic with the stored beta, accept iff p > 0.5."""
    beta = np.asarray(params["beta"], dtype=np.float64)
    mu = np.asarray(params["mean"], dtype=np.float64)
    sd = np.asarray(params["sd"], dtype=np.float64)
    if beta.shape != (len(names) + 1,) or mu.shape != (len(names),) or sd.shape != (len(names),) or np.any(sd <= 0):
        raise TransferError("stored gate parameters do not match the feature list")

    def accept(views: Sequence[Mapping[str, object]]) -> List[bool]:
        X = ana._matrix(views, names)
        z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd]) @ beta
        return [bool(p > 0.5) for p in 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))]
    return accept


def load_frozen(analysis_dir: str) -> Dict[str, object]:
    res = prov.verify_run(analysis_dir)
    if prov.sha256_file(os.path.join(analysis_dir, prov.RESULTS_NAME)) != FROZEN_ANALYSIS_SHA256:
        raise TransferError("%s is not the frozen CIRR analysis" % analysis_dir)
    gates = res["gates"]["seed%d" % GATE_SEED]["gates"]
    return {name: frozen_gate(gates[name]["params"], ana.feature_names(name)) for name in (METHOD, CONTROL)} | \
        {"lambda_star": res["lambda_star"]}


def summarise(views: Sequence[Mapping[str, object]], accept: Mapping[str, List[bool]], n_rep: int) -> Dict[str, object]:
    """§16 quantities for one set of views."""
    n = len(views)
    rescues = [i for i, v in enumerate(views) if v["changed"] and v["hit_fused"] > v["hit_base"]]
    breaks = [i for i, v in enumerate(views) if v["changed"] and v["hit_fused"] < v["hit_base"]]
    changed = sum(1 for v in views if v["changed"])
    out = {"n_queries": n, "changed": changed, "rescues_available": len(rescues), "breaks_available": len(breaks),
           "r1_base": 100.0 * sum(v["hit_base"] for v in views) / n,
           "r1_ungated_fusion": 100.0 * sum(v["hit_fused"] for v in views) / n,
           "u_ungated_fusion": (len(rescues) - 2.0 * len(breaks)) / n,
           "u_oracle": len(rescues) / n, "gates": {}}
    per_u = {}
    for name, flags in accept.items():
        ev = ana.evaluate_gate(views, flags)
        per_u[name] = ev["per_query_u"]
        out["gates"][name] = {"mean_utility_vs_base": ev["mean_utility"], "r1": ev["r1"],
                              "accepted": ev["accepted"], "acceptance_rate_of_changed": ev["accepted"] / changed
                              if changed else None,
                              "rescues_retained": ev["rescues"], "breaks_accepted": ev["breaks"]}
    out["oracle_ceiling_vs_control"] = out["u_oracle"] - out["gates"][CONTROL]["mean_utility_vs_base"]
    out["paired_difference"] = gs.paired_mean_bootstrap(per_u[METHOD], per_u[CONTROL], n_rep=n_rep)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", required=True, help="the frozen CIRR analysis run")
    ap.add_argument("--windows", required=True, nargs="+", help="scored window runs (one per category)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=gs.N_BOOTSTRAP)
    args = ap.parse_args(argv)
    if args.n_boot < 100:
        ap.error("--n-boot must be at least 100")
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    t0 = time.time()
    frozen = load_frozen(args.analysis)
    per_run, pooled, inputs = {}, [], {}
    for wdir in args.windows:
        res, rows = ana.load_windows(wdir)
        if float(res["lambda_star_frozen"]) != frozen["lambda_star"]:
            raise TransferError("%s was scored with another lambda*" % wdir)
        views = ana.gate_views(rows, frozen["lambda_star"], GATE_SEED)
        name = "%s/%s" % (res["target"], res["category"])
        if name in per_run:
            raise TransferError("window run %s given twice" % name)
        per_run[name] = summarise(views, {g: frozen[g](views) for g in (METHOD, CONTROL)}, args.n_boot)
        pooled.extend(views)
        inputs[name] = prov.sha256_file(os.path.join(wdir, prov.RESULTS_NAME))
    results = {"schema": "bbnv-transfer/1", "amendment": "v3.1 §16", "status": "exploratory",
               "gates": {"method": METHOD, "control": CONTROL, "seed": GATE_SEED, "refitted": False},
               "per_run": per_run,
               "pooled": summarise(pooled, {g: frozen[g](pooled) for g in (METHOD, CONTROL)}, args.n_boot),
               "inputs": {"analysis_results_sha256": FROZEN_ANALYSIS_SHA256, "windows_results_sha256": inputs},
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_transfer.py", "args": vars(args),
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("bbnv_transfer.py",)),
                            "runtime": prov.runtime_info(None)},
               "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {})
    prov.verify_run(args.out_dir)
    print("[transfer] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
