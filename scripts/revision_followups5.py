"""External-retriever windows (CLIP4Cir RN50x4_fullft): amendment v3.1 §25. CPU.

  --part windows   §25a: CLIP4Cir's own top-10 windows W_C from the §24 features (reference
                   excluded, ties to the lower row); arms A1 CLIP4Cir base, A2 + CLIP4Cir text
                   cosine, A3 + our text cosine; contrasts C1-C3; writes windows_c.npz, the
                   input of the §25b GPU pass (cirr_external_windows.py)
  --part verifier  §25c: with our verifier's scores on W_C (§25b output), arms A4 + our
                   verifier and A5 + our text cosine + our verifier; contrasts C4, C5

Every arm is bounded fusion on the window (z-scores within it). Primary lambda: tuned on
the tune half by R@1 over fusion_controls.LAMBDA_GRID (ties to the smaller value; for A5
the §21 two-dimensional rule). Report half primary, full set secondary.

  (from revua_cir/) python scripts/revision_followups5.py --part windows --root .. --out ../revision_work/followups25a_2026-11-04
  (from revua_cir/) python scripts/revision_followups5.py --part verifier --root .. --gpu-run ../revision_work/external_windows_c_2026-11-04 --out ../revision_work/followups25c_2026-11-04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402
import revision_followups2 as r2  # noqa: E402
import revision_followups3 as r3  # noqa: E402

AMENDMENT = "v3.1 §25 (2026-11-04)"
PROBE = "revision_work/clip4cir_probe_2026-11-03"
INPUTS = {
    "probe": (PROBE + "/results.json", "e9bdbf0cf106b1ea3c92e70adccb2e31cbf85dc5c41cc90fdb9c05de45b84029"),
    "c4c": (PROBE + "/features_RN50x4_fullft.npz", "505f0e33ef7a9bae8c258c2013e6d907c36a890c033e4049904f185e69cdaed4"),
    "features": r3.INPUTS["features"],
    "cache": r3.INPUTS["cache"],
}
CONFIG = "RN50x4_fullft"
K = 10
LAMBDA_T = r2.LAMBDA_T_S20            # 4.0, §20.2 (frozen, secondary)
LAMBDA_V = rf.DEPLOYED_LAMBDA         # 1.0, the deployed verifier fusion (frozen, secondary)
MIN_GPU_FIELDS = ("verification_score",)
GPU_SCHEMA = "cirr-external-windows/1"
GPU_CHECKPOINT_SHA = "677eddfb2238a4d729678346d3b5af49ddf6edb83b0ee565b089a03bf596c597"   # fc_aep_rv/epoch004.pt
GPU_CONFIG_SHA = "08d8cd16d78c4972e6558125224c8455015259cd2dd1344a02001a0726d50e15"       # edir_work/laptop_config.yaml


# ---------------------------------------------------------------------------
# windows and signals
# ---------------------------------------------------------------------------
def build_windows(img_c: np.ndarray, comp: np.ndarray, ref_row: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """CLIP4Cir top-K per query, scored exactly as the §24 probe ranked (float32 unit gallery
    times the float64 composed vector), reference excluded, ties to the lower row."""
    imgn = img_c / np.linalg.norm(img_c, axis=1, keepdims=True)
    cand, base = [], []
    for i in range(len(comp)):
        s = imgn @ comp[i].astype(np.float64)
        w = r3.top_k_excluding(s, int(ref_row[i]), K)
        cand.append(w)
        base.append(s[w])
    return np.asarray(cand, dtype=np.int64), np.asarray(base, dtype=np.float64)


def window_cosines(txt: np.ndarray, img: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """cos(text_i, image_c) for every window candidate, both towers unit-normalised in float64,
    with the same per-query product as §21's text_cosines (so the values are bit-identical)."""
    t = r3.unit_rows(txt)
    g = r3.unit_rows(img)
    return np.stack([g[cand[q]] @ t[q] for q in range(len(cand))])


def target_positions(cand: np.ndarray, tgt_row: Sequence[int]) -> List[Optional[int]]:
    out = []
    for i, w in enumerate(cand):
        hit = np.nonzero(w == int(tgt_row[i]))[0]
        out.append(int(hit[0]) if len(hit) else None)
    return out


# ---------------------------------------------------------------------------
# arms and tuning
# ---------------------------------------------------------------------------
def hits(score_fn: Callable[[int], Sequence[float]], pos: Sequence[Optional[int]], idx: Sequence[int]) -> List[int]:
    return [fc.hit_at_1(score_fn(i), {pos[i]} if pos[i] is not None else set()) for i in idx]


def tune_one(B, X, pos, idx) -> float:
    """Highest R@1 on idx over LAMBDA_GRID; ties to the smaller lambda."""
    best, best_h = None, -1
    for lam in fc.LAMBDA_GRID:
        h = sum(hits(lambda i: fc.rule_bounded(B[i], X[i], lam), pos, idx))
        if h > best_h:
            best, best_h = lam, h
    return best


def tune_three(B, T, X, pos, idx) -> Tuple[float, float]:
    """§21 rule: argmax R@1 over (lt, lx) in LAMBDA_GRID^2; ties to the smaller lx, then lt."""
    best = None
    for lx in fc.LAMBDA_GRID:
        for lt in fc.LAMBDA_GRID:
            val = sum(hits(lambda i: r2.three_term(B[i], T[i], X[i], lt, lx), pos, idx))
            key = (val, -lx, -lt)
            if best is None or key > best[0]:
                best = (key, lt, lx)
    return best[1], best[2]


def summarise_arms(arms: Dict[str, Callable], pos, idx, ref: str) -> Tuple[Dict[str, dict], Dict[str, List[int]]]:
    h = {a: hits(fn, pos, idx) for a, fn in arms.items()}
    return {a: {**rf.outcome(v, h[ref]), "vs_" + ref: rf.paired(v, h[ref]) if a != ref else None}
            for a, v in h.items()}, h


def contrasts(h: Dict[str, List[int]], pairs: Sequence[Tuple[str, str, str]]) -> Dict[str, dict]:
    out = {name: {"new": a, "ref": b, **rf.paired(h[a], h[b])} for name, a, b in pairs}
    for name, p in zip(out, rf.holm_adjusted([out[n]["p_mcnemar"] for n in out])):
        out[name]["p_holm"] = p
        out[name]["holm_significant"] = p <= 0.05
    return out


# ---------------------------------------------------------------------------
# loading and pre-flight
# ---------------------------------------------------------------------------
def load_all(root: str):
    paths = {k: os.path.join(root, p) for k, (p, _) in INPUTS.items()}
    hashes = {k: rf.sha256(p) for k, p in paths.items()}
    for k, (_, want) in INPUTS.items():
        if hashes[k] != want:
            raise SystemExit("ABORT: %s sha256 %s != recorded %s" % (k, hashes[k], want))
    with open(paths["probe"], encoding="utf-8") as fh:
        probe = json.load(fh)
    if not probe["gate"]["pass"]:
        raise SystemExit("ABORT: the §24 reproduction gate did not pass")
    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181:
        raise SystemExit("ABORT: expected the 4181-query CIRR cache")
    with np.load(paths["features"], allow_pickle=False) as z:
        ours = {k: z[k] for k in ("image_emb", "text_emb", "ref_row", "tgt_row", "image_paths")}
    with np.load(paths["c4c"], allow_pickle=False) as z:
        c4c = {k: z[k] for k in z.files}
    names = [os.path.splitext(os.path.basename(str(p)))[0] for p in ours["image_paths"]]
    if [str(n) for n in c4c["image_names"]] != names:
        raise SystemExit("ABORT: CLIP4Cir gallery rows differ from ours")
    if [int(q) for q in c4c["query_id"]] != [r["query_id"] for r in recs]:
        raise SystemExit("ABORT: CLIP4Cir query order differs from the cache")
    for k in ("ref_row", "tgt_row"):
        if not np.array_equal(c4c[k], ours[k]):
            raise SystemExit("ABORT: CLIP4Cir %s differs from ours" % k)
    return hashes, probe, recs, ours, c4c


def build_signals(recs, ours, c4c, probe) -> Dict[str, object]:
    ref_row, tgt_row = ours["ref_row"], ours["tgt_row"]
    cand, B = build_windows(c4c["image_emb"], c4c["composed"], ref_row)
    if any(int(ref_row[i]) in cand[i] for i in range(len(cand))) or any(len(set(w)) != K for w in cand):
        raise SystemExit("ABORT: a window holds its reference or a repeated row")
    pos = target_positions(cand, tgt_row)
    tune, report = fc.split_indices(len(recs))
    want = probe["configs"][CONFIG]["released"]
    for half, idx in (("full", list(range(len(recs)))), ("report", report)):
        r1 = 100.0 * np.mean([pos[i] == 0 for i in idx])
        cov = 100.0 * np.mean([pos[i] is not None for i in idx])
        if r1 != want[half]["R@1"] or cov != want[half]["R@10"]:
            raise SystemExit("ABORT: W_C %s R@1/coverage %r/%r != §24 %r/%r"
                             % (half, r1, cov, want[half]["R@1"], want[half]["R@10"]))
    # our T on our own windows must equal §21's T exactly (same definition, same code path)
    wb = np.asarray([r["candidate_gallery_ids"] for r in recs], dtype=np.int64)
    if not np.array_equal(window_cosines(ours["text_emb"], ours["image_emb"], wb),
                          np.asarray(r2.text_cosines(recs, ours, 2297))):
        raise SystemExit("ABORT: our text cosine does not reproduce §21's T")
    return {"cand": cand, "pos": pos, "tune": tune, "report": report,
            "B": B.tolist(),
            "T_C": window_cosines(c4c["text_emb"], c4c["image_emb"], cand).tolist(),
            "T_O": window_cosines(ours["text_emb"], ours["image_emb"], cand).tolist()}


def halves(sig) -> List[Tuple[str, List[int]]]:
    return [("report", sig["report"]), ("full", list(range(len(sig["pos"]))))]


# ---------------------------------------------------------------------------
# §25a
# ---------------------------------------------------------------------------
def part_windows(sig) -> dict:
    B, TC, TO, pos, tune = sig["B"], sig["T_C"], sig["T_O"], sig["pos"], sig["tune"]
    lam = {"A2": tune_one(B, TC, pos, tune), "A3": tune_one(B, TO, pos, tune)}
    arms = {"A1": lambda i: B[i],
            "A2": lambda i: fc.rule_bounded(B[i], TC[i], lam["A2"]),
            "A3": lambda i: fc.rule_bounded(B[i], TO[i], lam["A3"]),
            "A2_frozen": lambda i: fc.rule_bounded(B[i], TC[i], LAMBDA_T),
            "A3_frozen": lambda i: fc.rule_bounded(B[i], TO[i], LAMBDA_T),
            "T_C_alone": lambda i: TC[i], "T_O_alone": lambda i: TO[i]}
    out = {"lambda_tuned": lam, "lambda_frozen": LAMBDA_T}
    for half, idx in halves(sig):
        summ, h = summarise_arms(arms, pos, idx, "A1")
        out[half] = {"coverage": 100.0 * np.mean([pos[i] is not None for i in idx]), "arms": summ,
                     "contrasts": contrasts(h, (("C1", "A2", "A1"), ("C2", "A3", "A1"), ("C3", "A3", "A2"))),
                     "contrasts_frozen": contrasts(h, (("C1", "A2_frozen", "A1"), ("C2", "A3_frozen", "A1"),
                                                       ("C3", "A3_frozen", "A2_frozen")))}
    out["contrast_status"] = "report half primary (Holm over C1-C3); full set and frozen lambda secondary"
    return out


# ---------------------------------------------------------------------------
# §25c
# ---------------------------------------------------------------------------
def load_gpu_run(gpu_dir: str, cand: np.ndarray) -> Tuple[Dict[str, np.ndarray], dict]:
    with open(os.path.join(gpu_dir, "results.json"), encoding="utf-8") as fh:
        res = json.load(fh)
    man = res["manifest"]
    got = (res["schema"], res["n_queries"], man["checkpoint_sha256"], man["config_sha256"])
    if got != (GPU_SCHEMA, len(cand), GPU_CHECKPOINT_SHA, GPU_CONFIG_SHA):
        raise SystemExit("ABORT: GPU run is not the deployed checkpoint/config on all queries: %s" % (got,))
    npz = os.path.join(gpu_dir, res["output_name"])
    if rf.sha256(npz) != res["output_files"][res["output_name"]]:
        raise SystemExit("ABORT: GPU output hash differs from its results.json")
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    if not np.array_equal(arrays["cand"], cand):
        raise SystemExit("ABORT: the GPU run scored different windows from W_C")
    for f in MIN_GPU_FIELDS:
        if arrays[f].shape != cand.shape or not np.isfinite(arrays[f]).all():
            raise SystemExit("ABORT: GPU field %s has shape %s or non-finite values" % (f, arrays[f].shape))
    return arrays, res


def part_verifier(sig, V: Sequence[Sequence[float]]) -> dict:
    B, TO, pos, tune = sig["B"], sig["T_O"], sig["pos"], sig["tune"]
    lam3 = tune_one(B, TO, pos, tune)
    lam4 = tune_one(B, V, pos, tune)
    lt5, lx5 = tune_three(B, TO, V, pos, tune)
    arms = {"A1": lambda i: B[i],
            "A3": lambda i: fc.rule_bounded(B[i], TO[i], lam3),
            "A4": lambda i: fc.rule_bounded(B[i], V[i], lam4),
            "A5": lambda i: r2.three_term(B[i], TO[i], V[i], lt5, lx5),
            "A4_frozen": lambda i: fc.rule_bounded(B[i], V[i], LAMBDA_V),
            "V_alone": lambda i: V[i]}
    out = {"lambda_tuned": {"A3": lam3, "A4": lam4, "A5": {"lambda_T": lt5, "lambda_V": lx5}},
           "A5_verifier_excluded_by_tuning": lx5 == 0.0, "lambda_V_frozen": LAMBDA_V}
    for half, idx in halves(sig):
        summ, h = summarise_arms(arms, pos, idx, "A1")
        out[half] = {"arms": summ, "contrasts": contrasts(h, (("C4", "A4", "A1"), ("C5", "A5", "A3"))),
                     "frozen_C4": rf.paired(h["A4_frozen"], h["A1"])}
    out["contrast_status"] = "report half primary (Holm over C4, C5); full set and frozen lambda secondary"
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", required=True, choices=("windows", "verifier"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-run", help="§25b output directory (verifier part only)")
    args = ap.parse_args(argv)
    if (args.part == "verifier") != (args.gpu_run is not None):
        ap.error("--gpu-run is required for --part verifier and only for it")
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    t0 = time.time()
    hashes, probe, recs, ours, c4c = load_all(args.root)
    sig = build_signals(recs, ours, c4c, probe)
    print("[25] W_C reproduces §24 (R@1, coverage) and our T reproduces §21, %.0fs" % (time.time() - t0), flush=True)
    res = {"amendment": AMENDMENT, "part": args.part, "inputs_sha256": hashes,
           "status": "exploratory; specified before execution"}
    if args.part == "windows":
        res["A"] = part_windows(sig)
    else:
        arrays, gres = load_gpu_run(args.gpu_run, sig["cand"])
        res["gpu_run"] = {"dir": os.path.basename(os.path.normpath(args.gpu_run)),
                          "results_sha256": rf.sha256(os.path.join(args.gpu_run, "results.json")),
                          "parity": gres["parity"]}
        res["V"] = part_verifier(sig, arrays["verification_score"].astype(np.float64).tolist())
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("revision_followups5.py", "revision_followups3.py", "revision_followups2.py",
                             "revision_followups.py", "fusion_controls.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    os.makedirs(args.out)
    if args.part == "windows":
        with open(os.path.join(args.out, "windows_c.npz"), "xb") as fh:
            np.savez(fh, query_id=np.array([r["query_id"] for r in recs], dtype=np.int64), cand=sig["cand"],
                     ref_row=ours["ref_row"].astype(np.int64))
        res["windows_c_sha256"] = rf.sha256(os.path.join(args.out, "windows_c.npz"))
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
