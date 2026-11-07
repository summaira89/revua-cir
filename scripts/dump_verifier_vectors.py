"""Dump the verifier's internal vectors for every cached candidate (one GPU pass).

For each query of an existing score cache (its fixed candidate lists, so
nothing is re-ranked), this stores L2-normalised float32 vectors:

  e_inv  (N, d)     AEP invariant embedding (the preservation target)
  t_hat  (N, d)     edit-text embedding (the edit target)
  q_comp (N, d)     composed query (the base retriever's query)
  rec    (N, k, d)  FC(candidate, t_rev), the reverse reconstruction
  cinv   (N, k, d)  the candidate's invariant sub-space
  cedit  (N, k, d)  the candidate's editable sub-space
  c_pool (N, k, d)  the candidate's pooled image embedding (gallery side)

plus candidate_ids (N, k) and query_id (N,), and a JSON manifest beside the
.npz recording what the vectors belong to: cache SHA-256, config SHA-256,
checkpoint path and size, dataset/split/category, gallery size and digest,
the checkpoint's mixture weight w and calibrator (scale, bias).

Verification, all of which must pass or nothing is written:
  * every cache record is well formed, finite and in dataset order;
  * the vectors reproduce the cache's recon_cos, invariant_cos, edit_cos and
    base_score, and calibrator(mixture) reproduces verification_score,
    each to within its own tolerance in VECTOR_TOL, with every difference
    finite (computed in float64);
  * every stored tensor is finite;
  * after writing, the file is re-loaded and the same reproduction checks
    are repeated on the loaded arrays.

Usage
-----
  python scripts/dump_verifier_vectors.py --config CFG --checkpoint CKPT \\
      --module-combo fc+aep+rv --dataset cirr --cache CACHE --out ../vec/vec_cirr_val.npz
  python scripts/dump_verifier_vectors.py ... --dataset fashioniq --category dress \\
      --cache FIQ_CACHE --out ../vec/vec_fiq_dress.npz [--skip-if-complete]
  python scripts/dump_verifier_vectors.py ... --dataset edir --edir-root ../edir_data \\
      --edir-meta ../edir_meta/72581ea0 --cache ../edir_work/edir_cache.jsonl --out ../edir_work/vec_edir.npz
EDIR (v3.1 §19) has one split, "test"; its gallery is the 28,645-image corpus.

--skip-if-complete exits successfully, without loading the model, when the
output and its manifest already exist and verify: schema, vector-file digest,
and the identity of THIS run (dataset, category, split, full query count,
module combo, and the cache, config and checkpoint digests). Any other
existing state, including a complete output of a different run, is an error;
nothing is ever overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCHEMA = "revua-verifier-vectors/2"
# Per-field acceptance tolerances for reproducing cached values from the
# re-computed vectors. Rule: max(1e-4, 5 x the largest residual observed in
# the completed full-set checks of job 7517 -- CIRR, and FashionIQ dress up
# to its stop), rounded up to one significant figure. Observed maxima:
# base 1.9e-4, recon 3.1e-5, invariant 2.0e-5, edit 2.6e-6, verification
# 1.5e-6. A tolerance is an acceptance threshold, not a bound on downstream
# metric changes: the deployed baseline reads cached scores, but the BNV
# penalty is computed from these vectors against OTHER queries' targets, so
# its numerical sensitivity is measured separately (vector_repeatability.py).
VECTOR_TOL = {"recon_cos": 2e-4, "invariant_cos": 1e-4, "edit_cos": 1e-4,
              "verification_score": 1e-4, "base_score": 1e-3}
TERMS = ("recon_cos", "invariant_cos", "edit_cos")
FIELDS_K = ("candidate_gallery_ids", "base_score", "verification_score", "cos_sim") + TERMS


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_paths(out: str) -> dict:
    """Final and staging names for the vector file and its manifest."""
    stem = out[:-4]
    return {"npz": out, "manifest": stem + ".manifest.json",
            "npz_partial": stem + ".partial.npz", "manifest_partial": stem + ".manifest.partial.json"}


def output_state_error(paths: dict):
    """None if the output location is clean; otherwise a message naming the
    state and its recovery. Publication renames the vector file first and the
    manifest last, so an interruption can leave partials, a vector file
    without a manifest, or (if removed by hand) a manifest without vectors."""
    leftovers = [p for p in (paths["npz_partial"], paths["manifest_partial"])
                 if os.path.exists(p)]
    if leftovers:
        return ("incomplete artifacts from an interrupted run: %s; delete them and rerun"
                % ", ".join(leftovers))
    has_npz, has_man = os.path.exists(paths["npz"]), os.path.exists(paths["manifest"])
    if has_npz and not has_man:
        return ("%s has no manifest (an interrupted run left it); delete it and rerun"
                % paths["npz"])
    if has_man and not has_npz:
        return ("%s has no vector file beside it (orphan manifest); delete it and rerun"
                % paths["manifest"])
    if has_npz and has_man:
        return "%s already exists with its manifest; refusing to overwrite" % paths["npz"]
    return None


def tolerance_violations(worst: dict) -> list:
    """Fields whose worst reproduction error exceeds their own tolerance;
    every field in VECTOR_TOL must be present."""
    missing = sorted(set(VECTOR_TOL) - set(worst))
    if missing:
        raise ValueError("reproduction errors missing for %s" % missing)
    return ["%s %.3g > %.3g" % (f, worst[f], VECTOR_TOL[f])
            for f in sorted(VECTOR_TOL) if worst[f] > VECTOR_TOL[f]]


def split_of(dataset: str) -> str:
    """The evaluation split a dump covers: validation, except EDIR's single test split."""
    return "test" if dataset == "edir" else "val"


def run_identity(dataset: str, category, n_queries: int, module_combo: str,
                 cache: str, config: str, checkpoint_sha256: str) -> dict:
    """The manifest fields that make a full dump THIS run's dump."""
    return {"dataset": dataset, "category": category, "split": split_of(dataset), "n_queries": n_queries,
            "module_combo": module_combo, "cache_sha256": sha256_file(cache),
            "config_sha256": sha256_file(config), "checkpoint_sha256": checkpoint_sha256}


def complete_output(paths: dict, identity: dict) -> bool:
    """True if the vector file and its manifest both exist, the file matches
    the manifest's schema and digest, and every key of `identity` equals the
    manifest's value; False if the location is clean. Any other state
    (partials, one file only, failed verification, another run's output)
    raises ValueError."""
    problem = output_state_error(paths)
    if problem is None:
        return False
    if not (os.path.exists(paths["npz"]) and os.path.exists(paths["manifest"])):
        raise ValueError(problem)
    with open(paths["manifest"], encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("schema") != SCHEMA or man.get("npz_sha256") != sha256_file(paths["npz"]):
        raise ValueError("%s exists but does not verify against its manifest" % paths["npz"])
    differ = sorted(key for key in identity if man.get(key) != identity[key])
    if differ:
        raise ValueError("%s is complete but from a different run (differs in %s); "
                         "refusing to skip or overwrite" % (paths["npz"], ", ".join(differ)))
    return True


def load_cache(path: str):
    """Records in file order, validated: dicts, equal-length finite fields."""
    recs = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError("line %d is not a JSON object" % lineno)
            for f in FIELDS_K:
                if not isinstance(r.get(f), list) or not r[f]:
                    raise ValueError("line %d: field %s missing or empty" % (lineno, f))
            k = len(r["candidate_gallery_ids"])
            for f in FIELDS_K:
                if len(r[f]) != k:
                    raise ValueError("line %d: %s has length %d, expected %d"
                                     % (lineno, f, len(r[f]), k))
                if f != "candidate_gallery_ids" and not all(
                        isinstance(x, (int, float)) and not isinstance(x, bool)
                        and math.isfinite(x) for x in r[f]):
                    raise ValueError("line %d: %s has a non-numeric or non-finite value"
                                     % (lineno, f))
            ids = r["candidate_gallery_ids"]
            if not all(isinstance(c, int) and not isinstance(c, bool) for c in ids):
                raise ValueError("line %d: candidate ids must be integers" % lineno)
            if len(set(ids)) != len(ids):
                raise ValueError("line %d: duplicate candidate ids" % lineno)
            recs.append(r)
    if not recs:
        raise ValueError("%s contains no records" % path)
    return recs


def reproduction_errors(arr: dict, recs, w: float, a: float, b: float) -> dict:
    """Worst |vector-derived - cached| for each cached quantity. The stored
    vectors are float32; every dot product, the mixture and the sigmoid are
    evaluated in float64 so the check adds no rounding of its own."""
    import numpy as np
    f64 = {name: np.asarray(arr[name], dtype=np.float64)
           for name in ("e_inv", "t_hat", "q_comp", "rec", "cinv", "cedit", "c_pool")}
    e, t, qc = f64["e_inv"], f64["t_hat"], f64["q_comp"]
    got = {"recon_cos": np.einsum("nkd,nd->nk", f64["rec"], e),
           "invariant_cos": np.einsum("nkd,nd->nk", f64["cinv"], e),
           "edit_cos": np.einsum("nkd,nd->nk", f64["cedit"], t),
           "base_score": np.einsum("nkd,nd->nk", f64["c_pool"], qc)}
    mix = (1.0 - w) * 0.5 * (got["recon_cos"] + got["invariant_cos"]) + w * got["edit_cos"]
    z = a * mix + b
    e = np.exp(-np.abs(z))                       # sign-stable sigmoid: never overflows
    got["verification_score"] = np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))
    worst = {}
    for key, val in got.items():
        cached = np.array([r[key] for r in recs], dtype=np.float64)
        d = np.abs(val - cached)
        if not np.all(np.isfinite(d)):
            raise ValueError("non-finite difference in %s" % key)
        worst[key] = float(d.max())
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--dataset", required=True, choices=("cirr", "fashioniq", "edir"))
    ap.add_argument("--category", default=None, help="FashionIQ category")
    ap.add_argument("--edir-root", default=None, help="edir only: folder holding imgs_mc/")
    ap.add_argument("--edir-meta", default=None, help="edir only: folder with queries/corpus/instances.jsonl")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True, help="output path, must end in .npz")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=None,
                    help="probe run on the first N queries (alignment check before the full pass)")
    ap.add_argument("--skip-if-complete", action="store_true",
                    help="exit 0 if the output already exists, verifies and is this run's "
                         "(validated resume); full dumps only")
    args = ap.parse_args(argv)

    if not args.out.endswith(".npz"):
        ap.error("--out must end in .npz")
    paths = artifact_paths(args.out)
    if args.skip_if_complete and args.max_queries is not None:
        ap.error("--skip-if-complete resumes full dumps; a probe (--max-queries) is never skipped")
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.max_queries is not None and args.max_queries < 1:
        ap.error("--max-queries must be positive")
    if args.dataset == "fashioniq" and args.category not in ("dress", "shirt", "toptee"):
        ap.error("--category must be dress, shirt or toptee for fashioniq")
    if (args.dataset == "edir") != (args.edir_root is not None and args.edir_meta is not None):
        ap.error("--edir-root and --edir-meta are required for edir and only for it")
    if args.dataset == "edir" and args.category is not None:
        ap.error("--category does not apply to edir")
    for flag, path in (("--config", args.config), ("--checkpoint", args.checkpoint)):
        if not os.path.isfile(path):
            ap.error("%s %s is not a file" % (flag, path))

    try:
        recs = load_cache(args.cache)
    except (OSError, ValueError) as exc:
        ap.error("invalid cache %s: %s" % (args.cache, exc))
    k = len(recs[0]["candidate_gallery_ids"])
    if any(len(r["candidate_gallery_ids"]) != k for r in recs):
        ap.error("queries have unequal candidate windows")
    for i, r in enumerate(recs):
        if r.get("query_id") != i:
            ap.error("record %d has query_id %r; expected dataset order" % (i, r.get("query_id")))

    # Content identity of the weights (path and size alone cannot tell two
    # checkpoints apart); every dump of one study must carry the same digest.
    ckpt_sha = sha256_file(args.checkpoint)
    print("[dump] checkpoint sha256 %s" % ckpt_sha, flush=True)
    if args.skip_if_complete:
        try:
            identity = run_identity(args.dataset, args.category, len(recs), args.module_combo,
                                    args.cache, args.config, ckpt_sha)
            if complete_output(paths, identity):
                print("[dump] %s is complete, verifies and matches this run; skipping"
                      % paths["npz"], flush=True)
                return 0
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    problem = output_state_error(paths)
    if problem:
        ap.error(problem)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.collate import build_collate_fn
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform, safe_load_transform
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    tokenizer = build_tokenizer(cfg.backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "cirr":
        from evaluate_cirr import build_cirr_full_gallery
        ds = DATASET_REGISTRY["cirr"](root=os.path.join(cfg.data.root, "cirr"), split="val",
                                      transform=transform)
        gallery = build_cirr_full_gallery(ds)
    elif args.dataset == "edir":
        ds = DATASET_REGISTRY["edir"](root=args.edir_root, split="test", transform=transform,
                                      meta_dir=args.edir_meta)
        gallery = list(ds.corpus_paths)
        if {t["ref_path"] for t in ds.triplets} & set(gallery):
            ap.error("an EDIR reference image is in the gallery")
    else:
        from fashioniq_cache import official_gallery
        ds = DATASET_REGISTRY["fashioniq"](root=os.path.join(cfg.data.root, "fashioniq"),
                                           split="val", transform=transform,
                                           categories=[args.category])
        gallery = official_gallery(ds, args.category)
    if len(recs) != len(ds):
        ap.error("cache has %d records, dataset %d queries" % (len(recs), len(ds)))
    if any(not (0 <= c < len(gallery)) for r in recs for c in r["candidate_gallery_ids"]):
        ap.error("a cached candidate id lies outside the gallery")
    n_q = len(ds) if args.max_queries is None else min(args.max_queries, len(ds))
    recs = recs[:n_q]

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    if model.rv is None or model.aep is None or not model.rv.cfg.use_invariant_term:
        ap.error("checkpoint must include aep and an rv with the invariant term")
    w = float(torch.sigmoid(model.rv.edit_weight).item())
    cal_a = float(model.rv.calibrator.scale.item())
    cal_b = float(model.rv.calibrator.bias.item())
    print("[dump] %s/%s | %d queries | gallery %d | k=%d | w=%.6f | calibrator %.6f, %.6f"
          % (args.dataset, args.category or split_of(args.dataset), n_q, len(gallery), k, w, cal_a, cal_b),
          flush=True)

    loader = DataLoader(Subset(ds, range(n_q)), batch_size=args.batch_size, shuffle=False,
                        collate_fn=build_collate_fn(tokenizer), num_workers=0)
    parts = {name: [] for name in ("e_inv", "t_hat", "q_comp", "rec", "cinv", "cedit", "c_pool")}
    qi, t0 = 0, time.time()
    with torch.no_grad():
        for batch in loader:
            q = model.encode_query(batch["reference_pixel_values"].to(device),
                                   batch["edit_input_ids"].to(device),
                                   batch["edit_attention_mask"].to(device))
            text = q["text_out"].pooled
            inv = model.aep(text, q["ref_out"].pooled)["invariant_embed"]
            nb = text.size(0)
            # All candidates of this query batch in one pass (nb * k images);
            # each is paired with its own query's text and invariant target,
            # exactly as evaluate_dataset pairs them.
            ids = [c for j in range(nb) for c in recs[qi + j]["candidate_gallery_ids"]]
            pix = torch.stack([safe_load_transform(gallery[c], transform) for c in ids]).to(device)
            c_out = model.backbone.encode_image(pix)
            c_pool = F.normalize(c_out.pooled, dim=-1)
            text_k = text.repeat_interleave(k, dim=0)
            rv = model.rv(candidate_tokens=c_out.tokens, candidate_pooled=c_pool,
                          forward_text_pooled=text_k, invariant_embed=inv.repeat_interleave(k, dim=0))
            cinv, cedit = model.rv.candidate_decomp(c_pool, text_k)
            batch_vec = {
                "e_inv": F.normalize(inv, dim=-1), "t_hat": F.normalize(text, dim=-1),
                "q_comp": F.normalize(q["query"], dim=-1),
                "rec": F.normalize(rv["reconstructed"], dim=-1).view(nb, k, -1),
                "cinv": F.normalize(cinv, dim=-1).view(nb, k, -1),
                "cedit": F.normalize(cedit, dim=-1).view(nb, k, -1),
                "c_pool": c_pool.view(nb, k, -1),
            }
            for name, v in batch_vec.items():
                if not torch.isfinite(v).all():
                    print("FATAL: non-finite %s at queries %d-%d" % (name, qi, qi + nb - 1),
                          file=sys.stderr)
                    return 1
                parts[name].append(v.float().cpu())
            qi += nb
            if qi % 512 < args.batch_size:
                print("[dump] %d/%d queries, %.0fs" % (qi, n_q, time.time() - t0), flush=True)

    arr = {name: torch.cat(v).numpy() for name, v in parts.items()}
    worst = reproduction_errors(arr, recs, w, cal_a, cal_b)
    print("[dump] worst |vector-derived - cached|: %s" % worst, flush=True)
    bad = tolerance_violations(worst)
    if bad:
        print("FATAL: vectors do not reproduce the cache (%s); nothing written" % "; ".join(bad),
              file=sys.stderr)
        return 1

    cand = np.array([r["candidate_gallery_ids"] for r in recs], dtype=np.int64)
    qid = np.arange(n_q, dtype=np.int64)
    # Staged publication: both artifacts are written and verified under
    # .partial names first; the manifest (the completion marker the analysis
    # requires) is renamed into place last.
    np.savez(paths["npz_partial"], query_id=qid, candidate_ids=cand, **arr)
    with np.load(paths["npz_partial"]) as z:
        loaded = {name: z[name] for name in arr}
        ok_ids = (np.array_equal(z["candidate_ids"], cand) and np.array_equal(z["query_id"], qid))
    if not ok_ids or any(not np.all(np.isfinite(v)) for v in loaded.values()):
        os.remove(paths["npz_partial"])
        print("FATAL: re-loaded file failed identity/finiteness checks", file=sys.stderr)
        return 1
    worst_reload = reproduction_errors(loaded, recs, w, cal_a, cal_b)
    if tolerance_violations(worst_reload):
        os.remove(paths["npz_partial"])
        print("FATAL: re-loaded vectors do not reproduce the cache: %s" % worst_reload,
              file=sys.stderr)
        return 1

    manifest = {
        "schema": SCHEMA, "dataset": args.dataset, "split": split_of(args.dataset), "category": args.category,
        "n_queries": n_q, "k": k, "dim": int(arr["e_inv"].shape[1]), "dtype": "float32",
        "cache_path": os.path.abspath(args.cache), "cache_sha256": sha256_file(args.cache),
        "config_path": os.path.abspath(args.config), "config_sha256": sha256_file(args.config),
        "module_combo": args.module_combo,
        "checkpoint_path": os.path.abspath(args.checkpoint),
        "checkpoint_bytes": os.path.getsize(args.checkpoint), "checkpoint_sha256": ckpt_sha,
        "gallery_size": len(gallery),
        "gallery_sha256": hashlib.sha256("\n".join(gallery).encode("utf-8")).hexdigest(),
        "mixture_w": w, "calibrator_scale": cal_a, "calibrator_bias": cal_b,
        "tolerance_policy": VECTOR_TOL, "worst_reproduction_error": worst_reload,
        "runtime": {"query_batch_size": args.batch_size,
                    "max_candidates_per_pass": args.batch_size * k,
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                    "model_dtype": str(next(model.parameters()).dtype),
                    "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                    "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                    "float32_matmul_precision": torch.get_float32_matmul_precision()},
        "npz_sha256": sha256_file(paths["npz_partial"]),
    }
    with open(paths["manifest_partial"], "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, allow_nan=False)
    os.replace(paths["npz_partial"], paths["npz"])
    os.replace(paths["manifest_partial"], paths["manifest"])
    print("[dump] wrote %s and %s (%d queries) in %.0fs"
          % (paths["npz"], paths["manifest"], n_q, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
