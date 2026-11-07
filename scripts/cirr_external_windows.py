"""Our verifier on externally supplied CIRR val windows (amendment v3.1 §25b). One GPU pass.

The model calls are exactly those of cirr_split_windows.py: the gallery is encoded with
evaluate.encode_gallery_paths, each query with model.encode_query and model.aep, and every
window candidate with model.backbone.encode_image and model.rv. The only change is which
candidates are scored: the rows in --windows (CLIP4Cir's top-10 W_C from §25a) instead of the
query's own top-k. Our verifier therefore stays attached to its own encoders; no external
feature enters it.

Parity, all required before anything is written (otherwise WindowError, nothing written):
  own windows   the query's own top-k equals the deployed cache's candidate ids for every
                query, and its base scores are within VECTOR_TOL
  shared pairs  every (query, candidate) pair present in both W_C and the cache window
                reproduces the cache's verifier fields within VECTOR_TOL (each field depends
                on that pair only); at least MIN_PAIRS pairs
  features      text_emb and image_emb reproduce the Phase B features within COS_TOL

Usage (from revua_cir/, laptop, .venv_cluster)
  python scripts/cirr_external_windows.py --config ../edir_work/laptop_config.yaml \\
      --checkpoint ../checkpoints/ablation_collective_v3/fc_aep_rv/epoch004.pt --module-combo fc+aep+rv \\
      --windows ../revision_work/followups25a_2026-11-04/windows_c.npz \\
      --check-cache ../bbnv_caches/rv_score_cache_fc_aep_rv.jsonl --check-features ../bbnv_feats/v1/cirr_val \\
      --batch-size 32 --out-dir ../revision_work/external_windows_c_2026-11-04
  add --probe (and no --out-dir) to run the first batch only and write nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from cirr_split_windows import FIELDS, TOP, WindowError, split_resolver  # noqa: E402
from dump_verifier_vectors import VECTOR_TOL, load_cache  # noqa: E402

SCHEMA = "cirr-external-windows/1"
OUTPUT_NAME = "windows_c_scored.npz"
MIN_PAIRS = 1000


def check_windows(cand: np.ndarray, query_id: np.ndarray, ref_rows: np.ndarray, recs: Sequence[dict],
                  k: int, gallery_size: int) -> None:
    if cand.ndim != 2 or cand.shape != (len(recs), k):
        raise WindowError("windows have shape %s, expected (%d, %d)" % (cand.shape, len(recs), k))
    if [int(q) for q in query_id] != [r["query_id"] for r in recs]:
        raise WindowError("windows and cache disagree in query order")
    if cand.min() < 0 or cand.max() >= gallery_size:
        raise WindowError("window rows outside the gallery")
    for i, w in enumerate(cand):
        if len(set(w.tolist())) != k or int(ref_rows[i]) in w:
            raise WindowError("query %d: repeated row or its reference in the window" % i)


def check_own_windows(top: np.ndarray, top_s: np.ndarray, recs: Sequence[dict]) -> float:
    """The query's own top-k must be the cache's candidates, with base scores within tolerance."""
    for i, r in enumerate(recs):
        if list(map(int, top[i])) != r["candidate_gallery_ids"]:
            raise WindowError("query %d: own top-k differs from the deployed cache" % i)
    worst = float(np.abs(top_s - np.array([r["base_score"] for r in recs])).max())
    if worst > VECTOR_TOL["base_score"]:
        raise WindowError("base scores differ from the cache by %.3g" % worst)
    return worst


def shared_pair_parity(cand: np.ndarray, fields: Dict[str, np.ndarray], recs: Sequence[dict],
                       min_pairs: int) -> Dict[str, object]:
    """Every (query, candidate) pair in both W_C and the cache window must reproduce the cache."""
    worst = {f: 0.0 for f in FIELDS}
    n_pairs, n_queries = 0, 0
    for i, r in enumerate(recs):
        where = {g: j for j, g in enumerate(r["candidate_gallery_ids"])}
        shared = [(jc, where[int(g)]) for jc, g in enumerate(cand[i]) if int(g) in where]
        n_pairs += len(shared)
        n_queries += bool(shared)
        for jc, jb in shared:
            for f in FIELDS:
                worst[f] = max(worst[f], abs(float(fields[f][i, jc]) - float(r[f][jb])))
    if n_pairs < min_pairs:
        raise WindowError("only %d shared pairs to check (< %d)" % (n_pairs, min_pairs))
    bad = {f: worst[f] for f in FIELDS if f in VECTOR_TOL and worst[f] > VECTOR_TOL[f]}
    if bad:
        raise WindowError("shared pairs do not reproduce the cache: %s" % bad)
    return {"worst_error": worst, "n_pairs": n_pairs, "n_queries": n_queries}


def load_live_weights(path: str, model) -> Dict[str, int]:
    """Evaluation-only equivalent of revua_cir.utils.load_checkpoint (§25.3).

    The 5.6 GB checkpoint is 1.86 GiB of live weights plus 3.47 GiB of optimizer state.
    Staging it on the GPU ran out of VRAM (§25.2) and reading it into RAM exhausted Windows
    commit and crashed the laptop (§25.3), so it is memory-mapped from disk and only
    `live_model` is read: load_state_dict copies those tensors into the model exactly.
    Stricter than the original: a lazily-shaped buffer or a missing trainable parameter
    aborts instead of warning."""
    import torch
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    live_sd = state.get("live_model") or state["model"]
    own = model.state_dict()
    lazy = [n for n, v in live_sd.items() if n in own and own[n].numel() == 0 and torch.is_tensor(v) and v.numel()]
    if lazy:
        raise WindowError("checkpoint fills lazily-shaped buffers, not supported here: %s" % lazy[:3])
    missing, unexpected = model.load_state_dict(live_sd, strict=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    if [n for n in missing if n in trainable]:
        raise WindowError("trainable parameters missing from the checkpoint: %s"
                          % [n for n in missing if n in trainable][:3])
    info = {"loaded": len(live_sd) - len(unexpected), "missing_frozen": len(missing), "unexpected": len(unexpected)}
    del state, live_sd
    return info


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--windows", required=True, help="§25a windows_c.npz (query_id, cand, ref_row)")
    ap.add_argument("--check-cache", required=True)
    ap.add_argument("--check-features", required=True)
    ap.add_argument("--out-dir", help="fresh output directory (not with --probe)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: cfg.data.eval_batch_size")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    if args.probe == (args.out_dir is not None):
        ap.error("give --out-dir for a real run, and none for --probe")
    for path in (args.config, args.checkpoint, args.check_cache, args.windows):
        if not os.path.isfile(path):
            ap.error("%s is not a file" % path)
    if args.out_dir is not None:
        if os.path.exists(ext.man.staging_dir(args.out_dir)):
            ap.error("a staging directory from an interrupted run exists; use a fresh --out-dir")
        try:
            prov.prepare_fresh_dir(args.out_dir)
        except IOError as exc:
            ap.error(str(exc))

    import torch
    import torch.nn.functional as F
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda, but this Python environment cannot use CUDA")
    device = torch.device(args.device)
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import encode_gallery_paths
    from revua_cir.models.revua_cir import ReVUACIR

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    bs = args.batch_size or cfg.data.eval_batch_size
    k = cfg.reverse_verifier.top_k
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds, entries = split_resolver(os.path.join(cfg.data.root, "cirr"), "val")
    gallery = build_cirr_full_gallery(ds)
    gidx = {p: i for i, p in enumerate(gallery)}
    ref_paths = [ds._img_path(e["reference"]) for e in entries]
    ext.require_files(list(gallery) + ref_paths)
    ref_rows = np.array([gidx.get(p, -1) for p in ref_paths], dtype=np.int64)
    recs = load_cache(args.check_cache)
    with np.load(os.path.join(args.check_features, ext.FEATURES_NAME), allow_pickle=False) as z:
        feat = {k_: z[k_] for k_ in ("image_emb", "text_emb")}
    with np.load(args.windows, allow_pickle=False) as z:
        cand, query_id, w_ref = z["cand"].astype(np.int64), z["query_id"], z["ref_row"]
    if len(entries) != len(recs) or not np.array_equal(w_ref, ref_rows):
        raise WindowError("windows file, captions and gallery disagree on the reference rows")
    check_windows(cand, query_id, ref_rows, recs, k, len(gallery))
    n_q = min(bs, len(entries)) if args.probe else len(entries)
    print("[external] val | %d queries | gallery %d | k %d | batch %d | inputs checked in %.0fs"
          % (n_q, len(gallery), k, bs, time.time() - t0), flush=True)

    model = ReVUACIR(cfg).to(device)
    ckpt_info = load_live_weights(args.checkpoint, model)
    model.eval()
    off = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type != device.type]
    if off:
        raise WindowError("tensors left off %s after loading: %s" % (device, off[:5]))
    if model.rv is None or model.aep is None:
        raise WindowError("the checkpoint must include AEP and RV")
    tokenizer = build_tokenizer(cfg.backbone)
    with torch.no_grad():
        g = encode_gallery_paths(model, gallery, transform, device, bs).to(device)
        # fail fast: the gallery rows must be the Phase B rows that W_C indexes
        image_emb = g.float().cpu().numpy().astype(np.float64)
        icos = float((1.0 - np.einsum("nd,nd->n", image_emb, feat["image_emb"].astype(np.float64))).max())
        if icos > ext.COS_TOL:
            raise WindowError("gallery features do not reproduce Phase B (1 - cos %.3g)" % icos)
        print("[external] gallery reproduces Phase B (1 - cos %.3g), %.0fs" % (icos, time.time() - t0), flush=True)
        top, top_s, text = [], [], []
        win = {f: [] for f in FIELDS}
        for s in range(0, n_q, bs):
            chunk = entries[s:s + bs]
            ref = torch.stack([ext.ImageFiles([ds._img_path(e["reference"])], transform)[0] for e in chunk]).to(device)
            tok = tokenizer([e["caption"] for e in chunk])
            q = model.encode_query(ref, tok["input_ids"].to(device), tok["attention_mask"].to(device))
            scores = q["query"] @ g.t()
            for i, r in enumerate(ref_rows[s:s + len(chunk)]):
                if r >= 0:
                    scores[i, r] = float("-inf")
            ts, ti = scores.topk(min(TOP, g.size(0)), dim=-1)        # as cirr_split_windows, then the first k
            ts, ti = ts[:, :k], ti[:, :k]
            invariant = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]
            for b in range(len(chunk)):
                ids = cand[s + b].tolist()
                pix = torch.stack([ext.ImageFiles([gallery[i]], transform)[0] for i in ids]).to(device)
                c_out = model.backbone.encode_image(pix)
                rv = model.rv(candidate_tokens=c_out.tokens, candidate_pooled=F.normalize(c_out.pooled, dim=-1),
                              forward_text_pooled=q["text_out"].pooled[b:b + 1].expand(len(ids), -1),
                              invariant_embed=invariant[b:b + 1].expand(len(ids), -1))
                for f in FIELDS:
                    win[f].append(rv[f].float().cpu().numpy())
            top.append(ti.cpu().numpy())
            top_s.append(ts.float().cpu().numpy())
            text.append(F.normalize(q["text_out"].pooled, dim=-1).float().cpu().numpy())
            if (s // bs) % 8 == 0:
                print("[external] %d/%d queries, %.0fs" % (s + len(chunk), n_q, time.time() - t0), flush=True)
    fields = {f: np.stack(win[f]).astype(np.float32) for f in FIELDS}
    text_emb = np.concatenate(text).astype(np.float64)
    parity = {"own_windows_base_worst": check_own_windows(np.concatenate(top).astype(np.int64),
                                                          np.concatenate(top_s), recs[:n_q]),
              "shared_pairs": shared_pair_parity(cand[:n_q], fields, recs[:n_q], 0 if args.probe else MIN_PAIRS)}
    tcos = float((1.0 - np.einsum("nd,nd->n", text_emb, feat["text_emb"][:n_q].astype(np.float64))).max())
    if tcos > ext.COS_TOL:
        raise WindowError("features do not reproduce Phase B: text %.3g, image %.3g" % (tcos, icos))
    parity["features_max_1_minus_cos"] = {"text": tcos, "image": icos}
    print("[external] checkpoint %s | parity %s" % (ckpt_info, parity), flush=True)
    if args.probe:
        print("[external] PROBE %d queries in %.0fs | nothing written" % (n_q, time.time() - t0), flush=True)
        return 0
    arrays = {"cand": cand, **fields}
    results = {"schema": SCHEMA, "split": "val", "n_queries": n_q, "k": k, "gallery_size": len(gallery),
               "gallery_sha256": ext.gallery_digest(gallery), "parity": parity, "checkpoint_load": ckpt_info, "output_name": OUTPUT_NAME,
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "cirr_external_windows.py", "args": vars(args),
                            "config_sha256": prov.sha256_file(args.config),
                            "checkpoint_sha256": prov.sha256_file(args.checkpoint),
                            "windows_sha256": prov.sha256_file(args.windows),
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + (
                                "cirr_split_windows.py", "cirr_external_windows.py")),
                            "runtime": {"torch": torch.__version__,
                                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                        "batch_size": bs}},
               "wall_seconds": round(time.time() - t0, 1)}
    ext.publish_npz_run(args.out_dir, arrays, results, name=OUTPUT_NAME)
    print("[external] wrote, re-loaded and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
