"""CIRR retrieval windows and pooled features for any split, test1 included (v3.1 §18). One GPU pass.

`evaluate_dataset` writes the deployed score cache but needs ground truth,
and CIRRDataset cannot load test1 (its captions have no target_hard). This
script performs the SAME model calls on any split, from the captions and
split map directly:
  gallery     the full official split gallery (build_cirr_full_gallery order),
              encoded with evaluate.encode_gallery_paths (model.encode_gallery)
  query       model.encode_query(reference, caption); scores = query @ gallery.T,
              the query's own reference masked to -inf; top 50 kept
  window      the top K = cfg.reverse_verifier.top_k candidates: each encoded
              with model.backbone.encode_image and scored by model.rv with the
              query's text and its AEP invariant embedding, exactly as the
              cache's rerank block
  features    text_emb = normalize(text pooled) and image_emb = the gallery's
              normalised pooled embeddings (the BBNV Phase B definitions)
Images are loaded strictly (no gray placeholder).

On --split val the output must reproduce the deployed cache (--check-cache):
identical candidate ids, every field within VECTOR_TOL, and the Phase B
features (--check-features) within COS_TOL; otherwise nothing is written.

Usage
-----
  python scripts/cirr_split_windows.py --config CFG --checkpoint CKPT --module-combo fc+aep+rv \\
      --split val --check-cache CACHE --check-features ../bbnv_feats/v1/cirr_val --out-dir ../bbnv_runs/split_windows_val
  python scripts/cirr_split_windows.py ... --split test1 --out-dir ../bbnv_runs/split_windows_test1
  add --probe to run the first batch only and write nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from dump_verifier_vectors import VECTOR_TOL, load_cache  # noqa: E402

SCHEMA = "cirr-split-windows/1"
TOP = 50
FIELDS = ("cos_sim", "recon_cos", "invariant_cos", "edit_cos", "verification_score")
SPLITS = ("val", "test1")


class WindowError(ValueError):
    pass


def split_resolver(cirr_root: str, split: str):
    """A CIRRDataset carrying only the split map and path resolution (no
    triplets), so test1 -- whose captions lack target_hard -- can be resolved
    with exactly the dataset's own _img_path logic."""
    from revua_cir.data.datasets import CIRRDataset
    ds = CIRRDataset.__new__(CIRRDataset)
    ds.root = cirr_root
    ds.ann_root = CIRRDataset._resolve_ann_root(cirr_root, split)
    ds.image_root = CIRRDataset._resolve_image_root(cirr_root)
    with open(os.path.join(ds.ann_root, "image_splits", "split.rc2.%s.json" % split), encoding="utf-8") as fh:
        ds.id_to_path = json.load(fh)
    with open(os.path.join(ds.ann_root, "captions", "cap.rc2.%s.json" % split), encoding="utf-8") as fh:
        entries = json.load(fh)
    return ds, entries


def check_against_cache(cand: np.ndarray, fields: Dict[str, np.ndarray], base: np.ndarray,
                        recs: Sequence[dict]) -> Dict[str, float]:
    """Identical candidate ids and every field within VECTOR_TOL; returns the worst errors."""
    if len(recs) != len(cand):
        raise WindowError("cache has %d records, windows %d" % (len(recs), len(cand)))
    if any(list(map(int, c)) != r["candidate_gallery_ids"] for c, r in zip(cand, recs)):
        raise WindowError("window candidate ids differ from the deployed cache")
    worst = {"base_score": float(np.abs(base - np.array([r["base_score"] for r in recs])).max())}
    for f in FIELDS:
        worst[f] = float(np.abs(fields[f] - np.array([r[f] for r in recs])).max())
    missing = sorted(set(VECTOR_TOL) - set(worst))
    if missing:
        raise WindowError("no reproduction error computed for %s" % missing)
    bad = {f: worst[f] for f in VECTOR_TOL if worst[f] > VECTOR_TOL[f]}
    if bad:
        raise WindowError("windows do not reproduce the cache: %s" % bad)
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--check-cache", help="val only: the deployed score cache to reproduce")
    ap.add_argument("--check-features", help="val only: the Phase B cirr_val feature run to reproduce")
    ap.add_argument("--out-dir", help="fresh output directory (not with --probe)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: cfg.data.eval_batch_size")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args(argv)
    if (args.split == "val") != (args.check_cache is not None and args.check_features is not None):
        ap.error("--check-cache and --check-features are required for val and only for val")
    if args.probe == (args.out_dir is not None):
        ap.error("give --out-dir for a real run, and none for --probe")
    for path in (args.config, args.checkpoint, args.check_cache):
        if path is not None and not os.path.isfile(path):
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
    from revua_cir.utils import load_checkpoint

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    bs = args.batch_size or cfg.data.eval_batch_size
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds, entries = split_resolver(os.path.join(cfg.data.root, "cirr"), args.split)
    gallery = build_cirr_full_gallery(ds)
    gidx = {p: i for i, p in enumerate(gallery)}
    path_to_id = {ds._img_path(i): i for i in ds.id_to_path}
    if len(path_to_id) != len(gallery):
        raise WindowError("gallery paths and image ids are not one-to-one")
    ref_paths = [ds._img_path(e["reference"]) for e in entries]
    ext.require_files(list(gallery) + ref_paths)
    ref_rows = np.array([gidx.get(p, -1) for p in ref_paths], dtype=np.int64)
    if args.check_cache:
        recs = load_cache(args.check_cache)
        with np.load(os.path.join(args.check_features, ext.FEATURES_NAME), allow_pickle=False) as z:
            feat = {k: z[k] for k in ("image_emb", "text_emb")}
    n_q = min(bs, len(entries)) if args.probe else len(entries)
    print("[windows] %s | %d queries | gallery %d | batch %d | inputs checked in %.0fs"
          % (args.split, n_q, len(gallery), bs, time.time() - t0), flush=True)

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    if model.rv is None or model.aep is None:
        raise WindowError("the checkpoint must include AEP and RV")
    k = cfg.reverse_verifier.top_k
    tokenizer = build_tokenizer(cfg.backbone)
    with torch.no_grad():
        g = encode_gallery_paths(model, gallery, transform, device, bs).to(device)
        top50, top50s, text = [], [], []
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
            ts, ti = scores.topk(min(TOP, g.size(0)), dim=-1)
            invariant = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]
            for b in range(len(chunk)):
                ids = ti[b, :k].cpu().tolist()
                pix = torch.stack([ext.ImageFiles([gallery[i]], transform)[0] for i in ids]).to(device)
                c_out = model.backbone.encode_image(pix)
                rv = model.rv(candidate_tokens=c_out.tokens, candidate_pooled=F.normalize(c_out.pooled, dim=-1),
                              forward_text_pooled=q["text_out"].pooled[b:b + 1].expand(len(ids), -1),
                              invariant_embed=invariant[b:b + 1].expand(len(ids), -1))
                for f in FIELDS:
                    win[f].append(rv[f].float().cpu().numpy())
            top50.append(ti.cpu().numpy())
            top50s.append(ts.float().cpu().numpy())
            text.append(F.normalize(q["text_out"].pooled, dim=-1).float().cpu().numpy())
            if (s // bs) % 8 == 0:
                print("[windows] %d/%d queries, %.0fs" % (s + len(chunk), n_q, time.time() - t0), flush=True)
    arrays = {"top50": np.concatenate(top50).astype(np.int64), "top50_scores": np.concatenate(top50s),
              "text_emb": np.concatenate(text).astype(np.float32), "image_emb": g.float().cpu().numpy(),
              "ref_row": ref_rows[:n_q], "pairid": np.array([int(e["pairid"]) for e in entries[:n_q]], dtype=np.int64),
              "image_ids": np.array([path_to_id[p] for p in gallery])}
    for f in FIELDS:
        arrays[f] = np.stack(win[f]).astype(np.float32)
    ext.check_unit_rows("text_emb", arrays["text_emb"], 2)
    ext.check_unit_rows("image_emb", arrays["image_emb"], 2)
    info: Dict[str, object] = {}
    if args.check_cache:
        info["cache_worst_error"] = check_against_cache(arrays["top50"][:, :k], {f: arrays[f] for f in FIELDS},
                                                        arrays["top50_scores"][:, :k], recs[:n_q])
        tcos = float((1.0 - np.einsum("nd,nd->n", arrays["text_emb"].astype(np.float64),
                                      feat["text_emb"][:n_q].astype(np.float64))).max())
        icos = float((1.0 - np.einsum("nd,nd->n", arrays["image_emb"].astype(np.float64),
                                      feat["image_emb"].astype(np.float64))).max())
        if max(tcos, icos) > ext.COS_TOL:
            raise WindowError("features do not reproduce Phase B: text %.3g, image %.3g" % (tcos, icos))
        info["features_max_1_minus_cos"] = {"text": tcos, "image": icos}
        print("[windows] reproduces the cache %s and Phase B features %s" % (info["cache_worst_error"],
                                                                             info["features_max_1_minus_cos"]),
              flush=True)
    if args.probe:
        print("[windows] PROBE %d queries in %.0fs | nothing written" % (n_q, time.time() - t0), flush=True)
        return 0
    results = {"schema": SCHEMA, "split": args.split, "n_queries": n_q, "k": k, "gallery_size": len(gallery),
               "gallery_sha256": ext.gallery_digest(gallery), **info,
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "cirr_split_windows.py", "args": vars(args),
                            "config_sha256": prov.sha256_file(args.config),
                            "checkpoint_sha256": prov.sha256_file(args.checkpoint),
                            "captions_sha256": prov.sha256_file(os.path.join(
                                ds.ann_root, "captions", "cap.rc2.%s.json" % args.split)),
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("cirr_split_windows.py",)),
                            "runtime": {"torch": torch.__version__,
                                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                        "batch_size": bs}},
               "wall_seconds": round(time.time() - t0, 1)}
    ext.publish_npz_run(args.out_dir, arrays, results, name="windows.npz")
    print("[windows] wrote, re-loaded and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
