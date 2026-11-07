"""Tower embeddings for the CIRCO val windows (amendment v3.1 §27). One small GPU pass.

Encodes, with one backbone's matched towers, exactly what the CIRCO second stage needs:
  image_emb  normalize(backbone.encode_image(...).pooled) for every image in the cached top-10
             windows of the deployed CIRCO cache (2153 unique gallery rows)
  text_emb   normalize(backbone.encode_text(...).pooled) for the 220 relative captions
The gallery is CIRCO's official one, the sorted COCO2017-unlabeled list
(evaluate_circo.build_circo_full_gallery), so cache gallery indices map to files exactly.

  --checkpoint PATH   a trained checkpoint (memory-mapped live weights, §25.3)
  --pretrained        the backbone as constructed from cfg.backbone.pretrained_name
  --reproduce-base    deployed checkpoint only: also run model.encode_query on the 220 queries and
                      require q . image_emb to reproduce the cache's base_score for every window
                      candidate within VECTOR_TOL["base_score"] (checks the file mapping, the caption
                      field and the preprocessing); otherwise nothing is written

Usage (from revua_cir/, laptop, .venv_cluster)
  ... circo_towers.py --config ../edir_work/laptop_config.yaml --circo-root ../datasets/circo
      --cache ../rv_score_cache_circo_fc_aep_rv.jsonl --module-combo fc+aep+rv
      --checkpoint ../checkpoints/ablation_collective_v3/fc_aep_rv/epoch004.pt --reproduce-base
      --out-dir ../revision_work/circo_towers_deployed_2026-11-04
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from cirr_external_windows import load_live_weights  # noqa: E402
from cirr_split_windows import WindowError  # noqa: E402
from dump_verifier_vectors import VECTOR_TOL  # noqa: E402

SCHEMA = "circo-towers/1"
OUTPUT_NAME = "circo_towers.npz"


def image_id(path: str) -> int:
    return int(os.path.splitext(os.path.basename(path))[0])


def check_alignment(ann: Sequence[dict], recs: Sequence[dict], gallery: Sequence[str]) -> List[int]:
    """Cache and annotations must describe the same queries and images; returns window rows."""
    if len(ann) != len(recs):
        raise WindowError("%d annotations, %d cache records" % (len(ann), len(recs)))
    ids = [image_id(p) for p in gallery]
    pos = {i: k for k, i in enumerate(ids)}
    for a, r in zip(ann, recs):
        if r["query_id"] != a["id"]:
            raise WindowError("query order differs at id %r" % a["id"])
        if sorted(ids[g] for g in r["relevant_gallery_indices"]) != sorted(a["gt_img_ids"]):
            raise WindowError("query %d: ground truth does not map to the gallery" % a["id"])
        if pos[a["reference_img_id"]] in r["base_top50_gallery_ids"]:
            raise WindowError("query %d: reference inside the ranking" % a["id"])
        if r["base_top50_gallery_ids"][:len(r["candidate_gallery_ids"])] != r["candidate_gallery_ids"]:
            raise WindowError("query %d: window is not the head of the top-50" % a["id"])
    return sorted({int(g) for r in recs for g in r["candidate_gallery_ids"]})


def base_parity(q: np.ndarray, rows: Sequence[int], img: np.ndarray, recs: Sequence[dict]) -> float:
    at = {g: k for k, g in enumerate(rows)}
    worst = 0.0
    for i, r in enumerate(recs):
        got = img[[at[g] for g in r["candidate_gallery_ids"]]] @ q[i]
        worst = max(worst, float(np.abs(got - np.asarray(r["base_score"], dtype=np.float64)).max()))
    if worst > VECTOR_TOL["base_score"]:
        raise WindowError("q . image does not reproduce the cache base_score (worst %.3g)" % worst)
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--circo-root", required=True)
    ap.add_argument("--cache", required=True, help="the deployed CIRCO val score cache (jsonl)")
    ap.add_argument("--module-combo", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--pretrained", action="store_true")
    ap.add_argument("--reproduce-base", action="store_true")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args(argv)
    if args.reproduce_base and not args.checkpoint:
        ap.error("--reproduce-base needs the deployed --checkpoint")
    for path in (args.config, args.cache, args.checkpoint):
        if path is not None and not os.path.isfile(path):
            ap.error("%s is not a file" % path)
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
    from evaluate_circo import build_circo_full_gallery
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.models.revua_cir import ReVUACIR

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    gallery = build_circo_full_gallery(args.circo_root)
    ann_path = os.path.join(args.circo_root, "annotations", "val.json")
    with open(ann_path, encoding="utf-8") as fh:
        ann = json.load(fh)
    with open(args.cache, encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    rows = check_alignment(ann, recs, gallery)
    pos = {image_id(p): k for k, p in enumerate(gallery)}
    ext.require_files([gallery[g] for g in rows] + [gallery[pos[a["reference_img_id"]]] for a in ann])

    model = ReVUACIR(cfg)
    ckpt_info = load_live_weights(args.checkpoint, model) if args.checkpoint else {"pretrained": cfg.backbone.pretrained_name}
    model = model.to(device).eval()
    tokenizer = build_tokenizer(cfg.backbone)
    print("[circo-towers] %s | %d queries | %d window images | gallery %d | %.0fs"
          % (args.checkpoint or "pretrained", len(ann), len(rows), len(gallery), time.time() - t0), flush=True)
    img, txt, qs = [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), args.batch_size):
            pix = torch.stack([ext.ImageFiles([gallery[g]], transform)[0] for g in rows[s:s + args.batch_size]]).to(device)
            img.append(F.normalize(model.backbone.encode_image(pix).pooled, dim=-1).float().cpu().numpy())
        for s in range(0, len(ann), args.batch_size):
            chunk = ann[s:s + args.batch_size]
            tok = tokenizer([a["relative_caption"] for a in chunk])
            ids, mask = tok["input_ids"].to(device), tok["attention_mask"].to(device)
            txt.append(F.normalize(model.backbone.encode_text(ids, mask).pooled, dim=-1).float().cpu().numpy())
            if args.reproduce_base:
                ref = torch.stack([ext.ImageFiles([gallery[pos[a["reference_img_id"]]]], transform)[0]
                                   for a in chunk]).to(device)
                qs.append(model.encode_query(ref, ids, mask)["query"].float().cpu().numpy())
    arrays = {"rows": np.asarray(rows, dtype=np.int64), "query_id": np.asarray([a["id"] for a in ann], dtype=np.int64),
              "image_emb": np.concatenate(img).astype(np.float32), "text_emb": np.concatenate(txt).astype(np.float32)}
    ext.check_unit_rows("image_emb", arrays["image_emb"], 2)
    ext.check_unit_rows("text_emb", arrays["text_emb"], 2)
    parity = None
    if args.reproduce_base:
        parity = {"base_score_worst": base_parity(np.concatenate(qs).astype(np.float64), rows,
                                                  arrays["image_emb"].astype(np.float64), recs)}
    print("[circo-towers] encoded in %.0fs | parity %s" % (time.time() - t0, parity), flush=True)
    results = {"schema": SCHEMA, "split": "val", "n_queries": len(ann), "n_rows": len(rows),
               "source": "pretrained" if args.pretrained else "checkpoint", "checkpoint_load": ckpt_info,
               "parity_base": parity, "output_name": OUTPUT_NAME,
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "circo_towers.py", "args": vars(args),
                            "config_sha256": prov.sha256_file(args.config),
                            "cache_sha256": prov.sha256_file(args.cache),
                            "annotations_sha256": prov.sha256_file(ann_path),
                            "checkpoint_sha256": prov.sha256_file(args.checkpoint) if args.checkpoint else None,
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + (
                                "cirr_split_windows.py", "cirr_external_windows.py", "circo_towers.py")),
                            "runtime": {"torch": torch.__version__,
                                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                        "batch_size": args.batch_size}},
               "wall_seconds": round(time.time() - t0, 1)}
    ext.publish_npz_run(args.out_dir, arrays, results, name=OUTPUT_NAME)
    print("[circo-towers] wrote, re-loaded and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
