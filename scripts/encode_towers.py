"""Text and image tower embeddings of one backbone on CIRR val (amendment v3.1 §26b). One GPU pass.

For the encoder comparison on fixed CLIP4Cir windows, this encodes the 2297-image validation
gallery and the 4181 relative captions with the backbone towers only, exactly as the Phase B
features were produced:
  image_emb  normalize(model.backbone.encode_image(pixels).pooled)   (= encode_gallery's pooled)
  text_emb   normalize(model.backbone.encode_text(ids, mask).pooled) (= encode_query's text_out)
The image and text towers always come from the same weights.

  --checkpoint PATH   load the live weights of a trained checkpoint (memory-mapped, §25.3)
  --pretrained        no checkpoint: the backbone as constructed from cfg.backbone.pretrained_name
                      (openai/clip-vit-large-patch14), i.e. off-the-shelf CLIP
  --reproduce DIR     required parity for the deployed checkpoint: both arrays must reproduce the
                      Phase B features in DIR within COS_TOL, otherwise nothing is written

Usage (from revua_cir/, laptop, .venv_cluster)
  parity:     ... encode_towers.py --config ../edir_work/laptop_config.yaml --module-combo fc+aep+rv
                  --checkpoint ../checkpoints/ablation_collective_v3/fc_aep_rv/epoch004.pt
                  --reproduce ../bbnv_feats/v1/cirr_val --out-dir ../revision_work/towers_deployed_2026-11-04
  pretrained: ... encode_towers.py --config ../edir_work/laptop_config.yaml --module-combo fc+aep+rv
                  --pretrained --out-dir ../revision_work/towers_pretrained_2026-11-04
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from cirr_external_windows import load_live_weights  # noqa: E402
from cirr_split_windows import WindowError, split_resolver  # noqa: E402

SCHEMA = "cirr-towers/1"
OUTPUT_NAME = "towers.npz"


def max_one_minus_cos(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise WindowError("shape %s != %s" % (a.shape, b.shape))
    return float((1.0 - np.einsum("nd,nd->n", a.astype(np.float64), b.astype(np.float64))).max())


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--module-combo", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint")
    src.add_argument("--pretrained", action="store_true")
    ap.add_argument("--reproduce", help="Phase B feature directory the output must reproduce")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args(argv)
    if args.pretrained and args.reproduce:
        ap.error("--reproduce applies to a trained checkpoint only")
    for path in (args.config, args.checkpoint):
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
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.models.revua_cir import ReVUACIR

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds, entries = split_resolver(os.path.join(cfg.data.root, "cirr"), "val")
    gallery = build_cirr_full_gallery(ds)
    ext.require_files(list(gallery))
    want = None
    if args.reproduce:
        with np.load(os.path.join(args.reproduce, ext.FEATURES_NAME), allow_pickle=False) as z:
            want = {k: z[k] for k in ("image_emb", "text_emb")}

    model = ReVUACIR(cfg)
    ckpt_info = load_live_weights(args.checkpoint, model) if args.checkpoint else {"pretrained": cfg.backbone.pretrained_name}
    backbone = model.backbone.to(device).eval()
    del model
    tokenizer = build_tokenizer(cfg.backbone)
    print("[towers] %s | gallery %d | captions %d | %.0fs"
          % (args.checkpoint or "pretrained " + cfg.backbone.pretrained_name, len(gallery), len(entries),
             time.time() - t0), flush=True)
    img, txt = [], []
    with torch.no_grad():
        for s in range(0, len(gallery), args.batch_size):
            pix = torch.stack([ext.ImageFiles([p], transform)[0] for p in gallery[s:s + args.batch_size]]).to(device)
            img.append(F.normalize(backbone.encode_image(pix).pooled, dim=-1).float().cpu().numpy())
        for s in range(0, len(entries), args.batch_size):
            tok = tokenizer([e["caption"] for e in entries[s:s + args.batch_size]])
            out = backbone.encode_text(tok["input_ids"].to(device), tok["attention_mask"].to(device))
            txt.append(F.normalize(out.pooled, dim=-1).float().cpu().numpy())
    arrays = {"image_emb": np.concatenate(img).astype(np.float32), "text_emb": np.concatenate(txt).astype(np.float32)}
    ext.check_unit_rows("text_emb", arrays["text_emb"], 2)
    ext.check_unit_rows("image_emb", arrays["image_emb"], 2)
    parity = None
    if want is not None:
        parity = {k: max_one_minus_cos(arrays[k], want[k]) for k in ("image_emb", "text_emb")}
        if max(parity.values()) > ext.COS_TOL:
            raise WindowError("towers do not reproduce Phase B: %s" % parity)
    print("[towers] encoded in %.0fs | parity %s" % (time.time() - t0, parity), flush=True)
    results = {"schema": SCHEMA, "split": "val", "n_queries": len(entries), "gallery_size": len(gallery),
               "gallery_sha256": ext.gallery_digest(gallery), "source": "pretrained" if args.pretrained else "checkpoint",
               "checkpoint_load": ckpt_info, "parity_vs_phase_b": parity, "output_name": OUTPUT_NAME,
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "encode_towers.py", "args": vars(args),
                            "config_sha256": prov.sha256_file(args.config),
                            "checkpoint_sha256": prov.sha256_file(args.checkpoint) if args.checkpoint else None,
                            "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + (
                                "cirr_split_windows.py", "cirr_external_windows.py", "encode_towers.py")),
                            "runtime": {"torch": torch.__version__,
                                        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                        "batch_size": args.batch_size}},
               "wall_seconds": round(time.time() - t0, 1)}
    ext.publish_npz_run(args.out_dir, arrays, results, name=OUTPUT_NAME)
    print("[towers] wrote, re-loaded and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
