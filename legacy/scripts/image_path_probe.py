"""Locate an image-path discrepancy between two machines (v3.1 §19 diagnosis).

For the first --n images of the CIRR validation gallery (build_cirr_full_gallery
order), writes:
  ids         the image ids (so two machines are compared by id)
  pixel_sha   SHA-256 of each preprocessed float32 pixel tensor (eval transform,
              strict load_image): equal hashes = identical preprocessing
  emb_cpu     model.encode_gallery on CPU, float32 (no TF32 exists on CPU)
  emb_gpu_tf32 / emb_gpu_fp32   the same on CUDA with cudnn.allow_tf32 True / False
              (only with --gpu; matmul TF32 stays off, as in every project run)
plus the library versions. Run it on both machines, then compare with --compare.

  python scripts/image_path_probe.py --config CFG --checkpoint CKPT --n 8 --out probe_laptop.npz --gpu
  python scripts/image_path_probe.py --config CFG --checkpoint CKPT --n 8 --out probe_cluster.npz
  python scripts/image_path_probe.py --compare probe_laptop.npz probe_cluster.npz --phase-b FEATURES_DIR --windows WINDOWS_NPZ
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def one_minus_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return 1.0 - np.einsum("nd,nd->n", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def probe(config: str, checkpoint: str, n: int, out: str, gpu: bool) -> None:
    import PIL
    import torch
    import torchvision
    import bbnv_extract as ext
    from cirr_split_windows import split_resolver
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    cfg = ReVUAConfig.from_yaml(config).apply_module_ablation("fc+aep+rv")
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds, _ = split_resolver(os.path.join(cfg.data.root, "cirr"), "val")
    gallery = build_cirr_full_gallery(ds)[:n]
    path_to_id = {ds._img_path(i): i for i in ds.id_to_path}
    pix = torch.stack([ext.ImageFiles([p], transform)[0] for p in gallery]).float()
    sha = [hashlib.sha256(pix[i].contiguous().numpy().tobytes()).hexdigest() for i in range(len(gallery))]
    model = ReVUACIR(cfg)
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.eval()
    arrays = {"ids": np.array([path_to_id[p] for p in gallery]), "pixel_sha": np.array(sha)}
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        arrays["emb_cpu"] = model.encode_gallery(pix)["pooled"].float().numpy()
        if gpu:
            if not torch.cuda.is_available():
                raise SystemExit("--gpu given but CUDA is unavailable")
            model = model.cuda()
            for flag, key in ((True, "emb_gpu_tf32"), (False, "emb_gpu_fp32")):
                torch.backends.cudnn.allow_tf32 = flag
                arrays[key] = model.encode_gallery(pix.cuda())["pooled"].float().cpu().numpy()
            torch.backends.cudnn.allow_tf32 = True
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__, "PIL": PIL.__version__,
                "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0) if gpu else "none"}
    arrays["versions"] = np.array(json.dumps(versions))
    np.savez(out, **arrays)
    print(json.dumps(versions))
    print("wrote %s (%d images)" % (out, len(gallery)))


def compare(a_path: str, b_path: str, phase_b: Optional[str], windows: Optional[str]) -> None:
    a, b = np.load(a_path, allow_pickle=False), np.load(b_path, allow_pickle=False)
    if list(a["ids"]) != list(b["ids"]):
        raise SystemExit("the two probes cover different image ids")
    print("A:", str(a["versions"]))
    print("B:", str(b["versions"]))
    print("pixel tensors identical: %d of %d" % (int((a["pixel_sha"] == b["pixel_sha"]).sum()), len(a["ids"])))
    print("CPU emb A vs B: max 1-cos %.3g" % one_minus_cos(a["emb_cpu"], b["emb_cpu"]).max())
    ref = {}
    if phase_b and windows:
        with np.load(windows, allow_pickle=False) as w:
            row = {str(i): r for r, i in enumerate(w["image_ids"])}
        with np.load(os.path.join(phase_b, "features.npz"), allow_pickle=False) as f:
            ref["cluster cached"] = f["image_emb"][[row[str(i)] for i in a["ids"]]]
    for name, z in (("A", a), ("B", b)):
        for key in ("emb_cpu", "emb_gpu_tf32", "emb_gpu_fp32"):
            if key in z.files:
                for rname, r in ref.items():
                    print("%s %-13s vs %s: max 1-cos %.3g" % (name, key, rname, one_minus_cos(z[key], r).max()))
                if key != "emb_cpu":
                    print("%s %-13s vs %s emb_cpu: max 1-cos %.3g" % (name, key, name, one_minus_cos(z[key], z["emb_cpu"]).max()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--checkpoint")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("A_NPZ", "B_NPZ"))
    ap.add_argument("--phase-b", help="cluster Phase B cirr_val feature folder (with --compare)")
    ap.add_argument("--windows", help="cluster split_windows_val windows.npz, for the gallery id order (with --compare)")
    args = ap.parse_args(argv)
    if args.compare:
        compare(args.compare[0], args.compare[1], args.phase_b, args.windows)
        return 0
    if not (args.config and args.checkpoint and args.out):
        ap.error("--config, --checkpoint and --out are required to probe")
    if os.path.exists(args.out):
        ap.error("%s exists" % args.out)
    probe(args.config, args.checkpoint, args.n, args.out, args.gpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
