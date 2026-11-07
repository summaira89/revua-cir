"""Timing-only profile of the EDIR extraction costs on this machine (v3.1 §19.1, condition 6).

Measures, with no labels, no retrieval and no outcome of any kind:
  decode      load_image + eval transform per image (CPU), for corpus PNGs and
              reference JPGs, plus their pixel sizes
  encode      model.encode_gallery on pre-decoded pixels, at --batch-size
  window      one verifier window: encode 10 images (tokens) + model.rv, timed
              per window. The 10 images are the first corpus files, the same
              for every window, so no query's real candidates are touched.
Prints the per-unit costs and a projection for each planned stage. Nothing is written.

  python scripts/edir_profile.py --config CFG --checkpoint CKPT --edir-root ../edir_data --meta ../edir_meta/72581ea0
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_CORPUS, N_REF, N_WINDOWS, WINDOW = 256, 64, 32, 10


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--edir-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import EDIRDataset
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform, load_image
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    if not torch.cuda.is_available():
        ap.error("CUDA is not available")
    dev = torch.device("cuda")
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation("fc+aep+rv")
    tf = build_image_transform(cfg.data.image_size, train=False)
    ds = EDIRDataset(root=args.edir_root, split="test", transform=None, meta_dir=args.meta)
    corpus = ds.corpus_paths[:N_CORPUS]
    refs = sorted({t["ref_path"] for t in ds.triplets})[:N_REF]

    def decode(paths):
        t0 = time.perf_counter()
        pix = [tf(load_image(p)) for p in paths]
        return torch.stack(pix), (time.perf_counter() - t0) / len(paths)

    sizes = [Image.open(p).size for p in corpus[:64]]
    pix_c, dec_c = decode(corpus)
    _, dec_r = decode(refs)

    model = ReVUACIR(cfg).to(dev)
    load_checkpoint(args.checkpoint, model, map_location=dev)
    model.eval()
    tok = build_tokenizer(cfg.backbone)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model.encode_gallery(pix_c[:args.batch_size].to(dev))          # warm-up
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, len(pix_c), args.batch_size):
            model.encode_gallery(pix_c[s:s + args.batch_size].to(dev))
        torch.cuda.synchronize()
        enc = (time.perf_counter() - t0) / len(pix_c)

        t = tok([ds.triplets[0]["text"]])
        ref = pix_c[:1].to(dev)                                          # any image: timing only
        q = model.encode_query(ref, t["input_ids"].to(dev), t["attention_mask"].to(dev))
        inv = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]
        win_paths = ds.corpus_paths[:WINDOW]
        times = []
        for _ in range(N_WINDOWS + 1):
            t0 = time.perf_counter()
            pix = torch.stack([tf(load_image(p)) for p in win_paths]).to(dev)
            c = model.backbone.encode_image(pix)
            model.rv(candidate_tokens=c.tokens, candidate_pooled=F.normalize(c.pooled, dim=-1),
                     forward_text_pooled=q["text_out"].pooled.expand(WINDOW, -1),
                     invariant_embed=inv.expand(WINDOW, -1))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        win = float(np.median(times[1:]))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    w, h = zip(*sizes)
    print("corpus PNG size: median %dx%d (min %d, max %d px wide)" % (np.median(w), np.median(h), min(w), max(w)))
    print("decode+transform: corpus %.1f ms/img, reference %.1f ms/img (CPU, one process)" % (1e3 * dec_c, 1e3 * dec_r))
    print("GPU encode: %.1f ms/img at batch %d | one %d-candidate verifier window incl. decode: %.0f ms | peak GPU %.2f GiB"
          % (1e3 * enc, args.batch_size, WINDOW, 1e3 * win, peak))
    n_c, n_q = len(ds.corpus_paths), len(ds)
    n_ann = sum(len(a["neg"]) + 1 for a in ds.annotated)
    gallery = n_c * (dec_c + enc)
    queries = n_q * (dec_r + enc)                         # reference encode per query (the cache stage)
    windows = n_q * win
    annotated = n_ann / WINDOW * win
    print("projection (minutes): gallery pass %.0f | query pass %.0f | retrieval windows %.0f | annotated sets %.0f"
          % (gallery / 60, queries / 60, windows / 60, annotated / 60))
    print("  pipeline as written (gallery x2, windows x2): %.0f min" % ((2 * gallery + queries + 2 * windows + annotated) / 60))
    print("  with gallery and windows computed once:        %.0f min" % ((gallery + queries + windows + annotated) / 60))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
