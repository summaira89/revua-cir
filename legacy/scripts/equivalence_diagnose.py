"""Diagnose a failed §19 equivalence test (v3.1 §19). Read-only with respect to
every frozen artefact; writes one diagnostic npz.

Recomputes, for the first --n CIRR validation queries, exactly the retrieval
calls of cirr_split_windows.py (encode_gallery_paths over the full official
gallery, model.encode_query, reference masked, top 50), then reports:
  * gallery image embeddings vs the cluster's Phase B image_emb (all images)
  * text embeddings vs the cluster's Phase B text_emb (first n queries)
  * per query: is the top-K candidate list identical to the deployed cache?
    For a mismatch: the first differing position, and the cluster's own score
    gap between the candidates involved (a near-tie flip has a tiny gap)
  * stage timings and peak GPU memory
The verifier window (model.rv) is not run here; it is only needed once the
rankings agree.

  python scripts/equivalence_diagnose.py --config CFG --checkpoint CKPT --cache CACHE \\
      --features ../bbnv_feats/v1/cirr_val --n 200 --batch-size 32 --out ../edir_work/diagnose_200.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_extract as ext  # noqa: E402
from cirr_split_windows import TOP, split_resolver  # noqa: E402
from dump_verifier_vectors import load_cache  # noqa: E402


def compare_rankings(top: np.ndarray, top_scores: np.ndarray, recs: Sequence[dict], k: int) -> dict:
    """Per-query comparison of laptop top-k with the deployed cache."""
    same, mism = 0, []
    score_err = 0.0
    for i, r in enumerate(recs):
        lap = [int(x) for x in top[i, :k]]
        clu = [int(x) for x in r["candidate_gallery_ids"]]
        if lap == clu:
            same += 1
            score_err = max(score_err, float(np.abs(top_scores[i, :k] - np.asarray(r["base_score"])).max()))
            continue
        pos = next(j for j in range(k) if lap[j] != clu[j])
        cs = np.asarray(r["base_score"], dtype=np.float64)
        gap = float(cs[pos] - cs[pos + 1]) if pos + 1 < k else float("nan")
        mism.append({"query": i, "first_diff_pos": pos, "same_set": sorted(lap) == sorted(clu),
                     "cluster_gap_at_pos": gap,
                     "laptop_minus_cluster_score_at_pos": float(top_scores[i, pos] - cs[pos])})
    return {"n": len(recs), "identical": same, "mismatches": mism, "max_score_err_identical": score_err}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists" % args.out)

    import PIL
    import torch
    import torch.nn.functional as F
    import torchvision
    from evaluate_cirr import build_cirr_full_gallery
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import encode_gallery_paths
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    if not torch.cuda.is_available():
        ap.error("CUDA is not available")
    device = torch.device("cuda")
    print("torch %s | torchvision %s | PIL %s | %s" % (torch.__version__, torchvision.__version__,
                                                        PIL.__version__, torch.cuda.get_device_name(0)), flush=True)
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation("fc+aep+rv")
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds, entries = split_resolver(os.path.join(cfg.data.root, "cirr"), "val")
    gallery = build_cirr_full_gallery(ds)
    gidx = {p: i for i, p in enumerate(gallery)}
    entries = entries[:args.n]
    ref_paths = [ds._img_path(e["reference"]) for e in entries]
    ext.require_files(list(gallery) + ref_paths)
    ref_rows = np.array([gidx.get(p, -1) for p in ref_paths], dtype=np.int64)
    recs = load_cache(args.cache)[:args.n]
    with np.load(os.path.join(args.features, ext.FEATURES_NAME), allow_pickle=False) as z:
        feat = {k: z[k] for k in ("image_emb", "text_emb")}

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    tokenizer = build_tokenizer(cfg.backbone)
    k = cfg.reverse_verifier.top_k
    bs = args.batch_size
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        t0 = time.time()
        g = encode_gallery_paths(model, gallery, transform, device, bs).to(device)
        torch.cuda.synchronize()
        t_gal = time.time() - t0
        top, tops, text, qv = [], [], [], []
        t0 = time.time()
        for s in range(0, len(entries), bs):
            chunk = entries[s:s + bs]
            ref = torch.stack([ext.ImageFiles([ds._img_path(e["reference"])], transform)[0] for e in chunk]).to(device)
            tok = tokenizer([e["caption"] for e in chunk])
            q = model.encode_query(ref, tok["input_ids"].to(device), tok["attention_mask"].to(device))
            scores = q["query"] @ g.t()
            for i, r in enumerate(ref_rows[s:s + len(chunk)]):
                if r >= 0:
                    scores[i, r] = float("-inf")
            ts, ti = scores.topk(min(TOP, g.size(0)), dim=-1)
            top.append(ti.cpu().numpy())
            tops.append(ts.float().cpu().numpy())
            text.append(F.normalize(q["text_out"].pooled, dim=-1).float().cpu().numpy())
            qv.append(q["query"].float().cpu().numpy())
        torch.cuda.synchronize()
        t_q = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    top, tops = np.concatenate(top), np.concatenate(tops)
    text, image = np.concatenate(text), g.float().cpu().numpy()

    def one_minus_cos(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        return 1.0 - np.einsum("nd,nd->n", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    icos, tcos = one_minus_cos(image, feat["image_emb"]), one_minus_cos(text, feat["text_emb"][:len(entries)])
    rank = compare_rankings(top, tops, recs, k)
    print("timing: gallery %d images %.0fs (%.1f img/s) | %d queries %.0fs | peak GPU %.2f GiB"
          % (len(gallery), t_gal, len(gallery) / t_gal, len(entries), t_q, peak), flush=True)
    print("image_emb 1-cos vs cluster: max %.3g median %.3g, %d of %d above COS_TOL %.0e"
          % (icos.max(), np.median(icos), int((icos > ext.COS_TOL).sum()), len(icos), ext.COS_TOL))
    print("text_emb  1-cos vs cluster: max %.3g median %.3g, %d of %d above COS_TOL"
          % (tcos.max(), np.median(tcos), int((tcos > ext.COS_TOL).sum()), len(tcos)))
    print("top-%d identical to the cache: %d of %d; max |base score error| on identical: %.3g"
          % (k, rank["identical"], rank["n"], rank["max_score_err_identical"]))
    for m in rank["mismatches"][:15]:
        print("  mismatch q%d pos %d same_set=%s cluster_gap %.3g laptop-cluster score %.3g"
              % (m["query"], m["first_diff_pos"], m["same_set"], m["cluster_gap_at_pos"],
                 m["laptop_minus_cluster_score_at_pos"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, top50=top, top50_scores=tops, text_emb=text, image_emb=image, query=np.concatenate(qv),
             image_1mcos=icos, text_1mcos=tcos)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
