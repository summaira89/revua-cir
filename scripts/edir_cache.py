"""EDIR deployed score cache, protocol `edir_retrieval` (v3.1 §15.1, §15.5, §19). GPU.

The same model calls as fashioniq_cache.py: load_checkpoint, then
evaluate_dataset with rerank=True and cache_path, here with the full EDIR
corpus (28,645 images, corpus.jsonl order) as the gallery, so each query's
window is its deployed top-10 over the real corpus. No target is inserted and
no reference is in the gallery (asserted). Differences from fashioniq_cache.py:
  * no recall or other outcome is printed (EDIR outcomes stay unseen until the
    outputs are locked, §19);
  * data.eval_batch_size is set by --eval-batch-size (default 32, the laptop's
    memory; §19 allows batch sizes to differ) and recorded;
  * a manifest is written beside the cache: protocol, checkpoint, config,
    EDIR revision and metadata hashes, gallery digest, cache SHA-256, runtime.

  python scripts/edir_cache.py --config ../edir_work/laptop_config.yaml --checkpoint CKPT \\
      --module-combo fc+aep+rv --edir-root ../edir_data --meta ../edir_meta/72581ea0 \\
      --out ../edir_work/edir_cache.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402

SCHEMA = "edir-cache/1"
PROTOCOL = "edir_retrieval"
EDIR_REVISION = "72581ea0922cd01ccceb39fd3a69616170ca0125"
EXPECTED = {"queries": 5000, "corpus": 28645}
META_FILES = ("queries.jsonl", "corpus.jsonl", "instances.jsonl")


def check_cache(cache_path: str, ds, gallery) -> int:
    """Structural checks only (no outcome): one record per query in order, the
    target index is the query's own positive, and the window is K valid ids."""
    gidx = {p: i for i, p in enumerate(gallery)}
    n = 0
    with open(cache_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["query_id"] != n or r["target_gallery_index"] != gidx[ds.triplets[n]["target_path"]]:
                raise RuntimeError("cache record %d is out of order or has the wrong target" % n)
            cand = r["candidate_gallery_ids"]
            if len(cand) != 10 or len(set(cand)) != 10 or not all(0 <= c < len(gallery) for c in cand):
                raise RuntimeError("cache record %d has an invalid window" % n)
            n += 1
    if n != len(ds):
        raise RuntimeError("cache has %d records for %d queries" % (n, len(ds)))
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--edir-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True, help="cache path (.jsonl); must not exist")
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-queries", type=int, default=None, help="smoke test only; the real run uses all")
    args = ap.parse_args(argv)
    if not args.out.endswith(".jsonl"):
        ap.error("--out must end in .jsonl")
    manifest_path = args.out[:-6] + ".manifest.json"
    for p in (args.out, manifest_path):
        if os.path.exists(p):
            ap.error("%s exists; refusing to overwrite" % p)

    import PIL
    import torch
    import torchvision
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import EDIRDataset
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.evaluate import evaluate_dataset
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    cfg.data.eval_batch_size = args.eval_batch_size
    if cfg.reverse_verifier.top_k != 10:
        ap.error("the registered window is K = 10")
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds = EDIRDataset(root=args.edir_root, split="test", transform=transform, meta_dir=args.meta)
    gallery = list(ds.corpus_paths)
    if (len(ds), len(gallery)) != (EXPECTED["queries"], EXPECTED["corpus"]):
        print("FATAL: EDIR has %d queries and %d corpus images, expected %s"
              % (len(ds), len(gallery), EXPECTED), file=sys.stderr)
        return 1
    if {t["ref_path"] for t in ds.triplets} & set(gallery):
        print("FATAL: a reference image is in the retrieval gallery", file=sys.stderr)
        return 1
    n_q = len(ds) if args.max_queries is None else min(args.max_queries, len(ds))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[edir-cache] %s | %d queries | gallery %d | batch %d | %s"
          % (PROTOCOL, n_q, len(gallery), args.eval_batch_size, device), flush=True)

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    if model.rv is None:
        print("FATAL: model has no verifier (module combo must include rv)", file=sys.stderr)
        return 1
    tokenizer = build_tokenizer(cfg.backbone)
    # Written under a .partial name and renamed only after the structural check,
    # so an interrupted run never leaves a file under the final name.
    partial = args.out + ".partial"
    if os.path.exists(partial):
        os.remove(partial)                                 # an earlier interrupted attempt
    evaluate_dataset(model, ds, cfg, tokenizer, transform, device, ks=(1, 5, 10, 50), use_map=False,
                     use_group_recall=False, full_gallery_paths=gallery, max_queries=n_q,
                     rerank=True, cache_path=partial)      # returned metrics are deliberately not used
    if args.max_queries is None:
        check_cache(partial, ds, gallery)
    os.replace(partial, args.out)
    manifest = {
        "schema": SCHEMA, "protocol": PROTOCOL, "n_queries": n_q, "complete": args.max_queries is None,
        "gallery_size": len(gallery), "gallery_sha256": ext.gallery_digest(gallery),
        "corpus_ids_sha256": hashlib.sha256("\n".join(ds.corpus_ids).encode("utf-8")).hexdigest(),
        "cache_sha256": prov.sha256_file(args.out),
        "checkpoint_sha256": prov.sha256_file(args.checkpoint),
        "config_sha256": prov.sha256_file(args.config), "eval_batch_size": args.eval_batch_size,
        "edir_revision": EDIR_REVISION,
        "edir_meta_sha256": {f: prov.sha256_file(os.path.join(args.meta, f)) for f in META_FILES},
        "source_sha256": prov.source_hashes(("edir_cache.py",)),
        "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__, "PIL": PIL.__version__,
                    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
        "wall_seconds": round(time.time() - t0, 1)}
    with open(manifest_path, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("[edir-cache] wrote %s (%d records) and its manifest in %.0fs"
          % (args.out, n_q, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
