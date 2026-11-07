"""EDIR annotated-set scores, protocol `edir_annotated` (v3.1 §15.1, §15.2, §19.1). GPU.

Track A Q1a/Q1b discriminate each query's annotated candidates: its single
positive and its 4 or 5 annotated negatives, all edits of the query's own
source. This scores exactly those candidates with the deployed model, using
the model calls of evaluate_dataset's rerank block (as cirr_split_windows.py,
whose reproduction of the deployed cache was verified on all 4,181 CIRR
queries): model.encode_query on the reference and text; the AEP invariant
target; each candidate through model.backbone.encode_image and model.rv;
base_score = composed query . normalised candidate embedding.

The protocols stay separate (§19.1, condition 3): annotated candidates are
never inserted into a retrieval window, and this file is its own output with
protocol "edir_annotated". Candidates are stored in corpus-row order, so the
positive's position carries no information.

Label-free cross-check, required: every (query, candidate) pair that also
appears in the `edir_retrieval` cache must have the same base_score and verifier
terms there, within VECTOR_TOL (cos_sim within 1e-4). The two files are
computed by different code paths, so this ties this script's numbers to
evaluate_dataset's.

  python scripts/edir_annotated_cache.py --config CFG --checkpoint CKPT --module-combo fc+aep+rv \\
      --edir-root ../edir_data --meta ../edir_meta/72581ea0 \\
      --retrieval-cache ../edir_work/edir_cache.jsonl --out ../edir_work/edir_annotated.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_extract as ext  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from dump_verifier_vectors import VECTOR_TOL, load_cache  # noqa: E402
from edir_cache import EDIR_REVISION, EXPECTED, META_FILES  # noqa: E402

SCHEMA = "edir-annotated/1"
PROTOCOL = "edir_annotated"
FIELDS = ("base_score", "cos_sim", "recon_cos", "invariant_cos", "edit_cos", "verification_score")
CROSS_TOL = {**VECTOR_TOL, "cos_sim": 1e-4}


def annotated_rows(ds, gidx: Mapping[str, int], image_dir: str) -> List[Dict[str, object]]:
    """Per query: candidate corpus rows in ascending order, and the positive's position."""
    out = []
    for t, a in zip(ds.triplets, ds.annotated):
        rows = sorted(gidx[os.path.join(image_dir, f)] for f in [a["pos"]] + a["neg"])
        tgt = gidx[t["target_path"]]
        out.append({"rows": rows, "target_row": tgt, "target_pos": rows.index(tgt)})
    return out


def cross_check(records: Sequence[Mapping], retrieval: Sequence[Mapping]) -> Dict[str, object]:
    """Worst |annotated - retrieval| per field over every shared (query, candidate)."""
    worst = {f: 0.0 for f in FIELDS}
    shared = 0
    for a, r in zip(records, retrieval):
        pos_r = {c: i for i, c in enumerate(r["candidate_gallery_ids"])}
        for j, c in enumerate(a["candidate_gallery_ids"]):
            if c in pos_r:
                shared += 1
                for f in FIELDS:
                    worst[f] = max(worst[f], abs(a[f][j] - r[f][pos_r[c]]))
    bad = {f: worst[f] for f in FIELDS if worst[f] > CROSS_TOL[f]}
    return {"shared_pairs": shared, "worst": worst, "violations": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--edir-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--retrieval-cache", required=True, help="the edir_retrieval cache (for the cross-check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--query-batch", type=int, default=8)
    args = ap.parse_args(argv)
    if not args.out.endswith(".jsonl"):
        ap.error("--out must end in .jsonl")
    man_path = args.out[:-6] + ".manifest.json"
    for p in (args.out, man_path):
        if os.path.exists(p):
            ap.error("%s exists; refusing to overwrite" % p)
    rman_path = args.retrieval_cache[:-6] + ".manifest.json"
    with open(rman_path, encoding="utf-8") as fh:
        rman = json.load(fh)
    if rman.get("protocol") != "edir_retrieval" or not rman.get("complete") \
            or rman.get("cache_sha256") != prov.sha256_file(args.retrieval_cache):
        ap.error("%s is not a complete, verified edir_retrieval cache" % args.retrieval_cache)
    # Same checkpoint and config as the retrieval cache, checked BEFORE any GPU work:
    # a different config would change preprocessing and break the cross-check anyway.
    ckpt_sha, cfg_sha = prov.sha256_file(args.checkpoint), prov.sha256_file(args.config)
    if (ckpt_sha, cfg_sha) != (rman.get("checkpoint_sha256"), rman.get("config_sha256")):
        ap.error("the retrieval cache was made with another checkpoint or config")

    import PIL
    import torch
    import torch.nn.functional as F
    import torchvision
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import EDIRDataset
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    ds = EDIRDataset(root=args.edir_root, split="test", transform=transform, meta_dir=args.meta)
    gallery = list(ds.corpus_paths)
    if (len(ds), len(gallery)) != (EXPECTED["queries"], EXPECTED["corpus"]):
        ap.error("EDIR has %d queries and %d corpus images, expected %s" % (len(ds), len(gallery), EXPECTED))
    if ext.gallery_digest(gallery) != rman["gallery_sha256"] or rman["n_queries"] != len(ds):
        ap.error("the retrieval cache was made on another gallery or query set")
    gidx = {p: i for i, p in enumerate(gallery)}
    plan = annotated_rows(ds, gidx, os.path.join(ds.root, "imgs_mc"))     # ds.root: absolute, as the gallery
    retrieval = load_cache(args.retrieval_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[annotated] %s | %d queries | %d candidates | %s"
          % (PROTOCOL, len(ds), sum(len(p["rows"]) for p in plan), device), flush=True)

    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    if model.rv is None or model.aep is None:
        ap.error("the checkpoint must include AEP and RV")
    tokenizer = build_tokenizer(cfg.backbone)
    records = []
    with torch.no_grad():
        for s in range(0, len(ds), args.query_batch):
            idx = list(range(s, min(s + args.query_batch, len(ds))))
            ref = torch.stack([ext.ImageFiles([ds.triplets[i]["ref_path"]], transform)[0] for i in idx]).to(device)
            tok = tokenizer([ds.triplets[i]["text"] for i in idx])
            q = model.encode_query(ref, tok["input_ids"].to(device), tok["attention_mask"].to(device))
            inv = model.aep(q["text_out"].pooled, q["ref_out"].pooled)["invariant_embed"]
            owner = [b for b, i in enumerate(idx) for _ in plan[i]["rows"]]
            cand = [c for i in idx for c in plan[i]["rows"]]
            pix = torch.stack([ext.ImageFiles([gallery[c]], transform)[0] for c in cand]).to(device)
            c_out = model.backbone.encode_image(pix)
            c_pool = F.normalize(c_out.pooled, dim=-1)
            own = torch.tensor(owner, device=device)
            rv = model.rv(candidate_tokens=c_out.tokens, candidate_pooled=c_pool,
                          forward_text_pooled=q["text_out"].pooled[own], invariant_embed=inv[own])
            base = (q["query"][own] * c_pool).sum(dim=-1)
            vals = {"base_score": base, **{f: rv[f] for f in FIELDS if f != "base_score"}}
            vals = {f: v.float().cpu().tolist() for f, v in vals.items()}
            start = 0
            for i in idx:
                n = len(plan[i]["rows"])
                records.append({"protocol": PROTOCOL, "query_id": i, "target_gallery_index": plan[i]["target_row"],
                                "candidate_gallery_ids": plan[i]["rows"], "target_pos": plan[i]["target_pos"],
                                **{f: vals[f][start:start + n] for f in FIELDS}})
                start += n
            if (s // args.query_batch) % 64 == 0:
                print("[annotated] %d/%d queries, %.0fs" % (idx[-1] + 1, len(ds), time.time() - t0), flush=True)

    check = cross_check(records, retrieval)
    print("[annotated] cross-check with edir_retrieval: %d shared pairs, worst %s"
          % (check["shared_pairs"], check["worst"]), flush=True)
    if check["shared_pairs"] == 0 or check["violations"]:
        print("FATAL: annotated scores do not reproduce the retrieval cache (%s); nothing written"
              % (check["violations"] or "no shared pairs"), file=sys.stderr)
        return 1
    with open(args.out, "x", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    manifest = {"schema": SCHEMA, "protocol": PROTOCOL, "n_queries": len(records),
                "n_candidates": sum(len(r["candidate_gallery_ids"]) for r in records),
                "candidate_order": "ascending corpus row", "cross_check": check, "cross_tolerance": CROSS_TOL,
                "cache_sha256": prov.sha256_file(args.out), "retrieval_cache_sha256": rman["cache_sha256"],
                "gallery_sha256": rman["gallery_sha256"], "checkpoint_sha256": ckpt_sha,
                "config_sha256": cfg_sha, "query_batch": args.query_batch,
                "edir_revision": EDIR_REVISION,
                "edir_meta_sha256": {f: prov.sha256_file(os.path.join(args.meta, f)) for f in META_FILES},
                "source_sha256": prov.source_hashes(("edir_annotated_cache.py", "edir_cache.py")),
                "runtime": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                            "PIL": PIL.__version__,
                            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
                "wall_seconds": round(time.time() - t0, 1)}
    with open(man_path, "x", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("[annotated] wrote %s and its manifest in %.0fs" % (args.out, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
