"""BBNV Phase B feature extraction (design v3 §5.2; v3.1 §10). One GPU pass.

The BBNV heads are small MLPs on FROZEN pooled features of the deployed
ReVUA-CIR checkpoint:
  image  x_hat = normalize(model.backbone.encode_image(x).pooled)
         (identical to model.encode_gallery(x)["pooled"], i.e. `c_pool` of
         the verifier-vector dumps)
  text   t_hat = normalize(model.backbone.encode_text(t).pooled)
         (the text encoder alone, never mixed with an image; `t_hat` of the
         verifier-vector dumps)

Targets (one run each, into a fresh --out-dir):
  cirr_train   every CIRR-train image of the validated partition manifest
               (images.jsonl order) and every training caption
               (triplets.jsonl order). Before any GPU work the manifest is
               semantically re-validated against the raw captions and split
               map and its images.jsonl must carry the PINNED digest of
               amendment §10.
  cirr_val     the full CIRR validation gallery (evaluation order) and every
               validation query's text.
  fiq_val      one FashionIQ category (--category): its official validation
               gallery and every validation query's text.
  edir_test    EDIR (v3.1 §19; --edir-root, --edir-meta): the 28,645-image
               corpus (corpus.jsonl order) as image_emb, every query's text,
               and the 4,784 reference images as a SEPARATE block ref_emb.
               EDIR references are not gallery images, so there is no
               ref_row: identities are explicit and namespaced (image_ids
               `corpus:<file>`, ref_ids `ref:<file>`, sorted) and each query
               carries ref_index into ref_ids (§19.1, condition 1).
For cirr_val and fiq_val the reference and target of each query are stored
as gallery rows; for edir_test only the target is (tgt_row). The existing
verifier-vector dump of the SAME checkpoint (--vectors) is validated in full
(manifest identity, shapes, dtypes, finite unit rows, candidate ids in range
and unique per query) before the model is built, and the new features must reproduce every query's t_hat and every
cached window candidate's c_pool: angular distance (1 - cosine) <= COS_TOL.
COS_TOL bounds the angle between two vectors (for unit vectors a Euclidean
distance of at most sqrt(2 COS_TOL) ~ 0.014); it is not a bound on any
downstream score. The largest coordinate difference is reported beside it.

Images are loaded strictly: a missing or unreadable image is an error (no
gray placeholder). Arrays are float32, L2-normalised. --device cuda (the
default) fails at once if this Python environment cannot use CUDA.

Usage
-----
  python scripts/bbnv_extract.py --config CFG --checkpoint CKPT --module-combo fc+aep+rv \\
      --target cirr_train --manifest-dir ../bbnv_runs/manifest_groups_cluster --out-dir ../bbnv_feats/cirr_train
  python scripts/bbnv_extract.py ... --target cirr_val --vectors ../vec/vec_cirr_val.npz --out-dir ../bbnv_feats/cirr_val
  python scripts/bbnv_extract.py ... --target fiq_val --category dress --vectors ../vec/vec_fiq_dress.npz --out-dir ...
  --probe            encode ONE batch of texts and ONE batch of images (for a
                     validation target: the cached candidates of the probed
                     queries, so the reproduction check always has coverage);
                     writes nothing
  --skip-if-complete exit 0 if --out-dir already holds a verified run of this
                     target, category and checkpoint (resume after a stop);
                     any other non-empty state is an error

Output: features.npz, then results.json (completion marker) written into
<out-dir>.staging, verified, then moved to <out-dir> (one writer per directory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (_REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bbnv_manifest as man  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
from dump_verifier_vectors import SCHEMA as VECTOR_SCHEMA  # noqa: E402

SCHEMA = "bbnv-features/2"
# Amendment §10: the approved CIRR-train allocation. Kept in code and in the
# design document, independently of any result bundle.
APPROVED_IMAGES_SHA256 = "65448562ccd1aa4ac48a3f8ab60535d95a0aea5c941fc6e1d711b20ff4b622e5"
# Acceptance threshold for reproducing a dumped unit vector: max angular
# distance (1 - cosine). The dumps' own scalar tolerances are 1e-4.
COS_TOL = 1e-4
UNIT_TOL = 1e-4                  # |norm - 1| allowed for any stored or new unit vector
TARGETS = ("cirr_train", "cirr_val", "fiq_val", "edir_test")
# The verifier-vector dump each validation target is checked against: (dataset, split).
DUMP_OF = {"cirr_val": ("cirr", "val"), "fiq_val": ("fashioniq", "val"), "edir_test": ("edir", "test")}
FIQ_CATEGORIES = ("dress", "shirt", "toptee")
FEATURES_NAME = "features.npz"
# Code the features depend on beyond the BBNV scripts (repository-relative).
DEPENDENCIES = ("revua_cir/models/backbone.py", "revua_cir/models/revua_cir.py", "revua_cir/utils.py",
                "revua_cir/config.py", "revua_cir/data/datasets.py", "revua_cir/data/transforms.py",
                "revua_cir/data/tokenizer.py", "scripts/evaluate_cirr.py", "scripts/fashioniq_cache.py",
                "scripts/dump_verifier_vectors.py")


class ExtractError(ValueError):
    pass


# ------------------------------------------------------------ pure checks --
def gallery_digest(paths: Sequence[str]) -> str:
    """The verifier-vector dumps' gallery digest: SHA-256 of the newline-joined paths."""
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()


def require_files(paths: Sequence[str]) -> None:
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ExtractError("%d of %d images are missing, e.g. %s" % (len(missing), len(paths), missing[0]))


def rows_of(paths: Sequence[str], gallery_index: Mapping[str, int]) -> np.ndarray:
    """Gallery row of every path; a path outside the gallery is an error."""
    out = [gallery_index.get(p) for p in paths]
    if any(r is None for r in out):
        raise ExtractError("a reference/target image is not in the gallery: %s"
                           % paths[out.index(None)])
    return np.asarray(out, dtype=np.int64)


def edir_identities(ds, gallery_index: Mapping[str, int]) -> Dict[str, np.ndarray]:
    """EDIR identities (§19.1, condition 1): image_ids `corpus:<file>` in
    gallery order, ref_ids `ref:<file>` sorted, each query's ref_index into
    ref_ids and tgt_row into the gallery. No reference may be a corpus file."""
    if [gallery_index[p] for p in ds.corpus_paths] != list(range(len(ds.corpus_paths))):
        raise ExtractError("the EDIR gallery is not the corpus in corpus.jsonl order")
    ref_ids = sorted({t["meta"]["reference_id"] for t in ds.triplets})
    if not all(r.startswith("ref:") for r in ref_ids) or {r[4:] for r in ref_ids} & set(ds.corpus_files):
        raise ExtractError("an EDIR reference id is malformed or names a corpus image")
    pos = {r: i for i, r in enumerate(ref_ids)}
    return {"tgt_row": rows_of([t["target_path"] for t in ds.triplets], gallery_index),
            "ref_index": np.asarray([pos[t["meta"]["reference_id"]] for t in ds.triplets], dtype=np.int64),
            "image_ids": np.asarray(ds.corpus_ids), "ref_ids": np.asarray(ref_ids)}


def check_train_alignment(triplet_rows: Sequence[Mapping], dataset_triplets: Sequence[Mapping],
                          image_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """The manifest's triplets must be the training dataset's triplets, in the
    same order, with the same pairid, images and caption text. Returns the
    pairids and the image rows of every reference and target."""
    if len(triplet_rows) != len(dataset_triplets):
        raise ExtractError("manifest has %d triplets, dataset %d" % (len(triplet_rows), len(dataset_triplets)))
    row = {img: i for i, img in enumerate(image_ids)}
    for t, d in zip(triplet_rows, dataset_triplets):
        meta = d["meta"]
        if (t["pairid"] != meta["pairid"] or t["reference"] != meta["reference_id"]
                or t["target"] != meta["target_id"] or t["caption"] != d["text"]):
            raise ExtractError("manifest pairid %r disagrees with the training dataset" % (t["pairid"],))
    return {"pairid": np.asarray([t["pairid"] for t in triplet_rows], dtype=np.int64),
            "ref_row": np.asarray([row[t["reference"]] for t in triplet_rows], dtype=np.int64),
            "tgt_row": np.asarray([row[t["target"]] for t in triplet_rows], dtype=np.int64)}


def check_unit_rows(name: str, a: np.ndarray, ndim: int) -> None:
    """Finite float32 array of the given rank whose last-axis rows are unit vectors."""
    if a.dtype != np.float32 or a.ndim != ndim or a.size == 0 or not np.all(np.isfinite(a)):
        raise ExtractError("%s must be a non-empty finite %d-D float32 array" % (name, ndim))
    if np.abs(np.linalg.norm(a.astype(np.float64), axis=-1) - 1.0).max() > UNIT_TOL:
        raise ExtractError("%s rows are not unit vectors" % name)


def validate_dump(vec: Mapping[str, np.ndarray], vman: Mapping, n_queries: int, n_gallery: int,
                  dataset: str, category: Optional[str], module_combo: str, split: str = "val") -> None:
    """The verifier-vector dump's full contract, checked before it is used:
    the manifest names this dataset/split/category/combination; query_id is
    0..N-1; t_hat (N, D) and c_pool (N, K, D) are finite unit float32 rows;
    candidate_ids (N, K) are genuine integers in [0, n_gallery), unique per query."""
    want = {"schema": VECTOR_SCHEMA, "dataset": dataset, "split": split, "category": category,
            "module_combo": module_combo, "n_queries": n_queries, "gallery_size": n_gallery}
    differ = sorted(k for k, v in want.items() if vman.get(k) != v)
    if differ:
        raise ExtractError("vector-dump manifest differs in %s" % differ)
    qid, cand, t_hat, c_pool = vec["query_id"], vec["candidate_ids"], vec["t_hat"], vec["c_pool"]
    if qid.shape != (n_queries,) or not np.array_equal(qid, np.arange(n_queries)):
        raise ExtractError("query_id is not 0..%d in order" % (n_queries - 1))
    if cand.dtype.kind not in "iu" or cand.ndim != 2 or cand.shape[0] != n_queries or cand.shape[1] < 1:
        raise ExtractError("candidate_ids must be a 2-D integer array with one row per query")
    n, k = cand.shape
    d = vman.get("dim")
    if t_hat.shape != (n, d) or c_pool.shape != (n, k, d) or vman.get("k") != k:
        raise ExtractError("t_hat %s / c_pool %s do not match N=%d, K=%d, D=%s"
                           % (t_hat.shape, c_pool.shape, n, k, d))
    if cand.min() < 0 or cand.max() >= n_gallery:
        raise ExtractError("a candidate id lies outside the gallery [0, %d)" % n_gallery)
    if any(len(set(row.tolist())) != k for row in cand):
        raise ExtractError("a query lists a candidate twice")
    check_unit_rows("t_hat", t_hat, 2)
    check_unit_rows("c_pool", c_pool, 3)


def probe_image_rows(candidate_ids: np.ndarray, n_queries: int, limit: int) -> List[int]:
    """Gallery rows a probe encodes: the distinct cached candidates of the
    first n_queries queries, in order of first appearance, at most `limit`.
    Every one of them is a compared occurrence, so coverage is guaranteed."""
    rows: List[int] = []
    seen = set()
    for c in candidate_ids[:n_queries].ravel().tolist():
        if c not in seen:
            seen.add(c)
            rows.append(c)
            if len(rows) == limit:
                break
    return rows


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    cos = np.einsum("nd,nd->n", a64, b64) / (np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1))
    return np.clip(cos, -1.0, 1.0)


def reproduction_errors(text_emb: np.ndarray, image_emb: np.ndarray, image_rows: Sequence[int],
                        t_hat: np.ndarray, c_pool: np.ndarray, candidate_ids: np.ndarray) -> Dict[str, object]:
    """New vs dumped vectors, in float64, as true cosines. text_emb covers the
    first len(text_emb) queries; image_emb[i] is gallery row image_rows[i].
    Every cached candidate occurrence of those queries whose gallery row was
    encoded is compared; at least one must be."""
    nq = len(text_emb)
    if len(image_emb) != len(image_rows) or len(set(image_rows)) != len(image_rows):
        raise ExtractError("image_rows must name each encoded image exactly once")
    where = {int(r): i for i, r in enumerate(image_rows)}
    occ = [(q, j, where[int(c)]) for q in range(nq) for j, c in enumerate(candidate_ids[q]) if int(c) in where]
    if not occ:
        raise ExtractError("no cached candidate of the encoded queries was encoded")
    q_idx, k_idx, e_idx = (np.asarray(v, dtype=np.int64) for v in zip(*occ))
    new_img, old_img = image_emb[e_idx], c_pool[q_idx, k_idx]
    t_cos, i_cos = _cosine(text_emb, t_hat[:nq]), _cosine(new_img, old_img)
    return {"text_max_1_minus_cos": float((1.0 - t_cos).max()),
            "image_max_1_minus_cos": float((1.0 - i_cos).max()),
            "text_max_abs_coordinate": float(np.abs(text_emb.astype(np.float64) - t_hat[:nq]).max()),
            "image_max_abs_coordinate": float(np.abs(new_img.astype(np.float64) - old_img).max()),
            "n_text_compared": int(nq), "n_image_occurrences": len(occ),
            "n_distinct_images": len(set(e_idx.tolist()))}


def check_reproduction(errs: Mapping[str, object]) -> None:
    """Finite errors, positive comparison counts, and both angles within COS_TOL."""
    angles = ("text_max_1_minus_cos", "image_max_1_minus_cos")
    vals = [errs[k] for k in angles + ("text_max_abs_coordinate", "image_max_abs_coordinate")]
    if not all(isinstance(v, float) and np.isfinite(v) for v in vals):
        raise ExtractError("non-finite reproduction error: %s" % dict(errs))
    if min(errs["n_text_compared"], errs["n_image_occurrences"], errs["n_distinct_images"]) < 1:
        raise ExtractError("the reproduction check compared nothing: %s" % dict(errs))
    bad = {k: errs[k] for k in angles if errs[k] > COS_TOL}
    if bad:
        raise ExtractError("features do not reproduce the verifier vectors: %s" % bad)


def check_manifest(manifest_dir: str, cap_path: str, split_path: str) -> Dict[str, object]:
    """The partition manifest must be semantically valid against THESE caption
    and split files, built from exactly them, and carry the approved
    allocation digest. Returns the manifest's image and triplet rows."""
    with open(cap_path, encoding="utf-8") as fh:
        captions_obj = json.load(fh)
    with open(split_path, encoding="utf-8") as fh:
        split_obj = json.load(fh)
    res = man.validate_written_manifest(manifest_dir, captions_obj, split_obj)
    if res["manifest"]["inputs_sha256"] != {"captions": prov.sha256_file(cap_path),
                                            "splits": prov.sha256_file(split_path)}:
        raise ExtractError("manifest was built from other caption/split files than this dataset's")
    images_sha = prov.sha256_file(os.path.join(manifest_dir, "images.jsonl"))
    if images_sha != APPROVED_IMAGES_SHA256:
        raise ExtractError("images.jsonl digest %s is not the approved allocation" % images_sha)
    return {"images": man.read_jsonl(os.path.join(manifest_dir, "images.jsonl")),
            "triplets": man.read_jsonl(os.path.join(manifest_dir, "triplets.jsonl")),
            "images_sha256": images_sha}


def completed_run(out_dir: str, target: str, category: Optional[str], checkpoint_sha256: str) -> bool:
    """True if out_dir holds a verified run of this target, category and
    checkpoint; False if it is absent or empty (and has no staging leftover).
    Any other state is an error, never silently reused or overwritten."""
    if os.path.exists(man.staging_dir(out_dir)):
        raise ExtractError("%s exists (an interrupted run); inspect and remove it" % man.staging_dir(out_dir))
    if not os.path.isdir(out_dir) or not os.listdir(out_dir):
        return False
    try:
        res = prov.verify_run(out_dir)
    except IOError as exc:
        raise ExtractError("%s is not a complete verified run: %s" % (out_dir, exc)) from None
    if (res.get("schema"), res.get("target"), res.get("category"),
            res.get("manifest", {}).get("checkpoint_sha256")) != (SCHEMA, target, category, checkpoint_sha256):
        raise ExtractError("%s holds a different run; refusing to reuse or overwrite it" % out_dir)
    return True


def publish_npz_run(out_dir: str, arrays: Mapping[str, np.ndarray], results: Mapping,
                    name: str = FEATURES_NAME) -> None:
    """Write <name> (an .npz) into <out_dir>.staging, re-load and compare it,
    write results.json last, verify, then move the directory to out_dir (which
    must be the empty directory reserved at start-up)."""
    staging = man.staging_dir(out_dir)
    prov.prepare_fresh_dir(staging)
    path = os.path.join(staging, name)
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as z:
        if sorted(z.files) != sorted(arrays) or any(
                z[k].dtype != arrays[k].dtype or not np.array_equal(z[k], arrays[k]) for k in arrays):
            raise ExtractError("re-loaded %s differs from what was written" % name)
    payload = dict(results)
    payload["output_files"] = {name: prov.sha256_file(path)}
    prov.atomic_write_json(os.path.join(staging, prov.RESULTS_NAME), payload)
    prov.verify_run(staging)
    os.rmdir(out_dir)
    os.replace(staging, out_dir)


# ---------------------------------------------------------- GPU encoding --
class ImageFiles:
    """Strict image dataset over an explicit path list: an unreadable or
    corrupt file raises (revua_cir.data.transforms.load_image catches
    nothing), so no gray placeholder can enter the features."""

    def __init__(self, paths: Sequence[str], transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from revua_cir.data.transforms import load_image
        return self.transform(load_image(self.paths[i]))


def encode_images(model, paths: Sequence[str], transform, batch_size: int, device, workers: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    out = []
    with torch.no_grad():
        for pix in DataLoader(ImageFiles(paths, transform), batch_size=batch_size, shuffle=False,
                              num_workers=workers):
            out.append(F.normalize(model.backbone.encode_image(pix.to(device)).pooled, dim=-1).float().cpu())
    return torch.cat(out).numpy()


def encode_texts(model, tokenizer, texts: Sequence[str], batch_size: int, device) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    texts = list(texts)
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            tok = tokenizer(texts[s:s + batch_size])
            pooled = model.backbone.encode_text(tok["input_ids"].to(device), tok["attention_mask"].to(device)).pooled
            out.append(F.normalize(pooled, dim=-1).float().cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------ main --
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--target", required=True, choices=TARGETS)
    ap.add_argument("--category", choices=FIQ_CATEGORIES, help="fiq_val only")
    ap.add_argument("--manifest-dir", help="cirr_train only: the validated partition manifest")
    ap.add_argument("--edir-root", help="edir_test only: folder holding imgs_mc/")
    ap.add_argument("--edir-meta", help="edir_test only: folder with queries/corpus/instances.jsonl")
    ap.add_argument("--vectors", help="validation targets: verifier-vector dump of the same checkpoint")
    ap.add_argument("--out-dir", help="fresh (new or empty) output directory; not used by --probe")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--probe", action="store_true", help="one batch of texts and images; writes nothing")
    ap.add_argument("--skip-if-complete", action="store_true",
                    help="exit 0 if --out-dir already holds a verified run of this target and checkpoint")
    args = ap.parse_args(argv)
    is_train = args.target == "cirr_train"
    if args.batch_size < 1 or args.workers < 0:
        ap.error("--batch-size must be positive and --workers non-negative")
    if (args.target == "fiq_val") != (args.category is not None):
        ap.error("--category is required for fiq_val and only for it")
    if is_train != (args.manifest_dir is not None):
        ap.error("--manifest-dir is required for cirr_train and only for it")
    if (args.target == "edir_test") != (args.edir_root is not None and args.edir_meta is not None):
        ap.error("--edir-root and --edir-meta are required for edir_test and only for it")
    if (not is_train) != (args.vectors is not None):
        ap.error("--vectors is required for the validation targets and only for them")
    if args.probe == (args.out_dir is not None):
        ap.error("give --out-dir for a real run, and none for --probe")
    if args.probe and args.skip_if_complete:
        ap.error("--skip-if-complete applies to real runs only")
    for flag, path in (("--config", args.config), ("--checkpoint", args.checkpoint), ("--vectors", args.vectors)):
        if path is not None and not os.path.isfile(path):
            ap.error("%s %s is not a file" % (flag, path))
    ckpt_sha = prov.sha256_file(args.checkpoint)
    if args.out_dir is not None:
        try:
            if args.skip_if_complete and completed_run(args.out_dir, args.target, args.category, ckpt_sha):
                print("[extract] %s is complete, verified and of this run; skipping" % args.out_dir, flush=True)
                return 0
            if os.path.exists(man.staging_dir(args.out_dir)):
                raise ExtractError("%s exists (an interrupted run); use a fresh --out-dir"
                                   % man.staging_dir(args.out_dir))
            prov.prepare_fresh_dir(args.out_dir)
        except (IOError, ExtractError) as exc:
            ap.error(str(exc))

    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda, but this Python environment cannot use CUDA")
    device = torch.device(args.device)
    from revua_cir.config import ReVUAConfig
    from revua_cir.data.datasets import DATASET_REGISTRY
    from revua_cir.data.tokenizer import build_tokenizer
    from revua_cir.data.transforms import build_image_transform
    from revua_cir.models.revua_cir import ReVUACIR
    from revua_cir.utils import load_checkpoint

    t0 = time.time()
    cfg = ReVUAConfig.from_yaml(args.config).apply_module_ablation(args.module_combo)
    transform = build_image_transform(cfg.data.image_size, train=False)
    info: Dict[str, object] = {"target": args.target, "category": args.category}
    ref_paths: List[str] = []                       # edir_test only: the separate reference block

    # ---- inputs and every CPU check, before the model is built ----
    if is_train:
        ds = DATASET_REGISTRY["cirr"](root=os.path.join(cfg.data.root, "cirr"), split="train", transform=transform)
        mf = check_manifest(args.manifest_dir, os.path.join(ds.ann_root, "captions", "cap.rc2.train.json"),
                            os.path.join(ds.ann_root, "image_splits", "split.rc2.train.json"))
        image_rows, trip_rows = mf["images"], mf["triplets"]
        image_ids = [r["image_id"] for r in image_rows]
        idx = check_train_alignment(trip_rows, ds.triplets, image_ids)
        paths = [ds._img_path(i) for i in image_ids]
        texts = [t["caption"] for t in trip_rows]
        extra = {"image_ids": np.asarray(image_ids), "partition": np.asarray([r["partition"] for r in image_rows]),
                 **idx}
        info.update(manifest_dir=os.path.abspath(args.manifest_dir), manifest_images_sha256=mf["images_sha256"],
                    manifest_results_sha256=prov.sha256_file(os.path.join(args.manifest_dir, prov.RESULTS_NAME)))
    else:
        if args.target == "cirr_val":
            from evaluate_cirr import build_cirr_full_gallery
            ds = DATASET_REGISTRY["cirr"](root=os.path.join(cfg.data.root, "cirr"), split="val", transform=transform)
            paths = build_cirr_full_gallery(ds)
        elif args.target == "edir_test":
            ds = DATASET_REGISTRY["edir"](root=args.edir_root, split="test", transform=transform,
                                          meta_dir=args.edir_meta)
            paths = list(ds.corpus_paths)
        else:
            from fashioniq_cache import official_gallery
            ds = DATASET_REGISTRY["fashioniq"](root=os.path.join(cfg.data.root, "fashioniq"), split="val",
                                               transform=transform, categories=[args.category])
            paths = official_gallery(ds, args.category)
        if len(set(paths)) != len(paths):
            raise ExtractError("the gallery lists an image twice")
        gidx = {p: i for i, p in enumerate(paths)}
        texts = [t["text"] for t in ds.triplets]
        if args.target == "edir_test":
            extra = edir_identities(ds, gidx)
            ref_path_of = {t["meta"]["reference_id"]: t["ref_path"] for t in ds.triplets}
            ref_paths = [ref_path_of[r] for r in extra["ref_ids"]]
        else:
            extra = {"ref_row": rows_of([t["ref_path"] for t in ds.triplets], gidx),
                     "tgt_row": rows_of([t["target_path"] for t in ds.triplets], gidx)}
        vman_path = args.vectors[:-4] + ".manifest.json"
        with open(vman_path, encoding="utf-8") as fh:
            vman = json.load(fh)
        if vman.get("npz_sha256") != prov.sha256_file(args.vectors):
            raise ExtractError("%s does not match its manifest" % args.vectors)
        if vman.get("checkpoint_sha256") != ckpt_sha:
            raise ExtractError("the vector dump was made with a different checkpoint")
        if vman.get("gallery_sha256") != gallery_digest(paths):
            raise ExtractError("the vector dump's gallery is not this gallery in this order")
        with np.load(args.vectors, allow_pickle=False) as z:
            vec = {k: z[k] for k in ("query_id", "candidate_ids", "t_hat", "c_pool")}
        validate_dump(vec, vman, len(texts), len(paths), DUMP_OF[args.target][0],
                      args.category, args.module_combo, DUMP_OF[args.target][1])
        info.update(vectors=os.path.abspath(args.vectors), vectors_sha256=vman["npz_sha256"],
                    vectors_manifest_sha256=prov.sha256_file(vman_path),
                    config_sha256_matches_dump=vman.get("config_sha256") == prov.sha256_file(args.config))
    require_files(list(paths) + ref_paths)
    print("[extract] %s%s | %d images | %d texts%s | inputs checked in %.0fs"
          % (args.target, "/" + args.category if args.category else "", len(paths), len(texts),
             " | %d references" % len(ref_paths) if ref_paths else "", time.time() - t0), flush=True)

    # ---- which rows to encode: everything, or one probe batch ----
    q_rows = list(range(len(texts)))
    g_rows = list(range(len(paths)))
    if args.probe:
        q_rows = q_rows[:args.batch_size]
        g_rows = (probe_image_rows(vec["candidate_ids"], len(q_rows), args.batch_size) if not is_train
                  else g_rows[:args.batch_size])

    # ---- model ----
    model = ReVUACIR(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    tokenizer = build_tokenizer(cfg.backbone)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_img = time.time()
    image_emb = encode_images(model, [paths[r] for r in g_rows], transform, args.batch_size, device, args.workers)
    t_img = time.time() - t_img
    t_txt = time.time()
    text_emb = encode_texts(model, tokenizer, [texts[q] for q in q_rows], args.batch_size, device)
    t_txt = time.time() - t_txt
    check_unit_rows("image_emb", image_emb, 2)
    check_unit_rows("text_emb", text_emb, 2)
    if ref_paths:
        r_rows = list(range(len(ref_paths)))[:args.batch_size] if args.probe else list(range(len(ref_paths)))
        t_ref = time.time()
        ref_emb = encode_images(model, [ref_paths[r] for r in r_rows], transform, args.batch_size, device,
                                args.workers)
        t_img += time.time() - t_ref
        check_unit_rows("ref_emb", ref_emb, 2)
    timing = {"image_seconds": round(t_img, 2), "text_seconds": round(t_txt, 2),
              "images_done": len(image_emb), "texts_done": len(text_emb),
              "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if device.type == "cuda" else None}
    if not is_train:
        errs = reproduction_errors(text_emb, image_emb, g_rows, vec["t_hat"], vec["c_pool"], vec["candidate_ids"])
        print("[extract] reproduction vs verifier vectors: %s" % errs, flush=True)
        check_reproduction(errs)
        info["reproduction"] = errs
    if args.probe:
        proj = (t_img / len(image_emb)) * len(paths) + (t_txt / len(text_emb)) * len(texts)
        print("[extract] PROBE %s | encoding-only estimate for the full target %.1f min (the first batch "
              "includes start-up; excludes checks and writing) | nothing written" % (timing, proj / 60), flush=True)
        return 0

    root = ds.root if ref_paths else os.path.normpath(cfg.data.root)
    arrays = {"image_emb": image_emb, "text_emb": text_emb, "texts": np.asarray(texts),
              "image_paths": np.asarray([os.path.relpath(p, root).replace(os.sep, "/") for p in paths]), **extra}
    if ref_paths:
        arrays.update(ref_emb=ref_emb,
                      ref_paths=np.asarray([os.path.relpath(p, root).replace(os.sep, "/") for p in ref_paths]))
        info["n_refs"] = len(ref_paths)
    results = {
        "schema": SCHEMA, **info, "n_images": len(paths), "n_texts": len(texts), "dim": int(image_emb.shape[1]),
        "dtype": "float32", "cos_tol": COS_TOL, "gallery_sha256": gallery_digest(paths), "timing": timing,
        "preprocessing": {"image_size": cfg.data.image_size, "transform": repr(transform),
                          "tokenizer": cfg.backbone.pretrained_name,
                          "tokenizer_max_len": getattr(tokenizer, "max_len", None)},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_extract.py", "args": vars(args),
                     "config_sha256": prov.sha256_file(args.config), "checkpoint_sha256": ckpt_sha,
                     "source_sha256": prov.source_hashes(),
                     "dependency_sha256": {p: prov.sha256_file(os.path.join(_REPO, p)) for p in DEPENDENCIES},
                     "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                 "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                                 "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                 "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                                 "float32_matmul_precision": torch.get_float32_matmul_precision(),
                                 "batch_size": args.batch_size}}}
    publish_npz_run(args.out_dir, arrays, results)
    print("[extract] wrote, re-loaded and verified %s in %.0fs | %s"
          % (args.out_dir, time.time() - t0, timing), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
