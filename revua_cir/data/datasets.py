"""
Dataset adapters for the five corpora in the paper framework's dataset plan
(Section 5): FashionIQ, CIRR, CIRCO, LaSCo (train fallback), GeneCIS
(robustness ablation).

Each adapter expects the *official* annotation JSON released by that
dataset's authors, pointed at via `DataConfig.root`. Exact field names have
drifted slightly across dataset releases/versions over the years, so every
loader documents the schema it expects at the top of its `_load_annotations`
method and raises a clear error naming the missing field if your local copy
differs -- that is a five-minute fix (rename a key or adjust one line),
not a redesign.

Every adapter yields the same triplet dict, so `models/revua_cir.py` and
`train.py` never need to know which dataset produced a batch:
    {
        "reference_pixel_values": FloatTensor(3, H, W),
        "target_pixel_values":    FloatTensor(3, H, W),
        "edit_text": str,
        "attribute_targets": FloatTensor(vocab_size) or None,
        "meta": {"dataset": str, "id": Any, ...}
    }
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from revua_cir.data.transforms import safe_load_transform
from revua_cir.models.attribute_parser import RuleBasedAttributeExtractor, hash_attributes_to_vocab


class CIRTripletDatasetBase(Dataset):
    """Shared machinery: image loading, label-noise injection (Ablation #6),
    and weak-supervision attribute targets for the Attribute Edit Parser."""

    def __init__(self, transform, include_attribute_targets: bool = False, attribute_vocab_size: int = 512,
                 max_attributes: int = 8, noise_injection_rate: float = 0.0, seed: int = 42,
                 attribute_extractor=None):
        self.transform = transform
        self.include_attribute_targets = include_attribute_targets
        self.attribute_vocab_size = attribute_vocab_size
        self.max_attributes = max_attributes
        self.noise_injection_rate = noise_injection_rate
        self._rng = random.Random(seed)
        # `attribute_extractor` (fixed 2026-09-08): previously this always
        # hardcoded RuleBasedAttributeExtractor regardless of
        # cfg.attribute_parser.parser_backend -- "llm"/"hybrid" in a config
        # silently had no effect on the actual training data pipeline. Now
        # accepts any pre-built extractor (train.py constructs the right one
        # via attribute_parser.build_attribute_extractor(cfg.attribute_parser,
        # ...) and passes it through); defaults to the same rule-based
        # extractor as before when not given, so every existing call site
        # (including this project's own test fixtures) keeps working
        # unchanged.
        self._attr_extractor = (
            attribute_extractor if attribute_extractor is not None
            else (RuleBasedAttributeExtractor(use_spacy=False) if include_attribute_targets else None)
        )
        self.triplets: List[dict] = []  # populated by subclasses: {"ref_path", "target_path", "text", "meta"}

    def __len__(self):
        return len(self.triplets)

    def _maybe_inject_noise(self, idx: int, target_path: str) -> str:
        if self.noise_injection_rate <= 0 or self._rng.random() >= self.noise_injection_rate:
            return target_path
        # Ablation #6: swap the target for a random other sample's target -> simulates a noisy/weak label
        other = self._rng.randrange(len(self.triplets))
        return self.triplets[other]["target_path"]

    def __getitem__(self, idx: int) -> dict:
        item = self.triplets[idx]
        target_path = self._maybe_inject_noise(idx, item["target_path"])

        ref_tensor = safe_load_transform(item["ref_path"], self.transform)
        tgt_tensor = safe_load_transform(target_path, self.transform)

        out = {
            "reference_pixel_values": ref_tensor,
            "target_pixel_values": tgt_tensor,
            "edit_text": item["text"],
            "meta": item.get("meta", {}),
        }
        if self.include_attribute_targets:
            triples = self._attr_extractor.extract(item["text"])
            out["attribute_targets"] = hash_attributes_to_vocab(triples, self.attribute_vocab_size, self.max_attributes)
        else:
            out["attribute_targets"] = None
        return out


class FashionIQDataset(CIRTripletDatasetBase):
    """Expects the official layout from github.com/XiaoxiaoGuo/fashion-iq:
        {root}/captions/cap.{category}.{split}.json
            -> list of {"target": id, "candidate": id, "captions": [str, str]}
        {root}/images/{image_id}.jpg   (or .png -- both are probed)
        categories: dress, shirt, toptee
    NOTE: FashionIQ ships image *URLs*, not raw files (Section 9 risk register
    -- link rot). This adapter assumes you have already resolved/downloaded
    images into {root}/images/.

    Annotation and image locations are resolved independently (mirroring
    CIRRDataset), because third-party mirrors nest each differently and not
    always the same way. A real Kaggle upload (`binhhuunguyen/fashioniq`) hit
    both at once: captions/image_splits sit under a doubled repo-name path
    (`fashion-iq-master/fashion-iq-master/captions/...` -- a GitHub
    "Download ZIP" artifact from zipping an already-extracted repo), while
    images sit under `resized_images/resized_images/` with no per-category
    subfolder and no extension probing needed (`.jpg` only, confirmed against
    a real record).
    """

    CATEGORIES = ("dress", "shirt", "toptee")
    _ANNOTATION_PREFIXES = (
        "",
        os.path.join("fashion-iq-master", "fashion-iq-master"),
    )
    _IMAGE_SUBDIRS = (
        "images",
        os.path.join("resized_images", "resized_images"),
    )

    def __init__(self, root: str, split: str, transform, categories: Optional[List[str]] = None, **kwargs):
        super().__init__(transform, **kwargs)
        self.root = root
        self.ann_root = self._resolve_ann_root(root, split, categories or self.CATEGORIES)
        self.image_root = self._resolve_image_root(root)
        categories = categories or list(self.CATEGORIES)
        for cat in categories:
            ann_path = os.path.join(self.ann_root, "captions", f"cap.{cat}.{split}.json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path, encoding="utf-8") as f:
                entries = json.load(f)
            for e in entries:
                text = " and ".join(e.get("captions", [])) or e.get("caption", "")
                self.triplets.append({
                    "ref_path": self._img_path(e["candidate"]),
                    "target_path": self._img_path(e["target"]),
                    "text": text,
                    "meta": {"dataset": "fashioniq", "category": cat, "target_id": e["target"]},
                })

    @classmethod
    def _resolve_ann_root(cls, root: str, split: str, categories) -> str:
        """Captions live at {root}/ or, on some mirrors, nested under a
        doubled repo-name path (see class docstring)."""
        probe_cat = categories[0] if categories else cls.CATEGORIES[0]
        for prefix in cls._ANNOTATION_PREFIXES:
            candidate = os.path.join(root, prefix) if prefix else root
            if os.path.exists(os.path.join(candidate, "captions", f"cap.{probe_cat}.{split}.json")):
                return candidate
        return root  # let the open() below raise with the canonical path

    @classmethod
    def _resolve_image_root(cls, root: str) -> str:
        """Where category-less image files live. Independent of ann_root:
        a mirror can nest annotations under a repo-name path while images
        sit in a differently-named top-level folder entirely."""
        for sub in cls._IMAGE_SUBDIRS:
            if os.path.isdir(os.path.join(root, sub)):
                return os.path.join(root, sub)
        return os.path.join(root, "images")  # canonical path; probed per-file below anyway

    def _img_path(self, image_id: str) -> str:
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(self.image_root, f"{image_id}{ext}")
            if os.path.exists(p):
                return p
        return os.path.join(self.image_root, f"{image_id}.jpg")  # let safe_load_transform fall back to gray


class CIRRDataset(CIRTripletDatasetBase):
    """Expects the official layout from github.com/Cuberick-Orion/CIRR:
        {root}/captions/cap.rc2.{split}.json
            -> list of {"reference": id, "target_hard": id, "caption": str,
                        "img_set": {"members": [id, ...]}, "pairid": int}
        {root}/image_splits/split.rc2.{split}.json -> {image_id: relative_path}
        Images ultimately come from NLVR2 -- Section 9 flags a real access
        risk on that image set; see the paper framework's LaSCo fallback.

    Deviations tolerated because real CIRR checkouts/mirrors routinely have
    them, and the two axes (where annotations live, where images live) vary
    *independently* -- they are resolved separately, not derived from each
    other:

    * The annotation dirs may sit one level down in a cloned `CIRR_repo/`
      (the images are distributed separately from the annotation repo, so
      people end up with the repo nested inside the data dir), or nested as
      `CIRR/cirr/` (a common shape for third-party Kaggle mirrors of the
      official repo, which itself contains a `cirr/` subfolder).
    * The split map stores paths like `./dev/dev-244-0-img0.png`. Local
      checkouts of the official image release unpack that to `dev_img/`,
      `train_img/`, `test1_img/` (suffixed, flat for dev/test1, 100 numbered
      subdirs for train); some mirrors (e.g. a Kaggle re-upload) instead
      preserve the map's own `dev/`, `train/N/`, `test1/` naming one level
      under a `CIRR/` parent. We try the map's relative path as written
      first, against each candidate image root in turn, then retry with the
      `_img` suffix -- covering both shapes without hardcoding either.
    """

    def __init__(self, root: str, split: str, transform, **kwargs):
        super().__init__(transform, **kwargs)
        self.root = root
        self.ann_root = self._resolve_ann_root(root, split)
        self.image_root = self._resolve_image_root(root)

        split_map_path = os.path.join(self.ann_root, "image_splits", f"split.rc2.{split}.json")
        with open(split_map_path, encoding="utf-8") as f:
            self.id_to_path: Dict[str, str] = json.load(f)

        ann_path = os.path.join(self.ann_root, "captions", f"cap.rc2.{split}.json")
        with open(ann_path, encoding="utf-8") as f:
            entries = json.load(f)

        self.group_members: Dict[int, List[str]] = {}
        for e in entries:
            ref_id, tgt_id = e["reference"], e["target_hard"]
            self.triplets.append({
                "ref_path": self._img_path(ref_id),
                "target_path": self._img_path(tgt_id),
                "text": e["caption"],
                "meta": {"dataset": "cirr", "pairid": e.get("pairid"), "target_id": tgt_id,
                          "reference_id": ref_id,
                          # Official CIRR semantics (fixed 2026-09-09,
                          # code-audit finding #9): `img_set.members` is the
                          # full 6-image visually-similar group INCLUDING the
                          # reference image itself -- the actual subset
                          # protocol ranks only the 5 NON-reference members
                          # against the query (a query cannot correctly
                          # retrieve its own reference as "the edited
                          # target"). `reference_id` is threaded through
                          # separately so evaluate.py can compute that
                          # 5-candidate subset explicitly rather than relying
                          # on the reference happening to score low.
                          "group_members": e.get("img_set", {}).get("members", [])},
            })

    @staticmethod
    def _resolve_ann_root(root: str, split: str) -> str:
        """Annotations live at {root}/, or nested one level down as either
        {root}/CIRR_repo/ or {root}/CIRR/cirr/ (see class docstring)."""
        for candidate in (root, os.path.join(root, "CIRR_repo"), os.path.join(root, "CIRR", "cirr")):
            if os.path.exists(os.path.join(candidate, "captions", f"cap.rc2.{split}.json")):
                return candidate
        return root  # let the open() below raise with the canonical path

    @staticmethod
    def _resolve_image_root(root: str) -> str:
        """Where the split map's relative paths (`./dev/x.png` etc.) should be
        joined against. Independent of `_resolve_ann_root`: a mirror can nest
        annotations under `CIRR/cirr/` while images sit one level up at
        `CIRR/dev/...`, not alongside the annotations. Picks the first
        candidate that actually contains a `dev/` or `dev_img/` directory
        (dev is always present; test1/train are not universally, e.g. in a
        val-only download) rather than assuming a fixed relationship to
        ann_root.

        Added `img_raw/` and `CIRR/img_raw/` candidates (fixed 2026-09-09,
        code-audit finding #8): the official CIRR repo
        (github.com/Cuberick-Orion/CIRR) documents its own canonical release
        layout as `data/cirr/img_raw/{train,dev,test1}/`, which neither of
        the two candidates already here would resolve -- a user who follows
        the official repo's own instructions exactly (rather than one of the
        third-party mirror shapes this method already handles) would
        silently get every image replaced by safe_load_transform's gray
        placeholder, with no error. Checked in this order: bare `root`/
        `root/CIRR` first (unchanged priority, matches the two shapes
        already confirmed against real Kaggle mirrors), `img_raw` variants
        last, since a directory named literally `img_raw` is unambiguous and
        cannot collide with the other candidates' meaning.
        """
        for candidate in (root, os.path.join(root, "CIRR"),
                          os.path.join(root, "img_raw"), os.path.join(root, "CIRR", "img_raw")):
            if os.path.isdir(os.path.join(candidate, "dev")) or os.path.isdir(os.path.join(candidate, "dev_img")):
                return candidate
        return root

    def _img_path(self, image_id: str) -> str:
        rel = self.id_to_path.get(image_id, f"{image_id}.png")
        direct = os.path.normpath(os.path.join(self.image_root, rel))
        if os.path.exists(direct):
            return direct
        # The map says `./dev/x.png` / `./train/63/x.png`; some local checkouts
        # of the official image release unpack that to `dev_img/` (flat) and
        # `train_img/` (100 numbered subdirs) instead of the map's own naming.
        # Suffix the *leading* split dir and keep any nesting beneath it.
        parts = [p for p in os.path.normpath(rel).split(os.sep) if p not in (".", "")]
        if parts:
            suffixed = os.path.join(self.image_root, parts[0] + "_img", *parts[1:])
            if os.path.exists(suffixed):
                return suffixed
        return direct  # gray placeholder; verify_data.py reports the rate


class CIRCODataset(CIRTripletDatasetBase):
    """Expects the official layout from github.com/miccunifi/CIRCO:
        {root}/annotations/{split}.json
            -> list of {"id": int, "reference_img_id": id, "target_img_ids": [id, ...],
                        "relative_caption": str, "shared_concept": str}
        {root}/COCO2017_unlabeled/{filename}  (COCO id -> filename via id2filename map,
                                                 typically {id:012d}.jpg)
    CIRCO's defining feature -- multiple ground truths per query -- is why
    `meta["target_ids"]` (plural) is populated here; evaluate.py's mAP@K uses it.
    """

    def __init__(self, root: str, split: str, transform, **kwargs):
        super().__init__(transform, **kwargs)
        self.root = root
        ann_path = os.path.join(root, "annotations", f"{split}.json")
        with open(ann_path) as f:
            entries = json.load(f)
        for e in entries:
            target_ids = e.get("target_img_ids") or e.get("gt_img_ids") or [e.get("target_img_id")]
            primary_target = target_ids[0]
            self.triplets.append({
                "ref_path": self._img_path(e["reference_img_id"]),
                "target_path": self._img_path(primary_target),
                "text": e.get("relative_caption") or e.get("caption", ""),
                "meta": {"dataset": "circo", "query_id": e["id"], "target_ids": target_ids},
            })

    # The COCO2017 unlabeled archive unpacks to a nested `unlabeled2017/` dir,
    # so images usually sit at COCO2017_unlabeled/unlabeled2017/{id:012d}.jpg.
    _IMAGE_SUBDIRS = (os.path.join("COCO2017_unlabeled", "unlabeled2017"), "COCO2017_unlabeled")

    def _img_path(self, image_id) -> str:
        fname = f"{int(image_id):012d}.jpg" if isinstance(image_id, int) or str(image_id).isdigit() else str(image_id)
        for sub in self._IMAGE_SUBDIRS:
            p = os.path.join(self.root, sub, fname)
            if os.path.exists(p):
                return p
        return os.path.join(self.root, self._IMAGE_SUBDIRS[0], fname)


class LaScoDataset(CIRTripletDatasetBase):
    """Expects the official layout from github.com/levymsn/LaSCo:
        {root}/{split}.json -> list of {"query-image": [id, path], "target-image": [id, path], "query-text": str}
    Used as the large-scale open-domain TRAIN fallback if CIRR image access
    (NLVR2) is blocked -- see Section 5.1 / 9 of the paper framework.
    """

    def __init__(self, root: str, split: str, transform, **kwargs):
        super().__init__(transform, **kwargs)
        self.root = root
        ann_path = os.path.join(root, f"{split}.json")
        with open(ann_path) as f:
            entries = json.load(f)
        for e in entries:
            q_id, q_path = e["query-image"]
            t_id, t_path = e["target-image"]
            self.triplets.append({
                "ref_path": os.path.join(root, "images", q_path),
                "target_path": os.path.join(root, "images", t_path),
                "text": e.get("query-text", e.get("caption", "")),
                "meta": {"dataset": "lasco", "target_id": t_id},
            })


class GeneCISDataset(CIRTripletDatasetBase):
    """Expects the official layout from github.com/facebookresearch/genecis:
        {root}/{template}.json -> list of {"reference": id, "gallery": [id, ...],
                                            "target": id, "condition": str}
    Used only for the compositional/zero-shot robustness ablation
    (Section 8, Ablation #7 / Section 5.1) -- not a training split.
    """

    TEMPLATES = ("focus_attribute", "change_attribute", "focus_object", "change_object")

    def __init__(self, root: str, template: str, transform, image_dir: str = "images", **kwargs):
        super().__init__(transform, **kwargs)
        self.root = root
        ann_path = os.path.join(root, f"{template}.json")
        with open(ann_path) as f:
            entries = json.load(f)
        for e in entries:
            self.triplets.append({
                "ref_path": os.path.join(root, image_dir, str(e["reference"])),
                "target_path": os.path.join(root, image_dir, str(e["target"])),
                "text": e.get("condition", ""),
                "meta": {"dataset": "genecis", "template": template, "gallery": e.get("gallery", [])},
            })


def _read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class EDIRDataset(CIRTripletDatasetBase):
    """EDIR (Song et al., ACL 2026), the streamlined release registered in v3.1
    §15 (Hugging Face EDIR-BENCH/EDIR, revision 72581ea0...):
        {meta_dir}/queries.jsonl    {"id", "image" (reference), "text", "metadata": {"subcategory"}}
        {meta_dir}/corpus.jsonl     {"id", "image"}: the retrieval gallery (28,645 edited images)
        {meta_dir}/instances.jsonl  {"qid", "pos": [one image], "neg": [4 or 5 images]}
        {root}/imgs_mc/{image}      every reference and corpus image
    One split only ("test"). A query's target is its single annotated positive,
    a corpus image. Identities are namespaced, `corpus:<file>` and
    `ref:<file>`; no reference is a corpus image, and every annotated candidate
    is an edit of its own query's source (`<source>_<k>.png`), both checked.
    The annotated sets (Track A's protocol) are kept in `self.annotated`, apart
    from the retrieval triplets. Every image is checked to exist here, so the
    loader's gray-placeholder fallback cannot apply.
    """

    def __init__(self, root: str, split: str, transform, meta_dir: Optional[str] = None, **kwargs):
        super().__init__(transform, **kwargs)
        if split != "test":
            raise ValueError("EDIR has one split, 'test'; got %r" % split)
        # Absolute paths: every EDIR stage then names each image by the same
        # string however it was invoked, so gallery digests agree across stages.
        self.root = root = os.path.abspath(root)
        meta_dir = os.path.abspath(meta_dir or root)
        image_dir = os.path.join(root, "imgs_mc")
        queries = _read_jsonl(os.path.join(meta_dir, "queries.jsonl"))
        corpus = _read_jsonl(os.path.join(meta_dir, "corpus.jsonl"))
        instances = _read_jsonl(os.path.join(meta_dir, "instances.jsonl"))

        self.corpus_files: List[str] = [r["image"] for r in corpus]
        if len(set(self.corpus_files)) != len(self.corpus_files):
            raise ValueError("EDIR corpus lists an image twice")
        corpus_set = set(self.corpus_files)
        self.corpus_paths: List[str] = [os.path.join(image_dir, f) for f in self.corpus_files]
        self.corpus_ids: List[str] = ["corpus:" + f for f in self.corpus_files]

        inst = {r["qid"]: r for r in instances}
        qids = [q["id"] for q in queries]
        if len(inst) != len(instances) or len(set(qids)) != len(qids) or set(inst) != set(qids):
            raise ValueError("EDIR queries and instances do not correspond one to one")

        self.annotated: List[dict] = []
        for q in queries:
            r = inst[q["id"]]
            if len(r["pos"]) != 1:
                raise ValueError("EDIR query %s has %d positives, expected 1" % (q["id"], len(r["pos"])))
            pos, negs = r["pos"][0], list(r["neg"])
            if q["image"] in corpus_set:
                raise ValueError("EDIR reference %s is a corpus image" % q["image"])
            source = os.path.splitext(q["image"])[0]
            for c in [pos] + negs:
                if c not in corpus_set:
                    raise ValueError("EDIR candidate %s of query %s is not a corpus image" % (c, q["id"]))
                if not c.startswith(source + "_"):
                    raise ValueError("EDIR candidate %s is not an edit of source %s" % (c, source))
            if len(set([pos] + negs)) != len(negs) + 1:
                raise ValueError("EDIR query %s lists a candidate twice" % q["id"])
            self.triplets.append({
                "ref_path": os.path.join(image_dir, q["image"]),
                "target_path": os.path.join(image_dir, pos),
                "text": q["text"],
                "meta": {"dataset": "edir", "query_id": q["id"], "reference_id": "ref:" + q["image"],
                         "target_id": "corpus:" + pos, "source": source,
                         "subcategory": q.get("metadata", {}).get("subcategory")},
            })
            self.annotated.append({"pos": pos, "neg": negs})

        missing = sorted(p for p in set(self.corpus_paths) | {t["ref_path"] for t in self.triplets}
                         if not os.path.isfile(p))
        if missing:
            raise FileNotFoundError("%d EDIR images are missing, e.g. %s" % (len(missing), missing[0]))


DATASET_REGISTRY: Dict[str, Callable] = {
    "edir": EDIRDataset,
    "fashioniq": FashionIQDataset,
    "cirr": CIRRDataset,
    "circo": CIRCODataset,
    "lasco": LaScoDataset,
    "genecis": GeneCISDataset,
}


class ConcatTripletDataset(Dataset):
    """Concatenates several CIR datasets for joint training (Section 5.1: FashionIQ + CIRR/LaSCo)."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cum_sizes = []
        total = 0
        for d in datasets:
            total += len(d)
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def __getitem__(self, idx):
        for d, cum in zip(self.datasets, self.cum_sizes):
            if idx < cum:
                local_idx = idx - (cum - len(d))
                return d[local_idx]
        raise IndexError(idx)
