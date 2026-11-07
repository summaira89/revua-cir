"""BBNV Phase B partition manifest for CIRR train (v3.1 §4.4 as amended by §10, §8; design v3 §5.3).

Reads the CIRR train captions and the image-split map of the SAME release,
validates them, allocates every train image to a partition by linked group
(bbnv_partition), assigns every triplet to its images' partition, and checks
that every triplet of a training partition can draw N_NEGATIVES negatives from
the negative-bank partition after masking. The written manifest is then
re-validated SEMANTICALLY from the raw inputs by validate_written_manifest,
which shares no grouping or allocation code with bbnv_partition.

Triplet definition: (reference, target_hard, caption), the standard CIRR
training triplet. The training target is always target_hard.

Known positives (they link groups, and are masked from negatives):
  * every (reference, target_hard) pair;
  * every other target_soft entry with a value > 0 (1.0 match; 0.5 and 0.2
    partial matches). These extra soft positives are NOT additional training
    targets; they only ensure partial matches are never trained as negatives;
  * target_soft entries equal to the reference itself add no edge (the
    triplet's own images are masked anyway).
A -1.0 entry adds no positive edge. Masking is instruction-agnostic: an image
pair labelled -1.0 in one triplet is still masked if another annotation links
it positively (directly; the mask is the union of direct partners of either
image, see bbnv_partition.negatives_for).
CIRR train contains soft labels that contradict the hard target. Such triplets
are kept, as in the standard protocol, and listed in anomalies.jsonl.

Usage
-----
  python scripts/bbnv_manifest.py --captions <cap.rc2.train.json> --splits <split.rc2.train.json> --out-dir ../bbnv_runs/manifest_<date>

Output (fresh directory, ONE writer per directory: the empty-directory check
is not a lock):
  triplets.jsonl   one row per pairid, in source order: pairid, reference,
                   target, caption, partition
  images.jsonl     one row per split-map image, sorted: image_id, group, partition
  positives.jsonl  one row per image with a known positive: image_id, sorted partners
  anomalies.jsonl  one row per contradictory soft label: pairid, kind, label
  results.json     last, as the completion marker
Everything is first written to <out-dir>.staging and validated there; out-dir
receives the files only after validation has passed (publish_validated).
The allocation is a function of the annotations only; it is frozen and must
never be rebalanced after seeing development performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import deque
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_partition as part  # noqa: E402
import bbnv_provenance as prov  # noqa: E402

N_NEGATIVES = 1023                               # v3.1 §8: 1,023 per example, without replacement
SOFT_VALUES = (1.0, 0.5, 0.2, -1.0)              # the CIRR rc2 label set; anything else is an error
CAPTION_KEYS = ("pairid", "reference", "target_hard", "target_soft", "caption")
TRAINING_PARTITIONS = ("head", "dev_select", "dev_calibrate")
POOL_PARTITION = "negbank"
COMPANIONS = ("triplets.jsonl", "images.jsonl", "positives.jsonl", "anomalies.jsonl")


class ManifestError(ValueError):
    pass


def _label(val) -> str:
    return "%.1f" % float(val)


def load_split_map(obj) -> Set[str]:
    """The image-split map: {image_id: relative path}. Returns the id set."""
    if not isinstance(obj, dict) or not obj:
        raise ManifestError("split map must be a non-empty object {image_id: path}")
    for k, v in obj.items():
        part.canonical_id(k)
        if not isinstance(v, str) or not v:
            raise ManifestError("split map entry %r has no path" % (k,))
    return set(obj)


def parse_captions(rows, images: Set[str]
                   ) -> Tuple[List[Dict[str, object]], List[Tuple[str, str]], List[Dict[str, object]]]:
    """Validate caption rows. Returns (triplets, extra soft-positive pairs, anomaly rows).

    Every id (reference, hard target, soft entries) must be canonical and in
    the split map; pairids are unique integers; captions are non-empty
    strings; soft values come from SOFT_VALUES. Anomalies: the hard target
    not labelled 1.0 (kind 'hard_target_soft_label', label 'absent' or the
    value), and the reference itself labelled (kind 'reference_soft_label')."""
    if not isinstance(rows, list) or not rows:
        raise ManifestError("captions must be a non-empty list")
    triplets: List[Dict[str, object]] = []
    soft_pos: List[Tuple[str, str]] = []
    anomalies: List[Dict[str, object]] = []
    seen: Set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or any(k not in row for k in CAPTION_KEYS):
            raise ManifestError("caption row lacks one of %s: %r" % (CAPTION_KEYS, row))
        pid = row["pairid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid in seen:
            raise ManifestError("pairid %r is not a unique integer" % (pid,))
        seen.add(pid)
        ref, tgt = part.canonical_id(row["reference"]), part.canonical_id(row["target_hard"])
        if ref == tgt:
            raise ManifestError("pairid %d: reference equals target" % pid)
        cap = row["caption"]
        if not isinstance(cap, str) or not cap.strip():
            raise ManifestError("pairid %d: empty caption" % pid)
        soft = row["target_soft"]
        if not isinstance(soft, dict):
            raise ManifestError("pairid %d: target_soft is not an object" % pid)
        for img in (ref, tgt, *soft):
            if part.canonical_id(img) not in images:
                raise ManifestError("pairid %d: image %r is not in the split map" % (pid, img))
        for img, val in soft.items():
            if isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) not in SOFT_VALUES:
                raise ManifestError("pairid %d: soft value %r outside %s" % (pid, val, SOFT_VALUES))
            if img == ref:
                anomalies.append({"pairid": pid, "kind": "reference_soft_label", "label": _label(val)})
            elif val > 0 and img != tgt:
                soft_pos.append((ref, img))
        if tgt not in soft:
            anomalies.append({"pairid": pid, "kind": "hard_target_soft_label", "label": "absent"})
        elif soft[tgt] != 1.0:
            anomalies.append({"pairid": pid, "kind": "hard_target_soft_label", "label": _label(soft[tgt])})
        triplets.append({"pairid": pid, "reference": ref, "target": tgt, "caption": cap})
    return triplets, soft_pos, anomalies


def build_positives(triplets: Sequence[Mapping], soft_pos: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
    """Known positives: every triplet pair plus the extra soft positives, both directions."""
    pairs = list(triplets) + [{"reference": r, "target": t} for r, t in soft_pos]
    return part.known_positives(pairs)


def _min_eligible(kept: Mapping[str, Sequence[Mapping]], pool: Set[str],
                  positives: Mapping[str, Set[str]]) -> Dict[str, int]:
    """Fewest eligible pool negatives over the triplets of each training
    partition, with exactly the mask negatives_for applies. An empty training
    partition is an error."""
    out: Dict[str, int] = {}
    for name in TRAINING_PARTITIONS:
        if not kept[name]:
            raise ManifestError("training partition %s keeps no triplets" % name)
        out[name] = min(len(pool - ({t["reference"], t["target"]} | positives.get(t["reference"], set())
                                    | positives.get(t["target"], set())))
                        for t in kept[name])
    return out


def _anomaly_counts(anomalies: Sequence[Mapping]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {"hard_target_soft_label": {}, "reference_soft_label": {}}
    for a in anomalies:
        out[a["kind"]][a["label"]] = out[a["kind"]].get(a["label"], 0) + 1
    return {k: dict(sorted(v.items())) for k, v in out.items()}


def build(captions_obj, split_obj) -> Tuple[Dict[str, object], Dict[str, List[Dict[str, object]]]]:
    """Validated manifest. Returns (summary, companion rows by file name)."""
    images = load_split_map(split_obj)
    triplets, soft_pos, anomalies = parse_captions(captions_obj, images)
    positives = build_positives(triplets, soft_pos)
    image_part = part.allocate_images(triplets, positives, images)
    kept, counts = part.split_triplets(triplets, image_part)
    group = part.linked_groups(positives, images)
    pool = {img for img, p in image_part.items() if p == POOL_PARTITION}
    worst = _min_eligible(kept, pool, positives)
    short = {n: w for n, w in worst.items() if w < N_NEGATIVES}
    if short:
        raise ManifestError("negative pool too small after masking: %s < %d" % (short, N_NEGATIVES))
    group_trips: Dict[str, int] = {}
    for t in triplets:
        group_trips[group[t["reference"]]] = group_trips.get(group[t["reference"]], 0) + 1
    per_part = {n: sorted((c for g, c in group_trips.items() if image_part[g] == n), reverse=True)
                for n in part.PARTITION_NAMES}
    summary = {
        "allocation": "linked groups, greedy to target triplet shares (amendment §10); frozen",
        "target_triplet_shares_percent": {n: hi - lo for n, lo, hi in part.PARTITIONS},
        "triplet_counts": counts,
        "triplet_shares_percent": {n: round(100.0 * counts["kept_" + n] / counts["total"], 2)
                                   for n in part.PARTITION_NAMES},
        "linked_groups": {
            "with_triplets": len(group_trips),
            "without_triplets": len(set(group.values())) - len(group_trips),
            "per_partition": {n: {"groups_with_triplets": len(v), "largest_by_triplets": v[:3]}
                              for n, v in per_part.items()}},
        "images_allocated": {n: sum(p == n for p in image_part.values()) for n in part.PARTITION_NAMES},
        "images_used_by_triplets": {n: len(v) for n, v in part.images_by_partition(kept).items()},
        "negative_pool": {"partition": POOL_PARTITION, "size": len(pool), "n_negatives": N_NEGATIVES,
                          "min_eligible_after_masking": worst},
        "positives": {"hard_pairs": len(triplets), "extra_soft_pair_entries_value_gt_0": len(soft_pos),
                      "images_with_known_positives": len(positives)},
        "annotation_anomalies_kept": _anomaly_counts(anomalies),
        "policy": "training target = target_hard; positive edges = hard pairs + other target_soft > 0, "
                  "both directions; they define the linked groups and are masked from negatives "
                  "(union of direct partners, instruction-agnostic); -1.0 adds no edge",
    }
    companions = {
        "triplets.jsonl": [dict(t, partition=image_part[t["reference"]]) for t in triplets],
        "images.jsonl": [{"image_id": img, "group": group[img], "partition": image_part[img]}
                         for img in sorted(images)],
        "positives.jsonl": [{"image_id": img, "partners": sorted(positives[img])} for img in sorted(positives)],
        "anomalies.jsonl": anomalies,
    }
    return summary, companions


def read_jsonl(path: str) -> List[Dict[str, object]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def validate_written_manifest(out_dir: str, captions_obj, split_obj) -> Dict[str, object]:
    """Semantic check of a WRITTEN manifest against the raw inputs. Independent
    of bbnv_partition's grouping and allocation code: it rebuilds the positive
    edges from the raw rows, finds components by breadth-first search, and
    recomputes the bucket rule with hashlib. Raises ManifestError on the first
    disagreement; returns the validated results.json."""
    def need(ok: bool, msg: str) -> None:
        if not ok:
            raise ManifestError("written manifest invalid: " + msg)

    res = prov.verify_run(out_dir)                               # integrity first
    need(sorted(res["output_files"]) == sorted(COMPANIONS), "companion set %s" % sorted(res["output_files"]))
    rows = {n: read_jsonl(os.path.join(out_dir, n)) for n in COMPANIONS}
    names = {p[0] for p in part.PARTITIONS}
    # images: exactly one row per split-map image, valid partition
    img_rows = rows["images.jsonl"]
    need([r["image_id"] for r in img_rows] == sorted(split_obj), "image rows differ from the split map")
    pof = {r["image_id"]: r["partition"] for r in img_rows}
    gof = {r["image_id"]: r["group"] for r in img_rows}
    need(all(p in names for p in pof.values()), "unknown partition name")
    # triplets: source order, ids, captions, and partition agreement
    trip = rows["triplets.jsonl"]
    need(len(trip) == len(captions_obj), "triplet count differs from the captions")
    for t, src in zip(trip, captions_obj):
        need(t["pairid"] == src["pairid"] and t["reference"] == src["reference"]
             and t["target"] == src["target_hard"] and t["caption"] == src["caption"],
             "pairid %r does not match its source row" % (src["pairid"],))
        need(t["partition"] == pof[t["reference"]] == pof[t["target"]], "pairid %r crosses" % (t["pairid"],))
    # positive edges rebuilt from the raw rows
    adj: Dict[str, Set[str]] = {}
    for src in captions_obj:
        ref = src["reference"]
        for other in [src["target_hard"]] + [k for k, v in src["target_soft"].items() if v > 0 and k != ref]:
            adj.setdefault(ref, set()).add(other)
            adj.setdefault(other, set()).add(ref)
    need({r["image_id"]: set(r["partners"]) for r in rows["positives.jsonl"]} == adj, "positives.jsonl")
    # components by BFS: group key = smallest id; one partition per component
    seen: Set[str] = set()
    trip_images = {s["reference"] for s in captions_obj}
    for start in sorted(split_obj):
        if start in seen:
            continue
        comp, queue = {start}, deque([start])
        while queue:
            for nb in adj.get(queue.popleft(), ()):
                if nb not in comp:
                    comp.add(nb)
                    queue.append(nb)
        seen |= comp
        key = min(comp)
        need(all(gof[i] == key for i in comp), "group key of the component of %r" % key)
        need(len({pof[i] for i in comp}) == 1, "component of %r spans partitions" % key)
        if not comp & trip_images:                               # no triplet: the per-image bucket rule
            b = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
            need(any(lo <= b < hi and pof[key] == n for n, lo, hi in part.PARTITIONS),
                 "bucket rule for %r" % key)
    # recomputed counts and pool eligibility agree with the summary
    counts = {"kept_" + n: sum(t["partition"] == n for t in trip) for n in names}
    counts["total"] = len(trip)
    need(counts == res["triplet_counts"], "triplet counts")
    pool = {i for i, p in pof.items() if p == POOL_PARTITION}
    need(len(pool) == res["negative_pool"]["size"], "pool size")
    worst = {n: min(len(pool - ({t["reference"], t["target"]} | adj[t["reference"]] | adj[t["target"]]))
                    for t in trip if t["partition"] == n) for n in TRAINING_PARTITIONS}
    need(worst == res["negative_pool"]["min_eligible_after_masking"], "min eligible negatives")
    need(all(w >= N_NEGATIVES for w in worst.values()), "negative pool too small")
    need(_anomaly_counts(rows["anomalies.jsonl"]) == res["annotation_anomalies_kept"], "anomaly counts")
    return res


def staging_dir(out_dir: str) -> str:
    return os.path.normpath(out_dir) + ".staging"


def publish_validated(out_dir: str, summary: Mapping, companions: Mapping, captions_obj, split_obj) -> None:
    """Write the run into <out_dir>.staging, validate it there (integrity and
    semantics), and only then move it to out_dir. A manifest that fails
    validation therefore never appears under out_dir with a results.json; the
    staging directory is left for inspection and is never a valid input."""
    staging = staging_dir(out_dir)
    prov.write_run(staging, summary, companions)
    validate_written_manifest(staging, captions_obj, split_obj)
    os.rmdir(out_dir)                                   # the empty directory reserved at start-up
    os.replace(staging, out_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--captions", required=True, help="cap.rc2.train.json")
    ap.add_argument("--splits", required=True, help="split.rc2.train.json of the same release")
    ap.add_argument("--out-dir", required=True, help="fresh (new or empty) output directory")
    args = ap.parse_args(argv)
    if os.path.exists(staging_dir(args.out_dir)):
        ap.error("%s exists (an earlier failed run); use a fresh --out-dir" % staging_dir(args.out_dir))
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    with open(args.captions, encoding="utf-8") as fh:
        captions_obj = json.load(fh)
    with open(args.splits, encoding="utf-8") as fh:
        split_obj = json.load(fh)
    summary, companions = build(captions_obj, split_obj)
    summary["manifest"] = {
        "schema": prov.MANIFEST_SCHEMA, "script": "bbnv_manifest.py", "args": vars(args),
        "inputs_sha256": {"captions": prov.sha256_file(args.captions),
                          "splits": prov.sha256_file(args.splits)},
        "source_sha256": prov.source_hashes(), "runtime": prov.runtime_info(None)}
    publish_validated(args.out_dir, summary, companions, captions_obj, split_obj)
    c = summary["triplet_counts"]
    print("[manifest] %d triplets: %s | pool %d, min eligible %s | wrote, verified and validated %s"
          % (c["total"], {n: c["kept_" + n] for n in part.PARTITION_NAMES},
             summary["negative_pool"]["size"], summary["negative_pool"]["min_eligible_after_masking"],
             args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
