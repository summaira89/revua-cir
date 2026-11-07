"""BBNV data partition and negative sampling (v3.1 §4.4 and §8, as amended by §10).

Partitions and their target shares of CIRR-train TRIPLETS:
    head           70%   fit the heads
    negbank        15%   training negative-bank partition (supplies negatives;
                         disclosed as training data, not untouched)
    dev_select      7%   epochs, kappa
    dev_calibrate   8%   calibration, integration weights, lambda, beta,
                         comparator choice

Amendment §10 (20 Oct 2026): images are allocated by LINKED GROUP, not one by
one. A linked group is a connected component of the graph whose edges are
every triplet's (reference, target) pair and every other known positive pair.
Whole groups are allocated greedily and deterministically: groups in order of
decreasing triplet count (ties by group key, the smallest image id), each to
the partition with the largest shortfall against its target share (ties in
PARTITION_NAMES order). Images in a group with no triplet keep the original
rule, sha256(image id) mod 100 against the percentage ranges below.
Consequences, each asserted: no triplet crosses partitions (none is
discarded), no known positive pair crosses partitions, and no image id occurs
in two partitions in either role. The per-image rule of v3.1 §4.4 discarded
47% of CIRR-train triplets and left about 160 triplets per dev partition.

Image ids must be canonical, non-empty strings at this boundary: hashing and
masking then use one representation (source audit, finding 10).
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

PARTITIONS: Tuple[Tuple[str, int, int], ...] = (
    ("head", 0, 70), ("negbank", 70, 85), ("dev_select", 85, 92), ("dev_calibrate", 92, 100))
PARTITION_NAMES = tuple(p[0] for p in PARTITIONS)


class PartitionError(ValueError):
    pass


def _check_ranges() -> None:
    edges = [(lo, hi) for _, lo, hi in PARTITIONS]
    if edges[0][0] != 0 or edges[-1][1] != 100 or any(
            a[1] != b[0] for a, b in zip(edges, edges[1:])) or any(lo >= hi for lo, hi in edges):
        raise PartitionError("partition ranges must tile 0..99 without gaps or overlaps")


_check_ranges()


def canonical_id(image_id) -> str:
    """Image ids are non-empty strings without surrounding whitespace; any
    other representation (int, bytes, None, padded string) is rejected."""
    if not isinstance(image_id, str) or not image_id or image_id != image_id.strip():
        raise PartitionError("image id %r is not a canonical non-empty string" % (image_id,))
    return image_id


def bucket(image_id: str) -> int:
    """Deterministic bucket 0..99 from a canonical image id."""
    return int(hashlib.sha256(canonical_id(image_id).encode("utf-8")).hexdigest(), 16) % 100


def bucket_partition(image_id: str) -> str:
    """The original per-image rule, used only for images in a group with no triplet."""
    b = bucket(image_id)
    for name, lo, hi in PARTITIONS:
        if lo <= b < hi:
            return name
    raise PartitionError("bucket %d outside every partition" % b)   # unreachable if tiled


def _pair(trip: Mapping) -> Tuple[str, str]:
    try:
        ref, tgt = trip["reference"], trip["target"]
    except KeyError as exc:
        raise PartitionError("triplet lacks %s" % exc) from None
    ref, tgt = canonical_id(ref), canonical_id(tgt)
    if ref == tgt:
        raise PartitionError("triplet with identical reference and target: %r" % (ref,))
    return ref, tgt


def known_positives(triplets: Iterable[Mapping]) -> Dict[str, Set[str]]:
    """Every image's known positive partners, in both directions."""
    pos: Dict[str, Set[str]] = {}
    for trip in triplets:
        ref, tgt = _pair(trip)
        pos.setdefault(ref, set()).add(tgt)
        pos.setdefault(tgt, set()).add(ref)
    return pos


def linked_groups(positives: Mapping[str, Set[str]], images: Iterable[str] = ()) -> Dict[str, str]:
    """Image id -> group key (the smallest image id of its connected component)
    over the positive graph. Images without any positive form their own group."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for img in list(positives) + [canonical_id(i) for i in images]:
        parent.setdefault(canonical_id(img), img)
    for a, partners in positives.items():
        for b in partners:
            parent.setdefault(canonical_id(b), b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)        # the root is always the smallest id
    return {img: find(img) for img in parent}


def allocate_images(triplets: Sequence[Mapping], positives: Mapping[str, Set[str]],
                    images: Iterable[str] = ()) -> Dict[str, str]:
    """Image id -> partition, by whole linked group (module docstring).
    `positives` must contain every triplet pair; extra positive pairs only
    merge groups further."""
    for trip in triplets:
        ref, tgt = _pair(trip)
        if tgt not in positives.get(ref, ()):
            raise PartitionError("positives lack the triplet pair (%r, %r)" % (ref, tgt))
    group = linked_groups(positives, images)
    n_trip: Dict[str, int] = {}
    for trip in triplets:
        g = group[_pair(trip)[0]]
        n_trip[g] = n_trip.get(g, 0) + 1
    total = len(triplets)
    target = {name: (hi - lo) / 100.0 * total for name, lo, hi in PARTITIONS}
    got = {name: 0 for name in PARTITION_NAMES}
    group_part: Dict[str, str] = {}
    for g in sorted(n_trip, key=lambda k: (-n_trip[k], k)):
        name = max(PARTITION_NAMES, key=lambda n: target[n] - got[n])   # first maximum wins ties
        group_part[g] = name
        got[name] += n_trip[g]
    return {img: group_part[g] if g in group_part else bucket_partition(g)
            for img, g in group.items()}


def split_triplets(triplets: Iterable[Mapping], image_partition: Mapping[str, str]
                   ) -> Tuple[Dict[str, List[Mapping]], Dict[str, int]]:
    """Assign triplets ({'reference': id, 'target': id, ...}) to the partition
    of their images. A triplet whose images lie in different partitions, or an
    image without a partition, is an error (no triplet is silently discarded).
    Returns (triplets per partition, counts with 'kept_<name>' and 'total');
    role-disjointness is asserted before returning."""
    kept: Dict[str, List[Mapping]] = {n: [] for n in PARTITION_NAMES}
    total = 0
    for trip in triplets:
        total += 1
        ref, tgt = _pair(trip)
        if ref not in image_partition or tgt not in image_partition:
            raise PartitionError("triplet (%r, %r) has an unallocated image" % (ref, tgt))
        p_ref, p_tgt = image_partition[ref], image_partition[tgt]
        if p_ref != p_tgt:
            raise PartitionError("triplet (%r, %r) crosses %s and %s" % (ref, tgt, p_ref, p_tgt))
        kept[p_ref].append(trip)
    assert_role_disjoint(kept)
    counts = {"kept_" + n: len(kept[n]) for n in PARTITION_NAMES}
    counts["total"] = total
    return kept, counts


def images_by_partition(kept: Mapping[str, Sequence[Mapping]]) -> Dict[str, Set[str]]:
    """Image ids used by each partition, over both the reference and target roles."""
    return {n: {_pair(t)[0] for t in trips} | {_pair(t)[1] for t in trips}
            for n, trips in kept.items()}


def assert_role_disjoint(kept: Mapping[str, Sequence[Mapping]]) -> None:
    """No image id may occur, in either role, in two partitions."""
    imgs = images_by_partition(kept)
    names = list(imgs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = imgs[a] & imgs[b]
            if both:
                raise PartitionError("partitions %s and %s share %d image(s), e.g. %r"
                                     % (a, b, len(both), sorted(both)[0]))


def sample_negatives(pool_ids: Sequence[str], exclude_ids: Iterable[str], n: int,
                     rng: np.random.Generator) -> List[str]:
    """n distinct negatives drawn uniformly without replacement from the pool,
    after removing every excluded id (positives, duplicates). Fewer eligible
    images than requested is an error, never a silently smaller set (v3.1 §8)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise PartitionError("n must be a positive integer")
    pool = [canonical_id(i) for i in pool_ids]
    if len(set(pool)) != len(pool):
        raise PartitionError("negative pool contains duplicate image ids")
    excl = {canonical_id(i) for i in exclude_ids}
    eligible = [i for i in pool if i not in excl]
    if len(eligible) < n:
        raise PartitionError("only %d eligible negatives after masking, %d requested"
                             % (len(eligible), n))
    pick = rng.choice(len(eligible), size=int(n), replace=False)
    return [eligible[k] for k in pick]


def negatives_for(trip: Mapping, pool_ids: Sequence[str], positives: Mapping[str, Set[str]],
                  n: int, rng: np.random.Generator) -> List[str]:
    """Negatives for one training triplet: the triplet's own images and every
    known positive partner of either image are masked."""
    ref, tgt = _pair(trip)
    mask = {ref, tgt} | set(positives.get(ref, ())) | set(positives.get(tgt, ()))
    return sample_negatives(pool_ids, mask, n, rng)
