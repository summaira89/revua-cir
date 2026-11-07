"""BBNV Phase C head training on the real CIRR-train features (v3.1 §11). CPU.

For each direction (forward: (r_hat, t_hat) -> target; inverse: (c_hat,
t_hat) -> reference), each seed in SEEDS and each kappa in KAPPAS, one M = 1
head is trained on the HEAD partition with the sampled candidate-set
cross-entropy (the positive plus N_NEG negatives drawn uniformly without
replacement from the NEGBANK partition, re-drawn every epoch). The dev-select
NLL is evaluated after every epoch on one fixed negative set shared by every
configuration; for each (direction, seed) the (kappa, epoch) with the lowest
dev-select NLL is kept. Dev-calibrate and every validation set are untouched.

Inputs are checked before any training: the Phase B features (verified run,
schema, checkpoint, finite unit rows) and the partition manifest (verified
run, the pinned allocation digest, the same manifest the features were built
from, triplet alignment). Known positives never cross partitions (§10); this
is ASSERTED for every head and dev triplet, after which uniform sampling from
the whole pool is exactly the declared masked sampling.

Usage
-----
  python scripts/bbnv_train.py --features-dir ../bbnv_feats/v1/cirr_train \\
      --manifest-dir ../bbnv_runs/manifest_groups_v3_2026-10-20 --out-dir ../bbnv_heads/v1 [--threads 4]
  --smoke  2 epochs, kappas (5, 10), seed 0 only: a plumbing check whose
           output is marked smoke and must never be used for results

Output (fresh directory, written via <out-dir>.staging): heads.npz (the
selected head of every direction and seed, float32 parameters) and
results.json (protocol, full dev-NLL grids, training curves, selections).
"""
from __future__ import annotations

import os
import sys

# see bbnv_oracle.py: keep numpy's MKL off OpenMP so torch's runtime is the only one
_NUMPY_PRELOADED = "numpy" in sys.modules
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from typing import Dict, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_core as core  # noqa: E402
import bbnv_extract as ext  # noqa: E402
import bbnv_manifest as man  # noqa: E402
import bbnv_provenance as prov  # noqa: E402

SCHEMA = "bbnv-heads/1"
HEADS_NAME = "heads.npz"
FEATURES_SCHEMA = "bbnv-features/2"
DIRECTIONS = ("forward", "inverse")
SEEDS = tuple(sorted({s for pair in core.SEED_PAIRS for s in pair}))
KAPPAS = (5.0, 10.0, 20.0, 40.0, 80.0)
HIDDEN, EPOCHS, BATCH, LR, WEIGHT_DECAY, N_NEG = 512, 30, 256, 1e-3, 1e-4, 1023
DEV_NEG_SEED = 20261021
SMOKE = {"epochs": 2, "kappas": (5.0, 10.0), "seeds": (0,)}


class TrainError(ValueError):
    pass


# --------------------------------------------------------------- inputs --
def load_inputs(features_dir: str, manifest_dir: str) -> Dict[str, object]:
    """Verified Phase B cirr_train features bound to the approved manifest."""
    fres = prov.verify_run(features_dir)
    if fres.get("schema") != FEATURES_SCHEMA or fres.get("target") != "cirr_train":
        raise TrainError("%s is not a cirr_train feature run of schema %s" % (features_dir, FEATURES_SCHEMA))
    mres = prov.verify_run(manifest_dir)
    images_sha = prov.sha256_file(os.path.join(manifest_dir, "images.jsonl"))
    if images_sha != ext.APPROVED_IMAGES_SHA256 or fres.get("manifest_images_sha256") != images_sha:
        raise TrainError("features and manifest are not both of the approved allocation")
    if fres.get("manifest_results_sha256") != prov.sha256_file(os.path.join(manifest_dir, prov.RESULTS_NAME)):
        raise TrainError("the features were extracted from a different manifest run")
    with np.load(os.path.join(features_dir, ext.FEATURES_NAME), allow_pickle=False) as z:
        feat = {k: z[k] for k in z.files}
    ext.check_unit_rows("image_emb", feat["image_emb"], 2)
    ext.check_unit_rows("text_emb", feat["text_emb"], 2)
    trips = man.read_jsonl(os.path.join(manifest_dir, "triplets.jsonl"))
    ids = feat["image_ids"]
    if (len(trips) != len(feat["pairid"])
            or any(t["pairid"] != int(p) for t, p in zip(trips, feat["pairid"]))
            or any(ids[r] != t["reference"] or ids[g] != t["target"] or t["partition"] != feat["partition"][r]
                   for t, r, g in zip(trips, feat["ref_row"], feat["tgt_row"]))):
        raise TrainError("features and manifest triplets disagree")
    positives = {r["image_id"]: set(r["partners"])
                 for r in man.read_jsonl(os.path.join(manifest_dir, "positives.jsonl"))}
    return {"feat": feat, "positives": positives, "features_results": fres, "manifest_results": mres}


def split_examples(feat: Mapping[str, np.ndarray], positives: Mapping[str, set]) -> Dict[str, np.ndarray]:
    """Triplet indices of the head and dev-select partitions and the negbank
    pool's image rows. Asserts that no known positive of any used triplet's
    images lies in the pool (so uniform sampling from the pool is the masked
    sampling) and that each triplet's images share its partition."""
    part = feat["partition"]
    ids = feat["image_ids"]
    ref, tgt = feat["ref_row"], feat["tgt_row"]
    if not np.array_equal(part[ref], part[tgt]):
        raise TrainError("a triplet's reference and target lie in different partitions")
    pool = np.flatnonzero(part == "negbank")
    pool_ids = set(ids[pool].tolist())
    out = {"pool": pool}
    for name in ("head", "dev_select"):
        idx = np.flatnonzero(part[ref] == name)
        if len(idx) == 0:
            raise TrainError("partition %s has no triplets" % name)
        for i in idx:
            a, b = ids[ref[i]], ids[tgt[i]]
            if ({a, b} | positives.get(a, set()) | positives.get(b, set())) & pool_ids:
                raise TrainError("a known positive of triplet %d lies in the negative pool" % i)
        out[name] = idx
    if len(pool) < N_NEG:
        raise TrainError("negative pool has %d images, %d needed" % (len(pool), N_NEG))
    return out


def direction_arrays(feat: Mapping[str, np.ndarray], idx: np.ndarray, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    """(inputs, positives): forward conditions on the reference and predicts
    the target; inverse conditions on the target (as candidate) and predicts
    the reference. Inputs are [condition image, text] (2 x 768)."""
    if direction not in DIRECTIONS:
        raise TrainError("unknown direction %r" % (direction,))
    cond, pos = ((feat["ref_row"], feat["tgt_row"]) if direction == "forward"
                 else (feat["tgt_row"], feat["ref_row"]))
    img, txt = feat["image_emb"], feat["text_emb"]
    return (np.concatenate([img[cond[idx]], txt[idx]], axis=1).astype(np.float32),
            img[pos[idx]].astype(np.float32))


def sample_negative_columns(n_rows: int, pool_size: int, n_neg: int, rng: np.random.Generator) -> np.ndarray:
    """(n_rows, n_neg) pool columns, each row a uniform sample without replacement."""
    if not 1 <= n_neg <= pool_size:
        raise TrainError("cannot draw %d negatives from a pool of %d" % (n_neg, pool_size))
    return np.argsort(rng.random((n_rows, pool_size)), axis=1)[:, :n_neg]


# ---------------------------------------------------------------- model --
def build_head(seed: int):
    """The Phase A head body (bbnv_oracle.build_head) with M = 1 and NO kappa
    parameter: kappa is fixed per run (§11). float32."""
    import torch
    nn = torch.nn
    torch.manual_seed(seed)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(nn.Linear(2 * 768, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
            self.hyp = nn.Linear(HIDDEN, 768)

        def forward(self, x):
            return torch.nn.functional.normalize(self.hyp(self.body(x)), dim=-1)

    return Head().float()


def candidate_nll(hyp, pos, pool, neg_cols, kappa: float):
    """Per-example -log softmax over [positive, sampled negatives] of
    kappa * <hyp, x>: exactly the M = 1 component-normalised vMF over that
    candidate set. Pool logits are computed once and the sampled columns
    gathered, never materialising (B, n_neg, D) candidate tensors."""
    import torch
    pos_logit = kappa * (hyp * pos).sum(-1)
    neg_logit = torch.gather(kappa * hyp @ pool.T, 1, neg_cols)
    return torch.logsumexp(torch.cat([pos_logit[:, None], neg_logit], dim=1), dim=1) - pos_logit


def train_one(direction: str, seed: int, kappa: float, data: Mapping[str, np.ndarray], epochs: int) -> Dict[str, object]:
    """One configuration. The example order and the training negatives depend
    on (seed, direction, epoch) only, so every kappa of a seed sees the same
    data stream; the best-dev state is kept."""
    import torch
    d_idx = DIRECTIONS.index(direction)
    head = build_head(1000 * d_idx + seed)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x_tr, p_tr = (torch.from_numpy(a) for a in data["train"])
    x_dev, p_dev = (torch.from_numpy(a) for a in data["dev"])
    pool = torch.from_numpy(data["pool_emb"])
    dev_cols = torch.from_numpy(data["dev_cols"])
    n = len(x_tr)
    train_curve, dev_curve = [], []
    best = (math.inf, -1, None)
    for epoch in range(1, epochs + 1):
        order = np.random.default_rng([seed, d_idx, 0, epoch]).permutation(n)
        neg_rng = np.random.default_rng([seed, d_idx, 1, epoch])
        head.train()
        total = 0.0
        for s in range(0, n, BATCH):
            b = torch.from_numpy(order[s:s + BATCH])
            cols = torch.from_numpy(sample_negative_columns(len(b), len(pool), N_NEG, neg_rng))
            loss = candidate_nll(head(x_tr[b]), p_tr[b], pool, cols, kappa).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)
        head.eval()
        with torch.no_grad():
            dev = float(candidate_nll(head(x_dev), p_dev, pool, dev_cols, kappa).mean())
        if not (math.isfinite(total) and math.isfinite(dev)):
            raise TrainError("non-finite loss: %s seed %d kappa %g epoch %d" % (direction, seed, kappa, epoch))
        train_curve.append(total / n)
        dev_curve.append(dev)
        if dev < best[0]:
            best = (dev, epoch, {k: v.detach().clone() for k, v in head.state_dict().items()})
    return {"train_nll": train_curve, "dev_nll": dev_curve, "best_dev_nll": best[0], "best_epoch": best[1],
            "state": best[2]}


def select(runs: Mapping[float, Mapping[str, object]]) -> float:
    """The kappa whose best epoch has the lowest dev NLL; ties go to the smaller kappa."""
    return min(sorted(runs), key=lambda k: runs[k]["best_dev_nll"])


# ------------------------------------------------------------------ main --
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="plumbing check only; output marked smoke")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("--threads must be positive")
    if os.path.exists(man.staging_dir(args.out_dir)):
        ap.error("%s exists (an interrupted run); use a fresh --out-dir" % man.staging_dir(args.out_dir))
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    import torch
    torch.set_num_threads(args.threads)
    epochs = SMOKE["epochs"] if args.smoke else EPOCHS
    kappas = SMOKE["kappas"] if args.smoke else KAPPAS
    seeds = SMOKE["seeds"] if args.smoke else SEEDS

    t0 = time.time()
    inp = load_inputs(args.features_dir, args.manifest_dir)
    feat = inp["feat"]
    split = split_examples(feat, inp["positives"])
    pool_emb = feat["image_emb"][split["pool"]].astype(np.float32)
    dev_cols = sample_negative_columns(len(split["dev_select"]), len(pool_emb), N_NEG,
                                       np.random.default_rng(DEV_NEG_SEED))
    print("[train] head %d | dev_select %d | pool %d | inputs checked in %.0fs"
          % (len(split["head"]), len(split["dev_select"]), len(pool_emb), time.time() - t0), flush=True)

    arrays: Dict[str, np.ndarray] = {}
    grid: Dict[str, Dict[str, object]] = {}
    for direction in DIRECTIONS:
        data = {"train": direction_arrays(feat, split["head"], direction),
                "dev": direction_arrays(feat, split["dev_select"], direction),
                "pool_emb": pool_emb, "dev_cols": dev_cols}
        for seed in seeds:
            runs = {}
            for kappa in kappas:
                t = time.time()
                runs[kappa] = train_one(direction, seed, kappa, data, epochs)
                print("[train] %s seed %d kappa %g: best dev NLL %.4f at epoch %d (%.0fs)"
                      % (direction, seed, kappa, runs[kappa]["best_dev_nll"], runs[kappa]["best_epoch"],
                         time.time() - t), flush=True)
            k_best = select(runs)
            key = "%s_s%d" % (direction, seed)
            for name, val in runs[k_best]["state"].items():
                arrays["%s/%s" % (key, name)] = val.numpy()
            arrays["%s/kappa" % key] = np.asarray(k_best, dtype=np.float32)
            grid[key] = {"selected_kappa": k_best, "selected_epoch": runs[k_best]["best_epoch"],
                         "selected_dev_nll": runs[k_best]["best_dev_nll"],
                         "runs": {str(k): {kk: v for kk, v in r.items() if kk != "state"} for k, r in runs.items()}}
    results = {
        "schema": SCHEMA, "smoke": bool(args.smoke),
        "protocol": {"amendment": "v3.1 §11", "hidden": HIDDEN, "n_comp": 1, "epochs": epochs,
                     "kappas": list(kappas), "seeds": list(seeds), "batch": BATCH, "lr": LR,
                     "weight_decay": WEIGHT_DECAY, "n_neg": N_NEG, "dev_neg_seed": DEV_NEG_SEED,
                     "optimizer": "AdamW", "selection": "min dev_select NLL over (kappa, epoch)"},
        "data": {"head_triplets": int(len(split["head"])), "dev_select_triplets": int(len(split["dev_select"])),
                 "pool_images": int(len(pool_emb)), "chance_nll": math.log(N_NEG + 1)},
        "params_per_head": int(sum(p.numel() for p in build_head(0).parameters())),
        "selections": grid,
        "inputs": {"features_dir": os.path.abspath(args.features_dir),
                   "features_results_sha256": prov.sha256_file(os.path.join(args.features_dir, prov.RESULTS_NAME)),
                   "manifest_dir": os.path.abspath(args.manifest_dir),
                   "checkpoint_sha256": inp["features_results"]["manifest"]["checkpoint_sha256"]},
        "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_train.py", "args": vars(args),
                     "source_sha256": prov.source_hashes(),
                     "runtime": prov.runtime_info(_NUMPY_PRELOADED, args.threads)},
        "wall_seconds": round(time.time() - t0, 1)}
    ext.publish_npz_run(args.out_dir, arrays, results, name=HEADS_NAME)
    print("[train] wrote and verified %s in %.0fs | %s" % (
        args.out_dir, time.time() - t0,
        {k: (v["selected_kappa"], v["selected_epoch"], round(v["selected_dev_nll"], 4)) for k, v in grid.items()}),
        flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
