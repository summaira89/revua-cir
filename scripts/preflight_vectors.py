"""CPU preflight for resuming the verifier-vector dump (run before sbatch, and
again inside the job before the GPU is used).

Checks, all of which must pass:
  * the retained CIRR dump is complete, matches its manifest, and was made by
    THIS run (same cache, config, checkpoint and module combo; full query
    count), and passes the analysis's own loader (load_pair), so the BNV
    analysis will accept it;
  * every FashionIQ cache exists, is well formed and has the expected query
    count;
  * every FashionIQ output location is either clean (to be dumped) or holds a
    complete dump of THIS run (to be skipped); any other state is an error.

Prints one line per FashionIQ category: "<category> complete" or
"<category> to dump". Exit 0 on success, 2 on any problem.

Usage
-----
  python scripts/preflight_vectors.py --config CFG --checkpoint CKPT --module-combo fc+aep+rv \\
      --cirr CIRR_CACHE CIRR_NPZ \\
      --fiq DRESS_CACHE DRESS_NPZ SHIRT_CACHE SHIRT_NPZ TOPTEE_CACHE TOPTEE_NPZ
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bank_normalized_verification import FIQ_EXPECTED, load_pair  # noqa: E402
from dump_verifier_vectors import (artifact_paths, complete_output, load_cache,  # noqa: E402
                                   run_identity, sha256_file)

FIQ_ORDER = ("dress", "shirt", "toptee")


def check(config: str, checkpoint: str, module_combo: str, cirr, fiq) -> dict:
    """{category: True if complete, False if to dump}; raises ValueError
    (or OSError) naming the first problem."""
    ckpt_sha = sha256_file(checkpoint)
    cirr_cache, cirr_npz = cirr
    if not cirr_npz.endswith(".npz"):
        raise ValueError("the CIRR vector file must end in .npz")
    n_cirr = len(load_cache(cirr_cache))
    if not complete_output(artifact_paths(cirr_npz),
                           run_identity("cirr", None, n_cirr, module_combo,
                                        cirr_cache, config, ckpt_sha)):
        raise ValueError("no CIRR vectors at %s" % cirr_npz)
    load_pair(cirr_cache, cirr_npz)
    state = {}
    for cat, (cache, npz) in zip(FIQ_ORDER, fiq):
        if not npz.endswith(".npz"):
            raise ValueError("%s: the vector file must end in .npz" % cat)
        n = len(load_cache(cache))
        if n != FIQ_EXPECTED[cat]:
            raise ValueError("%s cache has %d records, expected %d" % (cat, n, FIQ_EXPECTED[cat]))
        state[cat] = complete_output(artifact_paths(npz),
                                     run_identity("fashioniq", cat, n, module_combo,
                                                  cache, config, ckpt_sha))
    return state


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--module-combo", required=True)
    ap.add_argument("--cirr", nargs=2, required=True, metavar=("CACHE", "NPZ"))
    ap.add_argument("--fiq", nargs=6, required=True,
                    metavar=("DRESS_CACHE", "DRESS_NPZ", "SHIRT_CACHE", "SHIRT_NPZ",
                             "TOPTEE_CACHE", "TOPTEE_NPZ"))
    args = ap.parse_args(argv)
    fiq = [(args.fiq[2 * i], args.fiq[2 * i + 1]) for i in range(len(FIQ_ORDER))]
    try:
        state = check(args.config, args.checkpoint, args.module_combo, args.cirr, fiq)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print("CIRR vectors verified: %s" % args.cirr[1])
    for cat in FIQ_ORDER:
        print("%s %s" % (cat, "complete" if state[cat] else "to dump"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
