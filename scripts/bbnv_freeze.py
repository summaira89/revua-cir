"""Freeze record for the EDIR confirmatory plan (v3.1 §15).

Writes a verified run whose results.json lists the SHA-256 of the design
document (which contains the final §15), every frozen artefact the EDIR
analysis will use, the BBNV sources, and the pinned EDIR revision with its
metadata and image-part digests. Run once, before any EDIR image is
downloaded; the record's own hash is then quoted with the plan.

Usage
-----
  python scripts/bbnv_freeze.py --out-dir ../bbnv_runs/edir_plan_freeze_2026-10-21
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_provenance as prov  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDIR_REVISION = "72581ea0922cd01ccceb39fd3a69616170ca0125"
EDIR_IMAGE_PARTS = {
    "imgs_mc.tar.partaa": "1d362d382aff666b55649901262186ead59aa3cd4619f3bc98c72002044bf23b",
    "imgs_mc.tar.partab": "75b2937f59a7fc8a4e80248be0b044db3f86d4d99465ab85e6aafa0094072e4f",
    "imgs_mc.tar.partac": "ec386fecb912df047ca31978f2531a0a0d5a080f3a97ed80fbdb3174e90a053a",
    "imgs_mc.tar.partad": "9ab46943aac775c168c06babfc9dbc8f1b94e05280b4747e3bf711eefdfe582f",
    "imgs_mc.tar.partae": "06f5c571506e914acd697a120c281fe547f20a76843d1fc124270defc72b06a9",
    "imgs_mc.tar.partaf": "32caba77e9e7151638a0f45e335030b47d7fad2354a61768e726620659e492e2",
    "imgs_mc.tar.partag": "49a2ea313422ed2ac6d8bb234fb3d701bc4506539f894ebea8c9f3d4dec7c19f",
}
FILES = {
    "design_amendments": "revua_cir/scripts/Bidirectional_Verification_Design_v3.1_Amendments_2026-10-19.md",
    "design_v3": "revua_cir/scripts/Bidirectional_Verification_Design_v3_2026-10-19.md",
    "cirr_results_doc": "revua_cir/scripts/BBNV_PhaseD_CIRR_Results_2026-10-21.md",
    "analysis_cirr_v1": "bbnv_runs/analysis_cirr_v1/results.json",
    "windows_cirr_v1": "bbnv_runs/windows_cirr_v1/results.json",
    "lambda_v1": "bbnv_runs/lambda_v1/results.json",
    "heads_v1": "bbnv_heads/v1/results.json",
    "heads_v1_npz": "bbnv_heads/v1/heads.npz",
    "features_cirr_train": "bbnv_feats/v1/cirr_train/results.json",
    "features_cirr_val": "bbnv_feats/v1/cirr_val/results.json",
    "manifest_groups_cluster": "bbnv_runs/manifest_groups_cluster/results.json",
    "edir_queries": "edir_meta/%s/queries.jsonl" % EDIR_REVISION[:8],
    "edir_corpus": "edir_meta/%s/corpus.jsonl" % EDIR_REVISION[:8],
    "edir_instances": "edir_meta/%s/instances.jsonl" % EDIR_REVISION[:8],
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    missing = [p for p in FILES.values() if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        ap.error("missing frozen artefacts: %s" % missing)
    for key in ("analysis_cirr_v1", "windows_cirr_v1", "lambda_v1", "heads_v1", "features_cirr_train",
                "features_cirr_val", "manifest_groups_cluster"):
        prov.verify_run(os.path.dirname(os.path.join(ROOT, FILES[key])))
    results = {
        "schema": "bbnv-edir-plan-freeze/1", "plan": "v3.1 §15 (final)",
        "edir": {"repo": "EDIR-BENCH/EDIR", "revision": EDIR_REVISION, "image_part_lfs_sha256": EDIR_IMAGE_PARTS},
        "sha256": {k: prov.sha256_file(os.path.join(ROOT, p)) for k, p in FILES.items()},
        "paths": FILES,
        "source_sha256": prov.source_hashes(prov.BBNV_SOURCES + ("bbnv_freeze.py",)),
    }
    prov.write_run(args.out_dir, results, {})
    prov.verify_run(args.out_dir)
    print("[freeze] wrote and verified %s | results.json sha256 %s"
          % (args.out_dir, prov.sha256_file(os.path.join(args.out_dir, prov.RESULTS_NAME))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
