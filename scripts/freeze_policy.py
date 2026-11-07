"""Freeze the CIRR-fitted policies into one file before any FashionIQ result.

Copies, verbatim and without re-fitting, from gate_suite.json (CIRR tune half):
fusion weights, the learned gate's logistic weights and standardisation, and
the gate thresholds chosen for the primary alpha=1 objective. Writes
frozen_policy.json and prints its SHA-256, which the pre-registration records.

Usage
-----
  python scripts/freeze_policy.py gate_suite.json frozen_policy.json
"""
import hashlib
import json
import sys

FEATURES = ["base_margin", "ver_gap", "base_gap", "fused_base_rank", "ver_margin"]


def main(src: str, dst: str) -> int:
    d = json.load(open(src, encoding="utf-8"))
    if d["n"] != 4181 or d["multipositive"]:
        print("refusing: source is not the CIRR gate_suite output", file=sys.stderr)
        return 1
    frozen = {"fitted_on": "CIRR validation tune half (seed 12345, n=2090)",
              "source_file": src, "features": FEATURES, "alpha_objective": 1.0, "rules": {}}
    for rname, r in d["rules"].items():
        g = r["gates"]
        frozen["rules"][rname] = {
            "fusion": "bounded" if rname == "bounded_frozen" else "convex",
            "param": r["param"],
            "weaker_param": r["weaker_param"],
            "logistic": {"weights": r["logistic"]["weights"],
                         "mean": r["logistic"]["mean"], "sd": r["logistic"]["sd"]},
            "thresholds": {"learned": g["learned"]["threshold"],
                           "vgap": g["vgap"]["threshold"],
                           "base": g["base"]["threshold"]},
        }
    text = json.dumps(frozen, indent=2, sort_keys=True)
    with open(dst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text + "\n")
    print("wrote %s  sha256 %s" % (dst, hashlib.sha256((text + "\n").encode()).hexdigest()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
