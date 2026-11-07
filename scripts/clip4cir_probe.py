"""CLIP4Cir external-retriever reproduction probe on CIRR validation: amendment v3.1 §24. Laptop GPU.

For each released CIRR configuration (RN50_fullft = reproduction gate, RN50x4_fullft =
external retriever) this encodes the 2297-image validation gallery and the 4181 queries
with the released CLIP + Combiner pair, computes the official CIRR validation metrics, and
measures peak GPU memory. Nothing from our own model is run.

Weights. CLIP4Cir's combiner training and validate.py load the fine-tuned CLIP through
clip.load on CUDA, i.e. into an fp16-converted model, and then call .float(). The primary
path reproduces that exactly with clip.model.build_model(sd).float(); the secondary path
reloads the exact fp32 tensors (strict) and is descriptive only.

Gate (fixed before any output): RN50_fullft, primary path, full validation set, every one
of R@1/5/10/50 and R_subset@1/2/3 within GATE_TOL pp of the published validation values
(TOMM 2023, Table 2, last row). The RN50x4 features are released for later analysis only
if the gate passes.

  (from revua_cir/) python scripts/clip4cir_probe.py --root .. --out ../revision_work/clip4cir_probe_2026-11-03
"""
from __future__ import annotations

import os
import sys

# Windows/Anaconda: numpy's MKL and pip's torch each ship an Intel OpenMP runtime and the
# process aborts once both start threads (OMP Error #15). As in bbnv_oracle.py, MKL's
# sequential mode keeps numpy off OpenMP (never KMP_DUPLICATE_LIB_OK, which can corrupt
# results). Effective only if set before numpy is first imported; recorded in the output.
_NUMPY_PRELOADED = "numpy" in sys.modules
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import argparse  # noqa: E402
import gc  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from typing import Dict, List, Sequence  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import fusion_controls as fc  # noqa: E402
import revision_followups as rf  # noqa: E402

AMENDMENT = "v3.1 §24 (2026-11-03)"
C4C = "baselines/clip4cir_extract/CIRR"
INPUTS = {
    "RN50_fullft_clip": (C4C + "/RN50_fullft/cirr_clip_RN50_fullft.pt",
                         "60866c3cf4d6b63240814e3bfcab4895dd863ca50c88b560368561249dcb48e0"),
    "RN50_fullft_comb": (C4C + "/RN50_fullft/cirr_comb_RN50_fullft.pt",
                         "7ed36cbfb3527fce72c839470934e222e2225fbb0fa96815c11c0100e608d8e0"),
    "RN50x4_fullft_clip": (C4C + "/RN50x4_fullft/cirr_clip_RN50x4_fullft.pt",
                           "7e730077485589379ebba4a52d2ca16e12c894a7ca3654aed5c674bcafb1f50e"),
    "RN50x4_fullft_comb": (C4C + "/RN50x4_fullft/cirr_comb_RN50x4_fullft.pt",
                           "a5c7bb1a4a9de2a5cad1b2cb26ee645d41f98de28459b4f77c2ca9e213903f77"),
    "c4c_combiner_py": ("baselines/CLIP4Cir/src/combiner.py",
                        "eed76ce258e9359d6d618e7f090aa34d0dc424219974ee2697c267864452e2d6"),
    "c4c_data_utils_py": ("baselines/CLIP4Cir/src/data_utils.py",
                          "b2e950fa12350510510a17bb05a9fec75ff825b2abae37969aa7613d3a7c9735"),
    "captions": ("cirr_val_unpacked/cirr_val/captions/cap.rc2.val.json",
                 "a85c3a1aa464f1af7229918e8018d08b8b20ce5dab479ffdf39d61113140f919"),
    "split": ("cirr_val_unpacked/cirr_val/image_splits/split.rc2.val.json",
              "de36f7a7219c8bb06347a67748cfc68405be7dad62853a1e31846c98e491235b"),
    "features": ("bbnv_feats/v1/cirr_val/features.npz",
                 "3a91e82caa69479ba6fe18ce93adbcca9345a454cc53812c3105d4927495033f"),
    "cache": ("bbnv_caches/rv_score_cache_fc_aep_rv.jsonl",
              "26d7fc552b4ae902d9ecfd0bd4ec2bad329e627536887ae4b241e5d9a85c0308"),
}
IMAGE_DIR = "cirr_val_unpacked/cirr_val"
# name: (feature dim, projection dim, hidden dim, input resolution), all read back from the tensors
CONFIGS = {"RN50_fullft": (1024, 4096, 8192, 224), "RN50x4_fullft": (640, 2560, 5120, 288)}
TARGET_RATIO = 1.25
PUBLISHED_RN50_VAL = {"R@1": 42.05, "R@5": 76.13, "R@10": 86.51, "R@50": 97.49,
                      "Rs@1": 70.15, "Rs@2": 87.18, "Rs@3": 94.40}
GATE_TOL = 0.5
OUR_BASE_FULL_R1 = 37.072
BATCH = 32


# ---------------------------------------------------------------------------
# metrics (pure numpy; tested on CPU)
# ---------------------------------------------------------------------------
def ranked_rows(scores: np.ndarray, exclude: int) -> np.ndarray:
    """Gallery rows by descending score, `exclude` removed, ties to the lower row."""
    s = scores.astype(np.float64)
    order = np.lexsort((np.arange(len(s)), -s))
    return order[order != exclude]


def query_ranks(scores: np.ndarray, exclude: int, target: int, members: Sequence[int]) -> Dict[str, int]:
    """0-based gallery rank of the target and its rank among the subset members (reference already removed)."""
    order = ranked_rows(scores, exclude)
    pos = int(np.nonzero(order == target)[0][0])
    mem = np.unique(np.asarray(members, dtype=np.int64))
    sub = order[np.isin(order, mem)]
    if len(sub) != len(mem) or target not in mem:
        raise SystemExit("ABORT: subset members missing from the ranking or target not a member")
    return {"rank": pos, "subset_rank": int(np.nonzero(sub == target)[0][0])}


def cirr_metrics(ranks: List[Dict[str, int]], idx: Sequence[int]) -> Dict[str, float]:
    out = {"R@%d" % k: 100.0 * np.mean([ranks[i]["rank"] < k for i in idx]) for k in (1, 5, 10, 50)}
    out.update({"Rs@%d" % k: 100.0 * np.mean([ranks[i]["subset_rank"] < k for i in idx]) for k in (1, 2, 3)})
    return out


def gate(metrics: Dict[str, float], published: Dict[str, float], tol: float) -> Dict[str, object]:
    dev = {k: metrics[k] - v for k, v in published.items()}
    return {"deviation_pp": dev, "tol_pp": tol, "pass": all(abs(d) <= tol for d in dev.values())}


def check_shapes(clip_sd: dict, comb_sd: dict, cfg: Sequence[int]) -> None:
    feat, proj, hid, res = cfg
    got_res = 32 * round((clip_sd["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    want = {"text_projection_layer.weight": (proj, feat), "combiner_layer.weight": (hid, 2 * proj),
            "output_layer.weight": (feat, hid)}
    bad = [k for k, s in want.items() if tuple(comb_sd[k].shape) != s]
    if bad or tuple(clip_sd["text_projection"].shape)[1] != feat or got_res != res:
        raise SystemExit("ABORT: checkpoint shapes disagree with config %s (%s, res %d)" % (cfg, bad, got_res))


# ---------------------------------------------------------------------------
# GPU part
# ---------------------------------------------------------------------------
def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_clip(clip_sd: dict, path: str):
    """'released' = clip.load-on-CUDA rounding (build_model converts to fp16, then .float());
    'exact_fp32' = the same model with the saved fp32 tensors reloaded strictly."""
    import clip.model as cm
    model = cm.build_model(dict(clip_sd)).float()
    if path == "exact_fp32":
        model.load_state_dict(clip_sd, strict=True)
    elif path != "released":
        raise ValueError(path)
    return model.eval()


def encode(model, comb, image_files: List[str], pre, captions: List[str], ref_row: np.ndarray, device):
    """Images are preprocessed one batch at a time, so host memory holds a single batch (§24.2)."""
    import clip
    import torch
    import torch.nn.functional as F
    from PIL import Image
    img, txt = [], []
    with torch.no_grad():
        for s in range(0, len(image_files), BATCH):
            batch = []
            for f in image_files[s:s + BATCH]:
                with Image.open(f) as im:
                    batch.append(pre(im))
            img.append(model.encode_image(torch.stack(batch).to(device)).float())
        img = torch.cat(img)
        for s in range(0, len(captions), BATCH):
            tok = clip.tokenize(captions[s:s + BATCH]).to(device)       # no truncation, as validate.py
            txt.append(model.encode_text(tok).float())
        txt = torch.cat(txt)
        ref = img[torch.as_tensor(ref_row, device=device)]
        comp = torch.cat([F.normalize(comb.combine_features(ref[s:s + BATCH], txt[s:s + BATCH]), dim=-1)
                          for s in range(0, len(captions), BATCH)])
    return img.cpu().numpy(), txt.cpu().numpy(), comp.cpu().numpy()


def run_config(name, paths, captions, image_files, ref_row, tgt_row, members, device):
    import torch
    cfg = CONFIGS[name]
    clip_sd = torch.load(paths[name + "_clip"], map_location="cpu", weights_only=True)["CLIP"]
    comb_sd = torch.load(paths[name + "_comb"], map_location="cpu", weights_only=True)["Combiner"]
    check_shapes(clip_sd, comb_sd, cfg)
    du = _load_module("c4c_data_utils", paths["c4c_data_utils_py"])
    cb = _load_module("c4c_combiner", paths["c4c_combiner_py"])
    pre = du.targetpad_transform(TARGET_RATIO, cfg[3])
    comb = cb.Combiner(cfg[0], cfg[1], cfg[2])
    comb.load_state_dict(comb_sd, strict=True)
    comb = comb.to(device).eval()
    texts = [c["caption"] for c in captions]
    out = {}
    feats = None
    for path in ("released", "exact_fp32"):
        model = build_clip(clip_sd, path).to(device)
        if model.visual.input_resolution != cfg[3]:
            raise SystemExit("ABORT: %s input resolution %d != %d" % (name, model.visual.input_resolution, cfg[3]))
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        img, txt, comp = encode(model, comb, image_files, pre, texts, ref_row, device)
        torch.cuda.synchronize()
        enc_s = time.time() - t1
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        imgn = img / np.linalg.norm(img, axis=1, keepdims=True)
        ranks = [query_ranks(imgn @ comp[i].astype(np.float64), int(ref_row[i]), int(tgt_row[i]), members[i])
                 for i in range(len(texts))]
        out[path] = {"encode_seconds": round(enc_s, 1), "peak_gpu_gib": round(peak, 3), "ranks": ranks}
        if path == "released":
            feats = {"image_emb": img, "text_emb": txt, "composed": comp}
        del model
        torch.cuda.empty_cache()
    return out, feats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        ap.error("%s exists; refusing to overwrite" % args.out)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("ABORT: CUDA not available")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    t0 = time.time()
    paths = {k: os.path.join(args.root, p) for k, (p, _) in INPUTS.items()}
    hashes = {k: rf.sha256(p) for k, p in paths.items()}
    for k, (_, want) in INPUTS.items():
        if hashes[k] != want:
            raise SystemExit("ABORT: %s sha256 %s != recorded %s" % (k, hashes[k], want))

    with open(paths["captions"], encoding="utf-8") as fh:
        captions = json.load(fh)
    with open(paths["split"], encoding="utf-8") as fh:
        split = json.load(fh)
    with np.load(paths["features"], allow_pickle=False) as z:
        names = [os.path.splitext(os.path.basename(str(p)))[0] for p in z["image_paths"]]
        ref_row, tgt_row = z["ref_row"].astype(np.int64), z["tgt_row"].astype(np.int64)
    if sorted(names) != sorted(split) or len(set(names)) != 2297:
        raise SystemExit("ABORT: gallery rows differ from the official validation split")
    row = {n: i for i, n in enumerate(names)}
    image_files = [os.path.join(args.root, IMAGE_DIR, split[n]) for n in names]
    recs, mp = fc.load_records(paths["cache"])
    if mp or len(recs) != 4181 or len(captions) != 4181:
        raise SystemExit("ABORT: expected 4181 aligned queries")
    members = []
    for i, c in enumerate(captions):
        if row[c["reference"]] != ref_row[i] or row[c["target_hard"]] != tgt_row[i]:
            raise SystemExit("ABORT: caption %d does not match the feature rows" % i)
        members.append([row[m] for m in c["img_set"]["members"] if row[m] != ref_row[i]])
    rel = [fc.relevant_set(r, mp) for r in recs]
    base = [fc.hit_at_1(r["base_score"], rel[i]) for i, r in enumerate(recs)]
    if round(100.0 * np.mean(base), 3) != OUR_BASE_FULL_R1:
        raise SystemExit("ABORT: our base R@1 does not reproduce %s" % OUR_BASE_FULL_R1)
    tune, report = fc.split_indices(len(recs))
    full = list(range(len(recs)))

    res = {"amendment": AMENDMENT, "inputs_sha256": hashes, "status": "reproduction probe; exploratory",
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "runtime": {"MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER"),
                       "mkl_guard_effective": not _NUMPY_PRELOADED}, "configs": {}}
    feats = {}
    for name in ("RN50_fullft", "RN50x4_fullft"):
        out, f = run_config(name, paths, captions, image_files, ref_row, tgt_row, members, device)
        entry = {}
        for path in ("released", "exact_fp32"):
            rk = out[path]["ranks"]
            hits = [int(r["rank"] == 0) for r in rk]
            entry[path] = {"encode_seconds": out[path]["encode_seconds"], "peak_gpu_gib": out[path]["peak_gpu_gib"],
                           "full": cirr_metrics(rk, full), "report": cirr_metrics(rk, report),
                           "vs_our_base_full": rf.paired(hits, base),
                           "vs_our_base_report": rf.paired([hits[i] for i in report], [base[i] for i in report])}
        res["configs"][name] = entry
        feats[name] = f
        del out
        gc.collect()
        torch.cuda.empty_cache()
        print("[24] %s done: released R@1 %.2f, exact-fp32 R@1 %.2f, peak %.2f GiB, %.0fs" % (
            name, entry["released"]["full"]["R@1"], entry["exact_fp32"]["full"]["R@1"],
            entry["released"]["peak_gpu_gib"], time.time() - t0), flush=True)
    g = gate(res["configs"]["RN50_fullft"]["released"]["full"], PUBLISHED_RN50_VAL, GATE_TOL)
    res["gate"] = g
    print("[24] reproduction gate %s: %s" % ("PASS" if g["pass"] else "FAIL",
                                             {k: round(v, 2) for k, v in g["deviation_pp"].items()}), flush=True)
    os.makedirs(args.out)
    if g["pass"]:
        for name, f in feats.items():
            with open(os.path.join(args.out, "features_%s.npz" % name), "xb") as fh:
                np.savez(fh, query_id=np.array([r["query_id"] for r in recs]), image_names=np.array(names),
                         ref_row=ref_row, tgt_row=tgt_row, **f)
    res["source_sha256"] = {f: rf.sha256(os.path.join(_HERE, f)) for f in
                            ("clip4cir_probe.py", "fusion_controls.py", "revision_followups.py")}
    res["wall_seconds"] = round(time.time() - t0, 1)
    out = os.path.join(args.out, "results.json")
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("[done] wrote %s (sha256 %s) in %.0fs" % (out, rf.sha256(out), time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
