"""Provenance and atomic output for BBNV runs (source audit, finding 9).

Every BBNV output records what produced it (source hashes, arguments, seeds,
data digests, runtime and thread configuration) and is written through a
temporary file that is re-read and validated before it replaces the final
path, so an interrupted run never leaves a truncated result behind.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
BBNV_SOURCES = ("bbnv_core.py", "bbnv_partition.py", "bbnv_oracle.py", "bbnv_profile.py",
                "bbnv_provenance.py", "bbnv_manifest.py", "bbnv_extract.py", "bbnv_train.py",
                "bbnv_score.py", "bbnv_analyze.py")
MANIFEST_SCHEMA = "bbnv-run-manifest/1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    """Digest of an array's dtype, shape and C-ordered bytes."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(repr(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def source_hashes(names: Sequence[str] = BBNV_SOURCES) -> Dict[str, str]:
    return {n: sha256_file(os.path.join(SCRIPTS, n)) for n in names}


def runtime_info(numpy_preloaded: Optional[bool], torch_threads: Optional[int] = None) -> Dict[str, object]:
    """Interpreter, library versions, thread configuration and whether the
    MKL guard was set before numpy was first imported (the guard is only
    effective if it was). numpy_preloaded=None means the script sets no guard
    (no numerical work), recorded as None rather than as effective. The guard
    only concerns MKL; the BLAS actually loaded is in blas_openmp_pools."""
    info: Dict[str, object] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "numpy": np.__version__,
        "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "mkl_guard_effective": None if numpy_preloaded is None else not numpy_preloaded,
    }
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    if torch_threads is not None:
        info["torch_threads_requested"] = torch_threads
    try:
        from threadpoolctl import threadpool_info
        info["blas_openmp_pools"] = [{k: p.get(k) for k in ("user_api", "internal_api",
                                                           "num_threads", "version")}
                                     for p in threadpool_info()]
    except ImportError:
        info["blas_openmp_pools"] = "threadpoolctl unavailable"
    return info


def process_rss_mb() -> Optional[float]:
    """Resident set size of this process, if psutil is available."""
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2 ** 20, 1)


def _atomic_write_text(path: str, text: str) -> str:
    """Write `text` to path.partial, close it, read it back, compare, then
    replace `path`. On ANY failure (write, read-back, mismatch, replace) the
    temporary file is removed after it has been closed, so nothing partial is
    ever left behind. Returns the SHA-256 of the final file."""
    tmp = path + ".partial"
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        with open(tmp, encoding="utf-8", newline="\n") as fh:
            back = fh.read()
        if back != text:
            raise IOError("re-read of %s does not match what was written" % tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return sha256_file(path)


def atomic_write_json(path: str, obj: Mapping) -> str:
    """JSON via a validated temporary file. Serialised before any file is
    opened, so an unserialisable object writes nothing."""
    return _atomic_write_text(path, json.dumps(obj, indent=2, allow_nan=False))


def atomic_write_jsonl(path: str, rows: Iterable[Mapping]) -> str:
    """One JSON object per line, via a validated temporary file; serialised first."""
    return _atomic_write_text(path, "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def prepare_fresh_dir(path: str) -> str:
    """Create an output directory that must be new or empty. Refusing a
    non-empty directory means a run can never overwrite, or be mixed with,
    the files of an earlier or interrupted run."""
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IOError("%s exists and is not a directory" % path)
        if os.listdir(path):
            raise IOError("output directory %s is not empty; use a fresh one" % path)
    else:
        os.makedirs(path)
    return path


RESULTS_NAME = "results.json"


def write_run(out_dir: str, results: Mapping, companions: Mapping[str, Iterable[Mapping]]) -> str:
    """Write a run into a fresh directory: every companion JSONL first, then
    results.json LAST, recording each companion's name and SHA-256. results.json
    is the completion marker: a directory without it is an incomplete run."""
    prepare_fresh_dir(out_dir)
    files = {}
    for name, rows in companions.items():
        if os.sep in name or name == RESULTS_NAME or not name.endswith(".jsonl"):
            raise ValueError("companion %r must be a plain *.jsonl file name" % name)
        files[name] = atomic_write_jsonl(os.path.join(out_dir, name), rows)
    payload = dict(results)
    payload["output_files"] = files
    return atomic_write_json(os.path.join(out_dir, RESULTS_NAME), payload)


def verify_run(out_dir: str) -> Dict[str, object]:
    """A run directory is complete and consistent: results.json exists, every
    companion it lists exists with the recorded SHA-256, no other files or
    leftover .partial files are present."""
    res_path = os.path.join(out_dir, RESULTS_NAME)
    if not os.path.exists(res_path):
        raise IOError("%s has no %s: incomplete run" % (out_dir, RESULTS_NAME))
    with open(res_path, encoding="utf-8") as fh:
        results = json.load(fh)
    files = results.get("output_files")
    if not isinstance(files, dict):
        raise IOError("results.json lists no output files")
    for name, sha in files.items():
        p = os.path.join(out_dir, name)
        if not os.path.exists(p) or sha256_file(p) != sha:
            raise IOError("%s is missing or does not match results.json" % name)
    extra = set(os.listdir(out_dir)) - set(files) - {RESULTS_NAME}
    if extra:
        raise IOError("unexpected files in %s: %s" % (out_dir, sorted(extra)))
    return results
