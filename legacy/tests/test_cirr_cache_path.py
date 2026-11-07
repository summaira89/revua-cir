"""Tests for lumiere/cirr_cache_path.sh, run through bash on temporary roots.
Runs without pytest; needs `bash` on PATH (Git Bash on Windows, any Linux)."""
import os
import shutil
import subprocess
import tempfile

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lumiere",
                      "cirr_cache_path.sh")
NAME = "rv_score_cache_fc_aep_rv.jsonl"


def _cache(path, lines=4181):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as fh:
        for _ in range(lines):
            fh.write("x" * 300 + "\n")          # > 1 MB in total, as the search requires


def _run(root, extra_path=None):
    env = dict(os.environ, SEARCH_ROOT=root.replace("\\", "/"))
    if extra_path:
        env["PATH"] = extra_path + os.pathsep + env["PATH"]
    return subprocess.run(["bash", SCRIPT.replace("\\", "/")], capture_output=True, text=True,
                          env=env)


def _tmp():
    return tempfile.mkdtemp()


def test_exactly_one_valid_cache_is_printed():
    root = _tmp()
    try:
        _cache(os.path.join(root, "a", NAME))
        r = _run(root)
        assert r.returncode == 0 and r.stdout.strip().endswith(NAME), r.stderr
    finally:
        shutil.rmtree(root)


def test_zero_or_several_matches_fail():
    root = _tmp()
    try:
        assert _run(root).returncode == 1                      # none
        _cache(os.path.join(root, "a", NAME))
        _cache(os.path.join(root, "b", NAME))
        r = _run(root)
        assert r.returncode == 1 and "found 2" in r.stderr     # several, all listed
    finally:
        shutil.rmtree(root)


def test_wrong_record_count_fails():
    root = _tmp()
    try:
        _cache(os.path.join(root, "a", NAME), lines=4500)
        assert _run(root).returncode == 1
    finally:
        shutil.rmtree(root)


def test_search_failure_after_one_match_fails():
    """Audit 3/3: one path printed, then find exits non-zero -> must fail."""
    root, fake = _tmp(), _tmp()
    try:
        good = os.path.join(root, "a", NAME)
        _cache(good)
        with open(os.path.join(fake, "find"), "w", newline="\n") as fh:
            fh.write("#!/bin/bash\necho '%s'\necho 'find: permission denied' >&2\nexit 1\n"
                     % good.replace("\\", "/"))
        os.chmod(os.path.join(fake, "find"), 0o755)
        r = _run(root, extra_path=fake)
        assert r.returncode == 1 and "failed or was incomplete" in r.stderr, r.stderr
    finally:
        shutil.rmtree(root)
        shutil.rmtree(fake)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print("ok  " + f.__name__)
    print("\n%d tests passed" % len(fns))
