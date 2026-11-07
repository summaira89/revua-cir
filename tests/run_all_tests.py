"""Run every test function in tests/test_*.py, each in a fresh interpreter, and
write a reproducible test manifest (tests/test_report.json):

  * per test: status (pass / skip / fail / infra_error / needs_fixture) and,
    for anything but pass, the exception type, message and traceback;
  * per module and overall: counts by status;
  * the code revision tested: SHA-256 of every scripts/*.py and tests/*.py;
  * the Python, NumPy and platform versions.

Tests that take arguments need a pytest fixture and are listed, not run.
Exit status 0 only if nothing failed or errored.

    python tests/run_all_tests.py
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from mutation_check import run_test  # noqa: E402  (same isolated, classified runner)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_names(path: Path):
    """(name, needs_fixture) for each top-level test function, without importing."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [(n.name, bool(n.args.args)) for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]


def has_testcase_classes(path: Path) -> bool:
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(isinstance(n, ast.ClassDef) and any(
        (isinstance(b, ast.Attribute) and b.attr == "TestCase")
        or (isinstance(b, ast.Name) and b.id == "TestCase") for b in n.bases)
        for n in tree.body)


def run_unittest_module(path: Path) -> dict:
    """Run a unittest-style module in a fresh interpreter; per-test outcomes."""
    import subprocess
    code = r'''
import json, sys, unittest
loader = unittest.TestLoader()
suite = loader.discover(start_dir=sys.argv[1], pattern=sys.argv[2])
res = unittest.TestResult()
suite.run(res)
out = {"run": res.testsRun,
       "fail": [[str(t), tb[-2000:]] for t, tb in res.failures + res.errors],
       "skip": [[str(t), r] for t, r in res.skipped]}
print("@@RESULT@@" + json.dumps(out))
'''
    proc = subprocess.run([sys.executable, "-c", code, str(path.parent), path.name],
                          capture_output=True, text=True, cwd=str(ROOT))
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
    return {"run": 0, "fail": [["<module>", proc.stderr[-2000:]]], "skip": []}


def main() -> int:
    import numpy
    report = {"python": sys.version.split()[0], "numpy": numpy.__version__,
              "platform": platform.platform(),
              "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
                              for p in sorted((ROOT / "scripts").glob("*.py"))
                              + sorted((ROOT / "tests").glob("*.py"))},
              "modules": {}}
    totals = {}
    for mod in sorted((ROOT / "tests").glob("test_*.py")):
        rows, counts = {}, {}
        for name, needs_fixture in test_names(mod):
            res = ({"status": "needs_fixture"} if needs_fixture
                   else run_test(ROOT, mod.name, name))
            rows[name] = res
            counts[res["status"]] = counts.get(res["status"], 0) + 1
        if has_testcase_classes(mod):
            u = run_unittest_module(mod)
            failed = {t for t, _ in u["fail"]}
            skipped = {t for t, _ in u["skip"]}
            rows["unittest"] = u
            counts["fail"] = counts.get("fail", 0) + len(failed)
            counts["skip"] = counts.get("skip", 0) + len(skipped)
            counts["pass"] = counts.get("pass", 0) + u["run"] - len(failed) - len(skipped)
        if not rows or sum(counts.values()) == 0:
            # a module from which nothing was collected is an error, never a silent zero
            rows["<collection>"] = {"status": "infra_error", "message": "no tests collected"}
            counts = {"infra_error": 1}
        for status, c in counts.items():
            totals[status] = totals.get(status, 0) + c
        report["modules"][mod.name] = {"counts": counts, "tests": rows}
        print("%-40s %s" % (mod.name, ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    report["totals"] = totals
    out = ROOT / "tests" / "test_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nTOTAL: %s" % ", ".join("%s %d" % kv for kv in sorted(totals.items())))
    print("manifest written to %s" % out)
    return 1 if totals.get("fail") or totals.get("infra_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
