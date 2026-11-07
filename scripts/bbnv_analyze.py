"""BBNV Phase D analysis of scored validation windows (v3.1 §13, §14). CPU.

Reads a verified `bbnv_score.py windows` run and computes, without any GPU
work and without re-scoring:

  arms (§14.1, exploratory)  for every arm (F, I, F+I, F+F', I+I', and the
       induced-inverse arm Ind) and every seed pair: report-half macro
       within-window AUROC and bounded-fusion R@1, plus the paired contrasts
       F+I - F, F+I - F+F' and I - Ind (query bootstrap for seed pair 0);
  gates (§14.2)  for every gate (base-only, inverse-only, forward-only,
       two-score stacker, delta, unsigned, same-direction): a 1:2 weighted L2
       logistic gate on the deployed bounded-fusion proposal, C chosen by
       5-fold stratified CV utility on the TUNE half, accepting when the
       weighted probability > 0.5; conservative controls (accept-none,
       base-margin threshold, base-only) with the strongest selected by the
       same CV utility; everything reported on the REPORT half, each gate
       minus the strongest control by paired query bootstrap.

Every arm uses V_H = mean_h z_W(s_h) + GAMMA * z_W(edit_cos) (bbnv_core.integrate)
and the frozen fusion z(base) + tanh(z(V)) (fusion_suite.r_bounded, lambda 1).
The forward evidence is s_F = raw - lambda* * marginal with the frozen
lambda* recorded by the scoring run (§12-13).

Usage
-----
  python scripts/bbnv_analyze.py --windows ../bbnv_runs/windows_cirr_v1 --out-dir ../bbnv_runs/analysis_cirr_v1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bbnv_core as core  # noqa: E402
import bbnv_manifest as man  # noqa: E402
import bbnv_provenance as prov  # noqa: E402
import bbnv_train as trn  # noqa: E402
import gate_suite as gs  # noqa: E402
from fusion_suite import r_bounded  # noqa: E402

SCHEMA = "bbnv-analysis/1"
WINDOWS_SCHEMA = "bbnv-windows/1"
GAMMA = 1.0
FUSION_LAMBDA = 1.0
ARMS = ("F", "I", "F+I", "F+F'", "I+I'", "Ind")
CONTRASTS = (("F+I", "F"), ("F+I", "F+F'"), ("I", "Ind"))
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
CV_SEED = 20261023
BREAK_WEIGHT = 2.0
GATES = {"base_only": (), "inverse_only": ("IE",), "forward_only": ("F", "S"),
         "stacker": ("IE", "Ind", "S"), "delta": ("Delta", "S"), "unsigned": ("absDelta", "S"),
         "same_direction": ("Same", "S")}
CONTROLS = ("accept_none", "base_margin", "base_only")


class AnalysisError(ValueError):
    pass


# --------------------------------------------------------------- inputs --
def load_windows(windows_dir: str) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    res = prov.verify_run(windows_dir)
    if res.get("schema") != WINDOWS_SCHEMA or res.get("smoke") is not False:
        raise AnalysisError("%s is not a full windows run of schema %s" % (windows_dir, WINDOWS_SCHEMA))
    rows = man.read_jsonl(os.path.join(windows_dir, "windows.jsonl"))
    if len(rows) != res["n_queries"] or [r["query"] for r in rows] != list(range(len(rows))):
        raise AnalysisError("windows.jsonl does not hold every query in order")
    return res, rows


def signals(row: Mapping[str, object], lam: float) -> Dict[str, np.ndarray]:
    """Per-head window signals: F<s> forward evidence (raw - lam * marginal),
    I<s> learned inverse, Ind<s> induced inverse, S<s> marginal."""
    out = {}
    for s in trn.SEEDS:
        f = row["heads"]["F%d" % s]
        out["F%d" % s] = core.forward_lambda_score(np.asarray(f["raw"]), np.asarray(f["marginal"]), lam)
        out["Ind%d" % s] = np.asarray(f["induced"], dtype=np.float64)
        out["S%d" % s] = np.asarray(f["marginal"], dtype=np.float64)
        out["I%d" % s] = np.asarray(row["heads"]["I%d" % s]["ie"], dtype=np.float64)
    return out


def arm_components(arm: str, pair: int) -> List[str]:
    """Head signals of an arm on seed pair `pair` (§13): (F_s, I_s) pairs, and
    the cyclic second head (s, s+1 mod 3) for F+F' and I+I'."""
    nxt = trn.SEEDS[(trn.SEEDS.index(pair) + 1) % len(trn.SEEDS)]
    table = {"F": ["F%d" % pair], "I": ["I%d" % pair], "F+I": ["F%d" % pair, "I%d" % pair],
             "F+F'": ["F%d" % pair, "F%d" % nxt], "I+I'": ["I%d" % pair, "I%d" % nxt], "Ind": ["Ind%d" % pair]}
    if arm not in table:
        raise AnalysisError("unknown arm %r" % (arm,))
    return table[arm]


def arm_score(sig: Mapping[str, np.ndarray], arm: str, pair: int, edit_cos: Sequence[float]) -> np.ndarray:
    return core.integrate([sig[c] for c in arm_components(arm, pair)], np.asarray(edit_cos, dtype=np.float64), GAMMA)


def fused_top(base: Sequence[float], verifier: Sequence[float]) -> int:
    return gs._top(r_bounded(list(base), list(verifier), FUSION_LAMBDA))


# ----------------------------------------------------------------- arms --
def arm_outcomes(rows: Sequence[Mapping[str, object]], lam: float) -> Dict[str, object]:
    """Per query: AUROC (None without a target in the window) and fused hit
    for every arm and seed pair; base and deployed hits."""
    out = {"auroc": {}, "hit": {}, "base_hit": [], "deployed_hit": []}
    for arm in ARMS:
        for s in trn.SEEDS:
            out["auroc"][(arm, s)], out["hit"][(arm, s)] = [], []
    for row in rows:
        tp = row["target_pos"]
        labels = np.arange(len(row["window"])) == tp
        sig = signals(row, lam)
        out["base_hit"].append(int(gs._top(row["base_score"]) == tp))
        out["deployed_hit"].append(int(fused_top(row["base_score"], row["verification_score"]) == tp))
        for arm in ARMS:
            for s in trn.SEEDS:
                v = arm_score(sig, arm, s, row["edit_cos"])
                out["auroc"][(arm, s)].append(core.pairwise_auroc(v, labels) if tp >= 0 else None)
                out["hit"][(arm, s)].append(int(fused_top(row["base_score"], v.tolist()) == tp))
    return out


def summarise_arms(outc: Mapping[str, object], idx: Sequence[int], n_rep: int) -> Dict[str, object]:
    pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
    arms = {}
    for (arm, s), au in outc["auroc"].items():
        m, n_def = core.macro_auroc(pick(au))
        hits = pick(outc["hit"][(arm, s)])
        arms["%s|%d" % (arm, s)] = {"macro_auroc": m, "n_defined": n_def, "r1": 100.0 * sum(hits) / len(hits)}
    contrasts = {}
    for a, b in CONTRASTS:
        per_pair = {}
        for s in trn.SEEDS:
            pm = core.paired_macro_auroc({i: outc["auroc"][(a, s)][i] for i in idx},
                                         {i: outc["auroc"][(b, s)][i] for i in idx})
            per_pair[s] = pm["difference"]
        xa = [outc["auroc"][(a, 0)][i] for i in idx if outc["auroc"][(a, 0)][i] is not None]
        xb = [outc["auroc"][(b, 0)][i] for i in idx if outc["auroc"][(a, 0)][i] is not None]
        contrasts["%s - %s" % (a, b)] = {
            "auroc_diff_by_pair": per_pair, "mean": float(np.mean(list(per_pair.values()))),
            "range": [min(per_pair.values()), max(per_pair.values())],
            "pair0_bootstrap": gs.paired_mean_bootstrap(xa, xb, n_rep=n_rep)}
    base, dep = pick(outc["base_hit"]), pick(outc["deployed_hit"])
    return {"arms": arms, "contrasts": contrasts, "n_queries": len(idx),
            "base_r1": 100.0 * sum(base) / len(base), "deployed_r1": 100.0 * sum(dep) / len(dep)}


# ---------------------------------------------------------------- gates --
def _signal_pair(x: np.ndarray, f: int, b: int) -> Tuple[float, float]:
    """(x_f - window mean, x_f - x_b): centred at the proposal, and its margin over base."""
    return float(x[f] - x.mean()), float(x[f] - x[b])


def gate_views(rows: Sequence[Mapping[str, object]], lam: float, seed: int) -> List[Dict[str, object]]:
    """Outcome and features of the deployed proposal for every query. Signals
    come from seed `seed` (the same-direction control uses I_s - I_{s+1})."""
    nxt = trn.SEEDS[(trn.SEEDS.index(seed) + 1) % len(trn.SEEDS)]
    views = []
    for row in rows:
        tp = row["target_pos"]
        v = gs.query_view({"base_score": row["base_score"], "verification_score": row["verification_score"]},
                          {tp} if tp >= 0 else set(), r_bounded, FUSION_LAMBDA)
        f = fused_top(row["base_score"], row["verification_score"])
        b = gs._top(row["base_score"])
        sig = signals(row, lam)
        ie, ind = sig["I%d" % seed], sig["Ind%d" % seed]
        delta = ie - ind
        raw_sig = {"IE": ie, "Ind": ind, "S": sig["S%d" % seed], "F": sig["F%d" % seed], "Delta": delta,
                   "absDelta": np.abs(delta - delta.mean()), "Same": ie - sig["I%d" % nxt]}
        extra = {}
        for name, x in raw_sig.items():
            extra[name + "_c"], extra[name + "_m"] = _signal_pair(x, f, b)
        v["feat"].update(extra)
        v["label"] = (1 if v["hit_fused"] and not v["hit_base"] else 0 if v["hit_base"] and not v["hit_fused"]
                      else None) if v["changed"] else None
        views.append(v)
    return views


def feature_names(gate: str) -> List[str]:
    if gate not in GATES:
        raise AnalysisError("unknown gate %r" % (gate,))
    return list(gs.FEATURES) + [n + suf for n in GATES[gate] for suf in ("_c", "_m")]


def fit_weighted_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float, iters: int = 100):
    """Weighted L2 logistic regression (Newton), features standardised on X;
    the intercept is not penalised. Returns (predict_proba, params)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    R = np.eye(Z.shape[1]) / C
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -500, 500)))
        g = Z.T @ (w * (p - y)) + R @ beta
        H = (Z * (w * p * (1 - p))[:, None]).T @ Z + R
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-10:
            break

    def predict(Xn: np.ndarray) -> np.ndarray:
        Zn = np.hstack([np.ones((len(Xn), 1)), (Xn - mu) / sd])
        return 1.0 / (1.0 + np.exp(-np.clip(Zn @ beta, -500, 500)))
    return predict, {"beta": beta.tolist(), "mean": mu.tolist(), "sd": sd.tolist(), "C": C}


def query_utility(view: Mapping[str, object], accept: bool) -> float:
    """u = 1[rescue] - 2 * 1[break] for an accepted changed proposal; 0 otherwise."""
    if not (view["changed"] and accept):
        return 0.0
    d = view["hit_fused"] - view["hit_base"]
    return 1.0 if d > 0 else -BREAK_WEIGHT if d < 0 else 0.0


def stratified_folds(labels: Sequence[int], k: int, seed: int) -> List[List[int]]:
    """k folds of positions: each class is shuffled (seeded) and dealt
    round-robin, the deal continuing across classes, so fold sizes differ by
    at most one and so does each class's count per fold."""
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(k)]
    j = 0
    for cls in (0, 1):
        for i in rng.permutation([i for i, y in enumerate(labels) if y == cls]).tolist():
            folds[j % k].append(i)
            j += 1
    return [sorted(f) for f in folds]


def _matrix(views: Sequence[Mapping[str, object]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[v["feat"][n] for n in names] for v in views], dtype=np.float64)


def _fit_gate(train: Sequence[Mapping[str, object]], names: Sequence[str], C: float):
    """Fit on the non-neutral changed views of `train`; returns accept(views) -> bools."""
    lab = [v for v in train if v["label"] is not None]
    if len({v["label"] for v in lab}) < 2:
        raise AnalysisError("a gate fit needs both rescues and breaks")
    y = np.asarray([v["label"] for v in lab], dtype=np.float64)
    w = np.where(y == 1, 1.0, BREAK_WEIGHT)
    predict, params = fit_weighted_logistic(_matrix(lab, names), y, w, C)
    return (lambda vs: [bool(p > 0.5) for p in predict(_matrix(vs, names))]), params


def _margin_gate(train: Sequence[Mapping[str, object]]):
    t, _ = gs.choose_threshold(list(train), lambda v: v["feat"]["base_margin"], "le", 1.0)
    return (lambda vs: [v["feat"]["base_margin"] <= t for v in vs]), {"threshold": t}


def cv_utility(views: Sequence[Mapping[str, object]], make_gate, k: int = N_FOLDS, seed: int = CV_SEED) -> float:
    """Mean held-out utility per query over k stratified folds of the
    non-neutral changed views (neutral and unchanged views carry u = 0
    whatever the gate does, so they enter only the training side)."""
    lab = [i for i, v in enumerate(views) if v["label"] is not None]
    folds = stratified_folds([views[i]["label"] for i in lab], k, seed)
    total = 0.0
    for f in folds:
        held = {lab[j] for j in f}
        accept = make_gate([v for i, v in enumerate(views) if i not in held])
        hv = [views[i] for i in sorted(held)]
        total += sum(query_utility(v, a) for v, a in zip(hv, accept(hv)))
    return total / len(views)


def select_C(views: Sequence[Mapping[str, object]], names: Sequence[str]) -> Tuple[float, Dict[str, float]]:
    scores = {C: cv_utility(views, lambda tr, C=C: _fit_gate(tr, names, C)[0]) for C in C_GRID}
    best = max(scores.values())
    return min(C for C, s in scores.items() if s == best), {str(C): s for C, s in scores.items()}


def evaluate_gate(views: Sequence[Mapping[str, object]], accept_flags: Sequence[bool]) -> Dict[str, object]:
    u = [query_utility(v, a) for v, a in zip(views, accept_flags)]
    hits = [v["hit_fused"] if (v["changed"] and a) else v["hit_base"] for v, a in zip(views, accept_flags)]
    acc = [v for v, a in zip(views, accept_flags) if v["changed"] and a]
    return {"mean_utility": float(np.mean(u)), "r1": 100.0 * sum(hits) / len(hits), "accepted": len(acc),
            "rescues": sum(1 for v in acc if v["hit_fused"] > v["hit_base"]),
            "breaks": sum(1 for v in acc if v["hit_fused"] < v["hit_base"]), "per_query_u": u}


def run_gates(tune: Sequence[Mapping[str, object]], report: Sequence[Mapping[str, object]],
              n_rep: int) -> Dict[str, object]:
    """Controls and gates (§14.2): C by CV on tune, final fit on all of tune,
    evaluation on report; each gate minus the strongest control."""
    makers = {"accept_none": lambda tr: (lambda vs: [False] * len(vs)),
              "base_margin": lambda tr: _margin_gate(tr)[0]}
    fitted = {}
    for gate in GATES:
        names = feature_names(gate)
        C, cv = select_C(tune, names)
        accept, params = _fit_gate(tune, names, C)
        fitted[gate] = {"accept": accept, "params": params, "cv_by_C": cv, "cv_utility": cv[str(C)]}
    control_cv = {"accept_none": 0.0, "base_margin": cv_utility(tune, makers["base_margin"]),
                  "base_only": fitted["base_only"]["cv_utility"]}
    strongest = max(CONTROLS, key=lambda c: (control_cv[c], -CONTROLS.index(c)))
    final = {"accept_none": [False] * len(report),
             "base_margin": _margin_gate(tune)[0](report)}
    for gate in GATES:
        final[gate] = fitted[gate]["accept"](report)
    evals = {name: evaluate_gate(report, flags) for name, flags in final.items()}
    ref = evals[strongest]["per_query_u"]
    out = {"strongest_control": strongest, "control_cv_utility": control_cv, "gates": {}}
    for name, ev in evals.items():
        entry = {k: v for k, v in ev.items() if k != "per_query_u"}
        if name in fitted:
            entry.update(C=fitted[name]["params"]["C"], cv_by_C=fitted[name]["cv_by_C"],
                         params=fitted[name]["params"])
        if name != strongest:
            entry["minus_strongest_control"] = gs.paired_mean_bootstrap(ev["per_query_u"], ref, n_rep=n_rep)
        out["gates"][name] = entry
    return out


# ------------------------------------------------------------------ main --
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=gs.N_BOOTSTRAP)
    args = ap.parse_args(argv)
    if args.n_boot < 100:
        ap.error("--n-boot must be at least 100")
    try:
        prov.prepare_fresh_dir(args.out_dir)
    except IOError as exc:
        ap.error(str(exc))
    t0 = time.time()
    res, rows = load_windows(args.windows)
    if res["target"] != "cirr_val" or res.get("split_seed") is None:
        raise AnalysisError("the gate and the tune/report analysis are declared for CIRR validation only")
    lam = float(res["lambda_star_frozen"])
    tune = [i for i, r in enumerate(rows) if r["split"] == "tune"]
    report = [i for i, r in enumerate(rows) if r["split"] == "report"]
    if not tune or not report or len(tune) + len(report) != len(rows):
        raise AnalysisError("every query must be in exactly one of tune / report")
    outc = arm_outcomes(rows, lam)
    arms = summarise_arms(outc, report, args.n_boot)
    gates = {}
    for s in trn.SEEDS:
        views = gate_views(rows, lam, s)
        gates["seed%d" % s] = run_gates([views[i] for i in tune], [views[i] for i in report], args.n_boot)
        print("[analyze] gates seed %d done (%.0fs)" % (s, time.time() - t0), flush=True)
    results = {"schema": SCHEMA, "amendment": "v3.1 §14", "lambda_star": lam, "gamma": GAMMA,
               "fusion": "bounded, lambda %g" % FUSION_LAMBDA, "n_tune": len(tune), "n_report": len(report),
               "report_half": arms, "gates": gates, "deployed_seed": core.DEPLOYED_SEED,
               "inputs": {"windows_results_sha256": prov.sha256_file(os.path.join(args.windows, prov.RESULTS_NAME))},
               "manifest": {"schema": prov.MANIFEST_SCHEMA, "script": "bbnv_analyze.py", "args": vars(args),
                            "source_sha256": prov.source_hashes(), "runtime": prov.runtime_info(None)},
               "wall_seconds": round(time.time() - t0, 1)}
    prov.write_run(args.out_dir, results, {})
    prov.verify_run(args.out_dir)
    print("[analyze] wrote and verified %s in %.0fs" % (args.out_dir, time.time() - t0), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
