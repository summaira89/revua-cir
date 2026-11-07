"""
Evaluation metrics (Section 6 of the paper framework), plus the diagnostic
metrics from Section 6.1 (verification agreement rate, ECE, latency is
timed in evaluate.py directly).

  * recall_at_k          -- FashionIQ / CIRR / LaSCo (single-target Recall@K)
  * mean_average_precision_at_k -- CIRCO (multi-ground-truth mAP@K)
  * expected_calibration_error  -- Uncertainty Gate calibration (Ablation #9)

CIRR's subset/group-recall protocol (six-member image set including the
reference, 5 non-reference candidates ranked after excluding it) is computed
directly inside evaluate.py's per-query loop, not via a helper here -- the
old `group_recall_at_k` helper that used to live in this module computed it
from an already-ranked, globally-truncated list, which could silently miss
a group's true best candidate; removed 2026-09-09 (code-audit finding #2)
once it had been fully superseded and confirmed dead (no remaining callers).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def recall_at_k(ranked_indices: torch.Tensor, target_indices: torch.Tensor, ks: Sequence[int]) -> Dict[str, float]:
    """ranked_indices: (B, N) gallery indices sorted by descending score.
    target_indices: (B,) the single correct gallery index per query.
    """
    ranked = ranked_indices.cpu()
    targets = target_indices.cpu().unsqueeze(1)
    hits = ranked == targets  # (B, N) bool
    out = {}
    for k in ks:
        k = min(k, ranked.size(1))
        out[f"recall_at_{k}"] = hits[:, :k].any(dim=1).float().mean().item() * 100
    return out


def mean_average_precision_at_k(ranked_indices: torch.Tensor, relevant_sets: List[set], ks: Sequence[int]) -> Dict[str, float]:
    """CIRCO-style mAP@K with multiple ground-truth targets per query.
    relevant_sets[i]: set of gallery indices that are valid targets for query i.
    """
    ranked = ranked_indices.cpu().numpy()
    out = {}
    for k in ks:
        aps = []
        for i in range(ranked.shape[0]):
            rel = relevant_sets[i]
            if not rel:
                continue
            hits, num_correct = 0, 0.0
            for rank, idx in enumerate(ranked[i, :k], start=1):
                if int(idx) in rel:
                    hits += 1
                    num_correct += hits / rank
            denom = min(len(rel), k)
            aps.append(num_correct / denom if denom > 0 else 0.0)
        out[f"map_at_{k}"] = float(np.mean(aps)) * 100 if aps else 0.0
    return out


def verification_agreement_rate(verification_scores: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of top-1 candidates whose reverse-verification score clears the calibrated threshold.

    Kept for backward compatibility with existing logs/plots. NOT a
    discrimination measure: `top_idx[:, 0]` is always the base ranker's own
    best guess, so this population is pre-filtered toward "plausible"
    regardless of whether RV can tell correct from incorrect. A
    well-calibrated *or* completely useless RV both tend toward 1.0 here --
    see `verification_auroc` (pooled ROC-AUC) or `verification_pairwise_
    win_rate` (per-query matched-pair comparison) below for the metrics
    that actually test the claim (fixed stale cross-reference 2026-09-09,
    code-audit finding #3 -- this docstring still pointed at
    `verification_discrimination_auc`, the pre-2026-09-09 name, which no
    longer exists in this module).
    """
    return (verification_scores >= threshold).float().mean().item()


def verification_pairwise_win_rate(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    """P(RV score of the true target > RV score of ITS OWN query's paired wrong
    candidate), averaged over queries -- exactly one positive/negative pair
    per query, matched pairwise, NOT pooled across the dataset.

    Renamed from `verification_discrimination_auc` (fixed 2026-09-09,
    code-audit finding B.8): this is NOT the same statistic as standard
    pooled ROC-AUC, which compares EVERY positive score in the dataset
    against EVERY negative score (all n_pos * n_neg pairs), not just the n
    matched pairs computed here. The two coincide only in degenerate cases
    (e.g. every query sharing an identical, fixed negative distribution);
    in general the matched-pair statistic can differ meaningfully from
    pooled AUROC, so calling it "AUC" without qualification overstated what
    it measures. Use `verification_auroc` below for the real
    pooled AUROC number a paper table should report; this function is kept
    for anyone specifically wanting the per-query matched-pair view (it
    answers "for a typical single query, does RV prefer its own true
    target over its own top-1-incorrect candidate", a different, also
    legitimate question from AUROC's "pooled over the whole population, can
    RV separate the positive and negative score distributions at all").
    0.5 = chance, 1.0 = perfect separation.
    """
    pos = positive_scores.detach().cpu().numpy()
    neg = negative_scores.detach().cpu().numpy()
    n = min(len(pos), len(neg))
    if n == 0:
        return float("nan")
    pos, neg = pos[:n], neg[:n]
    wins = (pos > neg).sum() + 0.5 * (pos == neg).sum()
    return float(wins / n)


def verification_auroc(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    """True pooled ROC-AUC (added 2026-09-09, code-audit finding B.8): treats
    every positive score as a "verified" (label=1) sample and every negative
    score as an "unverified" (label=0) sample, computed via the standard
    Mann-Whitney-U identity AUC = P(pos > neg) over ALL positive-negative
    pairs in the pool (not just one matched pair per query, unlike
    `verification_pairwise_win_rate` above) -- equivalent to
    sklearn.metrics.roc_auc_score(labels, scores) on the same pooled
    positive+negative scores, without adding a scikit-learn dependency for
    one metric. This is the number a paper claiming "AUROC" should report.
    0.5 = chance, 1.0 = perfect separation.
    """
    pos = positive_scores.detach().cpu().numpy()
    neg = negative_scores.detach().cpu().numpy()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-sum form of the Mann-Whitney U statistic: rank all pooled scores
    # (ties get the average rank), sum the ranks of the positive samples,
    # then AUC = (sum_of_positive_ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    # O((n_pos+n_neg) log(n_pos+n_neg)) via a single argsort, vs. O(n_pos*n_neg)
    # for a naive all-pairs comparison -- matters here since a full-gallery
    # CIRR eval pass can pool thousands of queries' pos/neg scores.
    pooled = np.concatenate([pos, neg])
    order = pooled.argsort(kind="stable")
    ranks = np.empty(len(pooled), dtype=np.float64)
    ranks[order] = np.arange(1, len(pooled) + 1, dtype=np.float64)
    # Average tied ranks (standard Mann-Whitney tie correction).
    sorted_scores = pooled[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    sum_pos_ranks = ranks[:n_pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    """Standard ECE: bins predictions by confidence, compares mean confidence to observed accuracy per bin."""
    confidences = confidences.detach().cpu().numpy()
    correct = correct.detach().cpu().numpy().astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def summarize(metric_dicts: Sequence[Dict[str, float]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d in metric_dicts:
        out.update(d)
    return out
