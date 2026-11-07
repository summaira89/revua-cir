"""Figures for the IVC manuscript, drawn from the locked results only (no hand-copied numbers).

  fig_overview.pdf  study design: candidate generation -> evidence -> integration -> evaluation
  fig_coverage.pdf  coverage versus within-window discrimination (CIRR held-out half, §22)
  fig_effects.pdf   paired incremental effects with 95% CIs (§20, §21, §25, §26c, §27)

Every input is hash-checked against its lock before use; vector output with TrueType fonts.

  (from revua_cir/) python scripts/make_ivc_figures.py --root .. --out ../manuscript/ivc/figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

LOCKED = {
    "s20": ("revision_work/followups_2026-11-03/results.json", "a8ef5f658dad278ef66843c79b59365db20cabcb3e9041b025c57a53a357dd30"),
    "s21": ("revision_work/followups21_2026-11-03/results.json", "e738bac8bd560a380fbe206551e367efafa92123cd161477a65a2adccb0067f6"),
    "s22": ("revision_work/followups22_2026-11-03/results.json", "516c1aea7b504abfce4ed27208f3f08482028c58a2cfda66f49dc19e295b6a7c"),
    "s25a": ("revision_work/followups25a_2026-11-04/results.json", "469628ab4bc609a42e7fef34ce6644567c6f8059131ee1e96801b79f151f8bcf"),
    "s25c": ("revision_work/followups25c_2026-11-04/results.json", "df703d87f098ebec4684b339b61fc9dff769cb3ee697f3808981431a1e930719"),
    "s26c": ("revision_work/followups26c_fc_2026-11-04/results.json", "7012dd9aef2608e34a58bf1a3265156721dc3c92e67d3aef3b2e46c25295e436"),
    "s27": ("revision_work/followups27_2026-11-04/results.json", "01cc116cbb6413417c0c46dd7e1574c6f07291371fe23fb201cf34ed0da3b79f"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # reference categorical slots 1-3, fixed order

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(root):
    out = {}
    for k, (p, want) in LOCKED.items():
        path = os.path.join(root, p)
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            raise SystemExit("ABORT: %s hash %s != lock %s" % (p, got, want))
        with open(path, encoding="utf-8") as fh:
            out[k] = json.load(fh)
    return out


# ---------------------------------------------------------------------------
def fig_overview(path):
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 37)
    ax.axis("off")
    cols = [
        (0.5, ["Query"], ["reference", "image $I_r$", "modification", "text $t$"]),
        (20.5, ["Candidate", "generation"], ["composed query", "CLIP4Cir RN50x4", "reference-image", "similarity",
                                             "top 10, reference", "excluded"]),
        (40.5, ["Second-stage", "evidence"], ["text cosine $T^{(e)}$:", "pretrained, FC-only,", "deployed, CLIP4Cir",
                                               "learned", "verifier $R$"]),
        (60.5, ["Integration"], ["bounded score", "fusion, weights", "from CIRR", "tuning half;", "acceptance",
                                 "policies"]),
        (80.5, ["Evaluation"], ["R@1, held-out", "half; utility", "at $\\alpha=1$; frozen", "transfer to", "CIRCO mAP@5"]),
    ]
    w = 17.5
    for x, title, lines in cols:
        ax.add_patch(FancyBboxPatch((x, 1), w, 34, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc="#f4f3f0", ec=MUTED, lw=0.8))
        for j, tl in enumerate(title):
            ax.text(x + w / 2, 32 - 3.2 * j, tl, ha="center", va="center", fontsize=7.5, color=INK, weight="bold")
        for j, ln in enumerate(lines):
            ax.text(x + w / 2, 23.5 - 3.6 * j, ln, ha="center", va="center", fontsize=6.6, color=INK)
    for (x0, _, _), (x1, _, _) in zip(cols, cols[1:]):
        ax.add_patch(FancyArrowPatch((x0 + w + 0.6, 18), (x1 - 0.6, 18), arrowstyle="-|>", mutation_scale=9,
                                     lw=0.9, color=MUTED))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_coverage(d, path):
    W = d["s22"]["W"]["report"]
    wins = [("Composed-query window", W["W_B"]), ("Reference-image window", W["W_I"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"width_ratios": [1, 2.2]})
    names = [n for n, _ in wins]
    cov = [100.0 * w["coverage"] for _, w in wins]
    a1.barh([1, 0], cov, color=[BLUE, ORANGE], height=0.55)
    for y, v in zip([1, 0], cov):
        a1.text(v + 2, y, "%.1f%%" % v, va="center", fontsize=7.5, color=INK)
    a1.set_yticks([1, 0], names)
    a1.set_xlim(0, 110)
    a1.set_xlabel("Target in window (%)")
    a1.set_title("(a) Coverage", loc="left", fontsize=8.5, color=INK)
    a1.grid(axis="x", color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    scorings = [("B", "composed score $B$", BLUE), ("T", "text cosine $T$", ORANGE), ("BT", "$B+T$ fused", AQUA)]
    x0 = [0, 1]
    bw = 0.25
    for j, (key, lab, col) in enumerate(scorings):
        vals = [w["scorings"][key]["r1_all"] for _, w in wins]
        xs = [x + (j - 1) * (bw + 0.02) for x in x0]
        a2.bar(xs, vals, width=bw, color=col, label=lab)
        for x, v in zip(xs, vals):
            a2.text(x, v + 0.8, "%.1f" % v, ha="center", fontsize=7, color=INK)
    a2.set_xticks(x0, names)
    a2.set_ylabel("R@1, all queries")
    a2.set_ylim(0, 55)
    a2.set_title("(b) End-to-end R@1 after reranking each window", loc="left", fontsize=8.5, color=INK)
    a2.grid(axis="y", color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    a2.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _row(label, c):
    return label, c["delta_pp"], c["ci_low"], c["ci_high"]


def fig_effects(d, path):
    t20 = d["s20"]["m2a"]["signals"]["T"]
    q1 = d["s21"]["A"]["contrasts"]["Q1"]["vs_BT"]
    a25 = d["s25a"]["A"]["report"]["contrasts"]
    c25 = d["s25c"]["V"]["report"]["contrasts"]
    e26 = d["s26c"]["E"]["report"]["fused_contrasts"]
    cirr = [
        ("Our windows", [_row("+T vs base", t20["vs_base"]),
                         _row("+T vs +verifier (deployed)", t20["vs_revua"]),
                         _row("+T+verifier vs +T", q1)]),
        ("CLIP4Cir windows", [_row("+T vs base", a25["C2"]),
                              _row("+verifier vs base", c25["C4"]),
                              _row("+T+verifier vs +T", c25["C5"]),
                              _row("Fused: T deployed vs T pretrained", e26["ours_vs_pretrained"]),
                              _row("Fused: T deployed vs T FC-only", e26["ours_vs_fc_only"])]),
    ]
    p27 = d["s27"]["primary"]

    def m5(c):
        return c["5"]["delta"], c["5"]["ci_low"], c["5"]["ci_high"]
    circo = [("+T (FC-only) vs base",) + m5(p27["primary"]["P1_T_fc_only_vs_base"]),
             ("+T (deployed) vs base",) + m5(p27["secondary"]["T_ours_vs_base"]),
             ("+verifier vs base",) + m5(p27["secondary"]["V_vs_base"]),
             ("+T+verifier vs +T",) + m5(p27["primary"]["P2_T_ours+V_vs_T_ours"])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    y, ticks, labels, heads = 0, [], [], []
    for gi, (grp, rows) in enumerate(cirr):
        col, mk = (BLUE, "o") if gi == 0 else (ORANGE, "s")
        ticks.append(y)
        labels.append(grp)
        heads.append((len(labels) - 1, col))
        y -= 1
        for lab, est, lo, hi in rows:
            a1.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round")
            a1.plot(est, y, marker=mk, color=col, ms=5, mec="#fcfcfb", mew=0.8)
            ticks.append(y)
            labels.append(lab)
            y -= 1
        y -= 0.6
    a1.axvline(0, color=MUTED, lw=0.8)
    a1.set_yticks(ticks, labels)
    a1.set_xlabel("Paired difference in R@1 (pp), 95% CI")
    a1.set_title("(a) CIRR held-out half (2091 queries)", loc="left", fontsize=8.5, color=INK)
    a1.grid(axis="x", color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    tl = a1.get_yticklabels()
    for idx, col in heads:
        tl[idx].set_color(col)
        tl[idx].set_fontweight("bold")
    for i, (lab, est, lo, hi) in enumerate(circo):
        a2.plot([lo, hi], [-i, -i], color=AQUA, lw=1.6, solid_capstyle="round")
        a2.plot(est, -i, marker="D", color=AQUA, ms=5, mec="#fcfcfb", mew=0.8)
    a2.axvline(0, color=MUTED, lw=0.8)
    a2.set_xlim(-8.5, 1.0)
    a2.set_yticks([-i for i in range(len(circo))], [r[0] for r in circo])
    a2.set_xlabel("Paired difference in mAP@5, 95% CI")
    a2.set_title("(b) CIRCO validation (220), CIRR weights", loc="left", fontsize=8.5, color=INK)
    a2.grid(axis="x", color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_graphical_abstract(d, path_pdf, path_png):
    """Graphical abstract: study design (left) and the key result (right), from the locked results."""
    va = d["s25c"]["V"]["report"]["arms"]
    p1 = d["s27"]["primary"]["primary"]["P1_T_fc_only_vs_base"]["5"]
    fig = plt.figure(figsize=(8.0, 3.4))
    ax = fig.add_axes([0.0, 0.0, 0.47, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 93, "Does a learned verifier add evidence\nbeyond a simple text cosine?", ha="center", va="center",
            fontsize=10.5, color=INK, weight="bold")
    boxes = [(78, "Query: reference image + edit text"), (62, "First stage: top-10 window\n(our retriever or CLIP4Cir)"),
             (44, "Second stage: text cosine\nvs learned verifier"), (26, "Matched fusion and tuning;\naccuracy, utility, transfer")]
    for y, label in boxes:
        ax.add_patch(FancyBboxPatch((10, y - 6.5), 80, 13, boxstyle="round,pad=0.4,rounding_size=2",
                                    fc="#f4f3f0", ec=MUTED, lw=0.9))
        ax.text(50, y, label, ha="center", va="center", fontsize=8.6, color=INK)
    for (y0, _), (y1, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((50, y0 - 7), (50, y1 + 7), arrowstyle="-|>", mutation_scale=10, lw=1.0, color=MUTED))
    ax.text(50, 8, "CIRCO with CIRR weights: FC-only text fusion\nchanges mAP@5 by \u2212%.2f" % -p1["delta"],
            ha="center", va="center",
            fontsize=8.8, color=ORANGE, weight="bold")
    b = fig.add_axes([0.58, 0.2, 0.39, 0.62])
    base = va["A1"]["r1"]
    arms = [("A4", "+ verifier"), ("A3", "+ text\ncosine"), ("A5", "+ text +\nverifier")]
    vals = [va[k]["r1"] - base for k, _ in arms]
    b.bar(range(3), vals, color=[BLUE, AQUA, BLUE], width=0.6)
    for i, v in enumerate(vals):
        b.text(i, v + 0.1, "%+.2f" % v, ha="center", fontsize=8.4, color=INK)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(range(3), [a for _, a in arms], fontsize=8.4)
    b.set_ylim(0, 4.8)
    b.set_ylabel("Change in R@1 (points)", fontsize=8.6)
    b.set_title("CLIP4Cir windows (CIRR, base R@1 %.2f):\ntext helps; no demonstrated verifier gain" % base,
                fontsize=8.8, color=INK, loc="left")
    b.grid(axis="y", color=GRID, lw=0.5)
    b.set_axisbelow(True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = load(a.root)
    os.makedirs(a.out, exist_ok=True)
    fig_overview(os.path.join(a.out, "fig_overview.pdf"))
    fig_coverage(d, os.path.join(a.out, "fig_coverage.pdf"))
    fig_effects(d, os.path.join(a.out, "fig_effects.pdf"))
    fig_graphical_abstract(d, os.path.join(a.out, "graphical_abstract.pdf"), os.path.join(a.out, "graphical_abstract.png"))
    print("wrote 3 figures and the graphical abstract to", a.out)


if __name__ == "__main__":
    main()
