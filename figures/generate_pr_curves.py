#!/usr/bin/env python3
"""Render precision--recall curves for the four IH/CH retrieval conditions.

Release/supplementary artifact backing the F1max/AUPR values in Table III of
the manuscript: each query's top-1 candidate carries its descriptor
confidence; sweeping an acceptance threshold yields precision (correct
accepted / accepted) over recall (correct accepted / all queries). Data come
from the same per-query CSVs as ``metrics_augment_20260725/pr_f1_summary.csv``
via ``slam/experiments/common/compute_retrieval_stats.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "slam/experiments/common"))
import compute_retrieval_stats as crs  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
})

INK = "#24313D"
MID = "#6B7783"
STYLE = {
    "SC": ("#8C9BAA", "-"),
    "SC++ (PC)": ("#B7A16C", "-"),
    "SOLiD": ("#7FA88C", "-"),
    "M2DP": ("#A98BA8", "-"),
    "LiDAR-Iris": ("#5B8DB8", "-"),
    "RING++": ("#C98A5A", "-"),
    "UpDown-SC": ("#B64342", "-"),
}
TITLES = {
    "ih_native": "IH native ($n_q$=320)",
    "ih_gravity": "IH +G ($n_q$=320)",
    "ch_native": "CH native ($n_q$=148)",
    "ch_gravity": "CH +G ($n_q$=148)",
}


def curve(scores: np.ndarray, correct: np.ndarray, ascending: bool,
          total: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores if ascending else -scores, kind="stable")
    flags = correct[order]
    tp = np.cumsum(flags)
    precision = tp / np.arange(1, len(flags) + 1)
    recall = tp / total
    return recall, precision


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2), facecolor="white",
                             sharex=True, sharey=True)
    for (cond, methods), ax in zip(crs.pr_files().items(), axes.flat):
        for method, (path, algo) in methods.items():
            if algo == "CANDIDATES":
                scores, flags, _, total = crs.updown_scores(cond)
            else:
                rows = crs.read_rows(path, algo)
                total = len(rows)
                scores = np.asarray([float(r["top1_score"]) for r in rows])
                flags = np.asarray([crs.as_bool(r["recall_at_1"]) for r in rows])
            recall, precision = curve(
                scores, flags, crs.DISTANCE_LIKE[method], total)
            color, ls = STYLE[method]
            is_ours = method == "UpDown-SC"
            ax.plot(recall, precision, color=color, linestyle=ls,
                    linewidth=1.9 if is_ours else 1.1,
                    zorder=6 if is_ours else 3,
                    label=method + (" (ours)" if is_ours else ""))
        ax.set_title(TITLES[cond], fontsize=8.5, color=INK, weight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.02)
        ax.grid(color="#E4E9ED", linewidth=0.5)
        ax.set_axisbelow(True)
    for ax in axes[1, :]:
        ax.set_xlabel("recall (correct accepted / all queries)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("precision", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Top-1 acceptance precision–recall "
                 "(threshold sweep on descriptor confidence)",
                 fontsize=9.5, weight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT / "pr_curves.pdf", bbox_inches="tight")
    fig.savefig(OUT / "pr_curves.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUT / "pr_curves.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote pr_curves.{pdf,png,svg}")
    make_paper_figure()


def make_paper_figure() -> None:
    """Single-column two-panel version for the manuscript (+G conditions)."""
    mpl.rcParams.update({"font.size": 5.4})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.72), facecolor="white",
                             sharey=True)
    conditions = {"ih_gravity": "IH +G ($n_q$=320)",
                  "ch_gravity": "CH +G ($n_q$=148)"}
    files = crs.pr_files()
    for (cond, title), ax in zip(conditions.items(), axes):
        for method, (path, algo) in files[cond].items():
            if algo == "CANDIDATES":
                scores, flags, _, total = crs.updown_scores(cond)
            else:
                rows = crs.read_rows(path, algo)
                total = len(rows)
                scores = np.asarray([float(r["top1_score"]) for r in rows])
                flags = np.asarray([crs.as_bool(r["recall_at_1"]) for r in rows])
            recall, precision = curve(
                scores, flags, crs.DISTANCE_LIKE[method], total)
            color, ls = STYLE[method]
            is_ours = method == "UpDown-SC"
            ax.plot(recall, precision, color=color, linestyle=ls,
                    linewidth=1.5 if is_ours else 0.8,
                    zorder=6 if is_ours else 3,
                    label=method + (" (ours)" if is_ours else ""))
        ax.set_title(title, fontsize=5.8, weight="bold", color=INK, pad=2)
        ax.set_xlim(0, 0.8)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.tick_params(labelsize=5.0, length=1.8, width=0.5, pad=1.5)
        ax.grid(color="#E9EDF0", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel("recall", fontsize=5.4, labelpad=1.5)
    axes[0].set_ylabel("precision", fontsize=5.4, labelpad=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=4.9,
               bbox_to_anchor=(0.5, -0.045), columnspacing=0.9,
               handlelength=1.4, handletextpad=0.4)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.30,
                        wspace=0.08)
    fig.savefig(OUT / "pr_curves_paper.pdf", bbox_inches="tight")
    fig.savefig(OUT / "pr_curves_paper.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "pr_curves_paper.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote pr_curves_paper.{pdf,png,svg}")


if __name__ == "__main__":
    main()
