#!/usr/bin/env python3
"""Redraw Figures 2 and 3 of the paper at their final printed size.

The values come unchanged from the CSV files written by simulate.py
(data/fig2_prompt.csv, data/fig3_errors.csv). Only the figure size and font
size differ from simulate.py, so that the text is about 7 pt in the PDF.
All values are EXPECTED values from a seeded simulation (not measured).

Usage: python make_paper_figures.py --data <dir with the CSV files> --out <figure dir>
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "purple": "#CC79A7",
         "sky": "#56B4E9", "vermillion": "#D55E00", "yellow": "#F0E442", "grey": "#999999"}
MODELS = ["gpt", "kaz", "llama", "qwen"]
ERR_CATS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
ERR_LABELS = {"E1": "E1 Retrieval miss", "E2": "E2 Citation missing", "E3": "E3 Wrong/invented provision",
              "E4": "E4 Incomplete answer", "E5": "E5 Misinterpretation", "E6": "E6 Cross-ref./repealed norm",
              "E7": "E7 Language error", "E8": "E8 Unjustified abstention"}


def set_style():
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "hatch.linewidth": 0.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def sim_label(ax, x=0.99, y=0.98, ha="right", va="top"):
    ax.text(x, y, "expected (simulation)", transform=ax.transAxes, ha=ha, va=va,
            fontsize=6, color="0.5", style="italic")


def fig_prompt(df, out):
    labels = ["GPT-\n6.1", "KazLLM-\n1.0-8B", "Llama-\n3.1-8B", "Qwen3.5-\n9B"]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 1.8), layout="constrained")
    x = np.arange(4)
    for ax, met, ylab in ((axes[0], "cor", "Legal correctness (%)"), (axes[1], "hal", "Hallucination rate (%)")):
        top = []
        for j, (lang, col, mk, lab) in enumerate((("kk", OKABE["blue"], "o", "Kazakh system prompt"),
                                                   ("ru", OKABE["orange"], "^", "Russian system prompt"))):
            sub = df[(df["metric"] == met) & (df["prompt"] == lang)].set_index("model").loc[MODELS]
            v, lo, hi = sub["value"].to_numpy(), sub["ci_lo"].to_numpy(), sub["ci_hi"].to_numpy()
            ax.errorbar(x + (j - 0.5) * 0.3, v, yerr=[v - lo, hi - v], fmt=mk, color=col, ms=3.2,
                        capsize=1.6, elinewidth=0.8, capthick=0.6, label=lab)
            top.append(hi)
        top = np.max(top, axis=0)
        sig = df[(df["metric"] == met) & (df["prompt"] == "kk")].set_index("model").loc[MODELS, "significant"]
        for i, s in enumerate(sig.astype(str).str.lower().eq("true").to_numpy()):
            if s:
                ax.text(x[i], top[i] + 0.8, "*", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylabel(ylab)
    axes[0].set_ylim(35, 88)
    axes[1].set_ylim(0, 40)
    sim_label(axes[1])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="outside upper center", frameon=False)
    fig.supxlabel("Generator model (RAG, k = 5)", fontsize=7)
    fig.savefig(out / "fig2_prompt_language.pdf", format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_errors(df, out):
    cols = [OKABE[k] for k in ("vermillion", "orange", "purple", "blue", "sky", "green", "yellow", "grey")]
    hatches = ["", "////", "", "\\\\\\\\", "....", "", "xxx", "----"]
    fig, ax = plt.subplots(figsize=(5.0, 1.85), layout="constrained")
    y = np.arange(4)[::-1]
    left = np.zeros(4)
    tab = df.set_index("model").loc[MODELS]
    for c, col, ht in zip(ERR_CATS, cols, hatches):
        vals = tab[c].to_numpy(float)
        ax.barh(y, vals, left=left, color=col, edgecolor="0.2", linewidth=0.3, hatch=ht,
                height=0.66, label=ERR_LABELS[c])
        left += vals
    for yi, tot in zip(y, tab["failed"].to_numpy()):
        ax.text(tot + 4, yi, f"{tot}", va="center", fontsize=6.5)
    ax.set_yticks(y, tab["model_name"].tolist())
    ax.set_xlim(0, 300)
    ax.set_xlabel("Failed answers (count)")
    fig.legend(ncol=4, loc="outside lower center", frameon=False, handlelength=1.4,
               handleheight=0.9, labelspacing=0.3, columnspacing=0.8)
    sim_label(ax)
    fig.savefig(out / "fig3_error_taxonomy.pdf", format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_style()
    fig_prompt(pd.read_csv(data / "fig2_prompt.csv"), out)
    fig_errors(pd.read_csv(data / "fig3_errors.csv"), out)


if __name__ == "__main__":
    main()
