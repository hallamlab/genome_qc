#!/usr/bin/env python3
"""Build a lollipop panel from genome-quality category summary tables."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summarize_metapathways_genomes import (  # noqa: E402
    apply_figure_typography,
    label_multi_panel_axes,
    plot_font_rc,
)


COUNT_METRICS = [
    ("total_bins", "Total"),
    ("hq_bins", "HQ"),
    ("mq_bins", "MQ"),
    ("lq_bins", "LQ"),
]

HALLMARK_METRICS = [
    ("mean_16S", "16S"),
    ("mean_23S", "23S"),
    ("mean_5S", "5S"),
    ("mean_trna_ge_18", "tRNA>=18"),
]

DEFAULT_ORDER = ["SAGs", "xPGs_SAGs", "MAGs", "xPGs_MAGs"]
DISPLAY_LABELS = {
    "SAGs": "SAGs",
    "xPGs_SAGs": "SAG-xPGs",
    "MAGs": "MAGs",
    "xPGs_MAGs": "MAG-xPGs",
}


def existing_category_order(values: pd.Series) -> list[str]:
    observed = [value for value in DEFAULT_ORDER if value in set(values.astype(str))]
    extras = sorted(set(values.astype(str)) - set(observed))
    return observed + extras


def load_count_summary(path: Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(path, sep="\t")
    if "category" not in frame.columns:
        raise ValueError(f"{path} does not contain a 'category' column")
    missing = [metric for metric, _ in COUNT_METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing count columns: {', '.join(missing)}")
    order = existing_category_order(frame["category"])
    counts = (
        frame.groupby("category", as_index=False)[[metric for metric, _ in COUNT_METRICS]]
        .sum()
        .set_index("category")
        .reindex(order)
        .reset_index()
    )
    return counts, order


def load_hallmark_summary(path: Path, order: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "category" not in frame.columns:
        raise ValueError(f"{path} does not contain a 'category' column")
    missing = [metric for metric, _ in HALLMARK_METRICS if metric not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing hallmark columns: {', '.join(missing)}")
    records = []
    by_category = frame.set_index("category")
    for category in order:
        if category not in by_category.index:
            continue
        for metric, label in HALLMARK_METRICS:
            value = pd.to_numeric(by_category.loc[category, metric], errors="coerce")
            if pd.notna(value):
                records.append(
                    {
                        "category": category,
                        "hallmark": label,
                        "fraction_present": float(np.clip(value, 0.0, 1.0)),
                    }
                )
    return pd.DataFrame.from_records(records)


def draw_grouped_lollipops(
    ax: plt.Axes,
    data: pd.DataFrame,
    order: list[str],
    hue_column: str,
    value_column: str,
    hue_order: list[str],
    palette: dict[str, str],
    ylabel: str,
    legend_title: str,
    ylim: tuple[float, float] | None = None,
    annotate_ints: bool = False,
) -> None:
    x_base = np.arange(len(order), dtype=float)
    display_order = [DISPLAY_LABELS.get(category, category) for category in order]
    offsets = np.linspace(-0.19, 0.19, len(hue_order)) if hue_order else []
    offset_map = dict(zip(hue_order, offsets))
    ymax = float(data[value_column].max()) if not data.empty else 1.0
    label_offset = ymax * 0.03

    for hue in hue_order:
        subset = data.loc[data[hue_column] == hue].set_index("category")
        for base, category in zip(x_base, order):
            if category not in subset.index:
                continue
            value = float(subset.loc[category, value_column])
            x_pos = base + offset_map[hue]
            color = palette[hue]
            ax.vlines(x_pos, 0, value, color=color, linewidth=2.5, zorder=2)
            ax.scatter(
                x_pos,
                value,
                s=72,
                color=color,
                edgecolor="black",
                linewidth=0.75,
                label=hue,
                zorder=3,
            )
            if annotate_ints:
                ax.text(x_pos, value + label_offset, f"{int(value)}", ha="center", va="bottom", fontsize=7)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        title=legend_title,
        frameon=False,
        ncol=min(len(hue_order), 4),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_base)
    ax.set_xticklabels(display_order, rotation=45, ha="right")
    ax.set_xlabel("Category")
    ax.set_xlim(-0.42, len(order) - 0.58)
    ax.grid(False, axis="x")
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_count_lollipops(ax: plt.Axes, counts: pd.DataFrame, order: list[str]) -> None:
    records = []
    for metric, label in COUNT_METRICS:
        for _, row in counts.iterrows():
            records.append(
                {
                    "category": row["category"],
                    "metric": label,
                    "genome_count": int(row[metric]),
                }
            )
    count_long = pd.DataFrame.from_records(records)
    hue_order = [label for _, label in COUNT_METRICS]
    palette = dict(zip(hue_order, ["#111111", "#555555", "#969696", "#d9d9d9"]))
    draw_grouped_lollipops(
        ax,
        count_long,
        order,
        "metric",
        "genome_count",
        hue_order,
        palette,
        "Genome count",
        "Quality class",
        annotate_ints=True,
    )


def draw_hallmark_lollipops(ax: plt.Axes, hallmarks: pd.DataFrame, order: list[str]) -> None:
    hallmark_order = [label for _, label in HALLMARK_METRICS if label in set(hallmarks["hallmark"])]
    palette = dict(zip(hallmark_order, ["#111111", "#555555", "#969696", "#d9d9d9"]))
    draw_grouped_lollipops(
        ax,
        hallmarks,
        order,
        "hallmark",
        "fraction_present",
        hallmark_order,
        palette,
        "Mean fraction present",
        "Hallmark",
        ylim=(0, 1.02),
    )


def build_panel(counts_path: Path, summary_path: Path, output_base: Path, title: str) -> None:
    counts, order = load_count_summary(counts_path)
    hallmarks = load_hallmark_summary(summary_path, order)
    if hallmarks.empty:
        raise ValueError("No hallmark values were available for plotting")

    plt.rcParams.update(plot_font_rc())
    sns.set_theme(style="whitegrid", context="paper", rc=plot_font_rc())
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 6.1), constrained_layout=False)
    count_ax, hallmark_ax = axes

    draw_count_lollipops(count_ax, counts, order)
    draw_hallmark_lollipops(hallmark_ax, hallmarks, order)

    if title:
        fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
        top = 0.78
    else:
        top = 0.84
    fig.subplots_adjust(top=top, bottom=0.19, left=0.08, right=0.985, wspace=0.16)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    label_multi_panel_axes(fig)
    apply_figure_typography(fig)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-summary", required=True, type=Path)
    parser.add_argument("--category-summary", required=True, type=Path)
    parser.add_argument("--output-base", required=True, type=Path)
    parser.add_argument(
        "--title",
        default="Genome counts and hallmark recovery by genome category",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_panel(args.count_summary, args.category_summary, args.output_base, args.title)
    print(args.output_base.with_suffix(".png"))
    print(args.output_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
