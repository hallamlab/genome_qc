#!/usr/bin/env python3
"""Draw per-module KO delta panels faceted by taxonomic scope."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import summarize_metapathways_genomes as mp_genomes  # noqa: E402
from kofam_nitrogen_module_ko_facets import (  # noqa: E402
    COMPARISONS,
    CYCLE_LABELS,
    MIN_PLOT_FONT_SIZE,
    ensure_plotting,
    save_nitrogen_figure,
    short_module_title,
    wrap_label,
)


DEFAULT_RESULTS_DIR = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/genome_qc_results"
)
DEFAULT_CYCLES = [
    "nitrogen",
    "sulfur",
    "carbon",
    "phosphorus",
    "iron",
    "trace_metals",
    "mobile_genetic_elements",
]
SCOPES = ["all", "gammaproteobacteria", "sup05"]
SCOPE_LABELS = {
    "all": "All genomes",
    "gammaproteobacteria": "Gammaproteobacteria",
    "sup05": "SUP05 + Thioglobus",
}
COMPARISON_PANEL_GROUPS = [
    ("Same to same", ["sag_xpg", "mag_xpg"]),
    ("Cross", ["sag_xpg_mag", "mag_xpg_sag"]),
    ("Type contrasts", ["mag_sag", "xpg_mag_xpg_sag"]),
]
COMPARISON_PANEL_LABEL_FONT_SIZE = 44


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cycles", nargs="+", default=DEFAULT_CYCLES)
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument("--delta-limit", type=float, default=100.0)
    parser.add_argument(
        "--panel-family",
        choices=["all", "scope", "comparison"],
        default="all",
        help="Which per-module panel family to render.",
    )
    return parser.parse_args()


def safe_filename(value: object, max_length: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "module")[:max_length]


def read_cycle_tables(results_dir: Path, cycle: str) -> pd.DataFrame:
    frames = []
    for scope in SCOPES:
        path = (
            results_dir
            / "elemental_cycles"
            / cycle
            / scope
            / f"kofam_{cycle}_modules.ko_delta_plotdata.tsv"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, sep="\t", low_memory=False)
        frame["taxon_filter"] = scope
        frame["taxon_label"] = SCOPE_LABELS[scope]
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table["scope_order"] = table["taxon_filter"].map({scope: index for index, scope in enumerate(SCOPES)})
    return table


def draw_scope_module_panel(
    module_data: pd.DataFrame,
    comparison_id: str,
    output_base: Path,
    cycle_label: str,
    delta_limit: float,
    q_threshold: float,
) -> list[Path]:
    comparison = module_data.loc[module_data["comparison_id"].eq(comparison_id)].copy()
    if comparison.empty:
        return []

    plt = ensure_plotting()
    scopes_present = [scope for scope in SCOPES if scope in set(comparison["taxon_filter"].astype(str))]
    if not scopes_present:
        return []

    max_kos = max(
        1,
        int(
            comparison.groupby("taxon_filter", sort=False)
            .size()
            .reindex(scopes_present)
            .fillna(0)
            .max()
        ),
    )
    fig_height = max(14.0, max_kos * 1.35 + 5.8)
    fig_width = max(36.0, 12.5 * len(scopes_present))
    fig, axes = plt.subplots(
        1,
        len(scopes_present),
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
        sharey=False,
    )
    axes = axes.ravel()

    module_id = str(comparison["module_id"].iat[0])
    module_name = str(comparison["module_name"].iat[0])
    comparison_label = str(comparison["comparison_label"].iat[0])
    first = str(comparison["base_category"].iat[0]).replace("_", " ")
    second = str(comparison["xpg_category"].iat[0]).replace("_", " ")

    wrote_any = False
    for ax, scope in zip(axes, scopes_present):
        scope_data = comparison.loc[comparison["taxon_filter"].eq(scope)].sort_values("ko_order")
        if scope_data.empty:
            ax.axis("off")
            continue

        y_positions = np.arange(scope_data.shape[0], dtype=float)
        deltas = scope_data["delta_prevalence_percent_points"].to_numpy(dtype=float)
        ci_low = scope_data["delta_ci95_low"].to_numpy(dtype=float)
        ci_high = scope_data["delta_ci95_high"].to_numpy(dtype=float)
        significant = scope_data["significant_q_le_threshold"].fillna(False).to_numpy(dtype=bool)
        one_category_only = scope_data["present_in_one_category_only"].fillna(False).to_numpy(dtype=bool)

        ax.axvline(0, color="black", linewidth=0.9, zorder=1)
        for y_pos, delta, low, high, is_significant, is_one_category_only in zip(
            y_positions,
            deltas,
            ci_low,
            ci_high,
            significant,
            one_category_only,
        ):
            if is_one_category_only:
                color = "black"
                facecolor = "white"
                edge = "black"
                size = 52
            elif is_significant:
                color = "black"
                facecolor = "black"
                edge = "black"
                size = 46
            else:
                color = "#bdbdbd"
                facecolor = "#bdbdbd"
                edge = "#737373"
                size = 34
            ax.errorbar(
                delta,
                y_pos,
                xerr=np.array([[delta - low], [high - delta]], dtype=float),
                fmt="none",
                ecolor=color,
                elinewidth=3.0,
                capsize=6.0,
                capthick=2.2,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                delta,
                y_pos,
                s=size * 3.8,
                facecolors=facecolor,
                edgecolors=edge,
                linewidths=1.8,
                zorder=3,
            )

        label_column = "ko_label" if "ko_label" in scope_data.columns else "ko"
        ax.set_yticks(y_positions)
        ax.set_yticklabels([wrap_label(label, width=24) for label in scope_data[label_column].tolist()])
        ax.invert_yaxis()
        ax.set_xlim(-abs(delta_limit), abs(delta_limit))
        ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=1.1)
        ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.9)
        ax.set_title(
            SCOPE_LABELS.get(scope, scope),
            fontsize=MIN_PLOT_FONT_SIZE + 4,
            fontweight="bold",
            pad=14,
        )
        ax.tick_params(axis="x", labelsize=MIN_PLOT_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=MIN_PLOT_FONT_SIZE)
        wrote_any = True

    if not wrote_any:
        plt.close(fig)
        return []

    fig.suptitle(
        wrap_label(
            f"{cycle_label}: {short_module_title(module_id, module_name)}\n{comparison_label}",
            width=82,
        ),
        fontsize=MIN_PLOT_FONT_SIZE + 10,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel(
        wrap_label(f"Prevalence difference, {second} - {first} (percentage points)", width=78),
        fontsize=MIN_PLOT_FONT_SIZE,
        y=0.055,
    )
    fig.text(
        0.5,
        0.018,
        wrap_label(
            f"Black points have BH-adjusted pairwise Fisher q<={q_threshold:g}; gray points are not significant; "
            "open black points are present in only one side of the comparison.",
            width=118,
        ),
        ha="center",
        va="bottom",
        fontsize=MIN_PLOT_FONT_SIZE,
        color="#4d4d4d",
    )
    fig.tight_layout(rect=[0.025, 0.10, 1.0, 0.90], w_pad=5.0, h_pad=3.5)
    save_nitrogen_figure(fig, output_base)
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def draw_comparison_facet_module_panel(
    module_data: pd.DataFrame,
    scope: str,
    output_base: Path,
    cycle_label: str,
    delta_limit: float,
    q_threshold: float,
) -> list[Path]:
    scope_data_all = module_data.loc[module_data["taxon_filter"].eq(scope)].copy()
    if scope_data_all.empty:
        return []

    available_comparisons = set(scope_data_all["comparison_id"].astype(str))
    required_comparisons = [comparison_id for _group, ids in COMPARISON_PANEL_GROUPS for comparison_id in ids]
    if not any(comparison_id in available_comparisons for comparison_id in required_comparisons):
        return []

    plt = ensure_plotting()
    max_kos = max(
        1,
        int(
            scope_data_all.groupby("comparison_id")
            .size()
            .reindex(required_comparisons)
            .fillna(0)
            .max()
        ),
    )
    fig_height = max(24.0, max_kos * 2.55 + 8.5)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(48.0, fig_height),
        squeeze=False,
        sharex=True,
        sharey=False,
    )

    module_id = str(scope_data_all["module_id"].iat[0])
    module_name = str(scope_data_all["module_name"].iat[0])
    scope_label = SCOPE_LABELS.get(scope, str(scope))
    wrote_any = False

    for col_index, (_group_label, comparison_ids) in enumerate(COMPARISON_PANEL_GROUPS):
        for row_index, comparison_id in enumerate(comparison_ids):
            ax = axes[row_index, col_index]
            comparison = scope_data_all.loc[scope_data_all["comparison_id"].eq(comparison_id)].sort_values("ko_order")
            if comparison.empty:
                ax.axis("off")
                continue

            y_positions = np.arange(comparison.shape[0], dtype=float)
            deltas = comparison["delta_prevalence_percent_points"].to_numpy(dtype=float)
            ci_low = comparison["delta_ci95_low"].to_numpy(dtype=float)
            ci_high = comparison["delta_ci95_high"].to_numpy(dtype=float)
            significant = comparison["significant_q_le_threshold"].fillna(False).to_numpy(dtype=bool)
            one_category_only = comparison["present_in_one_category_only"].fillna(False).to_numpy(dtype=bool)

            ax.axvline(0, color="black", linewidth=0.9, zorder=1)
            for y_pos, delta, low, high, is_significant, is_one_category_only in zip(
                y_positions,
                deltas,
                ci_low,
                ci_high,
                significant,
                one_category_only,
            ):
                if is_one_category_only:
                    color = "black"
                    facecolor = "white"
                    edge = "black"
                    size = 52
                elif is_significant:
                    color = "black"
                    facecolor = "black"
                    edge = "black"
                    size = 46
                else:
                    color = "#bdbdbd"
                    facecolor = "#bdbdbd"
                    edge = "#737373"
                    size = 34
                ax.errorbar(
                    delta,
                    y_pos,
                    xerr=np.array([[delta - low], [high - delta]], dtype=float),
                    fmt="none",
                    ecolor=color,
                    elinewidth=3.0,
                    capsize=6.0,
                    capthick=2.2,
                    alpha=0.85,
                    zorder=2,
                )
                ax.scatter(
                    delta,
                    y_pos,
                    s=size * 3.8,
                    facecolors=facecolor,
                    edgecolors=edge,
                    linewidths=1.8,
                    zorder=3,
                )

            label_column = "ko_label" if "ko_label" in comparison.columns else "ko"
            ax.set_yticks(y_positions)
            ax.set_yticklabels([wrap_label(label, width=22) for label in comparison[label_column].tolist()])
            ax.invert_yaxis()
            ax.set_xlim(-abs(delta_limit), abs(delta_limit))
            ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=1.1)
            ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.9)
            comparison_label = str(comparison["comparison_label"].iat[0])
            ax.set_title(
                wrap_label(comparison_label, width=28),
                fontsize=MIN_PLOT_FONT_SIZE + 3,
                fontweight="bold",
                pad=14,
            )
            ax.tick_params(axis="x", labelsize=MIN_PLOT_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=MIN_PLOT_FONT_SIZE)
            wrote_any = True

    if not wrote_any:
        plt.close(fig)
        return []

    fig.suptitle(
        wrap_label(f"{cycle_label}: {short_module_title(module_id, module_name)}\n{scope_label}", width=86),
        fontsize=MIN_PLOT_FONT_SIZE + 10,
        fontweight="bold",
        y=0.995,
    )
    fig.supxlabel(
        "Prevalence difference between comparison groups (percentage points)",
        fontsize=MIN_PLOT_FONT_SIZE,
        y=0.050,
    )
    fig.text(
        0.5,
        0.016,
        wrap_label(
            f"Black points have BH-adjusted pairwise Fisher q<={q_threshold:g}; gray points are not significant; "
            "open black points are present in only one side of the comparison.",
            width=124,
        ),
        ha="center",
        va="bottom",
        fontsize=MIN_PLOT_FONT_SIZE,
        color="#4d4d4d",
    )
    fig.tight_layout(rect=[0.025, 0.10, 1.0, 0.90], w_pad=4.8, h_pad=5.2)
    original_panel_label_size = mp_genomes.PLOT_FONT_SIZES.get("panel_label", MIN_PLOT_FONT_SIZE)
    mp_genomes.PLOT_FONT_SIZES["panel_label"] = max(
        COMPARISON_PANEL_LABEL_FONT_SIZE,
        original_panel_label_size,
    )
    try:
        save_nitrogen_figure(fig, output_base)
    finally:
        mp_genomes.PLOT_FONT_SIZES["panel_label"] = original_panel_label_size
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def write_cycle_panels(
    results_dir: Path,
    cycle: str,
    delta_limit: float,
    q_threshold: float,
    panel_family: str,
) -> list[Path]:
    table = read_cycle_tables(results_dir, cycle)
    scope_output_dir = results_dir / "elemental_cycles" / cycle / "per_module_scope_panels"
    comparison_output_dir = results_dir / "elemental_cycles" / cycle / "per_module_comparison_panels"
    scope_output_dir.mkdir(parents=True, exist_ok=True)
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    cycle_label = CYCLE_LABELS.get(cycle, cycle.replace("_", " ").title())

    written: list[Path] = []
    module_keys = (
        table.loc[:, ["module_order", "module_id", "module_name"]]
        .drop_duplicates()
        .sort_values(["module_order", "module_id"])
    )
    if panel_family in {"all", "scope"}:
        for comparison_id in COMPARISONS:
            comparison_dir = scope_output_dir / comparison_id
            comparison_dir.mkdir(parents=True, exist_ok=True)
            for module_record in module_keys.itertuples(index=False):
                module_id = str(module_record.module_id)
                module_name = str(module_record.module_name)
                module_data = table.loc[table["module_id"].astype(str).eq(module_id)]
                output_base = comparison_dir / f"{safe_filename(module_id)}_{safe_filename(module_name)}"
                written.extend(
                    draw_scope_module_panel(
                        module_data,
                        comparison_id,
                        output_base,
                        cycle_label,
                        delta_limit,
                        q_threshold,
                    )
                )
    if panel_family in {"all", "comparison"}:
        for scope in SCOPES:
            scope_dir = comparison_output_dir / scope
            scope_dir.mkdir(parents=True, exist_ok=True)
            for module_record in module_keys.itertuples(index=False):
                module_id = str(module_record.module_id)
                module_name = str(module_record.module_name)
                module_data = table.loc[table["module_id"].astype(str).eq(module_id)]
                output_base = scope_dir / f"{safe_filename(module_id)}_{safe_filename(module_name)}"
                written.extend(
                    draw_comparison_facet_module_panel(
                        module_data,
                        scope,
                        output_base,
                        cycle_label,
                        delta_limit,
                        q_threshold,
                    )
                )
    return written


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    written: list[Path] = []
    for cycle in args.cycles:
        cycle_written = write_cycle_panels(
            results_dir,
            cycle,
            args.delta_limit,
            args.q_threshold,
            args.panel_family,
        )
        written.extend(cycle_written)
        print(f"[done] {cycle}: wrote {len(cycle_written)} files")
    print("[done] wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
