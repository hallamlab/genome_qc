#!/usr/bin/env python3
"""Draw per-module KO delta panels faceted by taxonomic scope."""

from __future__ import annotations

import argparse
from functools import lru_cache
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
    PANEL_LABEL_FONT_SIZE,
    apply_minimum_font_size,
    ensure_plotting,
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
DEFAULT_KEGG_KO_HIERARCHY = (
    Path.home() / "MPDB_260131" / "functional_categories" / "kegg_mappings" / "ko00001.keg"
)
KO_SHORT_LABELS = {
    "K00368": "NirK",
    "K00370": "NarG/NxrA",
    "K00371": "NarH/NxrB",
    "K00374": "NarI",
    "K00376": "NosZ",
    "K02305": "NorC",
    "K02567": "NapA",
    "K02568": "NapB",
    "K04561": "NorB",
    "K15864": "NirS",
}

# KOfam definitions often include an explicit gene symbol (for example NifH or
# CysN/CysC), but also contain biochemical abbreviations such as NADH and CoA.
# This deliberately narrow pattern accepts gene-like mixed-case symbols and
# symbols ending in a number while leaving ambiguous abbreviations untouched.
GENE_SYMBOL_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][a-z]{1,8}[A-Z][A-Za-z0-9]*|[A-Z]{3,8}\d+)(?![A-Za-z0-9])"
)
NON_GENE_MIXED_CASE_TOKENS = {
    "AdoMet",
    "CoA",
    "CoB",
    "CoM",
    "FeS",
}


def normalize_kegg_symbol(value: str) -> str | None:
    """Convert a KEGG gene symbol to the protein-style display convention."""
    symbol = value.strip()
    if not symbol or re.fullmatch(r"K\d{5}", symbol, flags=re.IGNORECASE):
        return None
    if re.match(r"^E\d+(?:\.|$)", symbol, flags=re.IGNORECASE) or symbol.upper().startswith("TC."):
        return None
    if symbol[0].islower():
        symbol = symbol[0].upper() + symbol[1:]
    return symbol


@lru_cache(maxsize=4)
def load_kegg_ko_symbols(path_text: str = str(DEFAULT_KEGG_KO_HIERARCHY)) -> dict[str, str]:
    """Load KO-to-symbol aliases from a local KEGG ko00001 hierarchy."""
    path = Path(path_text).expanduser()
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    row_pattern = re.compile(r"^D\s+(K\d{5})\s+(.+?);\s*")
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = row_pattern.match(line)
            if not match:
                continue
            ko, raw_symbols = match.groups()
            if ko in mapping:
                continue
            symbols = []
            for raw_symbol in raw_symbols.split(","):
                symbol = normalize_kegg_symbol(raw_symbol)
                if symbol and symbol.casefold() not in {item.casefold() for item in symbols}:
                    symbols.append(symbol)
            if symbols:
                mapping[ko] = "/".join(symbols)
    return mapping


def infer_ko_short_label(ko: object, ko_name: object = "") -> tuple[str | None, str]:
    """Return a conservative short KO label and its provenance."""
    ko_text = str(ko).strip()
    curated = KO_SHORT_LABELS.get(ko_text)
    if curated:
        return curated, "curated"

    kegg_symbol = load_kegg_ko_symbols().get(ko_text)
    if kegg_symbol:
        return kegg_symbol, "kegg_ko00001_symbol"

    name = re.sub(r"\s+", " ", str(ko_name).strip())
    name = re.sub(r"\s*\(K\d+\)\s*$", "", name)
    # Bracketed carrier/cofactor names, such as [DsrC], do not identify the
    # encoded subunit and must not be promoted to gene labels.
    searchable = re.sub(r"\[[^]]+\]", " ", name)
    symbols = []
    for match in GENE_SYMBOL_PATTERN.finditer(searchable):
        symbol = match.group(0)
        if symbol in NON_GENE_MIXED_CASE_TOKENS:
            continue
        following = searchable[match.end() :]
        if re.match(
            r"(?:/[A-Za-z0-9]+)*\)?\s+family\b",
            following,
            flags=re.IGNORECASE,
        ):
            continue
        if symbol not in symbols:
            symbols.append(symbol)
    if symbols:
        return "/".join(symbols), "explicit_definition_symbol"
    return None, "description_fallback"


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
    parser.add_argument(
        "--module-id",
        help="Optional module ID filter for rendering a single module demo, e.g. M00529.",
    )
    return parser.parse_args()


def safe_filename(value: object, max_length: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "module")[:max_length]


def ko_axis_label(record: pd.Series) -> str:
    ko = str(record.get("ko", "")).strip()
    short_label, _source = infer_ko_short_label(ko, record.get("ko_name", ""))
    if short_label:
        return f"{short_label} ({ko})"
    ko_name = str(record.get("ko_name", "")).strip()
    if ko_name:
        return f"{ko_name} ({ko})"
    return str(record.get("ko_label", ko)).strip()


def module_axis_table(module_data: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["ko", "ko_order", "ko_name", "ko_label"] if column in module_data.columns]
    axis_table = (
        module_data.loc[:, columns]
        .drop_duplicates("ko")
        .sort_values(["ko_order", "ko"])
        .reset_index(drop=True)
    )
    axis_table["y_position"] = np.arange(axis_table.shape[0], dtype=float)
    axis_table["axis_label"] = axis_table.apply(ko_axis_label, axis=1)
    return axis_table


def symmetric_delta_limit(data: pd.DataFrame, fallback_limit: float) -> float:
    bounds = pd.to_numeric(
        pd.concat(
            [
                data.get("delta_ci95_low", pd.Series(dtype=float)),
                data.get("delta_ci95_high", pd.Series(dtype=float)),
                data.get("delta_prevalence_percent_points", pd.Series(dtype=float)),
            ],
            ignore_index=True,
        ),
        errors="coerce",
    ).dropna()
    if bounds.empty:
        return abs(fallback_limit)
    needed = float(np.nanmax(np.abs(bounds))) * 1.08
    rounded_limit = math.ceil(max(25.0, needed) / 5.0) * 5.0
    return min(rounded_limit, abs(fallback_limit))


def delta_ticks(limit: float) -> list[float]:
    max_tick = int(math.floor(limit / 25.0) * 25)
    max_tick = max(25, max_tick)
    return [float(tick) for tick in range(-max_tick, max_tick + 1, 25)]


def category_display_label(value: object) -> str:
    return str(value).replace("_", "-")


def add_category_side_headers(ax, left_label: str, right_label: str) -> None:
    ax.text(
        0.25,
        1.018,
        wrap_label(left_label, width=16),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=MIN_PLOT_FONT_SIZE + 1,
        fontweight="bold",
    )
    ax.text(
        0.75,
        1.018,
        wrap_label(right_label, width=16),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=MIN_PLOT_FONT_SIZE + 1,
        fontweight="bold",
    )


def save_cross_scope_figure(fig, output_base: Path | str) -> None:
    plt = ensure_plotting()
    mp_genomes.PLOT_FONT_SIZES["panel_label"] = max(
        PANEL_LABEL_FONT_SIZE,
        mp_genomes.PLOT_FONT_SIZES.get("panel_label", 0),
    )
    mp_genomes.apply_plot_style()
    mp_genomes.label_multi_panel_axes(fig)
    for ax in fig.axes:
        label_artist = getattr(ax, "_mp_panel_label_artist", None)
        if label_artist is None:
            continue
        label_artist.set_position((0.010, 1.025))
        label_artist.set_ha("right")
        label_artist.set_va("bottom")
    apply_minimum_font_size(fig)
    mp_genomes.apply_figure_typography(fig)
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


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

    axis_table = module_axis_table(comparison)
    ko_to_y = axis_table.set_index("ko")["y_position"].to_dict()
    y_ticks = axis_table["y_position"].to_numpy(dtype=float)
    y_labels = axis_table["axis_label"].tolist()
    max_kos = max(1, axis_table.shape[0])
    x_limit = symmetric_delta_limit(comparison, delta_limit)
    x_ticks = delta_ticks(x_limit)
    figure_size = max(16.0, max_kos * 0.7 + 9.0)
    fig, axes = plt.subplots(
        1,
        len(scopes_present),
        figsize=(figure_size, figure_size),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = axes.ravel()

    module_id = str(comparison["module_id"].iat[0])
    module_name = str(comparison["module_name"].iat[0])
    first = category_display_label(comparison["base_category"].iat[0])
    second = category_display_label(comparison["xpg_category"].iat[0])

    wrote_any = False
    for ax, scope in zip(axes, scopes_present):
        scope_data = comparison.loc[comparison["taxon_filter"].eq(scope)].sort_values("ko_order")
        if scope_data.empty:
            ax.axis("off")
            continue

        y_positions = scope_data["ko"].map(ko_to_y).to_numpy(dtype=float)
        deltas = scope_data["delta_prevalence_percent_points"].to_numpy(dtype=float)
        ci_low = scope_data["delta_ci95_low"].to_numpy(dtype=float)
        ci_high = scope_data["delta_ci95_high"].to_numpy(dtype=float)
        significant = scope_data["significant_q_le_threshold"].fillna(False).to_numpy(dtype=bool)
        one_category_only = scope_data["present_in_one_category_only"].fillna(False).to_numpy(dtype=bool)

        ax.axvline(0, color="black", linewidth=1.3, linestyle="--", zorder=1)
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

        ax.set_yticks(y_ticks)
        ax.set_yticklabels([wrap_label(label, width=18) for label in y_labels])
        ax.set_ylim(max(y_ticks) + 0.5, -0.5)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xticks(x_ticks)
        ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=1.1)
        ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.9)
        ax.set_title(
            SCOPE_LABELS.get(scope, scope),
            fontsize=MIN_PLOT_FONT_SIZE + 4,
            fontweight="bold",
            pad=40,
        )
        add_category_side_headers(ax, first, second)
        ax.tick_params(axis="x", labelsize=MIN_PLOT_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=MIN_PLOT_FONT_SIZE)
        wrote_any = True

    if not wrote_any:
        plt.close(fig)
        return []

    fig.tight_layout(rect=[0.025, 0.045, 1.0, 0.98], w_pad=2.4, h_pad=2.4)
    save_cross_scope_figure(fig, output_base)
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
    axis_table = module_axis_table(scope_data_all)
    ko_to_y = axis_table.set_index("ko")["y_position"].to_dict()
    y_ticks = axis_table["y_position"].to_numpy(dtype=float)
    y_labels = axis_table["axis_label"].tolist()
    max_kos = max(1, axis_table.shape[0])
    x_limit = symmetric_delta_limit(scope_data_all, delta_limit)
    x_ticks = delta_ticks(x_limit)
    figure_size = max(18.0, max_kos * 0.9 + 9.5)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(figure_size, figure_size),
        squeeze=False,
        sharex=True,
        sharey=True,
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

            y_positions = comparison["ko"].map(ko_to_y).to_numpy(dtype=float)
            deltas = comparison["delta_prevalence_percent_points"].to_numpy(dtype=float)
            ci_low = comparison["delta_ci95_low"].to_numpy(dtype=float)
            ci_high = comparison["delta_ci95_high"].to_numpy(dtype=float)
            significant = comparison["significant_q_le_threshold"].fillna(False).to_numpy(dtype=bool)
            one_category_only = comparison["present_in_one_category_only"].fillna(False).to_numpy(dtype=bool)

            ax.axvline(0, color="black", linewidth=1.3, linestyle="--", zorder=1)
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

            ax.set_yticks(y_ticks)
            ax.set_yticklabels([wrap_label(label, width=18) for label in y_labels])
            ax.set_ylim(max(y_ticks) + 0.5, -0.5)
            ax.set_xlim(-x_limit, x_limit)
            ax.set_xticks(x_ticks)
            ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=1.1)
            ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.9)
            left_label = category_display_label(comparison["base_category"].iat[0])
            right_label = category_display_label(comparison["xpg_category"].iat[0])
            add_category_side_headers(ax, left_label, right_label)
            ax.tick_params(axis="x", labelsize=MIN_PLOT_FONT_SIZE)
            ax.tick_params(axis="y", labelsize=MIN_PLOT_FONT_SIZE)
            wrote_any = True

    if not wrote_any:
        plt.close(fig)
        return []

    fig.tight_layout(rect=[0.025, 0.045, 1.0, 0.98], w_pad=2.4, h_pad=3.2)
    original_panel_label_size = mp_genomes.PLOT_FONT_SIZES.get("panel_label", MIN_PLOT_FONT_SIZE)
    mp_genomes.PLOT_FONT_SIZES["panel_label"] = max(
        COMPARISON_PANEL_LABEL_FONT_SIZE,
        original_panel_label_size,
    )
    try:
        save_cross_scope_figure(fig, output_base)
    finally:
        mp_genomes.PLOT_FONT_SIZES["panel_label"] = original_panel_label_size
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def write_cycle_panels(
    results_dir: Path,
    cycle: str,
    delta_limit: float,
    q_threshold: float,
    panel_family: str,
    module_id_filter: str | None,
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
    if module_id_filter:
        module_ids = module_keys["module_id"].astype(str)
        module_keys = module_keys.loc[
            module_ids.eq(module_id_filter) | module_ids.str.startswith(f"{module_id_filter}.")
        ].copy()
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
            args.module_id,
        )
        written.extend(cycle_written)
        print(f"[done] {cycle}: wrote {len(cycle_written)} files")
    print("[done] wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
