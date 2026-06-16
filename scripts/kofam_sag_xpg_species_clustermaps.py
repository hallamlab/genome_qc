#!/usr/bin/env python3
"""Draw SAG-xPG species-by-KO presence/absence clustermaps for elemental modules."""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kofam_nitrogen_module_ko_facets import (  # noqa: E402
    CYCLE_LABELS,
    DEFAULT_RESULTS_DIR,
    KEYWORD_CYCLE_PATTERNS,
    attach_ko_labels,
    load_ko_definition_map,
    load_ko_definition_map_rg,
    load_selected_ko_presence,
    load_taxonomy_table,
    prepare_cycle_modules,
    read_tsv,
    add_quality_tier,
    wrap_label,
)
from summarize_metapathways_genomes import apply_figure_typography, apply_plot_style, ensure_plotting  # noqa: E402


DEFAULT_CYCLES = [
    "nitrogen",
    "sulfur",
    "carbon",
    "phosphorus",
    "iron",
    "trace_metals",
    "mobile_genetic_elements",
]
MIN_FONT_SIZE = 22
TAXONOMY_ORDER_COLUMNS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cycles", nargs="+", default=DEFAULT_CYCLES)
    parser.add_argument("--min-genomes-per-species", type=int, default=1)
    return parser.parse_args()


def safe_filename(value: object, max_length: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "module")[:max_length]


def has_complete_species(value: object) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "n/a", "unknown", "unclassified"}:
        return False
    lower = text.casefold()
    incomplete_tokens = [
        " sp.",
        " spp.",
        " bacterium",
        " bacteria",
        " uncultured",
        " metagenome",
        " candidate ",
        " unidentified",
        " unclassified",
        " unknown",
        " cf.",
        " aff.",
    ]
    if any(token in f" {lower} " for token in incomplete_tokens):
        return False
    parts = text.replace("_", " ").split()
    return len(parts) >= 2


def clean_taxonomy_value(value: object) -> str:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "na", "n/a", "unknown", "unclassified"}:
        return ""
    return text


def most_common_text(values: pd.Series) -> str:
    cleaned = values.map(clean_taxonomy_value)
    cleaned = cleaned.loc[cleaned.ne("")]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    return str(counts.index[0])


def family_species_label(record: pd.Series) -> str:
    family = clean_taxonomy_value(record.get("Family", ""))
    species = clean_taxonomy_value(record.get("Species", ""))
    return f"{family} {species}".strip() if family else species


def wrap_function_label(label: object, width: int = 36) -> str:
    text = str(label).strip()
    match = re.match(r"^(.*)\s+(\(K\d+\))$", text)
    if match:
        name, ko_id = match.groups()
        wrapped = textwrap.wrap(name, width=width, break_long_words=False, break_on_hyphens=True)
        return "\n".join(wrapped + [ko_id])
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=True))


def load_sag_xpg_species_metadata(results_dir: Path) -> pd.DataFrame:
    metadata = read_tsv(results_dir / "metadata.tsv")
    for column in ["bin_id", "original_bin_id", "category", "sample_id"]:
        if column not in metadata.columns:
            raise ValueError(f"metadata.tsv is missing required column: {column}")
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.loc[metadata["category"].eq("xPG_SAGs")].copy()
    taxonomy = load_taxonomy_table(metadata, results_dir)
    metadata = metadata.merge(taxonomy, on=["category", "sample_id", "original_bin_id"], how="left")
    for column in TAXONOMY_ORDER_COLUMNS:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = add_quality_tier(metadata)
    metadata = metadata.loc[metadata["quality_tier"].isin(["high", "medium"])].copy()
    metadata["Species"] = metadata["Species"].fillna("").astype(str).str.strip()
    metadata = metadata.loc[metadata["Species"].map(has_complete_species)].copy()
    return metadata


def load_cycle_modules_with_labels(results_dir: Path, cycle: str) -> pd.DataFrame:
    module_path = results_dir / "SI_kofam.ko_to_module.tsv"
    cycle_modules, _module_summary, extra_ko_names = prepare_cycle_modules(results_dir, module_path, cycle)
    selected_kos = set(cycle_modules["ko"].astype(str))
    if cycle == "mobile_genetic_elements" or cycle in KEYWORD_CYCLE_PATTERNS:
        ko_names = load_ko_definition_map_rg(results_dir / "metadata.tsv", selected_kos)
    else:
        ko_names = load_ko_definition_map(results_dir / "metadata.tsv", selected_kos)
    ko_names.update({ko: name for ko, name in extra_ko_names.items() if ko not in ko_names})
    return attach_ko_labels(cycle_modules, ko_names)


def species_presence_absence_matrix(
    metadata: pd.DataFrame,
    presence: pd.DataFrame,
    module_data: pd.DataFrame,
    min_genomes_per_species: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ko_order = module_data.drop_duplicates("ko").loc[:, ["ko", "ko_label", "ko_order"]].sort_values("ko_order")
    species_counts = (
        metadata.groupby("Species", as_index=False)
        .agg(
            n_genomes=("bin_id", "nunique"),
            **{column: (column, most_common_text) for column in TAXONOMY_ORDER_COLUMNS if column != "Species"},
        )
        .loc[lambda frame: frame["n_genomes"].ge(min_genomes_per_species)]
    )
    if species_counts.empty:
        return pd.DataFrame(), species_counts
    species_counts["taxon_label"] = species_counts.apply(family_species_label, axis=1)
    species_counts = species_counts.sort_values(TAXONOMY_ORDER_COLUMNS, kind="mergesort")
    selected_species = set(species_counts["Species"].astype(str))
    bin_species = metadata.loc[metadata["Species"].isin(selected_species), ["bin_id", "Species"]].drop_duplicates()
    present_counts = (
        presence.merge(bin_species, on="bin_id", how="inner")
        .groupby(["Species", "ko"], as_index=False)
        .agg(n_present=("bin_id", "nunique"))
    )
    grid = (
        species_counts.loc[:, ["Species", "n_genomes"]]
        .merge(pd.DataFrame({"ko": ko_order["ko"].tolist()}), how="cross")
        .merge(present_counts, on=["Species", "ko"], how="left")
    )
    grid["n_present"] = grid["n_present"].fillna(0).astype(int)
    grid["present"] = grid["n_present"].gt(0).astype(int)
    species_order = species_counts["Species"].tolist()
    species_labels = species_counts.set_index("Species").loc[species_order, "taxon_label"].tolist()
    matrix = grid.pivot(index="Species", columns="ko", values="present").reindex(
        index=species_order,
        columns=ko_order["ko"].tolist(),
    )
    matrix.index = species_labels
    matrix.columns = [wrap_function_label(label) for label in ko_order.set_index("ko").loc[matrix.columns, "ko_label"]]
    return matrix, species_counts


def draw_clustermap(matrix: pd.DataFrame, output_base: Path, title: str) -> list[Path]:
    if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        return []
    sns = __import__("seaborn")
    plt = ensure_plotting()
    row_height = 0.78
    col_width = 0.82
    width = min(64.0, max(20.0, matrix.shape[1] * col_width + 11.0))
    height = min(108.0, max(20.0, matrix.shape[0] * row_height + 12.0))
    grid = sns.clustermap(
        matrix,
        row_cluster=False,
        col_cluster=False,
        cmap="Greys",
        vmin=0,
        vmax=1,
        linewidths=0.25,
        linecolor="#d9d9d9",
        figsize=(width, height),
        dendrogram_ratio=(0.01, 0.01),
        cbar_pos=None,
    )
    grid.ax_heatmap.set_xlabel("", fontsize=MIN_FONT_SIZE)
    grid.ax_heatmap.set_ylabel("Family Species in taxonomic order", fontsize=MIN_FONT_SIZE)
    grid.ax_heatmap.tick_params(
        axis="x",
        labelsize=MIN_FONT_SIZE,
        rotation=90,
        pad=12,
        top=False,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    grid.ax_heatmap.xaxis.tick_bottom()
    for label in grid.ax_heatmap.get_xticklabels():
        label.set_rotation_mode("anchor")
        label.set_ha("right")
        label.set_va("center")
        label.set_linespacing(0.95)
    grid.ax_heatmap.tick_params(axis="y", labelsize=MIN_FONT_SIZE, labelleft=True, labelright=False)
    grid.ax_heatmap.yaxis.tick_left()
    grid.ax_heatmap.yaxis.set_label_position("left")
    if getattr(grid, "ax_cbar", None) is not None:
        grid.ax_cbar.set_visible(False)
    for text in grid.fig.findobj(match=lambda artist: hasattr(artist, "get_fontsize")):
        try:
            if float(text.get_fontsize()) < MIN_FONT_SIZE:
                text.set_fontsize(MIN_FONT_SIZE)
        except Exception:
            continue
    grid.fig.subplots_adjust(top=0.96, bottom=0.38, left=0.42, right=0.96)
    apply_plot_style()
    apply_figure_typography(grid.fig)
    grid.fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    grid.fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(grid.fig)
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def write_cycle_clustermaps(
    results_dir: Path,
    cycle: str,
    metadata: pd.DataFrame,
    min_genomes_per_species: int,
) -> list[Path]:
    cycle_modules = load_cycle_modules_with_labels(results_dir, cycle)
    selected_kos = set(cycle_modules["ko"].astype(str))
    selected_bins = set(metadata["bin_id"].astype(str))
    presence = load_selected_ko_presence(results_dir / "all_kofam.mapper.unique.tsv", selected_bins, selected_kos)
    if presence.empty:
        presence = pd.DataFrame(columns=["bin_id", "ko", "present"])

    output_dir = results_dir / "elemental_cycles" / cycle / "sag_xpg_species_clustermaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    summary_rows = []

    modules = (
        cycle_modules.loc[:, ["module_order", "module_id", "module_name", "source_module_id", "source_module_name"]]
        .drop_duplicates()
        .sort_values(["module_order", "module_id"])
    )
    for module_record in modules.itertuples(index=False):
        module_id = str(module_record.module_id)
        module_name = str(module_record.module_name)
        module_data = cycle_modules.loc[cycle_modules["module_id"].astype(str).eq(module_id)].sort_values("ko_order")
        matrix, species_counts = species_presence_absence_matrix(
            metadata,
            presence,
            module_data,
            min_genomes_per_species,
        )
        matrix_path = output_dir / f"{safe_filename(module_id)}_{safe_filename(module_name)}.presence_absence_matrix.tsv"
        matrix.to_csv(matrix_path, sep="\t")
        written.append(matrix_path)
        title = f"{CYCLE_LABELS.get(cycle, cycle)}: {module_id} {module_name} SAG-xPG species KO presence/absence"
        output_base = output_dir / f"{safe_filename(module_id)}_{safe_filename(module_name)}"
        figure_paths = draw_clustermap(matrix, output_base, title)
        written.extend(figure_paths)
        summary_rows.append(
            {
                "cycle": cycle,
                "module_id": module_id,
                "module_name": module_name,
                "n_species": int(matrix.shape[0]),
                "n_kos": int(matrix.shape[1]),
                "figure_written": bool(figure_paths),
                "matrix_path": str(matrix_path),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / f"kofam_{cycle}_sag_xpg_species_clustermap_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    written.append(summary_path)
    return written


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    metadata = load_sag_xpg_species_metadata(results_dir)
    if metadata.empty:
        raise ValueError("No medium-or-better xPG_SAGs with complete Species names were found.")
    written: list[Path] = []
    for cycle in args.cycles:
        cycle_written = write_cycle_clustermaps(
            results_dir,
            cycle,
            metadata,
            args.min_genomes_per_species,
        )
        written.extend(cycle_written)
        print(f"[done] {cycle}: wrote {len(cycle_written)} files")
    print("[done] wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
