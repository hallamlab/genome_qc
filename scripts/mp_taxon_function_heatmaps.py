#!/usr/bin/env python3
"""Build genus/family functional heatmaps from MetaPathways wrapper summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_metapathways_wrapper import (
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    _heatmap_text_color,
    canonical_method_label,
    ensure_plotting,
    ordered_methods,
    sanitize_label,
    save_figure,
)


DEFAULT_MP_DIR = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98"
)
DEFAULT_ATLAS_TAXONOMY = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_genome_atlas_gunc_98/"
    "tables/other/genome_quality_annotated.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average MetaPathways functional marker/reference-mode evidence at genus "
            "and family levels and draw compact category-faceted heatmaps."
        )
    )
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--atlas-taxonomy", type=Path, default=DEFAULT_ATLAS_TAXONOMY)
    parser.add_argument("--prefix", default="metapathways_batch")
    parser.add_argument("--max-rows", type=int, default=45)
    parser.add_argument(
        "--annotate-cells",
        action="store_true",
        help="Draw numeric values in cells. Off by default to keep taxon-level heatmaps readable.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def best_genomes_path(mp_dir: Path, prefix: str) -> Path:
    return (
        mp_dir
        / "species_category_comparison"
        / "best_vs_best"
        / "tables"
        / "pathway"
        / f"{sanitize_label(prefix)}_species_category_best_genomes.tsv"
    )


def clean_taxonomy_value(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "unclassified", "unknown"}:
        return ""
    return text


def load_taxonomy_lookup(path: Path) -> pd.DataFrame:
    taxonomy = pd.read_csv(path, sep="\t", usecols=["Family", "Genus", "Species"])
    for column in ["Family", "Genus", "Species"]:
        taxonomy[column] = taxonomy[column].map(clean_taxonomy_value)
    taxonomy = taxonomy.loc[taxonomy["Species"].ne("")].copy()
    taxonomy = taxonomy.drop_duplicates(subset=["Species"], keep="first")
    return taxonomy


def load_best_genomes(mp_dir: Path, prefix: str, taxonomy_path: Path) -> pd.DataFrame:
    path = best_genomes_path(mp_dir, prefix)
    if not path.exists():
        raise FileNotFoundError(path)
    best = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"species_key", "category", "genome_id"}
    missing = sorted(required - set(best.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    taxonomy = load_taxonomy_lookup(taxonomy_path)
    best["species_key"] = best["species_key"].astype(str).str.strip()
    best["category"] = best["category"].astype(str).str.strip().map(canonical_method_label)
    merged = best.merge(taxonomy, left_on="species_key", right_on="Species", how="left")
    for column in ["Family", "Genus", "Species"]:
        merged[column] = merged[column].map(clean_taxonomy_value)
    return merged


def evidence_specs(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    specs = [
        ("marker_specific", "marker", "specific_gene_count", "Specific marker genes"),
        ("marker_generic", "marker", "generic_gene_count", "Generic marker genes"),
        ("reference_mode", "reference_mode", "accession_count", "Reference accessions"),
    ]
    available = []
    for evidence_id, prefix, suffix, label in specs:
        columns = [f"{prefix}_{mode_id}_{suffix}" for mode_id in ELEMENTAL_MODE_ORDER]
        if any(column in frame.columns for column in columns):
            available.append((evidence_id, prefix, suffix, label))
    return available


def aggregate_evidence(
    frame: pd.DataFrame,
    rank: str,
    evidence_id: str,
    value_prefix: str,
    value_suffix: str,
) -> pd.DataFrame:
    rows = []
    working = frame.loc[frame[rank].astype(str).str.strip().ne("")].copy()
    if working.empty:
        return pd.DataFrame()

    for mode_id in ELEMENTAL_MODE_ORDER:
        column = f"{value_prefix}_{mode_id}_{value_suffix}"
        if column not in working.columns:
            continue
        values = pd.to_numeric(working[column], errors="coerce").fillna(0.0)
        temp = working[[rank, "category", "species_key", "genome_id"]].copy()
        temp["mode_id"] = mode_id
        temp["mode_label"] = ELEMENTAL_MODE_LABELS.get(mode_id, mode_id)
        temp["value"] = values
        rows.append(temp)
    if not rows:
        return pd.DataFrame()

    long_df = pd.concat(rows, ignore_index=True)
    summary = (
        long_df.groupby([rank, "category", "mode_id", "mode_label"], dropna=False)
        .agg(
            mean_count=("value", "mean"),
            median_count=("value", "median"),
            max_count=("value", "max"),
            n_genomes=("genome_id", "nunique"),
            n_species=("species_key", "nunique"),
        )
        .reset_index()
        .rename(columns={rank: "taxon"})
    )
    summary["rank"] = rank
    summary["evidence"] = evidence_id
    return summary[
        [
            "rank",
            "taxon",
            "category",
            "mode_id",
            "mode_label",
            "evidence",
            "mean_count",
            "median_count",
            "max_count",
            "n_genomes",
            "n_species",
        ]
    ]


def select_taxa(summary: pd.DataFrame, max_rows: int) -> list[str]:
    score = (
        summary.groupby("taxon", dropna=False)
        .agg(n_species=("n_species", "max"), n_genomes=("n_genomes", "max"), signal=("mean_count", "mean"))
        .reset_index()
        .sort_values(["n_species", "n_genomes", "signal", "taxon"], ascending=[False, False, False, True])
    )
    taxa = score["taxon"].astype(str).tolist()
    if max_rows and max_rows > 0:
        taxa = taxa[:max_rows]
    return sorted(taxa)


def plot_faceted_heatmap(
    summary: pd.DataFrame,
    output_base: Path,
    rank: str,
    evidence_label: str,
    max_rows: int,
    annotate_cells: bool = False,
) -> bool:
    if summary.empty:
        return False
    taxa = select_taxa(summary, max_rows=max_rows)
    categories = ordered_methods(summary["category"].astype(str).tolist())
    modes = [mode_id for mode_id in ELEMENTAL_MODE_ORDER if summary["mode_id"].astype(str).eq(mode_id).any()]
    if not taxa or not categories or not modes:
        return False

    vmax = max(1.0, float(pd.to_numeric(summary["mean_count"], errors="coerce").max()))
    plt = ensure_plotting()
    n_cols = min(2, len(categories))
    n_rows = int(np.ceil(len(categories) / float(n_cols)))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(12.5, n_cols * len(modes) * 0.42), max(6.2, n_rows * len(taxa) * 0.13 + 1.9)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )
    axes_flat = axes.ravel()
    cmap = plt.colormaps["Greys"].copy() if hasattr(plt, "colormaps") else plt.cm.get_cmap("Greys").copy()
    cmap.set_bad(color="#ffffff")
    last_image = None

    for ax, category in zip(axes_flat, categories):
        subset = summary.loc[summary["category"].astype(str).eq(category)].copy()
        matrix = (
            subset.pivot_table(index="taxon", columns="mode_id", values="mean_count", aggfunc="mean")
            .reindex(index=taxa, columns=modes)
            .astype(float)
        )
        last_image = ax.imshow(matrix.values, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_title(category)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels([ELEMENTAL_MODE_LABELS.get(mode, mode) for mode in modes], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(taxa)))
        ax.set_yticklabels(taxa, fontsize=6)
        ax.set_xlabel("Functional mode")
        ax.set_ylabel(rank)
        if annotate_cells:
            for row_index in range(matrix.shape[0]):
                for col_index in range(matrix.shape[1]):
                    value = matrix.iat[row_index, col_index]
                    if pd.isna(value):
                        continue
                    text = f"{float(value):.1f}"
                    ax.text(
                        col_index,
                        row_index,
                        text,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=_heatmap_text_color(float(value), vmax),
                    )

    for index in range(len(categories), len(axes_flat)):
        axes_flat[index].axis("off")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes_flat[: len(categories)].tolist(), fraction=0.018, pad=0.015)
        cbar.set_label("Mean count per best representative")
    fig.suptitle(f"{rank}-level average {evidence_label} by category", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.012,
        "Rows are taxa; cells are category-specific means across best species-category representatives.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.18, right=0.94, bottom=0.20, top=0.90, wspace=0.10, hspace=0.42)
    save_figure(fig, str(output_base))
    return True


def main() -> None:
    args = parse_args()
    out_root = args.mp_dir / "taxon_function_summary"
    table_dir = out_root / "tables"
    plot_dir = out_root / "plots" / "heatmap"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    best = load_best_genomes(args.mp_dir, args.prefix, args.atlas_taxonomy)
    wrote = []
    best_out = table_dir / f"{sanitize_label(args.prefix)}_taxon_function_best_genomes.tsv"
    best.to_csv(best_out, sep="\t", index=False)
    wrote.append(best_out)

    all_summaries = []
    for rank in ["Family", "Genus"]:
        for evidence_id, value_prefix, value_suffix, evidence_label in evidence_specs(best):
            summary = aggregate_evidence(best, rank, evidence_id, value_prefix, value_suffix)
            if summary.empty:
                continue
            table_path = table_dir / f"{sanitize_label(args.prefix)}_{sanitize_label(rank.lower())}_{evidence_id}_summary.tsv"
            summary.to_csv(table_path, sep="\t", index=False)
            wrote.append(table_path)
            all_summaries.append(summary)
            plot_base = plot_dir / f"{sanitize_label(args.prefix)}_{sanitize_label(rank.lower())}_{evidence_id}_heatmap"
            if plot_faceted_heatmap(
                summary,
                plot_base,
                rank,
                evidence_label,
                max_rows=args.max_rows,
                annotate_cells=args.annotate_cells,
            ):
                wrote.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_out = table_dir / f"{sanitize_label(args.prefix)}_family_genus_function_summary.tsv"
        combined.to_csv(combined_out, sep="\t", index=False)
        wrote.append(combined_out)

    print(f"best_genomes={len(best)}")
    print(f"taxa_with_family={best['Family'].astype(str).str.strip().ne('').sum()}")
    print(f"taxa_with_genus={best['Genus'].astype(str).str.strip().ne('').sum()}")
    print(f"wrote={len(wrote)}")
    for path in wrote:
        print(path)


if __name__ == "__main__":
    main()
