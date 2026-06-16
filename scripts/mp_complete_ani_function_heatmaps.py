#!/usr/bin/env python3
"""Functional heatmaps for exact species represented by all genome categories."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_metapathways_wrapper import (  # noqa: E402
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    category_order,
    ensure_plotting,
    matching_id_aliases,
    sanitize_label,
    save_figure,
)


DEFAULT_MP_DIR = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98"
)
DEFAULT_COMPONENT_MEMBERS = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_genome_atlas_gunc_98/"
    "tables/components/best_sets_review_component_members.tsv"
)
DEFAULT_OUT_DIR = DEFAULT_MP_DIR / "complete_ani98_category_function_summary"
REQUIRED_CATEGORIES = ["MAGs", "SAGs", "xPGs_MAGs", "xPGs_SAGs"]
MP_CATEGORY_MAP = {"xPG_MAGs": "xPGs_MAGs", "xPG_SAGs": "xPGs_SAGs"}
TIER_RANK = {"high": 3, "medium": 2, "low": 1, "failed": 0, "": -1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create functional mode heatmaps from exact species that contain "
            "MAGs, SAGs, MAG-xPGs, and SAG-xPGs, with at least one strict high genome."
        )
    )
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--component-members", type=Path, default=DEFAULT_COMPONENT_MEMBERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="metapathways_batch")
    parser.add_argument("--purge", action="store_true", help="Remove the output directory before writing.")
    return parser.parse_args()


def normalize_category(value: object) -> str:
    text = str(value).strip()
    return MP_CATEGORY_MAP.get(text, text)


def clean_taxon(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "unknown", "unclassified"}:
        return ""
    return text


def clean_join(values) -> str:
    cleaned = sorted({clean_taxon(value) for value in values if clean_taxon(value)})
    return "; ".join(cleaned)


def mode_specs(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    specs = [
        ("marker_specific", "marker", "specific_gene_count", "Specific marker genes"),
        ("marker_generic", "marker", "generic_gene_count", "Generic marker genes"),
        ("reference_mode", "reference_mode", "accession_count", "Reference accessions"),
    ]
    return [
        spec
        for spec in specs
        if any(f"{spec[1]}_{mode_id}_{spec[2]}" in frame.columns for mode_id in ELEMENTAL_MODE_ORDER)
    ]


def load_mp_genomes(mp_dir: Path, prefix: str) -> pd.DataFrame:
    path = mp_dir / "tables" / "summary" / f"{sanitize_label(prefix)}_genome_summary.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"sample", "category", "genome_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    frame["category"] = frame["category"].map(normalize_category)
    frame["sample"] = frame["sample"].astype(str).str.strip()
    frame["genome_id"] = frame["genome_id"].astype(str).str.strip()
    return frame


def load_complete_component_members(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "best_set_scope",
        "best_set_name",
        "component_id",
        "category",
        "sample",
        "Genome_Id",
        "Bin Id",
        "pre_ani_bin_key",
        "Completeness",
        "Contamination",
        "qscore",
        "mimag_tier",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "gunc_strict_assessment_label",
        "component_member_count",
        "component_categories",
        "component_samples",
    ]
    members = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    members = members.loc[
        members["best_set_scope"].astype(str).eq("global")
        & members["best_set_name"].astype(str).eq("best_of_best")
    ].copy()
    members["category"] = members["category"].map(normalize_category)
    members["sample"] = members["sample"].astype(str).str.strip()
    members["Genome_Id"] = members["Genome_Id"].astype(str).str.strip()
    for column in ["Family", "Genus", "Species"]:
        members[column] = members[column].map(clean_taxon)
    members = members.loc[members["Species"].ne("")].copy()
    species_summary = (
        members.groupby("Species")
        .agg(
            categories=("category", lambda x: set(x.astype(str))),
            has_high=("mimag_tier", lambda x: any(str(value).strip().lower() == "high" for value in x)),
        )
        .reset_index()
    )
    complete_species = species_summary.loc[
        species_summary["categories"].map(lambda categories: set(REQUIRED_CATEGORIES).issubset(categories))
        & species_summary["has_high"].fillna(False),
        "Species",
    ].astype(str)
    complete = members.loc[members["Species"].isin(set(complete_species))].copy()
    complete = complete.rename(columns={"component_id": "ani_component_id"})
    complete["component_id"] = complete["Species"].astype(str)
    summary = (
        complete.groupby("component_id")
        .agg(
            n_genomes=("Genome_Id", "size"),
            n_samples=("sample", "nunique"),
            families=("Family", clean_join),
            genera=("Genus", clean_join),
            species=("Species", clean_join),
        )
        .reset_index()
    )
    counts = (
        complete.pivot_table(
            index="component_id",
            columns="category",
            values="Genome_Id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=REQUIRED_CATEGORIES, fill_value=0)
        .reset_index()
    )
    summary = summary.merge(counts, on="component_id", how="left")
    return complete, summary


def merge_mp_function_data(members: pd.DataFrame, mp_genomes: pd.DataFrame) -> pd.DataFrame:
    mp_working = mp_genomes.reset_index(drop=False).rename(columns={"index": "_mp_index"}).copy()
    alias_map = {}
    for row in mp_working[["_mp_index", "sample", "category", "genome_id"]].to_dict("records"):
        group_key = (str(row["sample"]).strip(), normalize_category(row["category"]))
        for alias in matching_id_aliases(row["genome_id"]):
            alias_map.setdefault((group_key, alias), set()).add(int(row["_mp_index"]))

    matched_indices = []
    match_methods = []
    match_counts = []
    for row in members.to_dict("records"):
        group_key = (str(row.get("sample", "")).strip(), normalize_category(row.get("category", "")))
        aliases = set()
        for column in ["Genome_Id", "Bin Id", "pre_ani_bin_key"]:
            aliases.update(matching_id_aliases(row.get(column, "")))
        candidates = set()
        for alias in aliases:
            candidates.update(alias_map.get((group_key, alias), set()))
        if len(candidates) == 1:
            matched_indices.append(next(iter(candidates)))
            match_methods.append("alias")
            match_counts.append(1)
        elif len(candidates) > 1:
            selected = sorted(candidates)[0]
            matched_indices.append(selected)
            match_methods.append("alias_first_sorted")
            match_counts.append(len(candidates))
        else:
            matched_indices.append(np.nan)
            match_methods.append("unmatched")
            match_counts.append(0)

    members_with_match = members.copy()
    members_with_match["_matched_mp_index"] = matched_indices
    members_with_match["_mp_match_method"] = match_methods
    members_with_match["_mp_match_count"] = match_counts
    missing = int(members_with_match["_matched_mp_index"].isna().sum())
    if missing:
        examples = members_with_match.loc[
            members_with_match["_matched_mp_index"].isna(),
            ["sample", "category", "Genome_Id", "pre_ani_bin_key"],
        ].head(10)
        raise ValueError(
            f"{missing} complete-component genomes could not be matched to the MP genome summary. "
            f"Examples: {examples.to_dict('records')}"
        )
    members_with_match["_matched_mp_index"] = members_with_match["_matched_mp_index"].astype(int)
    merged = members_with_match.merge(
        mp_working.drop(columns=["sample", "category"]),
        left_on="_matched_mp_index",
        right_on="_mp_index",
        how="left",
        suffixes=("", "_mp"),
    )
    for column in ["Completeness", "Contamination", "qscore"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["tier_rank"] = (
        merged["mimag_tier"].astype(str).str.lower().str.strip().map(TIER_RANK).fillna(-1).astype(float)
    )
    merged["component_taxon_label"] = merged.apply(component_taxon_label, axis=1)
    return merged


def component_taxon_label(row: pd.Series) -> str:
    genus = clean_taxon(row.get("Genus"))
    species = clean_taxon(row.get("Species"))
    family = clean_taxon(row.get("Family"))
    component = str(row.get("component_id", "")).strip()
    if genus and species:
        return f"{genus} | {species}"
    if species:
        return species
    if genus:
        return genus
    if family:
        return family
    return component


def representative_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        by=[
            "component_id",
            "category",
            "tier_rank",
            "qscore",
            "Completeness",
            "Contamination",
            "genome_id",
        ],
        ascending=[True, True, False, False, False, True, True],
        kind="mergesort",
    )
    return (
        ranked.groupby(["component_id", "category"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def summarize_selection(frame: pd.DataFrame, selection_name: str) -> pd.DataFrame:
    rows = []
    for evidence_id, value_prefix, value_suffix, evidence_label in mode_specs(frame):
        for mode_id in ELEMENTAL_MODE_ORDER:
            column = f"{value_prefix}_{mode_id}_{value_suffix}"
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            base = frame[
                [
                    "component_id",
                    "component_taxon_label",
                    "Family",
                    "Genus",
                    "Species",
                    "category",
                    "sample",
                    "Genome_Id",
                    "genome_id",
                    "mimag_tier",
                ]
            ].copy()
            base["selection"] = selection_name
            base["evidence"] = evidence_id
            base["evidence_label"] = evidence_label
            base["mode_id"] = mode_id
            base["mode_label"] = ELEMENTAL_MODE_LABELS.get(mode_id, mode_id)
            base["value"] = values
            rows.append(base)
    if not rows:
        return pd.DataFrame()
    long_df = pd.concat(rows, ignore_index=True)
    group_cols = [
        "selection",
        "component_id",
        "component_taxon_label",
        "Family",
        "Genus",
        "Species",
        "category",
        "evidence",
        "evidence_label",
        "mode_id",
        "mode_label",
    ]
    return (
        long_df.groupby(group_cols, dropna=False)
        .agg(
            mean_value=("value", "mean"),
            median_value=("value", "median"),
            max_value=("value", "max"),
            n_genomes=("Genome_Id", "nunique"),
            strict_mimag_tiers=("mimag_tier", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )


def ordered_components(summary: pd.DataFrame) -> list[str]:
    component_order = (
        summary[["component_id", "component_taxon_label"]]
        .drop_duplicates()
        .sort_values(["component_taxon_label", "component_id"], kind="mergesort")
    )
    return component_order["component_id"].astype(str).tolist()


def heatmap_text_color(value: float, vmax: float) -> str:
    if vmax <= 0:
        return "black"
    return "white" if value / vmax >= 0.58 else "black"


def plot_selection_heatmap(summary: pd.DataFrame, output_base: Path, selection_label: str, evidence_id: str) -> bool:
    subset = summary.loc[summary["evidence"].astype(str).eq(evidence_id)].copy()
    if subset.empty:
        return False
    components = ordered_components(subset)
    categories = [category for category in REQUIRED_CATEGORIES if subset["category"].astype(str).eq(category).any()]
    modes = [mode_id for mode_id in ELEMENTAL_MODE_ORDER if subset["mode_id"].astype(str).eq(mode_id).any()]
    labels = (
        subset[["component_id", "component_taxon_label"]]
        .drop_duplicates()
        .groupby("component_id")["component_taxon_label"]
        .agg(clean_join)
        .reset_index()
        .set_index("component_id")
        .reindex(components)["component_taxon_label"]
        .fillna(pd.Series(components, index=components))
        .astype(str)
        .tolist()
    )
    if not components or not categories or not modes:
        return False
    vmax = max(1.0, float(pd.to_numeric(subset["mean_value"], errors="coerce").max()))
    plt = ensure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(15, max(7.5, len(components) * 0.28 + 2.0)), squeeze=False, sharey=True)
    axes_flat = axes.ravel()
    cmap = plt.colormaps["Greys"].copy() if hasattr(plt, "colormaps") else plt.cm.get_cmap("Greys").copy()
    cmap.set_bad(color="#ffffff")
    image = None
    for ax, category in zip(axes_flat, categories):
        matrix = (
            subset.loc[subset["category"].astype(str).eq(category)]
            .pivot_table(index="component_id", columns="mode_id", values="mean_value", aggfunc="mean")
            .reindex(index=components, columns=modes)
        )
        image = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(category)
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels([ELEMENTAL_MODE_LABELS.get(mode, mode) for mode in modes], rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(components)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("Functional mode")
        ax.set_ylabel("Exact species")
    for idx in range(len(categories), len(axes_flat)):
        axes_flat[idx].axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes_flat[: len(categories)].tolist(), fraction=0.018, pad=0.015)
        cbar.set_label("Mean count")
    evidence_label = str(subset["evidence_label"].dropna().iloc[0])
    fig.suptitle(f"{selection_label}: {evidence_label} in HQ-anchored exact species sets", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Rows are exact species with all four categories and at least one strict high genome; categories are faceted.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.22, right=0.94, bottom=0.20, top=0.91, wspace=0.08, hspace=0.42)
    save_figure(fig, str(output_base))
    return True


def write_outputs(out_dir: Path, complete: pd.DataFrame, component_summary: pd.DataFrame, merged: pd.DataFrame) -> list[Path]:
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots" / "heatmap"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    wrote = []

    component_path = table_dir / "hq_species_complete_four_category_taxa.tsv"
    component_summary.to_csv(component_path, sep="\t", index=False)
    wrote.append(component_path)

    input_path = table_dir / "hq_species_complete_four_category_genomes.tsv"
    merged.to_csv(input_path, sep="\t", index=False)
    wrote.append(input_path)

    selections = {
        "representative": representative_rows(merged),
        "average": merged.copy(),
    }
    all_summaries = []
    for selection, frame in selections.items():
        selection_path = table_dir / f"{selection}_input_genomes.tsv"
        frame.to_csv(selection_path, sep="\t", index=False)
        wrote.append(selection_path)
        summary = summarize_selection(frame, selection)
        summary_path = table_dir / f"{selection}_functional_mode_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        wrote.append(summary_path)
        all_summaries.append(summary)
        for evidence_id in summary["evidence"].dropna().astype(str).unique():
            matrix = summary.loc[summary["evidence"].astype(str).eq(evidence_id)].pivot_table(
                index=["component_id", "component_taxon_label", "mode_id", "mode_label"],
                columns="category",
                values="mean_value",
                aggfunc="mean",
            )
            matrix = matrix.reindex(columns=REQUIRED_CATEGORIES)
            matrix_path = table_dir / f"{selection}_{evidence_id}_matrix.tsv"
            matrix.reset_index().to_csv(matrix_path, sep="\t", index=False)
            wrote.append(matrix_path)
            plot_base = plot_dir / f"{selection}_{evidence_id}_heatmap"
            if plot_selection_heatmap(
                summary,
                plot_base,
                "One representative per species/category" if selection == "representative" else "Average of all genomes per species/category",
                evidence_id,
            ):
                wrote.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_path = table_dir / "combined_representative_average_functional_mode_summary.tsv"
    combined_summary.to_csv(combined_path, sep="\t", index=False)
    wrote.append(combined_path)
    return wrote


def main() -> None:
    args = parse_args()
    if args.purge and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    mp_genomes = load_mp_genomes(args.mp_dir, args.prefix)
    complete, component_summary = load_complete_component_members(args.component_members)
    merged = merge_mp_function_data(complete, mp_genomes)
    wrote = write_outputs(args.out_dir, complete, component_summary, merged)
    print(f"mp_genome_rows={len(mp_genomes)}")
    print(f"complete_components={component_summary.shape[0]}")
    print(f"complete_component_genomes={len(merged)}")
    print("strict_mimag_tiers")
    print(merged["mimag_tier"].value_counts().to_string())
    print(f"wrote={len(wrote)}")
    for path in wrote:
        print(path)


if __name__ == "__main__":
    main()
