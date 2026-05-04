#!/usr/bin/env python3

import itertools
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.enrichment_stats import compute_rao_score, run_enrichment_dataframe
from summarize_metapathways_genomes import (
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    ensure_plotting,
    save_figure,
)
from summarize_metapathways_wrapper import (
    annotate_bar_values,
    benjamini_hochberg_adjust,
    build_metapathways_lookup,
    build_method_effectiveness_summary,
    canonical_method_label,
    category_order,
    choose_atlas_file,
    compute_functional_evidence_scores,
    exact_binomial_test_two_sided,
    match_atlas_genomes_to_metapathways,
    matching_id_aliases,
    method_family_label,
    method_sort_key,
    plot_category_mode_support_heatmaps,
    plot_method_effectiveness_panel,
    plot_sample_representative_heatmap,
    method_token,
    method_variant_flag,
    metapathways_best_set_sort_spec,
    ordered_methods,
    pair_sample_value,
    prepare_metapathways_best_set_candidates,
    sanitize_label,
    significance_stars,
    taxonomy_species_is_informative,
    taxonomy_value_is_informative,
)

PAIRED_QC_METRICS = [
    ("qscore", "Qscore"),
    ("integrity_score", "Integrity"),
    ("recoverability_score", "Recoverability"),
    ("mimag_quality_index", "MIMAG quality index"),
]

PAIRED_FUNCTION_METRICS = [
    ("informative_annotation_fraction", "Informative annotation fraction"),
    ("pathway_input_fraction", "Pathway-input fraction"),
    ("pathway_support_fraction", "Pathway-support fraction"),
    ("total_pathways", "Inferred pathways"),
    ("median_pathway_score", "Median pathway score"),
    ("mean_reaction_coverage", "Mean reaction coverage"),
]

MATCHED_LINEAGE_FUNCTION_METRICS = [
    ("total_orfs", "Total ORFs"),
    ("annotated_orfs", "Annotated ORFs"),
    ("annotation_fraction", "Annotation fraction"),
    ("informative_annotation_orfs", "Informative annotation ORFs"),
    ("informative_annotation_fraction", "Informative annotation fraction"),
    ("uncertain_annotation_fraction", "Uncertain annotation fraction"),
    ("pathway_input_fraction", "Pathway-input fraction"),
    ("pathway_support_fraction", "Pathway-support fraction"),
    ("total_pathways", "Inferred pathways"),
    ("median_pathway_score", "Median pathway score"),
    ("mean_reaction_coverage", "Mean reaction coverage"),
    ("marker_supported_orfs", "Marker-supported ORFs"),
    ("reference_mode_supported_accessions", "Reference-supported accessions"),
]


def run_enrichment_dataframe_compat(
    input_df,
    *,
    compute_associated_groups_if_missing=False,
    allow_empty_associated_groups=False,
):
    if input_df is None or input_df.empty:
        return pd.DataFrame()
    if len(input_df) >= 2:
        return run_enrichment_dataframe(
            input_df,
            compute_associated_groups_if_missing=compute_associated_groups_if_missing,
            allow_empty_associated_groups=allow_empty_associated_groups,
        )
    row = input_df.iloc[0].copy()
    groups = [str(column).split("_", 1)[1] for column in input_df.columns if str(column).startswith("p_")]
    props = np.array([float(pd.to_numeric(row.get(f"p_{group}", 0), errors="coerce")) for group in groups], dtype=float)
    reps = np.array([float(pd.to_numeric(row.get(f"N_{group}", 0), errors="coerce")) for group in groups], dtype=float)
    score, pvalue = compute_rao_score(props, reps)
    record = row.to_dict()
    record["unadjusted_p_value"] = pvalue
    record["adjusted_q_value"] = pvalue
    record["enrichment_score"] = max(0.0, float(score))
    return pd.DataFrame([record])

def load_atlas_inputs(args, require_shared=True, require_annotated=False):
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve() if args.genome_atlas_dir else None
    safe_compare = sanitize_label(args.atlas_compare_column)
    explicit_shared_best_tsv = getattr(args, "atlas_shared_best_tsv", None)
    explicit_annotated_tsv = getattr(args, "atlas_annotated_tsv", None)

    shared_preferred = f"{sanitize_label(args.atlas_prefix)}_compare_{safe_compare}_shared_best_genomes.tsv"
    shared_best_path = None
    if require_shared or explicit_shared_best_tsv:
        shared_best_path = choose_atlas_file(
            atlas_dir=atlas_dir,
            explicit_path=explicit_shared_best_tsv,
            pattern="*_shared_best_genomes.tsv",
            preferred_token=shared_preferred,
            required=require_shared,
        )
    annotated_preferred = f"{sanitize_label(args.atlas_prefix)}_annotated.tsv"
    annotated_path = choose_atlas_file(
        atlas_dir=atlas_dir,
        explicit_path=explicit_annotated_tsv,
        pattern="*_annotated.tsv",
        preferred_token=annotated_preferred,
        required=require_annotated,
    )

    shared_best_df = pd.read_csv(shared_best_path, sep="\t") if shared_best_path else pd.DataFrame()
    annotated_df = pd.read_csv(annotated_path, sep="\t") if annotated_path else pd.DataFrame()
    return shared_best_path, annotated_path, shared_best_df, annotated_df

def has_atlas_shared_best(args):
    if getattr(args, "atlas_shared_best_tsv", None):
        return True
    if not args.genome_atlas_dir:
        return False
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve()
    if not atlas_dir.exists():
        return False
    return any(path.is_file() for path in atlas_dir.rglob("*_shared_best_genomes.tsv"))

def has_atlas_annotated(args):
    if getattr(args, "atlas_annotated_tsv", None):
        return True
    if not args.genome_atlas_dir:
        return False
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve()
    if not atlas_dir.exists():
        return False
    return any(path.is_file() for path in atlas_dir.rglob("*_annotated.tsv"))

def prepare_atlas_shared_best(shared_best_df, annotated_df, args):
    compare_column = args.atlas_compare_column
    sample_column = args.atlas_sample_column
    genome_id_column = args.atlas_genome_id_column
    required = [compare_column, genome_id_column, "component_id"]
    for column in required:
        if column not in shared_best_df.columns:
            raise ValueError(
                f"Atlas shared-best table is missing required column '{column}'. "
                f"Provide the correct --atlas-* column names."
            )

    atlas_df = shared_best_df.copy()
    atlas_df[compare_column] = (
        atlas_df[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    atlas_df[genome_id_column] = atlas_df[genome_id_column].astype(str).str.strip()
    if sample_column in atlas_df.columns:
        atlas_df[sample_column] = atlas_df[sample_column].astype(str).str.strip()
    else:
        atlas_df[sample_column] = ""

    qc_fill_columns = [
        "qscore",
        "integrity_score",
        "recoverability_score",
        "mimag_quality_index",
        "Completeness",
        "Contamination",
    ]
    if not annotated_df.empty and genome_id_column in annotated_df.columns:
        annotated = annotated_df.copy()
        annotated[genome_id_column] = annotated[genome_id_column].astype(str).str.strip()
        if compare_column in annotated.columns:
            annotated[compare_column] = (
                annotated[compare_column]
                .astype(str)
                .str.strip()
                .map(canonical_method_label)
            )
        if sample_column in annotated.columns:
            annotated[sample_column] = annotated[sample_column].astype(str).str.strip()
            merge_keys = [sample_column, compare_column, genome_id_column]
            available_merge_keys = [key for key in merge_keys if key in atlas_df.columns and key in annotated.columns]
            if len(available_merge_keys) < 2:
                available_merge_keys = [genome_id_column]
        else:
            available_merge_keys = [genome_id_column]

        keep_cols = list(dict.fromkeys(available_merge_keys + qc_fill_columns))
        keep_cols = [column for column in keep_cols if column in annotated.columns]
        if keep_cols:
            atlas_df = atlas_df.merge(
                annotated[keep_cols].drop_duplicates(),
                on=available_merge_keys,
                how="left",
                suffixes=("", "_annotated"),
            )
            for column in qc_fill_columns:
                annotated_col = f"{column}_annotated"
                if annotated_col in atlas_df.columns:
                    atlas_df[column] = atlas_df[column] if column in atlas_df.columns else np.nan
                    atlas_df[column] = atlas_df[column].where(atlas_df[column].notna(), atlas_df[annotated_col])
                    atlas_df = atlas_df.drop(columns=[annotated_col])
    return atlas_df

def prepare_atlas_species_source(shared_best_df, annotated_df, args):
    compare_column = args.atlas_compare_column
    sample_column = args.atlas_sample_column
    genome_id_column = args.atlas_genome_id_column

    if (
        not annotated_df.empty
        and compare_column in annotated_df.columns
        and genome_id_column in annotated_df.columns
    ):
        atlas_df = annotated_df.copy()
        source_label = "annotated"
    else:
        atlas_df = prepare_atlas_shared_best(shared_best_df, annotated_df, args)
        source_label = "shared_best"

    atlas_df[compare_column] = (
        atlas_df[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    atlas_df[genome_id_column] = atlas_df[genome_id_column].astype(str).str.strip()
    if sample_column in atlas_df.columns:
        atlas_df[sample_column] = atlas_df[sample_column].astype(str).str.strip()
    else:
        atlas_df[sample_column] = ""
    return atlas_df, source_label

def source_sample_value(row):
    for column in ["source_sample", "mp_sample", "_atlas_sample", "sample"]:
        value = str(row.get(column, "")).strip()
        if value and value.lower() != "nan" and value != "all_samples":
            return value
    return ""

def resolve_representative_taxonomy(row):
    rank_priority = ["Species", "Genus", "Family", "Order", "Class", "Phylum", "Domain"]

    species_value = str(row.get("Species", "")).strip()
    if taxonomy_species_is_informative(species_value):
        return "species", species_value

    for rank_column, value_column in [
        ("mp_taxonomy_display_rank", "mp_taxonomy_display_value"),
        ("taxonomy_display_rank", "taxonomy_display_value"),
    ]:
        rank_value = str(row.get(rank_column, "")).strip().lower()
        value = str(row.get(value_column, "")).strip()
        if rank_value == "species" and taxonomy_species_is_informative(value):
            return "species", value

    for rank_name in rank_priority[1:]:
        value = str(row.get(rank_name, "")).strip()
        if taxonomy_value_is_informative(value):
            return rank_name.lower(), value

    for rank_column, value_column in [
        ("mp_taxonomy_display_rank", "mp_taxonomy_display_value"),
        ("taxonomy_display_rank", "taxonomy_display_value"),
    ]:
        rank_value = str(row.get(rank_column, "")).strip().lower()
        value = str(row.get(value_column, "")).strip()
        if rank_value in {"genus", "family", "order", "class", "phylum", "domain"} and taxonomy_value_is_informative(value):
            return rank_value, value
    return "", ""

def build_species_representative_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()
    required = {"mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame()

    working = matched_df.copy()
    working["_pair_sample"] = working.apply(pair_sample_value, axis=1)
    taxonomy_labels = working.apply(resolve_representative_taxonomy, axis=1)
    working["_taxonomy_rank"] = taxonomy_labels.map(lambda item: item[0])
    working["_taxonomy_label"] = taxonomy_labels.map(lambda item: item[1])
    working[compare_column] = working[compare_column].astype(str).str.strip()
    working["mp_genome_id"] = working["mp_genome_id"].astype(str).str.strip()

    working = working.loc[
        working["_pair_sample"].ne("")
        & working["_taxonomy_label"].ne("")
        & working[compare_column].ne("")
        & working["mp_genome_id"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame()

    sort_columns = []
    ascending = []
    for column in [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "mp_informative_annotation_fraction",
        "mp_informative_annotation_orfs",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
        "mp_total_pathways",
        "qscore",
    ]:
        if column in working.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.extend(["mp_genome_id"])
    ascending.extend([True])
    working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")

    selected = (
        working.groupby(["_pair_sample", compare_column, "_taxonomy_rank", "_taxonomy_label"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["sample"] = selected["_pair_sample"].astype(str)
    selected["taxonomy_species"] = selected["_taxonomy_label"].astype(str)
    selected["taxonomy_representative_rank"] = selected["_taxonomy_rank"].astype(str)
    selected["taxonomy_representative_label"] = selected["_taxonomy_label"].astype(str)
    selected["component_id"] = (
        selected["_taxonomy_rank"].astype(str) + ":" + selected["_taxonomy_label"].astype(str)
    )
    selected["selection_scope"] = "sample_method_taxonomy"
    selected = selected.drop(columns=["_pair_sample", "_taxonomy_rank", "_taxonomy_label"])
    return selected

def build_all_species_linked_member_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()
    required = {"mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame()

    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    working = matched_df.copy()
    working["source_sample"] = working.apply(pair_sample_value, axis=1)
    working["linked_sample"] = "all_samples"
    working["category"] = (
        working[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working[compare_column] = working["category"]
    working["mp_genome_id"] = working["mp_genome_id"].astype(str).str.strip()
    for column in taxonomy_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()

    species_mask = working["Species"].map(taxonomy_species_is_informative)
    complete_lineage_mask = pd.Series(True, index=working.index)
    for column in taxonomy_columns:
        complete_lineage_mask &= working[column].map(taxonomy_value_is_informative)

    working = working.loc[
        working["source_sample"].astype(str).str.strip().ne("")
        & working["category"].astype(str).str.strip().ne("")
        & working["mp_genome_id"].astype(str).str.strip().ne("")
        & species_mask
        & complete_lineage_mask
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["lineage_key"] = working[taxonomy_columns].agg(";".join, axis=1)
    working["component_id"] = working["lineage_key"].astype(str)
    category_counts = (
        working.groupby("component_id", dropna=False)["category"]
        .transform(lambda values: pd.Series(values).astype(str).nunique())
    )
    working["sample"] = "all_samples"
    working["taxonomy_species"] = working["Species"].astype(str)
    working["taxonomy_representative_rank"] = "species"
    working["taxonomy_representative_label"] = working["Species"].astype(str)
    working["selection_scope"] = "all_species_taxonomy_members"
    working["linked_set_member_type"] = "all_species_members"
    working["linked_set_n_categories"] = category_counts.astype(int)
    working = working.sort_values(
        by=["component_id", "source_sample", "category", "mp_genome_id"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return working

def build_complete_lineage_category_representatives(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {"mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame(), pd.DataFrame()

    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    working = matched_df.copy()
    working["source_sample"] = working.apply(pair_sample_value, axis=1)
    working["sample"] = "all_samples"
    working["category"] = (
        working[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working[compare_column] = working["category"]
    working["mp_genome_id"] = working["mp_genome_id"].astype(str).str.strip()
    for column in taxonomy_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()

    complete_lineage_mask = pd.Series(True, index=working.index)
    for column in taxonomy_columns:
        validator = taxonomy_species_is_informative if column == "Species" else taxonomy_value_is_informative
        complete_lineage_mask &= working[column].map(validator)

    working = working.loc[
        working["source_sample"].astype(str).str.strip().ne("")
        & working["category"].astype(str).str.strip().ne("")
        & working["mp_genome_id"].astype(str).str.strip().ne("")
        & complete_lineage_mask
    ].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    working["lineage_key"] = working[taxonomy_columns].agg(";".join, axis=1)
    working["component_id"] = working["lineage_key"].astype(str)
    working["taxonomy_species"] = working["Species"].astype(str)
    working["taxonomy_representative_rank"] = "species"
    working["taxonomy_representative_label"] = working["Species"].astype(str)
    working["selection_scope"] = "all_samples_complete_lineage_category_best"
    working["linked_set_member_type"] = "best_per_lineage_category"

    sort_columns = []
    ascending = []
    for column in [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "mp_informative_annotation_fraction",
        "mp_informative_annotation_orfs",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
        "mp_total_pathways",
        "qscore",
    ]:
        if column in working.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.extend(["source_sample", "mp_genome_id"])
    ascending.extend([True, True])
    working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")

    representatives = (
        working.groupby(["lineage_key", "component_id", "category"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    lineage_category_counts = (
        representatives.groupby("lineage_key", dropna=False)["category"]
        .transform(lambda values: pd.Series(values).astype(str).nunique())
    )
    representatives["linked_set_n_categories"] = lineage_category_counts.astype(int)
    representatives = representatives.loc[representatives["linked_set_n_categories"].ge(2)].copy()
    if representatives.empty:
        return pd.DataFrame(), pd.DataFrame()

    lineage_summary = (
        representatives.groupby(
            ["lineage_key", "component_id", *taxonomy_columns],
            dropna=False,
        )
        .agg(
            n_categories=("category", lambda s: int(pd.Series(s).astype(str).nunique())),
            categories_present=("category", lambda s: ";".join(ordered_methods(pd.Series(s).astype(str).drop_duplicates().tolist()))),
            n_representatives=("mp_genome_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            mean_mimag_quality_index=("mimag_quality_index", "mean"),
            mean_integrity_score=("integrity_score", "mean"),
            mean_recoverability_score=("recoverability_score", "mean"),
            mean_qscore=("qscore", "mean"),
        )
        .reset_index()
        .sort_values(
            by=["n_categories", "mean_mimag_quality_index", "Species", "component_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    return representatives, lineage_summary

def normalize_linked_representatives_for_summary(representative_df, compare_column, max_lineages=None):
    if representative_df.empty:
        return pd.DataFrame()

    working = representative_df.copy()
    rename_map = {
        "mp_genome_id": "genome_id",
        compare_column: "category",
        "mp_informative_annotation_fraction": "informative_annotation_fraction",
        "mp_informative_annotation_orfs": "informative_annotation_orfs",
        "mp_total_pathways": "total_pathways",
        "mp_marker_supported_orfs": "marker_supported_orfs",
        "mp_reference_mode_supported_accessions": "reference_mode_supported_accessions",
    }
    working = working.rename(columns={src: dst for src, dst in rename_map.items() if src in working.columns})
    if "category" not in working.columns or "genome_id" not in working.columns:
        return pd.DataFrame()

    if "lineage_key" not in working.columns:
        taxonomy_columns = [column for column in ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"] if column in working.columns]
        if taxonomy_columns:
            working["lineage_key"] = working[taxonomy_columns].astype(str).agg(";".join, axis=1)
        else:
            return pd.DataFrame()
    working["lineage_label"] = working["Species"].astype(str).str.strip()
    working.loc[working["lineage_label"].eq(""), "lineage_label"] = working["lineage_key"].astype(str)
    working["lineage_label"] = working["lineage_label"].map(
        lambda value: value if len(str(value)) <= 52 else str(value)[:49] + "..."
    )
    if max_lineages is not None and int(max_lineages) > 0:
        lineage_order = (
            working.groupby(["lineage_key", "lineage_label"], dropna=False)
            .agg(
                n_categories=("category", lambda s: int(pd.Series(s).astype(str).nunique())),
                mean_mimag_quality_index=("mimag_quality_index", "mean"),
            )
            .reset_index()
            .sort_values(
                by=["n_categories", "mean_mimag_quality_index", "lineage_label", "lineage_key"],
                ascending=[False, False, True, True],
                kind="mergesort",
            )
            .head(int(max_lineages))
        )
        keep_keys = set(lineage_order["lineage_key"].astype(str))
        working = working.loc[working["lineage_key"].astype(str).isin(keep_keys)].copy()
    if working.empty:
        return pd.DataFrame()

    lineage_rank_map = {
        lineage_key: index + 1
        for index, lineage_key in enumerate(
            working.groupby(["lineage_key", "lineage_label"], dropna=False)
            .size()
            .reset_index()
            .sort_values(by=["lineage_label", "lineage_key"], kind="mergesort")["lineage_key"].astype(str).tolist()
        )
    }
    lineage_label_map = (
        working[["lineage_key", "lineage_label"]]
        .drop_duplicates()
        .assign(lineage_key=lambda df: df["lineage_key"].astype(str))
        .set_index("lineage_key")["lineage_label"]
        .to_dict()
    )
    working["sample"] = working["lineage_key"].astype(str).map(
        lambda key: f"{lineage_rank_map.get(str(key), 0):03d} {lineage_label_map.get(str(key), str(key))}"
    )
    scored_df, _score_metrics = compute_functional_evidence_scores(working)
    return scored_df

def build_genome_representative_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()
    required = {"mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame()

    working = matched_df.copy()
    working["_pair_sample"] = working.apply(pair_sample_value, axis=1)
    working[compare_column] = working[compare_column].astype(str).str.strip()
    working["mp_genome_id"] = working["mp_genome_id"].astype(str).str.strip()

    working = working.loc[
        working["_pair_sample"].ne("")
        & working[compare_column].ne("")
        & working["mp_genome_id"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame()

    sort_columns = []
    ascending = []
    for column in [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "mp_informative_annotation_fraction",
        "mp_informative_annotation_orfs",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
        "mp_total_pathways",
        "qscore",
    ]:
        if column in working.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.extend(["mp_genome_id"])
    ascending.extend([True])
    working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")

    selected = (
        working.groupby(["_pair_sample", compare_column, "mp_genome_id"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["sample"] = selected["_pair_sample"].astype(str)
    selected["component_id"] = selected["mp_genome_id"].astype(str)
    selected["selection_scope"] = "sample_method_genome"
    selected = selected.drop(columns=["_pair_sample"], errors="ignore")
    return selected

def build_paired_component_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()

    rows = []
    metric_specs = []
    for metric, label in PAIRED_QC_METRICS:
        if metric in matched_df.columns:
            metric_specs.append((metric, metric, label))
    for metric, label in PAIRED_FUNCTION_METRICS:
        prefixed = f"mp_{metric}"
        if prefixed in matched_df.columns:
            metric_specs.append((metric, prefixed, label))

    for component_id, group in matched_df.groupby("component_id", dropna=False):
        component_rows = group.dropna(subset=["mp_genome_id"]).copy()
        if component_rows.empty:
            continue
        for left_row, right_row in itertools.combinations(component_rows.to_dict("records"), 2):
            sample_left = pair_sample_value(left_row)
            sample_right = pair_sample_value(right_row)
            if sample_left and sample_right and sample_left != sample_right:
                continue
            pair_sample = sample_left or sample_right
            if not pair_sample:
                continue
            cat_left = str(left_row.get(compare_column, ""))
            cat_right = str(right_row.get(compare_column, ""))
            if not cat_left or not cat_right or cat_left == cat_right:
                continue
            if method_sort_key(cat_left) <= method_sort_key(cat_right):
                row_a = left_row
                row_b = right_row
            else:
                row_a = right_row
                row_b = left_row

            row = {
                "component_id": component_id,
                "category_a": str(row_a.get(compare_column, "")),
                "category_b": str(row_b.get(compare_column, "")),
                "sample": pair_sample,
                "sample_a": sample_left,
                "sample_b": sample_right,
                "source_sample_a": source_sample_value(row_a),
                "source_sample_b": source_sample_value(row_b),
                "atlas_genome_a": str(row_a.get("_atlas_genome_id", "")),
                "atlas_genome_b": str(row_b.get("_atlas_genome_id", "")),
                "metapathways_genome_a": str(row_a.get("mp_genome_id", "")),
                "metapathways_genome_b": str(row_b.get("mp_genome_id", "")),
                "metapathways_label_a": str(row_a.get("mp_genome_label", "")),
                "metapathways_label_b": str(row_b.get("mp_genome_label", "")),
            }
            for metric_key, metric_column, _ in metric_specs:
                value_a = pd.to_numeric(pd.Series([row_a.get(metric_column)]), errors="coerce").iat[0]
                value_b = pd.to_numeric(pd.Series([row_b.get(metric_column)]), errors="coerce").iat[0]
                row[f"{metric_key}_a"] = value_a
                row[f"{metric_key}_b"] = value_b
                row[f"{metric_key}_delta"] = (
                    value_a - value_b if pd.notna(value_a) and pd.notna(value_b) else np.nan
                )
            rows.append(row)

    paired_df = pd.DataFrame(rows)
    if not paired_df.empty:
        paired_df["category_pair"] = paired_df["category_a"] + " | " + paired_df["category_b"]
    return paired_df

def summarize_paired_deltas(paired_df):
    if paired_df.empty:
        return pd.DataFrame()
    delta_columns = [column for column in paired_df.columns if column.endswith("_delta")]
    rows = []
    for category_pair, group in paired_df.groupby("category_pair", dropna=False):
        for delta_column in delta_columns:
            series = pd.to_numeric(group[delta_column], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "category_pair": category_pair,
                    "metric": delta_column[: -len("_delta")],
                    "n_pairs": int(series.size),
                    "median_delta": float(series.median()),
                    "mean_delta": float(series.mean()),
                    "std_delta": float(series.std(ddof=1)) if series.size > 1 else 0.0,
                    "positive_fraction": float((series > 0).mean()),
                }
            )
    return pd.DataFrame(rows)

def heatmap_text_color(value, vmax):
    if vmax <= 0:
        return "black"
    return "white" if float(value) >= (0.6 * float(vmax)) else "black"

def plot_pair_count_heatmap(paired_df, output_base):
    ensure_plotting()
    if paired_df.empty:
        return False

    categories = ordered_methods(
        set(paired_df["category_a"].astype(str)).union(set(paired_df["category_b"].astype(str)))
    )
    matrix = pd.DataFrame(0, index=categories, columns=categories, dtype=float)
    pair_counts = (
        paired_df.groupby(["category_a", "category_b"], dropna=False)
        .size()
        .reset_index(name="pair_count")
    )
    for row in pair_counts.itertuples(index=False):
        matrix.at[row.category_a, row.category_b] = float(row.pair_count)
        matrix.at[row.category_b, row.category_a] = float(row.pair_count)

    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(figsize=(max(7, len(categories) * 0.8), max(6, len(categories) * 0.7)))
    vmax = max(1.0, float(np.nanmax(matrix.values)))
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=90)
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Category")
    ax.set_ylabel("Category")
    ax.set_title("Like-to-like shared-best pair counts")
    for row_index in range(len(categories)):
        for col_index in range(len(categories)):
            value = int(round(float(matrix.iat[row_index, col_index])))
            color = heatmap_text_color(value, vmax)
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Pair count")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.16)
    save_figure(fig, output_base)
    return True

def plot_paired_delta_heatmaps(paired_df, output_base):
    ensure_plotting()
    if paired_df.empty:
        return False
    delta_columns = [column for column in paired_df.columns if column.endswith("_delta")]
    if not delta_columns:
        return False

    categories = ordered_methods(
        set(paired_df["category_a"].astype(str)).union(set(paired_df["category_b"].astype(str)))
    )
    metric_columns = []
    for metric, _ in PAIRED_QC_METRICS + PAIRED_FUNCTION_METRICS:
        column = f"{metric}_delta"
        if column in delta_columns and pd.to_numeric(paired_df[column], errors="coerce").notna().any():
            metric_columns.append(column)
    if not metric_columns:
        return False

    n_cols = min(4, len(metric_columns))
    n_rows = int(np.ceil(len(metric_columns) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, n_cols * 4.0), max(5.2, n_rows * 3.6)),
        squeeze=False,
    )

    label_lookup = {key: label for key, label in PAIRED_QC_METRICS + PAIRED_FUNCTION_METRICS}
    for index, delta_column in enumerate(metric_columns):
        row_index = index // n_cols
        col_index = index % n_cols
        ax = axes[row_index, col_index]
        metric = delta_column[: -len("_delta")]
        matrix = pd.DataFrame(np.nan, index=categories, columns=categories, dtype=float)
        for pair_row in paired_df[["category_a", "category_b", delta_column]].dropna().itertuples(index=False):
            matrix.at[pair_row.category_a, pair_row.category_b] = float(pair_row[2])
            matrix.at[pair_row.category_b, pair_row.category_a] = -float(pair_row[2])
        for category in categories:
            matrix.at[category, category] = 0.0

        finite = np.abs(matrix.values[np.isfinite(matrix.values)])
        vmax = max(1e-9, float(np.nanmax(finite)) if finite.size else 1.0)
        image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(categories)))
        ax.set_xticklabels(categories, rotation=90)
        ax.set_yticks(np.arange(len(categories)))
        ax.set_yticklabels(categories)
        ax.set_title(label_lookup.get(metric, metric))
        for r in range(len(categories)):
            for c in range(len(categories)):
                value = matrix.iat[r, c]
                if not np.isfinite(value):
                    continue
                ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Median delta (row - column)")

    for index in range(len(metric_columns), n_rows * n_cols):
        row_index = index // n_cols
        col_index = index % n_cols
        axes[row_index, col_index].axis("off")

    fig.suptitle("Like-to-like paired metric differences", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_base)
    return True

def matched_lineage_function_metric_specs(frame):
    specs = []
    for metric_id, metric_label in MATCHED_LINEAGE_FUNCTION_METRICS:
        column = f"mp_{metric_id}"
        if column in frame.columns and pd.to_numeric(frame[column], errors="coerce").notna().any():
            specs.append((metric_id, metric_label, column))
    return specs

def build_matched_lineage_function_category_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()
    if "component_id" not in matched_df.columns or compare_column not in matched_df.columns or "mp_genome_id" not in matched_df.columns:
        return pd.DataFrame()

    working = prepare_metapathways_best_set_candidates(matched_df, compare_column)
    if working.empty:
        return pd.DataFrame()
    helper_columns = [column for column in working.columns if str(column).startswith("__")]
    if helper_columns:
        working = working.drop(columns=helper_columns, errors="ignore")
    sort_columns, ascending = metapathways_best_set_sort_spec(working)
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    category_table = (
        working.groupby(["sample", "component_id", "category"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    category_table["lineage_component_id"] = category_table["component_id"].astype(str)
    category_table["selection_scope"] = "sample_component_category"
    category_table["selection_metric"] = (
        "coarse(mimag_quality_index)>gunc(class_gate_only)>"
        "coarse(integrity_score,recoverability_score)>16S_presence>"
        "mimag_quality_index>integrity_score>recoverability_score>"
        "mp_informative_annotation_fraction>mp_informative_annotation_orfs>"
        "mp_marker_supported_orfs>mp_reference_mode_supported_accessions>"
        "mp_total_pathways>qscore"
    )
    return category_table

def build_matched_lineage_function_pair_tables(category_table):
    if category_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    metric_specs = matched_lineage_function_metric_specs(category_table)
    if not metric_specs:
        return pd.DataFrame(), pd.DataFrame()

    wide_rows = []
    long_rows = []
    for (sample_value, component_id), group in category_table.groupby(["sample", "component_id"], dropna=False):
        categories = ordered_methods(group["category"].astype(str).tolist())
        if len(categories) < 2:
            continue
        category_rows = {
            str(row.get("category", "")).strip(): row
            for row in group.to_dict("records")
            if str(row.get("category", "")).strip()
        }
        for idx_a, raw_category_a in enumerate(categories):
            for raw_category_b in categories[idx_a + 1:]:
                category_a, category_b, _swapped, comparison_class = normalized_reporting_pair(raw_category_a, raw_category_b)
                if category_a not in category_rows or category_b not in category_rows:
                    continue
                row_a = category_rows[category_a]
                row_b = category_rows[category_b]
                wide_row = {
                    "sample": str(sample_value),
                    "component_id": str(component_id),
                    "category_a": category_a,
                    "category_b": category_b,
                    "category_pair": f"{category_a} -> {category_b}",
                    "comparison_class": comparison_class,
                    "genome_a": str(row_a.get("mp_genome_id", row_a.get("Genome_Id", ""))),
                    "genome_b": str(row_b.get("mp_genome_id", row_b.get("Genome_Id", ""))),
                    "atlas_genome_a": str(row_a.get("_atlas_genome_id", row_a.get("atlas_genome_id", ""))),
                    "atlas_genome_b": str(row_b.get("_atlas_genome_id", row_b.get("atlas_genome_id", ""))),
                }
                for metric_id, metric_label, column in metric_specs:
                    value_a = pd.to_numeric(pd.Series([row_a.get(column)]), errors="coerce").iat[0]
                    value_b = pd.to_numeric(pd.Series([row_b.get(column)]), errors="coerce").iat[0]
                    delta = value_b - value_a if pd.notna(value_a) and pd.notna(value_b) else np.nan
                    wide_row[f"{metric_id}_a"] = value_a
                    wide_row[f"{metric_id}_b"] = value_b
                    wide_row[f"{metric_id}_delta_b_minus_a"] = delta
                    long_rows.append(
                        {
                            "sample": str(sample_value),
                            "component_id": str(component_id),
                            "category_a": category_a,
                            "category_b": category_b,
                            "category_pair": f"{category_a} -> {category_b}",
                            "comparison_class": comparison_class,
                            "genome_a": wide_row["genome_a"],
                            "genome_b": wide_row["genome_b"],
                            "metric_id": metric_id,
                            "metric_label": metric_label,
                            "value_a": value_a,
                            "value_b": value_b,
                            "delta_b_minus_a": delta,
                            "winner_category": (
                                category_b if pd.notna(delta) and delta > 0
                                else category_a if pd.notna(delta) and delta < 0
                                else "tie" if pd.notna(delta)
                                else ""
                            ),
                        }
                    )
                wide_rows.append(wide_row)
    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)

def category_is_xpg(category):
    return "xpg" in method_token(category)

def category_family(category):
    token = method_token(category)
    if "sag" in token and "mag" not in token:
        return "SAG"
    if "mag" in token and "sag" not in token:
        return "MAG"
    return ""

def normalized_focused_pair(category_a, category_b):
    cat_a = canonical_method_label(category_a)
    cat_b = canonical_method_label(category_b)
    a_xpg = category_is_xpg(cat_a)
    b_xpg = category_is_xpg(cat_b)
    fam_a = category_family(cat_a)
    fam_b = category_family(cat_b)
    if a_xpg != b_xpg:
        if a_xpg:
            return cat_b, cat_a, "non_xpg_to_xpg"
        return cat_a, cat_b, "non_xpg_to_xpg"
    if {fam_a, fam_b} == {"SAG", "MAG"}:
        if fam_a == "SAG":
            return cat_a, cat_b, "sag_to_mag_variant" if a_xpg and b_xpg else "sag_to_mag"
        return cat_b, cat_a, "sag_to_mag_variant" if a_xpg and b_xpg else "sag_to_mag"
    return "", "", ""

def normalized_reporting_pair(category_a, category_b):
    cat_a = canonical_method_label(category_a)
    cat_b = canonical_method_label(category_b)
    a_xpg = category_is_xpg(cat_a)
    b_xpg = category_is_xpg(cat_b)
    fam_a = category_family(cat_a)
    fam_b = category_family(cat_b)
    if a_xpg != b_xpg:
        if a_xpg:
            return cat_b, cat_a, True, "non_xpg_to_xpg"
        return cat_a, cat_b, False, "non_xpg_to_xpg"
    if {fam_a, fam_b} == {"SAG", "MAG"}:
        comparison_class = "sag_to_mag_variant" if a_xpg and b_xpg else "sag_to_mag"
        if fam_a == "SAG":
            return cat_a, cat_b, False, comparison_class
        return cat_b, cat_a, True, comparison_class
    if method_sort_key(cat_a) <= method_sort_key(cat_b):
        return cat_a, cat_b, False, "other"
    return cat_b, cat_a, True, "other"

def filter_focused_matched_lineage_pairs(wide_df, long_df):
    if wide_df.empty or long_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    focused_long_rows = []
    focused_wide_rows = []
    wide_lookup = {
        (
            str(row.get("sample", "")),
            str(row.get("component_id", "")),
            str(row.get("category_a", "")),
            str(row.get("category_b", "")),
        ): row
        for row in wide_df.to_dict("records")
    }

    for row in long_df.to_dict("records"):
        raw_a = str(row.get("category_a", "")).strip()
        raw_b = str(row.get("category_b", "")).strip()
        focused_a, focused_b, comparison_class = normalized_focused_pair(raw_a, raw_b)
        if not comparison_class:
            continue
        output_row = dict(row)
        if focused_a == raw_a and focused_b == raw_b:
            value_a = row.get("value_a", np.nan)
            value_b = row.get("value_b", np.nan)
            delta = row.get("delta_b_minus_a", np.nan)
            genome_a = row.get("genome_a", "")
            genome_b = row.get("genome_b", "")
        else:
            value_a = row.get("value_b", np.nan)
            value_b = row.get("value_a", np.nan)
            raw_delta = pd.to_numeric(pd.Series([row.get("delta_b_minus_a")]), errors="coerce").iat[0]
            delta = -raw_delta if pd.notna(raw_delta) else np.nan
            genome_a = row.get("genome_b", "")
            genome_b = row.get("genome_a", "")
        output_row.update(
            {
                "category_a": focused_a,
                "category_b": focused_b,
                "category_pair": f"{focused_a} -> {focused_b}",
                "comparison_class": comparison_class,
                "genome_a": genome_a,
                "genome_b": genome_b,
                "value_a": value_a,
                "value_b": value_b,
                "delta_b_minus_a": delta,
                "winner_category": (
                    focused_b if pd.notna(delta) and delta > 0
                    else focused_a if pd.notna(delta) and delta < 0
                    else "tie" if pd.notna(delta)
                    else ""
                ),
            }
        )
        focused_long_rows.append(output_row)

    seen_wide = set()
    for row in focused_long_rows:
        key = (row.get("sample"), row.get("component_id"), row.get("category_a"), row.get("category_b"))
        if key in seen_wide:
            continue
        seen_wide.add(key)
        raw_key_forward = (str(row.get("sample")), str(row.get("component_id")), str(row.get("category_a")), str(row.get("category_b")))
        raw_key_reverse = (str(row.get("sample")), str(row.get("component_id")), str(row.get("category_b")), str(row.get("category_a")))
        source_row = wide_lookup.get(raw_key_forward) or wide_lookup.get(raw_key_reverse)
        if source_row is None:
            continue
        focused_wide = {
            "sample": row.get("sample"),
            "component_id": row.get("component_id"),
            "category_a": row.get("category_a"),
            "category_b": row.get("category_b"),
            "category_pair": row.get("category_pair"),
            "comparison_class": row.get("comparison_class"),
            "genome_a": row.get("genome_a"),
            "genome_b": row.get("genome_b"),
        }
        focused_wide_rows.append(focused_wide)

    return pd.DataFrame(focused_wide_rows), pd.DataFrame(focused_long_rows)

def matched_lineage_feature_columns(category_table):
    excluded_exact = {
        "mp__mp_index",
        "mp_genome_id",
        "mp_genome_label",
        "mp_sample",
        "mp_category",
        "mp_input_dir",
        "mp_genome_display_label",
        "mp_taxonomy_display_rank",
        "mp_taxonomy_display_value",
        "mp_taxonomy_display_status",
        "mp_taxonomy_match_method",
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "qscore",
        "Completeness",
        "Contamination",
    }
    feature_columns = []
    for column in category_table.columns:
        if not str(column).startswith("mp_"):
            continue
        if column in excluded_exact:
            continue
        values = pd.to_numeric(category_table[column], errors="coerce")
        if values.notna().any():
            feature_columns.append(column)
    return feature_columns

def feature_label_from_column(column):
    text = str(column)
    if text.startswith("mp_"):
        text = text[3:]
    return text

def build_matched_lineage_feature_delta_tables(category_table):
    if category_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    feature_columns = matched_lineage_feature_columns(category_table)
    if not feature_columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for (sample_value, component_id), group in category_table.groupby(["sample", "component_id"], dropna=False):
        categories = ordered_methods(group["category"].astype(str).tolist())
        if len(categories) < 2:
            continue
        category_rows = {
            str(row.get("category", "")).strip(): row
            for row in group.to_dict("records")
            if str(row.get("category", "")).strip()
        }
        for idx_a, raw_a in enumerate(categories):
            for raw_b in categories[idx_a + 1:]:
                focused_a, focused_b, comparison_class = normalized_focused_pair(raw_a, raw_b)
                if not comparison_class:
                    continue
                row_a = category_rows.get(focused_a)
                row_b = category_rows.get(focused_b)
                if row_a is None or row_b is None:
                    continue
                for column in feature_columns:
                    value_a = pd.to_numeric(pd.Series([row_a.get(column)]), errors="coerce").iat[0]
                    value_b = pd.to_numeric(pd.Series([row_b.get(column)]), errors="coerce").iat[0]
                    if pd.isna(value_a) and pd.isna(value_b):
                        continue
                    value_a = 0.0 if pd.isna(value_a) else float(value_a)
                    value_b = 0.0 if pd.isna(value_b) else float(value_b)
                    delta = value_b - value_a
                    rows.append(
                        {
                            "sample": str(sample_value),
                            "component_id": str(component_id),
                            "comparison_class": comparison_class,
                            "category_a": focused_a,
                            "category_b": focused_b,
                            "category_pair": f"{focused_a} -> {focused_b}",
                            "genome_a": str(row_a.get("mp_genome_id", row_a.get("Genome_Id", ""))),
                            "genome_b": str(row_b.get("mp_genome_id", row_b.get("Genome_Id", ""))),
                            "feature_id": feature_label_from_column(column),
                            "feature_column": column,
                            "value_a": value_a,
                            "value_b": value_b,
                            "delta_b_minus_a": delta,
                            "presence_a": int(value_a > 0),
                            "presence_b": int(value_b > 0),
                            "feature_status": (
                                "gained_in_b" if value_a <= 0 and value_b > 0
                                else "lost_in_b" if value_a > 0 and value_b <= 0
                                else "shared_present" if value_a > 0 and value_b > 0
                                else "shared_absent"
                            ),
                        }
                    )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return long_df, pd.DataFrame()

    summary_rows = []
    pvalues = []
    for keys, group in long_df.groupby(["comparison_class", "category_pair", "category_a", "category_b", "feature_id", "feature_column"], dropna=False):
        comparison_class, category_pair, category_a, category_b, feature_id, feature_column = keys
        deltas = pd.to_numeric(group["delta_b_minus_a"], errors="coerce").dropna()
        if deltas.empty:
            continue
        gained = int((group["feature_status"] == "gained_in_b").sum())
        lost = int((group["feature_status"] == "lost_in_b").sum())
        shared_present = int((group["feature_status"] == "shared_present").sum())
        shared_absent = int((group["feature_status"] == "shared_absent").sum())
        b_higher = int((deltas > 0).sum())
        a_higher = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())
        decisive = b_higher + a_higher
        pvalue = exact_binomial_test_two_sided(b_higher, decisive) if decisive > 0 else float("nan")
        pvalues.append(pvalue)
        summary_rows.append(
            {
                "comparison_class": comparison_class,
                "category_pair": category_pair,
                "category_a": category_a,
                "category_b": category_b,
                "feature_id": feature_id,
                "feature_column": feature_column,
                "n_pairs": int(deltas.size),
                "n_b_higher": b_higher,
                "n_a_higher": a_higher,
                "n_ties": ties,
                "fraction_b_higher": (float(b_higher) / float(decisive)) if decisive > 0 else float("nan"),
                "median_delta_b_minus_a": float(deltas.median()),
                "mean_delta_b_minus_a": float(deltas.mean()),
                "n_gained_in_b": gained,
                "n_lost_in_b": lost,
                "n_shared_present": shared_present,
                "n_shared_absent": shared_absent,
                "pvalue_sign": pvalue,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["qvalue_bh"] = benjamini_hochberg_adjust(summary_df["pvalue_sign"]).to_numpy(dtype=float)
        summary_df["significance"] = summary_df["qvalue_bh"].map(significance_stars)
        summary_df = summary_df.sort_values(
            by=["category_pair", "qvalue_bh", "feature_id"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return long_df, summary_df

def build_matched_lineage_all_member_feature_contrasts(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "component_id" not in matched_df.columns or compare_column not in matched_df.columns or "mp_genome_id" not in matched_df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    working = matched_df.copy()
    working["sample"] = working.apply(pair_sample_value, axis=1)
    working["source_sample"] = working.apply(source_sample_value, axis=1)
    working["category"] = working[compare_column].astype(str).str.strip().map(canonical_method_label)
    working["_raw_genome_id"] = working["mp_genome_id"].astype(str).str.strip()
    working["genome_id"] = np.where(
        working["sample"].astype(str).eq("all_samples"),
        working["source_sample"].astype(str) + "|" + working["_raw_genome_id"].astype(str),
        working["_raw_genome_id"].astype(str),
    )
    working["component_id"] = working["component_id"].astype(str).str.strip()
    working = working.loc[
        working["sample"].astype(str).str.strip().ne("")
        & working["source_sample"].astype(str).str.strip().ne("")
        & working["category"].astype(str).str.strip().ne("")
        & working["_raw_genome_id"].astype(str).str.strip().ne("")
        & working["component_id"].astype(str).str.strip().ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feature_columns = matched_lineage_feature_columns(working)
    if not feature_columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    member_rows = []
    contrast_rows = []
    for (sample_value, component_id), component_df in working.groupby(["sample", "component_id"], dropna=False):
        categories = ordered_methods(component_df["category"].astype(str).tolist())
        category_groups = {
            category: component_df.loc[component_df["category"].astype(str).eq(category)].copy()
            for category in categories
        }
        for category, category_df in category_groups.items():
            for row in category_df.to_dict("records"):
                genome_id = str(row.get("genome_id", "")).strip()
                for feature_column in feature_columns:
                    value = pd.to_numeric(pd.Series([row.get(feature_column)]), errors="coerce").iat[0]
                    if pd.isna(value) or float(value) <= 0:
                        continue
                    member_rows.append(
                        {
                            "sample": str(sample_value),
                            "component_id": str(component_id),
                            "category": category,
                            "genome_id": genome_id,
                            "feature_id": feature_label_from_column(feature_column),
                            "feature_column": feature_column,
                            "value": float(value),
                        }
                    )

        if len(categories) < 2:
            continue
        for idx_a, raw_a in enumerate(categories):
            for raw_b in categories[idx_a + 1:]:
                category_a, category_b, comparison_class = normalized_focused_pair(raw_a, raw_b)
                if not comparison_class:
                    continue
                group_a = category_groups.get(category_a, pd.DataFrame()).copy()
                group_b = category_groups.get(category_b, pd.DataFrame()).copy()
                if group_a.empty or group_b.empty:
                    continue
                genomes_a = group_a["genome_id"].astype(str).tolist()
                genomes_b = group_b["genome_id"].astype(str).tolist()
                for feature_column in feature_columns:
                    values_a = pd.to_numeric(group_a[feature_column], errors="coerce").fillna(0.0)
                    values_b = pd.to_numeric(group_b[feature_column], errors="coerce").fillna(0.0)
                    present_a = values_a.gt(0)
                    present_b = values_b.gt(0)
                    genomes_a_present = group_a.loc[present_a, "genome_id"].astype(str).tolist()
                    genomes_b_present = group_b.loc[present_b, "genome_id"].astype(str).tolist()
                    genomes_a_absent = group_a.loc[~present_a, "genome_id"].astype(str).tolist()
                    genomes_b_absent = group_b.loc[~present_b, "genome_id"].astype(str).tolist()
                    n_a = int(len(group_a))
                    n_b = int(len(group_b))
                    n_a_present = int(present_a.sum())
                    n_b_present = int(present_b.sum())
                    frac_a = float(n_a_present) / float(n_a) if n_a else np.nan
                    frac_b = float(n_b_present) / float(n_b) if n_b else np.nan
                    if n_a_present == n_a and n_b_present == 0:
                        status = "all_a_only"
                    elif n_a_present > 0 and n_b_present == 0:
                        status = "some_a_only"
                    elif n_b_present == n_b and n_a_present == 0:
                        status = "all_b_only"
                    elif n_b_present > 0 and n_a_present == 0:
                        status = "some_b_only"
                    elif n_a_present == n_a and n_b_present == n_b:
                        status = "shared_all"
                    elif n_a_present > 0 and n_b_present > 0:
                        status = "shared_partial"
                    else:
                        status = "absent_all"
                    contrast_rows.append(
                        {
                            "sample": str(sample_value),
                            "component_id": str(component_id),
                            "comparison_class": comparison_class,
                            "category_a": category_a,
                            "category_b": category_b,
                            "category_pair": f"{category_a} -> {category_b}",
                            "feature_id": feature_label_from_column(feature_column),
                            "feature_column": feature_column,
                            "n_genomes_a": n_a,
                            "n_genomes_b": n_b,
                            "n_present_a": n_a_present,
                            "n_present_b": n_b_present,
                            "fraction_present_a": frac_a,
                            "fraction_present_b": frac_b,
                            "delta_fraction_b_minus_a": frac_b - frac_a if pd.notna(frac_a) and pd.notna(frac_b) else np.nan,
                            "median_value_a": float(values_a.median()) if len(values_a) else np.nan,
                            "median_value_b": float(values_b.median()) if len(values_b) else np.nan,
                            "delta_median_value_b_minus_a": float(values_b.median() - values_a.median()) if len(values_a) and len(values_b) else np.nan,
                            "genomes_a_present": ";".join(genomes_a_present),
                            "genomes_b_present": ";".join(genomes_b_present),
                            "genomes_a_absent": ";".join(genomes_a_absent),
                            "genomes_b_absent": ";".join(genomes_b_absent),
                            "all_genomes_a": ";".join(genomes_a),
                            "all_genomes_b": ";".join(genomes_b),
                            "group_feature_status": status,
                        }
                    )

    member_df = pd.DataFrame(member_rows)
    contrast_df = pd.DataFrame(contrast_rows)
    if contrast_df.empty:
        return member_df, contrast_df, pd.DataFrame()

    summary_rows = []
    pvalues = []
    for keys, group in contrast_df.groupby(["comparison_class", "category_pair", "category_a", "category_b", "feature_id", "feature_column"], dropna=False):
        comparison_class, category_pair, category_a, category_b, feature_id, feature_column = keys
        delta_fraction = pd.to_numeric(group["delta_fraction_b_minus_a"], errors="coerce").dropna()
        delta_median = pd.to_numeric(group["delta_median_value_b_minus_a"], errors="coerce").dropna()
        b_higher = int((delta_fraction > 0).sum())
        a_higher = int((delta_fraction < 0).sum())
        ties = int((delta_fraction == 0).sum())
        decisive = b_higher + a_higher
        pvalue = exact_binomial_test_two_sided(b_higher, decisive) if decisive > 0 else float("nan")
        pvalues.append(pvalue)
        status_counts = group["group_feature_status"].astype(str).value_counts().to_dict()
        summary_rows.append(
            {
                "comparison_class": comparison_class,
                "category_pair": category_pair,
                "category_a": category_a,
                "category_b": category_b,
                "feature_id": feature_id,
                "feature_column": feature_column,
                "n_components": int(group["component_id"].nunique()),
                "n_all_a_only": int(status_counts.get("all_a_only", 0)),
                "n_some_a_only": int(status_counts.get("some_a_only", 0)),
                "n_all_b_only": int(status_counts.get("all_b_only", 0)),
                "n_some_b_only": int(status_counts.get("some_b_only", 0)),
                "n_shared_all": int(status_counts.get("shared_all", 0)),
                "n_shared_partial": int(status_counts.get("shared_partial", 0)),
                "n_absent_all": int(status_counts.get("absent_all", 0)),
                "n_b_fraction_higher": b_higher,
                "n_a_fraction_higher": a_higher,
                "n_fraction_ties": ties,
                "fraction_components_b_higher": (float(b_higher) / float(decisive)) if decisive > 0 else float("nan"),
                "median_delta_fraction_b_minus_a": float(delta_fraction.median()) if not delta_fraction.empty else np.nan,
                "median_delta_median_value_b_minus_a": float(delta_median.median()) if not delta_median.empty else np.nan,
                "pvalue_sign": pvalue,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["qvalue_bh"] = benjamini_hochberg_adjust(summary_df["pvalue_sign"]).to_numpy(dtype=float)
        summary_df["significance"] = summary_df["qvalue_bh"].map(significance_stars)
        summary_df = summary_df.sort_values(
            by=["category_pair", "qvalue_bh", "feature_id"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return member_df, contrast_df, summary_df

def build_matched_lineage_feature_group_review(contrast_df):
    columns = [
        "sample",
        "component_id",
        "comparison_class",
        "category_pair",
        "category_a",
        "category_b",
        "n_genomes_a",
        "n_genomes_b",
        "all_genomes_a",
        "all_genomes_b",
        "n_features_a_only_all_a",
        "features_a_only_all_a",
        "n_features_a_only_some_a",
        "features_a_only_some_a",
        "n_features_b_only_all_b",
        "features_b_only_all_b",
        "n_features_b_only_some_b",
        "features_b_only_some_b",
        "n_features_shared_all",
        "features_shared_all",
        "n_features_shared_partial",
        "features_shared_partial",
        "n_features_absent_all",
    ]
    if contrast_df is None or contrast_df.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "sample",
        "component_id",
        "comparison_class",
        "category_pair",
        "category_a",
        "category_b",
        "feature_id",
        "group_feature_status",
        "n_genomes_a",
        "n_genomes_b",
        "all_genomes_a",
        "all_genomes_b",
    }
    if not required.issubset(set(contrast_df.columns)):
        return pd.DataFrame(columns=columns)

    def feature_list(group, status):
        values = (
            group.loc[group["group_feature_status"].astype(str).eq(status), "feature_id"]
            .astype(str)
            .dropna()
            .drop_duplicates()
            .sort_values(kind="mergesort")
            .tolist()
        )
        return ";".join(values)

    rows = []
    group_columns = ["sample", "component_id", "comparison_class", "category_pair", "category_a", "category_b"]
    for keys, group in contrast_df.groupby(group_columns, dropna=False):
        sample, component_id, comparison_class, category_pair, category_a, category_b = keys
        status_counts = group["group_feature_status"].astype(str).value_counts().to_dict()
        first = group.iloc[0]
        rows.append(
            {
                "sample": str(sample),
                "component_id": str(component_id),
                "comparison_class": str(comparison_class),
                "category_pair": str(category_pair),
                "category_a": str(category_a),
                "category_b": str(category_b),
                "n_genomes_a": int(pd.to_numeric(pd.Series([first.get("n_genomes_a")]), errors="coerce").fillna(0).iat[0]),
                "n_genomes_b": int(pd.to_numeric(pd.Series([first.get("n_genomes_b")]), errors="coerce").fillna(0).iat[0]),
                "all_genomes_a": str(first.get("all_genomes_a", "")),
                "all_genomes_b": str(first.get("all_genomes_b", "")),
                "n_features_a_only_all_a": int(status_counts.get("all_a_only", 0)),
                "features_a_only_all_a": feature_list(group, "all_a_only"),
                "n_features_a_only_some_a": int(status_counts.get("some_a_only", 0)),
                "features_a_only_some_a": feature_list(group, "some_a_only"),
                "n_features_b_only_all_b": int(status_counts.get("all_b_only", 0)),
                "features_b_only_all_b": feature_list(group, "all_b_only"),
                "n_features_b_only_some_b": int(status_counts.get("some_b_only", 0)),
                "features_b_only_some_b": feature_list(group, "some_b_only"),
                "n_features_shared_all": int(status_counts.get("shared_all", 0)),
                "features_shared_all": feature_list(group, "shared_all"),
                "n_features_shared_partial": int(status_counts.get("shared_partial", 0)),
                "features_shared_partial": feature_list(group, "shared_partial"),
                "n_features_absent_all": int(status_counts.get("absent_all", 0)),
            }
        )
    review_df = pd.DataFrame(rows, columns=columns)
    if review_df.empty:
        return review_df
    return review_df.sort_values(
        by=["sample", "component_id", "category_pair"],
        kind="mergesort",
    ).reset_index(drop=True)

def build_matched_lineage_annotation_tables(matched_df, annotation_audit_df, compare_column):
    presence_columns = [
        "sample",
        "source_sample",
        "component_id",
        "lineage_key",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "category",
        "genome_id",
        "genome_axis_label",
        "annotation_text",
        "source_db",
        "annotation_category",
        "n_orfs",
        "orf_ids",
        "present",
    ]
    matrix_columns = ["annotation_text", "n_genomes_present", "source_dbs", "annotation_categories"]
    if matched_df is None or matched_df.empty or annotation_audit_df is None or annotation_audit_df.empty:
        return pd.DataFrame(columns=presence_columns), pd.DataFrame(columns=matrix_columns)
    required_matched = {"component_id", compare_column, "mp_genome_id"}
    required_audit = {"genome_id", "orf_id", "annotation_text"}
    if not required_matched.issubset(set(matched_df.columns)) or not required_audit.issubset(set(annotation_audit_df.columns)):
        return pd.DataFrame(columns=presence_columns), pd.DataFrame(columns=matrix_columns)

    matched = matched_df.copy()
    matched["sample"] = matched.apply(pair_sample_value, axis=1)
    matched["source_sample"] = matched.apply(source_sample_value, axis=1)
    matched["category"] = matched[compare_column].astype(str).str.strip().map(canonical_method_label)
    matched["genome_id"] = matched["mp_genome_id"].astype(str).str.strip()
    matched["component_id"] = matched["component_id"].astype(str).str.strip()
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    for column in taxonomy_columns:
        if column not in matched.columns:
            matched[column] = ""
        matched[column] = matched[column].astype(str).str.strip()
    matched = matched.loc[
        matched["sample"].astype(str).str.strip().ne("")
        & matched["source_sample"].astype(str).str.strip().ne("")
        & matched["category"].astype(str).str.strip().ne("")
        & matched["genome_id"].astype(str).str.strip().ne("")
        & matched["component_id"].astype(str).str.strip().ne("")
        & matched["Species"].astype(str).str.strip().ne("")
        & matched["Species"].astype(str).str.lower().ne("nan")
    ].copy()
    for column in taxonomy_columns:
        matched = matched.loc[
            matched[column].astype(str).str.strip().ne("")
            & matched[column].astype(str).str.lower().ne("nan")
        ].copy()
    matched["lineage_key"] = matched[taxonomy_columns].agg(";".join, axis=1)
    category_counts = (
        matched.groupby("lineage_key", dropna=False)["category"]
        .transform(lambda s: pd.Series(s).astype(str).nunique())
    )
    matched = matched.loc[category_counts.ge(2)].copy()
    if matched.empty:
        return pd.DataFrame(columns=presence_columns), pd.DataFrame(columns=matrix_columns)
    matched["genome_axis_label"] = (
        matched["sample"].astype(str)
        + "|"
        + matched["source_sample"].astype(str)
        + "|"
        + matched["component_id"].astype(str)
        + "|"
        + matched["category"].astype(str)
        + "|"
        + matched["genome_id"].astype(str)
    )
    matched_key = matched[
        [
            "sample",
            "source_sample",
            "component_id",
            "lineage_key",
            *taxonomy_columns,
            "category",
            "genome_id",
            "genome_axis_label",
        ]
    ].drop_duplicates()
    matched_key["annotation_sample"] = matched_key["source_sample"].astype(str)

    audit = annotation_audit_df.copy()
    audit["sample"] = audit.get("sample", "").astype(str).str.strip() if "sample" in audit.columns else ""
    audit["category"] = (
        audit.get("category", "").astype(str).str.strip().map(canonical_method_label)
        if "category" in audit.columns else ""
    )
    audit["genome_id"] = audit["genome_id"].astype(str).str.strip()
    audit["orf_id"] = audit["orf_id"].astype(str).str.strip()
    audit["annotation_text"] = audit["annotation_text"].astype(str).str.strip()
    audit = audit.loc[
        audit["sample"].astype(str).str.strip().ne("")
        & audit["category"].astype(str).str.strip().ne("")
        & audit["genome_id"].astype(str).str.strip().ne("")
        & audit["orf_id"].astype(str).str.strip().ne("")
        & audit["annotation_text"].astype(str).str.strip().ne("")
    ].copy()
    if "annotation_is_informative" in audit.columns:
        informative = audit["annotation_is_informative"].astype(str).str.lower().isin({"true", "1", "yes", "y"})
        audit = audit.loc[informative].copy()
    if audit.empty:
        return pd.DataFrame(columns=presence_columns), pd.DataFrame(columns=matrix_columns)
    for column in ["source_db", "annotation_category"]:
        if column not in audit.columns:
            audit[column] = ""

    audit_for_merge = audit[
        [
            "sample",
            "category",
            "genome_id",
            "orf_id",
            "annotation_text",
            "source_db",
            "annotation_category",
        ]
    ].rename(columns={"sample": "annotation_sample"})
    orf_long = matched_key.merge(
        audit_for_merge,
        on=["annotation_sample", "category", "genome_id"],
        how="inner",
    )
    if orf_long.empty:
        return pd.DataFrame(columns=presence_columns), pd.DataFrame(columns=matrix_columns)
    orf_long = orf_long.drop_duplicates().sort_values(
        by=["sample", "source_sample", "component_id", "category", "genome_id", "annotation_text", "orf_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    def join_unique(series):
        values = [str(value).strip() for value in series.dropna().astype(str) if str(value).strip()]
        return ";".join(sorted(set(values)))

    presence = (
        orf_long.groupby(
            [
                "sample",
                "source_sample",
                "component_id",
                "lineage_key",
                *taxonomy_columns,
                "category",
                "genome_id",
                "genome_axis_label",
                "annotation_text",
            ],
            dropna=False,
        )
        .agg(
            source_db=("source_db", join_unique),
            annotation_category=("annotation_category", join_unique),
            n_orfs=("orf_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            orf_ids=("orf_id", join_unique),
        )
        .reset_index()
    )
    presence["present"] = 1
    presence = presence.loc[:, presence_columns].sort_values(
        by=["sample", "source_sample", "component_id", "category", "genome_id", "annotation_text"],
        kind="mergesort",
    ).reset_index(drop=True)

    method_order = ordered_methods(matched_key["category"].astype(str).tolist())
    method_order_map = {category: index for index, category in enumerate(method_order)}
    matched_key_for_order = matched_key.copy()
    matched_key_for_order["_category_order"] = matched_key_for_order["category"].map(method_order_map).fillna(len(method_order_map)).astype(int)
    genome_order = matched_key_for_order.sort_values(
        by=["sample", "component_id", "lineage_key", "_category_order", "category", "source_sample", "genome_id"],
        kind="mergesort",
    )["genome_axis_label"].astype(str).drop_duplicates().tolist()
    annotation_meta = (
        presence.groupby("annotation_text", dropna=False)
        .agg(
            n_genomes_present=("genome_axis_label", lambda s: int(pd.Series(s).astype(str).nunique())),
            source_dbs=("source_db", join_unique),
            annotation_categories=("annotation_category", join_unique),
        )
        .reset_index()
    )
    annotation_order = annotation_meta.sort_values(
        by=["n_genomes_present", "annotation_text"],
        ascending=[False, True],
        kind="mergesort",
    )["annotation_text"].astype(str).tolist()
    matrix = presence.pivot_table(
        index="annotation_text",
        columns="genome_axis_label",
        values="present",
        aggfunc="max",
        fill_value=0,
    ).reindex(index=annotation_order, columns=genome_order, fill_value=0)
    matrix = matrix.reset_index()
    matrix = annotation_meta.merge(matrix, on="annotation_text", how="right")
    ordered_columns = matrix_columns + [column for column in genome_order if column in matrix.columns]
    matrix = matrix.loc[:, ordered_columns]
    return presence, matrix

def build_lineage_annotation_presence_matrix(presence_df):
    if presence_df is None or presence_df.empty:
        return pd.DataFrame()
    metadata_columns = [
        "annotation_text",
        "n_genomes_present",
        "source_db",
        "annotation_category",
        "categories_present",
        "categories_absent",
        "category_presence_pattern",
        "orf_ids_by_category",
    ]
    order_df = presence_df[["category", "genome_id", "genome_axis_label"]].drop_duplicates().copy()
    categories = ordered_methods(order_df["category"].astype(str).tolist())
    category_order = {category: index for index, category in enumerate(categories)}
    order_df["_category_order"] = order_df["category"].map(category_order).fillna(len(category_order)).astype(int)
    genome_order = (
        order_df.sort_values(
            by=["_category_order", "category", "genome_id"],
            kind="mergesort",
        )["genome_axis_label"].astype(str).tolist()
    )
    if not genome_order:
        return pd.DataFrame()

    def join_unique(series):
        values = [str(value).strip() for value in series.dropna().astype(str) if str(value).strip()]
        return ";".join(sorted(set(values)))

    matrix = presence_df.pivot_table(
        index="annotation_text",
        columns="genome_axis_label",
        values="present",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=genome_order, fill_value=0)
    category_presence = {}
    orf_by_category = {}
    for annotation_text, group in presence_df.groupby("annotation_text", dropna=False):
        present_categories = ordered_methods(group["category"].astype(str).drop_duplicates().tolist())
        category_presence[annotation_text] = {
            "categories_present": ";".join(present_categories),
            "categories_absent": ";".join([category for category in categories if category not in present_categories]),
            "category_presence_pattern": ";".join(
                f"{category}={'1' if category in present_categories else '0'}"
                for category in categories
            ),
        }
        category_bits = []
        for category, category_group in group.groupby("category", dropna=False):
            orfs = join_unique(category_group["orf_ids"])
            category_bits.append(f"{category}:{orfs}")
        orf_by_category[annotation_text] = "; ".join(category_bits)
    meta = (
        presence_df.groupby("annotation_text", dropna=False)
        .agg(
            n_genomes_present=("genome_axis_label", lambda s: int(pd.Series(s).astype(str).nunique())),
            source_db=("source_db", join_unique),
            annotation_category=("annotation_category", join_unique),
        )
        .reset_index()
    )
    for column in ["categories_present", "categories_absent", "category_presence_pattern"]:
        meta[column] = meta["annotation_text"].map(lambda value: category_presence.get(value, {}).get(column, ""))
    meta["orf_ids_by_category"] = meta["annotation_text"].map(lambda value: orf_by_category.get(value, ""))
    matrix = matrix.reset_index()
    matrix = meta.merge(matrix, on="annotation_text", how="right")
    matrix = matrix.sort_values(
        by=["n_genomes_present", "annotation_text"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return matrix.loc[:, metadata_columns + genome_order]

def write_lineage_annotation_presence_tables(presence_df, output_dir, combine_samples=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        for old_path in output_dir.rglob("*.tsv"):
            old_path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    if presence_df is None or presence_df.empty:
        return []

    wrote_paths = []
    index_rows = []
    group_columns = ["lineage_key"] if combine_samples else ["sample", "component_id", "lineage_key"]
    grouped = presence_df.groupby(group_columns, dropna=False)
    for raw_keys, group in grouped:
        if combine_samples:
            lineage_key = raw_keys[0] if isinstance(raw_keys, tuple) else raw_keys
            sample = "all_samples"
            component_id = "all_components"
        else:
            sample, component_id, lineage_key = raw_keys
        categories = ordered_methods(group["category"].astype(str).drop_duplicates().tolist())
        if len(categories) < 2:
            continue
        first = group.iloc[0]
        species = str(first.get("Species", "")).strip()
        samples = ";".join(sorted(group["sample"].astype(str).drop_duplicates().tolist()))
        components = ";".join(sorted(group["component_id"].astype(str).drop_duplicates().tolist()))
        safe_name = (
            sanitize_label(f"all_samples_{species}")
            if combine_samples
            else sanitize_label(f"{sample}_{component_id}_{species}")
        )
        table_path = output_dir / f"{safe_name}.annotation_presence.tsv"
        lineage_matrix = build_lineage_annotation_presence_matrix(group)
        if lineage_matrix.empty:
            continue
        lineage_matrix.to_csv(table_path, sep="\t", index=False)
        wrote_paths.append(table_path)
        index_rows.append(
            {
                "sample": str(sample),
                "component_id": str(component_id),
                "samples": samples,
                "component_ids": components,
                "lineage_key": str(lineage_key),
                "Domain": str(first.get("Domain", "")),
                "Phylum": str(first.get("Phylum", "")),
                "Class": str(first.get("Class", "")),
                "Order": str(first.get("Order", "")),
                "Family": str(first.get("Family", "")),
                "Genus": str(first.get("Genus", "")),
                "Species": species,
                "categories": ";".join(categories),
                "n_categories": len(categories),
                "sample_category_combinations": ";".join(
                    sorted(
                        (
                            group["sample"].astype(str)
                            + "|"
                            + group["category"].astype(str)
                        ).drop_duplicates().tolist()
                    )
                ),
                "n_sample_category_combinations": int(
                    (
                        group["sample"].astype(str)
                        + "|"
                        + group["category"].astype(str)
                    ).nunique()
                ),
                "n_genomes": int(group["genome_axis_label"].astype(str).nunique()),
                "n_annotations": int(lineage_matrix["annotation_text"].astype(str).nunique()),
                "table_path": str(table_path),
            }
        )
    if index_rows:
        index_path = output_dir / "lineage_annotation_presence_index.tsv"
        pd.DataFrame(index_rows).sort_values(
            by=["Species", "sample", "component_id"],
            kind="mergesort",
        ).to_csv(index_path, sep="\t", index=False)
        wrote_paths.insert(0, index_path)
    return [path for path in wrote_paths if path]

def cleanup_deprecated_matched_lineage_annotation_outputs(output_dir, paired_prefix, compare_label):
    output_dir = Path(output_dir)
    name_prefix = f"{paired_prefix}_atlas_paired_{compare_label}"
    deprecated_patterns = [
        f"{name_prefix}_matched_lineage_annotation_orf_long.tsv",
        f"{name_prefix}_matched_lineage_annotation_presence_long.tsv",
        f"{name_prefix}_matched_lineage_feature_group_review.tsv",
    ]
    removed = []
    for pattern in deprecated_patterns:
        for path in output_dir.rglob(pattern):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed

def plot_matched_lineage_annotation_presence_matrix(matrix_df, output_base, max_annotations=80):
    ensure_plotting()
    if matrix_df is None or matrix_df.empty or "annotation_text" not in matrix_df.columns:
        return False
    metadata_columns = {"annotation_text", "n_genomes_present", "source_dbs", "annotation_categories"}
    genome_columns = [column for column in matrix_df.columns if column not in metadata_columns]
    if not genome_columns:
        return False
    working = matrix_df.copy()
    for column in genome_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int)
    working["_presence_total"] = working[genome_columns].sum(axis=1)
    working = working.loc[working["_presence_total"].gt(0)].copy()
    if working.empty:
        return False
    if len(working) > int(max_annotations):
        working = working.sort_values(
            by=["_presence_total", "annotation_text"],
            ascending=[False, True],
            kind="mergesort",
        ).head(int(max_annotations)).copy()
    working = working.sort_values(
        by=["_presence_total", "annotation_text"],
        ascending=[True, True],
        kind="mergesort",
    )
    values = working[genome_columns].to_numpy(dtype=float)
    if values.size == 0:
        return False

    y_labels = [
        label if len(label) <= 90 else label[:87] + "..."
        for label in working["annotation_text"].astype(str).tolist()
    ]
    x_labels = []
    for label in genome_columns:
        parts = str(label).split("|")
        if len(parts) >= 4:
            x_labels.append(f"{parts[2]}|{parts[3]}")
        else:
            x_labels.append(str(label))

    plt_local = ensure_plotting()
    width = min(60, max(14, len(genome_columns) * 0.18))
    height = min(40, max(8, len(y_labels) * 0.24))
    fig, ax = plt_local.subplots(figsize=(width, height))
    image = ax.imshow(values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_title("Matched-lineage functional annotation presence/absence")
    ax.set_xlabel("Genome, ordered by sample | lineage component | category")
    ax.set_ylabel("Functional annotation")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=6 if len(y_labels) > 45 else 7)
    if len(x_labels) <= 160:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=4 if len(x_labels) > 80 else 6)
    else:
        tick_step = max(1, int(np.ceil(len(x_labels) / 120.0)))
        tick_positions = np.arange(0, len(x_labels), tick_step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([x_labels[idx] for idx in tick_positions], rotation=90, fontsize=3)

    previous_group = None
    for idx, label in enumerate(genome_columns):
        parts = str(label).split("|")
        group = tuple(parts[:3]) if len(parts) >= 3 else (str(label),)
        if previous_group is not None and group != previous_group:
            ax.axvline(idx - 0.5, color="#c9c9c9", linewidth=0.35)
        previous_group = group
    cbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Absent", "Present"])
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

def join_unique_values(series, separator=";"):
    values = [str(value).strip() for value in pd.Series(series).dropna().astype(str) if str(value).strip()]
    return separator.join(sorted(set(values)))

def prepare_global_lineage_universe(matched_df, compare_column):
    columns = [
        "lineage_key",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "category",
        "source_sample",
        "genome_id",
        "genome_axis_label",
    ]
    if matched_df is None or matched_df.empty or compare_column not in matched_df.columns:
        return pd.DataFrame(columns=columns)

    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    working = matched_df.copy()
    working["category"] = working[compare_column].astype(str).str.strip().map(canonical_method_label)
    working["source_sample"] = working.apply(source_sample_value, axis=1)
    working["genome_id"] = working.get("mp_genome_id", "").astype(str).str.strip()
    for column in taxonomy_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()
    working = working.loc[
        working["category"].astype(str).str.strip().ne("")
        & working["source_sample"].astype(str).str.strip().ne("")
        & working["genome_id"].astype(str).str.strip().ne("")
    ].copy()
    for column in taxonomy_columns:
        working = working.loc[
            working[column].astype(str).str.strip().ne("")
            & working[column].astype(str).str.lower().ne("nan")
            & working[column].astype(str).str.lower().ne("none")
        ].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["lineage_key"] = working[taxonomy_columns].agg(";".join, axis=1)
    working["genome_axis_label"] = (
        working["category"].astype(str)
        + "|"
        + working["source_sample"].astype(str)
        + "|"
        + working["genome_id"].astype(str)
    )
    return working.loc[:, columns].drop_duplicates().reset_index(drop=True)

def build_global_candidate_lineages(matched_df, compare_column):
    universe = prepare_global_lineage_universe(matched_df, compare_column)
    columns = [
        "candidate_rank",
        "passes_all_categories",
        "lineage_key",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
        "n_genomes_total",
        "n_categories",
        "categories_present",
        "missing_categories",
        "n_source_samples",
        "source_samples",
    ]
    if universe.empty:
        return pd.DataFrame(columns=columns)
    all_categories = ordered_methods(universe["category"].astype(str).drop_duplicates().tolist())
    rows = []
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    for lineage_key, group in universe.groupby("lineage_key", dropna=False):
        categories = ordered_methods(group["category"].astype(str).drop_duplicates().tolist())
        first = group.iloc[0]
        rows.append(
            {
                "lineage_key": str(lineage_key),
                **{column: str(first.get(column, "")) for column in taxonomy_columns},
                "n_genomes_total": int(group["genome_axis_label"].astype(str).nunique()),
                "n_categories": int(len(categories)),
                "categories_present": ";".join(categories),
                "missing_categories": ";".join([category for category in all_categories if category not in categories]),
                "n_source_samples": int(group["source_sample"].astype(str).nunique()),
                "source_samples": join_unique_values(group["source_sample"]),
            }
        )
    candidate_df = pd.DataFrame(rows)
    candidate_df["passes_all_categories"] = candidate_df["n_categories"].eq(len(all_categories))
    candidate_df = candidate_df.sort_values(
        by=["passes_all_categories", "n_categories", "n_genomes_total", "Species", "lineage_key"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    candidate_df.insert(0, "candidate_rank", np.arange(1, len(candidate_df) + 1, dtype=int))
    return candidate_df.loc[:, columns]

def all_category_candidate_lineage_keys(matched_df, compare_column):
    candidate_df = build_global_candidate_lineages(matched_df, compare_column)
    if candidate_df.empty or "passes_all_categories" not in candidate_df.columns:
        return set(), candidate_df
    selected = candidate_df.loc[candidate_df["passes_all_categories"].astype(bool)].copy()
    return set(selected["lineage_key"].astype(str)), candidate_df

def filter_to_all_category_candidate_lineages(matched_df, compare_column):
    if matched_df is None or matched_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    candidate_keys, candidate_df = all_category_candidate_lineage_keys(matched_df, compare_column)
    if not candidate_keys:
        return matched_df.iloc[0:0].copy(), candidate_df
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    working = matched_df.copy()
    for column in taxonomy_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()
    working["_candidate_lineage_key"] = working[taxonomy_columns].agg(";".join, axis=1)
    working = working.loc[working["_candidate_lineage_key"].astype(str).isin(candidate_keys)].copy()
    return working.drop(columns=["_candidate_lineage_key"], errors="ignore"), candidate_df

def fisher_exact_two_sided(a, b, c, d):
    values = pd.to_numeric(pd.Series([a, b, c, d]), errors="coerce").fillna(0).astype(int).clip(lower=0)
    a, b, c, d = [int(value) for value in values.tolist()]
    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2
    if total <= 0 or row1 <= 0 or row2 <= 0:
        return np.nan

    def log_choose(n, k):
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    def probability(x):
        return math.exp(
            log_choose(col1, x)
            + log_choose(total - col1, row1 - x)
            - log_choose(total, row1)
        )

    min_x = max(0, row1 - (total - col1))
    max_x = min(row1, col1)
    observed = probability(a)
    pvalue = 0.0
    for x in range(min_x, max_x + 1):
        p = probability(x)
        if p <= observed + 1e-12:
            pvalue += p
    return float(min(1.0, max(0.0, pvalue)))

def odds_ratio_with_pseudocount(a, b, c, d, pseudocount=0.5):
    values = pd.to_numeric(pd.Series([a, b, c, d]), errors="coerce").fillna(0).astype(float).clip(lower=0)
    a, b, c, d = [float(value) for value in values.tolist()]
    return ((a + pseudocount) * (d + pseudocount)) / ((b + pseudocount) * (c + pseudocount))

def add_bh_qvalues(frame, pvalue_column="pvalue"):
    output = frame.copy()
    if output.empty or pvalue_column not in output.columns:
        output["qvalue_bh"] = np.nan
        output["significance"] = ""
        return output
    output["qvalue_bh"] = benjamini_hochberg_adjust(output[pvalue_column]).to_numpy()
    output["significance"] = output["qvalue_bh"].map(significance_stars)
    return output

def build_annotation_entity_presence(annotation_presence_df):
    columns = [
        "lineage_key",
        "category",
        "source_sample",
        "genome_id",
        "genome_axis_label",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "source_db",
        "annotation_category",
        "count_value",
        "orf_ids",
    ]
    if annotation_presence_df is None or annotation_presence_df.empty:
        return pd.DataFrame(columns=columns)
    required = {"lineage_key", "category", "source_sample", "genome_id", "genome_axis_label", "annotation_text"}
    if not required.issubset(set(annotation_presence_df.columns)):
        return pd.DataFrame(columns=columns)
    working = annotation_presence_df.copy()
    working["entity_id"] = working["annotation_text"].astype(str).str.strip()
    working["entity_label"] = working["entity_id"]
    working["entity_kind"] = "functional_annotation"
    working["entity_group"] = working.get("annotation_category", "").astype(str).str.strip()
    working["count_value"] = pd.to_numeric(working.get("n_orfs", 1), errors="coerce").fillna(1)
    for column in ["source_db", "annotation_category", "orf_ids"]:
        if column not in working.columns:
            working[column] = ""
    return working.loc[:, columns].drop_duplicates().reset_index(drop=True)

def feature_label_from_column(column):
    label = str(column)
    if label.startswith("mp_"):
        label = label[3:]
    suffixes = [
        "_gene_count",
        "_core_gene_count",
        "_orf_count",
        "_accession_count",
        "_fraction",
        "_orfs",
        "_count",
    ]
    for suffix in suffixes:
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    return label

def build_feature_entity_presence(matched_df, compare_column):
    universe = prepare_global_lineage_universe(matched_df, compare_column)
    columns = [
        "lineage_key",
        "category",
        "source_sample",
        "genome_id",
        "genome_axis_label",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "source_db",
        "annotation_category",
        "count_value",
        "orf_ids",
    ]
    if universe.empty or matched_df is None or matched_df.empty:
        return pd.DataFrame(columns=columns)
    feature_columns = [
        column for column in matched_df.columns
        if (
            column.startswith("mp_marker_")
            or column.startswith("mp_reference_mode_")
        )
        and column not in {"mp_marker_gene_count"}
        and (
            column.endswith("_gene_count")
            or column.endswith("_core_gene_count")
            or column.endswith("_orf_count")
            or column.endswith("_accession_count")
        )
    ]
    if not feature_columns:
        return pd.DataFrame(columns=columns)
    working = matched_df.copy()
    working["_category"] = working[compare_column].astype(str).str.strip().map(canonical_method_label)
    working["_source_sample"] = working.apply(source_sample_value, axis=1)
    working["_genome_id"] = working.get("mp_genome_id", "").astype(str).str.strip()
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    for column in taxonomy_columns:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].astype(str).str.strip()
    working = working.loc[
        working["_category"].astype(str).str.strip().ne("")
        & working["_source_sample"].astype(str).str.strip().ne("")
        & working["_genome_id"].astype(str).str.strip().ne("")
    ].copy()
    for column in taxonomy_columns:
        working = working.loc[
            working[column].astype(str).str.strip().ne("")
            & working[column].astype(str).str.lower().ne("nan")
            & working[column].astype(str).str.lower().ne("none")
        ].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["_lineage_key"] = working[taxonomy_columns].agg(";".join, axis=1)
    working["_genome_axis_label"] = (
        working["_category"].astype(str)
        + "|"
        + working["_source_sample"].astype(str)
        + "|"
        + working["_genome_id"].astype(str)
    )
    rows = []
    base_columns = ["_lineage_key", "_category", "_source_sample", "_genome_id", "_genome_axis_label"]
    for column in feature_columns:
        values = pd.to_numeric(working[column], errors="coerce").fillna(0)
        subset = working.loc[values.gt(0), base_columns].copy()
        if subset.empty:
            continue
        subset["entity_id"] = feature_label_from_column(column)
        subset["entity_label"] = subset["entity_id"]
        subset["entity_kind"] = "marker_or_reference_feature"
        subset["entity_group"] = "marker" if column.startswith("mp_marker_") else "reference_mode"
        subset["source_db"] = subset["entity_group"]
        subset["annotation_category"] = subset["entity_group"]
        subset["count_value"] = values.loc[subset.index].to_numpy(dtype=float)
        subset["orf_ids"] = ""
        subset = subset.rename(
            columns={
                "_lineage_key": "lineage_key",
                "_category": "category",
                "_source_sample": "source_sample",
                "_genome_id": "genome_id",
                "_genome_axis_label": "genome_axis_label",
            }
        )
        rows.append(subset.loc[:, columns])
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True).drop_duplicates().reset_index(drop=True)

def build_global_entity_comparative_tables(universe_df, entity_presence_df, candidate_df, require_all_categories=True):
    category_summary_columns = [
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "source_db",
        "annotation_category",
        "category",
        "n_genomes",
        "n_present",
        "presence_fraction",
        "total_count_value",
        "mean_count_per_genome",
        "orf_ids",
    ]
    enrichment_columns = [
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "category",
        "n_present_in_category",
        "n_genomes_in_category",
        "presence_fraction_in_category",
        "n_present_other_categories",
        "n_genomes_other_categories",
        "presence_fraction_other_categories",
        "delta_fraction_category_minus_others",
        "odds_ratio",
        "log2_odds_ratio",
        "pvalue",
        "qvalue_bh",
        "significance",
    ]
    pairwise_columns = [
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "comparison_class",
        "category_a",
        "category_b",
        "category_pair",
        "n_present_a",
        "n_genomes_a",
        "presence_fraction_a",
        "n_present_b",
        "n_genomes_b",
        "presence_fraction_b",
        "delta_fraction_b_minus_a",
        "odds_ratio_b_over_a",
        "log2_odds_ratio_b_over_a",
        "pvalue",
        "qvalue_bh",
        "significance",
    ]
    status_columns = [
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "n_categories_present",
        "categories_present",
        "categories_absent",
        "status",
        "max_presence_fraction",
        "min_presence_fraction",
        "range_presence_fraction",
    ]
    if universe_df.empty or entity_presence_df.empty or candidate_df.empty:
        return (
            pd.DataFrame(columns=category_summary_columns),
            pd.DataFrame(columns=enrichment_columns),
            pd.DataFrame(columns=pairwise_columns),
            pd.DataFrame(columns=status_columns),
        )

    categories = ordered_methods(universe_df["category"].astype(str).drop_duplicates().tolist())
    selected_candidates = candidate_df.copy()
    if require_all_categories and "passes_all_categories" in selected_candidates.columns:
        selected_candidates = selected_candidates.loc[selected_candidates["passes_all_categories"].astype(bool)].copy()
    selected_lineages = set(selected_candidates["lineage_key"].astype(str))
    universe = universe_df.loc[universe_df["lineage_key"].astype(str).isin(selected_lineages)].copy()
    presence = entity_presence_df.loc[entity_presence_df["lineage_key"].astype(str).isin(selected_lineages)].copy()
    if not universe.empty:
        categories = ordered_methods(universe["category"].astype(str).drop_duplicates().tolist())
    if universe.empty or presence.empty:
        return (
            pd.DataFrame(columns=category_summary_columns),
            pd.DataFrame(columns=enrichment_columns),
            pd.DataFrame(columns=pairwise_columns),
            pd.DataFrame(columns=status_columns),
        )

    genome_counts = (
        universe.groupby(["lineage_key", "category"], dropna=False)
        .agg(
            n_genomes=("genome_axis_label", lambda s: int(pd.Series(s).astype(str).nunique())),
            Species=("Species", "first"),
        )
        .reset_index()
    )
    entity_meta = (
        presence.groupby(["lineage_key", "entity_id"], dropna=False)
        .agg(
            entity_label=("entity_label", "first"),
            entity_kind=("entity_kind", "first"),
            entity_group=("entity_group", join_unique_values),
            source_db=("source_db", join_unique_values),
            annotation_category=("annotation_category", join_unique_values),
        )
        .reset_index()
    )
    present_counts = (
        presence.groupby(["lineage_key", "entity_id", "category"], dropna=False)
        .agg(
            n_present=("genome_axis_label", lambda s: int(pd.Series(s).astype(str).nunique())),
            total_count_value=("count_value", "sum"),
            orf_ids=("orf_ids", join_unique_values),
        )
        .reset_index()
    )
    base = entity_meta.merge(genome_counts, on="lineage_key", how="inner")
    category_summary = base.merge(
        present_counts,
        on=["lineage_key", "entity_id", "category"],
        how="left",
    )
    category_summary["n_present"] = pd.to_numeric(category_summary["n_present"], errors="coerce").fillna(0).astype(int)
    category_summary["total_count_value"] = pd.to_numeric(category_summary["total_count_value"], errors="coerce").fillna(0.0)
    category_summary["presence_fraction"] = np.where(
        pd.to_numeric(category_summary["n_genomes"], errors="coerce").gt(0),
        category_summary["n_present"] / pd.to_numeric(category_summary["n_genomes"], errors="coerce"),
        np.nan,
    )
    category_summary["mean_count_per_genome"] = np.where(
        pd.to_numeric(category_summary["n_genomes"], errors="coerce").gt(0),
        category_summary["total_count_value"] / pd.to_numeric(category_summary["n_genomes"], errors="coerce"),
        np.nan,
    )
    category_summary["orf_ids"] = category_summary["orf_ids"].fillna("")
    category_summary = category_summary.loc[:, category_summary_columns]

    status_rows = []
    occurrence_rows = []
    pairwise_occurrence_rows = []
    for (lineage_key, entity_id), group in category_summary.groupby(["lineage_key", "entity_id"], dropna=False):
        group = group.copy()
        species = str(group["Species"].iloc[0])
        entity_label = str(group["entity_label"].iloc[0])
        entity_kind = str(group["entity_kind"].iloc[0])
        entity_group = str(group["entity_group"].iloc[0])
        source_db = str(group["source_db"].iloc[0]) if "source_db" in group.columns else ""
        annotation_category = str(group["annotation_category"].iloc[0]) if "annotation_category" in group.columns else ""
        present_map = {
            str(row.category): int(row.n_present)
            for row in group.itertuples(index=False)
        }
        genome_map = {
            str(row.category): int(row.n_genomes)
            for row in group.itertuples(index=False)
        }
        fraction_map = {
            str(row.category): float(row.presence_fraction) if pd.notna(row.presence_fraction) else np.nan
            for row in group.itertuples(index=False)
        }
        occurrence_row = {
            "lineage_key": str(lineage_key),
            "Species": species,
            "entity_id": str(entity_id),
            "entity_label": entity_label,
            "entity_kind": entity_kind,
            "entity_group": entity_group,
            "source_db": source_db,
            "annotation_category": annotation_category,
            "accession": f"{lineage_key}||{entity_id}",
        }
        props = np.array(
            [
                (present_map.get(category, 0) / genome_map.get(category, 0))
                if int(genome_map.get(category, 0)) > 0 else 0.0
                for category in categories
            ],
            dtype=float,
        )
        reps = np.array([int(genome_map.get(category, 0)) for category in categories], dtype=float)
        occurrence_row["associated_groups"] = ",".join(
            category
            for category, value in zip(categories, props)
            if np.count_nonzero(props) and value > ((np.sum(props * reps) / np.sum(reps)) if np.sum(reps) else 0.0)
        )
        for category, prop, rep in zip(categories, props, reps):
            occurrence_row[f"p_{category}"] = float(prop)
            occurrence_row[f"N_{category}"] = int(rep)
        occurrence_rows.append(occurrence_row)
        for category_a, category_b in itertools.combinations(categories, 2):
            category_a, category_b, _swapped, comparison_class = normalized_reporting_pair(category_a, category_b)
            a_present = int(present_map.get(category_a, 0))
            a_total = int(genome_map.get(category_a, 0))
            b_present = int(present_map.get(category_b, 0))
            b_total = int(genome_map.get(category_b, 0))
            pairwise_occurrence_rows.append(
                {
                    "lineage_key": str(lineage_key),
                    "Species": species,
                    "entity_id": str(entity_id),
                    "entity_label": entity_label,
                    "entity_kind": entity_kind,
                    "entity_group": entity_group,
                    "comparison_class": comparison_class,
                    "category_a": category_a,
                    "category_b": category_b,
                    "category_pair": f"{category_a} -> {category_b}",
                    "accession": f"{lineage_key}||{entity_id}||{sanitize_label(category_a)}||{sanitize_label(category_b)}",
                    "associated_groups": "",
                    "p_A": (a_present / a_total) if a_total else 0.0,
                    "N_A": a_total,
                    "p_B": (b_present / b_total) if b_total else 0.0,
                    "N_B": b_total,
                }
            )
        present_categories = [category for category in categories if present_map.get(category, 0) > 0]
        absent_categories = [category for category in categories if present_map.get(category, 0) <= 0]
        fractions = [value for value in fraction_map.values() if pd.notna(value)]
        if len(present_categories) == len(categories):
            status = "shared_all_categories"
        elif len(present_categories) == 1:
            status = f"unique_to_{sanitize_label(present_categories[0])}"
        else:
            status = "partial_category_overlap"
        status_rows.append(
            {
                "lineage_key": str(lineage_key),
                "Species": species,
                "entity_id": str(entity_id),
                "entity_label": entity_label,
                "entity_kind": entity_kind,
                "entity_group": entity_group,
                "n_categories_present": int(len(present_categories)),
                "categories_present": ";".join(present_categories),
                "categories_absent": ";".join(absent_categories),
                "status": status,
                "max_presence_fraction": float(max(fractions)) if fractions else np.nan,
                "min_presence_fraction": float(min(fractions)) if fractions else np.nan,
                "range_presence_fraction": float(max(fractions) - min(fractions)) if fractions else np.nan,
            }
        )

    if occurrence_rows:
        occurrence_df = pd.DataFrame(occurrence_rows)
        occurrence_enrichment_df = run_enrichment_dataframe_compat(
            occurrence_df,
            compute_associated_groups_if_missing=False,
            allow_empty_associated_groups=True,
        )
    else:
        occurrence_enrichment_df = pd.DataFrame()

    enrichment_rows = []
    if not occurrence_enrichment_df.empty:
        for row in occurrence_enrichment_df.to_dict("records"):
            associated_groups = {
                value.strip()
                for value in str(row.get("associated_groups", "")).split(",")
                if value.strip()
            }
            global_pvalue = float(pd.to_numeric(row.get("unadjusted_p_value"), errors="coerce"))
            global_qvalue = float(pd.to_numeric(row.get("adjusted_q_value"), errors="coerce"))
            global_score = float(pd.to_numeric(row.get("enrichment_score"), errors="coerce"))
            for category in categories:
                a = int(round(float(pd.to_numeric(row.get(f"p_{category}", 0), errors="coerce")) * float(pd.to_numeric(row.get(f"N_{category}", 0), errors="coerce"))))
                n_category = int(pd.to_numeric(row.get(f"N_{category}", 0), errors="coerce"))
                b = max(0, n_category - a)
                other_categories = [other for other in categories if other != category]
                c = int(
                    sum(
                        round(
                            float(pd.to_numeric(row.get(f"p_{other}", 0), errors="coerce"))
                            * float(pd.to_numeric(row.get(f"N_{other}", 0), errors="coerce"))
                        )
                        for other in other_categories
                    )
                )
                n_other = int(
                    sum(
                        int(pd.to_numeric(row.get(f"N_{other}", 0), errors="coerce"))
                        for other in other_categories
                    )
                )
                d = max(0, n_other - c)
                frac_category = (a / n_category) if n_category else np.nan
                frac_other = (c / n_other) if n_other else np.nan
                delta = (
                    frac_category - frac_other
                    if pd.notna(frac_category) and pd.notna(frac_other) else np.nan
                )
                odds = odds_ratio_with_pseudocount(a, b, c, d)
                is_associated = category in associated_groups and pd.notna(delta) and delta > 0
                enrichment_rows.append(
                    {
                        "lineage_key": str(row["lineage_key"]),
                        "Species": str(row["Species"]),
                        "entity_id": str(row["entity_id"]),
                        "entity_label": str(row["entity_label"]),
                        "entity_kind": str(row["entity_kind"]),
                        "entity_group": str(row["entity_group"]),
                        "category": category,
                        "n_present_in_category": a,
                        "n_genomes_in_category": n_category,
                        "presence_fraction_in_category": frac_category,
                        "n_present_other_categories": c,
                        "n_genomes_other_categories": n_other,
                        "presence_fraction_other_categories": frac_other,
                        "delta_fraction_category_minus_others": delta,
                        "odds_ratio": odds,
                        "log2_odds_ratio": float(np.log2(odds)) if odds > 0 else np.nan,
                        "pvalue": global_pvalue if is_associated else 1.0,
                        "qvalue_bh": global_qvalue if is_associated else 1.0,
                        "significance": significance_stars(global_qvalue if is_associated else 1.0),
                        "associated_groups": str(row.get("associated_groups", "")),
                        "enrichment_score": global_score,
                        "enrichment_method": "shaiber_willis_logit_rao_qvalue",
                    }
                )
    enrichment_df = pd.DataFrame(enrichment_rows, columns=enrichment_columns + ["associated_groups", "enrichment_score", "enrichment_method"])

    if pairwise_occurrence_rows:
        pairwise_occurrence_df = pd.DataFrame(pairwise_occurrence_rows)
        pairwise_stats_df = run_enrichment_dataframe_compat(
            pairwise_occurrence_df,
            compute_associated_groups_if_missing=True,
            allow_empty_associated_groups=True,
        )
    else:
        pairwise_stats_df = pd.DataFrame()

    pairwise_rows = []
    if not pairwise_stats_df.empty:
        for row in pairwise_stats_df.to_dict("records"):
            a_present = int(round(float(pd.to_numeric(row.get("p_A", 0), errors="coerce")) * float(pd.to_numeric(row.get("N_A", 0), errors="coerce"))))
            a_total = int(pd.to_numeric(row.get("N_A", 0), errors="coerce"))
            b_present = int(round(float(pd.to_numeric(row.get("p_B", 0), errors="coerce")) * float(pd.to_numeric(row.get("N_B", 0), errors="coerce"))))
            b_total = int(pd.to_numeric(row.get("N_B", 0), errors="coerce"))
            a_absent = max(0, a_total - a_present)
            b_absent = max(0, b_total - b_present)
            frac_a = (a_present / a_total) if a_total else np.nan
            frac_b = (b_present / b_total) if b_total else np.nan
            odds = odds_ratio_with_pseudocount(b_present, b_absent, a_present, a_absent)
            pairwise_rows.append(
                {
                    "lineage_key": str(row["lineage_key"]),
                    "Species": str(row["Species"]),
                    "entity_id": str(row["entity_id"]),
                    "entity_label": str(row["entity_label"]),
                    "entity_kind": str(row["entity_kind"]),
                    "entity_group": str(row["entity_group"]),
                    "comparison_class": str(row["comparison_class"]),
                    "category_a": str(row["category_a"]),
                    "category_b": str(row["category_b"]),
                    "category_pair": str(row["category_pair"]),
                    "n_present_a": a_present,
                    "n_genomes_a": a_total,
                    "presence_fraction_a": frac_a,
                    "n_present_b": b_present,
                    "n_genomes_b": b_total,
                    "presence_fraction_b": frac_b,
                    "delta_fraction_b_minus_a": (
                        frac_b - frac_a if pd.notna(frac_a) and pd.notna(frac_b) else np.nan
                    ),
                    "odds_ratio_b_over_a": odds,
                    "log2_odds_ratio_b_over_a": float(np.log2(odds)) if odds > 0 else np.nan,
                    "pvalue": float(pd.to_numeric(row.get("unadjusted_p_value"), errors="coerce")),
                    "qvalue_bh": float(pd.to_numeric(row.get("adjusted_q_value"), errors="coerce")),
                    "significance": significance_stars(float(pd.to_numeric(row.get("adjusted_q_value"), errors="coerce"))),
                    "associated_groups": str(row.get("associated_groups", "")),
                    "enrichment_score": float(pd.to_numeric(row.get("enrichment_score"), errors="coerce")),
                    "enrichment_method": "shaiber_willis_logit_rao_qvalue",
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows, columns=pairwise_columns + ["associated_groups", "enrichment_score", "enrichment_method"])
    status_df = pd.DataFrame(status_rows, columns=status_columns)
    if not enrichment_df.empty:
        enrichment_df = enrichment_df.sort_values(
            by=["qvalue_bh", "pvalue", "enrichment_score", "lineage_key", "entity_label", "category"],
            ascending=[True, True, False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    if not pairwise_df.empty:
        pairwise_df = pairwise_df.sort_values(
            by=["qvalue_bh", "pvalue", "enrichment_score", "lineage_key", "entity_label", "category_pair"],
            ascending=[True, True, False, True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return category_summary, enrichment_df, pairwise_df, status_df


def build_pooled_candidate_entity_comparative_tables(universe_df, entity_presence_df, candidate_df):
    if universe_df is None or universe_df.empty or entity_presence_df is None or entity_presence_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    if candidate_df is None or candidate_df.empty or "passes_all_categories" not in candidate_df.columns:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    selected_lineages = set(
        candidate_df.loc[candidate_df["passes_all_categories"].astype(bool), "lineage_key"].astype(str)
    )
    if not selected_lineages:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    pooled_universe = universe_df.loc[
        universe_df["lineage_key"].astype(str).isin(selected_lineages)
    ].copy()
    pooled_presence = entity_presence_df.loc[
        entity_presence_df["lineage_key"].astype(str).isin(selected_lineages)
    ].copy()
    if pooled_universe.empty or pooled_presence.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
    pooled_universe["lineage_key"] = "ALL_CANDIDATE_LINEAGES"
    pooled_universe["Species"] = "All candidate lineages"
    pooled_presence["lineage_key"] = "ALL_CANDIDATE_LINEAGES"
    pooled_candidate_df = pd.DataFrame(
        [{"lineage_key": "ALL_CANDIDATE_LINEAGES", "passes_all_categories": True}]
    )
    return build_global_entity_comparative_tables(
        universe_df=pooled_universe,
        entity_presence_df=pooled_presence,
        candidate_df=pooled_candidate_df,
        require_all_categories=False,
    )

def build_candidate_entity_review_table(category_summary_df, enrichment_df, pairwise_df, status_df, categories=None):
    output_columns = [
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "source_db",
        "annotation_category",
        "status",
        "n_categories_present",
        "categories_present",
        "categories_absent",
        "enriched_categories_q_lt_0_05",
        "enriched_categories_nominal_p_lt_0_05",
        "best_enriched_category",
        "best_enriched_category_qvalue",
        "best_enriched_category_pvalue",
        "best_enriched_category_delta_fraction",
        "best_enriched_category_log2_odds_ratio",
        "pairwise_significant_q_lt_0_05",
        "pairwise_nominal_p_lt_0_05",
        "max_presence_fraction",
        "min_presence_fraction",
        "range_presence_fraction",
    ]
    if category_summary_df is None or category_summary_df.empty:
        return pd.DataFrame(columns=output_columns)

    if categories is None:
        categories = ordered_methods(category_summary_df["category"].astype(str).drop_duplicates().tolist())
    else:
        categories = ordered_methods(categories)
    key_columns = ["lineage_key", "entity_id"]
    summary = (
        category_summary_df.groupby(key_columns, dropna=False)
        .agg(
            Species=("Species", "first"),
            entity_label=("entity_label", "first"),
            entity_kind=("entity_kind", "first"),
            entity_group=("entity_group", "first"),
            source_db=("source_db", "first"),
            annotation_category=("annotation_category", "first"),
        )
        .reset_index()
    )
    for category in categories:
        cat_safe = sanitize_label(category)
        subset = category_summary_df.loc[category_summary_df["category"].astype(str).eq(category)]
        maps = {
            "n_genomes": subset.set_index(key_columns)["n_genomes"].to_dict() if not subset.empty else {},
            "n_present": subset.set_index(key_columns)["n_present"].to_dict() if not subset.empty else {},
            "presence_fraction": subset.set_index(key_columns)["presence_fraction"].to_dict() if not subset.empty else {},
            "total_count_value": subset.set_index(key_columns)["total_count_value"].to_dict() if not subset.empty else {},
            "mean_count_per_genome": subset.set_index(key_columns)["mean_count_per_genome"].to_dict() if not subset.empty else {},
            "orf_ids": subset.set_index(key_columns)["orf_ids"].to_dict() if not subset.empty else {},
        }
        keys = list(zip(summary["lineage_key"].astype(str), summary["entity_id"].astype(str)))
        for metric, value_map in maps.items():
            summary[f"{cat_safe}_{metric}"] = [value_map.get(key, 0 if metric != "orf_ids" else "") for key in keys]

    if status_df is not None and not status_df.empty:
        status_cols = [
            "lineage_key",
            "entity_id",
            "n_categories_present",
            "categories_present",
            "categories_absent",
            "status",
            "max_presence_fraction",
            "min_presence_fraction",
            "range_presence_fraction",
        ]
        summary = summary.merge(
            status_df.loc[:, [column for column in status_cols if column in status_df.columns]].drop_duplicates(),
            on=key_columns,
            how="left",
        )
    else:
        for column in [
            "n_categories_present",
            "categories_present",
            "categories_absent",
            "status",
            "max_presence_fraction",
            "min_presence_fraction",
            "range_presence_fraction",
        ]:
            summary[column] = np.nan if "fraction" in column else ""

    if enrichment_df is not None and not enrichment_df.empty:
        enrichment = enrichment_df.copy()
        enrichment["qvalue_bh"] = pd.to_numeric(enrichment["qvalue_bh"], errors="coerce")
        enrichment["pvalue"] = pd.to_numeric(enrichment["pvalue"], errors="coerce")
        enrichment["delta_fraction_category_minus_others"] = pd.to_numeric(
            enrichment["delta_fraction_category_minus_others"],
            errors="coerce",
        )
        enrichment["log2_odds_ratio"] = pd.to_numeric(enrichment["log2_odds_ratio"], errors="coerce")
        positive = enrichment.loc[
            enrichment["delta_fraction_category_minus_others"].gt(0)
        ].copy()
        q_sig = (
            positive.loc[positive["qvalue_bh"].lt(0.05)]
            .groupby(key_columns, dropna=False)["category"]
            .apply(lambda s: ";".join(ordered_methods(s.astype(str).tolist())))
            .reset_index(name="enriched_categories_q_lt_0_05")
        )
        p_sig = (
            positive.loc[positive["pvalue"].lt(0.05)]
            .groupby(key_columns, dropna=False)["category"]
            .apply(lambda s: ";".join(ordered_methods(s.astype(str).tolist())))
            .reset_index(name="enriched_categories_nominal_p_lt_0_05")
        )
        best = positive.sort_values(
            by=["qvalue_bh", "pvalue", "delta_fraction_category_minus_others", "log2_odds_ratio"],
            ascending=[True, True, False, False],
            kind="mergesort",
            na_position="last",
        ).groupby(key_columns, as_index=False, sort=False).head(1)
        best = best.rename(
            columns={
                "category": "best_enriched_category",
                "qvalue_bh": "best_enriched_category_qvalue",
                "pvalue": "best_enriched_category_pvalue",
                "delta_fraction_category_minus_others": "best_enriched_category_delta_fraction",
                "log2_odds_ratio": "best_enriched_category_log2_odds_ratio",
            }
        )
        best_cols = [
            "lineage_key",
            "entity_id",
            "best_enriched_category",
            "best_enriched_category_qvalue",
            "best_enriched_category_pvalue",
            "best_enriched_category_delta_fraction",
            "best_enriched_category_log2_odds_ratio",
        ]
        for add_df in [q_sig, p_sig, best.loc[:, best_cols] if not best.empty else pd.DataFrame(columns=best_cols)]:
            summary = summary.merge(add_df, on=key_columns, how="left")
    else:
        for column in [
            "enriched_categories_q_lt_0_05",
            "enriched_categories_nominal_p_lt_0_05",
            "best_enriched_category",
            "best_enriched_category_qvalue",
            "best_enriched_category_pvalue",
            "best_enriched_category_delta_fraction",
            "best_enriched_category_log2_odds_ratio",
        ]:
            summary[column] = np.nan if column.endswith(("qvalue", "pvalue", "fraction", "ratio")) else ""

    if pairwise_df is not None and not pairwise_df.empty:
        pairwise = pairwise_df.copy()
        pairwise["qvalue_bh"] = pd.to_numeric(pairwise["qvalue_bh"], errors="coerce")
        pairwise["pvalue"] = pd.to_numeric(pairwise["pvalue"], errors="coerce")
        pairwise_q = (
            pairwise.loc[pairwise["qvalue_bh"].lt(0.05)]
            .groupby(key_columns, dropna=False)["category_pair"]
            .apply(lambda s: ";".join(sorted(set(s.astype(str)))))
            .reset_index(name="pairwise_significant_q_lt_0_05")
        )
        pairwise_p = (
            pairwise.loc[pairwise["pvalue"].lt(0.05)]
            .groupby(key_columns, dropna=False)["category_pair"]
            .apply(lambda s: ";".join(sorted(set(s.astype(str)))))
            .reset_index(name="pairwise_nominal_p_lt_0_05")
        )
        summary = summary.merge(pairwise_q, on=key_columns, how="left")
        summary = summary.merge(pairwise_p, on=key_columns, how="left")
    else:
        summary["pairwise_significant_q_lt_0_05"] = ""
        summary["pairwise_nominal_p_lt_0_05"] = ""

    for column in [
        "enriched_categories_q_lt_0_05",
        "enriched_categories_nominal_p_lt_0_05",
        "best_enriched_category",
        "pairwise_significant_q_lt_0_05",
        "pairwise_nominal_p_lt_0_05",
        "categories_present",
        "categories_absent",
        "status",
    ]:
        if column in summary.columns:
            summary[column] = summary[column].fillna("")

    category_columns = []
    for category in categories:
        cat_safe = sanitize_label(category)
        category_columns.extend(
            [
                f"{cat_safe}_n_genomes",
                f"{cat_safe}_n_present",
                f"{cat_safe}_presence_fraction",
                f"{cat_safe}_total_count_value",
                f"{cat_safe}_mean_count_per_genome",
                f"{cat_safe}_orf_ids",
            ]
        )
    ordered_columns = output_columns + [column for column in category_columns if column in summary.columns]
    for column in ordered_columns:
        if column not in summary.columns:
            summary[column] = np.nan if any(token in column for token in ["fraction", "qvalue", "pvalue", "ratio"]) else ""
    return summary.loc[:, ordered_columns].sort_values(
        by=[
            "lineage_key",
            "status",
            "best_enriched_category_qvalue",
            "range_presence_fraction",
            "entity_label",
        ],
        ascending=[True, True, True, False, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

def ordered_or_clustered_row_index(matrix, clustered=False):
    if matrix.empty:
        return []
    if not clustered or matrix.shape[0] < 3:
        return matrix.index.tolist()
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist
        values = matrix.to_numpy(dtype=float)
        distances = pdist(values, metric="euclidean")
        if distances.size == 0 or not np.isfinite(distances).all():
            return matrix.index.tolist()
        linkage_matrix = linkage(distances, method="average")
        return matrix.index[leaves_list(linkage_matrix)].tolist()
    except Exception:
        return matrix.index.tolist()

def select_entity_heatmap_rows(category_summary_df, enrichment_df, max_rows=250):
    if category_summary_df.empty:
        return []
    key_columns = ["lineage_key", "entity_id"]
    summary = (
        category_summary_df.groupby(key_columns, dropna=False)
        .agg(
            Species=("Species", "first"),
            entity_label=("entity_label", "first"),
            max_presence_fraction=("presence_fraction", "max"),
            min_presence_fraction=("presence_fraction", "min"),
            total_present=("n_present", "sum"),
        )
        .reset_index()
    )
    summary["range_presence_fraction"] = summary["max_presence_fraction"] - summary["min_presence_fraction"]
    if enrichment_df is not None and not enrichment_df.empty:
        enrich_score = (
            enrichment_df.groupby(key_columns, dropna=False)
            .agg(
                min_qvalue=("qvalue_bh", "min"),
                min_pvalue=("pvalue", "min"),
                max_abs_delta=("delta_fraction_category_minus_others", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            )
            .reset_index()
        )
        summary = summary.merge(enrich_score, on=key_columns, how="left")
    else:
        summary["min_qvalue"] = np.nan
        summary["min_pvalue"] = np.nan
        summary["max_abs_delta"] = np.nan
    summary = summary.sort_values(
        by=["min_qvalue", "min_pvalue", "max_abs_delta", "range_presence_fraction", "total_present", "Species", "entity_label"],
        ascending=[True, True, False, False, False, True, True],
        kind="mergesort",
        na_position="last",
    )
    return list(
        zip(
            summary["lineage_key"].astype(str).head(int(max_rows)).tolist(),
            summary["entity_id"].astype(str).head(int(max_rows)).tolist(),
        )
    )

def plot_entity_prevalence_heatmap(category_summary_df, enrichment_df, output_base, max_rows=250, clustered=False, title=None, categories=None):
    ensure_plotting()
    if category_summary_df is None or category_summary_df.empty:
        return False
    selected_keys = set(select_entity_heatmap_rows(category_summary_df, enrichment_df, max_rows=max_rows))
    if not selected_keys:
        return False
    working = category_summary_df.copy()
    working["_key"] = list(zip(working["lineage_key"].astype(str), working["entity_id"].astype(str)))
    working = working.loc[working["_key"].isin(selected_keys)].copy()
    if working.empty:
        return False
    working["row_label"] = (
        working["Species"].astype(str)
        + " | "
        + working["entity_label"].astype(str)
    )
    if categories is None:
        categories = ordered_methods(working["category"].astype(str).drop_duplicates().tolist())
    else:
        categories = ordered_methods(categories)
    matrix = working.pivot_table(
        index="row_label",
        columns="category",
        values="presence_fraction",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=categories, fill_value=0)
    order_stats = (
        working.groupby("row_label", dropna=False)
        .agg(
            max_presence=("presence_fraction", "max"),
            min_presence=("presence_fraction", "min"),
            total_present=("n_present", "sum"),
        )
        .reset_index()
    )
    order_stats["range_presence"] = order_stats["max_presence"] - order_stats["min_presence"]
    ordered_rows = order_stats.sort_values(
        by=["range_presence", "total_present", "row_label"],
        ascending=[False, False, True],
        kind="mergesort",
    )["row_label"].astype(str).tolist()
    matrix = matrix.reindex(index=[row for row in ordered_rows if row in matrix.index])
    matrix = matrix.reindex(index=ordered_or_clustered_row_index(matrix, clustered=clustered))
    if matrix.empty:
        return False
    y_labels = [
        label if len(label) <= 120 else label[:117] + "..."
        for label in matrix.index.astype(str).tolist()
    ]
    plt_local = ensure_plotting()
    width = max(8, len(categories) * 1.4 + 4)
    height = min(55, max(7, matrix.shape[0] * 0.18))
    fig, ax = plt_local.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(y_labels, fontsize=4 if matrix.shape[0] > 150 else 5)
    ax.set_xlabel("Genome category")
    ax.set_ylabel("Lineage | function")
    ax.set_title(title or ("HCA functional prevalence heatmap" if clustered else "Ordered functional prevalence heatmap"))
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Fraction of genomes with function")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

FOCUSED_CATEGORY_COMPARISONS = [
    "SAGs -> xPG_SAGs",
    "SAGs -> xPG_MAGs",
    "MAGs -> xPG_SAGs",
    "MAGs -> xPG_MAGs",
    "SAGs -> MAGs",
    "xPG_SAGs -> xPG_MAGs",
]

def category_pair_plot_order(category_pair):
    return {pair: idx for idx, pair in enumerate(FOCUSED_CATEGORY_COMPARISONS)}.get(str(category_pair), 100)

def category_pair_filename_token(category_pair):
    return sanitize_label(str(category_pair).replace(" -> ", "_vs_"))

def prepare_pairwise_differential_plot_rows(pairwise_df, min_genomes_per_category=3):
    if pairwise_df is None or pairwise_df.empty:
        return pd.DataFrame()
    required = {
        "lineage_key",
        "Species",
        "entity_id",
        "entity_label",
        "category_pair",
        "n_genomes_a",
        "n_genomes_b",
        "presence_fraction_a",
        "presence_fraction_b",
        "delta_fraction_b_minus_a",
    }
    if not required.issubset(set(pairwise_df.columns)):
        return pd.DataFrame()
    working = pairwise_df.copy()
    working = working.loc[working["category_pair"].astype(str).isin(set(FOCUSED_CATEGORY_COMPARISONS))].copy()
    if working.empty:
        return pd.DataFrame()
    numeric_columns = [
        "n_genomes_a",
        "n_genomes_b",
        "presence_fraction_a",
        "presence_fraction_b",
        "delta_fraction_b_minus_a",
        "qvalue_bh",
        "pvalue",
        "odds_ratio_b_over_a",
        "log2_odds_ratio_b_over_a",
    ]
    for column in numeric_columns:
        if column not in working.columns:
            working[column] = np.nan
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.loc[
        working["n_genomes_a"].ge(int(min_genomes_per_category))
        & working["n_genomes_b"].ge(int(min_genomes_per_category))
    ].copy()
    if working.empty:
        return pd.DataFrame()
    working = working.rename(columns={"qvalue_bh": "qvalue_bh_global"})
    working["qvalue_bh_pair"] = np.nan
    working["qvalue_bh_pair_lineage"] = np.nan
    for _, idx in working.groupby("category_pair", dropna=False).groups.items():
        working.loc[idx, "qvalue_bh_pair"] = benjamini_hochberg_adjust(working.loc[idx, "pvalue"]).to_numpy()
    for _, idx in working.groupby(["category_pair", "lineage_key"], dropna=False).groups.items():
        working.loc[idx, "qvalue_bh_pair_lineage"] = benjamini_hochberg_adjust(working.loc[idx, "pvalue"]).to_numpy()
    pvalue_for_plot = working["pvalue"].where(working["pvalue"].gt(0), np.nan)
    positive_pvalue = pvalue_for_plot.loc[pvalue_for_plot.gt(0)]
    pvalue_floor = max(1e-300, float(positive_pvalue.min()) / 10.0) if not positive_pvalue.empty else 1.0
    pvalue_for_plot = pvalue_for_plot.fillna(pvalue_floor).clip(lower=pvalue_floor)
    working["neg_log10_pvalue"] = (-np.log10(pvalue_for_plot)).clip(upper=50)
    qvalue_for_plot = working["qvalue_bh_pair"].where(working["qvalue_bh_pair"].gt(0), np.nan)
    positive_qvalue = qvalue_for_plot.loc[qvalue_for_plot.gt(0)]
    qvalue_floor = max(1e-300, float(positive_qvalue.min()) / 10.0) if not positive_qvalue.empty else 1.0
    qvalue_for_plot = qvalue_for_plot.fillna(qvalue_floor).clip(lower=qvalue_floor)
    working["neg_log10_qvalue_pair"] = (-np.log10(qvalue_for_plot)).clip(upper=50)
    working["mean_presence_fraction"] = (
        working["presence_fraction_a"].fillna(0) + working["presence_fraction_b"].fillna(0)
    ) / 2.0
    working["abs_delta"] = working["delta_fraction_b_minus_a"].abs()
    working["is_nominal_significant"] = working["pvalue"].lt(0.05)
    working["is_pair_bh_significant"] = working["qvalue_bh_pair"].lt(0.05)
    working["is_pair_lineage_bh_significant"] = working["qvalue_bh_pair_lineage"].lt(0.05)
    working["is_global_bh_significant"] = working["qvalue_bh_global"].lt(0.05)
    working["is_significant"] = working["is_nominal_significant"]
    lineage_label = working["Species"].astype(str).str.strip().copy()
    missing_lineage = lineage_label.eq("") | lineage_label.str.lower().isin({"nan", "none"})
    lineage_label = lineage_label.mask(missing_lineage, working["lineage_key"].astype(str))
    working["lineage_label"] = lineage_label
    working["plot_function_label"] = working["entity_label"].astype(str)
    working["volcano_effect_log2_odds_ratio"] = working["log2_odds_ratio_b_over_a"].replace([np.inf, -np.inf], np.nan)
    working["pair_order"] = working["category_pair"].map(category_pair_plot_order).astype(int)
    return working.sort_values(
        by=["pair_order", "qvalue_bh_pair", "pvalue", "abs_delta", "lineage_label", "plot_function_label"],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

def lineage_color_lookup(values, plt_local):
    lineages = sorted(pd.Series(values).astype(str).dropna().unique().tolist())
    if not lineages:
        return {}
    if len(lineages) <= 20:
        cmap = plt_local.get_cmap("tab20")
        return {lineage: cmap(idx % 20) for idx, lineage in enumerate(lineages)}
    cmap = plt_local.get_cmap("hsv")
    denominator = max(1, len(lineages))
    return {lineage: cmap(idx / denominator) for idx, lineage in enumerate(lineages)}

def add_significant_lineage_legend(ax, sig_df, color_lookup, max_lineages=12):
    if sig_df.empty:
        return
    lineage_counts = sig_df["lineage_label"].astype(str).value_counts()
    selected = lineage_counts.head(int(max_lineages)).index.astype(str).tolist()
    handles = [
        ax.scatter([], [], s=28, color=color_lookup.get(lineage, "#377eb8"), edgecolor="#333333", linewidth=0.25)
        for lineage in selected
    ]
    labels = [lineage if len(lineage) <= 45 else lineage[:42] + "..." for lineage in selected]
    if len(lineage_counts) > len(selected):
        handles.append(ax.scatter([], [], s=28, color="#777777", edgecolor="#333333", linewidth=0.25))
        labels.append(f"{len(lineage_counts) - len(selected)} additional lineages")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, title="Significant lineage")

def plot_category_pairwise_volcano(pair_df, category_pair, output_base, entity_label):
    if pair_df.empty:
        return False
    plt_local = ensure_plotting()
    plot_df = pair_df.copy()
    sig_df = plot_df.loc[plot_df["is_significant"].astype(bool)].copy()
    nonsig_df = plot_df.loc[~plot_df["is_significant"].astype(bool)].copy()
    color_lookup = lineage_color_lookup(sig_df["lineage_label"], plt_local)
    effect_values = pd.to_numeric(plot_df["volcano_effect_log2_odds_ratio"], errors="coerce")
    finite_effect = effect_values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_effect.empty:
        plot_df["volcano_effect_log2_odds_ratio"] = plot_df["delta_fraction_b_minus_a"]
        effect_label = "Presence fraction delta (B - A)"
        x_limit = 1.02
    else:
        max_abs_effect = float(finite_effect.abs().quantile(0.995))
        x_limit = min(12.0, max(2.0, max_abs_effect * 1.08))
        effect_label = "log2 odds ratio (B over A)"
    fig, ax = plt_local.subplots(figsize=(10.5, 7.2))
    if not nonsig_df.empty:
        ax.scatter(
            nonsig_df["volcano_effect_log2_odds_ratio"],
            nonsig_df["neg_log10_pvalue"],
            s=15,
            color="#cfcfcf",
            edgecolors="none",
            alpha=0.55,
            label="Nominal p >= 0.05",
        )
    for lineage, group in sig_df.groupby("lineage_label", dropna=False):
        edgecolors = np.where(group["is_pair_bh_significant"].astype(bool), "black", "#555555")
        ax.scatter(
            group["volcano_effect_log2_odds_ratio"],
            group["neg_log10_pvalue"],
            s=24,
            color=color_lookup.get(str(lineage), "#377eb8"),
            edgecolors=edgecolors,
            linewidth=np.where(group["is_pair_bh_significant"].astype(bool), 0.55, 0.25),
            alpha=0.9,
        )
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax.axhline(-np.log10(0.05), color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlim(-x_limit, x_limit)
    ax.set_xlabel(effect_label)
    ax.set_ylabel("-log10(Fisher exact p-value)")
    ax.set_title(f"{entity_label}: {category_pair} volcano")
    ax.text(
        0.98,
        0.02,
        (
            f"functions={len(plot_df)}; nominal p<0.05={len(sig_df)}; "
            f"pair-BH q<0.05={int(plot_df['is_pair_bh_significant'].sum())}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    add_significant_lineage_legend(ax, sig_df, color_lookup)
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

def plot_category_pairwise_ma(pair_df, category_pair, output_base, entity_label):
    if pair_df.empty:
        return False
    plt_local = ensure_plotting()
    plot_df = pair_df.copy()
    sig_df = plot_df.loc[plot_df["is_significant"].astype(bool)].copy()
    nonsig_df = plot_df.loc[~plot_df["is_significant"].astype(bool)].copy()
    color_lookup = lineage_color_lookup(sig_df["lineage_label"], plt_local)
    fig, ax = plt_local.subplots(figsize=(10.5, 7.2))
    if not nonsig_df.empty:
        ax.scatter(
            nonsig_df["mean_presence_fraction"],
            nonsig_df["delta_fraction_b_minus_a"],
            s=15,
            color="#cfcfcf",
            edgecolors="none",
            alpha=0.55,
            label="Nominal p >= 0.05",
        )
    for lineage, group in sig_df.groupby("lineage_label", dropna=False):
        edgecolors = np.where(group["is_pair_bh_significant"].astype(bool), "black", "#555555")
        ax.scatter(
            group["mean_presence_fraction"],
            group["delta_fraction_b_minus_a"],
            s=24,
            color=color_lookup.get(str(lineage), "#377eb8"),
            edgecolors=edgecolors,
            linewidth=np.where(group["is_pair_bh_significant"].astype(bool), 0.55, 0.25),
            alpha=0.9,
        )
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle=":")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("Mean presence fraction across A and B")
    ax.set_ylabel("Presence fraction delta (B - A)")
    ax.set_title(f"{entity_label}: {category_pair} MA-style differential plot")
    ax.text(
        0.98,
        0.02,
        (
            f"functions={len(plot_df)}; nominal p<0.05={len(sig_df)}; "
            f"pair-BH q<0.05={int(plot_df['is_pair_bh_significant'].sum())}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    add_significant_lineage_legend(ax, sig_df, color_lookup)
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

def write_category_pairwise_differential_plots(pairwise_df, output_dir, entity_label, min_genomes_per_category=3):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrote_paths = []
    plot_rows = prepare_pairwise_differential_plot_rows(
        pairwise_df,
        min_genomes_per_category=min_genomes_per_category,
    )
    all_data_out = output_dir / "category_pairwise_differential.filtered_plot_data.tsv"
    plot_rows.to_csv(all_data_out, sep="\t", index=False)
    wrote_paths.append(all_data_out)
    if plot_rows.empty:
        return wrote_paths
    for category_pair in FOCUSED_CATEGORY_COMPARISONS:
        pair_df = plot_rows.loc[plot_rows["category_pair"].astype(str).eq(category_pair)].copy()
        if pair_df.empty:
            continue
        token = category_pair_filename_token(category_pair)
        pair_data_out = output_dir / f"category_pairwise_differential.{token}.plot_data.tsv"
        pair_df.to_csv(pair_data_out, sep="\t", index=False)
        wrote_paths.append(pair_data_out)
        volcano_base = output_dir / f"category_pairwise_differential.{token}.volcano"
        ma_base = output_dir / f"category_pairwise_differential.{token}.ma_style"
        if plot_category_pairwise_volcano(pair_df, category_pair, volcano_base, entity_label):
            wrote_paths.extend([Path(str(volcano_base) + ".png"), Path(str(volcano_base) + ".pdf")])
        if plot_category_pairwise_ma(pair_df, category_pair, ma_base, entity_label):
            wrote_paths.extend([Path(str(ma_base) + ".png"), Path(str(ma_base) + ".pdf")])
    return wrote_paths

def plot_lineage_annotation_genome_heatmap(matrix_df, output_base, max_rows=250, clustered=False, title=None):
    ensure_plotting()
    if matrix_df is None or matrix_df.empty or "annotation_text" not in matrix_df.columns:
        return False
    metadata_columns = {
        "annotation_text",
        "n_genomes_present",
        "source_db",
        "annotation_category",
        "categories_present",
        "categories_absent",
        "category_presence_pattern",
        "orf_ids_by_category",
    }
    genome_columns = [column for column in matrix_df.columns if column not in metadata_columns]
    if not genome_columns:
        return False
    working = matrix_df.copy()
    for column in genome_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).clip(lower=0, upper=1)
    working["_presence_total"] = working[genome_columns].sum(axis=1)
    working = working.loc[working["_presence_total"].gt(0)].copy()
    if working.empty:
        return False
    working = working.sort_values(
        by=["_presence_total", "annotation_text"],
        ascending=[False, True],
        kind="mergesort",
    ).head(int(max_rows)).copy()
    matrix = working.set_index("annotation_text")[genome_columns].copy()
    matrix = matrix.reindex(index=ordered_or_clustered_row_index(matrix, clustered=clustered))
    y_labels = [
        label if len(label) <= 115 else label[:112] + "..."
        for label in matrix.index.astype(str).tolist()
    ]
    x_labels = []
    category_boundaries = []
    previous_category = None
    for idx, label in enumerate(genome_columns):
        parts = str(label).split("|")
        category = parts[0] if parts else str(label)
        if previous_category is not None and category != previous_category:
            category_boundaries.append(idx - 0.5)
        previous_category = category
        x_labels.append("|".join(parts[:3]) if len(parts) >= 3 else str(label))
    plt_local = ensure_plotting()
    width = min(65, max(12, len(genome_columns) * 0.22))
    height = min(55, max(8, matrix.shape[0] * 0.2))
    fig, ax = plt_local.subplots(figsize=(width, height))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title or ("HCA annotation-by-genome presence" if clustered else "Ordered annotation-by-genome presence"))
    ax.set_xlabel("Genome, grouped by category")
    ax.set_ylabel("Functional annotation")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(y_labels, fontsize=4 if matrix.shape[0] > 150 else 5)
    tick_step = max(1, int(np.ceil(len(x_labels) / 120.0)))
    tick_positions = np.arange(0, len(x_labels), tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([x_labels[idx] for idx in tick_positions], rotation=90, fontsize=3 if len(x_labels) > 100 else 5)
    for boundary in category_boundaries:
        ax.axvline(boundary, color="#d0d0d0", linewidth=0.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Absent", "Present"])
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

def plot_circular_feature_summary(category_summary_df, enrichment_df, output_base, max_features=48, categories=None):
    ensure_plotting()
    if category_summary_df is None or category_summary_df.empty:
        return False
    working = category_summary_df.copy()
    if "entity_kind" in working.columns:
        working = working.loc[working["entity_kind"].astype(str).eq("marker_or_reference_feature")].copy()
    if working.empty:
        return False
    feature_summary = (
        working.groupby(["entity_id", "entity_label", "entity_group", "category"], dropna=False)
        .agg(mean_prevalence=("presence_fraction", "mean"))
        .reset_index()
    )
    if categories is None:
        categories = ordered_methods(feature_summary["category"].astype(str).drop_duplicates().tolist())
    else:
        categories = ordered_methods(categories)
    matrix = feature_summary.pivot_table(
        index="entity_label",
        columns="category",
        values="mean_prevalence",
        aggfunc="max",
        fill_value=0,
    ).reindex(columns=categories, fill_value=0)
    if matrix.empty:
        return False
    score = pd.DataFrame(index=matrix.index)
    score["range_prevalence"] = matrix.max(axis=1) - matrix.min(axis=1)
    score["max_prevalence"] = matrix.max(axis=1)
    if enrichment_df is not None and not enrichment_df.empty:
        enrich = enrichment_df.loc[enrichment_df["entity_kind"].astype(str).eq("marker_or_reference_feature")].copy()
        if not enrich.empty:
            enrich_score = enrich.groupby("entity_label", dropna=False)["qvalue_bh"].min()
            score["min_qvalue"] = score.index.map(enrich_score).astype(float)
        else:
            score["min_qvalue"] = np.nan
    else:
        score["min_qvalue"] = np.nan
    selected = score.sort_values(
        by=["min_qvalue", "range_prevalence", "max_prevalence"],
        ascending=[True, False, False],
        kind="mergesort",
        na_position="last",
    ).head(int(max_features)).index.tolist()
    matrix = matrix.reindex(index=selected)
    n_features = matrix.shape[0]
    n_categories = matrix.shape[1]
    if n_features == 0 or n_categories == 0:
        return False
    theta = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    width = (2 * np.pi / max(1, n_features)) * 0.92
    plt_local = ensure_plotting()
    fig = plt_local.figure(figsize=(14, 14))
    ax = fig.add_subplot(111, projection="polar")
    cmap = plt_local.cm.viridis
    norm = plt_local.Normalize(vmin=0, vmax=1)
    for ring_index, category in enumerate(categories):
        values = matrix[category].to_numpy(dtype=float)
        for angle, value in zip(theta, values):
            ax.bar(
                angle,
                0.82,
                width=width,
                bottom=ring_index + 1,
                color=cmap(norm(value)),
                edgecolor="white",
                linewidth=0.2,
                align="edge",
            )
    ax.set_ylim(0, n_categories + 2.2)
    ax.set_yticks(np.arange(1.41, n_categories + 1.41))
    ax.set_yticklabels(categories, fontsize=10)
    label_step = max(1, int(np.ceil(n_features / 48.0)))
    ax.set_xticks(theta[::label_step] + width / 2)
    labels = [
        label if len(label) <= 28 else label[:25] + "..."
        for label in matrix.index.astype(str).tolist()[::label_step]
    ]
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", pad=12)
    ax.grid(False)
    ax.set_title("High-level marker/reference feature prevalence by genome category", y=1.08, fontsize=14)
    sm = plt_local.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.08)
    cbar.set_label("Mean lineage-level prevalence")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True

def color_for_group(group_value):
    palette = {
        "marker": "#1f77b4",
        "reference_mode": "#ff7f0e",
        "synthetic": "#2ca02c",
        "functional_annotation": "#4c78a8",
        "combined": "#9467bd",
    }
    text = str(group_value).strip()
    if text in palette:
        return palette[text]
    if not text:
        return "#777777"
    colors = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
    ]
    return colors[sum(ord(ch) for ch in text) % len(colors)]

def build_category_entity_visual_tables(category_summary_df, enrichment_df, categories=None):
    node_columns = [
        "node_id",
        "category",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "n_lineages_present",
        "mean_presence_fraction",
        "max_presence_fraction",
        "n_enriched_q_lt_0_05",
        "n_enriched_nominal_p_lt_0_05",
        "rank_score",
    ]
    link_columns = [
        "link_id",
        "category_a",
        "category_b",
        "category_pair",
        "entity_id",
        "entity_label",
        "entity_kind",
        "entity_group",
        "n_lineages_shared",
        "mean_prevalence_a",
        "mean_prevalence_b",
        "mean_min_prevalence",
        "mean_max_prevalence",
        "supporting_lineages",
        "rank_score",
    ]
    track_columns = [
        "category",
        "entity_group",
        "n_entities_present",
        "n_lineage_entity_observations",
        "mean_presence_fraction",
        "n_entities_enriched_q_lt_0_05",
        "n_entities_enriched_nominal_p_lt_0_05",
    ]
    if category_summary_df is None or category_summary_df.empty:
        return (
            pd.DataFrame(columns=node_columns),
            pd.DataFrame(columns=link_columns),
            pd.DataFrame(columns=track_columns),
        )
    if categories is None:
        categories = ordered_methods(category_summary_df["category"].astype(str).drop_duplicates().tolist())
    else:
        categories = ordered_methods(categories)

    working = category_summary_df.copy()
    working["category"] = working["category"].astype(str).str.strip().map(canonical_method_label)
    working["entity_id"] = working["entity_id"].astype(str).str.strip()
    working["entity_label"] = working["entity_label"].astype(str).str.strip()
    working["entity_kind"] = working.get("entity_kind", "").astype(str).str.strip()
    working["entity_group"] = working.get("entity_group", "").astype(str).str.strip()
    working["n_present"] = pd.to_numeric(working.get("n_present", 0), errors="coerce").fillna(0)
    working["presence_fraction"] = pd.to_numeric(working.get("presence_fraction", 0), errors="coerce").fillna(0)
    present = working.loc[
        working["category"].ne("")
        & working["entity_id"].ne("")
        & working["n_present"].gt(0)
    ].copy()
    if present.empty:
        return (
            pd.DataFrame(columns=node_columns),
            pd.DataFrame(columns=link_columns),
            pd.DataFrame(columns=track_columns),
        )

    enrichment_counts = pd.DataFrame()
    if enrichment_df is not None and not enrichment_df.empty:
        enrichment = enrichment_df.copy()
        enrichment["category"] = enrichment["category"].astype(str).str.strip().map(canonical_method_label)
        enrichment["entity_id"] = enrichment["entity_id"].astype(str).str.strip()
        enrichment["qvalue_bh"] = pd.to_numeric(enrichment.get("qvalue_bh"), errors="coerce")
        enrichment["pvalue"] = pd.to_numeric(enrichment.get("pvalue"), errors="coerce")
        enrichment["delta_fraction_category_minus_others"] = pd.to_numeric(
            enrichment.get("delta_fraction_category_minus_others"),
            errors="coerce",
        )
        enrichment = enrichment.loc[enrichment["delta_fraction_category_minus_others"].gt(0)].copy()
        if not enrichment.empty:
            enrichment_counts = (
                enrichment.groupby(["category", "entity_id"], dropna=False)
                .agg(
                    n_enriched_q_lt_0_05=("qvalue_bh", lambda s: int(pd.to_numeric(s, errors="coerce").lt(0.05).sum())),
                    n_enriched_nominal_p_lt_0_05=("pvalue", lambda s: int(pd.to_numeric(s, errors="coerce").lt(0.05).sum())),
                )
                .reset_index()
            )

    nodes = (
        present.groupby(["category", "entity_id"], dropna=False)
        .agg(
            entity_label=("entity_label", "first"),
            entity_kind=("entity_kind", "first"),
            entity_group=("entity_group", "first"),
            n_lineages_present=("lineage_key", lambda s: int(pd.Series(s).astype(str).nunique())),
            mean_presence_fraction=("presence_fraction", "mean"),
            max_presence_fraction=("presence_fraction", "max"),
        )
        .reset_index()
    )
    if not enrichment_counts.empty:
        nodes = nodes.merge(enrichment_counts, on=["category", "entity_id"], how="left")
    for column in ["n_enriched_q_lt_0_05", "n_enriched_nominal_p_lt_0_05"]:
        if column not in nodes.columns:
            nodes[column] = 0
        nodes[column] = pd.to_numeric(nodes[column], errors="coerce").fillna(0).astype(int)
    nodes["rank_score"] = (
        pd.to_numeric(nodes["n_enriched_q_lt_0_05"], errors="coerce").fillna(0) * 1000
        + pd.to_numeric(nodes["n_enriched_nominal_p_lt_0_05"], errors="coerce").fillna(0) * 100
        + pd.to_numeric(nodes["n_lineages_present"], errors="coerce").fillna(0) * 10
        + pd.to_numeric(nodes["mean_presence_fraction"], errors="coerce").fillna(0)
    )
    nodes["node_id"] = nodes["category"].astype(str) + "||" + nodes["entity_id"].astype(str)
    nodes = nodes.loc[:, node_columns].sort_values(
        by=["rank_score", "category", "entity_label"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    link_rows = []
    for (lineage_key, entity_id), group in present.groupby(["lineage_key", "entity_id"], dropna=False):
        category_rows = group.drop_duplicates(subset=["category"]).copy()
        if len(category_rows) < 2:
            continue
        records = category_rows.to_dict("records")
        for row_a, row_b in itertools.combinations(records, 2):
            category_a = str(row_a.get("category", "")).strip()
            category_b = str(row_b.get("category", "")).strip()
            if not category_a or not category_b or category_a == category_b:
                continue
            normalized_a, normalized_b, swapped, _comparison_class = normalized_reporting_pair(category_a, category_b)
            if swapped:
                row_a, row_b = row_b, row_a
            category_a, category_b = normalized_a, normalized_b
            prevalence_a = float(pd.to_numeric(pd.Series([row_a.get("presence_fraction")]), errors="coerce").fillna(0).iat[0])
            prevalence_b = float(pd.to_numeric(pd.Series([row_b.get("presence_fraction")]), errors="coerce").fillna(0).iat[0])
            link_rows.append(
                {
                    "lineage_key": str(lineage_key),
                    "Species": str(row_a.get("Species", "")),
                    "category_a": category_a,
                    "category_b": category_b,
                    "category_pair": f"{category_a} | {category_b}",
                    "entity_id": str(entity_id),
                    "entity_label": str(row_a.get("entity_label", "")),
                    "entity_kind": str(row_a.get("entity_kind", "")),
                    "entity_group": str(row_a.get("entity_group", "")),
                    "prevalence_a": prevalence_a,
                    "prevalence_b": prevalence_b,
                    "min_prevalence": min(prevalence_a, prevalence_b),
                    "max_prevalence": max(prevalence_a, prevalence_b),
                }
            )
    if link_rows:
        link_source = pd.DataFrame(link_rows)
        links = (
            link_source.groupby(["category_a", "category_b", "category_pair", "entity_id"], dropna=False)
            .agg(
                entity_label=("entity_label", "first"),
                entity_kind=("entity_kind", "first"),
                entity_group=("entity_group", "first"),
                n_lineages_shared=("lineage_key", lambda s: int(pd.Series(s).astype(str).nunique())),
                mean_prevalence_a=("prevalence_a", "mean"),
                mean_prevalence_b=("prevalence_b", "mean"),
                mean_min_prevalence=("min_prevalence", "mean"),
                mean_max_prevalence=("max_prevalence", "mean"),
                supporting_lineages=("lineage_key", join_unique_values),
            )
            .reset_index()
        )
        links["rank_score"] = (
            pd.to_numeric(links["n_lineages_shared"], errors="coerce").fillna(0) * 10
            + pd.to_numeric(links["mean_min_prevalence"], errors="coerce").fillna(0)
            + pd.to_numeric(links["mean_max_prevalence"], errors="coerce").fillna(0)
        )
        links["link_id"] = (
            links["category_a"].astype(str)
            + "||"
            + links["category_b"].astype(str)
            + "||"
            + links["entity_id"].astype(str)
        )
        links = links.loc[:, link_columns].sort_values(
            by=["rank_score", "category_pair", "entity_label"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        links = pd.DataFrame(columns=link_columns)

    tracks = (
        present.groupby(["category", "entity_group"], dropna=False)
        .agg(
            n_entities_present=("entity_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            n_lineage_entity_observations=("lineage_key", "size"),
            mean_presence_fraction=("presence_fraction", "mean"),
        )
        .reset_index()
    )
    if not enrichment_counts.empty:
        enrichment_track = nodes.groupby(["category", "entity_group"], dropna=False).agg(
            n_entities_enriched_q_lt_0_05=("n_enriched_q_lt_0_05", lambda s: int(pd.to_numeric(s, errors="coerce").gt(0).sum())),
            n_entities_enriched_nominal_p_lt_0_05=("n_enriched_nominal_p_lt_0_05", lambda s: int(pd.to_numeric(s, errors="coerce").gt(0).sum())),
        ).reset_index()
        tracks = tracks.merge(enrichment_track, on=["category", "entity_group"], how="left")
    for column in ["n_entities_enriched_q_lt_0_05", "n_entities_enriched_nominal_p_lt_0_05"]:
        if column not in tracks.columns:
            tracks[column] = 0
        tracks[column] = pd.to_numeric(tracks[column], errors="coerce").fillna(0).astype(int)
    if not tracks.empty:
        category_order = {category: index for index, category in enumerate(categories)}
        tracks["_category_order"] = tracks["category"].map(category_order).fillna(len(category_order)).astype(int)
        tracks = tracks.loc[:, track_columns + ["_category_order"]].sort_values(
            by=["_category_order", "n_entities_present", "entity_group"],
            ascending=[True, False, True],
            kind="mergesort",
        ).drop(columns=["_category_order"]).reset_index(drop=True)
    return nodes, links, tracks

def plot_category_function_circos(nodes_df, links_df, tracks_df, output_base, max_links=120, title=None):
    ensure_plotting()
    if nodes_df is None or nodes_df.empty or links_df is None or links_df.empty:
        return False
    categories = ordered_methods(
        set(nodes_df["category"].astype(str)).union(
            set(links_df["category_a"].astype(str)).union(set(links_df["category_b"].astype(str)))
        )
    )
    if len(categories) < 2:
        return False
    selected_links = links_df.sort_values(
        by=["rank_score", "n_lineages_shared", "entity_label"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(int(max_links)).copy()
    if selected_links.empty:
        return False

    plt_local = ensure_plotting()
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch, Wedge
    fig, ax = plt_local.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")

    n_categories = len(categories)
    gap = 8.0
    sector_span = (360.0 - n_categories * gap) / n_categories
    category_angles = {}
    for idx, category in enumerate(categories):
        start = 90.0 - idx * (sector_span + gap)
        end = start - sector_span
        mid = np.deg2rad((start + end) / 2.0)
        category_angles[category] = {
            "start": start,
            "end": end,
            "mid_rad": mid,
            "point": np.array([np.cos(mid), np.sin(mid)]) * 0.95,
        }
        wedge = Wedge(
            center=(0, 0),
            r=1.0,
            theta1=end,
            theta2=start,
            width=0.08,
            facecolor="#f2f2f2",
            edgecolor="#333333",
            linewidth=1.0,
        )
        ax.add_patch(wedge)
        label_point = np.array([np.cos(mid), np.sin(mid)]) * 1.13
        ax.text(
            label_point[0],
            label_point[1],
            category,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            rotation=np.rad2deg(mid) - 90,
            rotation_mode="anchor",
        )

    if tracks_df is not None and not tracks_df.empty:
        max_track = max(1.0, float(pd.to_numeric(tracks_df["n_entities_present"], errors="coerce").max()))
        for _, row in tracks_df.iterrows():
            category = str(row.get("category", ""))
            if category not in category_angles:
                continue
            group = str(row.get("entity_group", ""))
            group_rows = tracks_df.loc[tracks_df["category"].astype(str).eq(category)].reset_index(drop=True)
            group_index = int(group_rows.index[group_rows["entity_group"].astype(str).eq(group)][0]) if group in group_rows["entity_group"].astype(str).tolist() else 0
            n_groups = max(1, len(group_rows))
            start = category_angles[category]["start"]
            end = category_angles[category]["end"]
            sub_span = abs(start - end) / n_groups
            theta2 = start - group_index * sub_span
            theta1 = theta2 - sub_span * 0.88
            value = float(pd.to_numeric(pd.Series([row.get("n_entities_present")]), errors="coerce").fillna(0).iat[0])
            color = color_for_group(group)
            wedge = Wedge(
                center=(0, 0),
                r=0.88,
                theta1=theta1,
                theta2=theta2,
                width=0.045 + 0.055 * min(1.0, value / max_track),
                facecolor=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=0.75,
            )
            ax.add_patch(wedge)

    max_support = max(1.0, float(pd.to_numeric(selected_links["n_lineages_shared"], errors="coerce").max()))
    for _, row in selected_links.iterrows():
        category_a = str(row.get("category_a", ""))
        category_b = str(row.get("category_b", ""))
        if category_a not in category_angles or category_b not in category_angles:
            continue
        p0 = category_angles[category_a]["point"]
        p3 = category_angles[category_b]["point"]
        support = float(pd.to_numeric(pd.Series([row.get("n_lineages_shared")]), errors="coerce").fillna(1).iat[0])
        control_scale = 0.18
        verts = [
            tuple(p0),
            tuple(p0 * control_scale),
            tuple(p3 * control_scale),
            tuple(p3),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor="none",
            edgecolor=color_for_group(row.get("entity_group", "")),
            linewidth=0.35 + 3.2 * math.sqrt(support / max_support),
            alpha=0.28,
            zorder=1,
        )
        ax.add_patch(patch)

    ax.text(
        0,
        0,
        f"top {len(selected_links)} links",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_title(title or "Category functional sharing Circos-style summary", fontsize=15, pad=20)
    save_figure(fig, output_base)
    return True

def plot_category_function_hive(nodes_df, links_df, output_base, max_links=160, title=None):
    ensure_plotting()
    if nodes_df is None or nodes_df.empty or links_df is None or links_df.empty:
        return False
    categories = ordered_methods(
        set(nodes_df["category"].astype(str)).union(
            set(links_df["category_a"].astype(str)).union(set(links_df["category_b"].astype(str)))
        )
    )
    if len(categories) < 2:
        return False
    selected_links = links_df.sort_values(
        by=["rank_score", "n_lineages_shared", "entity_label"],
        ascending=[False, False, True],
        kind="mergesort",
    ).head(int(max_links)).copy()
    if selected_links.empty:
        return False
    selected_node_ids = set()
    for row in selected_links.itertuples(index=False):
        selected_node_ids.add(f"{row.category_a}||{row.entity_id}")
        selected_node_ids.add(f"{row.category_b}||{row.entity_id}")
    selected_nodes = nodes_df.loc[nodes_df["node_id"].astype(str).isin(selected_node_ids)].copy()
    if selected_nodes.empty:
        return False

    angle_map = {
        category: (2.0 * np.pi * idx / len(categories)) + (np.pi / 2.0)
        for idx, category in enumerate(categories)
    }
    node_positions = {}
    max_rank = max(1.0, float(pd.to_numeric(selected_nodes["rank_score"], errors="coerce").max()))
    for category in categories:
        subset = selected_nodes.loc[selected_nodes["category"].astype(str).eq(category)].copy()
        subset = subset.sort_values(
            by=["rank_score", "entity_label"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if subset.empty:
            continue
        for idx, row in subset.iterrows():
            radius = 0.23 + 0.72 * ((idx + 1) / (len(subset) + 1))
            angle = angle_map[category]
            position = np.array([np.cos(angle), np.sin(angle)]) * radius
            node_positions[str(row["node_id"])] = position

    plt_local = ensure_plotting()
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch
    fig, ax = plt_local.subplots(figsize=(13, 13))
    ax.set_aspect("equal")
    ax.axis("off")
    for category in categories:
        angle = angle_map[category]
        start = np.array([np.cos(angle), np.sin(angle)]) * 0.18
        end = np.array([np.cos(angle), np.sin(angle)]) * 1.02
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#333333", linewidth=1.2, zorder=2)
        label = np.array([np.cos(angle), np.sin(angle)]) * 1.14
        ax.text(label[0], label[1], category, ha="center", va="center", fontsize=12, fontweight="bold")

    max_support = max(1.0, float(pd.to_numeric(selected_links["n_lineages_shared"], errors="coerce").max()))
    for _, row in selected_links.iterrows():
        node_a = f"{row.get('category_a')}||{row.get('entity_id')}"
        node_b = f"{row.get('category_b')}||{row.get('entity_id')}"
        if node_a not in node_positions or node_b not in node_positions:
            continue
        p0 = node_positions[node_a]
        p3 = node_positions[node_b]
        support = float(pd.to_numeric(pd.Series([row.get("n_lineages_shared")]), errors="coerce").fillna(1).iat[0])
        verts = [
            tuple(p0),
            tuple(p0 * 0.25),
            tuple(p3 * 0.25),
            tuple(p3),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor="none",
            edgecolor=color_for_group(row.get("entity_group", "")),
            linewidth=0.35 + 2.8 * math.sqrt(support / max_support),
            alpha=0.24,
            zorder=1,
        )
        ax.add_patch(patch)

    for _, row in selected_nodes.iterrows():
        position = node_positions.get(str(row["node_id"]))
        if position is None:
            continue
        rank = float(pd.to_numeric(pd.Series([row.get("rank_score")]), errors="coerce").fillna(0).iat[0])
        size = 18 + 90 * math.sqrt(max(0.0, rank) / max_rank)
        ax.scatter(
            [position[0]],
            [position[1]],
            s=size,
            color=color_for_group(row.get("entity_group", "")),
            edgecolor="white",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )
    ax.text(0, 0, f"top {len(selected_links)} links", ha="center", va="center", fontsize=10, color="#444444")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_title(title or "Category functional sharing hive plot", fontsize=15, pad=20)
    save_figure(fig, output_base)
    return True

def write_visual_network_outputs(category_summary_df, enrichment_df, output_dir, max_links=160, title_prefix="", categories=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_df, links_df, tracks_df = build_category_entity_visual_tables(category_summary_df, enrichment_df, categories=categories)
    nodes_out = output_dir / "category_function_nodes.tsv"
    links_out = output_dir / "category_function_links.tsv"
    tracks_out = output_dir / "category_function_tracks.tsv"
    nodes_df.to_csv(nodes_out, sep="\t", index=False)
    links_df.to_csv(links_out, sep="\t", index=False)
    tracks_df.to_csv(tracks_out, sep="\t", index=False)
    wrote_paths = [nodes_out, links_out, tracks_out]
    circos_base = output_dir / "category_function_sharing_circos"
    hive_base = output_dir / "category_function_sharing_hive"
    wrote_circos = plot_category_function_circos(
        nodes_df,
        links_df,
        tracks_df,
        circos_base,
        max_links=max_links,
        title=f"{title_prefix}: category functional sharing" if title_prefix else None,
    )
    wrote_hive = plot_category_function_hive(
        nodes_df,
        links_df,
        hive_base,
        max_links=max_links,
        title=f"{title_prefix}: category functional sharing hive plot" if title_prefix else None,
    )
    if wrote_circos:
        wrote_paths.extend([Path(str(circos_base) + ".png"), Path(str(circos_base) + ".pdf")])
    if wrote_hive:
        wrote_paths.extend([Path(str(hive_base) + ".png"), Path(str(hive_base) + ".pdf")])
    return wrote_paths

def write_lineage_detail_outputs(
    universe_df,
    annotation_presence_df,
    annotation_category_summary_df,
    annotation_enrichment_df,
    candidate_df,
    output_dir,
    max_heatmap_rows=250,
    max_lineage_plots=25,
):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wrote_paths = []
    if universe_df.empty or annotation_presence_df.empty or candidate_df.empty:
        return wrote_paths
    selected = candidate_df.loc[candidate_df["passes_all_categories"].astype(bool)].copy()
    if selected.empty:
        return wrote_paths
    selected = selected.sort_values(
        by=["n_categories", "n_genomes_total", "Species"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    index_rows = []
    for detail_index, row in selected.iterrows():
        lineage_key = str(row["lineage_key"])
        safe_name = f"{int(detail_index) + 1:03d}_{sanitize_label(str(row.get('Species', lineage_key)))}"
        lineage_dir = output_dir / safe_name
        lineage_dir.mkdir(parents=True, exist_ok=True)
        lineage_universe = universe_df.loc[universe_df["lineage_key"].astype(str).eq(lineage_key)].copy()
        lineage_presence = annotation_presence_df.loc[annotation_presence_df["lineage_key"].astype(str).eq(lineage_key)].copy()
        lineage_summary = annotation_category_summary_df.loc[
            annotation_category_summary_df["lineage_key"].astype(str).eq(lineage_key)
        ].copy()
        lineage_enrichment = annotation_enrichment_df.loc[
            annotation_enrichment_df["lineage_key"].astype(str).eq(lineage_key)
        ].copy()
        if lineage_universe.empty or lineage_presence.empty:
            continue
        genome_order = lineage_universe.copy()
        categories = ordered_methods(genome_order["category"].astype(str).drop_duplicates().tolist())
        lineage_review = build_candidate_entity_review_table(
            category_summary_df=lineage_summary,
            enrichment_df=lineage_enrichment,
            pairwise_df=pd.DataFrame(),
            status_df=pd.DataFrame(),
            categories=categories,
        )
        category_order = {category: order for order, category in enumerate(categories)}
        genome_order["_category_order"] = genome_order["category"].map(category_order).fillna(len(category_order)).astype(int)
        genome_columns = genome_order.sort_values(
            by=["_category_order", "category", "source_sample", "genome_id"],
            kind="mergesort",
        )["genome_axis_label"].astype(str).drop_duplicates().tolist()
        matrix = lineage_presence.pivot_table(
            index="entity_label",
            columns="genome_axis_label",
            values="count_value",
            aggfunc=lambda s: 1,
            fill_value=0,
        ).reindex(columns=genome_columns, fill_value=0)
        annotation_meta = (
            lineage_presence.groupby("entity_label", dropna=False)
            .agg(
                n_genomes_present=("genome_axis_label", lambda s: int(pd.Series(s).astype(str).nunique())),
                source_db=("source_db", join_unique_values),
                annotation_category=("annotation_category", join_unique_values),
            )
            .reset_index()
            .rename(columns={"entity_label": "annotation_text"})
        )
        matrix_df = matrix.reset_index().rename(columns={"entity_label": "annotation_text"})
        matrix_df = annotation_meta.merge(matrix_df, on="annotation_text", how="right")
        category_bits = []
        for annotation_text, group in lineage_presence.groupby("entity_label", dropna=False):
            present_categories = ordered_methods(group["category"].astype(str).drop_duplicates().tolist())
            category_bits.append((annotation_text, ";".join(present_categories)))
        category_map = dict(category_bits)
        matrix_df["categories_present"] = matrix_df["annotation_text"].map(category_map).fillna("")
        matrix_df["categories_absent"] = matrix_df["categories_present"].map(
            lambda value: ";".join([category for category in categories if category not in str(value).split(";")])
        )
        matrix_df["category_presence_pattern"] = matrix_df["categories_present"].map(
            lambda value: ";".join(
                f"{category}={'1' if category in str(value).split(';') else '0'}"
                for category in categories
            )
        )
        matrix_df["orf_ids_by_category"] = ""
        ordered_meta = [
            "annotation_text",
            "n_genomes_present",
            "source_db",
            "annotation_category",
            "categories_present",
            "categories_absent",
            "category_presence_pattern",
            "orf_ids_by_category",
        ]
        matrix_df = matrix_df.loc[:, ordered_meta + genome_columns].sort_values(
            by=["n_genomes_present", "annotation_text"],
            ascending=[False, True],
            kind="mergesort",
        )
        matrix_out = lineage_dir / "annotation_genome_presence_matrix.tsv"
        summary_out = lineage_dir / "annotation_category_summary.tsv"
        enrichment_out = lineage_dir / "annotation_category_enrichment.tsv"
        review_out = lineage_dir / "annotation_function_review.tsv"
        lineage_summary.to_csv(summary_out, sep="\t", index=False)
        lineage_enrichment.to_csv(enrichment_out, sep="\t", index=False)
        lineage_review.to_csv(review_out, sep="\t", index=False)
        matrix_df.to_csv(matrix_out, sep="\t", index=False)
        wrote_paths.extend([matrix_out, summary_out, enrichment_out, review_out])
        ordered_base = lineage_dir / "annotation_genome_presence_ordered_heatmap"
        hca_base = lineage_dir / "annotation_genome_presence_hca_heatmap"
        wrote_ordered = False
        wrote_hca = False
        if detail_index < int(max_lineage_plots):
            wrote_ordered = plot_lineage_annotation_genome_heatmap(
                matrix_df,
                ordered_base,
                max_rows=max_heatmap_rows,
                clustered=False,
                title=f"{row.get('Species', lineage_key)} annotation presence by genome",
            )
            wrote_hca = plot_lineage_annotation_genome_heatmap(
                matrix_df,
                hca_base,
                max_rows=max_heatmap_rows,
                clustered=True,
                title=f"{row.get('Species', lineage_key)} annotation presence by genome (HCA rows)",
            )
            if wrote_ordered:
                wrote_paths.extend([Path(str(ordered_base) + ".png"), Path(str(ordered_base) + ".pdf")])
            if wrote_hca:
                wrote_paths.extend([Path(str(hca_base) + ".png"), Path(str(hca_base) + ".pdf")])
        index_rows.append(
            {
                "candidate_rank": int(row.get("candidate_rank", detail_index + 1)),
                "lineage_key": lineage_key,
                "Species": str(row.get("Species", "")),
                "n_genomes_total": int(row.get("n_genomes_total", 0)),
                "categories_present": str(row.get("categories_present", "")),
                "n_annotations": int(matrix_df["annotation_text"].astype(str).nunique()),
                "detail_dir": str(lineage_dir),
                "matrix_path": str(matrix_out),
                "ordered_heatmap_png": str(Path(str(ordered_base) + ".png")) if wrote_ordered else "",
                "hca_heatmap_png": str(Path(str(hca_base) + ".png")) if wrote_hca else "",
            }
        )
    if index_rows:
        index_out = output_dir / "lineage_detail_index.tsv"
        pd.DataFrame(index_rows).to_csv(index_out, sep="\t", index=False)
        wrote_paths.insert(0, index_out)
    return wrote_paths

def run_linked_global_comparative_analysis(
    matched_df,
    annotation_presence_df,
    compare_column,
    output_dir,
    prefix,
    analysis_subdir_name="linked_comparative_analysis",
    max_heatmap_functions=250,
    max_lineage_detail_plots=25,
):
    output_dir = Path(output_dir)
    comparative_dir = output_dir / str(analysis_subdir_name)
    if comparative_dir.exists():
        shutil.rmtree(comparative_dir)
    comparative_dir.mkdir(parents=True, exist_ok=True)
    wrote_paths = []

    universe_df = prepare_global_lineage_universe(matched_df, compare_column)
    candidate_df = build_global_candidate_lineages(matched_df, compare_column)
    candidate_out = comparative_dir / "candidate_lineages.tsv"
    selected_candidate_out = comparative_dir / "candidate_lineages.all_categories.tsv"
    universe_out = comparative_dir / "global_linked_genome_universe.tsv"
    universe_df.to_csv(universe_out, sep="\t", index=False)
    candidate_df.to_csv(candidate_out, sep="\t", index=False)
    selected_candidates = (
        candidate_df.loc[candidate_df["passes_all_categories"].astype(bool)].copy()
        if not candidate_df.empty and "passes_all_categories" in candidate_df.columns
        else pd.DataFrame()
    )
    candidate_categories = (
        ordered_methods(universe_df.loc[
            universe_df["lineage_key"].astype(str).isin(set(selected_candidates["lineage_key"].astype(str)))
        ]["category"].astype(str).drop_duplicates().tolist())
        if not universe_df.empty and not selected_candidates.empty
        else ordered_methods(universe_df["category"].astype(str).drop_duplicates().tolist())
    )
    selected_candidates.to_csv(selected_candidate_out, sep="\t", index=False)
    wrote_paths.extend([universe_out, candidate_out, selected_candidate_out])

    annotation_entity_presence_df = build_annotation_entity_presence(annotation_presence_df)
    feature_entity_presence_df = build_feature_entity_presence(matched_df, compare_column)
    combined_entity_presence_df = pd.concat(
        [
            annotation_entity_presence_df,
            feature_entity_presence_df,
        ],
        ignore_index=True,
    ) if (not annotation_entity_presence_df.empty or not feature_entity_presence_df.empty) else pd.DataFrame()

    for entity_label, entity_df in [
        ("functional_annotations", annotation_entity_presence_df),
        ("marker_reference_features", feature_entity_presence_df),
        ("combined_entities", combined_entity_presence_df),
    ]:
        entity_dir = comparative_dir / entity_label
        entity_dir.mkdir(parents=True, exist_ok=True)
        presence_out = entity_dir / "entity_presence_long.tsv"
        entity_df.to_csv(presence_out, sep="\t", index=False)
        wrote_paths.append(presence_out)
        category_summary_df, enrichment_df, pairwise_df, status_df = build_global_entity_comparative_tables(
            universe_df=universe_df,
            entity_presence_df=entity_df,
            candidate_df=candidate_df,
            require_all_categories=True,
        )
        review_df = build_candidate_entity_review_table(
            category_summary_df=category_summary_df,
            enrichment_df=enrichment_df,
            pairwise_df=pairwise_df,
            status_df=status_df,
            categories=candidate_categories,
        )
        category_summary_out = entity_dir / "category_presence_summary.tsv"
        enrichment_out = entity_dir / "category_enrichment_vs_others.tsv"
        pairwise_out = entity_dir / "category_pairwise_differential.tsv"
        status_out = entity_dir / "shared_unique_status.tsv"
        review_out = entity_dir / "candidate_lineage_function_review.tsv"
        category_summary_df.to_csv(category_summary_out, sep="\t", index=False)
        enrichment_df.to_csv(enrichment_out, sep="\t", index=False)
        pairwise_df.to_csv(pairwise_out, sep="\t", index=False)
        status_df.to_csv(status_out, sep="\t", index=False)
        review_df.to_csv(review_out, sep="\t", index=False)
        wrote_paths.extend([category_summary_out, enrichment_out, pairwise_out, status_out, review_out])
        pairwise_plot_paths = write_category_pairwise_differential_plots(
            pairwise_df=pairwise_df,
            output_dir=entity_dir / "category_pairwise_differential_plots",
            entity_label=entity_label.replace("_", " ").title(),
            min_genomes_per_category=3,
        )
        wrote_paths.extend(pairwise_plot_paths)
        ordered_base = entity_dir / "category_prevalence_ordered_heatmap"
        hca_base = entity_dir / "category_prevalence_hca_heatmap"
        wrote_ordered = plot_entity_prevalence_heatmap(
            category_summary_df,
            enrichment_df,
            ordered_base,
            max_rows=max_heatmap_functions,
            clustered=False,
            categories=candidate_categories,
            title=f"{entity_label.replace('_', ' ').title()}: ordered category prevalence",
        )
        wrote_hca = plot_entity_prevalence_heatmap(
            category_summary_df,
            enrichment_df,
            hca_base,
            max_rows=max_heatmap_functions,
            clustered=True,
            categories=candidate_categories,
            title=f"{entity_label.replace('_', ' ').title()}: HCA category prevalence",
        )
        if wrote_ordered:
            wrote_paths.extend([Path(str(ordered_base) + ".png"), Path(str(ordered_base) + ".pdf")])
        if wrote_hca:
            wrote_paths.extend([Path(str(hca_base) + ".png"), Path(str(hca_base) + ".pdf")])
        network_paths = write_visual_network_outputs(
            category_summary_df=category_summary_df,
            enrichment_df=enrichment_df,
            output_dir=entity_dir / "visual_networks",
            max_links=max(30, min(240, int(max_heatmap_functions))),
            title_prefix=entity_label.replace("_", " ").title(),
            categories=candidate_categories,
        )
        wrote_paths.extend(network_paths)
        if entity_label == "marker_reference_features":
            circular_base = entity_dir / "category_feature_prevalence_circular_summary"
            wrote_circular = plot_circular_feature_summary(
                category_summary_df,
                enrichment_df,
                circular_base,
                max_features=min(64, max(16, int(max_heatmap_functions))),
                categories=candidate_categories,
            )
            if wrote_circular:
                wrote_paths.extend([Path(str(circular_base) + ".png"), Path(str(circular_base) + ".pdf")])
        if entity_label == "functional_annotations":
            lineage_paths = write_lineage_detail_outputs(
                universe_df=universe_df,
                annotation_presence_df=entity_df,
                annotation_category_summary_df=category_summary_df,
                annotation_enrichment_df=enrichment_df,
                candidate_df=candidate_df,
                output_dir=comparative_dir / "candidate_lineage_sets",
                max_heatmap_rows=max_heatmap_functions,
                max_lineage_plots=max_lineage_detail_plots,
            )
            wrote_paths.extend(lineage_paths)

        pooled_dir = entity_dir / "pooled_all_candidate_lineages"
        pooled_dir.mkdir(parents=True, exist_ok=True)
        pooled_category_summary_df, pooled_enrichment_df, pooled_pairwise_df, pooled_status_df = (
            build_pooled_candidate_entity_comparative_tables(
                universe_df=universe_df,
                entity_presence_df=entity_df,
                candidate_df=candidate_df,
            )
        )
        pooled_review_df = build_candidate_entity_review_table(
            category_summary_df=pooled_category_summary_df,
            enrichment_df=pooled_enrichment_df,
            pairwise_df=pooled_pairwise_df,
            status_df=pooled_status_df,
            categories=candidate_categories,
        )
        pooled_category_summary_out = pooled_dir / "category_presence_summary.tsv"
        pooled_enrichment_out = pooled_dir / "category_enrichment_vs_others.tsv"
        pooled_pairwise_out = pooled_dir / "category_pairwise_differential.tsv"
        pooled_status_out = pooled_dir / "shared_unique_status.tsv"
        pooled_review_out = pooled_dir / "candidate_lineage_function_review.tsv"
        pooled_category_summary_df.to_csv(pooled_category_summary_out, sep="\t", index=False)
        pooled_enrichment_df.to_csv(pooled_enrichment_out, sep="\t", index=False)
        pooled_pairwise_df.to_csv(pooled_pairwise_out, sep="\t", index=False)
        pooled_status_df.to_csv(pooled_status_out, sep="\t", index=False)
        pooled_review_df.to_csv(pooled_review_out, sep="\t", index=False)
        wrote_paths.extend(
            [
                pooled_category_summary_out,
                pooled_enrichment_out,
                pooled_pairwise_out,
                pooled_status_out,
                pooled_review_out,
            ]
        )
        pooled_pairwise_plot_paths = write_category_pairwise_differential_plots(
            pairwise_df=pooled_pairwise_df,
            output_dir=pooled_dir / "category_pairwise_differential_plots",
            entity_label=f"{entity_label.replace('_', ' ').title()} pooled across candidate lineages",
            min_genomes_per_category=3,
        )
        wrote_paths.extend(pooled_pairwise_plot_paths)
        pooled_ordered_base = pooled_dir / "category_prevalence_ordered_heatmap"
        pooled_hca_base = pooled_dir / "category_prevalence_hca_heatmap"
        wrote_pooled_ordered = plot_entity_prevalence_heatmap(
            pooled_category_summary_df,
            pooled_enrichment_df,
            pooled_ordered_base,
            max_rows=max_heatmap_functions,
            clustered=False,
            categories=candidate_categories,
            title=f"{entity_label.replace('_', ' ').title()}: pooled candidate-lineage category prevalence",
        )
        wrote_pooled_hca = plot_entity_prevalence_heatmap(
            pooled_category_summary_df,
            pooled_enrichment_df,
            pooled_hca_base,
            max_rows=max_heatmap_functions,
            clustered=True,
            categories=candidate_categories,
            title=f"{entity_label.replace('_', ' ').title()}: pooled candidate-lineage HCA prevalence",
        )
        if wrote_pooled_ordered:
            wrote_paths.extend([Path(str(pooled_ordered_base) + ".png"), Path(str(pooled_ordered_base) + ".pdf")])
        if wrote_pooled_hca:
            wrote_paths.extend([Path(str(pooled_hca_base) + ".png"), Path(str(pooled_hca_base) + ".pdf")])

    manifest_out = comparative_dir / "README_outputs.tsv"
    pd.DataFrame(
        [
            {
                "path": str(path),
                "description": "linked comparative analysis output",
            }
            for path in wrote_paths
        ]
    ).to_csv(manifest_out, sep="\t", index=False)
    wrote_paths.append(manifest_out)
    return [path for path in wrote_paths if path]

def summarize_matched_lineage_function_deltas(long_df):
    columns = [
        "category_pair",
        "category_a",
        "category_b",
        "metric_id",
        "metric_label",
        "n_pairs",
        "n_category_b_higher",
        "n_category_a_higher",
        "n_ties",
        "fraction_category_b_higher",
        "median_delta_b_minus_a",
        "mean_delta_b_minus_a",
        "pvalue_sign",
        "qvalue_bh",
        "significance",
    ]
    if long_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    pvalues = []
    for group_keys, group in long_df.groupby(["category_pair", "category_a", "category_b", "metric_id", "metric_label"], dropna=False):
        category_pair, category_a, category_b, metric_id, metric_label = group_keys
        deltas = pd.to_numeric(group["delta_b_minus_a"], errors="coerce").dropna()
        if deltas.empty:
            continue
        n_b = int((deltas > 0).sum())
        n_a = int((deltas < 0).sum())
        n_tie = int((deltas == 0).sum())
        decisive = n_a + n_b
        pvalue = exact_binomial_test_two_sided(n_b, decisive) if decisive > 0 else float("nan")
        pvalues.append(pvalue)
        rows.append(
            {
                "category_pair": category_pair,
                "category_a": category_a,
                "category_b": category_b,
                "metric_id": metric_id,
                "metric_label": metric_label,
                "n_pairs": int(deltas.size),
                "n_category_b_higher": n_b,
                "n_category_a_higher": n_a,
                "n_ties": n_tie,
                "fraction_category_b_higher": (float(n_b) / float(decisive)) if decisive > 0 else float("nan"),
                "median_delta_b_minus_a": float(deltas.median()),
                "mean_delta_b_minus_a": float(deltas.mean()),
                "pvalue_sign": pvalue,
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return pd.DataFrame(columns=columns)
    summary["qvalue_bh"] = benjamini_hochberg_adjust(summary["pvalue_sign"]).to_numpy(dtype=float)
    summary["significance"] = summary["qvalue_bh"].map(significance_stars)
    return summary.loc[:, columns].sort_values(
        by=["category_pair", "metric_id"],
        kind="mergesort",
    ).reset_index(drop=True)

def plot_matched_lineage_function_summary(summary_df, output_base):
    ensure_plotting()
    if summary_df.empty:
        return False
    metric_order = [
        metric_id
        for metric_id, _label in MATCHED_LINEAGE_FUNCTION_METRICS
        if metric_id in set(summary_df["metric_id"].astype(str))
    ]
    pair_order = ordered_methods([])
    pair_values = summary_df["category_pair"].astype(str).drop_duplicates().tolist()
    if pair_values:
        pair_order = pair_values
    if not metric_order or not pair_order:
        return False

    median_matrix = summary_df.pivot_table(
        index="category_pair",
        columns="metric_id",
        values="median_delta_b_minus_a",
        aggfunc="first",
    ).reindex(index=pair_order, columns=metric_order)
    fraction_matrix = summary_df.pivot_table(
        index="category_pair",
        columns="metric_id",
        values="fraction_category_b_higher",
        aggfunc="first",
    ).reindex(index=pair_order, columns=metric_order)
    annotation_matrix = summary_df.pivot_table(
        index="category_pair",
        columns="metric_id",
        values="significance",
        aggfunc="first",
    ).reindex(index=pair_order, columns=metric_order).fillna("")
    keep_rows = median_matrix.notna().any(axis=1) | fraction_matrix.notna().any(axis=1)
    keep_cols = median_matrix.notna().any(axis=0) | fraction_matrix.notna().any(axis=0)
    median_matrix = median_matrix.loc[keep_rows, keep_cols]
    fraction_matrix = fraction_matrix.loc[keep_rows, keep_cols]
    annotation_matrix = annotation_matrix.loc[keep_rows, keep_cols]
    if median_matrix.empty or fraction_matrix.empty:
        return False
    pair_order = median_matrix.index.astype(str).tolist()
    metric_order = median_matrix.columns.astype(str).tolist()
    metric_labels = {
        metric_id: metric_label
        for metric_id, metric_label in MATCHED_LINEAGE_FUNCTION_METRICS
    }
    x_labels = [metric_labels.get(metric_id, metric_id) for metric_id in metric_order]

    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        1,
        2,
        figsize=(max(15, len(metric_order) * 1.0), max(5.5, len(pair_order) * 0.55)),
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    finite = np.abs(median_matrix.values[np.isfinite(median_matrix.values)])
    vmax = max(1e-9, float(np.nanmax(finite)) if finite.size else 1.0)
    image0 = axes[0].imshow(median_matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    axes[0].set_title("Median delta (category B - category A)")
    axes[0].set_xticks(np.arange(len(metric_order)))
    axes[0].set_xticklabels(x_labels, rotation=90, fontsize=8)
    axes[0].set_yticks(np.arange(len(pair_order)))
    axes[0].set_yticklabels(pair_order, fontsize=8)
    for row_idx in range(len(pair_order)):
        for col_idx in range(len(metric_order)):
            value = median_matrix.iat[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            star = str(annotation_matrix.iat[row_idx, col_idx] or "")
            axes[0].text(col_idx, row_idx, f"{value:.2f}\n{star}", ha="center", va="center", fontsize=7)
    cbar0 = fig.colorbar(image0, ax=axes[0], fraction=0.03, pad=0.02)
    cbar0.set_label("Median delta")

    image1 = axes[1].imshow(fraction_matrix.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Fraction category B higher")
    axes[1].set_xticks(np.arange(len(metric_order)))
    axes[1].set_xticklabels(x_labels, rotation=90, fontsize=8)
    axes[1].set_yticks(np.arange(len(pair_order)))
    axes[1].set_yticklabels(pair_order, fontsize=8)
    for row_idx in range(len(pair_order)):
        for col_idx in range(len(metric_order)):
            value = fraction_matrix.iat[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            star = str(annotation_matrix.iat[row_idx, col_idx] or "")
            color = "white" if float(value) >= 0.55 else "black"
            axes[1].text(col_idx, row_idx, f"{value:.2f}\n{star}", ha="center", va="center", fontsize=7, color=color)
    cbar1 = fig.colorbar(image1, ax=axes[1], fraction=0.03, pad=0.02)
    cbar1.set_label("Fraction")

    fig.suptitle("Matched-lineage functional annotation comparison", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_base)
    return True

def plot_matched_lineage_feature_status_summary(feature_summary_df, output_base, top_n=25):
    ensure_plotting()
    if feature_summary_df is None or feature_summary_df.empty:
        return False
    required = {"category_pair", "feature_id", "n_gained_in_b", "n_lost_in_b", "n_shared_present"}
    if not required.issubset(set(feature_summary_df.columns)):
        return False

    working = feature_summary_df.copy()
    for column in ["n_gained_in_b", "n_lost_in_b", "n_shared_present"]:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    working["activity_score"] = working["n_gained_in_b"] + working["n_lost_in_b"] + working["n_shared_present"]
    top_features = (
        working.groupby("feature_id")["activity_score"]
        .max()
        .sort_values(ascending=False)
        .head(int(top_n))
        .index
        .tolist()
    )
    if not top_features:
        return False
    plot_df = working.loc[working["feature_id"].isin(top_features)].copy()
    pair_order = plot_df["category_pair"].astype(str).drop_duplicates().tolist()
    feature_order = (
        plot_df.groupby("feature_id")["activity_score"]
        .max()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    statuses = []
    for column, title in [
        ("n_gained_in_b", "Gained in category B"),
        ("n_lost_in_b", "Lost in category B"),
        ("n_shared_present", "Shared present"),
    ]:
        if column in plot_df.columns and pd.to_numeric(plot_df[column], errors="coerce").fillna(0).gt(0).any():
            statuses.append((column, title))
    if not statuses:
        return False
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        1,
        len(statuses),
        figsize=(max(14, len(statuses) * 4.6), max(7, len(feature_order) * 0.32)),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (column, title) in zip(axes, statuses):
        matrix = plot_df.pivot_table(
            index="feature_id",
            columns="category_pair",
            values=column,
            aggfunc="max",
            fill_value=0,
        ).reindex(index=feature_order, columns=pair_order, fill_value=0)
        matrix = matrix.loc[matrix.sum(axis=1).gt(0), matrix.sum(axis=0).gt(0)]
        if matrix.empty:
            ax.axis("off")
            continue
        filtered_features = matrix.index.astype(str).tolist()
        filtered_pairs = matrix.columns.astype(str).tolist()
        vmax = max(1.0, float(np.nanmax(matrix.values)) if matrix.size else 1.0)
        image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(filtered_pairs)))
        ax.set_xticklabels(filtered_pairs, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(filtered_features)))
        ax.set_yticklabels(filtered_features, fontsize=7)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(round(float(matrix.iat[row_idx, col_idx])))
                if value == 0:
                    continue
                color = "white" if value >= 0.55 * vmax else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=6, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Genome pairs")
    fig.suptitle("Matched-lineage feature gain/loss/shared status", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True

def plot_matched_lineage_group_contrast_summary(group_summary_df, output_base, top_n=25):
    ensure_plotting()
    if group_summary_df is None or group_summary_df.empty:
        return False
    status_specs = [
        ("n_all_a_only", "All A only"),
        ("n_some_a_only", "Some A only"),
        ("n_all_b_only", "All B only"),
        ("n_some_b_only", "Some B only"),
        ("n_shared_all", "Shared by all"),
        ("n_shared_partial", "Shared partial"),
    ]
    required = {"category_pair", "feature_id", *(column for column, _title in status_specs)}
    if not required.issubset(set(group_summary_df.columns)):
        return False

    working = group_summary_df.copy()
    status_columns = [column for column, _title in status_specs]
    for column in status_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0)
    working["activity_score"] = working[status_columns].sum(axis=1)
    if not working["activity_score"].gt(0).any():
        return False

    top_features = (
        working.groupby("feature_id")["activity_score"]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n))
        .index
        .tolist()
    )
    if not top_features:
        return False
    plot_df = working.loc[working["feature_id"].isin(top_features)].copy()
    pair_order = plot_df["category_pair"].astype(str).drop_duplicates().tolist()
    feature_order = (
        plot_df.groupby("feature_id")["activity_score"]
        .sum()
        .sort_values(ascending=True)
        .index
        .tolist()
    )

    active_statuses = []
    for column, title in status_specs:
        if pd.to_numeric(plot_df[column], errors="coerce").fillna(0).gt(0).any():
            active_statuses.append((column, title))
    if not active_statuses:
        return False

    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        1,
        len(active_statuses),
        figsize=(max(14, len(active_statuses) * 4.2), max(7, len(feature_order) * 0.32)),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()
    any_panel = False
    for ax, (column, title) in zip(axes, active_statuses):
        matrix = plot_df.pivot_table(
            index="feature_id",
            columns="category_pair",
            values=column,
            aggfunc="sum",
            fill_value=0,
        ).reindex(index=feature_order, columns=pair_order, fill_value=0)
        matrix = matrix.loc[matrix.sum(axis=1).gt(0), matrix.sum(axis=0).gt(0)]
        if matrix.empty:
            ax.axis("off")
            continue
        any_panel = True
        filtered_features = matrix.index.astype(str).tolist()
        filtered_pairs = matrix.columns.astype(str).tolist()
        vmax = max(1.0, float(np.nanmax(matrix.values)) if matrix.size else 1.0)
        image = ax.imshow(matrix.values, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(filtered_pairs)))
        ax.set_xticklabels(filtered_pairs, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(filtered_features)))
        ax.set_yticklabels(filtered_features, fontsize=7)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = int(round(float(matrix.iat[row_idx, col_idx])))
                if value == 0:
                    continue
                color = "white" if value >= 0.55 * vmax else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=6, color=color)
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Lineage groups")

    if not any_panel:
        plt_local.close(fig)
        return False
    fig.suptitle("Matched-lineage all-member feature contrast", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True

def select_focus_method_pairs(paired_df):
    if paired_df.empty:
        return pd.DataFrame()
    required = {"category_a", "category_b"}
    if not required.issubset(set(paired_df.columns)):
        return pd.DataFrame()

    pair_counts = (
        paired_df.groupby(["category_a", "category_b"], dropna=False)
        .size()
        .reset_index(name="pair_count")
    )
    if pair_counts.empty:
        return pd.DataFrame()

    rows = []
    for family in ["MAG", "SAG"]:
        family_lower = family.lower()
        family_pairs = pair_counts.loc[
            pair_counts["category_a"].astype(str).str.lower().str.contains(family_lower)
            & pair_counts["category_b"].astype(str).str.lower().str.contains(family_lower)
        ].copy()
        if family_pairs.empty:
            continue
        family_pairs["variant_a"] = family_pairs["category_a"].map(method_variant_flag)
        family_pairs["variant_b"] = family_pairs["category_b"].map(method_variant_flag)
        preferred = family_pairs.loc[
            family_pairs["variant_a"].ne(family_pairs["variant_b"])
        ].copy()
        candidate = preferred if not preferred.empty else family_pairs
        candidate = candidate.sort_values(
            by=["pair_count", "category_a", "category_b"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        if candidate.empty:
            continue
        best = candidate.iloc[0]
        rows.append(
            {
                "focus_family": family,
                "category_a": str(best["category_a"]),
                "category_b": str(best["category_b"]),
                "pair_count": int(best["pair_count"]),
                "selected_variant_contrast": bool(
                    method_variant_flag(best["category_a"]) != method_variant_flag(best["category_b"])
                ),
                "category_pair": f"{best['category_a']} | {best['category_b']}",
            }
        )
    return pd.DataFrame(rows)

def plot_focus_pair_metric_scatter_grid(
    paired_df,
    metric_specs,
    category_a,
    category_b,
    output_base,
    title,
):
    ensure_plotting()
    if paired_df.empty:
        return False

    available_specs = []
    for metric, label in metric_specs:
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        if col_a in paired_df.columns and col_b in paired_df.columns:
            series_a = pd.to_numeric(paired_df[col_a], errors="coerce")
            series_b = pd.to_numeric(paired_df[col_b], errors="coerce")
            if series_a.notna().any() and series_b.notna().any():
                available_specs.append((metric, label))
    if not available_specs:
        return False

    n_cols = min(3, len(available_specs))
    n_rows = int(np.ceil(len(available_specs) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(11, n_cols * 4.2), max(5.5, n_rows * 3.9)),
        squeeze=False,
    )

    for index, (metric, metric_label) in enumerate(available_specs):
        row_index = index // n_cols
        col_index = index % n_cols
        ax = axes[row_index, col_index]
        col_a = f"{metric}_a"
        col_b = f"{metric}_b"
        plot_df = paired_df[[col_a, col_b]].copy()
        plot_df[col_a] = pd.to_numeric(plot_df[col_a], errors="coerce")
        plot_df[col_b] = pd.to_numeric(plot_df[col_b], errors="coerce")
        plot_df = plot_df.dropna(subset=[col_a, col_b])
        if plot_df.empty:
            ax.axis("off")
            continue

        x_values = plot_df[col_a].values
        y_values = plot_df[col_b].values
        min_value = float(min(np.nanmin(x_values), np.nanmin(y_values)))
        max_value = float(max(np.nanmax(x_values), np.nanmax(y_values)))
        span = max(1e-9, max_value - min_value)
        pad = span * 0.05
        x_min = min_value - pad
        x_max = max_value + pad

        ax.scatter(
            x_values,
            y_values,
            s=22,
            color="#4d4d4d",
            edgecolors="black",
            linewidths=0.25,
            alpha=0.85,
        )
        ax.plot([x_min, x_max], [x_min, x_max], linestyle="--", color="black", linewidth=0.9, alpha=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        ax.set_xlabel(category_a)
        ax.set_ylabel(category_b)
        ax.set_title(metric_label)
        ax.grid(color="#e5e5e5", linewidth=0.6, linestyle="-")

        n_points = int(plot_df.shape[0])
        median_delta = float(np.median(x_values - y_values))
        ax.text(
            0.02,
            0.98,
            f"n={n_points}; median delta={median_delta:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.8, "pad": 2.0},
        )

    for index in range(len(available_specs), n_rows * n_cols):
        row_index = index // n_cols
        col_index = index % n_cols
        axes[row_index, col_index].axis("off")

    n_components = int(paired_df["component_id"].nunique()) if "component_id" in paired_df.columns else int(len(paired_df))
    fig.suptitle(f"{title} (shared components={n_components})", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_base)
    return True

def build_focus_genome_axis_delta_table(focus_pair_df, category_a, category_b, metric_specs):
    if focus_pair_df.empty:
        return pd.DataFrame()

    category_a_is_variant = method_variant_flag(category_a)
    category_b_is_variant = method_variant_flag(category_b)
    invert_sign = bool(category_a_is_variant and not category_b_is_variant)
    baseline_category = category_b if invert_sign else category_a
    contrast_category = category_a if invert_sign else category_b

    rows = []
    for row in focus_pair_df.to_dict("records"):
        sample_value = str(row.get("sample", "")).strip()
        component_id = str(row.get("component_id", "")).strip()
        component_label = f"{sample_value} | {component_id}" if sample_value else component_id
        out_row = {
            "sample": sample_value,
            "component_id": component_id,
            "component_label": component_label,
            "baseline_category": str(baseline_category),
            "contrast_category": str(contrast_category),
            "category_a": str(category_a),
            "category_b": str(category_b),
            "orientation": f"{baseline_category} - {contrast_category}",
            "atlas_genome_baseline": str(row.get("atlas_genome_b" if invert_sign else "atlas_genome_a", "")),
            "atlas_genome_contrast": str(row.get("atlas_genome_a" if invert_sign else "atlas_genome_b", "")),
            "metapathways_genome_baseline": str(
                row.get("metapathways_genome_b" if invert_sign else "metapathways_genome_a", "")
            ),
            "metapathways_genome_contrast": str(
                row.get("metapathways_genome_a" if invert_sign else "metapathways_genome_b", "")
            ),
        }
        for metric, _label in metric_specs:
            delta_column = f"{metric}_delta"
            delta_value = pd.to_numeric(pd.Series([row.get(delta_column)]), errors="coerce").iat[0]
            if pd.notna(delta_value) and invert_sign:
                delta_value = -float(delta_value)
            out_row[f"{metric}_delta"] = delta_value
        rows.append(out_row)

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return delta_df
    delta_df = delta_df.sort_values(by=["sample", "component_id"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return delta_df

def plot_focus_genome_axis_delta_heatmap(delta_df, metric_specs, output_base, title, max_rows=250):
    ensure_plotting()
    if delta_df.empty:
        return False

    metric_columns = []
    metric_labels = []
    for metric, label in metric_specs:
        column = f"{metric}_delta"
        if column in delta_df.columns and pd.to_numeric(delta_df[column], errors="coerce").notna().any():
            metric_columns.append(column)
            metric_labels.append(label)
    if not metric_columns:
        return False

    plot_df = delta_df.copy()
    if len(plot_df) > int(max_rows):
        score = pd.Series(0.0, index=plot_df.index)
        for column in metric_columns:
            score = score.add(pd.to_numeric(plot_df[column], errors="coerce").abs().fillna(0.0), fill_value=0.0)
        plot_df = (
            plot_df.assign(_plot_abs_delta_score=score)
            .sort_values(by=["_plot_abs_delta_score", "component_label"], ascending=[False, True], kind="mergesort")
            .head(int(max_rows))
            .sort_values(by=["component_label"], ascending=[True], kind="mergesort")
            .drop(columns=["_plot_abs_delta_score"], errors="ignore")
            .reset_index(drop=True)
        )
        plotted_rows_out = Path(str(output_base) + "_plotted_rows.tsv")
        plot_df.to_csv(plotted_rows_out, sep="\t", index=False)

    matrix = plot_df[metric_columns].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.empty:
        return False
    color_matrix = matrix.copy()
    for column in color_matrix.columns:
        series = pd.to_numeric(color_matrix[column], errors="coerce")
        finite = np.abs(series.values[np.isfinite(series.values)])
        column_max_abs = float(np.nanmax(finite)) if finite.size else 0.0
        if column_max_abs <= 0:
            color_matrix[column] = 0.0
        else:
            color_matrix[column] = series / column_max_abs

    row_labels = plot_df["component_label"].astype(str).tolist()
    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    y_font = 8 if n_rows <= 70 else 6 if n_rows <= 180 else 5
    x_font = 9 if n_cols <= 6 else 8

    vmax = 1.0

    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(10, n_cols * 1.8 + 3), max(6, n_rows * 0.22)),
    )
    image = ax.imshow(color_matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(metric_labels, rotation=90, fontsize=x_font)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=y_font)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Sample | Genome component")
    title_suffix = f" (top {len(plot_df)} of {len(delta_df)} rows)" if len(delta_df) > len(plot_df) else ""
    ax.set_title(f"{title}{title_suffix}")

    if n_rows <= 50 and n_cols <= 8:
        for row_index in range(n_rows):
            for col_index in range(n_cols):
                value = matrix.iat[row_index, col_index]
                if pd.isna(value):
                    continue
                scaled_value = color_matrix.iat[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=("white" if abs(float(scaled_value)) >= 0.6 else "black"),
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Column-scaled delta (baseline - contrast)")
    fig.subplots_adjust(left=0.34, right=0.99, top=0.93, bottom=0.2)
    save_figure(fig, output_base)
    return True

def _row_explicit_id_aliases(row):
    alias_columns = [
        "_atlas_genome_id",
        "Genome_Id",
        "SAG_ID",
        "Bin Id",
        "genome_id",
        "mp_genome_id",
        "mp_genome_label",
        "fasta_path",
        "ani_record_id",
        "ani_fasta_path",
    ]
    aliases = set()
    for column in alias_columns:
        if column not in row:
            continue
        raw_value = str(row.get(column, "")).strip()
        if not raw_value:
            continue
        if raw_value.lower() in {"nan", "none", "null", "na", "n/a"}:
            continue
        if column in {"fasta_path", "ani_fasta_path"}:
            raw_value = Path(raw_value).name
        aliases.update(matching_id_aliases(raw_value))

    cleaned = set()
    for alias in aliases:
        text = str(alias).strip()
        if not text:
            continue
        if text.lower() in {"nan", "none", "null", "na", "n/a"}:
            continue
        if len(text) < 3:
            continue
        cleaned.add(text)
    return cleaned

def _preferred_alias_label(alias_set, fallback_value):
    aliases = sorted(
        list(alias_set),
        key=lambda value: ("." in value, "/" in value, len(value), value),
    )
    if aliases:
        return aliases[0]
    return str(fallback_value).strip()

def _metric_column_specs_for_pairing(frame):
    specs = []
    for metric, label in PAIRED_QC_METRICS:
        if metric in frame.columns:
            specs.append((metric, metric, label))
    for metric, label in PAIRED_FUNCTION_METRICS:
        mp_column = f"mp_{metric}"
        if mp_column in frame.columns:
            specs.append((metric, mp_column, label))
    return specs

def build_focus_explicit_id_pair_tables(source_df, compare_column, category_a, category_b):
    if source_df.empty or compare_column not in source_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    category_a = str(category_a).strip()
    category_b = str(category_b).strip()
    if not category_a or not category_b:
        return pd.DataFrame(), pd.DataFrame()

    category_a_is_variant = method_variant_flag(category_a)
    category_b_is_variant = method_variant_flag(category_b)
    invert_sign = bool(category_a_is_variant and not category_b_is_variant)
    baseline_category = category_b if invert_sign else category_a
    contrast_category = category_a if invert_sign else category_b

    working = source_df.copy()
    working["_sample_key"] = working.apply(pair_sample_value, axis=1)
    working["_category_key"] = (
        working[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working = working.loc[
        working["_sample_key"].ne("")
        & working["_category_key"].isin([baseline_category, contrast_category])
    ].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    working["_mapping_aliases"] = working.apply(_row_explicit_id_aliases, axis=1)
    working = working.loc[working["_mapping_aliases"].map(bool)].copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame()

    metric_specs = _metric_column_specs_for_pairing(working)
    baseline_df = working.loc[working["_category_key"].eq(baseline_category)].copy()
    contrast_df = working.loc[working["_category_key"].eq(contrast_category)].copy()
    if baseline_df.empty or contrast_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    audit_rows = []
    pair_rows = []
    for sample_value in sorted(set(baseline_df["_sample_key"]).union(set(contrast_df["_sample_key"]))):
        baseline_sample = baseline_df.loc[baseline_df["_sample_key"].eq(sample_value)].copy().reset_index(drop=True)
        contrast_sample = contrast_df.loc[contrast_df["_sample_key"].eq(sample_value)].copy().reset_index(drop=True)
        if baseline_sample.empty or contrast_sample.empty:
            continue

        contrast_alias_index = {}
        contrast_records = contrast_sample.to_dict("records")
        for index, record in enumerate(contrast_records):
            for alias in record.get("_mapping_aliases", set()):
                contrast_alias_index.setdefault(alias, set()).add(index)

        proposals = []
        for baseline_index, baseline_record in enumerate(baseline_sample.to_dict("records")):
            baseline_aliases = set(baseline_record.get("_mapping_aliases", set()))
            candidate_indices = set()
            for alias in baseline_aliases:
                candidate_indices.update(contrast_alias_index.get(alias, set()))

            base_id = str(baseline_record.get("_atlas_genome_id", "") or baseline_record.get("mp_genome_id", ""))
            if not candidate_indices:
                audit_rows.append(
                    {
                        "sample": sample_value,
                        "baseline_category": baseline_category,
                        "contrast_category": contrast_category,
                        "baseline_genome": base_id,
                        "status": "unmatched",
                        "candidate_count": 0,
                        "mapping_alias": "",
                    }
                )
                continue

            scored = []
            for contrast_index in sorted(candidate_indices):
                contrast_record = contrast_records[contrast_index]
                overlap = baseline_aliases & set(contrast_record.get("_mapping_aliases", set()))
                if not overlap:
                    continue
                scored.append((contrast_index, len(overlap), sorted(overlap)[0]))

            if not scored:
                audit_rows.append(
                    {
                        "sample": sample_value,
                        "baseline_category": baseline_category,
                        "contrast_category": contrast_category,
                        "baseline_genome": base_id,
                        "status": "unmatched",
                        "candidate_count": 0,
                        "mapping_alias": "",
                    }
                )
                continue

            max_overlap = max(item[1] for item in scored)
            top_scored = [item for item in scored if item[1] == max_overlap]
            if len(top_scored) > 1:
                audit_rows.append(
                    {
                        "sample": sample_value,
                        "baseline_category": baseline_category,
                        "contrast_category": contrast_category,
                        "baseline_genome": base_id,
                        "status": "ambiguous",
                        "candidate_count": int(len(top_scored)),
                        "mapping_alias": "",
                    }
                )
                continue

            selected_index, overlap_count, alias_value = top_scored[0]
            proposals.append(
                {
                    "baseline_index": baseline_index,
                    "contrast_index": selected_index,
                    "overlap_count": int(overlap_count),
                    "mapping_alias": alias_value,
                }
            )

        if not proposals:
            continue

        by_contrast_index = {}
        for proposal in proposals:
            by_contrast_index.setdefault(proposal["contrast_index"], []).append(proposal)

        final_pairs = []
        for contrast_index, grouped in by_contrast_index.items():
            if len(grouped) == 1:
                final_pairs.append(grouped[0])
                continue
            grouped = sorted(grouped, key=lambda item: (-item["overlap_count"], item["baseline_index"]))
            winner = grouped[0]
            final_pairs.append(winner)
            for rejected in grouped[1:]:
                rejected_row = baseline_sample.iloc[int(rejected["baseline_index"])]
                rejected_id = str(rejected_row.get("_atlas_genome_id", "") or rejected_row.get("mp_genome_id", ""))
                audit_rows.append(
                    {
                        "sample": sample_value,
                        "baseline_category": baseline_category,
                        "contrast_category": contrast_category,
                        "baseline_genome": rejected_id,
                        "status": "duplicate_conflict",
                        "candidate_count": int(len(grouped)),
                        "mapping_alias": str(rejected.get("mapping_alias", "")),
                    }
                )

        for pair in final_pairs:
            baseline_record = baseline_sample.iloc[int(pair["baseline_index"])].to_dict()
            contrast_record = contrast_records[int(pair["contrast_index"])]
            mapping_alias = str(pair.get("mapping_alias", "")).strip()
            baseline_aliases = set(baseline_record.get("_mapping_aliases", set()))
            contrast_aliases = set(contrast_record.get("_mapping_aliases", set()))
            alias_for_label = _preferred_alias_label(
                baseline_aliases & contrast_aliases,
                mapping_alias or baseline_record.get("_atlas_genome_id", ""),
            )
            component_label = f"{sample_value} | {alias_for_label}" if sample_value else alias_for_label

            row = {
                "sample": sample_value,
                "component_id": alias_for_label,
                "component_label": component_label,
                "mapping_alias": mapping_alias,
                "overlap_count": int(pair.get("overlap_count", 0)),
                "baseline_category": baseline_category,
                "contrast_category": contrast_category,
                "category_a": baseline_category,
                "category_b": contrast_category,
                "orientation": f"{baseline_category} - {contrast_category}",
                "atlas_genome_baseline": str(
                    baseline_record.get("_atlas_genome_id", "") or baseline_record.get("Genome_Id", "")
                ),
                "atlas_genome_contrast": str(
                    contrast_record.get("_atlas_genome_id", "") or contrast_record.get("Genome_Id", "")
                ),
                "metapathways_genome_baseline": str(baseline_record.get("mp_genome_id", "")),
                "metapathways_genome_contrast": str(contrast_record.get("mp_genome_id", "")),
            }
            for metric_key, metric_column, _metric_label in metric_specs:
                value_baseline = pd.to_numeric(pd.Series([baseline_record.get(metric_column)]), errors="coerce").iat[0]
                value_contrast = pd.to_numeric(pd.Series([contrast_record.get(metric_column)]), errors="coerce").iat[0]
                row[f"{metric_key}_baseline"] = value_baseline
                row[f"{metric_key}_contrast"] = value_contrast
                row[f"{metric_key}_delta"] = (
                    value_baseline - value_contrast
                    if pd.notna(value_baseline) and pd.notna(value_contrast)
                    else np.nan
                )
            pair_rows.append(row)
            audit_rows.append(
                {
                    "sample": sample_value,
                    "baseline_category": baseline_category,
                    "contrast_category": contrast_category,
                    "baseline_genome": row["atlas_genome_baseline"] or row["metapathways_genome_baseline"],
                    "contrast_genome": row["atlas_genome_contrast"] or row["metapathways_genome_contrast"],
                    "status": "matched",
                    "candidate_count": 1,
                    "mapping_alias": mapping_alias,
                    "overlap_count": int(pair.get("overlap_count", 0)),
                }
            )

    audit_df = pd.DataFrame(audit_rows)
    pair_df = pd.DataFrame(pair_rows)
    if not audit_df.empty:
        audit_df = audit_df.sort_values(
            by=["sample", "status", "baseline_genome"],
            ascending=[True, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    if not pair_df.empty:
        pair_df = pair_df.sort_values(
            by=["sample", "component_id"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return audit_df, pair_df

def plot_focus_before_after_lines(pair_df, metric_specs, output_base, title):
    ensure_plotting()
    if pair_df.empty:
        return False

    available = []
    for metric, label in metric_specs:
        baseline_col = f"{metric}_baseline"
        contrast_col = f"{metric}_contrast"
        if baseline_col in pair_df.columns and contrast_col in pair_df.columns:
            baseline_series = pd.to_numeric(pair_df[baseline_col], errors="coerce")
            contrast_series = pd.to_numeric(pair_df[contrast_col], errors="coerce")
            if baseline_series.notna().any() and contrast_series.notna().any():
                available.append((metric, label))
    if not available:
        return False

    n_cols = min(3, len(available))
    n_rows = int(np.ceil(len(available) / float(n_cols)))
    baseline_label = str(pair_df["baseline_category"].iloc[0]) if "baseline_category" in pair_df.columns else "Before"
    contrast_label = str(pair_df["contrast_category"].iloc[0]) if "contrast_category" in pair_df.columns else "After"
    n_pairs = int(pair_df.shape[0])
    alpha = 0.35 if n_pairs <= 200 else 0.2

    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(11, n_cols * 4.2), max(5.5, n_rows * 3.8)),
        squeeze=False,
    )
    for index, (metric, metric_label) in enumerate(available):
        row_index = index // n_cols
        col_index = index % n_cols
        ax = axes[row_index, col_index]
        baseline_col = f"{metric}_baseline"
        contrast_col = f"{metric}_contrast"
        plotting = pair_df[[baseline_col, contrast_col]].copy()
        plotting[baseline_col] = pd.to_numeric(plotting[baseline_col], errors="coerce")
        plotting[contrast_col] = pd.to_numeric(plotting[contrast_col], errors="coerce")
        plotting = plotting.dropna(subset=[baseline_col, contrast_col])
        if plotting.empty:
            ax.axis("off")
            continue

        x_values = np.array([0.0, 1.0], dtype=float)
        for values in plotting[[baseline_col, contrast_col]].to_numpy():
            ax.plot(
                x_values,
                values,
                color="#7a7a7a",
                linewidth=0.7,
                alpha=alpha,
                zorder=1,
            )
        ax.scatter(
            np.zeros(len(plotting)),
            plotting[baseline_col].values,
            s=10,
            color="#4d4d4d",
            alpha=0.7,
            zorder=2,
        )
        ax.scatter(
            np.ones(len(plotting)),
            plotting[contrast_col].values,
            s=10,
            color="#1f1f1f",
            alpha=0.7,
            zorder=2,
        )
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels([baseline_label, contrast_label], rotation=20)
        ax.set_title(metric_label)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)

        median_delta = float(np.median(plotting[baseline_col].values - plotting[contrast_col].values))
        ax.text(
            0.02,
            0.98,
            f"n={int(plotting.shape[0])}; median delta={median_delta:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.8, "pad": 2.0},
        )

    for index in range(len(available), n_rows * n_cols):
        row_index = index // n_cols
        col_index = index % n_cols
        axes[row_index, col_index].axis("off")

    fig.suptitle(f"{title} (mapped pairs={n_pairs})", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_base)
    return True

def summarize_idmap_sample_method_mean_deltas(pair_df, metric_specs, focus_family):
    if pair_df.empty:
        return pd.DataFrame()
    required = {"sample", "baseline_category", "contrast_category", "orientation"}
    if not required.issubset(set(pair_df.columns)):
        return pd.DataFrame()

    rows = []
    group_columns = ["sample", "baseline_category", "contrast_category", "orientation"]
    for group_values, group in pair_df.groupby(group_columns, dropna=False):
        sample_value, baseline_category, contrast_category, orientation = group_values
        for metric, metric_label in metric_specs:
            delta_column = f"{metric}_delta"
            if delta_column not in group.columns:
                continue
            series = pd.to_numeric(group[delta_column], errors="coerce").dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "focus_family": str(focus_family),
                    "sample": str(sample_value),
                    "baseline_category": str(baseline_category),
                    "contrast_category": str(contrast_category),
                    "orientation": str(orientation),
                    "metric": str(metric),
                    "metric_label": str(metric_label),
                    "n_pairs": int(series.size),
                    "mean_delta": float(series.mean()),
                    "median_delta": float(series.median()),
                }
            )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    summary_df["sample_method"] = summary_df.apply(
        lambda row: f"{row['sample']} | {row['orientation']}",
        axis=1,
    )
    return summary_df

def plot_idmap_sample_method_delta_heatmap(summary_df, metric_specs, output_base, title):
    ensure_plotting()
    if summary_df.empty:
        return False
    required = {"sample_method", "metric", "metric_label", "mean_delta"}
    if not required.issubset(set(summary_df.columns)):
        return False

    metric_order = []
    metric_label_lookup = {}
    for metric, label in metric_specs:
        metric_order.append(metric)
        metric_label_lookup[metric] = label
    working = summary_df.copy()
    working["metric"] = working["metric"].astype(str)
    working["sample_method"] = working["sample_method"].astype(str)
    working["mean_delta"] = pd.to_numeric(working["mean_delta"], errors="coerce")
    working = working.dropna(subset=["mean_delta"])
    if working.empty:
        return False

    row_order = (
        working[["sample_method", "focus_family", "sample", "orientation"]]
        .drop_duplicates()
        .sort_values(by=["focus_family", "sample", "orientation"], ascending=[True, True, True], kind="mergesort")
        ["sample_method"]
        .tolist()
    )
    matrix = (
        working.pivot_table(
            index="sample_method",
            columns="metric",
            values="mean_delta",
            aggfunc="mean",
        )
        .reindex(index=row_order)
        .reindex(columns=metric_order)
    )
    matrix = matrix.rename(columns=metric_label_lookup)
    if matrix.empty:
        return False

    color_matrix = matrix.copy()
    for column in color_matrix.columns:
        series = pd.to_numeric(color_matrix[column], errors="coerce")
        finite = np.abs(series.values[np.isfinite(series.values)])
        max_abs = float(np.nanmax(finite)) if finite.size else 0.0
        if max_abs <= 0:
            color_matrix[column] = 0.0
        else:
            color_matrix[column] = series / max_abs

    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    y_font = 8 if n_rows <= 60 else 7 if n_rows <= 150 else 6
    x_font = 9 if n_cols <= 7 else 8
    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(11, n_cols * 2.0 + 3.5), max(7, n_rows * 0.32)),
    )
    image = ax.imshow(color_matrix.values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90, fontsize=x_font)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index.astype(str).tolist(), fontsize=y_font)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Sample | Method delta")
    ax.set_title(title)

    if n_rows <= 60 and n_cols <= 8:
        for row_index in range(n_rows):
            for col_index in range(n_cols):
                raw_value = matrix.iat[row_index, col_index]
                if pd.isna(raw_value):
                    continue
                scaled_value = color_matrix.iat[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    f"{float(raw_value):.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=("white" if abs(float(scaled_value)) >= 0.6 else "black"),
                )

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Column-scaled mean delta")
    fig.subplots_adjust(left=0.34, right=0.99, top=0.95, bottom=0.2)
    save_figure(fig, output_base)
    return True

def build_method_overlap_entities(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame(), []
    required = {"component_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame(), []

    working = matched_df.copy()
    working["_sample_key"] = working.apply(pair_sample_value, axis=1)
    working["_category_key"] = (
        working[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working["component_id"] = working["component_id"].astype(str).str.strip()
    working = working.loc[
        working["_sample_key"].ne("") & working["_category_key"].ne("") & working["component_id"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame(), []

    method_counts = working["_category_key"].value_counts(dropna=False).to_dict()
    methods = ordered_methods(working["_category_key"].unique().tolist(), counts=method_counts)
    rows = []
    grouped = (
        working.groupby(["_sample_key", "component_id"], dropna=False)["_category_key"]
        .agg(lambda values: ordered_methods(set(values)))
        .reset_index()
    )
    for row in grouped.to_dict("records"):
        sample_value = str(row.get("_sample_key", ""))
        component_id = str(row.get("component_id", ""))
        present_methods = list(row.get("_category_key", []))
        present_set = set(present_methods)
        out_row = {
            "sample": sample_value,
            "component_id": component_id,
            "entity_id": f"{sample_value}|{component_id}",
            "present_methods": ";".join(present_methods),
            "present_method_count": len(present_methods),
        }
        for method in methods:
            out_row[f"is_{sanitize_label(method)}"] = int(method in present_set)
        rows.append(out_row)

    return pd.DataFrame(rows), methods

def summarize_method_overlap_intersections(entity_df, methods):
    if entity_df.empty or not methods:
        return pd.DataFrame(), {}
    method_columns = [f"is_{sanitize_label(method)}" for method in methods]
    for column in method_columns:
        if column not in entity_df.columns:
            return pd.DataFrame(), {}

    method_sizes = {}
    for method, column in zip(methods, method_columns):
        method_sizes[method] = int(pd.to_numeric(entity_df[column], errors="coerce").fillna(0).sum())

    counts = {}
    for row in entity_df.to_dict("records"):
        present_methods = tuple(
            method for method, column in zip(methods, method_columns)
            if int(pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce").fillna(0).iat[0]) > 0
        )
        if not present_methods:
            continue
        counts[present_methods] = counts.get(present_methods, 0) + 1

    rows = []
    for present_methods, count in counts.items():
        row = {
            "intersection_methods": ";".join(present_methods),
            "intersection_size": int(len(present_methods)),
            "entity_count": int(count),
        }
        for method in methods:
            row[f"is_{sanitize_label(method)}"] = int(method in present_methods)
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df, method_sizes
    summary_df = summary_df.sort_values(
        by=["entity_count", "intersection_size", "intersection_methods"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return summary_df, method_sizes

def plot_method_overlap_upset(intersections_df, methods, method_sizes, output_base, title):
    ensure_plotting()
    if intersections_df.empty or not methods:
        return False

    plot_df = intersections_df.copy().reset_index(drop=True)
    x_positions = np.arange(len(plot_df))
    method_columns = [f"is_{sanitize_label(method)}" for method in methods]
    y_positions = np.arange(len(methods))

    plt_local = ensure_plotting()
    fig = plt_local.figure(figsize=(max(12, len(plot_df) * 0.5), max(7.5, len(methods) * 0.65 + 3.5)))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.8, 5.5],
        height_ratios=[3.2, 2.2],
        hspace=0.06,
        wspace=0.08,
    )
    ax_blank = fig.add_subplot(grid[0, 0])
    ax_top = fig.add_subplot(grid[0, 1])
    ax_set = fig.add_subplot(grid[1, 0])
    ax_matrix = fig.add_subplot(grid[1, 1], sharex=ax_top)
    ax_blank.axis("off")

    counts = pd.to_numeric(plot_df["entity_count"], errors="coerce").fillna(0.0)
    ax_top.bar(x_positions, counts, color="#7f7f7f", edgecolor="black", linewidth=0.7)
    for x, value in zip(x_positions, counts.tolist()):
        ax_top.text(x, float(value), f"{int(round(float(value)))}", ha="center", va="bottom", fontsize=7)
    ax_top.set_ylabel("Count")
    ax_top.set_title(title)
    ax_top.set_xticks([])
    ax_top.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_top.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.6)

    max_count = max(1.0, float(counts.max()))
    ax_top.set_ylim(0, max_count * 1.15)

    for index, row in plot_df.iterrows():
        included = []
        for method_index, column in enumerate(method_columns):
            is_included = int(pd.to_numeric(pd.Series([row.get(column, 0)]), errors="coerce").fillna(0).iat[0]) > 0
            color = "black" if is_included else "#c9c9c9"
            ax_matrix.scatter(index, method_index, s=30, color=color, zorder=3)
            if is_included:
                included.append(method_index)
        if len(included) >= 2:
            ax_matrix.plot([index, index], [min(included), max(included)], color="black", linewidth=1.1, zorder=2)

    ax_matrix.set_yticks(y_positions)
    ax_matrix.set_yticklabels(methods)
    ax_matrix.yaxis.tick_right()
    ax_matrix.tick_params(axis="y", which="major", labelleft=False, labelright=True, pad=6)
    ax_matrix.set_ylim(-0.5, len(methods) - 0.5)
    ax_matrix.invert_yaxis()
    ax_matrix.set_xlabel("")
    ax_matrix.set_xticks([])
    ax_matrix.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax_matrix.grid(axis="x", color="#efefef", linestyle="-", linewidth=0.5)

    set_sizes = [method_sizes.get(method, 0) for method in methods]
    ax_set.barh(y_positions, set_sizes, color="#9c9c9c", edgecolor="black", linewidth=0.7)
    for y, value in zip(y_positions, set_sizes):
        ax_set.text(float(value), y, f"{int(value)}", va="center", ha="left", fontsize=7)
    ax_set.set_yticks(y_positions)
    ax_set.set_yticklabels([])
    ax_set.invert_yaxis()
    ax_set.set_xlabel("Set size")
    ax_set.grid(axis="x", color="#efefef", linestyle="-", linewidth=0.5)

    fig.subplots_adjust(left=0.12, right=0.995, top=0.95, bottom=0.14)
    save_figure(fig, output_base)
    return True

def plot_species_method_presence_heatmap(entity_df, methods, output_base, title, entity_label="Species"):
    ensure_plotting()
    if entity_df.empty or not methods:
        return False

    method_columns = [f"is_{sanitize_label(method)}" for method in methods]
    if not all(column in entity_df.columns for column in method_columns):
        return False

    working = entity_df.copy()
    for column in method_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0).astype(int).clip(lower=0, upper=1)
    if "sample" in working.columns:
        working["sample"] = working["sample"].astype(str).str.strip()
    else:
        working["sample"] = ""
    if "component_id" in working.columns:
        working["component_id"] = working["component_id"].astype(str).str.strip()
    else:
        working["component_id"] = working.get("entity_id", "").astype(str)
    working["species_row_label"] = working.apply(
        lambda row: f"{row['sample']} | {row['component_id']}" if str(row.get("sample", "")).strip() else str(row.get("component_id", "")),
        axis=1,
    )
    working = working.sort_values(
        by=["present_method_count", "sample", "component_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    matrix = working[method_columns].copy()
    matrix.columns = methods
    labels = working["species_row_label"].astype(str).tolist()
    if matrix.empty:
        return False

    n_rows = matrix.shape[0]
    n_cols = matrix.shape[1]
    y_font = 8 if n_rows <= 50 else 7 if n_rows <= 120 else 6
    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(10, n_cols * 1.35 + 1.5), max(7, n_rows * 0.32)),
    )
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels, fontsize=y_font)
    ax.set_xlabel("Method")
    ax.set_ylabel(f"Sample | {entity_label}")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Presence (1) / Absence (0)")
    fig.subplots_adjust(left=0.42, right=0.99, top=0.95, bottom=0.12)
    save_figure(fig, output_base)
    return True

def plot_taxonomy_rank_method_heatmaps(
    genome_df,
    output_base_prefix,
    category_column="category",
    genome_id_column="mp_genome_id",
    ranks=None,
):
    ensure_plotting()
    if genome_df.empty or category_column not in genome_df.columns:
        return []
    if genome_id_column not in genome_df.columns:
        genome_id_column = "genome_id" if "genome_id" in genome_df.columns else None
    if genome_id_column is None:
        return []

    if ranks is None:
        ranks = ["Phylum", "Class", "Order", "Family", "Genus", "Species"]

    wrote_paths = []
    category_values = genome_df[category_column].astype(str).str.strip()
    category_counts = category_values.loc[category_values.ne("")].value_counts(dropna=False).to_dict()
    categories = ordered_methods(
        category_values.loc[category_values.ne("")].unique().tolist(),
        counts=category_counts,
    )
    if not categories:
        return wrote_paths

    for rank in ranks:
        if rank not in genome_df.columns:
            continue
        working = genome_df[[rank, category_column, genome_id_column]].copy()
        working[rank] = working[rank].astype(str).str.strip()
        working[category_column] = working[category_column].astype(str).str.strip()
        working[genome_id_column] = working[genome_id_column].astype(str).str.strip()
        working = working.loc[
            working[rank].ne("")
            & working[category_column].ne("")
            & working[genome_id_column].ne("")
            & working[rank].str.lower().ne("nan")
            & working[rank].str.lower().ne("none")
        ].copy()
        if working.empty:
            continue

        matrix = (
            working.groupby([rank, category_column])[genome_id_column]
            .nunique()
            .reset_index(name="n_genomes")
            .pivot_table(
                index=rank,
                columns=category_column,
                values="n_genomes",
                fill_value=0,
                aggfunc="sum",
            )
            .reindex(columns=categories, fill_value=0)
        )
        if matrix.empty:
            continue
        row_order = matrix.sum(axis=1).sort_values(ascending=False).index.tolist()
        matrix = matrix.reindex(index=row_order)

        matrix_out = Path(f"{output_base_prefix}_{sanitize_label(rank)}_count_matrix.tsv")
        matrix.reset_index().to_csv(matrix_out, sep="\t", index=False)
        wrote_paths.append(matrix_out)

        n_rows = matrix.shape[0]
        n_cols = matrix.shape[1]
        y_font = 8 if n_rows <= 70 else 7 if n_rows <= 180 else 6
        plt_local = ensure_plotting()
        fig, ax = plt_local.subplots(
            figsize=(max(9, n_cols * 1.35 + 2.0), max(6, n_rows * 0.32)),
        )
        vmax = max(1.0, float(np.nanmax(matrix.values)))
        image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(matrix.index.astype(str).tolist(), fontsize=y_font)
        ax.set_xlabel("Method")
        ax.set_ylabel(rank)
        ax.set_title(f"{rank} genome counts by method")
        for row_index in range(n_rows):
            for col_index in range(n_cols):
                value = int(round(float(matrix.iat[row_index, col_index])))
                ax.text(
                    col_index,
                    row_index,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=heatmap_text_color(value, vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Genome count")
        fig.subplots_adjust(left=0.42, right=0.99, top=0.95, bottom=0.14)
        plot_base = Path(f"{output_base_prefix}_{sanitize_label(rank)}_count_heatmap")
        save_figure(fig, plot_base)
        wrote_paths.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])
    return wrote_paths

def select_complete_shared_representatives(matched_df, compare_column):
    if matched_df.empty or compare_column not in matched_df.columns:
        return pd.DataFrame(), []
    required = {"component_id", "mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame(), []

    working = matched_df.copy()
    working = working.dropna(subset=["component_id", "mp_genome_id"])
    working["_pair_sample"] = working.apply(pair_sample_value, axis=1)
    working = working.loc[working["_pair_sample"].astype(str).str.strip().ne("")].copy()
    working[compare_column] = working[compare_column].astype(str).str.strip()
    working = working.loc[working[compare_column].ne("")].copy()
    category_counts = working[compare_column].astype(str).value_counts(dropna=False).to_dict()
    categories = ordered_methods(
        working[compare_column].dropna().astype(str).unique().tolist(),
        counts=category_counts,
    )
    if not categories:
        return pd.DataFrame(), []

    expected_categories = len(categories)
    component_category_counts = (
        working.groupby(["_pair_sample", "component_id"])[compare_column]
        .nunique()
    )
    full_keys = component_category_counts.loc[component_category_counts.eq(expected_categories)].index.tolist()
    if not full_keys:
        return pd.DataFrame(), categories

    full_keys_set = set(full_keys)
    full_df = working.loc[
        working.apply(lambda row: (row["_pair_sample"], row["component_id"]) in full_keys_set, axis=1)
    ].copy()
    rank_columns = []
    ascending = []
    for column in [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "mp_informative_annotation_fraction",
        "mp_informative_annotation_orfs",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
        "mp_total_pathways",
        "qscore",
    ]:
        if column in full_df.columns:
            rank_columns.append(column)
            ascending.append(False)
    if "mp_genome_id" in full_df.columns:
        rank_columns.append("mp_genome_id")
        ascending.append(True)
    if rank_columns:
        full_df = full_df.sort_values(by=rank_columns, ascending=ascending, kind="mergesort")
    selected = (
        full_df.groupby(["_pair_sample", "component_id", compare_column], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["sample"] = selected["_pair_sample"].astype(str)
    selected = selected.drop(columns=["_pair_sample"])
    return selected, categories

def _metric_panel_matrix(frame, category_column, metric_specs):
    available = [(column, label) for column, label in metric_specs if column in frame.columns]
    if not available:
        return pd.DataFrame()
    matrix_rows = []
    category_values = ordered_methods(frame[category_column].astype(str).unique().tolist())
    for category in category_values:
        subset = frame.loc[frame[category_column].astype(str).eq(category)]
        row = {"category": category}
        for column, label in available:
            series = pd.to_numeric(subset[column], errors="coerce").dropna()
            row[label] = float(series.median()) if not series.empty else np.nan
        matrix_rows.append(row)
    matrix = pd.DataFrame(matrix_rows).set_index("category")
    return matrix

def plot_complete_shared_summary_panel(complete_df, output_base, category_column="category"):
    ensure_plotting()
    if complete_df.empty or category_column not in complete_df.columns:
        return False

    quality_specs = [
        ("qscore", "Qscore"),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
        ("mimag_quality_index", "MIMAG index"),
    ]
    function_specs = [
        ("mp_informative_annotation_fraction", "Informative fraction"),
        ("mp_total_pathways", "Inferred pathways"),
        ("mp_marker_supported_orfs", "Marker-supported ORFs"),
        ("mp_reference_mode_supported_accessions", "Reference-supported accessions"),
    ]
    quality_matrix = _metric_panel_matrix(complete_df, category_column, quality_specs)
    function_matrix = _metric_panel_matrix(complete_df, category_column, function_specs)
    if quality_matrix.empty and function_matrix.empty:
        return False

    n_components = int(complete_df["component_id"].nunique()) if "component_id" in complete_df.columns else int(len(complete_df))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(2, 1, figsize=(max(10, (quality_matrix.shape[1] if not quality_matrix.empty else 4) * 1.4), 10), squeeze=False)
    axes = axes.ravel()
    panels = [
        ("Genome atlas quality summary (median by method)", quality_matrix),
        ("MetaPathways functional summary (median by method)", function_matrix),
    ]
    for ax, (title, matrix) in zip(axes, panels):
        if matrix.empty:
            ax.axis("off")
            continue
        vmax = float(np.nanmax(matrix.values)) if np.isfinite(matrix.values).any() else 0.0
        if vmax <= 0:
            vmax = 1.0
        image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.astype(str).tolist())
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix.iat[row_index, col_index]
                if pd.isna(value):
                    continue
                text = f"{float(value):.2f}" if float(value) < 10 else f"{float(value):.1f}"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=heatmap_text_color(float(value), vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Median value")
    fig.suptitle(
        f"Shared-across-all-methods representative panel (components={n_components})",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True

def build_pair_elemental_play_by_play(matched_df, compare_column):
    if matched_df.empty or compare_column not in matched_df.columns:
        return pd.DataFrame()
    if "component_id" not in matched_df.columns:
        return pd.DataFrame()

    rows = []
    for component_id, group in matched_df.groupby("component_id", dropna=False):
        component_rows = group.dropna(subset=["mp_genome_id"]).copy()
        if component_rows.empty:
            continue
        records = component_rows.to_dict("records")
        for left_row, right_row in itertools.combinations(records, 2):
            sample_left = pair_sample_value(left_row)
            sample_right = pair_sample_value(right_row)
            if sample_left and sample_right and sample_left != sample_right:
                continue
            pair_sample = sample_left or sample_right
            if not pair_sample:
                continue
            category_left = str(left_row.get(compare_column, ""))
            category_right = str(right_row.get(compare_column, ""))
            if not category_left or not category_right or category_left == category_right:
                continue
            if method_sort_key(category_left) <= method_sort_key(category_right):
                row_a, row_b = left_row, right_row
            else:
                row_a, row_b = right_row, left_row

            category_pair = f"{row_a.get(compare_column, '')} | {row_b.get(compare_column, '')}"
            for mode_id in ELEMENTAL_MODE_ORDER:
                marker_col = f"mp_marker_{mode_id}_gene_count"
                ref_col = f"mp_reference_mode_{mode_id}_accession_count"
                marker_a = pd.to_numeric(pd.Series([row_a.get(marker_col)]), errors="coerce").iat[0]
                marker_b = pd.to_numeric(pd.Series([row_b.get(marker_col)]), errors="coerce").iat[0]
                ref_a = pd.to_numeric(pd.Series([row_a.get(ref_col)]), errors="coerce").iat[0]
                ref_b = pd.to_numeric(pd.Series([row_b.get(ref_col)]), errors="coerce").iat[0]
                rows.append(
                    {
                        "component_id": component_id,
                        "sample": pair_sample,
                        "sample_a": sample_left,
                        "sample_b": sample_right,
                        "category_a": str(row_a.get(compare_column, "")),
                        "category_b": str(row_b.get(compare_column, "")),
                        "category_pair": category_pair,
                        "metapathways_genome_a": str(row_a.get("mp_genome_id", "")),
                        "metapathways_genome_b": str(row_b.get("mp_genome_id", "")),
                        "mode_id": mode_id,
                        "mode_label": ELEMENTAL_MODE_LABELS.get(mode_id, mode_id),
                        "marker_count_a": marker_a,
                        "marker_count_b": marker_b,
                        "marker_delta": marker_a - marker_b if pd.notna(marker_a) and pd.notna(marker_b) else np.nan,
                        "reference_accession_count_a": ref_a,
                        "reference_accession_count_b": ref_b,
                        "reference_delta": ref_a - ref_b if pd.notna(ref_a) and pd.notna(ref_b) else np.nan,
                    }
                )
    return pd.DataFrame(rows)

def summarize_pair_elemental_play_by_play(play_df):
    if play_df.empty:
        return pd.DataFrame()
    rows = []
    group_columns = ["category_pair", "mode_id", "mode_label"]
    include_sample = "sample" in play_df.columns
    if include_sample:
        group_columns = ["sample"] + group_columns
    for group_values, group in play_df.groupby(group_columns, dropna=False):
        if include_sample:
            sample_value, category_pair, mode_id, mode_label = group_values
        else:
            category_pair, mode_id, mode_label = group_values
            sample_value = ""
        marker_series = pd.to_numeric(group["marker_delta"], errors="coerce").dropna()
        ref_series = pd.to_numeric(group["reference_delta"], errors="coerce").dropna()
        rows.append(
            {
                "sample": sample_value,
                "category_pair": category_pair,
                "mode_id": mode_id,
                "mode_label": mode_label,
                "n_pairs": int(len(group)),
                "marker_median_delta": float(marker_series.median()) if not marker_series.empty else np.nan,
                "marker_mean_delta": float(marker_series.mean()) if not marker_series.empty else np.nan,
                "marker_positive_fraction": float((marker_series > 0).mean()) if not marker_series.empty else np.nan,
                "reference_median_delta": float(ref_series.median()) if not ref_series.empty else np.nan,
                "reference_mean_delta": float(ref_series.mean()) if not ref_series.empty else np.nan,
                "reference_positive_fraction": float((ref_series > 0).mean()) if not ref_series.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)

def plot_pair_elemental_play_by_play(play_df, output_base):
    ensure_plotting()
    if play_df.empty:
        return False
    summary_df = summarize_pair_elemental_play_by_play(play_df)
    if summary_df.empty:
        return False

    mode_order = [ELEMENTAL_MODE_LABELS.get(mode_id, mode_id) for mode_id in ELEMENTAL_MODE_ORDER]
    pair_order = (
        play_df.groupby("category_pair")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    marker_matrix = (
        summary_df.pivot_table(
            index="category_pair",
            columns="mode_label",
            values="marker_median_delta",
            aggfunc="mean",
        )
        .reindex(index=pair_order)
        .reindex(columns=mode_order, fill_value=np.nan)
    )
    ref_matrix = (
        summary_df.pivot_table(
            index="category_pair",
            columns="mode_label",
            values="reference_median_delta",
            aggfunc="mean",
        )
        .reindex(index=pair_order)
        .reindex(columns=mode_order, fill_value=np.nan)
    )
    if marker_matrix.empty and ref_matrix.empty:
        return False

    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(2, 1, figsize=(max(12, len(mode_order) * 0.8), max(8, len(pair_order) * 0.5 + 5)), squeeze=False)
    axes = axes.ravel()
    panels = [
        ("Elemental marker gene deltas (median; row - column)", marker_matrix),
        ("Reference GO-accession deltas (median; row - column)", ref_matrix),
    ]
    for ax, (title, matrix) in zip(axes, panels):
        if matrix.empty:
            ax.axis("off")
            continue
        finite = np.abs(matrix.values[np.isfinite(matrix.values)])
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        vmax = max(1e-9, vmax)
        image = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90, fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.astype(str).tolist())
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix.iat[row_index, col_index]
                if pd.isna(value):
                    continue
                ax.text(col_index, row_index, f"{float(value):.2f}", ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Median delta")
    fig.suptitle("Method-paired elemental play-by-play panel", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True


def run_atlas_linked_comparisons(
    args,
    output_dir,
    combined_genome_df,
    combined_annotation_audit_df=None,
    prefix_override=None,
):
    if combined_annotation_audit_df is None:
        combined_annotation_audit_df = pd.DataFrame()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    linkage_mode = str(getattr(args, "atlas_linkage_mode", "species_all")).strip().lower()
    if linkage_mode == "shared_best":
        shared_best_path, annotated_path, shared_best_df, annotated_df = load_atlas_inputs(
            args,
            require_shared=True,
            require_annotated=False,
        )
        atlas_df = prepare_atlas_shared_best(shared_best_df, annotated_df, args)
    else:
        shared_best_path, annotated_path, shared_best_df, annotated_df = load_atlas_inputs(
            args,
            require_shared=False,
            require_annotated=True,
        )
        atlas_df, _atlas_source = prepare_atlas_species_source(shared_best_df, annotated_df, args)

    mp_df, exact_map, alias_map = build_metapathways_lookup(combined_genome_df)
    atlas_match_audit_df, matched_df = match_atlas_genomes_to_metapathways(
        atlas_df=atlas_df,
        mp_df=mp_df,
        exact_map=exact_map,
        alias_map=alias_map,
        args=args,
    )
    if args.min_mimag_tier:
        tier_order = {"low": 0, "medium": 1, "high": 2}
        min_value = tier_order[str(args.min_mimag_tier).strip().lower()]
        if "mimag_tier" not in matched_df.columns:
            raise ValueError(
                "--min-mimag-tier was provided, but 'mimag_tier' is missing from atlas-linked matched genomes."
            )
        matched_df["_mimag_tier_norm"] = matched_df["mimag_tier"].astype(str).str.lower().str.strip()
        matched_df["_mimag_tier_value"] = matched_df["_mimag_tier_norm"].map(tier_order)
        matched_df = matched_df.loc[matched_df["_mimag_tier_value"].fillna(-1).ge(min_value)].copy()
        if matched_df.empty:
            raise ValueError(
                f"No atlas-linked genomes remain after applying --min-mimag-tier {args.min_mimag_tier}."
            )
        matched_df = matched_df.drop(columns=["_mimag_tier_norm", "_mimag_tier_value"], errors="ignore")

    compare_label = sanitize_label(args.atlas_compare_column)
    linked_prefix = sanitize_label(prefix_override) if prefix_override else sanitize_label(args.prefix)
    linked_dir = output_dir / f"{linked_prefix}_linked_{compare_label}"

    cleanup_deprecated_matched_lineage_annotation_outputs(output_dir, linked_prefix, compare_label)
    deprecated_dirs = [
        output_dir / "linked_comparative_analysis",
        output_dir / "complete_shared_linked_comparative_analysis",
        output_dir / f"{linked_prefix}_atlas_paired_{compare_label}_matched_lineage_annotation_presence",
        linked_dir,
    ]
    for path in deprecated_dirs:
        if path.exists():
            shutil.rmtree(path)
    for old_path in output_dir.glob(f"{linked_prefix}_atlas_paired_{compare_label}*"):
        if old_path.is_dir():
            shutil.rmtree(old_path)
        elif old_path.is_file():
            old_path.unlink()

    tables_dir = linked_dir / "tables"
    plots_dir = linked_dir / "plots"
    enrichment_plot_dir = plots_dir / "diff"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    enrichment_plot_dir.mkdir(parents=True, exist_ok=True)

    matched_out = tables_dir / "genome_matches.tsv"
    audit_out = tables_dir / "match_audit.tsv"
    matched_df.to_csv(matched_out, sep="\t", index=False)
    atlas_match_audit_df.to_csv(audit_out, sep="\t", index=False)

    representative_df, lineage_summary_df = build_complete_lineage_category_representatives(
        matched_df=matched_df,
        compare_column=args.atlas_compare_column,
    )
    if representative_df.empty:
        raise ValueError(
            "No complete species-level lineages with representatives in two or more categories remained. "
            "Check taxonomy completeness and atlas-to-MetaPathways genome matching."
        )

    representative_out = tables_dir / "species_reps.tsv"
    lineage_summary_out = tables_dir / "lineage_summary.tsv"
    representative_df.to_csv(representative_out, sep="\t", index=False)
    lineage_summary_df.to_csv(lineage_summary_out, sep="\t", index=False)

    enrichment_member_df = build_all_species_linked_member_table(
        matched_df=matched_df,
        compare_column=args.atlas_compare_column,
    )
    if enrichment_member_df.empty:
        enrichment_member_df = representative_df.copy()
    enrichment_member_out = tables_dir / "linked_members.tsv"
    enrichment_member_df.to_csv(enrichment_member_out, sep="\t", index=False)

    paired_df = build_paired_component_table(representative_df, args.atlas_compare_column)
    pair_summary_df = summarize_paired_deltas(paired_df)
    paired_out = tables_dir / "lineage_pairs.tsv"
    pair_summary_out = tables_dir / "lineage_pair_deltas.tsv"
    paired_df.to_csv(paired_out, sep="\t", index=False)
    pair_summary_df.to_csv(pair_summary_out, sep="\t", index=False)

    normalized_representatives_df = normalize_linked_representatives_for_summary(
        representative_df=representative_df,
        compare_column=args.atlas_compare_column,
        max_lineages=getattr(args, "max_lineage_detail_plots", 25),
    )
    category_summary_df = (
        build_method_effectiveness_summary(
            normalized_representatives_df,
            normalized_representatives_df,
            category_column="category",
        )
        if not normalized_representatives_df.empty else pd.DataFrame()
    )
    if not normalized_representatives_df.empty:
        normalized_representatives_df.to_csv(tables_dir / "heatmap_reps.tsv", sep="\t", index=False)
    if not category_summary_df.empty:
        category_summary_df.to_csv(tables_dir / "category_summary.tsv", sep="\t", index=False)

    matched_annotation_presence_df, matched_annotation_matrix_df = build_matched_lineage_annotation_tables(
        matched_df=representative_df,
        annotation_audit_df=combined_annotation_audit_df,
        compare_column=args.atlas_compare_column,
    )
    matched_annotation_presence_df.to_csv(tables_dir / "ann_presence.tsv", sep="\t", index=False)
    matched_annotation_matrix_df.to_csv(tables_dir / "ann_matrix.tsv", sep="\t", index=False)

    elemental_play_df = build_pair_elemental_play_by_play(
        matched_df=representative_df,
        compare_column=args.atlas_compare_column,
    )
    elemental_summary_df = summarize_pair_elemental_play_by_play(elemental_play_df)
    elemental_play_df.to_csv(tables_dir / "elemental_pairs.tsv", sep="\t", index=False)
    elemental_summary_df.to_csv(tables_dir / "elemental_summary.tsv", sep="\t", index=False)

    enrichment_annotation_presence_df, _enrichment_annotation_matrix_df = build_matched_lineage_annotation_tables(
        matched_df=enrichment_member_df,
        annotation_audit_df=combined_annotation_audit_df,
        compare_column=args.atlas_compare_column,
    )
    universe_df = prepare_global_lineage_universe(enrichment_member_df, args.atlas_compare_column)
    candidate_df = build_global_candidate_lineages(enrichment_member_df, args.atlas_compare_column)
    annotation_entity_presence_df = build_annotation_entity_presence(enrichment_annotation_presence_df)
    feature_entity_presence_df = build_feature_entity_presence(enrichment_member_df, args.atlas_compare_column)
    differential_entities = [
        ("functional_annotations", annotation_entity_presence_df),
        ("marker_reference_features", feature_entity_presence_df),
    ]
    wrote_paths = [
        matched_out,
        audit_out,
        representative_out,
        lineage_summary_out,
        enrichment_member_out,
        paired_out,
        pair_summary_out,
        tables_dir / "ann_presence.tsv",
        tables_dir / "ann_matrix.tsv",
        tables_dir / "elemental_pairs.tsv",
        tables_dir / "elemental_summary.tsv",
    ]
    if not normalized_representatives_df.empty:
        wrote_paths.append(tables_dir / "heatmap_reps.tsv")
    if not category_summary_df.empty:
        wrote_paths.append(tables_dir / "category_summary.tsv")

    for entity_label, entity_df in differential_entities:
        if entity_df.empty:
            continue
        short_label = "ann" if entity_label == "functional_annotations" else "marker_ref"
        category_presence_df, enrichment_df, pairwise_df, _status_df = build_global_entity_comparative_tables(
            universe_df=universe_df,
            entity_presence_df=entity_df,
            candidate_df=candidate_df,
            require_all_categories=False,
        )
        if not category_presence_df.empty:
            out_path = tables_dir / f"{short_label}_prevalence.tsv"
            category_presence_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
        if not enrichment_df.empty:
            out_path = tables_dir / f"{short_label}_enrichment.tsv"
            enrichment_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
        if not pairwise_df.empty:
            out_path = tables_dir / f"{short_label}_pairwise.tsv"
            pairwise_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
            plot_dir = enrichment_plot_dir / short_label
            plot_paths = write_category_pairwise_differential_plots(
                pairwise_df=pairwise_df,
                output_dir=plot_dir,
                entity_label=entity_label.replace("_", " ").title(),
                min_genomes_per_category=3,
            )
            kept_plot_paths = []
            for path in plot_dir.rglob("*.plot_data.tsv"):
                if path.exists():
                    path.unlink()
            for plot_path in plot_paths:
                path = Path(plot_path)
                if path.exists():
                    kept_plot_paths.append(path)
            wrote_paths.extend(kept_plot_paths)

        pooled_presence_df, pooled_enrichment_df, pooled_pairwise_df, _pooled_status_df = (
            build_pooled_candidate_entity_comparative_tables(
                universe_df=universe_df,
                entity_presence_df=entity_df,
                candidate_df=candidate_df,
            )
        )
        if not pooled_presence_df.empty:
            out_path = tables_dir / f"{short_label}_pooled_all_candidates_prevalence.tsv"
            pooled_presence_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
        if not pooled_enrichment_df.empty:
            out_path = tables_dir / f"{short_label}_pooled_all_candidates_enrichment.tsv"
            pooled_enrichment_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
        if not pooled_pairwise_df.empty:
            out_path = tables_dir / f"{short_label}_pooled_all_candidates_pairwise.tsv"
            pooled_pairwise_df.to_csv(out_path, sep="\t", index=False)
            wrote_paths.append(out_path)
            pooled_plot_dir = enrichment_plot_dir / f"{short_label}_pooled_all_candidates"
            pooled_plot_paths = write_category_pairwise_differential_plots(
                pairwise_df=pooled_pairwise_df,
                output_dir=pooled_plot_dir,
                entity_label=f"{entity_label.replace('_', ' ').title()} pooled across candidate lineages",
                min_genomes_per_category=3,
            )
            kept_pooled_plot_paths = []
            for path in pooled_plot_dir.rglob("*.plot_data.tsv"):
                if path.exists():
                    path.unlink()
            for plot_path in pooled_plot_paths:
                path = Path(plot_path)
                if path.exists():
                    kept_pooled_plot_paths.append(path)
            wrote_paths.extend(kept_pooled_plot_paths)

    lineage_heatmap_base = plots_dir / "lineage_heatmap"
    mode_heatmap_base = plots_dir / "mode_support"
    category_panel_base = plots_dir / "category_summary"
    elemental_panel_base = plots_dir / "elemental_deltas"
    annotation_heatmap_base = plots_dir / "annotation_heatmap"

    if not normalized_representatives_df.empty and plot_sample_representative_heatmap(
        normalized_representatives_df,
        lineage_heatmap_base,
        category_column="category",
        sample_column="sample",
    ):
        wrote_paths.extend([Path(str(lineage_heatmap_base) + ".png"), Path(str(lineage_heatmap_base) + ".pdf")])

    if not normalized_representatives_df.empty and plot_category_mode_support_heatmaps(
        normalized_representatives_df,
        mode_heatmap_base,
        category_column="category",
    ):
        wrote_paths.extend([Path(str(mode_heatmap_base) + ".png"), Path(str(mode_heatmap_base) + ".pdf")])

    if not category_summary_df.empty and plot_method_effectiveness_panel(
        category_summary_df,
        category_panel_base,
        category_column="category",
    ):
        wrote_paths.extend([Path(str(category_panel_base) + ".png"), Path(str(category_panel_base) + ".pdf")])

    if not elemental_play_df.empty and plot_pair_elemental_play_by_play(
        elemental_play_df,
        elemental_panel_base,
    ):
        wrote_paths.extend([Path(str(elemental_panel_base) + ".png"), Path(str(elemental_panel_base) + ".pdf")])

    if not matched_annotation_matrix_df.empty and plot_matched_lineage_annotation_presence_matrix(
        matched_annotation_matrix_df,
        annotation_heatmap_base,
    ):
        wrote_paths.extend([Path(str(annotation_heatmap_base) + ".png"), Path(str(annotation_heatmap_base) + ".pdf")])

    if shared_best_path:
        wrote_paths.append(shared_best_path)
    if annotated_path:
        wrote_paths.append(annotated_path)
    return [path for path in wrote_paths if path]
