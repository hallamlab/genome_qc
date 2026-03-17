#!/usr/bin/env python3

import argparse
import itertools
import math
import os
import re
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd

plt = None
sns = None

SCORE_COLUMN = "qscore"
SCORE_LABEL = "Master qscore"
TIER_PALETTE = {"low": "#d9d9d9", "medium": "#7f7f7f", "high": "#1a1a1a"}
RIGHT_MARGIN = 0.70
TAXONOMY_RANKS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
SHARED_QUALITY_METRICS = [
    (SCORE_COLUMN, SCORE_LABEL),
    ("Completeness", "Completeness (%)"),
    ("Contamination", "Contamination (%)"),
    ("integrity_score", "Integrity"),
    ("recoverability_score", "Recoverability"),
    ("rrna_16S_score", "16S score"),
    ("rrna_23S_score", "23S score"),
    ("rrna_5S_score", "5S score"),
    ("trna_ge_18", "tRNA>=18"),
]
SHARED_DIFFERENCE_METRICS = [
    (SCORE_COLUMN, SCORE_LABEL),
    ("Completeness", "Completeness (%)"),
    ("Contamination", "Contamination (%)"),
    ("integrity_score", "Integrity"),
    ("recoverability_score", "Recoverability"),
    ("16S_rRNA", "16S copies"),
    ("23S_rRNA", "23S copies"),
    ("5S_rRNA", "5S copies"),
    ("trna_unique", "Unique tRNAs"),
]
FAVORABLE_NEGATIVE_METRICS = {"Contamination"}

REQUIRED_COLUMNS = [
    "Genome_Id",
    "Bin Id",
    "Completeness",
    "Contamination",
    "qscore",
    "23S_rRNA",
    "5S_rRNA",
    "trna_unique",
]


def ensure_plotting():
    global plt, sns
    if plt is not None and sns is not None:
        return plt, sns

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        import seaborn as sns_mod
    except Exception as exc:
        raise RuntimeError(
            "Plotting dependencies could not be imported. "
            "This environment currently has a matplotlib/pandas compatibility issue."
        ) from exc

    plt = plt_mod
    sns = sns_mod
    return plt, sns


def clean_group_series(series):
    cleaned = series.copy()
    if pd.api.types.is_object_dtype(cleaned) or pd.api.types.is_string_dtype(cleaned):
        cleaned = cleaned.map(lambda value: value.strip() if isinstance(value, str) else value)
        cleaned = cleaned.replace("", np.nan)
    return cleaned


def resolve_grouping_series(frame, group_column):
    if not group_column or group_column not in frame.columns:
        return None

    resolved = clean_group_series(frame[group_column]).astype(object)
    if group_column not in TAXONOMY_RANKS:
        return resolved

    missing = resolved.isna()
    if not missing.any():
        return resolved

    rank_index = TAXONOMY_RANKS.index(group_column)
    for higher_rank in reversed(TAXONOMY_RANKS[:rank_index]):
        if higher_rank not in frame.columns:
            continue
        higher_values = clean_group_series(frame[higher_rank])
        fill_mask = missing & higher_values.notna()
        if fill_mask.any():
            resolved.loc[fill_mask] = "unclassified_" + higher_values.loc[fill_mask].astype(str)
            missing = resolved.isna()
        if not missing.any():
            break

    if missing.any():
        resolved.loc[missing] = "unclassified_unknown"
    return resolved


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute a continuous MIMAG-style genome quality index and generate "
            "a multi-panel atlas from Master_genome_QC.tsv."
        )
    )
    parser.add_argument("master_tsv", help="Path to the master genome QC TSV file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <master-dir>/genome_quality_atlas.",
    )
    parser.add_argument(
        "--prefix",
        default="genome_quality",
        help="Prefix for output files. Default: genome_quality",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Optional column used for group summaries/statistics and the atlas "
            "comparison panel. Example: Domain, Phylum, sample_set."
        ),
    )
    parser.add_argument(
        "--sample-column",
        default=None,
        help=(
            "Optional sample/replicate column used for sample-aware category "
            "count comparisons. Example: sample, site, replicate."
        ),
    )
    parser.add_argument(
        "--matched-samples-only",
        action="store_true",
        help=(
            "Restrict category-comparison outputs to samples present in every "
            "category. FastANI overlap/dedup comparisons are then constrained "
            "to within-sample matches only."
        ),
    )
    parser.add_argument(
        "--top-n-groups",
        type=int,
        default=12,
        help="Maximum number of groups to show in the comparison panel. Default: 12",
    )
    parser.add_argument(
        "--compare-column",
        action="append",
        default=[],
        help=(
            "Category column to compare across. May be passed multiple times. "
            "Columns can come from the master table or from --compare-map."
        ),
    )
    parser.add_argument(
        "--compare-map",
        default=None,
        help=(
            "Optional TSV/CSV containing Genome_Id or Bin Id plus one or more "
            "custom category columns to merge before comparison plotting."
        ),
    )
    parser.add_argument(
        "--compare-map-key",
        default=None,
        help=(
            "Join key for --compare-map. Defaults to auto-detecting Genome_Id "
            "or Bin Id."
        ),
    )
    parser.add_argument(
        "--ani-compare-column",
        action="append",
        default=[],
        help=(
            "Category column for FastANI-based overlap comparison. May be passed "
            "multiple times."
        ),
    )
    parser.add_argument(
        "--ani-results",
        default=None,
        help=(
            "Optional existing FastANI results file to reuse instead of running "
            "FastANI."
        ),
    )
    parser.add_argument(
        "--ani-genome-dir",
        default=None,
        help=(
            "Directory containing genome FASTA files. Used when running FastANI "
            "unless --ani-fasta-column is provided."
        ),
    )
    parser.add_argument(
        "--ani-fasta-column",
        default=None,
        help="Column containing genome FASTA paths for FastANI.",
    )
    parser.add_argument(
        "--ani-fasta-exts",
        default=".fasta,.fa,.fna,.fasta.gz,.fa.gz,.fna.gz",
        help="Comma-separated FASTA extensions used when resolving files from --ani-genome-dir.",
    )
    parser.add_argument(
        "--ani-threshold",
        type=float,
        default=95.0,
        help="ANI threshold for calling genomes matched. Default: 95.0",
    )
    parser.add_argument(
        "--ani-af-threshold",
        type=float,
        default=0.5,
        help="Minimum alignment fraction for calling genomes matched. Default: 0.5",
    )
    parser.add_argument(
        "--ani-threads",
        type=int,
        default=1,
        help="Number of threads for FastANI. Default: 1",
    )
    parser.add_argument(
        "--ani-top-overlaps",
        type=int,
        default=15,
        help="Maximum number of overlap combinations to show in the UpSet plot. Default: 15",
    )
    return parser


def ensure_columns(frame):
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if "16S_rRNA" not in frame.columns and "contains_16S" not in frame.columns:
        missing.append("16S_rRNA|contains_16S")
    if missing:
        raise ValueError(
            "Input table is missing required columns: " + ", ".join(missing)
        )


def read_table_auto(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def sanitize_token(value):
    token = str(value).strip()
    allowed = []
    for char in token:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    cleaned = "".join(allowed).strip("_")
    return cleaned or "comparison"


def merge_compare_map(frame, compare_map_path, compare_map_key=None):
    compare_df = read_table_auto(compare_map_path)
    possible_keys = ["Genome_Id", "Bin Id"]

    join_key = compare_map_key
    if join_key is None:
        for key in possible_keys:
            if key in frame.columns and key in compare_df.columns:
                join_key = key
                break

    if join_key is None:
        raise ValueError(
            "--compare-map must contain one of the join keys shared with the master table: "
            + ", ".join(possible_keys)
        )
    if join_key not in frame.columns or join_key not in compare_df.columns:
        raise ValueError(f"Join key '{join_key}' is not present in both tables.")

    additional_columns = [
        column for column in compare_df.columns
        if column != join_key and column not in frame.columns
    ]
    if not additional_columns:
        warnings.warn(
            "--compare-map does not add any new columns; keeping the original master table unchanged.",
            RuntimeWarning,
        )
        return frame

    keep_columns = [join_key] + additional_columns
    return frame.merge(compare_df[keep_columns], on=join_key, how="left")


def prepare_compare_frame(frame, compare_column):
    if compare_column not in frame.columns:
        raise ValueError(f"Comparison column '{compare_column}' was not found.")

    compare_df = frame.copy()
    compare_df[compare_column] = (
        compare_df[compare_column]
        .fillna("Unassigned")
        .astype(str)
        .str.strip()
        .replace("", "Unassigned")
    )
    compare_df = compare_df.dropna(subset=[SCORE_COLUMN])
    if compare_df.empty:
        raise ValueError(f"No usable rows available for comparison column '{compare_column}'.")
    return compare_df


def filter_to_matched_samples(compare_df, compare_column, sample_column=None):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if not selected_sample or selected_sample not in compare_df.columns:
        return compare_df.copy(), selected_sample

    filtered = compare_df.copy()
    filtered[selected_sample] = (
        filtered[selected_sample]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    filtered = filtered.loc[filtered[selected_sample] != ""].copy()
    if filtered.empty:
        return filtered, selected_sample

    n_categories = filtered[compare_column].astype(str).nunique()
    if n_categories < 2:
        return filtered, selected_sample

    sample_scope = (
        filtered.groupby(selected_sample)[compare_column]
        .nunique()
    )
    matched_samples = sample_scope.loc[sample_scope == n_categories].index.astype(str)
    filtered = filtered.loc[filtered[selected_sample].isin(matched_samples)].copy()
    return filtered, selected_sample


PREFERRED_METHOD_ORDER = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]


def preferred_method_rank(label):
    text = str(label).strip()
    lowered = text.lower()
    exact_map = {name.lower(): index for index, name in enumerate(PREFERRED_METHOD_ORDER)}
    if lowered in exact_map:
        return exact_map[lowered]
    has_sag = "sag" in lowered
    has_mag = "mag" in lowered
    has_xpg = "xpg" in lowered
    if has_sag and not has_xpg:
        return 0
    if has_sag and has_xpg:
        return 1
    if has_mag and not has_xpg:
        return 2
    if has_mag and has_xpg:
        return 3
    return 999


def ordered_unique_categories(values):
    unique_values = sorted(
        {
            str(value).strip()
            for value in values
            if str(value).strip() != ""
            and str(value).strip().lower() not in {"nan", "none", "null"}
        }
    )
    return sorted(unique_values, key=lambda value: (preferred_method_rank(value), value.lower()))


def category_order(frame, compare_column):
    median_lookup = (
        frame.groupby(compare_column)[SCORE_COLUMN]
        .median()
        .to_dict()
    )
    categories = ordered_unique_categories(frame[compare_column].astype(str).tolist())
    return sorted(
        categories,
        key=lambda category: (
            preferred_method_rank(category),
            0 if preferred_method_rank(category) < 999 else -float(median_lookup.get(category, float("-inf"))),
            category.lower(),
        ),
    )


def grayscale_palette(n_values, start=0.2, stop=0.85):
    if n_values <= 1:
        return ["#4d4d4d"]
    values = np.linspace(start, stop, n_values)
    return [str(value) for value in values]


def apply_tight_layout(fig, rect=(0, 0, 1, 1)):
    fig.tight_layout(rect=rect, pad=1.2, w_pad=2.0, h_pad=2.0)


def style_long_ticklabels(ax, axis="x", rotation=90, size=8):
    ax.tick_params(axis=axis, rotation=rotation, labelsize=size)


def split_extensions(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def strip_known_extension(filename, extensions):
    for extension in sorted(extensions, key=len, reverse=True):
        if filename.endswith(extension):
            return filename[: -len(extension)]
    return os.path.splitext(filename)[0]


def resolve_fasta_paths(frame, fasta_column=None, genome_dir=None, fasta_exts=None):
    fasta_exts = fasta_exts or [".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz"]
    resolved = frame.copy()

    if fasta_column:
        if fasta_column not in resolved.columns:
            raise ValueError(f"FASTA column '{fasta_column}' was not found.")
        resolved["ani_fasta_path"] = [
            os.path.abspath(os.path.expanduser(str(path)))
            for path in resolved[fasta_column].astype(str)
        ]
        missing = [path for path in resolved["ani_fasta_path"] if not os.path.exists(path)]
        if missing:
            raise ValueError(
                "Some FASTA paths from the requested column do not exist. Example: "
                + missing[0]
            )
        resolved["ani_record_id"] = resolved["ani_fasta_path"].astype(str)
        return resolved

    if not genome_dir:
        raise ValueError(
            "FastANI requires either --ani-fasta-column or --ani-genome-dir."
        )

    path_map = {}
    genome_dir = os.path.abspath(os.path.expanduser(genome_dir))
    for root, _, files in os.walk(genome_dir):
        for name in files:
            if any(name.endswith(extension) for extension in fasta_exts):
                stem = strip_known_extension(name, fasta_exts)
                full_path = os.path.abspath(os.path.join(root, name))
                if stem not in path_map:
                    path_map[stem] = full_path

    if not path_map:
        raise ValueError(f"No FASTA files were found under '{genome_dir}'.")

    resolved_paths = []
    missing_ids = []
    for _, row in resolved.iterrows():
        candidates = []
        if "Bin Id" in resolved.columns:
            candidates.append(str(row["Bin Id"]))
        if "Genome_Id" in resolved.columns:
            candidates.append(str(row["Genome_Id"]))
        found = None
        for candidate in candidates:
            if candidate in path_map:
                found = path_map[candidate]
                break
        resolved_paths.append(found)
        if found is None:
            missing_ids.append(candidates[0] if candidates else "unknown")

    if missing_ids:
        raise ValueError(
            "Could not resolve FASTA paths for some genomes. Example missing ID: "
            + missing_ids[0]
        )

    resolved["ani_fasta_path"] = resolved_paths
    resolved["ani_record_id"] = resolved["ani_fasta_path"].astype(str)
    return resolved


def run_fastani(frame, output_base, threads=1, existing_results=None):
    ensure_fastani = shutil.which("fastANI") or shutil.which("fastani")
    raw_output = output_base + "_fastani_raw.tsv"

    if existing_results:
        raw_output = os.path.abspath(os.path.expanduser(existing_results))
        if not os.path.exists(raw_output):
            raise ValueError(f"FastANI results file was not found: {raw_output}")
        return raw_output

    if not ensure_fastani:
        raise ValueError(
            "FastANI was requested but no fastANI binary was found in PATH."
        )

    fasta_list_path = output_base + "_fastani_genomes.txt"
    unique_paths = (
        frame["ani_fasta_path"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    with open(fasta_list_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(unique_paths) + "\n")

    cmd = [
        ensure_fastani,
        "--ql",
        fasta_list_path,
        "--rl",
        fasta_list_path,
        "-o",
        raw_output,
        "-t",
        str(int(threads)),
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "FastANI failed.\nSTDOUT:\n"
            + completed.stdout
            + "\nSTDERR:\n"
            + completed.stderr
        )
    return raw_output


def load_fastani_pairs(raw_output, frame):
    pair_df = pd.read_csv(
        raw_output,
        sep="\t",
        header=None,
        names=["query_path", "reference_path", "ani", "fragments_mapped", "total_fragments"],
    )
    if pair_df.empty:
        return pair_df

    metadata_columns = ["ani_record_id", "ani_fasta_path", "Genome_Id"]
    optional_columns = [
        "Bin Id",
        "source_dir",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
    ]
    for column in optional_columns:
        if column in frame.columns:
            metadata_columns.append(column)
    metadata_df = frame[metadata_columns].drop_duplicates("ani_record_id").copy()
    path_to_record = dict(
        zip(
            metadata_df["ani_fasta_path"].astype(str),
            metadata_df["ani_record_id"].astype(str),
        )
    )
    pair_df["query_path"] = pair_df["query_path"].astype(str)
    pair_df["reference_path"] = pair_df["reference_path"].astype(str)
    pair_df["query_record"] = pair_df["query_path"].map(path_to_record)
    pair_df["reference_record"] = pair_df["reference_path"].map(path_to_record)
    pair_df = pair_df.dropna(subset=["query_record", "reference_record"]).copy()
    pair_df = pair_df.loc[pair_df["query_record"] != pair_df["reference_record"]].copy()
    if pair_df.empty:
        return pair_df

    pair_df["ani"] = pd.to_numeric(pair_df["ani"], errors="coerce")
    pair_df["fragments_mapped"] = pd.to_numeric(pair_df["fragments_mapped"], errors="coerce")
    pair_df["total_fragments"] = pd.to_numeric(pair_df["total_fragments"], errors="coerce")
    pair_df["af_query"] = np.where(
        pair_df["total_fragments"] > 0,
        pair_df["fragments_mapped"] / pair_df["total_fragments"],
        np.nan,
    )
    pair_df["record_a"] = pair_df[["query_record", "reference_record"]].min(axis=1)
    pair_df["record_b"] = pair_df[["query_record", "reference_record"]].max(axis=1)

    aggregated = (
        pair_df.groupby(["record_a", "record_b"], as_index=False)
        .agg(
            ani_mean=("ani", "mean"),
            ani_max=("ani", "max"),
            af_min=("af_query", "min"),
            af_max=("af_query", "max"),
            n_directions=("af_query", "size"),
        )
    )
    left_meta = metadata_df.add_prefix("a_").rename(columns={"a_ani_record_id": "record_a"})
    right_meta = metadata_df.add_prefix("b_").rename(columns={"b_ani_record_id": "record_b"})
    aggregated = aggregated.merge(left_meta, on="record_a", how="left")
    aggregated = aggregated.merge(right_meta, on="record_b", how="left")
    aggregated["genome_a"] = aggregated["a_Genome_Id"]
    aggregated["genome_b"] = aggregated["b_Genome_Id"]
    if "a_Bin Id" in aggregated.columns:
        aggregated["bin_id_a"] = aggregated["a_Bin Id"]
        aggregated["bin_id_b"] = aggregated["b_Bin Id"]
    return aggregated


def connected_components(nodes, edges):
    adjacency = {node: set() for node in nodes}
    for node_a, node_b in edges:
        adjacency.setdefault(node_a, set()).add(node_b)
        adjacency.setdefault(node_b, set()).add(node_a)

    visited = set()
    components = []
    for node in nodes:
        if node in visited:
            continue
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited))
        components.append(sorted(component))
    return components


def summarize_fastani_matches(frame, pair_df, compare_column, ani_threshold, af_threshold):
    compare_df = prepare_compare_frame(frame, compare_column)
    record_to_category = dict(
        zip(compare_df["ani_record_id"].astype(str), compare_df[compare_column].astype(str))
    )
    pair_summary = pair_df.copy()
    if pair_summary.empty:
        pair_summary = pd.DataFrame(
            columns=[
                "record_a",
                "record_b",
                "genome_a",
                "genome_b",
                "ani_mean",
                "ani_max",
                "af_min",
                "af_max",
                "n_directions",
                "category_a",
                "category_b",
                "category_pair",
                "passes_threshold",
            ]
        )
        category_summary = (
            compare_df.groupby(compare_column)
            .agg(n_genomes=("Genome_Id", "size"))
            .reset_index()
        )
        category_summary["matched_genomes"] = 0
        category_summary["matched_fraction"] = 0.0
        return compare_df, pair_summary, pd.DataFrame(), category_summary

    pair_summary["category_a"] = pair_summary["record_a"].map(record_to_category)
    pair_summary["category_b"] = pair_summary["record_b"].map(record_to_category)
    pair_summary = pair_summary.dropna(subset=["category_a", "category_b"]).copy()
    normalized_pairs = pair_summary.apply(
        lambda row: sorted([str(row["category_a"]), str(row["category_b"])]),
        axis=1,
        result_type="expand",
    )
    pair_summary["category_1"] = normalized_pairs[0]
    pair_summary["category_2"] = normalized_pairs[1]
    pair_summary["category_pair"] = pair_summary.apply(
        lambda row: " | ".join([str(row["category_1"]), str(row["category_2"])]),
        axis=1,
    )
    pair_summary["passes_threshold"] = (
        pair_summary["ani_mean"].ge(ani_threshold)
        & pair_summary["af_min"].ge(af_threshold)
    )

    category_pair_summary = (
        pair_summary.groupby(["category_1", "category_2"], dropna=False)
        .agg(
            pair_count=("ani_mean", "size"),
            matched_pair_count=("passes_threshold", "sum"),
            median_ani=("ani_mean", "median"),
            median_af=("af_min", "median"),
            max_ani=("ani_max", "max"),
            max_af=("af_max", "max"),
        )
        .reset_index()
        .rename(columns={"category_1": "category_a", "category_2": "category_b"})
    )

    matched_genomes = set(
        pair_summary.loc[pair_summary["passes_threshold"], ["record_a", "record_b"]]
        .stack()
        .astype(str)
        .tolist()
    )
    category_summary = (
        compare_df.assign(
            ani_matched=compare_df["ani_record_id"].astype(str).isin(matched_genomes)
        )
        .groupby(compare_column)
        .agg(
            n_genomes=("Genome_Id", "size"),
            matched_genomes=("ani_matched", "sum"),
        )
        .reset_index()
    )
    category_summary["matched_fraction"] = np.where(
        category_summary["n_genomes"] > 0,
        category_summary["matched_genomes"] / category_summary["n_genomes"],
        np.nan,
    )
    return compare_df, pair_summary, category_pair_summary, category_summary


def filter_fastani_pairs_within_sample(compare_df, pair_summary, sample_column=None):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if pair_summary.empty or not selected_sample or selected_sample not in compare_df.columns:
        return pair_summary.copy(), selected_sample

    sample_map = dict(
        zip(compare_df["ani_record_id"].astype(str), compare_df[selected_sample].astype(str))
    )
    filtered = pair_summary.copy()
    filtered["sample_a"] = filtered["record_a"].map(sample_map)
    filtered["sample_b"] = filtered["record_b"].map(sample_map)
    filtered = filtered.dropna(subset=["sample_a", "sample_b"]).copy()
    filtered = filtered.loc[filtered["sample_a"] == filtered["sample_b"]].copy()
    return filtered, selected_sample


def build_fastani_components(compare_df, pair_summary, compare_column):
    record_to_category = dict(
        zip(compare_df["ani_record_id"].astype(str), compare_df[compare_column].astype(str))
    )
    all_nodes = sorted(compare_df["ani_record_id"].astype(str).tolist())
    filtered_edges = pair_summary.loc[pair_summary["passes_threshold"], ["record_a", "record_b"]]
    edge_list = [tuple(row) for row in filtered_edges.itertuples(index=False, name=None)]
    components = connected_components(all_nodes, edge_list)
    record_to_label = dict(
        zip(compare_df["ani_record_id"].astype(str), compare_df["Genome_Id"].astype(str))
    )

    component_rows = []
    overlap_rows = []
    for index, genomes in enumerate(components, start=1):
        categories = sorted({record_to_category.get(genome, "Unassigned") for genome in genomes})
        component_rows.append(
            {
                "component_id": f"component_{index:04d}",
                "n_genomes": len(genomes),
                "n_categories": len(categories),
                "categories": ";".join(categories),
                "records": ";".join(genomes),
                "genomes": ";".join(record_to_label.get(genome, genome) for genome in genomes),
            }
        )
        overlap_rows.append(
            {
                "component_id": f"component_{index:04d}",
                "categories": tuple(categories),
                "overlap_key": " & ".join(categories),
                "n_categories": len(categories),
                "n_genomes": len(genomes),
            }
        )

    component_df = pd.DataFrame(component_rows)
    overlap_df = (
        pd.DataFrame(overlap_rows)
        .groupby(["overlap_key", "n_categories"], as_index=False)
        .agg(
            component_count=("component_id", "size"),
            total_genomes=("n_genomes", "sum"),
        )
    )
    return component_df, overlap_df


def component_map_from_pairs(compare_df, pair_summary):
    if compare_df.empty or "ani_record_id" not in compare_df.columns:
        return {}

    all_nodes = sorted(compare_df["ani_record_id"].astype(str).tolist())
    node_set = set(all_nodes)
    if pair_summary.empty:
        edge_list = []
    else:
        filtered_edges = pair_summary.loc[pair_summary["passes_threshold"], ["record_a", "record_b"]]
        edge_list = [
            (record_a, record_b)
            for record_a, record_b in filtered_edges.itertuples(index=False, name=None)
            if record_a in node_set and record_b in node_set
        ]
    components = connected_components(all_nodes, edge_list)
    component_map = {}
    for index, nodes in enumerate(components, start=1):
        component_id = f"component_{index:04d}"
        for node in nodes:
            component_map[node] = component_id
    return component_map


def build_shared_best_genome_table(compare_df, pair_summary, compare_column):
    if compare_df.empty or "ani_record_id" not in compare_df.columns:
        return pd.DataFrame()

    working = compare_df.copy()
    working["ani_record_id"] = working["ani_record_id"].astype(str)
    working["component_id"] = working["ani_record_id"].map(component_map_from_pairs(working, pair_summary))
    working = working.dropna(subset=["component_id"]).copy()
    if working.empty:
        return pd.DataFrame()

    component_n_categories = (
        working.groupby("component_id")[compare_column]
        .nunique()
        .rename("component_n_categories")
    )
    shared_component_ids = component_n_categories.loc[component_n_categories > 1].index
    working = working.loc[working["component_id"].isin(shared_component_ids)].copy()
    if working.empty:
        return pd.DataFrame()

    component_categories = (
        working.groupby("component_id")[compare_column]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .rename("component_categories")
    )
    component_member_count = working.groupby("component_id").size().rename("component_member_count")
    category_member_count = (
        working.groupby(["component_id", compare_column])
        .size()
        .rename("category_member_count_in_component")
    )
    total_categories = working[compare_column].astype(str).nunique()

    sort_specs = [
        (SCORE_COLUMN, False),
        ("Completeness", False),
        ("integrity_score", False),
        ("recoverability_score", False),
        ("Contamination", True),
        ("Genome_Id", True),
        ("Bin Id", True),
    ]
    sort_columns = [column for column, _ in sort_specs if column in working.columns]
    sort_ascending = [ascending for column, ascending in sort_specs if column in working.columns]
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=sort_ascending, kind="mergesort")

    best_df = (
        working.groupby(["component_id", compare_column], as_index=False, sort=False)
        .head(1)
        .copy()
    )
    best_df["component_categories"] = best_df["component_id"].map(component_categories)
    best_df["component_n_categories"] = best_df["component_id"].map(component_n_categories)
    best_df["component_member_count"] = best_df["component_id"].map(component_member_count)
    best_df["category_member_count_in_component"] = (
        best_df.set_index(["component_id", compare_column]).index.map(category_member_count)
    )
    best_df["shared_scope"] = np.where(
        best_df["component_n_categories"].eq(total_categories),
        "complete",
        "partial",
    )
    best_df["best_selection_metric"] = SCORE_COLUMN
    return best_df

def feature_present(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).gt(0).astype(int)

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    truthy = {"true", "t", "yes", "y", "present", "1"}
    return normalized.isin(truthy).astype(int)


def capped_feature_score(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).clip(lower=0, upper=1)

    return feature_present(series).astype(float)


def mimag_tier(frame):
    tiers = np.full(len(frame), "low", dtype=object)
    medium = (frame["Completeness"] >= 50) & (frame["Contamination"] < 10)
    high = (frame["Completeness"] > 90) & (frame["Contamination"] < 5)
    tiers[medium] = "medium"
    tiers[high] = "high"
    return pd.Categorical(tiers, categories=["low", "medium", "high"], ordered=True)


def compute_quality_index(frame):
    annotated = frame.copy()

    numeric_columns = [
        "Completeness",
        "Contamination",
        "qscore",
        "16S_rRNA",
        "23S_rRNA",
        "5S_rRNA",
        "trna_unique",
    ]
    for column in numeric_columns:
        if column in annotated.columns:
            annotated[column] = pd.to_numeric(annotated[column], errors="coerce")

    has_16s_source = "16S_rRNA" if "16S_rRNA" in annotated.columns else "contains_16S"
    annotated["rrna_16S_score"] = capped_feature_score(annotated[has_16s_source])
    annotated["rrna_23S_score"] = capped_feature_score(annotated["23S_rRNA"])
    annotated["rrna_5S_score"] = capped_feature_score(annotated["5S_rRNA"])
    annotated["has_16S"] = annotated["rrna_16S_score"].gt(0).astype(int)
    annotated["has_23S"] = annotated["rrna_23S_score"].gt(0).astype(int)
    annotated["has_5S"] = annotated["rrna_5S_score"].gt(0).astype(int)
    annotated["trna_ge_18"] = annotated["trna_unique"].fillna(0).ge(18).astype(int)

    annotated["mC"] = np.clip((annotated["Completeness"] - 50.0) / 40.0, 0.0, 1.0)
    annotated["mK"] = np.clip((10.0 - annotated["Contamination"]) / 5.0, 0.0, 1.0)
    annotated["integrity_score"] = np.minimum(annotated["mC"], annotated["mK"])

    recoverability_matrix = annotated[
        ["rrna_16S_score", "rrna_23S_score", "rrna_5S_score", "trna_ge_18"]
    ]
    annotated["recoverability_score"] = recoverability_matrix.mean(axis=1)
    feature_matrix = annotated[["has_16S", "has_23S", "has_5S", "trna_ge_18"]]
    annotated["mimag_quality_index"] = np.sqrt(
        annotated["integrity_score"] * annotated["recoverability_score"]
    )
    annotated["recovered_feature_count"] = feature_matrix.sum(axis=1)
    annotated["mimag_tier"] = mimag_tier(annotated)
    annotated["recovery_pattern"] = feature_matrix.apply(
        lambda row: "".join(str(int(value)) for value in row), axis=1
    )

    pattern_names = []
    for _, row in feature_matrix.iterrows():
        labels = []
        if row["has_16S"]:
            labels.append("16S")
        if row["has_23S"]:
            labels.append("23S")
        if row["has_5S"]:
            labels.append("5S")
        if row["trna_ge_18"]:
            labels.append("tRNA>=18")
        pattern_names.append("+".join(labels) if labels else "none")
    annotated["recovery_pattern_label"] = pattern_names
    return annotated


def summarize(frame, group_column=None):
    overall = pd.DataFrame(
        [
            {
                "n_genomes": len(frame),
                "median_qscore": frame[SCORE_COLUMN].median(),
                "mean_qscore": frame[SCORE_COLUMN].mean(),
                "median_integrity": frame["integrity_score"].median(),
                "median_recoverability": frame["recoverability_score"].median(),
                "median_mimag_quality_index": frame["mimag_quality_index"].median(),
                "high_quality_n": int((frame["mimag_tier"] == "high").sum()),
                "medium_quality_n": int((frame["mimag_tier"] == "medium").sum()),
                "low_quality_n": int((frame["mimag_tier"] == "low").sum()),
            }
        ]
    )

    feature_rates = (
        frame[["has_16S", "has_23S", "has_5S", "trna_ge_18"]]
        .mean()
        .rename("fraction_present")
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    group_summary = None
    if group_column and group_column in frame.columns:
        grouped = frame.copy()
        grouped["__group_label"] = resolve_grouping_series(grouped, group_column)
        grouped = grouped.dropna(subset=["__group_label"])
        if not grouped.empty:
            group_summary = (
                grouped.groupby("__group_label", dropna=True)
                .agg(
                    n_genomes=("Genome_Id", "size"),
                    median_qscore=(SCORE_COLUMN, "median"),
                    mean_qscore=(SCORE_COLUMN, "mean"),
                    median_integrity=("integrity_score", "median"),
                    median_recoverability=("recoverability_score", "median"),
                    median_mimag_quality_index=("mimag_quality_index", "median"),
                )
                .sort_values(
                    by=["median_qscore", "n_genomes"],
                    ascending=[False, False],
                )
                .reset_index()
                .rename(columns={"__group_label": group_column})
            )
    return overall, feature_rates, group_summary


def run_nonparametric_test(frame, group_column):
    if not group_column or group_column not in frame.columns:
        return None

    grouped = frame.copy()
    grouped["__group_label"] = resolve_grouping_series(grouped, group_column)
    grouped = grouped.dropna(subset=["__group_label"])
    if grouped.empty:
        return None

    value_counts = grouped["__group_label"].value_counts()
    valid_groups = value_counts[value_counts > 0].index.tolist()
    if len(valid_groups) < 2:
        return None

    group_values = [
        grouped.loc[grouped["__group_label"] == group, SCORE_COLUMN].values
        for group in valid_groups
    ]

    try:
        from scipy import stats
    except ImportError:
        warnings.warn(
            "scipy is not installed; skipping nonparametric group statistics.",
            RuntimeWarning,
        )
        return None

    if len(valid_groups) == 2:
        stat, pvalue = stats.mannwhitneyu(
            group_values[0], group_values[1], alternative="two-sided"
        )
        return pd.DataFrame(
            [
                {
                    "test": "Mann-Whitney U",
                    "group_column": group_column,
                    "group_a": valid_groups[0],
                    "group_b": valid_groups[1],
                    "statistic": stat,
                    "pvalue": pvalue,
                }
            ]
        )

    stat, pvalue = stats.kruskal(*group_values)
    return pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis",
                "group_column": group_column,
                "n_groups": len(valid_groups),
                "groups": ";".join(map(str, valid_groups)),
                "statistic": stat,
                "pvalue": pvalue,
            }
        ]
    )


def add_quality_thresholds(ax):
    ax.axvline(50, color="#6b7280", linestyle="--", linewidth=1)
    ax.axhline(10, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(90, color="#111827", linestyle="--", linewidth=1)
    ax.axhline(5, color="#111827", linestyle="--", linewidth=1)
    ax.set_xlim(-1, 101)
    ymax = max(12.0, min(20.0, float(ax.get_ylim()[1])))
    ax.set_ylim(-0.25, ymax)


def plot_completeness_contamination(ax, frame):
    tiers = {"low": "o", "medium": "s", "high": "D"}
    scatter = None
    for tier, marker in tiers.items():
        subset = frame.loc[frame["mimag_tier"] == tier]
        if subset.empty:
            continue
        scatter = ax.scatter(
            subset["Completeness"],
            subset["Contamination"],
            c=subset[SCORE_COLUMN],
            cmap="Greys",
            s=35,
            marker=marker,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )
    add_quality_thresholds(ax)
    ax.set_xlabel("Completeness (%)")
    ax.set_ylabel("Contamination (%)")
    ax.set_title("Threshold landscape")
    return scatter


def plot_component_scatter(ax, frame):
    scatter = ax.scatter(
        frame["integrity_score"],
        frame["recoverability_score"],
        c=frame[SCORE_COLUMN],
        cmap="Greys",
        s=35,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Integrity (I)")
    ax.set_ylabel("Recoverability (R)")
    ax.set_title("Integrity vs recoverability")
    return scatter


def plot_quality_distribution(ax, frame):
    sns.histplot(
        data=frame,
        x=SCORE_COLUMN,
        hue="mimag_tier",
        hue_order=["low", "medium", "high"],
        palette=TIER_PALETTE,
        bins=20,
        edgecolor="white",
        alpha=0.8,
        multiple="stack",
        legend=False,
        ax=ax,
    )
    qmin = float(frame[SCORE_COLUMN].min())
    qmax = float(frame[SCORE_COLUMN].max())
    if math.isfinite(qmin) and math.isfinite(qmax) and qmin != qmax:
        padding = max((qmax - qmin) * 0.05, 1.0)
        ax.set_xlim(qmin - padding, qmax + padding)
    ax.set_xlabel(SCORE_LABEL)
    ax.set_ylabel("Genome count")
    ax.set_title("qscore distribution")


def plot_feature_recovery(ax, frame):
    feature_rates = (
        frame[["rrna_16S_score", "rrna_23S_score", "rrna_5S_score", "trna_ge_18"]]
        .mean()
        .rename(
            {
                "rrna_16S_score": "16S",
                "rrna_23S_score": "23S",
                "rrna_5S_score": "5S",
                "trna_ge_18": "tRNA>=18",
            }
        )
        .sort_values(ascending=False)
    )
    feature_df = feature_rates.rename_axis("feature").reset_index(name="value")
    sns.barplot(
        data=feature_df,
        x="feature",
        y="value",
        hue="feature",
        palette=["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3"],
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean recoverability contribution")
    ax.set_title("Hallmark feature contributions")
    for index, value in enumerate(feature_df["value"]):
        ax.text(index, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)


def plot_recovery_patterns(ax, frame):
    features = frame[["has_16S", "has_23S", "has_5S", "trna_ge_18"]].copy()
    patterns = (
        frame["recovery_pattern_label"]
        .value_counts()
        .rename_axis("recovery_pattern_label")
        .reset_index(name="count")
        .head(8)
    )

    if patterns.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No recovery patterns available", ha="center", va="center")
        return

    heat_rows = []
    labels = []
    ranks = []
    for _, row in patterns.iterrows():
        label = row["recovery_pattern_label"]
        mask = frame["recovery_pattern_label"] == label
        subset = features.loc[mask]
        counts = subset.sum(axis=0).tolist()
        heat_rows.append(counts)
        labels.append(label)
        ranks.append(int(subset.iloc[0].sum()))

    heat_df = pd.DataFrame(
        heat_rows,
        columns=["16S", "23S", "5S", "tRNA>=18"],
        index=labels,
    )
    heat_df["quality_rank"] = ranks
    heat_df["pattern_count"] = patterns.set_index("recovery_pattern_label").loc[
        heat_df.index, "count"
    ].values
    heat_df = heat_df.sort_values(
        by=["quality_rank", "pattern_count"],
        ascending=[False, False],
    ).drop(columns=["quality_rank", "pattern_count"])
    sns.heatmap(
        heat_df,
        cmap="Greys",
        cbar=True,
        cbar_kws={"label": "Genome count"},
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Top recovery patterns (genome counts)")


def choose_group_column(frame, requested_group):
    if requested_group and requested_group in frame.columns:
        return requested_group
    for fallback in ["Phylum", "Domain", "Class"]:
        if fallback in frame.columns:
            return fallback
    return None


def plot_group_panel(ax, frame, group_column, top_n):
    if not group_column or group_column not in frame.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "No grouping column available", ha="center", va="center")
        return None

    subset = frame.copy()
    subset["__group_label"] = resolve_grouping_series(subset, group_column)
    subset = subset.dropna(subset=["__group_label"])
    if subset.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No grouping data available", ha="center", va="center")
        return None

    counts = subset["__group_label"].value_counts()
    total_groups = int(len(counts))
    keep = counts.head(top_n).index
    shown_groups = int(len(keep))
    subset = subset.loc[subset["__group_label"].isin(keep)].copy()
    order = (
        subset.groupby("__group_label")[SCORE_COLUMN]
        .median()
        .sort_values(ascending=False)
        .index
    )
    sns.boxplot(
        data=subset,
        x="__group_label",
        y=SCORE_COLUMN,
        order=order,
        color="#bdbdbd",
        fliersize=1.5,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=subset,
        x="__group_label",
        y=SCORE_COLUMN,
        order=order,
        color="#1a1a1a",
        size=2,
        alpha=0.35,
        ax=ax,
    )
    ax.set_xlabel(group_column)
    ax.set_ylabel(SCORE_LABEL)
    ax.set_title(f"qscore by {group_column} ({shown_groups}/{total_groups})")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    return subset


def facet_layout(n_panels, ncols=3, width=4.8, height=4.2):
    ensure_plotting()
    ncols = max(1, min(ncols, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * width, nrows * height),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def plot_compare_threshold_facets(frame, compare_column, output_base):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    fig, axes = facet_layout(len(order), ncols=2, width=5.4, height=4.2)
    used_axes = []
    scatter = None
    for ax, category in zip(axes, order):
        subset = compare_df.loc[compare_df[compare_column] == category]
        scatter = ax.scatter(
            subset["Completeness"],
            subset["Contamination"],
            c=subset[SCORE_COLUMN],
            cmap="Greys",
            s=28,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )
        add_quality_thresholds(ax)
        ax.set_title(f"{category}\n(n={len(subset)})", fontsize=10)
        ax.set_xlabel("Completeness (%)")
        ax.set_ylabel("Contamination (%)")
        used_axes.append(ax)
    for ax in axes[len(order):]:
        ax.axis("off")
    add_qscore_colorbar(fig, scatter, cax_rect=[0.84, 0.2, 0.02, 0.58])
    fig.suptitle(f"Threshold landscape by {compare_column}", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, RIGHT_MARGIN, 0.95])
    save_figure(fig, output_base + "_threshold_facets")


def plot_compare_component_facets(frame, compare_column, output_base):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    fig, axes = facet_layout(len(order), ncols=2, width=5.4, height=4.2)
    used_axes = []
    scatter = None
    for ax, category in zip(axes, order):
        subset = compare_df.loc[compare_df[compare_column] == category]
        scatter = ax.scatter(
            subset["integrity_score"],
            subset["recoverability_score"],
            c=subset[SCORE_COLUMN],
            cmap="Greys",
            s=28,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{category}\n(n={len(subset)})", fontsize=10)
        ax.set_xlabel("Integrity (I)")
        ax.set_ylabel("Recoverability (R)")
        used_axes.append(ax)
    for ax in axes[len(order):]:
        ax.axis("off")
    add_qscore_colorbar(fig, scatter, cax_rect=[0.84, 0.2, 0.02, 0.58])
    fig.suptitle(f"Integrity vs recoverability by {compare_column}", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, RIGHT_MARGIN, 0.95])
    save_figure(fig, output_base + "_component_facets")


def plot_compare_qscore(ax, frame, compare_column):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    sns.boxplot(
        data=compare_df,
        x=compare_column,
        y=SCORE_COLUMN,
        order=order,
        color="#bdbdbd",
        fliersize=1.5,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=compare_df,
        x=compare_column,
        y=SCORE_COLUMN,
        order=order,
        color="#1a1a1a",
        size=2,
        alpha=0.3,
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel(SCORE_LABEL)
    ax.set_title(f"qscore by {compare_column}")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)


def plot_compare_hallmarks(ax, frame, compare_column):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    feature_df = (
        compare_df.groupby(compare_column)[
            ["rrna_16S_score", "rrna_23S_score", "rrna_5S_score", "trna_ge_18"]
        ]
        .mean()
        .reindex(order)
        .rename(
            columns={
                "rrna_16S_score": "16S",
                "rrna_23S_score": "23S",
                "rrna_5S_score": "5S",
                "trna_ge_18": "tRNA>=18",
            }
        )
        .reset_index()
        .melt(id_vars=compare_column, var_name="feature", value_name="value")
    )
    sns.barplot(
        data=feature_df,
        x=compare_column,
        y="value",
        hue="feature",
        order=order,
        hue_order=["16S", "23S", "5S", "tRNA>=18"],
        palette=["#1a1a1a", "#4d4d4d", "#808080", "#b3b3b3"],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel(compare_column)
    ax.set_ylabel("Mean contribution")
    ax.set_title(f"Hallmark contributions by {compare_column}")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    place_axis_legend_right(ax, title="Feature")


def plot_compare_recovery_patterns(ax, frame, compare_column):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    top_patterns = (
        compare_df["recovery_pattern_label"]
        .value_counts()
        .head(8)
        .index
    )
    heat_df = (
        compare_df.loc[compare_df["recovery_pattern_label"].isin(top_patterns)]
        .groupby(["recovery_pattern_label", compare_column])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top_patterns, columns=order, fill_value=0)
    )
    pattern_rank = (
        compare_df.loc[compare_df["recovery_pattern_label"].isin(top_patterns), [
            "recovery_pattern_label",
            "recovered_feature_count",
        ]]
        .drop_duplicates("recovery_pattern_label")
        .set_index("recovery_pattern_label")["recovered_feature_count"]
    )
    heat_df["quality_rank"] = pattern_rank.reindex(heat_df.index).fillna(0)
    heat_df["pattern_total"] = heat_df.drop(columns=["quality_rank"], errors="ignore").sum(axis=1)
    heat_df = heat_df.sort_values(
        by=["quality_rank", "pattern_total"],
        ascending=[False, False],
    ).drop(columns=["quality_rank", "pattern_total"])
    sns.heatmap(
        heat_df,
        cmap="Greys",
        cbar=True,
        cbar_kws={"label": "Genome count"},
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f",
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel("")
    ax.set_title(f"Recovery patterns by {compare_column}")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)


def plot_compare_metric_panels(frame, compare_column, output_base):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    metric_info = [
        ("Completeness", "Completeness (%)"),
        ("Contamination", "Contamination (%)"),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=False)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metric_info):
        sns.boxplot(
            data=compare_df,
            x=compare_column,
            y=metric,
            order=order,
            color="#bdbdbd",
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )
        sns.stripplot(
            data=compare_df,
            x=compare_column,
            y=metric,
            order=order,
            color="#1a1a1a",
            size=1.8,
            alpha=0.25,
            ax=ax,
        )
        ax.set_xlabel(compare_column)
        ax.set_ylabel(label)
        ax.set_title(label)
        style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    fig.suptitle(f"Assembly and recoverability metrics by {compare_column}", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base + "_metric_panels")


def plot_compare_taxonomy_heatmap(ax, frame, compare_column, group_column=None, top_n_groups=12, exclude_unclassified=False):
    selected_group = choose_group_column(frame, group_column)
    if not selected_group or selected_group not in frame.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "No taxonomy grouping available", ha="center", va="center")
        return False

    compare_df = prepare_compare_frame(frame, compare_column).copy()
    compare_df["__taxonomy_label"] = resolve_grouping_series(compare_df, selected_group)
    if exclude_unclassified:
        compare_df = compare_df.loc[
            ~compare_df["__taxonomy_label"].astype(str).str.startswith("unclassified_")
        ].copy()
        if compare_df.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No classified taxonomy groups available", ha="center", va="center")
            return False
    compare_order = category_order(compare_df, compare_column)
    tax_counts = (
        compare_df["__taxonomy_label"]
        .dropna()
        .value_counts()
    )
    total_groups = int(len(tax_counts))
    taxa = tax_counts.head(top_n_groups).index
    shown_groups = int(len(taxa))
    heat_df = (
        compare_df.loc[compare_df["__taxonomy_label"].isin(taxa)]
        .groupby(["__taxonomy_label", compare_column])[SCORE_COLUMN]
        .median()
        .unstack(fill_value=np.nan)
        .reindex(index=taxa, columns=compare_order)
    )
    sns.heatmap(
        heat_df,
        cmap="Greys",
        cbar=True,
        cbar_kws={"label": f"Median {SCORE_LABEL}"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel(selected_group)
    ax.set_title(f"Median qscore: {selected_group} ({shown_groups}/{total_groups}) x {compare_column}")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    return True


def summarize_compare(frame, compare_column):
    compare_df = prepare_compare_frame(frame, compare_column)
    order = category_order(compare_df, compare_column)
    summary = (
        compare_df.groupby(compare_column)
        .agg(
            n_genomes=("Genome_Id", "size"),
            median_qscore=(SCORE_COLUMN, "median"),
            mean_qscore=(SCORE_COLUMN, "mean"),
            median_completeness=("Completeness", "median"),
            median_contamination=("Contamination", "median"),
            median_integrity=("integrity_score", "median"),
            median_recoverability=("recoverability_score", "median"),
            mean_16S=("rrna_16S_score", "mean"),
            mean_23S=("rrna_23S_score", "mean"),
            mean_5S=("rrna_5S_score", "mean"),
            mean_trna_ge_18=("trna_ge_18", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    return compare_df, summary


def summarize_taxonomy_quality(compare_df, compare_column, taxonomy_column=None, exclude_unclassified=False):
    order = category_order(compare_df, compare_column)
    summary = pd.DataFrame({compare_column: order})

    tier_counts = (
        compare_df.groupby([compare_column, "mimag_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=order, columns=["low", "medium", "high"], fill_value=0)
        .rename(columns={"low": "lq_genomes", "medium": "mq_genomes", "high": "hq_genomes"})
        .reset_index()
    )
    summary = summary.merge(tier_counts, on=compare_column, how="left")

    selected_taxonomy = choose_group_column(compare_df, taxonomy_column)
    if selected_taxonomy and selected_taxonomy in compare_df.columns:
        taxonomy_df = compare_df[[compare_column]].copy()
        taxonomy_df["__taxonomy_label"] = resolve_grouping_series(compare_df, selected_taxonomy)
        taxonomy_df = taxonomy_df.dropna(subset=["__taxonomy_label"])
        if exclude_unclassified:
            taxonomy_df = taxonomy_df.loc[
                ~taxonomy_df["__taxonomy_label"].astype(str).str.startswith("unclassified_")
            ].copy()
        taxonomy_presence = (
            taxonomy_df.drop_duplicates([compare_column, "__taxonomy_label"])
            .groupby("__taxonomy_label")[compare_column]
            .nunique()
        )
        taxonomy_with_scope = taxonomy_df.drop_duplicates([compare_column, "__taxonomy_label"]).copy()
        taxonomy_with_scope["taxonomy_scope"] = taxonomy_with_scope["__taxonomy_label"].map(
            taxonomy_presence
        ).map(lambda count: "unique_taxa" if count == 1 else "shared_taxa")
        taxonomy_counts = (
            taxonomy_with_scope.groupby([compare_column, "taxonomy_scope"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=order, columns=["unique_taxa", "shared_taxa"], fill_value=0)
            .reset_index()
        )
        summary = summary.merge(taxonomy_counts, on=compare_column, how="left")
        summary["taxonomy_column"] = selected_taxonomy
    else:
        summary["unique_taxa"] = np.nan
        summary["shared_taxa"] = np.nan
        summary["taxonomy_column"] = ""

    fill_columns = ["lq_genomes", "mq_genomes", "hq_genomes", "unique_taxa", "shared_taxa"]
    for column in fill_columns:
        if column in summary.columns:
            summary[column] = summary[column].fillna(0).astype(int)
    return summary, selected_taxonomy


def choose_sample_column(frame, requested_sample_column):
    if requested_sample_column and requested_sample_column in frame.columns:
        return requested_sample_column
    for fallback in ["sample", "Sample", "sample_id", "Sample_ID"]:
        if fallback in frame.columns:
            return fallback
    return None


def normalize_dedupe_token(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if "." in text:
        pieces = text.split(".")
        for end in range(len(pieces), 0, -1):
            candidate = ".".join(pieces[:end]).strip()
            if candidate and candidate.lower() not in {"nan", "none", "null"}:
                text = candidate
        text = pieces[0].strip()
    match = re.match(r"^(bin_\d+)", text)
    if match:
        return match.group(1)
    return text


def best_set_key(row):
    for column in ["SAG_ID", "Bin Id", "Genome_Id", "genome_id", "ani_record_id"]:
        token = normalize_dedupe_token(row.get(column, ""))
        if token:
            return token
    return ""


def build_best_sample_method_set_from_ani(
    compare_df,
    compare_column,
    sample_column=None,
    ani_settings=None,
    output_base=None,
):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if not selected_sample or selected_sample not in compare_df.columns:
        return None, selected_sample

    if ani_settings is None:
        return None, selected_sample

    fasta_column = getattr(ani_settings, "ani_fasta_column", None)
    genome_dir = getattr(ani_settings, "ani_genome_dir", None)
    fasta_exts = split_extensions(getattr(ani_settings, "ani_fasta_exts", ".fasta,.fa,.fna,.fasta.gz,.fa.gz,.fna.gz"))
    ani_threads = int(getattr(ani_settings, "ani_threads", 1))
    ani_threshold = float(getattr(ani_settings, "ani_threshold", 95.0))
    ani_af_threshold = float(getattr(ani_settings, "ani_af_threshold", 0.5))
    existing_results = getattr(ani_settings, "ani_results", None)

    try:
        working = resolve_fasta_paths(
            compare_df.copy(),
            fasta_column=fasta_column,
            genome_dir=genome_dir,
            fasta_exts=fasta_exts,
        )
    except Exception as exc:
        warnings.warn(
            f"ANI-based best-set selection fallback: could not resolve FASTA paths ({exc})",
            RuntimeWarning,
        )
        return None, selected_sample

    if not existing_results and output_base:
        reuse_path = output_base + "_fastani_fastani_raw.tsv"
        if os.path.exists(reuse_path):
            existing_results = reuse_path

    ani_run_base = (output_base + "_bestset_fastani") if output_base else "bestset_fastani"
    try:
        raw_output = run_fastani(
            working,
            output_base=ani_run_base,
            threads=ani_threads,
            existing_results=existing_results,
        )
        pair_summary = load_fastani_pairs(raw_output, working)
    except Exception as exc:
        warnings.warn(
            f"ANI-based best-set selection fallback: FastANI failed ({exc})",
            RuntimeWarning,
        )
        return None, selected_sample

    working[selected_sample] = working[selected_sample].fillna("").astype(str).str.strip()
    working[compare_column] = working[compare_column].fillna("").astype(str).str.strip()
    working = working.loc[
        working[selected_sample].ne("")
        & working[compare_column].ne("")
        & working["ani_record_id"].astype(str).ne("")
    ].copy()
    if working.empty:
        return None, selected_sample

    if pair_summary.empty:
        pair_summary = pd.DataFrame(
            columns=[
                "record_a",
                "record_b",
                "ani_mean",
                "af_min",
                "passes_threshold",
            ]
        )
    else:
        sample_map = dict(
            zip(working["ani_record_id"].astype(str), working[selected_sample].astype(str))
        )
        category_map = dict(
            zip(working["ani_record_id"].astype(str), working[compare_column].astype(str))
        )
        pair_summary = pair_summary.copy()
        pair_summary["record_a"] = pair_summary["record_a"].astype(str)
        pair_summary["record_b"] = pair_summary["record_b"].astype(str)
        pair_summary["sample_a"] = pair_summary["record_a"].map(sample_map)
        pair_summary["sample_b"] = pair_summary["record_b"].map(sample_map)
        pair_summary["category_a"] = pair_summary["record_a"].map(category_map)
        pair_summary["category_b"] = pair_summary["record_b"].map(category_map)
        pair_summary = pair_summary.dropna(
            subset=["sample_a", "sample_b", "category_a", "category_b"]
        ).copy()
        pair_summary = pair_summary.loc[
            pair_summary["sample_a"].astype(str).eq(pair_summary["sample_b"].astype(str))
            & pair_summary["category_a"].astype(str).eq(pair_summary["category_b"].astype(str))
        ].copy()
        pair_summary["passes_threshold"] = (
            pd.to_numeric(pair_summary["ani_mean"], errors="coerce").ge(ani_threshold)
            & pd.to_numeric(pair_summary["af_min"], errors="coerce").ge(ani_af_threshold)
        )

    record_to_component = {}
    for (sample_value, method_value), group in working.groupby([selected_sample, compare_column], sort=False):
        subset = group.copy()
        subset_records = set(subset["ani_record_id"].astype(str).tolist())
        if pair_summary.empty:
            subset_pairs = pair_summary
        else:
            subset_pairs = pair_summary.loc[
                pair_summary["record_a"].astype(str).isin(subset_records)
                & pair_summary["record_b"].astype(str).isin(subset_records)
                & pair_summary["sample_a"].astype(str).eq(str(sample_value))
                & pair_summary["category_a"].astype(str).eq(str(method_value))
            ].copy()
        local_component_map = component_map_from_pairs(subset, subset_pairs)
        for record_id, local_component in local_component_map.items():
            record_to_component[str(record_id)] = (
                f"{sample_value}|{method_value}|{local_component}"
            )

    working["ani_component_id"] = working["ani_record_id"].astype(str).map(record_to_component)
    missing_component = working["ani_component_id"].isna()
    if missing_component.any():
        working.loc[missing_component, "ani_component_id"] = (
            working.loc[missing_component, selected_sample].astype(str)
            + "|"
            + working.loc[missing_component, compare_column].astype(str)
            + "|singleton|"
            + working.loc[missing_component, "ani_record_id"].astype(str)
        )

    tier_rank = {"high": 0, "medium": 1, "low": 2}
    working["__mimag_rank"] = (
        working["mimag_tier"].astype(str).str.lower().map(tier_rank).fillna(3).astype(int)
    )
    sort_specs = [
        (selected_sample, True),
        (compare_column, True),
        ("ani_component_id", True),
        ("__mimag_rank", True),
        ("integrity_score", False),
        ("recoverability_score", False),
        ("qscore", False),
        ("Completeness", False),
        ("Contamination", True),
        ("Genome_Id", True),
        ("Bin Id", True),
    ]
    sort_columns = [column for column, _ in sort_specs if column in working.columns]
    sort_ascending = [ascending for column, ascending in sort_specs if column in working.columns]
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=sort_ascending, kind="mergesort")

    selected = (
        working.groupby([selected_sample, compare_column, "ani_component_id"], as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected["best_set_key"] = selected["ani_component_id"].astype(str)
    selected["best_set_strategy"] = "ani_component"
    selected["best_selection_metric"] = "ani_component>mimag_tier>integrity>recoverability>qscore"
    selected = selected.drop(columns=["__mimag_rank"], errors="ignore")
    return selected, selected_sample


def build_best_sample_method_set(compare_df, compare_column, sample_column=None, ani_settings=None, output_base=None):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if not selected_sample or selected_sample not in compare_df.columns:
        return None, None

    ani_selected, ani_sample_column = build_best_sample_method_set_from_ani(
        compare_df=compare_df,
        compare_column=compare_column,
        sample_column=sample_column,
        ani_settings=ani_settings,
        output_base=output_base,
    )
    if ani_selected is not None and not ani_selected.empty:
        return ani_selected, ani_sample_column

    working = compare_df.copy()
    working[selected_sample] = working[selected_sample].astype(str).str.strip()
    working[compare_column] = working[compare_column].astype(str).str.strip()
    working = working.loc[
        working[selected_sample].ne("")
        & working[compare_column].ne("")
    ].copy()
    if working.empty:
        return None, selected_sample

    working["best_set_key"] = working.apply(best_set_key, axis=1)
    fallback_mask = working["best_set_key"].astype(str).eq("")
    if fallback_mask.any():
        working.loc[fallback_mask, "best_set_key"] = (
            working.loc[fallback_mask, "Genome_Id"].astype(str).str.strip()
        )
    working = working.loc[working["best_set_key"].astype(str).ne("")].copy()
    if working.empty:
        return None, selected_sample

    tier_rank = {"high": 0, "medium": 1, "low": 2}
    working["__mimag_rank"] = (
        working["mimag_tier"].astype(str).str.lower().map(tier_rank).fillna(3).astype(int)
    )
    sort_specs = [
        (selected_sample, True),
        (compare_column, True),
        ("best_set_key", True),
        ("__mimag_rank", True),
        ("integrity_score", False),
        ("recoverability_score", False),
        ("qscore", False),
        ("Completeness", False),
        ("Contamination", True),
        ("Genome_Id", True),
        ("Bin Id", True),
    ]
    sort_columns = [column for column, _ in sort_specs if column in working.columns]
    sort_ascending = [ascending for column, ascending in sort_specs if column in working.columns]
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=sort_ascending, kind="mergesort")

    selected = (
        working.groupby([selected_sample, compare_column, "best_set_key"], as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected["best_set_strategy"] = "label_key"
    selected["best_selection_metric"] = "mimag_tier>integrity>recoverability>qscore"
    selected = selected.drop(columns=["__mimag_rank"], errors="ignore")
    return selected, selected_sample


def summarize_sample_method_counts(compare_df, compare_column, sample_column=None):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if not selected_sample:
        return None, None

    sample_df = compare_df[[selected_sample, compare_column, "mimag_tier"]].copy()
    sample_df[selected_sample] = sample_df[selected_sample].astype(str).str.strip()
    sample_df = sample_df.loc[sample_df[selected_sample] != ""].copy()
    if sample_df.empty:
        return None, selected_sample

    counts = (
        sample_df.groupby([selected_sample, compare_column, "mimag_tier"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"high": "hq_bins", "medium": "mq_bins", "low": "lq_bins"})
        .reset_index()
    )
    for column in ["hq_bins", "mq_bins", "lq_bins"]:
        if column not in counts.columns:
            counts[column] = 0
    counts["total_bins"] = counts["hq_bins"] + counts["mq_bins"] + counts["lq_bins"]
    counts["hqmq_bins"] = counts["hq_bins"] + counts["mq_bins"]
    return counts, selected_sample


def run_sample_count_tests(sample_counts, compare_column, sample_column):
    if sample_counts is None or sample_counts.empty:
        return None

    try:
        from scipy import stats
    except ImportError:
        warnings.warn(
            "scipy is not installed; skipping sample-aware category count tests.",
            RuntimeWarning,
        )
        return None

    categories = ordered_unique_categories(sample_counts[compare_column].astype(str).drop_duplicates().tolist())
    metrics = ["total_bins", "hq_bins", "mq_bins", "lq_bins", "hqmq_bins"]
    results = []

    for metric in metrics:
        full_pivot = (
            sample_counts.pivot_table(
                index=sample_column,
                columns=compare_column,
                values=metric,
                aggfunc="first",
            )
        )
        full_pivot.columns = full_pivot.columns.astype(str)
        available_categories = [category for category in categories if category in full_pivot.columns]
        if len(available_categories) < 2:
            continue

        for left, right in itertools.combinations(available_categories, 2):
            pair_pivot = full_pivot[[left, right]].dropna()
            if pair_pivot.empty:
                continue
            try:
                stat, pvalue = stats.wilcoxon(
                    pair_pivot[left],
                    pair_pivot[right],
                    alternative="two-sided",
                )
            except ValueError:
                stat, pvalue = np.nan, np.nan
            results.append(
                {
                    "test": "Wilcoxon signed-rank",
                    "metric": metric,
                    "sample_column": sample_column,
                    "group_column": compare_column,
                    "group_a": left,
                    "group_b": right,
                    "n_samples": int(len(pair_pivot)),
                    "group_a_median": float(pair_pivot[left].median()),
                    "group_b_median": float(pair_pivot[right].median()),
                    "group_a_mean": float(pair_pivot[left].mean()),
                    "group_b_mean": float(pair_pivot[right].mean()),
                    "median_difference_a_minus_b": float((pair_pivot[left] - pair_pivot[right]).median()),
                    "mean_difference_a_minus_b": float((pair_pivot[left] - pair_pivot[right]).mean()),
                    "statistic": stat,
                    "pvalue": pvalue,
                }
            )

        global_pivot = full_pivot[available_categories].dropna()
        if global_pivot.empty or global_pivot.shape[1] < 2:
            continue
        try:
            stat, pvalue = stats.friedmanchisquare(*[global_pivot[column] for column in global_pivot.columns])
        except ValueError:
            stat, pvalue = np.nan, np.nan
        results.append(
            {
                "test": "Friedman",
                "metric": metric,
                "sample_column": sample_column,
                "group_column": compare_column,
                "n_samples": int(len(global_pivot)),
                "n_groups": int(global_pivot.shape[1]),
                "groups": ";".join(map(str, global_pivot.columns.tolist())),
                "statistic": stat,
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(results) if results else None


def benjamini_hochberg_adjust(series):
    values = pd.to_numeric(series, errors="coerce").values.astype(float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(values)
    if not valid_mask.any():
        return adjusted
    valid_values = values[valid_mask]
    n_values = len(valid_values)
    order = np.argsort(valid_values)
    ordered = valid_values[order]
    scaled = ordered * n_values / np.arange(1, n_values + 1, dtype=float)
    monotonic = np.minimum.accumulate(scaled[::-1])[::-1]
    monotonic = np.clip(monotonic, 0.0, 1.0)
    reordered = np.empty_like(monotonic)
    reordered[order] = monotonic
    adjusted[valid_mask] = reordered
    return adjusted


def run_method_significance_tests(compare_df, compare_column, sample_counts=None, sample_count_stats=None, sample_column=None):
    if compare_df.empty or compare_column not in compare_df.columns:
        return None

    try:
        from scipy import stats
    except ImportError:
        warnings.warn(
            "scipy is not installed; skipping consolidated method significance tests.",
            RuntimeWarning,
        )
        return None

    categories = category_order(compare_df, compare_column)
    if len(categories) < 2:
        return None

    rows = []
    per_genome_metrics = [
        (SCORE_COLUMN, SCORE_LABEL),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
        ("mimag_quality_index", "MIMAG quality index"),
        ("Completeness", "Completeness (%)"),
        ("Contamination", "Contamination (%)"),
    ]
    for metric, metric_label in per_genome_metrics:
        if metric not in compare_df.columns:
            continue
        metric_df = compare_df[[compare_column, metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric])
        if metric_df.empty:
            continue

        present_categories = [
            category for category in categories
            if metric_df.loc[metric_df[compare_column].astype(str).eq(category)].shape[0] > 0
        ]
        if len(present_categories) < 2:
            continue

        for group_a, group_b in itertools.combinations(present_categories, 2):
            values_a = metric_df.loc[metric_df[compare_column].astype(str).eq(group_a), metric].astype(float)
            values_b = metric_df.loc[metric_df[compare_column].astype(str).eq(group_b), metric].astype(float)
            if values_a.empty or values_b.empty:
                continue
            try:
                statistic, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            median_a = float(values_a.median())
            median_b = float(values_b.median())
            mean_a = float(values_a.mean())
            mean_b = float(values_b.mean())
            median_delta = median_a - median_b
            mean_delta = mean_a - mean_b
            winner = group_a if median_delta > 0 else group_b if median_delta < 0 else "tie"
            rows.append(
                {
                    "analysis_scope": "per_genome_metric_pairwise",
                    "test": "Mann-Whitney U",
                    "metric": metric,
                    "metric_label": metric_label,
                    "group_column": compare_column,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "group_a_median": median_a,
                    "group_b_median": median_b,
                    "group_a_mean": mean_a,
                    "group_b_mean": mean_b,
                    "median_difference_a_minus_b": median_delta,
                    "mean_difference_a_minus_b": mean_delta,
                    "winner_by_median": winner,
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

        grouped_values = [
            metric_df.loc[metric_df[compare_column].astype(str).eq(category), metric].astype(float).values
            for category in present_categories
        ]
        if len(grouped_values) >= 2:
            try:
                statistic, pvalue = stats.kruskal(*grouped_values)
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            rows.append(
                {
                    "analysis_scope": "per_genome_metric_global",
                    "test": "Kruskal-Wallis",
                    "metric": metric,
                    "metric_label": metric_label,
                    "group_column": compare_column,
                    "n_groups": int(len(present_categories)),
                    "groups": ";".join(map(str, present_categories)),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

    if sample_count_stats is None and sample_counts is not None and sample_column:
        sample_count_stats = run_sample_count_tests(sample_counts, compare_column, sample_column)
    if sample_count_stats is not None and not sample_count_stats.empty:
        count_stats = sample_count_stats.copy()
        count_stats["analysis_scope"] = np.where(
            count_stats["test"].astype(str).str.contains("Friedman", case=False, na=False),
            "sample_count_global",
            "sample_count_pairwise",
        )
        count_stats["metric_label"] = count_stats["metric"].astype(str)
        if "group_a_median" in count_stats.columns and "group_b_median" in count_stats.columns:
            group_a_series = (
                count_stats["group_a"].astype(str)
                if "group_a" in count_stats.columns
                else pd.Series([""] * len(count_stats), index=count_stats.index, dtype=str)
            )
            group_b_series = (
                count_stats["group_b"].astype(str)
                if "group_b" in count_stats.columns
                else pd.Series([""] * len(count_stats), index=count_stats.index, dtype=str)
            )
            count_stats["winner_by_median"] = np.where(
                pd.to_numeric(count_stats["group_a_median"], errors="coerce")
                > pd.to_numeric(count_stats["group_b_median"], errors="coerce"),
                group_a_series,
                np.where(
                    pd.to_numeric(count_stats["group_a_median"], errors="coerce")
                    < pd.to_numeric(count_stats["group_b_median"], errors="coerce"),
                    group_b_series,
                    "tie",
                ),
            )
        rows.extend(count_stats.to_dict("records"))

    if not rows:
        return None

    results = pd.DataFrame(rows)
    if "pvalue" in results.columns:
        results["qvalue_bh"] = benjamini_hochberg_adjust(results["pvalue"])
        results["significant_p05"] = pd.to_numeric(results["pvalue"], errors="coerce").lt(0.05)
        results["significant_q05"] = pd.to_numeric(results["qvalue_bh"], errors="coerce").lt(0.05)
    return results


def plot_sample_method_counts(sample_counts, compare_column, sample_column, output_base):
    ensure_plotting()
    if sample_counts is None or sample_counts.empty:
        return False

    order = ordered_unique_categories(sample_counts[compare_column].astype(str).tolist())
    total_counts = (
        sample_counts.groupby(compare_column)[
            ["total_bins", "hq_bins", "mq_bins", "lq_bins", "hqmq_bins"]
        ]
        .sum()
        .reindex(order)
        .reset_index()
    )
    metrics = [
        ("total_bins", "Total bins"),
        ("hq_bins", "HQ bins"),
        ("mq_bins", "MQ bins"),
        ("lq_bins", "LQ bins"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=False)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, metrics):
        sns.barplot(
            data=total_counts,
            x=compare_column,
            y=metric,
            order=order,
            hue=compare_column,
            palette=grayscale_palette(len(order), start=0.2, stop=0.75),
            dodge=False,
            legend=False,
            edgecolor="black",
            ax=ax,
        )
        ax.set_xlabel(compare_column)
        ax.set_ylabel(label)
        ax.set_title(f"{label} totals by {compare_column}")
        style_long_ticklabels(ax, axis="x", rotation=90, size=8)
        for index, value in enumerate(total_counts[metric].tolist()):
            ax.text(index, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        f"Total bin counts by {compare_column} (sample-aware stats reported separately)",
        fontsize=16,
        y=0.99,
    )
    apply_tight_layout(fig, rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base + "_sample_count_summary")
    return True


def summarize_sample_breakout(compare_df, compare_column, sample_column=None):
    selected_sample = choose_sample_column(compare_df, sample_column)
    if not selected_sample:
        return None, None

    metric_columns = [
        compare_column,
        selected_sample,
        "mimag_tier",
        SCORE_COLUMN,
        "Completeness",
        "Contamination",
        "integrity_score",
        "recoverability_score",
    ]
    sample_df = compare_df[metric_columns].copy()
    sample_df[selected_sample] = sample_df[selected_sample].astype(str).str.strip()
    sample_df = sample_df.loc[sample_df[selected_sample] != ""].copy()
    if sample_df.empty:
        return None, selected_sample

    summary = (
        sample_df.groupby([selected_sample, compare_column])
        .agg(
            n_genomes=("mimag_tier", "size"),
            median_qscore=(SCORE_COLUMN, "median"),
            median_completeness=("Completeness", "median"),
            median_contamination=("Contamination", "median"),
            median_integrity=("integrity_score", "median"),
            median_recoverability=("recoverability_score", "median"),
        )
        .reset_index()
    )

    tier_counts = (
        sample_df.groupby([selected_sample, compare_column, "mimag_tier"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"high": "hq_bins", "medium": "mq_bins", "low": "lq_bins"})
        .reset_index()
    )
    for column in ["hq_bins", "mq_bins", "lq_bins"]:
        if column not in tier_counts.columns:
            tier_counts[column] = 0
    summary = summary.merge(
        tier_counts[[selected_sample, compare_column, "hq_bins", "mq_bins", "lq_bins"]],
        on=[selected_sample, compare_column],
        how="left",
    )
    summary["total_bins"] = summary["hq_bins"] + summary["mq_bins"] + summary["lq_bins"]
    return summary, selected_sample


def apply_heatmap_text_contrast(ax, matrix):
    if matrix.empty:
        return
    values = matrix.to_numpy(dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        for text in ax.texts:
            text.set_color("#1a1a1a")
        return
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    denominator = vmax - vmin
    flat_values = values.flatten(order="C")
    for text, value in zip(ax.texts, flat_values):
        if not np.isfinite(value):
            text.set_color("#1a1a1a")
            continue
        normalized = 0.0 if denominator == 0 else (float(value) - vmin) / denominator
        text.set_color("white" if normalized >= 0.6 else "#1a1a1a")


def plot_sample_breakout_dashboard(sample_summary, compare_column, sample_column, output_base):
    ensure_plotting()
    if sample_summary is None or sample_summary.empty:
        return False

    category_order = ordered_unique_categories(sample_summary[compare_column].astype(str).tolist())
    sample_order = (
        sample_summary.groupby(sample_column)["total_bins"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    metrics = [
        ("total_bins", "Total bins", ".0f"),
        ("hq_bins", "HQ bins", ".0f"),
        ("mq_bins", "MQ bins", ".0f"),
        ("lq_bins", "LQ bins", ".0f"),
        ("median_qscore", "Median qscore", ".2f"),
        ("median_integrity", "Median integrity", ".2f"),
        ("median_recoverability", "Median recoverability", ".2f"),
        ("median_completeness", "Median completeness", ".1f"),
        ("median_contamination", "Median contamination", ".2f"),
    ]

    figure_width = max(15, len(category_order) * 1.2 + 8)
    figure_height = max(12, min(28, len(sample_order) * 0.32 + 9))
    fig, axes = plt.subplots(3, 3, figsize=(figure_width, figure_height), sharex=False, sharey=False)
    axes = axes.ravel()

    for ax, (metric, label, fmt) in zip(axes, metrics):
        matrix = (
            sample_summary.pivot_table(
                index=sample_column,
                columns=compare_column,
                values=metric,
                aggfunc="first",
            )
            .reindex(index=sample_order, columns=category_order)
            .fillna(0)
        )
        sns.heatmap(
            matrix,
            cmap="Greys",
            linewidths=0.5,
            linecolor="white",
            annot=True,
            fmt=fmt,
            cbar=False,
            annot_kws={"fontsize": 7},
            ax=ax,
        )
        apply_heatmap_text_contrast(ax, matrix)
        ax.set_xlabel(compare_column)
        ax.set_ylabel(sample_column)
        ax.set_title(label, pad=10)
        style_long_ticklabels(ax, axis="x", rotation=90, size=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(f"Per-sample method breakout by {compare_column}", fontsize=16, y=0.995)
    apply_tight_layout(fig, rect=[0, 0, 1, 0.975])
    save_figure(fig, output_base + "_sample_breakout_dashboard")
    return True


def make_annotation_color_map(values, palette_name="husl"):
    unique_values = [value for value in pd.Series(values).astype(str).drop_duplicates().tolist() if value != ""]
    if not unique_values:
        return {}
    from matplotlib.colors import to_hex

    palette = sns.color_palette(palette_name, n_colors=len(unique_values))
    return {value: to_hex(palette[index]) for index, value in enumerate(unique_values)}


def build_taxonomy_clustermap_inputs(frame, rank):
    sample_column = choose_sample_column(frame, "sample")
    if (
        not sample_column
        or sample_column not in frame.columns
        or "category" not in frame.columns
        or rank not in frame.columns
    ):
        return None, None, None

    working = frame.copy()
    working[sample_column] = clean_group_series(working[sample_column])
    working["category"] = clean_group_series(working["category"])
    working = working.dropna(subset=[sample_column, "category", SCORE_COLUMN]).copy()
    if working.empty:
        return None, None, None

    working["sample_category_label"] = (
        working[sample_column].astype(str) + " | " + working["category"].astype(str)
    )

    rank_df = working[[sample_column, "category", "sample_category_label", rank, SCORE_COLUMN]].copy()
    rank_df[rank] = clean_group_series(rank_df[rank])
    rank_df = rank_df.dropna(subset=[rank]).copy()
    if rank_df.empty:
        return None, None, None

    rank_df = rank_df.loc[
        ~rank_df[rank].astype(str).str.startswith("unclassified_")
    ].copy()
    if rank_df.empty:
        return None, None, None

    rank_df["taxonomy_rank"] = rank
    rank_df["taxonomy_value"] = rank_df[rank].astype(str)
    rank_df["taxonomy_label"] = rank_df["taxonomy_rank"] + ": " + rank_df["taxonomy_value"]
    long_df = rank_df[["sample_category_label", "taxonomy_rank", "taxonomy_value", "taxonomy_label", SCORE_COLUMN]].copy()
    matrix = (
        long_df.groupby(["taxonomy_label", "sample_category_label"])[SCORE_COLUMN]
        .mean()
        .unstack(fill_value=np.nan)
    )
    matrix = matrix.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if matrix.empty or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return None, None, None

    row_meta = (
        long_df[["taxonomy_label", "taxonomy_rank", "taxonomy_value"]]
        .drop_duplicates()
        .set_index("taxonomy_label")
        .reindex(matrix.index)
        .reset_index()
    )
    column_meta = (
        working[[sample_column, "category", "sample_category_label"]]
        .drop_duplicates()
        .set_index("sample_category_label")
        .reindex(matrix.columns)
        .reset_index()
        .rename(columns={sample_column: "sample"})
    )
    column_meta["sample"] = column_meta["sample"].astype(str).str.strip()
    column_meta["category"] = column_meta["category"].astype(str).str.strip()
    column_meta["__category_rank"] = column_meta["category"].map(preferred_method_rank).fillna(999).astype(int)
    column_meta = column_meta.sort_values(
        by=["sample", "__category_rank", "category", "sample_category_label"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ordered_columns = column_meta["sample_category_label"].astype(str).tolist()
    matrix = matrix.reindex(columns=ordered_columns)
    column_meta = column_meta.drop(columns=["__category_rank"], errors="ignore")
    return matrix, row_meta, column_meta


def save_taxonomy_clustermap(frame, output_base):
    ensure_plotting()
    wrote_files = []
    sample_color_map = {}
    category_color_map = {}

    sample_column = choose_sample_column(frame, "sample")
    if sample_column and sample_column in frame.columns:
        sample_values = (
            clean_group_series(frame[sample_column])
            .dropna()
            .astype(str)
            .str.strip()
        )
        sample_values = sample_values.loc[sample_values.ne("")]
        sample_color_map = make_annotation_color_map(sample_values.tolist(), palette_name="tab20")
    if "category" in frame.columns:
        category_values = (
            clean_group_series(frame["category"])
            .dropna()
            .astype(str)
            .str.strip()
        )
        category_values = category_values.loc[category_values.ne("")]
        category_color_map = make_annotation_color_map(
            ordered_unique_categories(category_values.tolist()),
            palette_name="Set1",
        )

    for rank in TAXONOMY_RANKS:
        matrix, row_meta, column_meta = build_taxonomy_clustermap_inputs(frame, rank)
        if matrix is None:
            continue
        cluster_matrix = matrix.fillna(0.0)
        column_meta = (
            column_meta.set_index("sample_category_label")
            .reindex(cluster_matrix.columns)
            .reset_index()
        )

        if not sample_color_map:
            sample_color_map = make_annotation_color_map(column_meta["sample"], palette_name="tab20")
        if not category_color_map:
            category_color_map = make_annotation_color_map(column_meta["category"], palette_name="Set1")

        sample_colors = (
            column_meta["sample"]
            .astype(str)
            .map(sample_color_map)
            .fillna("#7f7f7f")
        )
        category_colors = (
            column_meta["category"]
            .astype(str)
            .map(category_color_map)
            .fillna("#1a1a1a")
        )
        # Build col_colors positionally (not label-aligned), so each color strip
        # maps exactly to the corresponding heatmap column.
        if len(sample_colors) != cluster_matrix.shape[1] or len(category_colors) != cluster_matrix.shape[1]:
            warnings.warn(
                f"Unexpected clustermap metadata length mismatch for rank={rank}; skipping.",
                RuntimeWarning,
            )
            continue
        col_colors = pd.DataFrame(
            {
                "sample": sample_colors.to_numpy(dtype=object),
                "category": category_colors.to_numpy(dtype=object),
            },
            index=cluster_matrix.columns,
        ).reindex(cluster_matrix.columns).fillna("#7f7f7f")

        figure_width = max(14, min(34, 9 + cluster_matrix.shape[1] * 0.42))
        figure_height = max(11, min(34, 6.5 + cluster_matrix.shape[0] * 0.23))
        # Keep annotation-strip rows close to heatmap cell scale.
        col_colors_ratio = min(0.06, max(0.012, 2.0 / float(max(cluster_matrix.shape[0], 2))))
        grid = sns.clustermap(
            cluster_matrix,
            cmap="Greys",
            metric="euclidean",
            method="average",
            row_cluster=True,
            col_cluster=False,
            linewidths=0.1,
            linecolor="#f2f2f2",
            figsize=(figure_width, figure_height),
            col_colors=col_colors,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": f"Mean {SCORE_LABEL}"},
            dendrogram_ratio=(0.14, 0.12),
            colors_ratio=(0.04, col_colors_ratio),
            cbar_pos=(0.80, 0.10, 0.018, 0.24),
        )
        grid.fig.subplots_adjust(left=0.06, right=0.71, top=0.92, bottom=0.10)
        grid.fig.suptitle(f"{rank} clustermap across samples and methods", fontsize=16, y=0.995)
        grid.ax_heatmap.set_xlabel("Sample | category")
        grid.ax_heatmap.set_ylabel(f"{rank} label")
        grid.ax_heatmap.tick_params(axis="x", labelrotation=90, labelsize=7, pad=1)
        grid.ax_heatmap.tick_params(axis="y", labelsize=6, pad=1)

        if hasattr(grid, "ax_col_colors") and grid.ax_col_colors is not None:
            grid.ax_col_colors.set_yticks([])
            grid.ax_col_colors.set_ylabel("Sample / Category", fontsize=8, rotation=90, labelpad=8)
            grid.ax_col_colors.tick_params(axis="x", bottom=False, labelbottom=False)
            grid.ax_col_colors.set_aspect("auto")

        present_samples = (
            column_meta["sample"].astype(str).str.strip().drop_duplicates().tolist()
        )
        present_samples = [label for label in present_samples if label in sample_color_map]
        present_categories = (
            column_meta["category"].astype(str).str.strip().drop_duplicates().tolist()
        )
        present_categories = [label for label in present_categories if label in category_color_map]
        present_categories = sorted(
            present_categories,
            key=lambda label: (preferred_method_rank(label), str(label).lower()),
        )

        sample_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=str(sample_color_map[label]),
                label=str(label),
                markersize=7,
            )
            for label in present_samples
        ]
        category_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                color=str(category_color_map[label]),
                label=str(label),
                markersize=7,
            )
            for label in present_categories
        ]

        legend_x = 0.74
        legend_top = 0.96
        legend_gap = 0.035
        sample_block_height = 0.0
        category_block_height = 0.0
        if sample_handles:
            sample_ncol = max(1, int(math.ceil(len(sample_handles) / 10.0)))
            sample_rows = int(math.ceil(len(sample_handles) / float(sample_ncol)))
            sample_block_height = 0.05 + sample_rows * 0.030
            sample_legend = grid.fig.legend(
                handles=sample_handles,
                title="Sample",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(legend_x, legend_top),
                borderaxespad=0.0,
                ncol=sample_ncol,
                fontsize=8,
                title_fontsize=9,
            )
            grid.fig.add_artist(sample_legend)
        if category_handles:
            category_anchor_y = legend_top - (sample_block_height if sample_handles else 0.0) - (legend_gap if sample_handles else 0.0)
            category_ncol = max(1, int(math.ceil(len(category_handles) / 8.0)))
            category_rows = int(math.ceil(len(category_handles) / float(category_ncol)))
            category_block_height = 0.05 + category_rows * 0.030
            category_legend = grid.fig.legend(
                handles=category_handles,
                title="Category",
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(legend_x, max(0.08, category_anchor_y)),
                borderaxespad=0.0,
                ncol=category_ncol,
                fontsize=8,
                title_fontsize=9,
            )
            grid.fig.add_artist(category_legend)
        if hasattr(grid, "cax") and grid.cax is not None:
            cbar_top = legend_top - sample_block_height
            if category_handles:
                cbar_top = cbar_top - legend_gap - category_block_height
            cbar_top = max(0.36, cbar_top - 0.03)
            cbar_bottom = 0.10
            cbar_height = max(0.22, cbar_top - cbar_bottom)
            grid.cax.set_position([0.76, cbar_bottom, 0.020, cbar_height])
            grid.cax.tick_params(labelsize=8)
            grid.cax.yaxis.set_label_position("right")
            grid.cax.yaxis.tick_right()

        rank_token = sanitize_token(rank)
        png_path = output_base + f"_taxonomy_clustermap_{rank_token}.png"
        pdf_path = output_base + f"_taxonomy_clustermap_{rank_token}.pdf"
        grid.fig.savefig(png_path, dpi=300, bbox_inches="tight")
        grid.fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(grid.fig)

        matrix_path = output_base + f"_taxonomy_clustermap_{rank_token}_matrix.tsv"
        filled_matrix_path = output_base + f"_taxonomy_clustermap_{rank_token}_matrix_filled.tsv"
        row_meta_path = output_base + f"_taxonomy_clustermap_{rank_token}_rows.tsv"
        column_meta_path = output_base + f"_taxonomy_clustermap_{rank_token}_columns.tsv"
        matrix.to_csv(matrix_path, sep="\t")
        cluster_matrix.to_csv(filled_matrix_path, sep="\t")
        row_meta.to_csv(row_meta_path, sep="\t", index=False)
        column_meta.to_csv(column_meta_path, sep="\t", index=False)
        wrote_files.extend(
            [
                matrix_path,
                filled_matrix_path,
                row_meta_path,
                column_meta_path,
                png_path,
                pdf_path,
            ]
        )
    return wrote_files


def plot_compare_taxonomy_quality_summary(ax_taxonomy, ax_quality, compare_df, compare_column, taxonomy_column=None):
    summary_df, selected_taxonomy = summarize_taxonomy_quality(compare_df, compare_column, taxonomy_column)
    order = summary_df[compare_column].tolist()

    if selected_taxonomy:
        taxonomy_plot = summary_df.set_index(compare_column)[["shared_taxa", "unique_taxa"]].rename(
            columns={"shared_taxa": "Shared", "unique_taxa": "Unique"}
        )
        taxonomy_plot.plot(
            kind="bar",
            stacked=True,
            color=["#808080", "#1a1a1a"],
            edgecolor="black",
            linewidth=0.5,
            ax=ax_taxonomy,
        )
        ax_taxonomy.set_xlabel(compare_column)
        ax_taxonomy.set_ylabel("Taxon count")
        ax_taxonomy.set_title(f"{selected_taxonomy}: shared vs unique")
        style_long_ticklabels(ax_taxonomy, axis="x", rotation=90, size=8)
        place_axis_legend_right(ax_taxonomy, title="Taxonomy scope")
    else:
        ax_taxonomy.axis("off")
        ax_taxonomy.text(0.5, 0.5, "No taxonomy column available", ha="center", va="center")

    quality_plot = summary_df.set_index(compare_column)[["hq_genomes", "mq_genomes", "lq_genomes"]].rename(
        columns={"hq_genomes": "HQ", "mq_genomes": "MQ", "lq_genomes": "LQ"}
    )
    quality_plot.plot(
        kind="bar",
        stacked=True,
        color=["#1a1a1a", "#7f7f7f", "#d9d9d9"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax_quality,
    )
    ax_quality.set_xlabel(compare_column)
    ax_quality.set_ylabel("Genome count")
    ax_quality.set_title("HQ / MQ / LQ genomes")
    style_long_ticklabels(ax_quality, axis="x", rotation=90, size=8)
    place_axis_legend_right(ax_quality, title="Quality tier")
    return summary_df, selected_taxonomy


def plot_compare_metric(ax, compare_df, compare_column, metric, label):
    order = category_order(compare_df, compare_column)
    sns.boxplot(
        data=compare_df,
        x=compare_column,
        y=metric,
        order=order,
        color="#bdbdbd",
        fliersize=1.5,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=compare_df,
        x=compare_column,
        y=metric,
        order=order,
        color="#1a1a1a",
        size=1.8,
        alpha=0.25,
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel(label)
    ax.set_title(label)
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)


def plot_compare_quality_tiers(ax, compare_df, compare_column):
    summary_df, _ = summarize_taxonomy_quality(compare_df, compare_column, taxonomy_column=None)
    if summary_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No category summary available", ha="center", va="center")
        return

    quality_plot = summary_df.set_index(compare_column)[["hq_genomes", "mq_genomes", "lq_genomes"]].rename(
        columns={"hq_genomes": "HQ", "mq_genomes": "MQ", "lq_genomes": "LQ"}
    )
    quality_plot.plot(
        kind="bar",
        stacked=True,
        color=["#1a1a1a", "#7f7f7f", "#d9d9d9"],
        edgecolor="black",
        linewidth=0.5,
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel("Genome count")
    ax.set_title("HQ / MQ / LQ genomes")
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    place_axis_legend_right(ax, title="Quality tier")


def save_grouped_atlas(frame, output_base, compare_column="category", group_column=None, top_n_groups=12):
    compare_df = prepare_compare_frame(frame, compare_column)
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))

    plot_compare_qscore(axes[0, 0], compare_df, compare_column)
    plot_compare_metric(axes[0, 1], compare_df, compare_column, "integrity_score", "Integrity")
    plot_compare_metric(axes[0, 2], compare_df, compare_column, "recoverability_score", "Recoverability")
    plot_compare_hallmarks(axes[1, 0], compare_df, compare_column)
    plot_compare_recovery_patterns(axes[1, 1], compare_df, compare_column)
    if group_column and choose_group_column(compare_df, group_column):
        plot_compare_taxonomy_heatmap(
            axes[1, 2],
            compare_df,
            compare_column,
            group_column=group_column,
            top_n_groups=top_n_groups,
            exclude_unclassified=True,
        )
    else:
        plot_compare_quality_tiers(axes[1, 2], compare_df, compare_column)

    fig.suptitle(f"Genome Quality Atlas by {compare_column}", fontsize=18, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 0.78, 0.97])
    save_figure(fig, output_base + "_atlas")


def save_compare_outputs(
    frame,
    output_base,
    compare_column,
    group_column=None,
    top_n_groups=12,
    sample_column=None,
    matched_samples_only=False,
    ani_settings=None,
):
    ensure_plotting()
    compare_df = prepare_compare_frame(frame, compare_column)
    if matched_samples_only:
        selected_sample = choose_sample_column(compare_df, sample_column)
        if not selected_sample:
            raise ValueError(
                f"--matched-samples-only requires a usable sample column for comparison column '{compare_column}'."
            )
        compare_df, selected_sample = filter_to_matched_samples(
            compare_df,
            compare_column,
            sample_column=sample_column,
        )
        if compare_df.empty:
            raise ValueError(
                f"No matched-sample rows remain for comparison column '{compare_column}'."
            )
        sample_column = selected_sample
    _, compare_summary = summarize_compare(compare_df, compare_column)
    safe_name = sanitize_token(compare_column)
    compare_base = output_base + f"_compare_{safe_name}"

    compare_summary.to_csv(compare_base + "_summary.tsv", sep="\t", index=False)
    taxonomy_quality_summary, _ = summarize_taxonomy_quality(compare_df, compare_column, group_column)
    taxonomy_quality_summary.to_csv(
        compare_base + "_taxonomy_quality_summary.tsv",
        sep="\t",
        index=False,
    )
    stats_df = run_nonparametric_test(compare_df, compare_column)
    if stats_df is not None:
        stats_df.to_csv(compare_base + "_stats.tsv", sep="\t", index=False)
    sample_counts, selected_sample = summarize_sample_method_counts(
        compare_df,
        compare_column,
        sample_column=sample_column,
    )
    sample_count_stats = run_sample_count_tests(
        sample_counts,
        compare_column,
        selected_sample,
    ) if sample_counts is not None else None
    if sample_counts is not None:
        sample_counts.to_csv(compare_base + "_sample_count_summary.tsv", sep="\t", index=False)
    if sample_count_stats is not None:
        sample_count_stats.to_csv(compare_base + "_sample_count_stats.tsv", sep="\t", index=False)
    method_significance = run_method_significance_tests(
        compare_df=compare_df,
        compare_column=compare_column,
        sample_counts=sample_counts,
        sample_count_stats=sample_count_stats,
        sample_column=selected_sample,
    )
    if method_significance is not None:
        method_significance.to_csv(compare_base + "_method_significance.tsv", sep="\t", index=False)

    best_set_df, best_set_sample_column = build_best_sample_method_set(
        compare_df=compare_df,
        compare_column=compare_column,
        sample_column=sample_column,
        ani_settings=ani_settings,
        output_base=compare_base,
    )
    best_set_significance = None
    if best_set_df is not None and not best_set_df.empty:
        best_set_df.to_csv(compare_base + "_best_sample_method_selected.tsv", sep="\t", index=False)
        best_set_counts, _ = summarize_sample_method_counts(
            best_set_df,
            compare_column,
            sample_column=best_set_sample_column,
        )
        best_set_count_stats = run_sample_count_tests(
            best_set_counts,
            compare_column,
            best_set_sample_column,
        ) if best_set_counts is not None else None
        best_set_significance = run_method_significance_tests(
            compare_df=best_set_df,
            compare_column=compare_column,
            sample_counts=best_set_counts,
            sample_count_stats=best_set_count_stats,
            sample_column=best_set_sample_column,
        )
        if best_set_significance is not None:
            best_set_significance.to_csv(
                compare_base + "_method_significance_best_sample_method.tsv",
                sep="\t",
                index=False,
            )
    sample_breakout_summary, selected_breakout_sample = summarize_sample_breakout(
        compare_df,
        compare_column,
        sample_column=sample_column,
    )
    if sample_breakout_summary is not None:
        sample_breakout_summary.to_csv(compare_base + "_sample_breakout_summary.tsv", sep="\t", index=False)

    plot_compare_threshold_facets(compare_df, compare_column, compare_base)
    plot_compare_component_facets(compare_df, compare_column, compare_base)

    fig, ax = plt.subplots(figsize=(max(7, len(category_order(compare_df, compare_column)) * 0.8), 6.5))
    plot_compare_qscore(ax, compare_df, compare_column)
    apply_tight_layout(fig)
    save_figure(fig, compare_base + "_qscore")

    fig, ax = plt.subplots(figsize=(max(8, len(category_order(compare_df, compare_column)) * 0.9), 6.5))
    plot_compare_hallmarks(ax, compare_df, compare_column)
    apply_tight_layout(fig, rect=[0, 0, RIGHT_MARGIN, 1])
    save_figure(fig, compare_base + "_hallmarks")

    fig, ax = plt.subplots(figsize=(max(8, len(category_order(compare_df, compare_column)) * 0.9), 6.5))
    plot_compare_recovery_patterns(ax, compare_df, compare_column)
    apply_tight_layout(fig, rect=[0, 0, 0.80, 1])
    save_figure(fig, compare_base + "_recovery_patterns")

    plot_compare_metric_panels(compare_df, compare_column, compare_base)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, len(category_order(compare_df, compare_column)) * 1.5), 11.5))
    plot_compare_taxonomy_quality_summary(
        axes[0],
        axes[1],
        compare_df,
        compare_column,
        taxonomy_column=group_column,
    )
    fig.suptitle(f"Taxonomy and quality tradeoffs by {compare_column}", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 0.74, 0.95])
    save_figure(fig, compare_base + "_taxonomy_quality_summary")

    wrote_sample_plot = plot_sample_method_counts(
        sample_counts,
        compare_column,
        selected_sample,
        compare_base,
    )
    wrote_sample_breakout = plot_sample_breakout_dashboard(
        sample_breakout_summary,
        compare_column,
        selected_breakout_sample,
        compare_base,
    )

    fig, ax = plt.subplots(figsize=(max(8, len(category_order(compare_df, compare_column)) * 0.9), 7))
    wrote_taxonomy = plot_compare_taxonomy_heatmap(
        ax,
        compare_df,
        compare_column,
        group_column=group_column,
        top_n_groups=top_n_groups,
    )
    if wrote_taxonomy:
        apply_tight_layout(fig, rect=[0, 0, 0.80, 1])
        save_figure(fig, compare_base + "_taxonomy_heatmap")
    else:
        plt.close(fig)

    wrote_files = [
        compare_base + "_summary.tsv",
        compare_base + "_taxonomy_quality_summary.tsv",
        compare_base + "_sample_count_summary.tsv",
        compare_base + "_threshold_facets.png",
        compare_base + "_threshold_facets.pdf",
        compare_base + "_component_facets.png",
        compare_base + "_component_facets.pdf",
        compare_base + "_qscore.png",
        compare_base + "_qscore.pdf",
        compare_base + "_taxonomy_quality_summary.png",
        compare_base + "_taxonomy_quality_summary.pdf",
        compare_base + "_sample_count_summary.png",
        compare_base + "_sample_count_summary.pdf",
        compare_base + "_sample_breakout_summary.tsv",
        compare_base + "_sample_breakout_dashboard.png",
        compare_base + "_sample_breakout_dashboard.pdf",
        compare_base + "_hallmarks.png",
        compare_base + "_hallmarks.pdf",
        compare_base + "_recovery_patterns.png",
        compare_base + "_recovery_patterns.pdf",
        compare_base + "_metric_panels.png",
        compare_base + "_metric_panels.pdf",
        compare_base + "_taxonomy_heatmap.png",
        compare_base + "_taxonomy_heatmap.pdf",
    ]
    if stats_df is not None:
        wrote_files.append(compare_base + "_stats.tsv")
    if sample_count_stats is not None:
        wrote_files.append(compare_base + "_sample_count_stats.tsv")
    if method_significance is not None:
        wrote_files.append(compare_base + "_method_significance.tsv")
    if best_set_df is not None and not best_set_df.empty:
        wrote_files.append(compare_base + "_best_sample_method_selected.tsv")
    if best_set_significance is not None:
        wrote_files.append(compare_base + "_method_significance_best_sample_method.tsv")
    if not wrote_taxonomy:
        wrote_files = [
            path for path in wrote_files
            if "_taxonomy_heatmap." not in path
        ]
    if sample_counts is None:
        wrote_files = [
            path for path in wrote_files
            if "_sample_count_summary." not in path and "_sample_count_summary.tsv" not in path
        ]
    if sample_count_stats is None:
        wrote_files = [
            path for path in wrote_files
            if not path.endswith("_sample_count_stats.tsv")
        ]
    if method_significance is None:
        wrote_files = [
            path for path in wrote_files
            if not path.endswith("_method_significance.tsv")
        ]
    if best_set_df is None or best_set_df.empty:
        wrote_files = [
            path for path in wrote_files
            if not path.endswith("_best_sample_method_selected.tsv")
        ]
    if best_set_significance is None:
        wrote_files = [
            path for path in wrote_files
            if not path.endswith("_method_significance_best_sample_method.tsv")
        ]
    if sample_breakout_summary is None:
        wrote_files = [
            path for path in wrote_files
            if "_sample_breakout_" not in path
        ]
    if not wrote_sample_plot:
        wrote_files = [
            path for path in wrote_files
            if "_sample_count_summary." not in path
        ]
    if not wrote_sample_breakout:
        wrote_files = [
            path for path in wrote_files
            if "_sample_breakout_dashboard." not in path
        ]
    return wrote_files


def plot_fastani_scatter(ax, pair_summary, ani_threshold, af_threshold):
    if pair_summary.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FastANI pairs available", ha="center", va="center")
        return

    fail_df = pair_summary.loc[~pair_summary["passes_threshold"]]
    pass_df = pair_summary.loc[pair_summary["passes_threshold"]]
    if not fail_df.empty:
        ax.scatter(
            fail_df["ani_mean"],
            fail_df["af_min"],
            c="#bdbdbd",
            s=20,
            alpha=0.5,
            edgecolors="black",
            linewidths=0.2,
            label="below threshold",
        )
    if not pass_df.empty:
        ax.scatter(
            pass_df["ani_mean"],
            pass_df["af_min"],
            c="#1a1a1a",
            s=22,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.2,
            label="matched",
        )
    ax.axvline(ani_threshold, linestyle="--", color="#4d4d4d", linewidth=1)
    ax.axhline(af_threshold, linestyle="--", color="#4d4d4d", linewidth=1)
    ax.set_xlabel("Mean ANI (%)")
    ax.set_ylabel("Minimum alignment fraction")
    ax.set_title("FastANI pair matches")
    place_axis_legend_right(ax)


def plot_fastani_category_heatmap(ax, category_pair_summary, compare_column, metric="matched_pair_count"):
    if category_pair_summary.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FastANI category pairs available", ha="center", va="center")
        return False

    categories = ordered_unique_categories(
        list(category_pair_summary["category_a"].astype(str))
        + list(category_pair_summary["category_b"].astype(str))
    )
    matrix = pd.DataFrame(0.0, index=categories, columns=categories)
    for row in category_pair_summary.itertuples(index=False):
        matrix.loc[str(row.category_a), str(row.category_b)] = getattr(row, metric)
        matrix.loc[str(row.category_b), str(row.category_a)] = getattr(row, metric)

    label = "Matched pair count" if metric == "matched_pair_count" else "Median ANI (%)"
    sns.heatmap(
        matrix,
        cmap="Greys",
        cbar=True,
        cbar_kws={"label": label},
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f" if metric == "matched_pair_count" else ".1f",
        ax=ax,
    )
    ax.set_xlabel(compare_column)
    ax.set_ylabel(compare_column)
    ax.set_title(f"{label} by {compare_column}")
    return True


def plot_fastani_match_fraction(ax, category_summary, compare_column):
    if category_summary.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "No FastANI category summary available", ha="center", va="center")
        return

    summary = category_summary.copy()
    order = ordered_unique_categories(summary[compare_column].astype(str).tolist())
    summary[compare_column] = pd.Categorical(
        summary[compare_column].astype(str),
        categories=order,
        ordered=True,
    )
    summary = summary.sort_values(compare_column, kind="mergesort")
    sns.barplot(
        data=summary,
        x=compare_column,
        y="matched_fraction",
        hue=compare_column,
        palette=grayscale_palette(len(summary)),
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylim(0, 1.16)
    ax.set_xlabel(compare_column)
    ax.set_ylabel("Fraction of genomes matched")
    ax.set_title("FastANI matched fraction by category", pad=14)
    style_long_ticklabels(ax, axis="x", rotation=90, size=8)
    for index, value in enumerate(summary["matched_fraction"]):
        ax.text(index, min(1.08, value + 0.025), f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def plot_upset_overlap(ax_bar, ax_matrix, overlap_df, categories, top_n=15):
    ensure_plotting()
    from matplotlib.ticker import MaxNLocator

    if overlap_df.empty:
        ax_bar.axis("off")
        ax_matrix.axis("off")
        ax_bar.text(0.5, 0.5, "No overlap components available", ha="center", va="center")
        return

    plot_df = overlap_df.copy()
    plot_df["category_list"] = plot_df["overlap_key"].str.split(" & ")
    plot_df = plot_df.sort_values(
        by=["component_count", "n_categories", "total_genomes"],
        ascending=[False, False, False],
    ).head(top_n)
    x_positions = np.arange(len(plot_df))

    ax_bar.bar(x_positions, plot_df["component_count"], color="#4d4d4d", edgecolor="black", linewidth=0.5)
    ax_bar.set_ylabel("Component count")
    ax_bar.set_title("Category overlap components")
    ax_bar.set_xticks([])
    ax_bar.yaxis.set_major_locator(MaxNLocator(integer=True))

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        present = set(row["category_list"])
        y_points = []
        for y_index, category in enumerate(categories):
            included = category in present
            color = "#1a1a1a" if included else "#d9d9d9"
            ax_matrix.scatter(
                idx,
                y_index,
                s=40,
                c=color,
                edgecolors="black",
                linewidths=0.3,
                zorder=3,
            )
            if included:
                y_points.append(y_index)
        if len(y_points) > 1:
            ax_matrix.plot([idx, idx], [min(y_points), max(y_points)], color="#4d4d4d", linewidth=1)

    ax_matrix.set_yticks(np.arange(len(categories)))
    ax_matrix.set_yticklabels(categories)
    ax_matrix.set_xticks(x_positions)
    ax_matrix.set_xticklabels([""] * len(plot_df))
    ax_matrix.set_xlabel("Overlap combinations")
    ax_matrix.set_ylabel("")
    ax_matrix.invert_yaxis()
    ax_matrix.grid(False)


def summarize_ani_component_counts(compare_df, pair_summary, compare_column):
    if compare_df.empty or "ani_record_id" not in compare_df.columns:
        return pd.DataFrame(columns=[compare_column, "ani_dedup_components"])

    all_nodes = sorted(compare_df["ani_record_id"].astype(str).tolist())
    node_set = set(all_nodes)
    if pair_summary.empty:
        component_map = {node: f"component_{index:04d}" for index, node in enumerate(all_nodes, start=1)}
    else:
        filtered_edges = pair_summary.loc[pair_summary["passes_threshold"], ["record_a", "record_b"]]
        edge_list = [
            (record_a, record_b)
            for record_a, record_b in filtered_edges.itertuples(index=False, name=None)
            if record_a in node_set and record_b in node_set
        ]
        components = connected_components(all_nodes, edge_list)
        component_map = {}
        for index, nodes in enumerate(components, start=1):
            component_id = f"component_{index:04d}"
            for node in nodes:
                component_map[node] = component_id

    component_df = compare_df[[compare_column, "ani_record_id"]].copy()
    component_df["ani_record_id"] = component_df["ani_record_id"].astype(str)
    component_df["component_id"] = component_df["ani_record_id"].map(component_map)
    summary = (
        component_df.dropna(subset=["component_id"])
        .drop_duplicates([compare_column, "component_id"])
        .groupby(compare_column)
        .size()
        .rename("ani_dedup_components")
        .reset_index()
    )
    return summary


def compact_quality_subset(compare_df):
    return compare_df.loc[
        compare_df["mimag_tier"].astype(str).isin(["medium", "high"])
    ].copy()


def compact_quality_tier_map():
    return [("medium", "MQ"), ("high", "HQ")]


def summarize_shared_best_quality(shared_best_df, compare_column):
    if shared_best_df.empty:
        return pd.DataFrame(columns=[compare_column, "n_shared_best_genomes", "n_shared_components", "n_complete_components"])

    order = category_order(shared_best_df, compare_column)
    summary = (
        shared_best_df.groupby(compare_column)
        .agg(
            n_shared_best_genomes=("ani_record_id", "size"),
            n_shared_components=("component_id", "nunique"),
            n_complete_components=("shared_scope", lambda values: int((pd.Series(values) == "complete").sum())),
            median_qscore=(SCORE_COLUMN, "median"),
            mean_qscore=(SCORE_COLUMN, "mean"),
            median_completeness=("Completeness", "median"),
            median_contamination=("Contamination", "median"),
            median_integrity=("integrity_score", "median"),
            median_recoverability=("recoverability_score", "median"),
            mean_16S=("rrna_16S_score", "mean"),
            mean_23S=("rrna_23S_score", "mean"),
            mean_5S=("rrna_5S_score", "mean"),
            mean_trna_ge_18=("trna_ge_18", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    numeric_columns = [column for column in summary.columns if column != compare_column]
    for column in numeric_columns:
        if column.startswith("n_"):
            summary[column] = summary[column].fillna(0).astype(int)
    return summary


def run_shared_best_quality_tests(shared_best_df, compare_column):
    if shared_best_df.empty:
        return None

    categories = category_order(shared_best_df, compare_column)
    if len(categories) < 2:
        return None

    try:
        from scipy import stats
    except ImportError:
        warnings.warn(
            "scipy is not installed; skipping shared-genome quality statistics.",
            RuntimeWarning,
        )
        return None

    rows = []
    for metric, metric_label in SHARED_QUALITY_METRICS:
        if metric not in shared_best_df.columns:
            continue

        metric_df = shared_best_df[["component_id", compare_column, metric]].dropna()
        if metric_df.empty:
            continue

        if len(categories) > 2:
            complete_pivot = (
                metric_df.pivot_table(index="component_id", columns=compare_column, values=metric, aggfunc="first")
                .reindex(columns=categories)
                .dropna()
            )
            if len(complete_pivot) >= 2:
                statistic, pvalue = stats.friedmanchisquare(
                    *(complete_pivot[category].values for category in categories)
                )
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label,
                        "test": "Friedman",
                        "comparison_scope": "all_categories_complete",
                        "group_a": "",
                        "group_b": "",
                        "n_components": len(complete_pivot),
                        "statistic": statistic,
                        "pvalue": pvalue,
                    }
                )

        for category_a, category_b in itertools.combinations(categories, 2):
            pair_pivot = (
                metric_df.loc[metric_df[compare_column].isin([category_a, category_b])]
                .pivot_table(index="component_id", columns=compare_column, values=metric, aggfunc="first")
                .reindex(columns=[category_a, category_b])
                .dropna()
            )
            if len(pair_pivot) < 2:
                continue
            statistic, pvalue = stats.wilcoxon(
                pair_pivot[category_a].values,
                pair_pivot[category_b].values,
                zero_method="zsplit",
            )
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "test": "Wilcoxon signed-rank",
                    "comparison_scope": "pairwise_shared_components",
                    "group_a": category_a,
                    "group_b": category_b,
                    "n_components": len(pair_pivot),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

    if not rows:
        return None
    return pd.DataFrame(rows)


def build_shared_best_difference_table(shared_best_df, compare_column):
    if shared_best_df.empty:
        return pd.DataFrame()

    categories = category_order(shared_best_df, compare_column)
    metadata_columns = [
        column for column in [
            "component_categories",
            "component_n_categories",
            "component_member_count",
            "shared_scope",
        ]
        if column in shared_best_df.columns
    ]
    rows = []

    for category_a, category_b in itertools.combinations(categories, 2):
        subset = shared_best_df.loc[
            shared_best_df[compare_column].isin([category_a, category_b])
        ].copy()
        if subset.empty:
            continue

        pivot_base = (
            subset.pivot_table(
                index="component_id",
                columns=compare_column,
                values="ani_record_id",
                aggfunc="first",
            )
            .reindex(columns=[category_a, category_b])
            .dropna()
        )
        if pivot_base.empty:
            continue

        metadata_df = (
            subset.drop_duplicates("component_id")
            .set_index("component_id")[metadata_columns]
            .reindex(pivot_base.index)
            if metadata_columns
            else pd.DataFrame(index=pivot_base.index)
        )
        pair_df = metadata_df.copy()
        pair_df["group_a"] = category_a
        pair_df["group_b"] = category_b
        pair_df["pair_label"] = f"{category_a} - {category_b}"

        for metric, _ in SHARED_DIFFERENCE_METRICS:
            if metric not in subset.columns:
                continue
            metric_pivot = (
                subset.pivot_table(
                    index="component_id",
                    columns=compare_column,
                    values=metric,
                    aggfunc="first",
                )
                .reindex(columns=[category_a, category_b])
                .dropna()
            )
            metric_pivot = metric_pivot.reindex(pivot_base.index).dropna()
            if metric_pivot.empty:
                continue
            pair_df = pair_df.reindex(metric_pivot.index)
            metric_name = sanitize_token(metric)
            pair_df[f"{metric_name}_a"] = metric_pivot[category_a].values
            pair_df[f"{metric_name}_b"] = metric_pivot[category_b].values
            pair_df[f"{metric_name}_diff"] = (
                metric_pivot[category_a].values - metric_pivot[category_b].values
            )

        genome_label_map = (
            subset.set_index(["component_id", compare_column])["Genome_Id"]
            if "Genome_Id" in subset.columns
            else None
        )
        bin_label_map = (
            subset.set_index(["component_id", compare_column])["Bin Id"]
            if "Bin Id" in subset.columns
            else None
        )
        record_map = subset.set_index(["component_id", compare_column])["ani_record_id"]

        for component_id in pair_df.index:
            key_a = (component_id, category_a)
            key_b = (component_id, category_b)
            pair_df.loc[component_id, "record_a"] = record_map.get(key_a)
            pair_df.loc[component_id, "record_b"] = record_map.get(key_b)
            if genome_label_map is not None:
                pair_df.loc[component_id, "genome_a"] = genome_label_map.get(key_a)
                pair_df.loc[component_id, "genome_b"] = genome_label_map.get(key_b)
            if bin_label_map is not None:
                pair_df.loc[component_id, "bin_a"] = bin_label_map.get(key_a)
                pair_df.loc[component_id, "bin_b"] = bin_label_map.get(key_b)

        rows.append(pair_df.reset_index())

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_shared_best_difference_panels(diff_df, output_base):
    ensure_plotting()
    if diff_df.empty:
        return False

    score_diff_column = f"{sanitize_token(SCORE_COLUMN)}_diff"
    pair_order = (
        diff_df.groupby("pair_label")[score_diff_column]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    pair_counts = diff_df.groupby("pair_label").size().reindex(pair_order).fillna(0).astype(int)
    pair_label_map = {
        pair_label: f"{pair_label}\n(n={pair_counts.loc[pair_label]})"
        for pair_label in pair_order
    }
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=False)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, SHARED_DIFFERENCE_METRICS):
        diff_column = f"{sanitize_token(metric)}_diff"
        if diff_column not in diff_df.columns:
            ax.axis("off")
            continue
        plot_df = diff_df.dropna(subset=[diff_column]).copy()
        if plot_df.empty:
            ax.axis("off")
            continue
        plot_df["pair_label_display"] = plot_df["pair_label"].map(pair_label_map)
        display_order = [pair_label_map[pair_label] for pair_label in pair_order]
        max_abs = float(np.nanmax(np.abs(plot_df[diff_column].values))) if len(plot_df) else 0.0
        x_limit = max(0.05, max_abs * 1.15)
        if metric in FAVORABLE_NEGATIVE_METRICS:
            ax.axvspan(-x_limit, 0, color="#efefef", zorder=0)
        else:
            ax.axvspan(0, x_limit, color="#efefef", zorder=0)
        sns.stripplot(
            data=plot_df,
            x=diff_column,
            y="pair_label_display",
            order=display_order,
            orient="h",
            color="#1a1a1a",
            size=2.2,
            alpha=0.35,
            jitter=0.18,
            ax=ax,
        )
        summary_df = (
            plot_df.groupby("pair_label_display")[diff_column]
            .agg(
                median="median",
                q25=lambda values: values.quantile(0.25),
                q75=lambda values: values.quantile(0.75),
            )
            .reindex(display_order)
        )
        y_positions = np.arange(len(display_order))
        ax.hlines(y_positions, summary_df["q25"], summary_df["q75"], color="#4d4d4d", linewidth=2.0, zorder=3)
        ax.scatter(summary_df["median"], y_positions, marker="D", s=26, c="#000000", zorder=4)
        ax.axvline(0, linestyle="--", color="#4d4d4d", linewidth=1)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_xlabel(f"Delta {label} (A - B)")
        ax.set_ylabel("")
        ax.set_title(f"Delta {label}")
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("FastANI-shared best genome differences", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base + "_shared_best_differences")
    return True


def plot_shared_best_quality_panels(shared_best_df, compare_column, output_base):
    ensure_plotting()
    if shared_best_df.empty:
        return False

    order = category_order(shared_best_df, compare_column)
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=False)
    axes = axes.ravel()
    score_like_metrics = {
        SCORE_COLUMN,
        "integrity_score",
        "recoverability_score",
        "rrna_16S_score",
        "rrna_23S_score",
        "rrna_5S_score",
        "trna_ge_18",
    }

    for ax, (metric, label) in zip(axes, SHARED_QUALITY_METRICS):
        if metric not in shared_best_df.columns:
            ax.axis("off")
            continue
        sns.boxplot(
            data=shared_best_df,
            x=compare_column,
            y=metric,
            order=order,
            color="#bdbdbd",
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )
        sns.stripplot(
            data=shared_best_df,
            x=compare_column,
            y=metric,
            order=order,
            color="#1a1a1a",
            size=2,
            alpha=0.35,
            ax=ax,
        )
        if metric in score_like_metrics:
            ax.set_ylim(-0.02, 1.02 if metric != SCORE_COLUMN else max(1.02, float(shared_best_df[metric].max()) + 0.05))
        ax.set_xlabel(compare_column)
        ax.set_ylabel(label)
        ax.set_title(label)
        style_long_ticklabels(ax, axis="x", rotation=90, size=8)

    fig.suptitle(f"FastANI-shared best genomes by {compare_column}", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base + "_shared_best_quality")
    return True


def extract_compact_unique_taxa_genomes(compare_df, compare_column, taxonomy_column=None):
    selected_taxonomy = choose_group_column(compare_df, taxonomy_column)
    base_columns = [
        column for column in [
            compare_column,
            "sample",
            "Genome_Id",
            "Bin Id",
            "mimag_tier",
            SCORE_COLUMN,
            "ani_record_id",
            "fasta_path",
        ]
        if column in compare_df.columns
    ]
    if not selected_taxonomy or selected_taxonomy not in compare_df.columns:
        return pd.DataFrame(columns=base_columns + ["taxonomy_column", "resolved_taxonomy_label"]), selected_taxonomy

    compact_df = compact_quality_subset(compare_df)
    if compact_df.empty:
        return pd.DataFrame(columns=base_columns + [selected_taxonomy, "compact_quality_tier", "taxonomy_column", "resolved_taxonomy_label"]), selected_taxonomy

    unique_frames = []
    for tier_value, tier_label in compact_quality_tier_map():
        tier_df = compact_df.loc[compact_df["mimag_tier"].astype(str) == tier_value].copy()
        if tier_df.empty:
            continue
        tier_df["resolved_taxonomy_label"] = resolve_grouping_series(tier_df, selected_taxonomy)
        tier_df = tier_df.dropna(subset=["resolved_taxonomy_label"])
        tier_df = tier_df.loc[
            ~tier_df["resolved_taxonomy_label"].astype(str).str.startswith("unclassified_")
        ].copy()
        if tier_df.empty:
            continue

        taxon_presence = (
            tier_df[[compare_column, "resolved_taxonomy_label"]]
            .drop_duplicates()
            .groupby("resolved_taxonomy_label")[compare_column]
            .nunique()
        )
        unique_taxa = taxon_presence.loc[taxon_presence == 1].index
        tier_unique = tier_df.loc[
            tier_df["resolved_taxonomy_label"].isin(unique_taxa)
        ].copy()
        if tier_unique.empty:
            continue
        tier_unique["taxonomy_column"] = selected_taxonomy
        tier_unique["compact_quality_tier"] = tier_label
        unique_frames.append(tier_unique)

    if unique_frames:
        unique_genomes = pd.concat(unique_frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=base_columns + [selected_taxonomy, "compact_quality_tier", "taxonomy_column", "resolved_taxonomy_label"]), selected_taxonomy

    export_columns = base_columns.copy()
    if selected_taxonomy in unique_genomes.columns:
        export_columns.append(selected_taxonomy)
    export_columns.extend(["compact_quality_tier", "taxonomy_column", "resolved_taxonomy_label"])
    return unique_genomes.loc[:, export_columns], selected_taxonomy


def summarize_compact_method_metrics(compare_df, compare_column, taxonomy_column, pair_summary):
    order = category_order(compare_df, compare_column)
    compact = pd.DataFrame({compare_column: order})
    compact["total_bins"] = (
        compare_df.groupby(compare_column)
        .size()
        .reindex(order)
        .fillna(0)
        .astype(int)
        .values
    )
    selected_taxonomy = choose_group_column(compare_df, taxonomy_column)
    compact_subset = compact_quality_subset(compare_df)
    for tier_value, tier_label in compact_quality_tier_map():
        tier_df = compare_df.loc[compare_df["mimag_tier"].astype(str) == tier_value].copy()
        compact[f"{tier_label.lower()}_bins"] = (
            tier_df.groupby(compare_column)
            .size()
            .reindex(order)
            .fillna(0)
            .astype(int)
            .values
        )
        taxonomy_summary, selected_taxonomy = summarize_taxonomy_quality(
            tier_df,
            compare_column,
            taxonomy_column,
            exclude_unclassified=True,
        )
        compact = compact.merge(
            taxonomy_summary[[compare_column, "unique_taxa"]].rename(
                columns={"unique_taxa": f"{tier_label.lower()}_unique_taxa"}
            ),
            on=compare_column,
            how="left",
        )
        ani_component_summary = summarize_ani_component_counts(tier_df, pair_summary, compare_column)
        compact = compact.merge(
            ani_component_summary.rename(
                columns={"ani_dedup_components": f"{tier_label.lower()}_ani_dedup_components"}
            ),
            on=compare_column,
            how="left",
        )
    if not compact_subset.empty:
        taxonomy_summary, selected_taxonomy = summarize_taxonomy_quality(
            compact_subset,
            compare_column,
            taxonomy_column,
            exclude_unclassified=True,
        )
        compact = compact.merge(
            taxonomy_summary[[compare_column, "shared_taxa", "unique_taxa"]].rename(
                columns={"unique_taxa": "group_specific_taxa"}
            ),
            on=compare_column,
            how="left",
        )
    for column in [
        "mq_bins",
        "hq_bins",
        "mq_ani_dedup_components",
        "hq_ani_dedup_components",
        "shared_taxa",
        "group_specific_taxa",
    ]:
        compact[column] = compact[column].fillna(0).astype(int)
    return compact, selected_taxonomy


def plot_compact_method_summary(compact_df, compare_column, output_base, taxonomy_label=None):
    ensure_plotting()
    if compact_df.empty:
        return False

    metrics = [
        ("total_bins", "Total bins", None),
        (("mq_bins", "hq_bins"), "MQ / HQ bins", "Genome tier"),
        ("taxonomy_scope", f"{taxonomy_label}" if taxonomy_label else "Taxa", "Taxonomy scope"),
        (("mq_ani_dedup_components", "hq_ani_dedup_components"), "ANI-deduped genome clusters", "Genome tier"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=False)
    axes = axes.ravel()
    order = compact_df[compare_column].tolist()
    palette = grayscale_palette(len(order), start=0.2, stop=0.75)
    tier_palette = {"MQ": "#8c8c8c", "HQ": "#1a1a1a"}

    for ax, (metric, label, legend_title) in zip(axes, metrics):
        if metric == "taxonomy_scope":
            plot_df = compact_df[[compare_column, "shared_taxa", "group_specific_taxa"]].rename(
                columns={"shared_taxa": "Shared", "group_specific_taxa": "Group-specific"}
            ).set_index(compare_column)
            plot_df.plot(
                kind="bar",
                stacked=True,
                color=["#808080", "#1a1a1a"],
                edgecolor="black",
                linewidth=0.5,
                ax=ax,
            )
            place_axis_legend_right(ax, title=legend_title)
            stacked_totals = plot_df.sum(axis=1).tolist()
            max_total = max(stacked_totals) if stacked_totals else 0
            ax.set_ylim(0, max(1, max_total) * 1.14)
            for index, value in enumerate(stacked_totals):
                ax.text(index, value + max(0.5, max_total * 0.02), f"{int(value)}", ha="center", va="bottom", fontsize=8)
        elif isinstance(metric, tuple):
            plot_df = compact_df[[compare_column, metric[0], metric[1]]].melt(
                id_vars=compare_column,
                var_name="compact_metric",
                value_name="value",
            )
            plot_df["tier_label"] = plot_df["compact_metric"].map(
                {
                    metric[0]: "MQ",
                    metric[1]: "HQ",
                }
            )
            sns.barplot(
                data=plot_df,
                x=compare_column,
                y="value",
                hue="tier_label",
                order=order,
                hue_order=["MQ", "HQ"],
                palette=tier_palette,
                edgecolor="black",
                ax=ax,
            )
            place_axis_legend_right(ax, title=legend_title)
            for patch in ax.patches:
                height = patch.get_height()
                if np.isnan(height):
                    continue
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        else:
            sns.barplot(
                data=compact_df,
                x=compare_column,
                y=metric,
                hue=compare_column,
                order=order,
                palette=palette,
                dodge=False,
                legend=False,
                edgecolor="black",
                ax=ax,
            )
            for index, value in enumerate(compact_df[metric].tolist()):
                ax.text(index, value, f"{int(value)}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel(compare_column)
        ax.set_ylabel(label)
        ax.set_title(
            f"{label}: shared vs group-specific" if metric == "taxonomy_scope" else label
        )
        style_long_ticklabels(ax, axis="x", rotation=90, size=8)

    fig.suptitle(f"Compact method summary by {compare_column}", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, RIGHT_MARGIN, 0.96])
    save_figure(fig, output_base + "_compact_summary")
    return True


def save_fastani_outputs(
    frame,
    output_base,
    compare_column,
    args,
):
    ensure_plotting()
    safe_name = sanitize_token(compare_column)
    compare_base = output_base + f"_compare_{safe_name}"
    ani_base = compare_base + "_fastani"

    fastani_frame = prepare_compare_frame(frame, compare_column)
    if args.matched_samples_only:
        selected_sample = choose_sample_column(fastani_frame, args.sample_column)
        if not selected_sample:
            raise ValueError(
                f"--matched-samples-only requires a usable sample column for FastANI comparison column '{compare_column}'."
            )
        fastani_frame, selected_sample = filter_to_matched_samples(
            fastani_frame,
            compare_column,
            sample_column=args.sample_column,
        )
        if fastani_frame.empty:
            raise ValueError(
                f"No matched-sample rows remain for FastANI comparison column '{compare_column}'."
            )
    else:
        selected_sample = choose_sample_column(fastani_frame, args.sample_column)

    fastani_frame = resolve_fasta_paths(
        fastani_frame,
        fasta_column=args.ani_fasta_column,
        genome_dir=args.ani_genome_dir,
        fasta_exts=split_extensions(args.ani_fasta_exts),
    )
    raw_output = run_fastani(
        fastani_frame,
        output_base=ani_base,
        threads=args.ani_threads,
        existing_results=args.ani_results,
    )
    pair_summary = load_fastani_pairs(raw_output, fastani_frame)
    if args.matched_samples_only:
        pair_summary, _ = filter_fastani_pairs_within_sample(
            fastani_frame,
            pair_summary,
            sample_column=selected_sample,
        )
    compare_df, pair_summary, category_pair_summary, category_summary = summarize_fastani_matches(
        fastani_frame,
        pair_summary,
        compare_column,
        args.ani_threshold,
        args.ani_af_threshold,
    )
    component_df, overlap_df = build_fastani_components(compare_df, pair_summary, compare_column)

    pair_summary.to_csv(ani_base + "_pairs.tsv", sep="\t", index=False)
    category_pair_summary.to_csv(ani_base + "_category_pairs.tsv", sep="\t", index=False)
    category_summary.to_csv(ani_base + "_category_summary.tsv", sep="\t", index=False)
    component_df.to_csv(ani_base + "_components.tsv", sep="\t", index=False)
    overlap_df.to_csv(ani_base + "_overlap_summary.tsv", sep="\t", index=False)
    compact_summary_df, taxonomy_label = summarize_compact_method_metrics(
        compare_df,
        compare_column,
        args.group_column,
        pair_summary,
    )
    shared_best_df = build_shared_best_genome_table(compare_df, pair_summary, compare_column)
    shared_best_diff_df = build_shared_best_difference_table(shared_best_df, compare_column)
    shared_best_summary_df = summarize_shared_best_quality(shared_best_df, compare_column)
    shared_best_stats_df = run_shared_best_quality_tests(shared_best_df, compare_column)
    compact_unique_taxa_genomes_df, _ = extract_compact_unique_taxa_genomes(
        compare_df,
        compare_column,
        args.group_column,
    )
    compact_summary_df.to_csv(compare_base + "_compact_summary.tsv", sep="\t", index=False)
    shared_best_df.to_csv(compare_base + "_shared_best_genomes.tsv", sep="\t", index=False)
    shared_best_diff_df.to_csv(compare_base + "_shared_best_differences.tsv", sep="\t", index=False)
    shared_best_summary_df.to_csv(compare_base + "_shared_best_summary.tsv", sep="\t", index=False)
    if shared_best_stats_df is not None:
        shared_best_stats_df.to_csv(compare_base + "_shared_best_stats.tsv", sep="\t", index=False)
    compact_unique_taxa_genomes_df.to_csv(
        compare_base + "_compact_unique_taxa_genomes.tsv",
        sep="\t",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plot_fastani_scatter(ax, pair_summary, args.ani_threshold, args.ani_af_threshold)
    apply_tight_layout(fig, rect=[0, 0, RIGHT_MARGIN, 1])
    save_figure(fig, ani_base + "_scatter")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    plot_fastani_category_heatmap(ax, category_pair_summary, compare_column, metric="matched_pair_count")
    apply_tight_layout(fig, rect=[0, 0, 0.80, 1])
    save_figure(fig, ani_base + "_match_heatmap")

    fig, ax = plt.subplots(figsize=(max(7, len(category_summary) * 0.9), 6))
    plot_fastani_match_fraction(ax, category_summary, compare_column)
    apply_tight_layout(fig)
    save_figure(fig, ani_base + "_matched_fraction")

    categories = category_order(compare_df, compare_column)
    fig, (ax_bar, ax_matrix) = plt.subplots(
        2,
        1,
        figsize=(max(8, args.ani_top_overlaps * 0.6), max(6, len(categories) * 0.5 + 3)),
        gridspec_kw={"height_ratios": [2, 1.5]},
        sharex=True,
    )
    plot_upset_overlap(ax_bar, ax_matrix, overlap_df, categories, top_n=args.ani_top_overlaps)
    fig.suptitle(f"FastANI overlap UpSet by {compare_column}", fontsize=16, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 1, 0.96])
    save_figure(fig, ani_base + "_upset")

    wrote_compact = plot_compact_method_summary(
        compact_summary_df,
        compare_column,
        compare_base,
        taxonomy_label=taxonomy_label,
    )
    wrote_shared_best = plot_shared_best_quality_panels(
        shared_best_df,
        compare_column,
        compare_base,
    )
    wrote_shared_best_diff = plot_shared_best_difference_panels(
        shared_best_diff_df,
        compare_base,
    )

    wrote_files = [
        raw_output,
        ani_base + "_pairs.tsv",
        ani_base + "_category_pairs.tsv",
        ani_base + "_category_summary.tsv",
        ani_base + "_components.tsv",
        ani_base + "_overlap_summary.tsv",
        compare_base + "_compact_summary.tsv",
        compare_base + "_shared_best_genomes.tsv",
        compare_base + "_shared_best_differences.tsv",
        compare_base + "_shared_best_summary.tsv",
        compare_base + "_compact_unique_taxa_genomes.tsv",
        ani_base + "_scatter.png",
        ani_base + "_scatter.pdf",
        ani_base + "_match_heatmap.png",
        ani_base + "_match_heatmap.pdf",
        ani_base + "_matched_fraction.png",
        ani_base + "_matched_fraction.pdf",
        ani_base + "_upset.png",
        ani_base + "_upset.pdf",
    ]
    if wrote_compact:
        wrote_files.extend(
            [
                compare_base + "_compact_summary.png",
                compare_base + "_compact_summary.pdf",
            ]
        )
    if shared_best_stats_df is not None:
        wrote_files.append(compare_base + "_shared_best_stats.tsv")
    if wrote_shared_best:
        wrote_files.extend(
            [
                compare_base + "_shared_best_quality.png",
                compare_base + "_shared_best_quality.pdf",
            ]
        )
    if wrote_shared_best_diff:
        wrote_files.extend(
            [
                compare_base + "_shared_best_differences.png",
                compare_base + "_shared_best_differences.pdf",
            ]
        )
    return wrote_files


def place_axis_legend_right(ax, title=None):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.legend(
        handles,
        labels,
        title=title,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )


def add_shared_tier_legend(fig, anchor_x=0.81, anchor_y=0.82):
    ensure_plotting()
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TIER_PALETTE["low"],
            markeredgecolor="black",
            color="black",
            label="low",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markerfacecolor=TIER_PALETTE["medium"],
            markeredgecolor="black",
            color="black",
            label="medium",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor=TIER_PALETTE["high"],
            markeredgecolor="black",
            color="black",
            label="high",
        ),
    ]
    fig.legend(
        handles=handles,
        labels=["low", "medium", "high"],
        title="MIMAG tier",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(anchor_x, anchor_y),
        ncol=1,
    )


def add_qscore_colorbar(fig, mappable, axes=None, cax_rect=None):
    if mappable is None:
        return
    if cax_rect is not None:
        cax = fig.add_axes(cax_rect)
        colorbar = fig.colorbar(mappable, cax=cax)
    else:
        colorbar = fig.colorbar(
            mappable,
            ax=axes,
            fraction=0.03,
            pad=0.04,
        )
    colorbar.set_label(SCORE_LABEL)


def save_figure(fig, output_base):
    fig.savefig(output_base + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(output_base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def save_individual_plots(frame, output_base, group_column=None, top_n_groups=12):
    ensure_plotting()
    selected_group = choose_group_column(frame, group_column)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter = plot_completeness_contamination(ax, frame)
    add_qscore_colorbar(fig, scatter, cax_rect=[0.84, 0.18, 0.025, 0.38])
    add_shared_tier_legend(fig, anchor_x=0.80, anchor_y=0.76)
    apply_tight_layout(fig, rect=[0, 0, 0.66, 1])
    save_figure(fig, output_base + "_threshold_landscape")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = plot_component_scatter(ax, frame)
    add_qscore_colorbar(fig, scatter, cax_rect=[0.84, 0.18, 0.025, 0.60])
    apply_tight_layout(fig, rect=[0, 0, 0.74, 1])
    save_figure(fig, output_base + "_integrity_recoverability")

    fig, ax = plt.subplots(figsize=(7, 6))
    plot_quality_distribution(ax, frame)
    add_shared_tier_legend(fig, anchor_x=0.80, anchor_y=0.76)
    apply_tight_layout(fig, rect=[0, 0, 0.66, 1])
    save_figure(fig, output_base + "_qscore_distribution")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_feature_recovery(ax, frame)
    apply_tight_layout(fig)
    save_figure(fig, output_base + "_hallmark_contributions")

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_recovery_patterns(ax, frame)
    apply_tight_layout(fig, rect=[0, 0, 0.80, 1])
    save_figure(fig, output_base + "_recovery_patterns")

    fig, ax = plt.subplots(figsize=(max(8, min(16, top_n_groups + 2)), 6.5))
    plot_group_panel(ax, frame, selected_group, top_n_groups)
    apply_tight_layout(fig)
    save_figure(fig, output_base + "_taxonomy_qscore")


def save_atlas(frame, output_base, group_column=None, top_n_groups=12):
    ensure_plotting()
    sns.set_theme(style="whitegrid", context="notebook")
    if "category" in frame.columns and frame["category"].astype(str).nunique() > 1:
        save_grouped_atlas(
            frame,
            output_base=output_base,
            compare_column="category",
            group_column=group_column,
            top_n_groups=top_n_groups,
        )
        return
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    scatter = plot_completeness_contamination(axes[0, 0], frame)
    component_scatter = plot_component_scatter(axes[0, 1], frame)
    plot_quality_distribution(axes[0, 2], frame)
    plot_feature_recovery(axes[1, 0], frame)
    plot_recovery_patterns(axes[1, 1], frame)
    selected_group = choose_group_column(frame, group_column)
    plot_group_panel(axes[1, 2], frame, selected_group, top_n_groups)
    add_shared_tier_legend(fig, anchor_x=0.81, anchor_y=0.78)

    add_qscore_colorbar(fig, component_scatter, cax_rect=[0.85, 0.22, 0.018, 0.34])

    fig.suptitle("Genome Quality Atlas", fontsize=18, y=0.99)
    apply_tight_layout(fig, rect=[0, 0, 0.68, 0.97])
    save_figure(fig, output_base + "_atlas")


def main():
    args = build_parser().parse_args()
    master_path = os.path.abspath(args.master_tsv)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(master_path), "genome_quality_atlas")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    master = pd.read_csv(master_path, sep="\t")
    ensure_columns(master)
    if args.compare_map:
        master = merge_compare_map(master, args.compare_map, args.compare_map_key)
    annotated = compute_quality_index(master)

    output_base = os.path.join(output_dir, args.prefix)
    annotated.to_csv(output_base + "_annotated.tsv", sep="\t", index=False)

    overall_summary, feature_rates, group_summary = summarize(
        annotated, group_column=args.group_column
    )
    overall_summary.to_csv(output_base + "_summary.tsv", sep="\t", index=False)
    feature_rates.to_csv(output_base + "_feature_recovery.tsv", sep="\t", index=False)

    stats_df = run_nonparametric_test(annotated, args.group_column)
    if group_summary is not None:
        group_summary.to_csv(output_base + "_group_summary.tsv", sep="\t", index=False)
    if stats_df is not None:
        stats_df.to_csv(output_base + "_group_stats.tsv", sep="\t", index=False)

    save_atlas(
        annotated,
        output_base=output_base,
        group_column=args.group_column,
        top_n_groups=args.top_n_groups,
    )
    save_individual_plots(
        annotated,
        output_base=output_base,
        group_column=args.group_column,
        top_n_groups=args.top_n_groups,
    )
    clustermap_files = save_taxonomy_clustermap(
        annotated,
        output_base=output_base,
    )

    compare_files = []
    seen_compare_columns = []
    for compare_column in args.compare_column:
        if compare_column not in seen_compare_columns:
            seen_compare_columns.append(compare_column)
    for compare_column in seen_compare_columns:
        compare_files.extend(
            save_compare_outputs(
                annotated,
                output_base=output_base,
                compare_column=compare_column,
                group_column=args.group_column,
                top_n_groups=args.top_n_groups,
                sample_column=args.sample_column,
                matched_samples_only=args.matched_samples_only,
                ani_settings=args,
            )
        )

    ani_files = []
    seen_ani_columns = []
    for compare_column in args.ani_compare_column:
        if compare_column not in seen_ani_columns:
            seen_ani_columns.append(compare_column)
    for compare_column in seen_ani_columns:
        ani_files.extend(
            save_fastani_outputs(
                annotated,
                output_base=output_base,
                compare_column=compare_column,
                args=args,
            )
        )

    print("Wrote:")
    print(output_base + "_annotated.tsv")
    print(output_base + "_summary.tsv")
    print(output_base + "_feature_recovery.tsv")
    if group_summary is not None:
        print(output_base + "_group_summary.tsv")
    if stats_df is not None:
        print(output_base + "_group_stats.tsv")
    print(output_base + "_atlas.png")
    print(output_base + "_atlas.pdf")
    print(output_base + "_threshold_landscape.png")
    print(output_base + "_threshold_landscape.pdf")
    print(output_base + "_integrity_recoverability.png")
    print(output_base + "_integrity_recoverability.pdf")
    print(output_base + "_qscore_distribution.png")
    print(output_base + "_qscore_distribution.pdf")
    print(output_base + "_hallmark_contributions.png")
    print(output_base + "_hallmark_contributions.pdf")
    print(output_base + "_recovery_patterns.png")
    print(output_base + "_recovery_patterns.pdf")
    print(output_base + "_taxonomy_qscore.png")
    print(output_base + "_taxonomy_qscore.pdf")
    for path in clustermap_files:
        print(path)
    for path in compare_files:
        print(path)
    for path in ani_files:
        print(path)


if __name__ == "__main__":
    main()
