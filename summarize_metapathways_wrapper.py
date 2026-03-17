#!/usr/bin/env python3

import argparse
import glob
import itertools
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_metapathways_genomes import (
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    build_annotation_quality_table,
    build_annotation_source_table,
    build_elemental_mode_summary_table,
    build_elemental_summary_table,
    build_marker_summary_table,
    build_summary_tables,
    ensure_plotting,
    id_aliases,
    load_marker_manifest,
    save_figure,
    sanitize_label,
    write_outputs,
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

PLOT_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTENSIONS = {".tsv", ".csv", ".txt", ".json", ".yaml", ".yml", ".parquet"}
STYLE_KEYWORDS = [
    ("clustermap", "clustermap"),
    ("heatmap", "heatmap"),
    ("upset", "upset"),
    ("compact", "compact"),
    ("deduplicated", "compact"),
    ("representative", "compact"),
    ("panel", "panel"),
    ("facet", "facet"),
    ("distribution", "distribution"),
    ("summary", "summary"),
    ("taxonomy", "taxonomy"),
    ("marker", "marker"),
    ("reference_mode", "reference_mode"),
    ("elemental", "elemental"),
    ("pathway", "pathway"),
    ("annotation", "annotation"),
    ("quality", "quality"),
    ("sample_count", "sample_count"),
    ("presence", "presence"),
    ("pair", "pairs"),
]

PREFERRED_METHOD_ORDER = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]
PREFERRED_METHOD_TOKENS = {
    "sag": 0,
    "sags": 0,
    "xpgssag": 1,
    "xpgssags": 1,
    "xpgsag": 1,
    "xpgsags": 1,
    "mag": 2,
    "mags": 2,
    "xpgsmag": 3,
    "xpgsmags": 3,
}


def method_token(value):
    return "".join(character for character in str(value).strip().lower() if character.isalnum())


def method_rank(value):
    token = method_token(value)
    if token in PREFERRED_METHOD_TOKENS:
        return PREFERRED_METHOD_TOKENS[token]
    if ("xpg" in token) and ("sag" in token):
        return 1
    if ("xpg" in token) and ("mag" in token):
        return 3
    if ("sag" in token) and ("mag" not in token):
        return 0
    if ("mag" in token) and ("sag" not in token):
        return 2
    return None


def method_sort_key(value, counts=None):
    text = str(value).strip()
    rank = method_rank(text)
    if rank is not None:
        return (0, rank, text.lower())
    if counts is not None:
        return (1, -int(counts.get(text, 0)), text.lower())
    return (1, 0, text.lower())


def canonical_method_label(value):
    text = str(value).strip()
    rank = method_rank(text)
    if rank == 0:
        return "SAGs"
    if rank == 1:
        return "xPG_SAGs"
    if rank == 2:
        return "MAGs"
    if rank == 3:
        return "xPG_MAGs"
    return text


def ordered_methods(values, counts=None):
    cleaned = [str(value).strip() for value in values if str(value).strip() != ""]
    if not cleaned:
        return []
    unique = []
    seen = set()
    for value in cleaned:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return sorted(unique, key=lambda value: method_sort_key(value, counts=counts))


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Batch combine MetaPathways summaries from a manifest. Manifest entries can "
            "point to raw MetaPathways results directories or to existing "
            "summarize_metapathways_genomes.py output directories."
        )
    )
    parser.add_argument(
        "manifest_tsv",
        help=(
            "Manifest TSV. Preferred columns: sample, category, input_dir. "
            "Headerless 3-column or 2-column manifests are also supported. "
            "The third column may be a raw results directory or an existing summary directory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Combined output directory. Defaults to <manifest-dir>/metapathways_batch_summary",
    )
    parser.add_argument(
        "--prefix",
        default="metapathways_batch",
        help="Combined output prefix. Default: metapathways_batch",
    )
    parser.add_argument(
        "--individual-subdir",
        default="genome_function_summary",
        help=(
            "Per-results output subdirectory name when manifest paths point to raw results "
            "directories. Default: genome_function_summary"
        ),
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Pathway score threshold used for high-confidence summaries. Default: 0.8",
    )
    parser.add_argument(
        "--marker-manifest",
        default=None,
        help=(
            "Optional marker manifest passed through when manifest rows point to raw "
            "MetaPathways results directories."
        ),
    )
    parser.add_argument(
        "--reference-mappings-dir",
        default=None,
        help=(
            "Optional normalized reference mapping directory built by "
            "scripts/build_metabolism_reference_mappings.py. Used when manifest rows "
            "point to raw MetaPathways results directories."
        ),
    )
    parser.add_argument(
        "--genome-atlas-dir",
        default=None,
        help=(
            "Optional genome atlas combined output directory. If provided, the wrapper "
            "builds like-to-like paired MetaPathways comparisons using atlas shared-best genomes."
        ),
    )
    parser.add_argument(
        "--atlas-shared-best-tsv",
        default=None,
        help=(
            "Optional explicit path to atlas *_shared_best_genomes.tsv. Overrides auto-detection "
            "from --genome-atlas-dir."
        ),
    )
    parser.add_argument(
        "--atlas-annotated-tsv",
        default=None,
        help=(
            "Optional explicit path to atlas *_annotated.tsv. Used to backfill QC metrics if needed."
        ),
    )
    parser.add_argument(
        "--atlas-prefix",
        default="genome_quality",
        help="Atlas file prefix for auto-detection. Default: genome_quality",
    )
    parser.add_argument(
        "--atlas-compare-column",
        default="category",
        help="Comparison/category column used in atlas shared-best outputs. Default: category",
    )
    parser.add_argument(
        "--atlas-sample-column",
        default="sample",
        help="Sample column name expected in atlas outputs. Default: sample",
    )
    parser.add_argument(
        "--atlas-genome-id-column",
        default="Genome_Id",
        help="Genome ID column name in atlas outputs. Default: Genome_Id",
    )
    parser.add_argument(
        "--atlas-disable-alias-fallback",
        action="store_true",
        help=(
            "Disable conservative ID alias fallback when joining atlas genomes to "
            "MetaPathways genome IDs."
        ),
    )
    parser.add_argument(
        "--best-subset-tsv",
        default=None,
        help=(
            "Optional selected-genomes TSV from genome_quality_atlas_wrapper "
            "(e.g., best_sets_review_selected_genomes.tsv). If provided, "
            "wrapper writes additional subset outputs for those ANI representatives."
        ),
    )
    parser.add_argument(
        "--best-subset-scope",
        default="global",
        help=(
            "Subset scope filter when best-set columns are available. "
            "Use 'global', 'sample', or 'all'. Default: global"
        ),
    )
    parser.add_argument(
        "--best-subset-name",
        default=None,
        help=(
            "Optional best_set_name filter for --best-subset-tsv (for example, a sample name)."
        ),
    )
    parser.add_argument(
        "--best-subset-id-column",
        default="Genome_Id",
        help="Genome ID column in --best-subset-tsv. Default: Genome_Id",
    )
    parser.add_argument(
        "--best-subset-sample-column",
        default="sample",
        help="Sample column for subset matching. Default: sample",
    )
    parser.add_argument(
        "--best-subset-category-column",
        default="category",
        help="Category column for subset matching. Default: category",
    )
    parser.add_argument(
        "--min-mimag-tier",
        default="medium",
        choices=["low", "medium", "high"],
        help=(
            "Optional minimum MIMAG tier filter for atlas-linked comparisons. "
            "Default: medium (keeps medium+high). Use 'low' to disable filtering or "
            "'high' for high only."
        ),
    )
    parser.add_argument(
        "--disable-auto-atlas",
        action="store_true",
        help=(
            "Disable auto-detection of genome atlas outputs. By default the wrapper "
            "tries to find a sibling/nearby combined_genome_atlas directory and run "
            "atlas-linked + best-ANI subset comparisons automatically."
        ),
    )
    parser.add_argument(
        "--skip-atlas-linked",
        action="store_true",
        help="Skip atlas-linked paired comparisons.",
    )
    parser.add_argument(
        "--skip-best-subset",
        action="store_true",
        help="Skip best-ANI subset comparisons.",
    )
    parser.add_argument(
        "--skip-organize-outputs",
        action="store_true",
        help=(
            "Skip moving outputs into plots/ and tables/ subdirectories grouped "
            "by style and type."
        ),
    )
    return parser


def read_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    raw = pd.read_csv(manifest_path, sep="\t", header=None, comment="#")
    raw = raw.dropna(how="all")
    raw = raw.applymap(lambda value: value.strip() if isinstance(value, str) else value)
    raw = raw.loc[~raw.eq("").all(axis=1)].reset_index(drop=True)
    if raw.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    first_row = [str(value).strip().lower() for value in raw.iloc[0].tolist()]
    header_like = (
        set(first_row) >= {"sample", "category", "results_dir"}
        or set(first_row) >= {"sample", "category", "summary_dir"}
        or set(first_row) >= {"sample", "category", "input_dir"}
        or set(first_row) >= {"category", "results_dir"}
        or set(first_row) >= {"category", "summary_dir"}
        or set(first_row) >= {"category", "input_dir"}
    )
    if header_like:
        frame = pd.read_csv(manifest_path, sep="\t", comment="#")
        frame.columns = [str(column).strip() for column in frame.columns]
    elif raw.shape[1] >= 3:
        frame = raw.iloc[:, :3].copy()
        frame.columns = ["sample", "category", "input_dir"]
    elif raw.shape[1] == 2:
        frame = raw.iloc[:, :2].copy()
        frame.columns = ["category", "input_dir"]
        frame.insert(0, "sample", frame["category"])
    else:
        raise ValueError(
            "Manifest must have either 3 columns (sample, category, input_dir) "
            "or 2 columns (category, input_dir)."
        )

    if "input_dir" not in frame.columns:
        if "results_dir" in frame.columns:
            frame = frame.rename(columns={"results_dir": "input_dir"})
        elif "summary_dir" in frame.columns:
            frame = frame.rename(columns={"summary_dir": "input_dir"})

    required = ["sample", "category", "input_dir"]
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"Manifest is missing required column: {column}")
    frame = frame[required].copy()
    frame["sample"] = frame["sample"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip().map(canonical_method_label)
    frame["input_dir"] = frame["input_dir"].astype(str).str.strip()
    if frame[required].eq("").any().any():
        raise ValueError("Manifest contains blank sample/category/input_dir values.")
    return frame


def expand_manifest_inputs(frame):
    expanded_rows = []
    for row in frame.to_dict("records"):
        pattern = str(row["input_dir"]).strip()
        matches = sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True))
        if not matches:
            raise FileNotFoundError(f"Manifest pattern did not match any paths: {pattern}")
        for match in matches:
            expanded_rows.append(
                {
                    "sample": row["sample"],
                    "category": row["category"],
                    "input_dir": str(Path(match).resolve()),
                }
            )
    return pd.DataFrame(expanded_rows, columns=["sample", "category", "input_dir"])


def add_context_columns(frame, sample, category, input_dir):
    output = frame.copy()
    output.insert(0, "input_dir", str(input_dir))
    output.insert(0, "category", category)
    output.insert(0, "sample", sample)
    if "genome_id" in output.columns:
        output.insert(
            0,
            "genome_label",
            output.apply(
                lambda row: f"{row['sample']}|{row['category']}|{row['genome_id']}",
                axis=1,
            ),
        )
    return output


def classify_output_file(path_obj):
    suffix = path_obj.suffix.lower()
    stem = path_obj.stem.lower()
    if suffix in PLOT_EXTENSIONS:
        kind = "plots"
    elif suffix in TABLE_EXTENSIONS:
        kind = "tables"
    else:
        return None, None
    style = "other"
    for needle, label in STYLE_KEYWORDS:
        if needle in stem:
            style = label
            break
    return kind, style


def move_output_file(source_path, target_path):
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.move(str(source), str(target))


def next_available_target(target):
    target_path = Path(target)
    if not target_path.exists():
        return target_path
    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def organize_directory_outputs(directory):
    directory_path = Path(directory).resolve()
    if not directory_path.exists() or not directory_path.is_dir():
        return 0, None
    if directory_path.name == "_organized":
        return 0, None

    rows = []
    for entry in sorted(directory_path.iterdir(), key=lambda item: item.name):
        if not entry.is_file():
            continue
        if entry.name == "organization_index.tsv":
            continue
        kind, style = classify_output_file(entry)
        if kind is None:
            continue
        target = directory_path / kind / style / entry.name
        if target.exists():
            target = next_available_target(target)
        source_before_move = str(entry)
        move_output_file(entry, target)
        rows.append(
            {
                "source_path": source_before_move,
                "organized_path": str(target.resolve()),
                "kind": kind,
                "style": style,
            }
        )

    if not rows:
        return 0, None
    index_path = directory_path / "tables" / "summary" / "organization_index.tsv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, sep="\t", index=False)
    return len(rows), index_path


def organize_output_tree(root_dir):
    root_path = Path(root_dir).resolve()
    if not root_path.exists() or not root_path.is_dir():
        return 0, []

    total_files = 0
    index_paths = []
    for current_dir, dirnames, _ in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in {"plots", "tables", "_organized"}]
        count, index_path = organize_directory_outputs(current_dir)
        if count > 0:
            total_files += count
            if index_path is not None:
                index_paths.append(index_path)
    return total_files, index_paths


def build_combined_pathway_matrix(pathway_long, value_column):
    if pathway_long.empty or value_column not in pathway_long.columns:
        return pd.DataFrame()
    matrix = (
        pathway_long.pivot_table(
            index="genome_label",
            columns="PWY_NAME",
            values=value_column,
            aggfunc="max",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
        .reset_index()
    )
    matrix.columns.name = None
    return matrix


def find_single_file(directory, pattern, required=True):
    matches = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not matches:
        matches = sorted(path for path in directory.rglob(pattern) if path.is_file())
    if not matches:
        if required:
            raise FileNotFoundError(f"Expected file matching '{pattern}' under {directory}")
        return None
    if len(matches) > 1:
        matches.sort(
            key=lambda path: (
                len(path.parts),
                -path.stat().st_mtime,
                str(path),
            )
        )
    return matches[0]


def find_preferred_file(directory, preferred_name, fallback_pattern, required=True):
    preferred = directory / preferred_name
    if preferred.exists():
        return preferred
    recursive_preferred = sorted(path for path in directory.rglob(preferred_name) if path.is_file())
    if recursive_preferred:
        recursive_preferred.sort(
            key=lambda path: (
                len(path.parts),
                -path.stat().st_mtime,
                str(path),
            )
        )
        return recursive_preferred[0]
    return find_single_file(directory, fallback_pattern, required=required)


def detect_input_mode(input_dir):
    resolved = Path(input_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {resolved}")
    if (resolved / "pgdb").exists():
        return "results", resolved
    if list(resolved.glob("*_genome_summary.tsv")) or list(resolved.rglob("*_genome_summary.tsv")):
        return "summary", resolved
    raise ValueError(
        f"Could not determine input type for {resolved}. Expected a MetaPathways results "
        "directory with 'pgdb/' or a summary directory containing '*_genome_summary.tsv'."
    )


def load_existing_summary(summary_dir):
    genome_path = find_single_file(summary_dir, "*_genome_summary.tsv")
    prefix = genome_path.name[: -len("_genome_summary.tsv")]
    pathway_path = find_preferred_file(
        summary_dir,
        f"{prefix}_pathway_long.tsv",
        "*_pathway_long.tsv",
    )
    pathway_orf_path = find_preferred_file(
        summary_dir,
        f"{prefix}_pathway_orf_long.tsv",
        "*_pathway_orf_long.tsv",
        required=False,
    )
    annotation_path = find_preferred_file(
        summary_dir,
        f"{prefix}_annotation_summary.tsv",
        "*_annotation_summary.tsv",
        required=False,
    )
    annotation_quality_path = find_preferred_file(
        summary_dir,
        f"{prefix}_annotation_quality_summary.tsv",
        "*_annotation_quality_summary.tsv",
        required=False,
    )
    marker_summary_path = find_preferred_file(
        summary_dir,
        f"{prefix}_marker_summary.tsv",
        "*_marker_summary.tsv",
        required=False,
    )
    reference_mode_summary_path = find_preferred_file(
        summary_dir,
        f"{prefix}_reference_mode_summary.tsv",
        "*_reference_mode_summary.tsv",
        required=False,
    )
    elemental_annotation_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_annotation_summary.tsv",
        "*_elemental_annotation_summary.tsv",
        required=False,
    )
    elemental_mode_annotation_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_mode_annotation_summary.tsv",
        "*_elemental_mode_annotation_summary.tsv",
        required=False,
    )
    elemental_pathway_support_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_pathway_support_summary.tsv",
        "*_elemental_pathway_support_summary.tsv",
        required=False,
    )
    elemental_mode_pathway_support_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_mode_pathway_support_summary.tsv",
        "*_elemental_mode_pathway_support_summary.tsv",
        required=False,
    )
    elemental_pathway_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_pathway_summary.tsv",
        "*_elemental_pathway_summary.tsv",
        required=False,
    )
    elemental_mode_pathway_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_mode_pathway_summary.tsv",
        "*_elemental_mode_pathway_summary.tsv",
        required=False,
    )

    genome_summary = pd.read_csv(genome_path, sep="\t")
    pathway_long = pd.read_csv(pathway_path, sep="\t")
    pathway_orf_long = pd.read_csv(pathway_orf_path, sep="\t") if pathway_orf_path else pd.DataFrame()
    annotation_summary = (
        pd.read_csv(annotation_path, sep="\t")
        if annotation_path else build_annotation_source_table(genome_summary)
    )
    annotation_quality_summary = (
        pd.read_csv(annotation_quality_path, sep="\t")
        if annotation_quality_path else build_annotation_quality_table(genome_summary)
    )
    marker_summary = (
        pd.read_csv(marker_summary_path, sep="\t")
        if marker_summary_path else build_marker_summary_table(genome_summary)
    )
    reference_mode_summary = (
        pd.read_csv(reference_mode_summary_path, sep="\t")
        if reference_mode_summary_path else pd.DataFrame()
    )
    elemental_annotation_summary = (
        pd.read_csv(elemental_annotation_path, sep="\t")
        if elemental_annotation_path else build_elemental_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
    )
    elemental_mode_annotation_summary = (
        pd.read_csv(elemental_mode_annotation_path, sep="\t")
        if elemental_mode_annotation_path else build_elemental_mode_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
    )
    elemental_pathway_support_summary = (
        pd.read_csv(elemental_pathway_support_path, sep="\t")
        if elemental_pathway_support_path else build_elemental_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
    )
    elemental_mode_pathway_support_summary = (
        pd.read_csv(elemental_mode_pathway_support_path, sep="\t")
        if elemental_mode_pathway_support_path else build_elemental_mode_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
    )
    elemental_pathway_summary = (
        pd.read_csv(elemental_pathway_path, sep="\t")
        if elemental_pathway_path else build_elemental_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
    )
    elemental_mode_pathway_summary = (
        pd.read_csv(elemental_mode_pathway_path, sep="\t")
        if elemental_mode_pathway_path else build_elemental_mode_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
    )
    return (
        genome_summary,
        annotation_summary,
        annotation_quality_summary,
        marker_summary,
        reference_mode_summary,
        elemental_annotation_summary,
        elemental_mode_annotation_summary,
        elemental_pathway_support_summary,
        elemental_mode_pathway_support_summary,
        elemental_pathway_summary,
        elemental_mode_pathway_summary,
        pathway_long,
        pathway_orf_long,
    )


def choose_atlas_file(atlas_dir, explicit_path, pattern, preferred_token=None, required=True):
    if explicit_path:
        resolved = Path(explicit_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Atlas file not found: {resolved}")
        return resolved

    if atlas_dir is None:
        if required:
            raise ValueError("Atlas directory is required when an explicit atlas file path is not provided.")
        return None

    atlas_base = Path(atlas_dir).expanduser().resolve()
    matches = sorted(path for path in atlas_base.glob(pattern) if path.is_file())
    if not matches:
        matches = sorted(path for path in atlas_base.rglob(pattern) if path.is_file())
    if not matches:
        if required:
            raise FileNotFoundError(f"No atlas files matched '{pattern}' in {atlas_dir}")
        return None
    if len(matches) == 1:
        return matches[0]

    if preferred_token:
        filtered = [path for path in matches if preferred_token in path.name]
        if len(filtered) == 1:
            return filtered[0]
        if len(filtered) > 1:
            matches = filtered

    if required:
        raise ValueError(
            f"Multiple atlas files matched '{pattern}' in {atlas_dir}. "
            f"Provide an explicit path. Matches: {', '.join(str(path) for path in matches)}"
        )
    return None


def load_atlas_inputs(args):
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve() if args.genome_atlas_dir else None
    safe_compare = sanitize_label(args.atlas_compare_column)

    shared_preferred = f"{sanitize_label(args.atlas_prefix)}_compare_{safe_compare}_shared_best_genomes.tsv"
    shared_best_path = choose_atlas_file(
        atlas_dir=atlas_dir,
        explicit_path=args.atlas_shared_best_tsv,
        pattern="*_shared_best_genomes.tsv",
        preferred_token=shared_preferred,
        required=True,
    )
    annotated_preferred = f"{sanitize_label(args.atlas_prefix)}_annotated.tsv"
    annotated_path = choose_atlas_file(
        atlas_dir=atlas_dir,
        explicit_path=args.atlas_annotated_tsv,
        pattern="*_annotated.tsv",
        preferred_token=annotated_preferred,
        required=False,
    )

    shared_best_df = pd.read_csv(shared_best_path, sep="\t")
    annotated_df = pd.read_csv(annotated_path, sep="\t") if annotated_path else pd.DataFrame()
    return shared_best_path, annotated_path, shared_best_df, annotated_df


def auto_detect_atlas_dir(manifest_path, output_dir, manifest_df):
    candidates = []

    def add_candidate(path_obj):
        if path_obj is None:
            return
        resolved = Path(path_obj).expanduser().resolve()
        if resolved not in candidates:
            candidates.append(resolved)

    add_candidate(output_dir.parent / "combined_genome_atlas")
    add_candidate(manifest_path.parent / "combined_genome_atlas")
    add_candidate(output_dir)

    if isinstance(manifest_df, pd.DataFrame) and "input_dir" in manifest_df.columns and not manifest_df.empty:
        input_paths = [str(Path(path).expanduser().resolve()) for path in manifest_df["input_dir"].astype(str).tolist()]
        try:
            common = Path(os.path.commonpath(input_paths))
            add_candidate(common)
            add_candidate(common.parent)
            add_candidate(common.parent / "combined_genome_atlas")
        except Exception:
            pass

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        has_shared = any(path.is_file() for path in candidate.rglob("*_shared_best_genomes.tsv"))
        has_subset = (candidate / "best_sets_review_selected_genomes.tsv").exists()
        if has_shared or has_subset:
            return candidate
    return None


def has_atlas_shared_best(args):
    if args.atlas_shared_best_tsv:
        return True
    if not args.genome_atlas_dir:
        return False
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve()
    if not atlas_dir.exists():
        return False
    return any(path.is_file() for path in atlas_dir.rglob("*_shared_best_genomes.tsv"))


def resolve_best_subset_path(args):
    if args.best_subset_tsv:
        resolved = Path(args.best_subset_tsv).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Best-subset TSV was not found: {resolved}")
        return resolved
    if args.genome_atlas_dir:
        candidate = Path(args.genome_atlas_dir).expanduser().resolve() / "best_sets_review_selected_genomes.tsv"
        if candidate.exists():
            return candidate
    return None


def load_best_subset_table(args):
    subset_path = resolve_best_subset_path(args)
    if subset_path is None:
        return None, pd.DataFrame()

    subset_df = pd.read_csv(subset_path, sep="\t")
    if subset_df.empty:
        raise ValueError(f"Best-subset TSV is empty: {subset_path}")

    scope_value = str(args.best_subset_scope).strip().lower()
    if scope_value not in {"all", "global", "sample"}:
        raise ValueError("--best-subset-scope must be one of: all, global, sample")
    if scope_value != "all" and "best_set_scope" in subset_df.columns:
        subset_df = subset_df.loc[
            subset_df["best_set_scope"].astype(str).str.strip().str.lower().eq(scope_value)
        ].copy()
    if args.best_subset_name and "best_set_name" in subset_df.columns:
        subset_df = subset_df.loc[
            subset_df["best_set_name"].astype(str).str.strip().eq(str(args.best_subset_name).strip())
        ].copy()
    if subset_df.empty:
        raise ValueError(
            "Best-subset filtering removed all rows. Check --best-subset-scope / --best-subset-name."
        )
    return subset_path, subset_df


def build_best_subset_lookup(subset_df, args):
    candidate_id_columns = [
        args.best_subset_id_column,
        "Genome_Id",
        "genome_id",
        "Bin Id",
    ]
    selected_id_column = next((column for column in candidate_id_columns if column in subset_df.columns), None)
    if selected_id_column is None:
        raise ValueError(
            "--best-subset-tsv does not contain a usable genome ID column. "
            f"Tried: {', '.join([column for column in candidate_id_columns if column])}"
        )

    sample_column = args.best_subset_sample_column if args.best_subset_sample_column in subset_df.columns else None
    category_column = args.best_subset_category_column if args.best_subset_category_column in subset_df.columns else None

    alias_columns = [selected_id_column]
    for optional_column in [
        "SAG_ID",
        "Bin Id",
        "Genome_Id",
        "genome_id",
        "fasta_path",
    ]:
        if optional_column in subset_df.columns and optional_column not in alias_columns:
            alias_columns.append(optional_column)

    tuple_aliases = set()
    category_aliases = set()
    id_aliases_set = set()
    for row in subset_df.to_dict("records"):
        aliases = set()
        for column in alias_columns:
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            if column == "fasta_path":
                value = Path(value).name
            aliases.update(id_aliases(value))
        if not aliases:
            continue
        category_value = (
            canonical_method_label(str(row.get(category_column, "")).strip())
            if category_column else ""
        )
        if sample_column and category_column:
            sample_value = str(row.get(sample_column, "")).strip()
            if sample_value and category_value:
                for alias in aliases:
                    tuple_aliases.add((sample_value, category_value, alias))
        if category_value:
            for alias in aliases:
                category_aliases.add((category_value, alias))
        for alias in aliases:
            id_aliases_set.add(alias)
    return {
        "id_column": selected_id_column,
        "sample_column": sample_column,
        "category_column": category_column,
        "tuple_aliases": tuple_aliases,
        "category_aliases": category_aliases,
        "id_aliases": id_aliases_set,
    }


def filter_frame_by_best_subset(frame, subset_lookup, sample_column="sample", category_column="category", genome_column="genome_id"):
    if frame is None or frame.empty or genome_column not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else frame

    working = frame.copy()
    working[genome_column] = working[genome_column].astype(str).str.strip()
    if sample_column in working.columns:
        working[sample_column] = working[sample_column].astype(str).str.strip()
    if category_column in working.columns:
        working[category_column] = (
            working[category_column]
            .astype(str)
            .str.strip()
            .map(canonical_method_label)
        )

    tuple_aliases = subset_lookup.get("tuple_aliases", set())
    category_aliases = subset_lookup.get("category_aliases", set())
    id_aliases_set = subset_lookup.get("id_aliases", set())
    use_tuples = bool(tuple_aliases) and sample_column in working.columns and category_column in working.columns

    keep_mask = []
    for row in working.to_dict("records"):
        aliases = id_aliases(row.get(genome_column, ""))
        if not aliases:
            keep_mask.append(False)
            continue
        if use_tuples:
            sample_value = str(row.get(sample_column, "")).strip()
            category_value = canonical_method_label(str(row.get(category_column, "")).strip())
            matched = any((sample_value, category_value, alias) in tuple_aliases for alias in aliases)
            if (not matched) and (not tuple_aliases) and category_value and category_aliases:
                matched = any((category_value, alias) in category_aliases for alias in aliases)
        else:
            matched = bool(aliases & id_aliases_set)
        keep_mask.append(matched)
    return working.loc[np.array(keep_mask, dtype=bool)].copy()

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


def build_metapathways_lookup(genome_summary_df):
    required = ["sample", "category", "genome_id"]
    for column in required:
        if column not in genome_summary_df.columns:
            raise ValueError(f"Combined MetaPathways genome summary is missing required column '{column}'.")

    working = genome_summary_df.copy().reset_index(drop=True)
    working["_mp_index"] = working.index.astype(int)
    working["_sample_key"] = working["sample"].astype(str).str.strip()
    working["_category_key"] = (
        working["category"]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working["_genome_key"] = working["genome_id"].astype(str).str.strip()
    working["_alias_set"] = working["_genome_key"].apply(id_aliases)

    exact_map = {}
    alias_map = {}
    for mp_index, sample_key, category_key, genome_key, alias_set in zip(
        working["_mp_index"].tolist(),
        working["_sample_key"].tolist(),
        working["_category_key"].tolist(),
        working["_genome_key"].tolist(),
        working["_alias_set"].tolist(),
    ):
        group_key = (sample_key, category_key)
        exact_map.setdefault((group_key, genome_key), []).append(int(mp_index))
        for alias in alias_set:
            alias_map.setdefault((group_key, alias), set()).add(int(mp_index))
    return working, exact_map, alias_map


def match_atlas_genomes_to_metapathways(atlas_df, mp_df, exact_map, alias_map, args):
    sample_column = args.atlas_sample_column
    compare_column = args.atlas_compare_column
    genome_id_column = args.atlas_genome_id_column
    use_alias_fallback = not bool(args.atlas_disable_alias_fallback)

    atlas_alias_columns = [genome_id_column]
    for optional_column in ["SAG_ID", "Bin Id", "genome_id", "fasta_path", "ani_record_id", "ani_fasta_path"]:
        if optional_column in atlas_df.columns and optional_column not in atlas_alias_columns:
            atlas_alias_columns.append(optional_column)

    match_rows = []
    matched_mp_indices = []
    for atlas_idx, row in atlas_df.reset_index(drop=True).iterrows():
        sample_key = str(row.get(sample_column, "")).strip()
        category_key = str(row.get(compare_column, "")).strip()
        genome_key = str(row.get(genome_id_column, "")).strip()
        group_key = (sample_key, category_key)
        raw_identifiers = set()
        for column in atlas_alias_columns:
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            if column in {"fasta_path", "ani_fasta_path"}:
                value = Path(value).name
            raw_identifiers.add(value)

        status = "unmatched"
        method = ""
        matched_indices = []
        if not category_key or not raw_identifiers:
            status = "missing_required"
        else:
            exact_candidates = set()
            for raw_identifier in raw_identifiers:
                exact_candidates.update(exact_map.get((group_key, raw_identifier), []))
            if len(exact_candidates) == 1:
                status = "matched"
                method = "exact"
                matched_indices = sorted(exact_candidates)
            elif len(exact_candidates) > 1:
                status = "ambiguous"
                method = "exact"
                matched_indices = sorted(set(exact_candidates))
            elif use_alias_fallback:
                alias_candidates = set()
                alias_pool = set()
                for raw_identifier in raw_identifiers:
                    alias_pool.update(id_aliases(raw_identifier))
                for alias in alias_pool:
                    alias_candidates.update(alias_map.get((group_key, alias), set()))
                if len(alias_candidates) == 1:
                    status = "matched"
                    method = "alias"
                    matched_indices = sorted(alias_candidates)
                elif len(alias_candidates) > 1:
                    status = "ambiguous"
                    method = "alias"
                    matched_indices = sorted(alias_candidates)

        matched_index = matched_indices[0] if len(matched_indices) == 1 else np.nan
        if len(matched_indices) == 1:
            matched_mp_indices.append(int(matched_index))

        match_rows.append(
            {
                "_atlas_row_index": atlas_idx,
                "_atlas_sample": sample_key,
                "_atlas_category": category_key,
                "_atlas_genome_id": genome_key,
                "_match_status": status,
                "_match_method": method,
                "_match_count": len(matched_indices),
                "_matched_mp_index": matched_index,
            }
        )

    match_df = pd.DataFrame(match_rows)
    merged = atlas_df.reset_index(drop=True).merge(match_df, left_index=True, right_on="_atlas_row_index", how="left")
    matched = merged.loc[merged["_match_status"].eq("matched")].copy()

    mp_prefixed = mp_df.drop(columns=["_alias_set"]).copy()
    mp_prefixed = mp_prefixed.add_prefix("mp_")
    matched = matched.merge(
        mp_prefixed,
        left_on="_matched_mp_index",
        right_on="mp__mp_index",
        how="left",
    )
    return merged, matched


def pair_sample_value(row):
    for column in ["mp_sample", "sample", "_atlas_sample"]:
        value = str(row.get(column, "")).strip()
        if value and value.lower() != "nan":
            return value
    return ""


def taxonomy_species_is_informative(value):
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "null", "unknown"}:
        return False
    if lowered.startswith("unclassified"):
        return False
    return True


def resolve_species_label(row):
    species_value = str(row.get("Species", "")).strip()
    if taxonomy_species_is_informative(species_value):
        return species_value

    for rank_column, value_column in [
        ("mp_taxonomy_display_rank", "mp_taxonomy_display_value"),
        ("taxonomy_display_rank", "taxonomy_display_value"),
    ]:
        rank_value = str(row.get(rank_column, "")).strip().lower()
        value = str(row.get(value_column, "")).strip()
        if rank_value == "species" and taxonomy_species_is_informative(value):
            return value
    return ""


def build_species_representative_table(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()
    required = {"mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame()

    working = matched_df.copy()
    working["_pair_sample"] = working.apply(pair_sample_value, axis=1)
    working["_species_label"] = working.apply(resolve_species_label, axis=1)
    working[compare_column] = working[compare_column].astype(str).str.strip()
    working["mp_genome_id"] = working["mp_genome_id"].astype(str).str.strip()

    working = working.loc[
        working["_pair_sample"].ne("")
        & working["_species_label"].ne("")
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
        "qscore",
        "mp_informative_annotation_fraction",
        "mp_total_pathways",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
    ]:
        if column in working.columns:
            sort_columns.append(column)
            ascending.append(False)
    sort_columns.extend(["mp_genome_id"])
    ascending.extend([True])
    working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")

    selected = (
        working.groupby(["_pair_sample", compare_column, "_species_label"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected["sample"] = selected["_pair_sample"].astype(str)
    selected["taxonomy_species"] = selected["_species_label"].astype(str)
    selected["component_id"] = selected["taxonomy_species"].astype(str)
    selected["selection_scope"] = "sample_method_species"
    selected = selected.drop(columns=["_pair_sample", "_species_label"])
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


def method_family_label(method_name):
    value = str(method_name).strip().lower()
    if "mag" in value:
        return "MAG"
    if "sag" in value:
        return "SAG"
    return ""


def method_variant_flag(method_name):
    value = str(method_name).strip().lower()
    variant_tokens = ["xpg", "variant"]
    return any(token in value for token in variant_tokens)


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


def plot_focus_genome_axis_delta_heatmap(delta_df, metric_specs, output_base, title):
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

    matrix = delta_df[metric_columns].copy()
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

    row_labels = delta_df["component_label"].astype(str).tolist()
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
    ax.set_title(title)

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
    fig.subplots_adjust(left=0.34, right=0.99, top=0.95, bottom=0.2)
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
        aliases.update(id_aliases(raw_value))

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
    y_font = 8 if n_rows <= 70 else 6 if n_rows <= 180 else 5
    x_font = 9 if n_cols <= 7 else 8
    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(10, n_cols * 1.9 + 3), max(6, n_rows * 0.24)),
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

    fig.subplots_adjust(left=0.08, right=0.995, top=0.95, bottom=0.14)
    save_figure(fig, output_base)
    return True


def plot_species_method_presence_heatmap(entity_df, methods, output_base, title):
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
    y_font = 8 if n_rows <= 60 else 6 if n_rows <= 140 else 5
    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(9, n_cols * 1.2), max(6, n_rows * 0.22)),
    )
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(methods, rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(labels, fontsize=y_font)
    ax.set_xlabel("Method")
    ax.set_ylabel("Sample | Species")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Presence (1) / Absence (0)")
    fig.subplots_adjust(left=0.36, right=0.99, top=0.95, bottom=0.12)
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
        y_font = 8 if n_rows <= 80 else 6 if n_rows <= 220 else 5
        plt_local = ensure_plotting()
        fig, ax = plt_local.subplots(
            figsize=(max(8, n_cols * 1.2 + 1.5), max(5, n_rows * 0.22)),
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
        fig.subplots_adjust(left=0.36, right=0.99, top=0.95, bottom=0.14)
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
        "qscore",
        "mp_informative_annotation_fraction",
        "mp_total_pathways",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
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


def category_order(frame, category_column="category"):
    if frame.empty or category_column not in frame.columns:
        return []
    counts = (
        frame.groupby(category_column)
        .size()
        .to_dict()
    )
    return ordered_methods(frame[category_column].astype(str).tolist(), counts=counts)


def annotate_bar_values(ax, values, fmt="int"):
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        numeric = float(value)
        if fmt == "float2":
            label = f"{numeric:.2f}"
        elif fmt == "float1":
            label = f"{numeric:.1f}"
        else:
            label = f"{int(round(numeric))}"
        ax.text(index, numeric, label, ha="center", va="bottom", fontsize=8)


def build_compact_category_summary(genome_df, category_column="category"):
    if genome_df.empty or category_column not in genome_df.columns:
        return pd.DataFrame(), "informative_annotation_orfs"

    working = genome_df.copy()
    informative_column = (
        "informative_annotation_orfs"
        if "informative_annotation_orfs" in working.columns
        else "annotated_orfs"
    )
    for column in ["total_orfs", informative_column, "total_pathways"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(0.0)

    summary = (
        working.groupby(category_column, dropna=False)
        .agg(
            n_genomes=("genome_id", "nunique"),
            total_orfs=("total_orfs", "sum"),
            informative_orfs=(informative_column, "sum"),
            total_pathways=("total_pathways", "sum"),
        )
        .reset_index()
    )
    order = category_order(working, category_column=category_column)
    summary[category_column] = summary[category_column].astype(str)
    summary = summary.set_index(category_column).reindex(order).reset_index()
    return summary, informative_column


def plot_combined_category_compact_summary(
    genome_df,
    output_base,
    category_column="category",
    sample_column="sample",
):
    ensure_plotting()
    if genome_df.empty or category_column not in genome_df.columns:
        return False

    working = genome_df.copy()
    informative_column = (
        "informative_annotation_orfs"
        if "informative_annotation_orfs" in working.columns
        else "annotated_orfs"
    )
    for column in ["total_orfs", informative_column, "total_pathways"]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    order = category_order(working, category_column=category_column)
    if not order:
        return False

    metrics = [
        ("genomes_per_sample", "Genomes per sample"),
        ("total_orfs", "Total ORFs per genome"),
        (informative_column, "Informative ORFs per genome"),
        ("total_pathways", "Inferred pathways per genome"),
    ]
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(2, 2, figsize=(max(12, len(order) * 0.9), 10), sharex=False)
    axes = axes.ravel()
    samples = (
        working[sample_column].astype(str).str.strip().loc[lambda series: series.ne("")].unique().tolist()
        if sample_column in working.columns
        else []
    )

    for ax, (metric, title) in zip(axes, metrics):
        grouped_values = []
        if metric == "genomes_per_sample":
            if not samples:
                ax.axis("off")
                continue
            for category in order:
                subset = (
                    working.loc[working[category_column].astype(str).eq(category)]
                    .groupby(sample_column)["genome_id"]
                    .nunique()
                    .reindex(samples, fill_value=0)
                )
                grouped_values.append(subset.values.astype(float))
        else:
            if metric not in working.columns:
                ax.axis("off")
                continue
            for category in order:
                series = (
                    pd.to_numeric(
                        working.loc[working[category_column].astype(str).eq(category), metric],
                        errors="coerce",
                    )
                    .dropna()
                )
                grouped_values.append(series.values.astype(float))

        if not any(len(values) for values in grouped_values):
            ax.axis("off")
            continue

        box = ax.boxplot(grouped_values, patch_artist=True, labels=order, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#c0c0c0")
            patch.set_edgecolor("black")
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.3)
        for whisker in box["whiskers"]:
            whisker.set_color("black")
        for cap in box["caps"]:
            cap.set_color("black")
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xticklabels(order, rotation=90)
        ax.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.6)
        if metric == "genomes_per_sample":
            ax.set_ylim(bottom=0)

    n_samples = int(working[sample_column].astype(str).nunique()) if sample_column in working.columns else int(working.shape[0])
    dedup_note = "species-deduplicated within sample-method"
    fig.suptitle(
        f"MetaPathways compact summary ({dedup_note}; samples={n_samples})",
        fontsize=16,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_base)
    return True


def plot_combined_category_distributions(genome_df, output_base, category_column="category"):
    ensure_plotting()
    if genome_df.empty or category_column not in genome_df.columns:
        return False
    order = category_order(genome_df, category_column=category_column)
    if not order:
        return False

    metrics = [
        ("mimag_quality_index", "MIMAG quality index"),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
        ("qscore", "Qscore"),
        ("informative_annotation_fraction", "Informative annotation fraction"),
        ("pathway_input_fraction", "Pathway-input fraction"),
        ("pathway_support_fraction", "Pathway-support fraction"),
        ("total_pathways", "Inferred pathways"),
        ("marker_supported_orfs", "Marker-supported ORFs"),
        ("reference_mode_supported_accessions", "Reference-supported accessions"),
    ]
    available_metrics = []
    for metric, title in metrics:
        if metric not in genome_df.columns:
            continue
        values = pd.to_numeric(genome_df[metric], errors="coerce")
        if values.notna().any():
            available_metrics.append((metric, title))
    if not available_metrics:
        return False

    n_cols = 3
    n_rows = int(np.ceil(len(available_metrics) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, len(order) * 1.0), max(7.0, n_rows * 3.6)),
        sharex=False,
        squeeze=False,
    )
    axes = axes.ravel()
    wrote_any = False

    for ax, (metric, title) in zip(axes, available_metrics):
        metric_values = pd.to_numeric(genome_df[metric], errors="coerce")
        grouped = []
        for category in order:
            series = metric_values.loc[genome_df[category_column].astype(str).eq(category)].dropna()
            grouped.append(series.values)
        if not any(len(values) for values in grouped):
            ax.axis("off")
            continue
        box = ax.boxplot(grouped, patch_artist=True, labels=order)
        for patch in box["boxes"]:
            patch.set_facecolor("#c0c0c0")
            patch.set_edgecolor("black")
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        ax.set_title(title)
        ax.set_xticklabels(order, rotation=90)
        ax.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.6)
        if metric.endswith("_fraction") or metric in {"integrity_score", "recoverability_score", "mimag_quality_index"}:
            ax.set_ylim(0, 1.02)
        wrote_any = True

    for index in range(len(available_metrics), len(axes)):
        axes[index].axis("off")

    if not wrote_any:
        plt_local.close(fig)
        return False
    fig.suptitle("MetaPathways category metric distributions (quality first, then function)", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save_figure(fig, output_base)
    return True


def _heatmap_text_color(value, vmax):
    if vmax <= 0:
        return "black"
    return "white" if float(value) >= (0.55 * float(vmax)) else "black"


def plot_sample_category_count_heatmap(genome_df, output_base, category_column="category", sample_column="sample"):
    ensure_plotting()
    required = {category_column, sample_column}
    if genome_df.empty or not required.issubset(set(genome_df.columns)):
        return False

    matrix = (
        genome_df.groupby([sample_column, category_column])["genome_id"]
        .nunique()
        .reset_index(name="n_genomes")
        .pivot_table(
            index=sample_column,
            columns=category_column,
            values="n_genomes",
            fill_value=0,
            aggfunc="sum",
        )
        .sort_index()
    )
    matrix = matrix.reindex(columns=category_order(genome_df, category_column=category_column), fill_value=0)
    if matrix.empty:
        return False

    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(figsize=(max(8, matrix.shape[1] * 0.9), max(5, matrix.shape[0] * 0.55)))
    vmax = max(1.0, float(np.nanmax(matrix.values)))
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str).tolist())
    ax.set_xlabel("Category")
    ax.set_ylabel("Sample")
    ax.set_title("Genome counts by sample and category")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(round(float(matrix.iat[row_index, col_index])))
            ax.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color=_heatmap_text_color(value, vmax),
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Genome count")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def _category_mode_matrix(genome_df, category_column, value_prefix, value_suffix):
    columns = [f"{value_prefix}_{mode_id}_{value_suffix}" for mode_id in ELEMENTAL_MODE_ORDER]
    available = [column for column in columns if column in genome_df.columns]
    if not available:
        return pd.DataFrame(), []
    matrix = (
        genome_df.groupby(category_column)[available]
        .mean(numeric_only=True)
        .reindex(index=category_order(genome_df, category_column=category_column))
    )
    matrix.columns = [
        column.replace(f"{value_prefix}_", "").replace(f"_{value_suffix}", "")
        for column in matrix.columns
    ]
    return matrix.fillna(0.0), matrix.columns.tolist()


def plot_category_mode_support_heatmaps(genome_df, output_base, category_column="category"):
    ensure_plotting()
    if genome_df.empty or category_column not in genome_df.columns:
        return False

    marker_matrix, _ = _category_mode_matrix(genome_df, category_column, "marker", "gene_count")
    ref_matrix, _ = _category_mode_matrix(genome_df, category_column, "reference_mode", "accession_count")

    panels = []
    if not marker_matrix.empty:
        panels.append(("Marker gene support (mean per genome)", marker_matrix))
    if not ref_matrix.empty:
        panels.append(("Reference accession support (mean per genome)", ref_matrix))
    if not panels:
        return False

    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        1,
        len(panels),
        figsize=(max(10, len(panels) * 9), max(5.5, max(panel[1].shape[0] for panel in panels) * 0.5)),
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (title, matrix) in zip(axes, panels):
        modes = matrix.columns.astype(str).tolist()
        labels = [ELEMENTAL_MODE_LABELS.get(mode_id, mode_id) for mode_id in modes]
        vmax = max(1.0, float(np.nanmax(matrix.values)))
        image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.astype(str).tolist())
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = float(matrix.iat[row_index, col_index])
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=_heatmap_text_color(value, vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Mean count per genome")
    fig.suptitle("Category-level elemental mode support comparison", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_base)
    return True


def plot_category_pathway_presence_heatmap(pathway_df, genome_df, output_base, category_column="category", top_n=30):
    ensure_plotting()
    required_cols = {category_column, "PWY_NAME", "genome_label"}
    if pathway_df.empty or not required_cols.issubset(set(pathway_df.columns)):
        return False
    if genome_df.empty or category_column not in genome_df.columns or "genome_label" not in genome_df.columns:
        return False

    total_by_category = (
        genome_df.groupby(category_column)["genome_label"]
        .nunique()
        .rename("total_genomes")
    )
    top_pathways = (
        pathway_df.groupby("PWY_NAME")["genome_label"]
        .nunique()
        .sort_values(ascending=False)
        .head(max(1, int(top_n)))
        .index
        .tolist()
    )
    if not top_pathways:
        return False

    subset = pathway_df.loc[pathway_df["PWY_NAME"].isin(top_pathways)].copy()
    count_matrix = (
        subset.groupby([category_column, "PWY_NAME"])["genome_label"]
        .nunique()
        .reset_index(name="genome_count")
        .pivot_table(
            index=category_column,
            columns="PWY_NAME",
            values="genome_count",
            fill_value=0,
            aggfunc="sum",
        )
    )
    count_matrix = count_matrix.reindex(index=category_order(genome_df, category_column=category_column), columns=top_pathways, fill_value=0)
    fraction_matrix = count_matrix.copy().astype(float)
    for category in fraction_matrix.index:
        denom = float(total_by_category.get(category, 0))
        if denom > 0:
            fraction_matrix.loc[category] = fraction_matrix.loc[category] / denom
        else:
            fraction_matrix.loc[category] = 0.0

    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(10, len(top_pathways) * 0.33), max(5, fraction_matrix.shape[0] * 0.6))
    )
    vmax = max(1.0, float(np.nanmax(fraction_matrix.values)))
    image = ax.imshow(fraction_matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(fraction_matrix.shape[1]))
    ax.set_xticklabels(fraction_matrix.columns.astype(str).tolist(), rotation=90, fontsize=8)
    ax.set_yticks(np.arange(fraction_matrix.shape[0]))
    ax.set_yticklabels(fraction_matrix.index.astype(str).tolist())
    ax.set_xlabel("Pathway")
    ax.set_ylabel("Category")
    ax.set_title("Top pathway presence by category (color=fraction, text=count)")
    for row_index in range(fraction_matrix.shape[0]):
        for col_index in range(fraction_matrix.shape[1]):
            frac = float(fraction_matrix.iat[row_index, col_index])
            count = int(round(float(count_matrix.iat[row_index, col_index])))
            ax.text(
                col_index,
                row_index,
                str(count),
                ha="center",
                va="center",
                fontsize=7,
                color=_heatmap_text_color(frac, vmax),
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction of genomes with pathway")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def compute_functional_evidence_scores(genome_df):
    frame = genome_df.copy()
    quality_metrics = [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "qscore",
    ]
    function_metrics = [
        "informative_annotation_fraction",
        "total_pathways",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
    ]
    available_quality = [metric for metric in quality_metrics if metric in frame.columns]
    available_function = [metric for metric in function_metrics if metric in frame.columns]
    available = available_quality + available_function
    if not available:
        frame["functional_evidence_score"] = np.nan
        frame["quality_evidence_score"] = np.nan
        frame["combined_evidence_score"] = np.nan
        return frame, []
    normalized = pd.DataFrame(index=frame.index)
    for metric in available:
        values = pd.to_numeric(frame[metric], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            normalized[metric] = np.nan
            continue
        min_value = float(valid.min())
        max_value = float(valid.max())
        if max_value <= min_value:
            normalized[metric] = np.where(values.notna(), 1.0, np.nan)
        else:
            normalized[metric] = (values - min_value) / (max_value - min_value)
    if available_quality:
        frame["quality_evidence_score"] = normalized[available_quality].mean(axis=1, skipna=True)
    else:
        frame["quality_evidence_score"] = np.nan
    if available_function:
        frame["functional_evidence_score"] = normalized[available_function].mean(axis=1, skipna=True)
    else:
        frame["functional_evidence_score"] = np.nan
    frame["combined_evidence_score"] = pd.concat(
        [frame["quality_evidence_score"], frame["functional_evidence_score"]],
        axis=1,
    ).mean(axis=1, skipna=True)
    return frame, available


def representative_sort_spec(frame):
    preferred = [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "qscore",
        "functional_evidence_score",
        "informative_annotation_fraction",
        "total_pathways",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
        "genome_id",
    ]
    sort_columns = []
    ascending = []
    for column in preferred:
        if column not in frame.columns:
            continue
        sort_columns.append(column)
        ascending.append(column == "genome_id")
    return sort_columns, ascending


def dedup_species_key(row):
    species_value = str(row.get("Species", "")).strip()
    if taxonomy_species_is_informative(species_value):
        return species_value

    display_rank = str(row.get("taxonomy_display_rank", "")).strip().lower()
    display_value = str(row.get("taxonomy_display_value", "")).strip()
    if display_rank == "species" and taxonomy_species_is_informative(display_value):
        return display_value
    return np.nan


def select_sample_method_deduplicated_genomes(genome_df, category_column="category", sample_column="sample"):
    required = {category_column, sample_column, "genome_id"}
    if genome_df.empty or not required.issubset(set(genome_df.columns)):
        return pd.DataFrame()

    working = genome_df.copy()
    working[sample_column] = working[sample_column].astype(str).str.strip()
    working[category_column] = working[category_column].astype(str).str.strip().map(canonical_method_label)
    working["genome_id"] = working["genome_id"].astype(str).str.strip()
    working = working.loc[
        working[sample_column].ne("")
        & working[category_column].ne("")
        & working["genome_id"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["__species_dedup_key"] = working.apply(dedup_species_key, axis=1)
    working = working.loc[working["__species_dedup_key"].notna()].copy()
    if working.empty:
        return pd.DataFrame()
    sort_columns, ascending = representative_sort_spec(working)
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    deduped = (
        working.groupby([sample_column, category_column, "__species_dedup_key"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    duplicate_count = int(
        deduped.duplicated(subset=[sample_column, category_column, "__species_dedup_key"]).sum()
    )
    if duplicate_count > 0:
        raise ValueError(
            "Sample-method species deduplication failed; duplicate representatives remain."
        )
    return deduped.drop(columns=["__species_dedup_key"], errors="ignore")


def select_sample_representatives(genome_df, category_column="category", sample_column="sample"):
    required = {category_column, sample_column, "genome_id", "functional_evidence_score"}
    if genome_df.empty or not required.issubset(set(genome_df.columns)):
        return pd.DataFrame()
    working = genome_df.copy()
    sort_columns, ascending = representative_sort_spec(working)
    if not sort_columns:
        return pd.DataFrame()
    representatives = (
        working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
        .groupby([sample_column, category_column], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return representatives


def build_method_effectiveness_summary(genome_df, representative_df, category_column="category"):
    if genome_df.empty or category_column not in genome_df.columns:
        return pd.DataFrame()

    working = genome_df.copy()
    for metric in [
        "functional_evidence_score",
        "informative_annotation_fraction",
        "total_pathways",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
        "qscore",
        "integrity_score",
        "recoverability_score",
        "mimag_quality_index",
        "combined_evidence_score",
        "quality_evidence_score",
    ]:
        if metric in working.columns:
            working[metric] = pd.to_numeric(working[metric], errors="coerce")

    if "mimag_tier" in working.columns:
        tiers = working["mimag_tier"].astype(str).str.lower().str.strip()
        working["n_hq"] = tiers.eq("high").astype(int)
        working["n_mq"] = tiers.eq("medium").astype(int)

    agg_spec = {
        "n_genomes": ("genome_id", "nunique"),
    }
    optional_medians = [
        ("median_mimag_quality_index", "mimag_quality_index"),
        ("median_integrity_score", "integrity_score"),
        ("median_recoverability_score", "recoverability_score"),
        ("median_qscore", "qscore"),
        ("median_quality_evidence_score", "quality_evidence_score"),
        ("median_functional_evidence_score", "functional_evidence_score"),
        ("median_combined_evidence_score", "combined_evidence_score"),
        ("median_informative_annotation_fraction", "informative_annotation_fraction"),
        ("median_total_pathways", "total_pathways"),
        ("median_marker_supported_orfs", "marker_supported_orfs"),
        ("median_reference_mode_supported_accessions", "reference_mode_supported_accessions"),
    ]
    for output_column, source_column in optional_medians:
        if source_column in working.columns:
            agg_spec[output_column] = (source_column, "median")
    for optional_count in ["n_hq", "n_mq"]:
        if optional_count in working.columns:
            agg_spec[optional_count] = (optional_count, "sum")

    summary = working.groupby(category_column, dropna=False).agg(**agg_spec).reset_index()
    if not representative_df.empty and category_column in representative_df.columns:
        representative_working = representative_df.copy()
        sort_columns, ascending = representative_sort_spec(representative_working)
        winners = (
            representative_working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
            .groupby("sample", as_index=False, sort=False)
            .head(1)[category_column]
        )
        wins = (
            winners.value_counts(dropna=False)
            .rename_axis(category_column)
            .reset_index(name="sample_win_count")
        )
        wins[category_column] = wins[category_column].astype(str)
        summary[category_column] = summary[category_column].astype(str)
        summary = summary.merge(wins, on=category_column, how="left")
    else:
        summary["sample_win_count"] = 0
    summary["sample_win_count"] = pd.to_numeric(summary["sample_win_count"], errors="coerce").fillna(0).astype(int)
    for optional_count in ["n_hq", "n_mq"]:
        if optional_count in summary.columns:
            summary[optional_count] = pd.to_numeric(summary[optional_count], errors="coerce").fillna(0).astype(int)
    return summary


def plot_sample_representative_heatmap(representative_df, output_base, category_column="category", sample_column="sample"):
    ensure_plotting()
    required = {category_column, sample_column}
    if representative_df.empty or not required.issubset(set(representative_df.columns)):
        return False

    metric_specs = [
        ("mimag_quality_index", "MIMAG quality index"),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
        ("qscore", "Qscore"),
        ("functional_evidence_score", "Functional evidence score"),
        ("informative_annotation_fraction", "Informative annotation fraction"),
        ("total_pathways", "Inferred pathways"),
    ]
    available_specs = [(metric, title) for metric, title in metric_specs if metric in representative_df.columns]
    if not available_specs:
        return False

    panel_matrices = []
    ordered_categories = category_order(representative_df, category_column=category_column)
    for metric, title in available_specs:
        matrix = (
            representative_df.pivot_table(
                index=sample_column,
                columns=category_column,
                values=metric,
                aggfunc="max",
            )
            .fillna(0.0)
            .sort_index()
        )
        matrix = matrix.reindex(columns=ordered_categories, fill_value=0.0)
        if matrix.empty:
            continue
        panel_matrices.append((metric, title, matrix))
    if not panel_matrices:
        return False

    n_cols = 3
    n_rows = int(np.ceil(len(panel_matrices) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, max(matrix.shape[1] for _, _, matrix in panel_matrices) * 1.1), max(6.5, n_rows * 4.5)),
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (metric, title, matrix) in zip(axes, panel_matrices):
        vmax_value = float(np.nanmax(matrix.values)) if np.isfinite(matrix.values).any() else 0.0
        if metric.endswith("_fraction") or metric in {"integrity_score", "recoverability_score", "mimag_quality_index", "functional_evidence_score"}:
            vmax = max(1.0, vmax_value)
            value_fmt = "float2"
        else:
            vmax = max(1.0, vmax_value)
            value_fmt = "float1"
        image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns.astype(str).tolist(), rotation=90)
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index.astype(str).tolist())
        ax.set_xlabel("Category")
        ax.set_ylabel("Sample")
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = float(matrix.iat[row_index, col_index])
                text = f"{value:.2f}" if value_fmt == "float2" else f"{value:.1f}"
                ax.text(
                    col_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_heatmap_text_color(value, vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(title)
    for index in range(len(panel_matrices), len(axes)):
        axes[index].axis("off")
    fig.suptitle("Sample-wise representative genomes: quality then functional evidence", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save_figure(fig, output_base)
    return True


def plot_method_effectiveness_panel(summary_df, output_base, category_column="category"):
    ensure_plotting()
    if summary_df.empty or category_column not in summary_df.columns:
        return False

    working = summary_df.copy()
    counts = pd.to_numeric(working.get("n_genomes", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
    category_values = working[category_column].astype(str).tolist()
    counts_map = {category: int(count) for category, count in zip(category_values, counts)}
    order = ordered_methods(category_values, counts=counts_map)
    working[category_column] = working[category_column].astype(str)
    working = working.set_index(category_column).reindex(order).reset_index()
    x = np.arange(len(order))
    metrics = [
        ("n_genomes", "Genome count"),
        ("sample_win_count", "Sample win count"),
        ("n_hq", "HQ genomes"),
        ("n_mq", "MQ genomes"),
        ("median_mimag_quality_index", "Median MIMAG index"),
        ("median_integrity_score", "Median integrity"),
        ("median_recoverability_score", "Median recoverability"),
        ("median_qscore", "Median qscore"),
        ("median_quality_evidence_score", "Median quality evidence"),
        ("median_functional_evidence_score", "Median functional evidence"),
        ("median_combined_evidence_score", "Median combined evidence"),
        ("median_informative_annotation_fraction", "Median informative fraction"),
        ("median_total_pathways", "Median inferred pathways"),
        ("median_reference_mode_supported_accessions", "Median ref accessions"),
    ]
    available_metrics = [(metric, title) for metric, title in metrics if metric in working.columns]
    if not available_metrics:
        return False
    n_cols = 4
    n_rows = int(np.ceil(len(available_metrics) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(15, len(order) * 1.0), max(8.5, n_rows * 3.8)),
        sharex=False,
        squeeze=False,
    )
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, available_metrics):
        values = pd.to_numeric(working[metric], errors="coerce").fillna(0.0)
        ax.bar(x, values, color="#7f7f7f", edgecolor="black", linewidth=0.6)
        if "fraction" in metric or "score" in metric or metric in {"median_mimag_quality_index", "median_integrity_score", "median_recoverability_score"}:
            annotate_bar_values(ax, values.tolist(), fmt="float2")
        elif "qscore" in metric:
            annotate_bar_values(ax, values.tolist(), fmt="float1")
        else:
            annotate_bar_values(ax, values.tolist(), fmt="int")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90)
        if "fraction" in metric or "score" in metric or metric in {"median_mimag_quality_index", "median_integrity_score", "median_recoverability_score"}:
            ax.set_ylim(0, max(1.02, float(values.max()) + 0.05))
    for index in range(len(available_metrics), len(axes)):
        axes[index].axis("off")
    fig.suptitle("Method-level effectiveness: genome quality and functional evidence", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    save_figure(fig, output_base)
    return True


def run_combined_summary_plots(output_dir, prefix, combined_genome_df, combined_pathway_df):
    output_dir = Path(output_dir)
    wrote_paths = []
    base = output_dir / f"{sanitize_label(prefix)}_compare_category"
    scored_df, score_metrics = compute_functional_evidence_scores(combined_genome_df)
    representative_df = select_sample_representatives(scored_df)
    compact_df = select_sample_method_deduplicated_genomes(scored_df)
    compact_summary_df, _compact_informative_column = build_compact_category_summary(compact_df)
    method_summary_df = build_method_effectiveness_summary(scored_df, representative_df)
    representatives_out = Path(str(base) + "_sample_representatives.tsv")
    compact_set_out = Path(str(base) + "_compact_deduplicated_genomes.tsv")
    compact_summary_out = Path(str(base) + "_compact_summary_table.tsv")
    method_summary_out = Path(str(base) + "_method_effectiveness_summary.tsv")
    representative_df.to_csv(representatives_out, sep="\t", index=False)
    compact_df.to_csv(compact_set_out, sep="\t", index=False)
    compact_summary_df.to_csv(compact_summary_out, sep="\t", index=False)
    method_summary_df.to_csv(method_summary_out, sep="\t", index=False)
    wrote_paths.extend([representatives_out, compact_set_out, compact_summary_out, method_summary_out])

    plot_specs = [
        (
            "compact_summary",
            lambda: plot_combined_category_compact_summary(
                compact_df,
                str(base) + "_compact_summary",
            ),
        ),
        ("distribution_facets", lambda: plot_combined_category_distributions(scored_df, str(base) + "_distribution_facets")),
        ("sample_count_heatmap", lambda: plot_sample_category_count_heatmap(scored_df, str(base) + "_sample_count_heatmap")),
        ("mode_support_heatmaps", lambda: plot_category_mode_support_heatmaps(scored_df, str(base) + "_mode_support_heatmaps")),
        ("pathway_presence_heatmap", lambda: plot_category_pathway_presence_heatmap(combined_pathway_df, scored_df, str(base) + "_pathway_presence_heatmap")),
        ("sample_representative_heatmap", lambda: plot_sample_representative_heatmap(representative_df, str(base) + "_sample_representative_heatmap")),
        ("method_effectiveness_panel", lambda: plot_method_effectiveness_panel(method_summary_df, str(base) + "_method_effectiveness_panel")),
    ]
    for label, plotter in plot_specs:
        wrote = bool(plotter())
        if wrote:
            plot_base = Path(str(base) + f"_{label}")
            wrote_paths.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])
    return wrote_paths


def run_atlas_linked_comparisons(args, output_dir, combined_genome_df, prefix_override=None):
    shared_best_path, annotated_path, shared_best_df, annotated_df = load_atlas_inputs(args)
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
    paired_prefix = sanitize_label(prefix_override) if prefix_override else sanitize_label(args.prefix)
    paired_base = output_dir / f"{paired_prefix}_atlas_paired_{compare_label}"
    matched_out = Path(str(paired_base) + "_matched_genomes.tsv")
    audit_out = Path(str(paired_base) + "_match_audit.tsv")
    species_representatives_out = Path(str(paired_base) + "_species_representatives.tsv")
    pairs_out = Path(str(paired_base) + "_pairs.tsv")
    pair_summary_out = Path(str(paired_base) + "_pair_delta_summary.tsv")

    matched_df.to_csv(matched_out, sep="\t", index=False)
    atlas_match_audit_df.to_csv(audit_out, sep="\t", index=False)
    species_representatives_df = build_species_representative_table(
        matched_df=matched_df,
        compare_column=args.atlas_compare_column,
    )
    if species_representatives_df.empty:
        raise ValueError(
            "No species-classified genomes remained after species representative selection. "
            "Ensure GTDB Species annotations are present and informative."
        )
    species_representatives_df.to_csv(species_representatives_out, sep="\t", index=False)

    paired_df = build_paired_component_table(species_representatives_df, args.atlas_compare_column)
    paired_df.to_csv(pairs_out, sep="\t", index=False)
    pair_summary_df = summarize_paired_deltas(paired_df)
    pair_summary_df.to_csv(pair_summary_out, sep="\t", index=False)

    overlap_entities_df, overlap_methods = build_method_overlap_entities(
        matched_df=species_representatives_df,
        compare_column=args.atlas_compare_column,
    )
    overlap_intersections_df, overlap_method_sizes = summarize_method_overlap_intersections(
        entity_df=overlap_entities_df,
        methods=overlap_methods,
    )
    overlap_entities_out = Path(str(paired_base) + "_method_overlap_entities.tsv")
    overlap_intersections_out = Path(str(paired_base) + "_method_overlap_intersections.tsv")
    species_presence_matrix_out = Path(str(paired_base) + "_species_method_presence_matrix.tsv")
    overlap_entities_df.to_csv(overlap_entities_out, sep="\t", index=False)
    overlap_intersections_df.to_csv(overlap_intersections_out, sep="\t", index=False)
    overlap_entities_df.to_csv(species_presence_matrix_out, sep="\t", index=False)
    taxonomy_heatmap_paths = plot_taxonomy_rank_method_heatmaps(
        genome_df=species_representatives_df,
        output_base_prefix=str(paired_base) + "_taxonomy_rank",
        category_column=args.atlas_compare_column,
        genome_id_column="mp_genome_id",
        ranks=["Phylum", "Class", "Order", "Family", "Genus", "Species"],
    )

    complete_shared_df, _expected_categories = select_complete_shared_representatives(
        matched_df=species_representatives_df,
        compare_column=args.atlas_compare_column,
    )
    complete_shared_out = Path(str(paired_base) + "_complete_shared_representatives.tsv")
    complete_shared_df.to_csv(complete_shared_out, sep="\t", index=False)

    elemental_play_by_play_df = build_pair_elemental_play_by_play(
        matched_df=species_representatives_df,
        compare_column=args.atlas_compare_column,
    )
    elemental_play_by_play_out = Path(str(paired_base) + "_elemental_play_by_play.tsv")
    elemental_play_by_play_df.to_csv(elemental_play_by_play_out, sep="\t", index=False)
    elemental_play_by_play_summary_df = summarize_pair_elemental_play_by_play(elemental_play_by_play_df)
    elemental_play_by_play_summary_out = Path(str(paired_base) + "_elemental_play_by_play_summary.tsv")
    elemental_play_by_play_summary_df.to_csv(elemental_play_by_play_summary_out, sep="\t", index=False)
    focused_pairs_df = select_focus_method_pairs(paired_df)
    focused_pairs_out = Path(str(paired_base) + "_focus_method_pairs.tsv")
    focused_pairs_df.to_csv(focused_pairs_out, sep="\t", index=False)

    wrote_paths = [
        shared_best_path,
        matched_out,
        audit_out,
        species_representatives_out,
        pairs_out,
        pair_summary_out,
        overlap_entities_out,
        overlap_intersections_out,
        species_presence_matrix_out,
        *taxonomy_heatmap_paths,
        complete_shared_out,
        elemental_play_by_play_out,
        elemental_play_by_play_summary_out,
        focused_pairs_out,
    ]
    if annotated_path:
        wrote_paths.append(annotated_path)

    count_plot_base = Path(str(paired_base) + "_count_heatmap")
    delta_plot_base = Path(str(paired_base) + "_delta_heatmaps")
    complete_panel_base = Path(str(paired_base) + "_complete_shared_panel")
    elemental_panel_base = Path(str(paired_base) + "_elemental_play_by_play_panel")
    overlap_panel_base = Path(str(paired_base) + "_method_overlap_upset")
    species_presence_plot_base = Path(str(paired_base) + "_species_method_presence")
    wrote_count = plot_pair_count_heatmap(paired_df, count_plot_base)
    wrote_delta = plot_paired_delta_heatmaps(paired_df, delta_plot_base)
    wrote_overlap = plot_method_overlap_upset(
        intersections_df=overlap_intersections_df,
        methods=overlap_methods,
        method_sizes=overlap_method_sizes,
        output_base=overlap_panel_base,
        title="Method overlap (sample-qualified species representatives)",
    )
    wrote_species_presence = plot_species_method_presence_heatmap(
        entity_df=overlap_entities_df,
        methods=overlap_methods,
        output_base=species_presence_plot_base,
        title="Species presence/absence across methods (sample-qualified)",
    )
    wrote_complete = plot_complete_shared_summary_panel(
        complete_df=complete_shared_df,
        output_base=complete_panel_base,
        category_column=args.atlas_compare_column,
    )
    wrote_elemental = plot_pair_elemental_play_by_play(
        play_df=elemental_play_by_play_df,
        output_base=elemental_panel_base,
    )
    if wrote_count:
        wrote_paths.extend([Path(str(count_plot_base) + ".png"), Path(str(count_plot_base) + ".pdf")])
    if wrote_delta:
        wrote_paths.extend([Path(str(delta_plot_base) + ".png"), Path(str(delta_plot_base) + ".pdf")])
    if wrote_overlap:
        wrote_paths.extend([Path(str(overlap_panel_base) + ".png"), Path(str(overlap_panel_base) + ".pdf")])
    if wrote_species_presence:
        wrote_paths.extend([Path(str(species_presence_plot_base) + ".png"), Path(str(species_presence_plot_base) + ".pdf")])
    if wrote_complete:
        wrote_paths.extend([Path(str(complete_panel_base) + ".png"), Path(str(complete_panel_base) + ".pdf")])
    if wrote_elemental:
        wrote_paths.extend([Path(str(elemental_panel_base) + ".png"), Path(str(elemental_panel_base) + ".pdf")])

    idmap_sample_method_atlas_summaries = []
    idmap_sample_method_metapathways_summaries = []

    for focus_row in focused_pairs_df.to_dict("records"):
        focus_family = str(focus_row.get("focus_family", "")).strip()
        category_a = str(focus_row.get("category_a", "")).strip()
        category_b = str(focus_row.get("category_b", "")).strip()
        if not focus_family or not category_a or not category_b:
            continue
        focus_pair_df = paired_df.loc[
            paired_df["category_a"].astype(str).eq(category_a)
            & paired_df["category_b"].astype(str).eq(category_b)
        ].copy()
        if focus_pair_df.empty:
            continue
        focus_slug = sanitize_label(focus_family)
        focus_pairs_out = Path(str(paired_base) + f"_focus_{focus_slug}_pairs.tsv")
        focus_pair_df.to_csv(focus_pairs_out, sep="\t", index=False)
        wrote_paths.append(focus_pairs_out)

        idmap_audit_df, idmap_pair_df = build_focus_explicit_id_pair_tables(
            source_df=matched_df,
            compare_column=args.atlas_compare_column,
            category_a=category_a,
            category_b=category_b,
        )
        idmap_audit_out = Path(str(paired_base) + f"_focus_{focus_slug}_idmap_audit.tsv")
        idmap_pairs_out = Path(str(paired_base) + f"_focus_{focus_slug}_idmap_pairs.tsv")
        idmap_audit_df.to_csv(idmap_audit_out, sep="\t", index=False)
        idmap_pair_df.to_csv(idmap_pairs_out, sep="\t", index=False)
        wrote_paths.extend([idmap_audit_out, idmap_pairs_out])
        atlas_sample_method_summary_df = summarize_idmap_sample_method_mean_deltas(
            pair_df=idmap_pair_df,
            metric_specs=PAIRED_QC_METRICS,
            focus_family=focus_family,
        )
        metapathways_sample_method_summary_df = summarize_idmap_sample_method_mean_deltas(
            pair_df=idmap_pair_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            focus_family=focus_family,
        )
        focus_summary_out = Path(str(paired_base) + f"_focus_{focus_slug}_idmap_sample_method_mean_delta_summary.tsv")
        pd.concat(
            [atlas_sample_method_summary_df, metapathways_sample_method_summary_df],
            ignore_index=True,
        ).to_csv(focus_summary_out, sep="\t", index=False)
        wrote_paths.append(focus_summary_out)
        if not atlas_sample_method_summary_df.empty:
            idmap_sample_method_atlas_summaries.append(atlas_sample_method_summary_df)
        if not metapathways_sample_method_summary_df.empty:
            idmap_sample_method_metapathways_summaries.append(metapathways_sample_method_summary_df)

        focus_delta_df = build_focus_genome_axis_delta_table(
            focus_pair_df=focus_pair_df,
            category_a=category_a,
            category_b=category_b,
            metric_specs=PAIRED_QC_METRICS + PAIRED_FUNCTION_METRICS,
        )
        focus_delta_out = Path(str(paired_base) + f"_focus_{focus_slug}_genome_axis_deltas.tsv")
        focus_delta_df.to_csv(focus_delta_out, sep="\t", index=False)
        wrote_paths.append(focus_delta_out)

        atlas_focus_base = Path(str(paired_base) + f"_focus_{focus_slug}_genome_x_genome_atlas")
        metapathways_focus_base = Path(str(paired_base) + f"_focus_{focus_slug}_genome_x_genome_metapathways")
        atlas_genome_axis_delta_base = Path(str(paired_base) + f"_focus_{focus_slug}_genome_axis_delta_atlas")
        metapathways_genome_axis_delta_base = Path(
            str(paired_base) + f"_focus_{focus_slug}_genome_axis_delta_metapathways"
        )
        idmap_delta_atlas_base = Path(str(paired_base) + f"_focus_{focus_slug}_idmap_genome_axis_delta_atlas")
        idmap_delta_metapathways_base = Path(
            str(paired_base) + f"_focus_{focus_slug}_idmap_genome_axis_delta_metapathways"
        )
        idmap_before_after_atlas_base = Path(str(paired_base) + f"_focus_{focus_slug}_idmap_before_after_atlas")
        idmap_before_after_metapathways_base = Path(
            str(paired_base) + f"_focus_{focus_slug}_idmap_before_after_metapathways"
        )
        wrote_atlas_focus = plot_focus_pair_metric_scatter_grid(
            paired_df=focus_pair_df,
            metric_specs=PAIRED_QC_METRICS,
            category_a=category_a,
            category_b=category_b,
            output_base=atlas_focus_base,
            title=f"{focus_family} shared genomes: atlas genome-vs-genome comparison",
        )
        wrote_metapathways_focus = plot_focus_pair_metric_scatter_grid(
            paired_df=focus_pair_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            category_a=category_a,
            category_b=category_b,
            output_base=metapathways_focus_base,
            title=f"{focus_family} shared genomes: MetaPathways genome-vs-genome comparison",
        )
        wrote_atlas_genome_axis_delta = plot_focus_genome_axis_delta_heatmap(
            delta_df=focus_delta_df,
            metric_specs=PAIRED_QC_METRICS,
            output_base=atlas_genome_axis_delta_base,
            title=f"{focus_family} shared genomes: atlas deltas by genome ({focus_delta_df['orientation'].iloc[0] if not focus_delta_df.empty else ''})",
        )
        wrote_metapathways_genome_axis_delta = plot_focus_genome_axis_delta_heatmap(
            delta_df=focus_delta_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            output_base=metapathways_genome_axis_delta_base,
            title=f"{focus_family} shared genomes: MetaPathways deltas by genome ({focus_delta_df['orientation'].iloc[0] if not focus_delta_df.empty else ''})",
        )
        wrote_idmap_atlas_delta = plot_focus_genome_axis_delta_heatmap(
            delta_df=idmap_pair_df,
            metric_specs=PAIRED_QC_METRICS,
            output_base=idmap_delta_atlas_base,
            title=f"{focus_family} explicit ID-mapped deltas by genome ({idmap_pair_df['orientation'].iloc[0] if not idmap_pair_df.empty else ''})",
        )
        wrote_idmap_metapathways_delta = plot_focus_genome_axis_delta_heatmap(
            delta_df=idmap_pair_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            output_base=idmap_delta_metapathways_base,
            title=f"{focus_family} explicit ID-mapped MetaPathways deltas ({idmap_pair_df['orientation'].iloc[0] if not idmap_pair_df.empty else ''})",
        )
        wrote_idmap_atlas_before_after = plot_focus_before_after_lines(
            pair_df=idmap_pair_df,
            metric_specs=PAIRED_QC_METRICS,
            output_base=idmap_before_after_atlas_base,
            title=f"{focus_family} explicit ID-mapped before/after (atlas)",
        )
        wrote_idmap_metapathways_before_after = plot_focus_before_after_lines(
            pair_df=idmap_pair_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            output_base=idmap_before_after_metapathways_base,
            title=f"{focus_family} explicit ID-mapped before/after (MetaPathways)",
        )
        if wrote_atlas_focus:
            wrote_paths.extend([Path(str(atlas_focus_base) + ".png"), Path(str(atlas_focus_base) + ".pdf")])
        if wrote_metapathways_focus:
            wrote_paths.extend([Path(str(metapathways_focus_base) + ".png"), Path(str(metapathways_focus_base) + ".pdf")])
        if wrote_atlas_genome_axis_delta:
            wrote_paths.extend(
                [Path(str(atlas_genome_axis_delta_base) + ".png"), Path(str(atlas_genome_axis_delta_base) + ".pdf")]
            )
        if wrote_metapathways_genome_axis_delta:
            wrote_paths.extend(
                [
                    Path(str(metapathways_genome_axis_delta_base) + ".png"),
                    Path(str(metapathways_genome_axis_delta_base) + ".pdf"),
                ]
            )
        if wrote_idmap_atlas_delta:
            wrote_paths.extend([Path(str(idmap_delta_atlas_base) + ".png"), Path(str(idmap_delta_atlas_base) + ".pdf")])
        if wrote_idmap_metapathways_delta:
            wrote_paths.extend(
                [Path(str(idmap_delta_metapathways_base) + ".png"), Path(str(idmap_delta_metapathways_base) + ".pdf")]
            )
        if wrote_idmap_atlas_before_after:
            wrote_paths.extend(
                [Path(str(idmap_before_after_atlas_base) + ".png"), Path(str(idmap_before_after_atlas_base) + ".pdf")]
            )
        if wrote_idmap_metapathways_before_after:
            wrote_paths.extend(
                [
                    Path(str(idmap_before_after_metapathways_base) + ".png"),
                    Path(str(idmap_before_after_metapathways_base) + ".pdf"),
                ]
            )

    if idmap_sample_method_atlas_summaries or idmap_sample_method_metapathways_summaries:
        atlas_summary_all_df = (
            pd.concat(idmap_sample_method_atlas_summaries, ignore_index=True)
            if idmap_sample_method_atlas_summaries else pd.DataFrame()
        )
        metapathways_summary_all_df = (
            pd.concat(idmap_sample_method_metapathways_summaries, ignore_index=True)
            if idmap_sample_method_metapathways_summaries else pd.DataFrame()
        )
        all_summary_df = pd.concat(
            [atlas_summary_all_df, metapathways_summary_all_df],
            ignore_index=True,
        )
        idmap_sample_method_summary_out = Path(str(paired_base) + "_idmap_sample_method_mean_delta_summary.tsv")
        all_summary_df.to_csv(idmap_sample_method_summary_out, sep="\t", index=False)
        wrote_paths.append(idmap_sample_method_summary_out)

        atlas_summary_plot_base = Path(str(paired_base) + "_idmap_sample_method_mean_delta_atlas")
        metapathways_summary_plot_base = Path(str(paired_base) + "_idmap_sample_method_mean_delta_metapathways")
        wrote_atlas_summary_plot = plot_idmap_sample_method_delta_heatmap(
            summary_df=atlas_summary_all_df,
            metric_specs=PAIRED_QC_METRICS,
            output_base=atlas_summary_plot_base,
            title="Explicit ID-mapped sample/method mean deltas (atlas)",
        )
        wrote_metapathways_summary_plot = plot_idmap_sample_method_delta_heatmap(
            summary_df=metapathways_summary_all_df,
            metric_specs=PAIRED_FUNCTION_METRICS,
            output_base=metapathways_summary_plot_base,
            title="Explicit ID-mapped sample/method mean deltas (MetaPathways)",
        )
        if wrote_atlas_summary_plot:
            wrote_paths.extend([Path(str(atlas_summary_plot_base) + ".png"), Path(str(atlas_summary_plot_base) + ".pdf")])
        if wrote_metapathways_summary_plot:
            wrote_paths.extend(
                [Path(str(metapathways_summary_plot_base) + ".png"), Path(str(metapathways_summary_plot_base) + ".pdf")]
            )

    return wrote_paths


def write_combined_tables(
    output_dir,
    prefix,
    manifest_df,
    combined_genome_df,
    combined_annotation_df,
    combined_annotation_quality_df,
    combined_marker_df,
    combined_reference_mode_df,
    combined_elemental_annotation_df,
    combined_elemental_mode_annotation_df,
    combined_elemental_pathway_support_df,
    combined_elemental_mode_pathway_support_df,
    combined_elemental_pathway_df,
    combined_elemental_mode_pathway_df,
    combined_pathway_df,
    combined_pathway_orf_df,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / f"{prefix}_manifest.tsv"
    genome_out = output_dir / f"{prefix}_genome_summary.tsv"
    annotation_out = output_dir / f"{prefix}_annotation_summary.tsv"
    annotation_quality_out = output_dir / f"{prefix}_annotation_quality_summary.tsv"
    marker_out = output_dir / f"{prefix}_marker_summary.tsv"
    reference_mode_out = output_dir / f"{prefix}_reference_mode_summary.tsv"
    elemental_annotation_out = output_dir / f"{prefix}_elemental_annotation_summary.tsv"
    elemental_mode_annotation_out = output_dir / f"{prefix}_elemental_mode_annotation_summary.tsv"
    elemental_pathway_support_out = output_dir / f"{prefix}_elemental_pathway_support_summary.tsv"
    elemental_mode_pathway_support_out = output_dir / f"{prefix}_elemental_mode_pathway_support_summary.tsv"
    elemental_pathway_out = output_dir / f"{prefix}_elemental_pathway_summary.tsv"
    elemental_mode_pathway_out = output_dir / f"{prefix}_elemental_mode_pathway_summary.tsv"
    pathway_out = output_dir / f"{prefix}_pathway_long.tsv"
    pathway_orf_out = output_dir / f"{prefix}_pathway_orf_long.tsv"
    presence_out = output_dir / f"{prefix}_pathway_presence_matrix.tsv"
    score_out = output_dir / f"{prefix}_pathway_score_matrix.tsv"
    coverage_out = output_dir / f"{prefix}_pathway_coverage_matrix.tsv"

    manifest_df.to_csv(manifest_out, sep="\t", index=False)
    combined_genome_df.to_csv(genome_out, sep="\t", index=False)
    combined_annotation_df.to_csv(annotation_out, sep="\t", index=False)
    combined_annotation_quality_df.to_csv(annotation_quality_out, sep="\t", index=False)
    combined_marker_df.to_csv(marker_out, sep="\t", index=False)
    combined_reference_mode_df.to_csv(reference_mode_out, sep="\t", index=False)
    combined_elemental_annotation_df.to_csv(elemental_annotation_out, sep="\t", index=False)
    combined_elemental_mode_annotation_df.to_csv(elemental_mode_annotation_out, sep="\t", index=False)
    combined_elemental_pathway_support_df.to_csv(elemental_pathway_support_out, sep="\t", index=False)
    combined_elemental_mode_pathway_support_df.to_csv(elemental_mode_pathway_support_out, sep="\t", index=False)
    combined_elemental_pathway_df.to_csv(elemental_pathway_out, sep="\t", index=False)
    combined_elemental_mode_pathway_df.to_csv(elemental_mode_pathway_out, sep="\t", index=False)
    combined_pathway_df.to_csv(pathway_out, sep="\t", index=False)
    combined_pathway_orf_df.to_csv(pathway_orf_out, sep="\t", index=False)
    build_combined_pathway_matrix(
        combined_pathway_df.assign(pathway_present=1),
        "pathway_present",
    ).to_csv(presence_out, sep="\t", index=False)
    build_combined_pathway_matrix(combined_pathway_df, "PWY_SCORE").to_csv(score_out, sep="\t", index=False)
    build_combined_pathway_matrix(combined_pathway_df, "reaction_coverage_fraction").to_csv(
        coverage_out,
        sep="\t",
        index=False,
    )
    return [
        manifest_out,
        genome_out,
        annotation_out,
        annotation_quality_out,
        marker_out,
        reference_mode_out,
        elemental_annotation_out,
        elemental_mode_annotation_out,
        elemental_pathway_support_out,
        elemental_mode_pathway_support_out,
        elemental_pathway_out,
        elemental_mode_pathway_out,
        pathway_out,
        pathway_orf_out,
        presence_out,
        score_out,
        coverage_out,
    ]


def main():
    args = build_parser().parse_args()
    manifest = expand_manifest_inputs(read_manifest(args.manifest_tsv))
    manifest_path = Path(args.manifest_tsv).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest_path.parent / "metapathways_batch_summary"
    )
    prefix = sanitize_label(args.prefix)
    organize_roots = {str(output_dir)}

    if (not args.disable_auto_atlas) and (not args.genome_atlas_dir) and (not args.atlas_shared_best_tsv):
        detected_atlas_dir = auto_detect_atlas_dir(manifest_path, output_dir, manifest)
        if detected_atlas_dir is not None:
            args.genome_atlas_dir = str(detected_atlas_dir)
            print(f"[done] auto-detected genome atlas dir: {detected_atlas_dir}")
        else:
            print(
                "[warn] no genome atlas directory auto-detected; atlas-linked and best-ANI subset "
                "comparisons require --genome-atlas-dir (or explicit atlas TSV flags)."
            )

    combined_genome = []
    combined_annotation = []
    combined_annotation_quality = []
    combined_marker = []
    combined_reference_mode = []
    combined_elemental_annotation = []
    combined_elemental_mode_annotation = []
    combined_elemental_pathway_support = []
    combined_elemental_mode_pathway_support = []
    combined_elemental_pathway = []
    combined_elemental_mode_pathway = []
    combined_pathway = []
    combined_pathway_orf = []
    marker_manifest = (
        load_marker_manifest(Path(args.marker_manifest).expanduser().resolve())
        if args.marker_manifest
        else None
    )

    print(f"[start] manifest rows: {len(manifest)}")
    for index, row in manifest.reset_index(drop=True).iterrows():
        sample = row["sample"]
        category = row["category"]
        input_mode, input_dir = detect_input_mode(row["input_dir"])
        individual_output_dir = input_dir / args.individual_subdir
        print(
            f"[start] ({index + 1}/{len(manifest)}) sample '{sample}' "
            f"category '{category}' from {input_dir} [{input_mode}]"
        )
        if input_mode == "results":
            (
                genome_summary,
                pathway_long,
                pathway_orf_long,
                annotation_audit_long,
                pathway_audit_long,
                marker_audit_long,
                reference_mode_audit_long,
            ) = build_summary_tables(
                results_dir=input_dir,
                high_conf_threshold=args.high_confidence_threshold,
                marker_manifest=marker_manifest,
                reference_mappings_dir=args.reference_mappings_dir,
            )
            annotation_summary = build_annotation_source_table(genome_summary)
            annotation_quality_summary = build_annotation_quality_table(genome_summary)
            marker_summary = build_marker_summary_table(genome_summary)
            reference_mode_summary = genome_summary[
                [
                    column for column in genome_summary.columns
                    if column == "genome_id" or column.startswith("reference_mode_")
                ]
            ].copy()
            elemental_annotation_summary = build_elemental_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
            elemental_mode_annotation_summary = build_elemental_mode_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
            elemental_pathway_support_summary = build_elemental_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
            elemental_mode_pathway_support_summary = build_elemental_mode_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
            elemental_pathway_summary = build_elemental_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
            elemental_mode_pathway_summary = build_elemental_mode_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
            write_outputs(
                output_dir=individual_output_dir,
                prefix=prefix,
                genome_summary=genome_summary,
                pathway_long=pathway_long,
                pathway_orf_long=pathway_orf_long,
                annotation_audit_long=annotation_audit_long,
                pathway_audit_long=pathway_audit_long,
                marker_audit_long=marker_audit_long,
                reference_mode_audit_long=reference_mode_audit_long,
                marker_manifest=marker_manifest,
            )
            organize_roots.add(str(individual_output_dir))
            print(f"[done] ({index + 1}/{len(manifest)}) outputs in: {individual_output_dir}")
        else:
            (
                genome_summary,
                annotation_summary,
                annotation_quality_summary,
                marker_summary,
                reference_mode_summary,
                elemental_annotation_summary,
                elemental_mode_annotation_summary,
                elemental_pathway_support_summary,
                elemental_mode_pathway_support_summary,
                elemental_pathway_summary,
                elemental_mode_pathway_summary,
                pathway_long,
                pathway_orf_long,
            ) = load_existing_summary(input_dir)
            print(f"[done] ({index + 1}/{len(manifest)}) loaded existing summary: {input_dir}")

        combined_genome.append(add_context_columns(genome_summary, sample, category, input_dir))
        combined_annotation.append(add_context_columns(annotation_summary, sample, category, input_dir))
        combined_annotation_quality.append(add_context_columns(annotation_quality_summary, sample, category, input_dir))
        combined_marker.append(add_context_columns(marker_summary, sample, category, input_dir))
        combined_reference_mode.append(add_context_columns(reference_mode_summary, sample, category, input_dir))
        combined_elemental_annotation.append(add_context_columns(elemental_annotation_summary, sample, category, input_dir))
        combined_elemental_mode_annotation.append(add_context_columns(elemental_mode_annotation_summary, sample, category, input_dir))
        combined_elemental_pathway_support.append(add_context_columns(elemental_pathway_support_summary, sample, category, input_dir))
        combined_elemental_mode_pathway_support.append(add_context_columns(elemental_mode_pathway_support_summary, sample, category, input_dir))
        combined_elemental_pathway.append(add_context_columns(elemental_pathway_summary, sample, category, input_dir))
        combined_elemental_mode_pathway.append(add_context_columns(elemental_mode_pathway_summary, sample, category, input_dir))
        combined_pathway.append(add_context_columns(pathway_long, sample, category, input_dir))
        combined_pathway_orf.append(add_context_columns(pathway_orf_long, sample, category, input_dir))

    combined_genome_df = (
        pd.concat(combined_genome, ignore_index=True)
        if combined_genome
        else pd.DataFrame()
    )
    combined_annotation_df = (
        pd.concat(combined_annotation, ignore_index=True)
        if combined_annotation
        else pd.DataFrame()
    )
    combined_annotation_quality_df = (
        pd.concat(combined_annotation_quality, ignore_index=True)
        if combined_annotation_quality
        else pd.DataFrame()
    )
    combined_marker_df = (
        pd.concat(combined_marker, ignore_index=True)
        if combined_marker
        else pd.DataFrame()
    )
    combined_reference_mode_df = (
        pd.concat(combined_reference_mode, ignore_index=True)
        if combined_reference_mode
        else pd.DataFrame()
    )
    combined_elemental_annotation_df = (
        pd.concat(combined_elemental_annotation, ignore_index=True)
        if combined_elemental_annotation
        else pd.DataFrame()
    )
    combined_elemental_mode_annotation_df = (
        pd.concat(combined_elemental_mode_annotation, ignore_index=True)
        if combined_elemental_mode_annotation
        else pd.DataFrame()
    )
    combined_elemental_pathway_support_df = (
        pd.concat(combined_elemental_pathway_support, ignore_index=True)
        if combined_elemental_pathway_support
        else pd.DataFrame()
    )
    combined_elemental_mode_pathway_support_df = (
        pd.concat(combined_elemental_mode_pathway_support, ignore_index=True)
        if combined_elemental_mode_pathway_support
        else pd.DataFrame()
    )
    combined_elemental_pathway_df = (
        pd.concat(combined_elemental_pathway, ignore_index=True)
        if combined_elemental_pathway
        else pd.DataFrame()
    )
    combined_elemental_mode_pathway_df = (
        pd.concat(combined_elemental_mode_pathway, ignore_index=True)
        if combined_elemental_mode_pathway
        else pd.DataFrame()
    )
    combined_pathway_df = (
        pd.concat(combined_pathway, ignore_index=True)
        if combined_pathway
        else pd.DataFrame()
    )
    combined_pathway_orf_df = (
        pd.concat(combined_pathway_orf, ignore_index=True)
        if combined_pathway_orf
        else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths = write_combined_tables(
        output_dir=output_dir,
        prefix=prefix,
        manifest_df=manifest,
        combined_genome_df=combined_genome_df,
        combined_annotation_df=combined_annotation_df,
        combined_annotation_quality_df=combined_annotation_quality_df,
        combined_marker_df=combined_marker_df,
        combined_reference_mode_df=combined_reference_mode_df,
        combined_elemental_annotation_df=combined_elemental_annotation_df,
        combined_elemental_mode_annotation_df=combined_elemental_mode_annotation_df,
        combined_elemental_pathway_support_df=combined_elemental_pathway_support_df,
        combined_elemental_mode_pathway_support_df=combined_elemental_mode_pathway_support_df,
        combined_elemental_pathway_df=combined_elemental_pathway_df,
        combined_elemental_mode_pathway_df=combined_elemental_mode_pathway_df,
        combined_pathway_df=combined_pathway_df,
        combined_pathway_orf_df=combined_pathway_orf_df,
    )

    print("[start] combined category summary/comparison plots")
    combined_plot_paths = run_combined_summary_plots(
        output_dir=output_dir,
        prefix=prefix,
        combined_genome_df=combined_genome_df,
        combined_pathway_df=combined_pathway_df,
    )
    written_paths.extend(combined_plot_paths)
    print(f"[done] combined category plots: {len(combined_plot_paths)} files")

    atlas_ready = has_atlas_shared_best(args)
    if args.skip_atlas_linked:
        print("[skip] atlas-linked paired comparisons skipped by --skip-atlas-linked")
    elif atlas_ready:
        print("[start] atlas-linked paired comparisons")
        atlas_paths = run_atlas_linked_comparisons(
            args=args,
            output_dir=output_dir,
            combined_genome_df=combined_genome_df,
        )
        written_paths.extend(atlas_paths)
        print(f"[done] atlas-linked paired outputs: {len(atlas_paths)} files")
    else:
        print(
            "[warn] atlas-linked paired comparisons not run; no shared-best atlas TSV was found. "
            "Provide --genome-atlas-dir or --atlas-shared-best-tsv."
        )

    subset_path, subset_table = (None, pd.DataFrame())
    if args.skip_best_subset:
        print("[skip] best-ANI subset comparisons skipped by --skip-best-subset")
    else:
        subset_path, subset_table = load_best_subset_table(args)
    if subset_path is not None:
        print(f"[start] best-ANI subset from: {subset_path}")
        subset_lookup = build_best_subset_lookup(subset_table, args)
        subset_genome_df = filter_frame_by_best_subset(combined_genome_df, subset_lookup)
        subset_annotation_df = filter_frame_by_best_subset(combined_annotation_df, subset_lookup)
        subset_annotation_quality_df = filter_frame_by_best_subset(combined_annotation_quality_df, subset_lookup)
        subset_marker_df = filter_frame_by_best_subset(combined_marker_df, subset_lookup)
        subset_reference_mode_df = filter_frame_by_best_subset(combined_reference_mode_df, subset_lookup)
        subset_elemental_annotation_df = filter_frame_by_best_subset(combined_elemental_annotation_df, subset_lookup)
        subset_elemental_mode_annotation_df = filter_frame_by_best_subset(combined_elemental_mode_annotation_df, subset_lookup)
        subset_elemental_pathway_support_df = filter_frame_by_best_subset(combined_elemental_pathway_support_df, subset_lookup)
        subset_elemental_mode_pathway_support_df = filter_frame_by_best_subset(
            combined_elemental_mode_pathway_support_df,
            subset_lookup,
        )
        subset_elemental_pathway_df = filter_frame_by_best_subset(combined_elemental_pathway_df, subset_lookup)
        subset_elemental_mode_pathway_df = filter_frame_by_best_subset(combined_elemental_mode_pathway_df, subset_lookup)
        subset_pathway_df = filter_frame_by_best_subset(combined_pathway_df, subset_lookup)
        subset_pathway_orf_df = filter_frame_by_best_subset(combined_pathway_orf_df, subset_lookup)

        if subset_genome_df.empty:
            raise ValueError(
                "Best-ANI subset selection produced zero genomes in combined MetaPathways outputs."
            )

        subset_prefix = sanitize_label(f"{prefix}_bestani")
        subset_paths = write_combined_tables(
            output_dir=output_dir,
            prefix=subset_prefix,
            manifest_df=manifest,
            combined_genome_df=subset_genome_df,
            combined_annotation_df=subset_annotation_df,
            combined_annotation_quality_df=subset_annotation_quality_df,
            combined_marker_df=subset_marker_df,
            combined_reference_mode_df=subset_reference_mode_df,
            combined_elemental_annotation_df=subset_elemental_annotation_df,
            combined_elemental_mode_annotation_df=subset_elemental_mode_annotation_df,
            combined_elemental_pathway_support_df=subset_elemental_pathway_support_df,
            combined_elemental_mode_pathway_support_df=subset_elemental_mode_pathway_support_df,
            combined_elemental_pathway_df=subset_elemental_pathway_df,
            combined_elemental_mode_pathway_df=subset_elemental_mode_pathway_df,
            combined_pathway_df=subset_pathway_df,
            combined_pathway_orf_df=subset_pathway_orf_df,
        )
        written_paths.extend(subset_paths)
        print(f"[done] best-ANI subset tables: {len(subset_paths)} files; genomes={len(subset_genome_df)}")

        print("[start] best-ANI subset category summary/comparison plots")
        subset_plot_paths = run_combined_summary_plots(
            output_dir=output_dir,
            prefix=subset_prefix,
            combined_genome_df=subset_genome_df,
            combined_pathway_df=subset_pathway_df,
        )
        written_paths.extend(subset_plot_paths)
        print(f"[done] best-ANI subset category plots: {len(subset_plot_paths)} files")

        if atlas_ready and (not args.skip_atlas_linked):
            print("[start] best-ANI subset atlas-linked paired comparisons")
            subset_atlas_paths = run_atlas_linked_comparisons(
                args=args,
                output_dir=output_dir,
                combined_genome_df=subset_genome_df,
                prefix_override=subset_prefix,
            )
            written_paths.extend(subset_atlas_paths)
            print(f"[done] best-ANI subset atlas-linked outputs: {len(subset_atlas_paths)} files")
    elif not args.skip_best_subset:
        print(
            "[warn] best-ANI subset comparisons not run; no best-set table was found. "
            "Provide --best-subset-tsv or --genome-atlas-dir containing best_sets_review_selected_genomes.tsv."
        )

    if not args.skip_organize_outputs:
        print("[start] organizing outputs by type/style")
        organized_files_total = 0
        organized_index_paths = []
        for root_dir in sorted(set(organize_roots)):
            organized_files, index_paths = organize_output_tree(root_dir)
            organized_files_total += organized_files
            organized_index_paths.extend(index_paths)
        written_paths.extend(organized_index_paths)
        print(
            f"[done] organized outputs: {organized_files_total} files "
            f"({len(organized_index_paths)} index tables)"
        )

    print(f"[done] combined outputs in: {output_dir}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
