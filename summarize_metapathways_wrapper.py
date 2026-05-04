#!/usr/bin/env python3

import argparse
import glob
import hashlib
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from summarize_metapathways_genomes import (
    MARKER_MANIFEST_PATH,
    DEFAULT_REFERENCE_MAPPINGS_DIR,
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    build_annotation_quality_table,
    build_annotation_source_table,
    build_elemental_mode_summary_table,
    build_elemental_summary_table,
    build_marker_summary_table,
    build_summary_tables,
    ensure_plotting,
    load_allowed_genomes,
    load_marker_manifest,
    resolve_optional_path_arg,
    load_taxonomy_label_lookup,
    save_figure,
    sanitize_label,
    write_outputs,
)

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
SPECIES_LINEAGE_RANKS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


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
        "--workers",
        type=int,
        default=1,
        help=(
            "Worker count passed through to summarize_metapathways_genomes.py-style "
            "per-results summarization. Default: 1"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Legacy global thread override. If >0, this value is used for both "
            "--workers and --prep-workers."
        ),
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=0,
        help=(
            "Heavy preprocessing worker count passed through to "
            "summarize_metapathways_genomes.py. Use 0 to reuse --workers. Default: 0"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Progress update interval for per-results summarization. Default: 10",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose per-results summarization progress messages.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Heartbeat interval for long per-results stages. Default: 60",
    )
    parser.add_argument(
        "--genome-filter-tsv",
        default=None,
        help=(
            "Optional atlas/master-style TSV used to restrict which genomes are "
            "summarized in raw MetaPathways results directories."
        ),
    )
    parser.add_argument(
        "--filter-id-column",
        default=None,
        help=(
            "Optional genome ID column in --genome-filter-tsv. Defaults to "
            "auto-detecting Bin Id, genome_id, or Genome_Id."
        ),
    )
    parser.add_argument(
        "--filter-tier-column",
        default="mimag_tier",
        help="Tier column in --genome-filter-tsv. Default: mimag_tier",
    )
    parser.add_argument(
        "--include-tiers",
        default="medium,high",
        help="Comma-separated tiers to keep from --genome-filter-tsv. Default: medium,high",
    )
    parser.add_argument(
        "--taxonomy-label-tsv",
        default=None,
        help=(
            "Optional atlas/master-style TSV used to replace genome IDs in plot labels "
            "with taxonomy labels. Defaults to --genome-filter-tsv when available."
        ),
    )
    parser.add_argument(
        "--taxonomy-id-column",
        default=None,
        help=(
            "Optional genome ID column in --taxonomy-label-tsv. Defaults to "
            "--filter-id-column, then auto-detection."
        ),
    )
    parser.add_argument(
        "--marker-manifest",
        default=str(MARKER_MANIFEST_PATH),
        help=(
            "Marker manifest passed through to MetaPathways summary building. "
            "Default: <cwd>/config/metabolism_marker_manifest.tsv when available, "
            "otherwise <repo>/config/metabolism_marker_manifest.tsv. "
            "Use 'none' to disable curated marker denominators."
        ),
    )
    parser.add_argument(
        "--reference-mappings-dir",
        default=str(DEFAULT_REFERENCE_MAPPINGS_DIR),
        help=(
            "Normalized reference mapping directory built by "
            "scripts/build_metabolism_reference_mappings.py. Default: "
            "<cwd>/reference_mappings when available, otherwise <repo>/reference_mappings. "
            "Use 'none' to disable reference-term augmentation."
        ),
    )
    parser.add_argument(
        "--reference-chunk-size",
        type=int,
        default=500000,
        help="Rows per chunk when streaming accession reference mappings. Default: 500000",
    )
    parser.add_argument(
        "--reference-progress-rows",
        type=int,
        default=2000000,
        help="Progress update interval while loading reference mappings. Default: 2000000",
    )
    parser.add_argument(
        "--reference-force-full-index",
        action="store_true",
        help="Force loading/building the full accession reference lookup.",
    )
    parser.add_argument(
        "--experimental-mobility-screen",
        action="store_true",
        help=(
            "Run the separate experimental candidate mobility marker screen from "
            "annotation text and write dedicated tables/plots."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-tsv",
        default=None,
        help=(
            "Optional metadata TSV used to resolve genome_type as SAG or MAG for "
            "the experimental mobility screen."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-column",
        default=None,
        help=(
            "Column in --experimental-mobility-genome-type-tsv containing SAG/MAG "
            "labels. If omitted, common column names are tried."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-id-column",
        default=None,
        help=(
            "Genome ID column in --experimental-mobility-genome-type-tsv. "
            "If omitted, common ID columns are tried."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-include-broad-screen",
        action="store_true",
        help=(
            "Also write separate optional broad-screen mobility tables. These are "
            "kept out of the main prevalence table and figure."
        ),
    )
    parser.add_argument(
        "--genome-atlas-dir",
        default=None,
        help=(
            "Optional genome atlas combined output directory. If provided, the wrapper "
            "builds MetaPathways best-of-sample / best-of-best outputs."
        ),
    )
    parser.add_argument(
        "--atlas-compare-column",
        default="category",
        help="Comparison/category column used in atlas component outputs. Default: category",
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
        "--atlas-shared-best-tsv",
        default=None,
        help=(
            "Optional explicit atlas *_shared_best_genomes.tsv path for "
            "atlas-linked shared-best comparisons."
        ),
    )
    parser.add_argument(
        "--atlas-annotated-tsv",
        default=None,
        help=(
            "Optional explicit atlas *_annotated.tsv path for atlas-linked "
            "species_all comparisons."
        ),
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
        "--skip-atlas-linked",
        action="store_true",
        help="Skip atlas-linked category comparison outputs.",
    )
    parser.add_argument(
        "--atlas-linkage-mode",
        choices=["species_all", "shared_best"],
        default="species_all",
        help=(
            "Atlas-linked comparison design. Default: species_all, which keeps "
            "complete species-level lineages across categories."
        ),
    )
    parser.add_argument(
        "--atlas-representative-level",
        choices=["species", "genome"],
        default="species",
        help=(
            "Representative level used only with --atlas-linkage-mode shared_best. "
            "Default: species"
        ),
    )
    parser.add_argument(
        "--skip-atlas-linked-annotation-presence",
        action="store_true",
        help="Skip atlas-linked annotation presence matrices for faster runs.",
    )
    parser.add_argument(
        "--max-heatmap-functions",
        type=int,
        default=250,
        help="Maximum functions/features drawn in atlas-linked heatmaps. Default: 250",
    )
    parser.add_argument(
        "--max-lineage-detail-plots",
        type=int,
        default=25,
        help="Maximum atlas-linked lineage detail heatmaps to draw. Default: 25",
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
        default="low",
        choices=["low", "medium", "high"],
        help=(
            "Optional minimum MIMAG tier filter for atlas-backed best-of selection. "
            "Default: low (keeps all matched genomes). Use 'medium' to keep medium+high or "
            "'high' for high only."
        ),
    )
    parser.add_argument(
        "--disable-auto-atlas",
        action="store_true",
        help=(
            "Disable auto-detection of genome atlas outputs. By default the wrapper "
            "tries to find a sibling/nearby combined_genome_atlas directory for "
            "best-of and best-ANI subset outputs."
        ),
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
    parser.add_argument(
        "--run-denovo-phylogeny",
        action="store_true",
        help=(
            "Build de novo 16S and GTDB-marker phylogenies for all atlas-matched "
            "HQ/nonchimeric genomes and all MQ-or-better/nonchimeric genomes. "
            "Best-representative subsets are exported and built automatically."
        ),
    )
    parser.add_argument(
        "--denovo-phylogeny-threads",
        type=int,
        default=1,
        help=(
            "Threads passed to the phylogeny helper for Barrnap, MAFFT, and "
            "GTDB-Tk. Default: 1"
        ),
    )
    parser.add_argument(
        "--denovo-gtdbtk-data-path",
        default=None,
        help=(
            "Optional GTDBTK_DATA_PATH override passed to the phylogeny helper "
            "when building GTDB marker alignments/trees."
        ),
    )
    return parser


def phylogeny_script_path():
    path = Path(__file__).resolve().with_name("best_set_phylogeny.py")
    if not path.exists():
        raise ValueError(f"Phylogeny script was not found: {path}")
    return path


def run_denovo_phylogeny(python_exe, output_dir, threads, gtdbtk_data_path=None):
    phylogeny_script = phylogeny_script_path()
    cmd = [
        str(python_exe),
        str(phylogeny_script),
        str(Path(output_dir).resolve()),
        "--threads",
        str(max(1, int(threads))),
    ]
    if gtdbtk_data_path:
        cmd.extend(["--gtdbtk-data-path", str(Path(gtdbtk_data_path).expanduser().resolve())])
    print(f"[start] de novo phylogeny: {output_dir}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"De novo phylogeny failed with exit code {completed.returncode}: {' '.join(cmd)}"
        )
    print(f"[done] de novo phylogeny: {output_dir}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


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
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(lambda value: value.strip() if isinstance(value, str) else value)
    frame["sample"] = frame["sample"].astype(str).str.strip()
    frame["category"] = frame["category"].astype(str).str.strip().map(canonical_method_label)
    frame["input_dir"] = frame["input_dir"].astype(str).str.strip()
    if frame[required].eq("").any().any():
        raise ValueError("Manifest contains blank sample/category/input_dir values.")
    return frame


def manifest_row_value(row, column, default=None):
    value = row.get(column, default)
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return default
    return text


def resolve_manifest_row_inputs(row):
    pattern = str(row["input_dir"]).strip()
    matches = sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True))
    if not matches:
        raise FileNotFoundError(f"Manifest pattern did not match any paths: {pattern}")
    return [Path(match).resolve() for match in matches]


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


def matching_id_aliases(value):
    text = str(value).strip()
    aliases = []
    if not text:
        return aliases

    def add(candidate):
        candidate = str(candidate).strip()
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    add(text)
    path_name = Path(text).name
    if path_name != text:
        add(path_name)
        text = path_name

    stem = text
    for suffix in [".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            add(stem)
            break

    if "." in stem:
        parts = stem.split(".")
        for end in range(len(parts) - 1, 0, -1):
            add(".".join(parts[:end]))

    bin_match = re.match(r"^(bin_\d+)", stem)
    if bin_match:
        add(bin_match.group(1))

    for sag_match in re.finditer(r"\b(AB-\d+_[A-Za-z]\d+_AB-\d+(?:_\d+)?)\b", stem):
        add(sag_match.group(1))

    genome_match = re.match(
        r"^(.+?)(?:\.best_match.*|\.majority_rule.*|\.ocsvm.*|\.intersect.*|\.xPG.*)$",
        stem,
    )
    if genome_match:
        add(genome_match.group(1))

    return aliases


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


def effective_worker_settings(args):
    if int(args.threads) > 0:
        return int(args.threads), int(args.threads)
    workers = max(1, int(args.workers))
    prep_workers = int(args.prep_workers) if int(args.prep_workers) > 0 else workers
    return workers, max(1, prep_workers)


def build_wrapper_filter_context(args):
    allowed_genomes = None
    filter_id_column = None
    if args.genome_filter_tsv:
        include_tiers = [
            token.strip()
            for token in str(args.include_tiers).split(",")
            if token.strip()
        ]
        allowed_genomes, filter_id_column = load_allowed_genomes(
            args.genome_filter_tsv,
            filter_id_column=args.filter_id_column,
            filter_tier_column=args.filter_tier_column,
            include_tiers=include_tiers,
        )

    taxonomy_tsv = args.taxonomy_label_tsv or args.genome_filter_tsv
    taxonomy_lookup = None
    taxonomy_id_column = None
    taxonomy_ambiguous_alias_count = 0
    if taxonomy_tsv:
        taxonomy_lookup, taxonomy_id_column, taxonomy_ambiguous_alias_count = load_taxonomy_label_lookup(
            taxonomy_tsv,
            taxonomy_id_column=args.taxonomy_id_column or args.filter_id_column,
        )

    return {
        "allowed_genomes": allowed_genomes,
        "filter_id_column": filter_id_column,
        "taxonomy_lookup": taxonomy_lookup,
        "taxonomy_id_column": taxonomy_id_column,
        "taxonomy_ambiguous_alias_count": taxonomy_ambiguous_alias_count,
        "taxonomy_tsv": taxonomy_tsv,
    }


def build_row_filter_context(args, row):
    filter_tsv = manifest_row_value(row, "genome_filter_tsv", args.genome_filter_tsv)
    filter_id_column = manifest_row_value(row, "filter_id_column", args.filter_id_column)
    filter_tier_column = manifest_row_value(row, "filter_tier_column", args.filter_tier_column)
    include_tiers = manifest_row_value(row, "include_tiers", args.include_tiers)
    taxonomy_tsv = manifest_row_value(row, "taxonomy_label_tsv", args.taxonomy_label_tsv or filter_tsv)
    taxonomy_id_column = manifest_row_value(row, "taxonomy_id_column", args.taxonomy_id_column or filter_id_column)

    allowed_genomes = None
    selected_filter_id_column = None
    if filter_tsv:
        allowed_genomes, selected_filter_id_column = load_allowed_genomes(
            Path(filter_tsv).expanduser().resolve(),
            filter_id_column=filter_id_column,
            filter_tier_column=filter_tier_column,
            include_tiers=[
                token.strip()
                for token in str(include_tiers).split(",")
                if token.strip()
            ],
        )

    taxonomy_lookup = None
    selected_taxonomy_id_column = None
    taxonomy_ambiguous_alias_count = 0
    if taxonomy_tsv:
        taxonomy_lookup, selected_taxonomy_id_column, taxonomy_ambiguous_alias_count = load_taxonomy_label_lookup(
            Path(taxonomy_tsv).expanduser().resolve(),
            taxonomy_id_column=taxonomy_id_column,
        )

    return {
        "filter_tsv": str(Path(filter_tsv).expanduser().resolve()) if filter_tsv else None,
        "allowed_genomes": allowed_genomes,
        "filter_id_column": selected_filter_id_column or filter_id_column,
        "taxonomy_lookup": taxonomy_lookup,
        "taxonomy_tsv": str(Path(taxonomy_tsv).expanduser().resolve()) if taxonomy_tsv else None,
        "taxonomy_id_column": selected_taxonomy_id_column or taxonomy_id_column,
        "taxonomy_ambiguous_alias_count": taxonomy_ambiguous_alias_count,
    }


def default_group_output_dir(input_paths, subdir_name):
    if not input_paths:
        return None
    if len(input_paths) == 1:
        return input_paths[0] / subdir_name
    common_root = Path(os.path.commonpath([str(path) for path in input_paths]))
    return common_root / subdir_name


def resolve_row_output_dir(row, input_paths, args):
    explicit = manifest_row_value(row, "output_dir", None)
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_group_output_dir(input_paths, args.individual_subdir)


def build_atlas_linked_args(args, prefix):
    return SimpleNamespace(
        genome_atlas_dir=args.genome_atlas_dir,
        atlas_shared_best_tsv=getattr(args, "atlas_shared_best_tsv", None),
        atlas_annotated_tsv=getattr(args, "atlas_annotated_tsv", None),
        atlas_prefix="genome_quality",
        atlas_compare_column=args.atlas_compare_column,
        atlas_sample_column=args.atlas_sample_column,
        atlas_genome_id_column=args.atlas_genome_id_column,
        atlas_linkage_mode=args.atlas_linkage_mode,
        atlas_representative_level=args.atlas_representative_level,
        atlas_disable_alias_fallback=bool(args.atlas_disable_alias_fallback),
        min_mimag_tier=args.min_mimag_tier,
        max_heatmap_functions=args.max_heatmap_functions,
        max_lineage_detail_plots=args.max_lineage_detail_plots,
        prefix=prefix,
    )


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
    skip_dirs = {
        "plots",
        "tables",
        "_organized",
        "selected_set",
        "phylogeny",
    }
    for current_dir, dirnames, _ in os.walk(root_path):
        dirnames[:] = [
            name for name in dirnames
            if (
                name not in skip_dirs
                and "matched_lineage_annotation_presence" not in name
                and not name.endswith("linked_comparative_analysis")
            )
        ]
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


def load_existing_summary(summary_dir, marker_manifest=None):
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
    annotation_audit_path = find_preferred_file(
        summary_dir,
        f"{prefix}_elemental_annotation_audit.tsv",
        "*_elemental_annotation_audit.tsv",
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
    annotation_audit_long = (
        pd.read_csv(annotation_audit_path, sep="\t")
        if annotation_audit_path else pd.DataFrame()
    )
    marker_summary = (
        pd.read_csv(marker_summary_path, sep="\t")
        if marker_summary_path else build_marker_summary_table(genome_summary, marker_manifest=marker_manifest)
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
        annotation_audit_long,
    )


def summarize_results_group(input_dirs, group_output_dir, prefix, row_filter_context, marker_manifest, args):
    genome_parts = []
    pathway_parts = []
    pathway_orf_parts = []
    annotation_audit_parts = []
    pathway_audit_parts = []
    marker_audit_parts = []
    reference_mode_audit_parts = []
    wrote_paths = []
    skipped_inputs = []

    total_dirs = len(input_dirs)
    for index, results_dir in enumerate(input_dirs, start=1):
        print(f"[start] ({index}/{total_dirs}) summarizing results directory: {results_dir}")
        try:
            (
                part_genome_summary,
                part_pathway_long,
                part_pathway_orf_long,
                part_annotation_audit_long,
                part_pathway_audit_long,
                part_marker_audit_long,
                part_reference_mode_audit_long,
            ) = build_summary_tables(
                results_dir=results_dir,
                high_conf_threshold=args.high_confidence_threshold,
                allowed_genomes=row_filter_context["allowed_genomes"],
                taxonomy_label_lookup=row_filter_context["taxonomy_lookup"],
                marker_manifest=marker_manifest,
                reference_mappings_dir=args.reference_mappings_dir,
                workers=effective_worker_settings(args)[0],
                progress=not args.quiet,
                progress_interval=args.progress_interval,
                reference_chunk_size=args.reference_chunk_size,
                reference_progress_rows=args.reference_progress_rows,
                heartbeat_seconds=args.heartbeat_seconds,
                reference_force_full_index=bool(args.reference_force_full_index),
                prep_workers=effective_worker_settings(args)[1],
            )
        except ValueError as exc:
            message = str(exc)
            if "No MetaPathways genome records remain after applying --genome-filter-tsv." in message:
                skipped_inputs.append(
                    {
                        "input_dir": str(results_dir),
                        "error_type": exc.__class__.__name__,
                        "error_message": message,
                    }
                )
                print(f"[warn] ({index}/{total_dirs}) skipped after filter: {results_dir}")
                continue
            raise

        individual_output_dir = results_dir / args.individual_subdir
        print(f"[start] ({index}/{total_dirs}) writing individual output set: {individual_output_dir}")
        wrote_paths.extend(
            write_outputs(
                individual_output_dir,
                prefix,
                part_genome_summary,
                part_pathway_long,
                part_pathway_orf_long,
                part_annotation_audit_long,
                part_pathway_audit_long,
                marker_audit_long=part_marker_audit_long,
                reference_mode_audit_long=part_reference_mode_audit_long,
                marker_manifest=marker_manifest,
                progress=not args.quiet,
                heartbeat_seconds=args.heartbeat_seconds,
                experimental_mobility_screen=bool(args.experimental_mobility_screen),
                experimental_mobility_genome_type_tsv=args.experimental_mobility_genome_type_tsv,
                experimental_mobility_genome_type_column=args.experimental_mobility_genome_type_column,
                experimental_mobility_genome_type_id_column=args.experimental_mobility_genome_type_id_column,
                experimental_mobility_include_broad_screen=bool(args.experimental_mobility_include_broad_screen),
            )
        )
        print(f"[done] ({index}/{total_dirs}) genomes={len(part_genome_summary):,} from {results_dir}")

        genome_parts.append(part_genome_summary)
        pathway_parts.append(part_pathway_long)
        pathway_orf_parts.append(part_pathway_orf_long)
        annotation_audit_parts.append(part_annotation_audit_long)
        pathway_audit_parts.append(part_pathway_audit_long)
        marker_audit_parts.append(part_marker_audit_long)
        reference_mode_audit_parts.append(part_reference_mode_audit_long)

    if not genome_parts:
        raise ValueError("No valid MetaPathways summaries remained after grouped results processing.")

    genome_summary = pd.concat(genome_parts, ignore_index=True)
    pathway_long = pd.concat(pathway_parts, ignore_index=True)
    pathway_orf_long = pd.concat(pathway_orf_parts, ignore_index=True)
    annotation_audit_long = pd.concat(annotation_audit_parts, ignore_index=True)
    pathway_audit_long = pd.concat(pathway_audit_parts, ignore_index=True)
    marker_audit_long = pd.concat(marker_audit_parts, ignore_index=True)
    reference_mode_audit_long = pd.concat(reference_mode_audit_parts, ignore_index=True)

    print(f"[start] writing grouped output set: {group_output_dir}")
    wrote_paths.extend(
        write_outputs(
            group_output_dir,
            prefix,
            genome_summary,
            pathway_long,
            pathway_orf_long,
            annotation_audit_long,
            pathway_audit_long,
            marker_audit_long=marker_audit_long,
            reference_mode_audit_long=reference_mode_audit_long,
            marker_manifest=marker_manifest,
            progress=not args.quiet,
            heartbeat_seconds=args.heartbeat_seconds,
            experimental_mobility_screen=bool(args.experimental_mobility_screen),
            experimental_mobility_genome_type_tsv=args.experimental_mobility_genome_type_tsv,
            experimental_mobility_genome_type_column=args.experimental_mobility_genome_type_column,
            experimental_mobility_genome_type_id_column=args.experimental_mobility_genome_type_id_column,
            experimental_mobility_include_broad_screen=bool(args.experimental_mobility_include_broad_screen),
        )
    )
    print(f"[done] grouped MetaPathways outputs in: {group_output_dir}")

    annotation_summary = build_annotation_source_table(genome_summary)
    annotation_quality_summary = build_annotation_quality_table(genome_summary)
    marker_summary = build_marker_summary_table(genome_summary, marker_manifest=marker_manifest)
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

    return {
        "genome_summary": genome_summary,
        "annotation_summary": annotation_summary,
        "annotation_quality_summary": annotation_quality_summary,
        "marker_summary": marker_summary,
        "reference_mode_summary": reference_mode_summary,
        "elemental_annotation_summary": elemental_annotation_summary,
        "elemental_mode_annotation_summary": elemental_mode_annotation_summary,
        "elemental_pathway_support_summary": elemental_pathway_support_summary,
        "elemental_mode_pathway_support_summary": elemental_mode_pathway_support_summary,
        "elemental_pathway_summary": elemental_pathway_summary,
        "elemental_mode_pathway_summary": elemental_mode_pathway_summary,
        "pathway_long": pathway_long,
        "pathway_orf_long": pathway_orf_long,
        "annotation_audit_long": annotation_audit_long,
        "wrote_paths": wrote_paths,
        "skipped_inputs": skipped_inputs,
    }


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
    if preferred_token:
        for candidate in [
            atlas_base / preferred_token,
            atlas_base / "tables" / "other" / preferred_token,
            atlas_base / "tables" / "components" / preferred_token,
        ]:
            if candidate.is_file():
                return candidate

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


def load_atlas_component_members(args):
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve() if args.genome_atlas_dir else None
    component_members_path = choose_atlas_file(
        atlas_dir=atlas_dir,
        explicit_path=None,
        pattern="best_sets_review_component_members.tsv",
        preferred_token="best_sets_review_component_members.tsv",
        required=True,
    )
    component_members_df = pd.read_csv(component_members_path, sep="\t")
    return component_members_path, component_members_df


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
        has_component_members = any(path.is_file() for path in candidate.rglob("best_sets_review_component_members.tsv"))
        has_subset = any(path.is_file() for path in candidate.rglob("best_sets_review_selected_genomes.tsv"))
        if has_component_members or has_subset:
            return candidate
    return None


def has_atlas_component_members(args):
    if not args.genome_atlas_dir:
        return False
    atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve()
    if not atlas_dir.exists():
        return False
    return any(path.is_file() for path in atlas_dir.rglob("best_sets_review_component_members.tsv"))


def resolve_best_subset_path(args):
    if args.best_subset_tsv:
        resolved = Path(args.best_subset_tsv).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Best-subset TSV was not found: {resolved}")
        return resolved
    if args.genome_atlas_dir:
        atlas_dir = Path(args.genome_atlas_dir).expanduser().resolve()
        direct_candidate = atlas_dir / "best_sets_review_selected_genomes.tsv"
        if direct_candidate.exists():
            return direct_candidate
        preferred_candidate = atlas_dir / "tables" / "selected" / "best_sets_review_selected_genomes.tsv"
        if preferred_candidate.exists():
            return preferred_candidate
        matches = sorted(path for path in atlas_dir.rglob("best_sets_review_selected_genomes.tsv") if path.is_file())
        if matches:
            return matches[0]
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
            aliases.update(matching_id_aliases(value))
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
        aliases = set(matching_id_aliases(row.get(genome_column, "")))
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


def best_set_has_16s(frame):
    if "contains_16S" in frame.columns:
        contains_series = pd.to_numeric(frame["contains_16S"], errors="coerce").fillna(0)
        return contains_series.gt(0).astype(int)
    if "16S_rRNA" in frame.columns:
        rrna_series = pd.to_numeric(frame["16S_rRNA"], errors="coerce").fillna(0)
        return rrna_series.gt(0).astype(int)
    return pd.Series(0, index=frame.index, dtype=int)


def coarse_quality_bin(series, step):
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric / float(step)).round().fillna(float("-inf"))


def gunc_assessment_rank(series):
    if series is None:
        return pd.Series(dtype=int)
    rank_map = {
        "likely_not_chimeric": 0,
        "low_reference_uncertain": 1,
        "unscored": 1,
        "credible_chimeric_signal": 2,
    }
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unscored", "none": "unscored", "null": "unscored"})
    )
    return cleaned.map(rank_map).fillna(2).astype(int)


def gunc_confident_call_mask(series):
    if series is None:
        return pd.Series(dtype=bool)
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unscored", "none": "unscored", "null": "unscored"})
    )
    return cleaned.isin({"likely_not_chimeric", "credible_chimeric_signal"})


def metapathways_best_set_sort_spec(frame):
    preferred = [
        "__mimag_quality_bin",
        "__gunc_strict_rank",
        "__integrity_bin",
        "__recoverability_bin",
        "__has_16s",
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "mp_informative_annotation_fraction",
        "mp_informative_annotation_orfs",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
        "mp_total_pathways",
        "qscore",
        "mp_genome_id",
    ]
    sort_columns = []
    ascending = []
    for column in preferred:
        if column not in frame.columns:
            continue
        sort_columns.append(column)
        ascending.append(column in {"__gunc_strict_rank", "mp_genome_id"})
    return sort_columns, ascending


def prepare_metapathways_best_set_candidates(matched_df, compare_column):
    if matched_df.empty:
        return pd.DataFrame()

    required = {"component_id", "mp_genome_id", compare_column}
    if not required.issubset(set(matched_df.columns)):
        return pd.DataFrame()

    working = matched_df.copy()
    numeric_columns = [
        "mimag_quality_index",
        "integrity_score",
        "recoverability_score",
        "qscore",
        "mp_informative_annotation_orfs",
        "mp_informative_annotation_fraction",
        "mp_total_pathways",
        "mp_marker_supported_orfs",
        "mp_reference_mode_supported_accessions",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working["__mimag_quality_bin"] = coarse_quality_bin(working.get("mimag_quality_index"), 0.02)
    if "gunc_strict_assessment" in working.columns:
        working["__gunc_strict_rank"] = gunc_assessment_rank(working["gunc_strict_assessment"])
    working["__integrity_bin"] = coarse_quality_bin(working.get("integrity_score"), 0.02)
    working["__recoverability_bin"] = coarse_quality_bin(working.get("recoverability_score"), 0.02)
    working["__has_16s"] = best_set_has_16s(working)

    working["sample"] = working.apply(pair_sample_value, axis=1)
    working["category"] = (
        working[compare_column]
        .astype(str)
        .str.strip()
        .map(canonical_method_label)
    )
    working["Genome_Id"] = working["mp_genome_id"].astype(str).str.strip()
    working["genome_id"] = working["mp_genome_id"].astype(str).str.strip()
    working["atlas_genome_id"] = working["_atlas_genome_id"].astype(str).str.strip() if "_atlas_genome_id" in working.columns else ""
    working["component_id"] = working["component_id"].astype(str).str.strip()
    working = working.loc[
        working["sample"].astype(str).str.strip().ne("")
        & working["category"].astype(str).str.strip().ne("")
        & working["Genome_Id"].astype(str).str.strip().ne("")
        & working["component_id"].astype(str).str.strip().ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame()

    sort_columns, ascending = metapathways_best_set_sort_spec(working)
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    return working.reset_index(drop=True)


def _build_best_set_scope_table(working, group_columns, scope_label):
    if working.empty:
        return pd.DataFrame()
    ranked = working.copy()
    ranked["best_set_scope"] = scope_label
    if scope_label == "sample":
        ranked["best_set_name"] = ranked["sample"].astype(str)
    else:
        ranked["best_set_name"] = "global"
    ranked["selection_rank_within_set"] = ranked.groupby(group_columns, dropna=False).cumcount() + 1
    ranked["selected"] = ranked["selection_rank_within_set"].eq(1)
    ranked["best_set_label"] = np.where(
        ranked["best_set_scope"].astype(str).eq("sample"),
        "best_of_sample",
        "best_of_best",
    )
    ranked["selection_metric"] = (
        "coarse(mimag_quality_index)>gunc(class_gate_only)>"
        "coarse(integrity_score,recoverability_score)>16S_presence>"
        "mimag_quality_index>integrity_score>recoverability_score>"
        "mp_informative_annotation_fraction>mp_informative_annotation_orfs>"
        "mp_marker_supported_orfs>mp_reference_mode_supported_accessions>"
        "mp_total_pathways>qscore"
    )
    return ranked


def build_metapathways_best_set_tables(matched_df, compare_column):
    working = prepare_metapathways_best_set_candidates(matched_df, compare_column)
    if working.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if {"best_set_scope", "best_set_name"}.issubset(set(working.columns)):
        scope_values = working["best_set_scope"].astype(str).str.strip().str.lower()
        sample_working = working.loc[scope_values.eq("sample")].copy()
        global_working = working.loc[scope_values.eq("global")].copy()
        if sample_working.empty:
            sample_working = working.copy()
        if global_working.empty:
            global_working = working.copy()
    else:
        sample_working = working.copy()
        global_working = working.copy()

    sample_review_df = _build_best_set_scope_table(
        working=sample_working,
        group_columns=["best_set_name", "component_id"],
        scope_label="sample",
    )
    global_review_df = _build_best_set_scope_table(
        working=global_working,
        group_columns=["component_id"],
        scope_label="global",
    )
    review_df = pd.concat([sample_review_df, global_review_df], ignore_index=True)
    if review_df.empty:
        return review_df, pd.DataFrame(), pd.DataFrame()
    helper_columns = [column for column in review_df.columns if str(column).startswith("__")]
    if helper_columns:
        review_df = review_df.drop(columns=helper_columns, errors="ignore")

    selected_df = review_df.loc[review_df["selected"].fillna(False)].copy().reset_index(drop=True)
    contribution_df = (
        selected_df.groupby(["best_set_scope", "best_set_name", "best_set_label", "category"], dropna=False)
        .size()
        .reset_index(name="selected_genome_count")
    )
    if not contribution_df.empty:
        contribution_df["category"] = contribution_df["category"].astype(str).str.strip().map(canonical_method_label)
        category_counts = (
            contribution_df.groupby("category")["selected_genome_count"].sum().to_dict()
        )
        method_order = ordered_methods(contribution_df["category"].astype(str).tolist(), counts=category_counts)
        order_map = {category: index for index, category in enumerate(method_order)}
        contribution_df["__method_order"] = contribution_df["category"].map(order_map).fillna(len(order_map)).astype(int)
        contribution_df = (
            contribution_df.sort_values(
                by=["best_set_scope", "best_set_name", "__method_order", "category"],
                ascending=[True, True, True, True],
                kind="mergesort",
            )
            .drop(columns=["__method_order"])
            .reset_index(drop=True)
        )
    return review_df, selected_df, contribution_df


def run_metapathways_best_sets(
    args,
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
    combined_annotation_audit_df=None,
):
    component_members_path, atlas_df = load_atlas_component_members(args)
    if atlas_df.empty:
        raise ValueError(f"Atlas component-member table is empty: {component_members_path}")
    mp_df, exact_map, alias_map = build_metapathways_lookup(combined_genome_df)
    _atlas_match_audit_df, matched_df = match_atlas_genomes_to_metapathways(
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
                "--min-mimag-tier was provided, but 'mimag_tier' is missing from atlas-backed matched genomes."
            )
        matched_df["_mimag_tier_norm"] = matched_df["mimag_tier"].astype(str).str.lower().str.strip()
        matched_df["_mimag_tier_value"] = matched_df["_mimag_tier_norm"].map(tier_order)
        matched_df = matched_df.loc[matched_df["_mimag_tier_value"].fillna(-1).ge(min_value)].copy()
        matched_df = matched_df.drop(columns=["_mimag_tier_norm", "_mimag_tier_value"], errors="ignore")
    if matched_df.empty:
        raise ValueError("No atlas-backed MetaPathways genomes are available for best-set selection.")

    review_df, selected_df, contribution_df = build_metapathways_best_set_tables(
        matched_df=matched_df,
        compare_column=args.atlas_compare_column,
    )
    if selected_df.empty:
        raise ValueError("MetaPathways best-set selection produced zero selected genomes.")

    output_dir = Path(output_dir)
    wrote_paths = []
    review_out = output_dir / "best_sets_review_candidates.tsv"
    selected_out = output_dir / "best_sets_review_selected_genomes.tsv"
    contribution_out = output_dir / "best_sets_review_category_contributions.tsv"
    review_df.to_csv(review_out, sep="\t", index=False)
    selected_df.to_csv(selected_out, sep="\t", index=False)
    contribution_df.to_csv(contribution_out, sep="\t", index=False)
    wrote_paths.extend([review_out, selected_out, contribution_out])
    review_selection_overall_frames = []
    review_selection_pairwise_frames = []

    for (scope_value, set_name), selected_rows in selected_df.groupby(["best_set_scope", "best_set_name"], dropna=False):
        scope_text = str(scope_value).strip().lower()
        set_name_text = str(set_name).strip()
        if scope_text == "sample":
            subset_dir = output_dir / "best_of_sample" / sanitize_label(set_name_text)
        else:
            subset_dir = output_dir / "best_of_best"
        subset_dir.mkdir(parents=True, exist_ok=True)
        selected_set_dir = subset_dir / "selected_set"
        selected_rows = selected_rows.copy().reset_index(drop=True)
        selected_rows, selected_fasta_paths = export_selected_set_fastas(selected_rows, selected_set_dir)

        subset_lookup = build_best_subset_lookup(selected_rows.copy(), args)
        subset_genome_df = filter_frame_by_best_subset(combined_genome_df, subset_lookup)
        subset_annotation_df = filter_frame_by_best_subset(combined_annotation_df, subset_lookup)
        subset_annotation_quality_df = filter_frame_by_best_subset(combined_annotation_quality_df, subset_lookup)
        subset_annotation_audit_df = filter_frame_by_best_subset(combined_annotation_audit_df, subset_lookup)
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
            continue

        subset_prefix = sanitize_label(prefix)
        subset_selected_out = subset_dir / "selected_genomes.tsv"
        subset_contribution_out = subset_dir / "category_contributions.tsv"
        subset_selected_count_out = subset_dir / "selected_category_count_summary.tsv"
        subset_selection_pref_out = subset_dir / "selection_preference_by_category.tsv"
        subset_selection_pairwise_out = subset_dir / "selection_preference_pairwise.tsv"
        selected_rows.to_csv(subset_selected_out, sep="\t", index=False)
        subset_contribution_df = contribution_df.loc[
            contribution_df["best_set_scope"].astype(str).eq(str(scope_value))
            & contribution_df["best_set_name"].astype(str).eq(str(set_name))
        ].copy()
        subset_contribution_df.to_csv(subset_contribution_out, sep="\t", index=False)
        selected_count_df = summarize_selected_category_counts(selected_rows, category_column="category")
        selected_count_df.to_csv(subset_selected_count_out, sep="\t", index=False)
        subset_review_df = review_df.loc[
            review_df["best_set_scope"].astype(str).eq(str(scope_value))
            & review_df["best_set_name"].astype(str).eq(str(set_name))
        ].copy()
        subset_selection_pref_df, subset_selection_pairwise_df = summarize_selection_preferences(
            subset_review_df,
            category_column="category",
        )
        subset_selection_pref_df.to_csv(subset_selection_pref_out, sep="\t", index=False)
        subset_selection_pairwise_df.to_csv(subset_selection_pairwise_out, sep="\t", index=False)
        wrote_paths.extend(
            [
                subset_selected_out,
                subset_contribution_out,
                subset_selected_count_out,
                subset_selection_pref_out,
                subset_selection_pairwise_out,
                *selected_fasta_paths,
            ]
        )
        if not subset_selection_pref_df.empty:
            review_selection_overall = subset_selection_pref_df.copy()
            review_selection_overall["best_set_scope"] = str(scope_value)
            review_selection_overall["best_set_name"] = str(set_name)
            review_selection_overall_frames.append(review_selection_overall)
        if not subset_selection_pairwise_df.empty:
            review_selection_pairwise = subset_selection_pairwise_df.copy()
            review_selection_pairwise["best_set_scope"] = str(scope_value)
            review_selection_pairwise["best_set_name"] = str(set_name)
            review_selection_pairwise_frames.append(review_selection_pairwise)

        subset_paths = write_combined_tables(
            output_dir=subset_dir,
            prefix=subset_prefix,
            manifest_df=manifest_df,
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
            combined_annotation_audit_df=subset_annotation_audit_df,
        )
        wrote_paths.extend(subset_paths)
        plot_paths = run_combined_summary_plots(
            output_dir=subset_dir,
            prefix=subset_prefix,
            combined_genome_df=subset_genome_df,
            combined_pathway_df=subset_pathway_df,
        )
        wrote_paths.extend(plot_paths)
        selected_count_plot_base = subset_dir / f"{subset_prefix}_selected_category_count_summary"
        if plot_selected_category_counts(selected_count_df, str(selected_count_plot_base), category_column="category"):
            wrote_paths.extend([Path(str(selected_count_plot_base) + ".png"), Path(str(selected_count_plot_base) + ".pdf")])
        selection_pref_plot_base = subset_dir / f"{subset_prefix}_selection_preference_summary"
        if plot_selection_preference_summary(
            subset_selection_pref_df,
            subset_selection_pairwise_df,
            str(selection_pref_plot_base),
            category_column="category",
        ):
            wrote_paths.extend([Path(str(selection_pref_plot_base) + ".png"), Path(str(selection_pref_plot_base) + ".pdf")])

    if review_selection_overall_frames:
        review_selection_overall_out = output_dir / "best_sets_review_selection_preference_by_category.tsv"
        pd.concat(review_selection_overall_frames, ignore_index=True).to_csv(
            review_selection_overall_out,
            sep="\t",
            index=False,
        )
        wrote_paths.append(review_selection_overall_out)
    if review_selection_pairwise_frames:
        review_selection_pairwise_out = output_dir / "best_sets_review_selection_preference_pairwise.tsv"
        pd.concat(review_selection_pairwise_frames, ignore_index=True).to_csv(
            review_selection_pairwise_out,
            sep="\t",
            index=False,
        )
        wrote_paths.append(review_selection_pairwise_out)
    return [path for path in wrote_paths if path]


def gunc_nonchimeric_mask(frame):
    if frame.empty:
        return pd.Series(dtype=bool)
    for column in ["gunc_strict_assessment", "gunc_assessment"]:
        if column in frame.columns:
            cleaned = (
                frame[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .replace({"nan": "", "none": "", "null": ""})
            )
            return ~cleaned.eq("credible_chimeric_signal")
    for column in ["gunc_strict_chimera", "gunc_credible_chimeric_signal"]:
        if column in frame.columns:
            cleaned = frame[column].astype(str).str.strip().str.lower()
            return ~cleaned.isin({"true", "1", "yes", "y"})
    raise ValueError(
        "Cannot build de novo phylogeny cohorts because no GUNC chimera-assessment column "
        "was found in atlas-matched genomes."
    )


def prepare_denovo_phylogeny_candidates(matched_df):
    if matched_df.empty:
        return pd.DataFrame()
    if "mimag_tier" not in matched_df.columns:
        raise ValueError("Cannot build de novo phylogeny cohorts because 'mimag_tier' is missing.")

    working = matched_df.copy().reset_index(drop=True)
    working["_tier_norm"] = working["mimag_tier"].astype(str).str.strip().str.lower()
    working["_gunc_nonchimeric"] = gunc_nonchimeric_mask(working)
    working = working.loc[
        working["_tier_norm"].isin({"high", "medium"})
        & working["_gunc_nonchimeric"]
    ].copy()
    if working.empty:
        return working

    working["sample"] = working.apply(pair_sample_value, axis=1)
    if "mp_category" in working.columns:
        category_series = working["mp_category"]
    elif "category" in working.columns:
        category_series = working["category"]
    else:
        category_series = pd.Series("", index=working.index)
    working["category"] = category_series.astype(str).str.strip().map(canonical_method_label)

    if "mp_genome_id" in working.columns:
        genome_series = working["mp_genome_id"]
    elif "genome_id" in working.columns:
        genome_series = working["genome_id"]
    else:
        genome_series = pd.Series("", index=working.index)
    working["Genome_Id"] = genome_series.astype(str).str.strip()
    working["genome_id"] = working["Genome_Id"]
    working["atlas_genome_id"] = working["_atlas_genome_id"].astype(str).str.strip() if "_atlas_genome_id" in working.columns else ""
    working["denovo_phylogeny_selection"] = np.where(
        working["_tier_norm"].eq("high"),
        "hq_nonchimeric",
        "mq_nonchimeric",
    )
    working["denovo_phylogeny_selection_rule"] = (
        "hq_nonchimeric: mimag_tier=high + not confirmed chimeric by GUNC; "
        "mq_nonchimeric: mimag_tier in {medium,high} + not confirmed chimeric by GUNC; "
        "best-representative subsets are also exported for each cohort; "
        "keeps sample/category variants; removes exact duplicate selected rows"
    )
    working = working.loc[
        working["sample"].astype(str).str.strip().ne("")
        & working["category"].astype(str).str.strip().ne("")
        & working["Genome_Id"].astype(str).str.strip().ne("")
    ].copy()
    return working


def annotate_denovo_best_representatives(candidates, output_dir):
    working = candidates.copy()
    working["best_representative"] = "false"
    if working.empty:
        return working

    best_table = Path(output_dir) / "best_of_best" / "selected_set" / "master.tsv"
    if not best_table.exists():
        return working

    best_df = pd.read_csv(best_table, sep="\t", dtype=str).fillna("")
    required = ["sample", "category", "Genome_Id"]
    if not all(column in best_df.columns for column in required):
        return working

    best_keys = {
        (
            str(row["sample"]).strip().lower(),
            canonical_method_label(str(row["category"]).strip()).lower(),
            str(row["Genome_Id"]).strip().lower(),
        )
        for _, row in best_df.iterrows()
    }
    candidate_keys = zip(
        working["sample"].astype(str).str.strip().str.lower(),
        working["category"].astype(str).str.strip().map(canonical_method_label).str.lower(),
        working["Genome_Id"].astype(str).str.strip().str.lower(),
    )
    working["best_representative"] = ["true" if key in best_keys else "false" for key in candidate_keys]
    return working


def build_denovo_phylogeny_sets(args, output_dir, combined_genome_df):
    component_members_path, atlas_df = load_atlas_component_members(args)
    if atlas_df.empty:
        raise ValueError(f"Atlas component-member table is empty: {component_members_path}")
    mp_df, exact_map, alias_map = build_metapathways_lookup(combined_genome_df)
    match_audit_df, matched_df = match_atlas_genomes_to_metapathways(
        atlas_df=atlas_df,
        mp_df=mp_df,
        exact_map=exact_map,
        alias_map=alias_map,
        args=args,
    )
    candidates = prepare_denovo_phylogeny_candidates(matched_df)
    candidates = annotate_denovo_best_representatives(candidates, output_dir)

    output_dir = Path(output_dir)
    phylogeny_root = output_dir / "denovo_phylogeny"
    phylogeny_root.mkdir(parents=True, exist_ok=True)
    wrote_paths = []

    audit_path = phylogeny_root / "atlas_match_audit.tsv"
    match_audit_df.to_csv(audit_path, sep="\t", index=False)
    wrote_paths.append(audit_path)

    candidate_path = phylogeny_root / "denovo_phylogeny_candidates.tsv"
    candidates.to_csv(candidate_path, sep="\t", index=False)
    wrote_paths.append(candidate_path)

    best_mask = candidates["best_representative"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    hq_mask = candidates["_tier_norm"].eq("high")
    mq_or_better_mask = candidates["_tier_norm"].isin({"medium", "high"})
    cohort_filters = [
        ("hq_nonchimeric", hq_mask),
        ("hq_nonchimeric_best_representatives", hq_mask & best_mask),
        ("mq_nonchimeric", mq_or_better_mask),
        ("mq_nonchimeric_best_representatives", mq_or_better_mask & best_mask),
    ]
    for cohort_name, cohort_mask in cohort_filters:
        cohort_dir = phylogeny_root / cohort_name
        selected_set_dir = cohort_dir / "selected_set"
        cohort_df = candidates.loc[cohort_mask].copy().reset_index(drop=True)
        cohort_df = deduplicate_phylogeny_rows(cohort_df)
        if cohort_df.empty:
            cohort_dir.mkdir(parents=True, exist_ok=True)
            empty_path = cohort_dir / "selected_genomes.tsv"
            cohort_df.to_csv(empty_path, sep="\t", index=False)
            wrote_paths.append(empty_path)
            continue
        cohort_df, fasta_paths = export_selected_set_fastas(cohort_df, selected_set_dir)
        selected_out = cohort_dir / "selected_genomes.tsv"
        cohort_df.to_csv(selected_out, sep="\t", index=False)
        wrote_paths.extend([selected_out] + fasta_paths)

    return [path for path in wrote_paths if path]


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
    working["_alias_set"] = working["_genome_key"].apply(lambda value: set(matching_id_aliases(value)))

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
    for optional_column in [
        "SAG_ID",
        "Bin Id",
        "genome_id",
        "fasta_path",
        "ani_record_id",
        "ani_fasta_path",
        "pre_ani_bin_key",
    ]:
        if optional_column in atlas_df.columns and optional_column not in atlas_alias_columns:
            atlas_alias_columns.append(optional_column)

    match_rows = []
    matched_mp_indices = []
    for atlas_idx, row in atlas_df.reset_index(drop=True).iterrows():
        sample_key = str(row.get(sample_column, "")).strip()
        category_key = canonical_method_label(str(row.get(compare_column, "")).strip())
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
                    alias_pool.update(matching_id_aliases(raw_identifier))
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
    for column in ["linked_sample", "mp_sample", "sample", "_atlas_sample"]:
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


def taxonomy_value_is_informative(value):
    text = str(value).strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"nan", "none", "na", "n/a", "null", "unknown"}:
        return False
    if lowered.startswith("unclassified"):
        return False
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


def benjamini_hochberg_adjust(pvalues):
    numeric = pd.to_numeric(pd.Series(pvalues), errors="coerce")
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return adjusted
    order = valid.sort_values().index.tolist()
    ordered = valid.loc[order].to_numpy(dtype=float)
    n = float(len(ordered))
    bh = np.empty(len(ordered), dtype=float)
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        rank = reverse_index + 1.0
        candidate = (ordered[reverse_index] * n) / rank
        running = min(running, candidate)
        bh[reverse_index] = min(1.0, running)
    adjusted.loc[order] = bh
    return adjusted


def significance_stars(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
    if pd.isna(numeric):
        return ""
    if numeric < 0.0001:
        return "****"
    if numeric < 0.001:
        return "***"
    if numeric < 0.01:
        return "**"
    if numeric < 0.05:
        return "*"
    return ""


def exact_binomial_test_two_sided(k, n):
    try:
        n_int = int(n)
        k_int = int(k)
    except Exception:
        return float("nan")
    if n_int <= 0:
        return float("nan")
    pmf_obs = math.comb(n_int, k_int) / (2.0 ** n_int)
    total = 0.0
    for value in range(n_int + 1):
        pmf_value = math.comb(n_int, value) / (2.0 ** n_int)
        if pmf_value <= pmf_obs + 1e-15:
            total += pmf_value
    return min(float(total), 1.0)


def build_compact_summary_stats(genome_df, category_column="category", sample_column="sample"):
    if genome_df.empty or category_column not in genome_df.columns:
        return pd.DataFrame()

    try:
        from scipy import stats
    except ImportError:
        return pd.DataFrame()

    working = genome_df.copy()
    working[category_column] = working[category_column].astype(str).str.strip().map(canonical_method_label)
    order = category_order(working, category_column=category_column)
    if len(order) < 2:
        return pd.DataFrame()

    informative_column = (
        "informative_annotation_orfs"
        if "informative_annotation_orfs" in working.columns
        else "annotated_orfs"
    )
    metric_specs = [
        ("genomes_per_sample", "Genomes per sample", "sample"),
        ("total_orfs", "Total ORFs per genome", "genome"),
        (informative_column, "Informative ORFs per genome", "genome"),
        ("total_pathways", "Inferred pathways per genome", "genome"),
        ("mimag_quality_index", "MIMAG quality index", "genome"),
        ("integrity_score", "Integrity", "genome"),
        ("recoverability_score", "Recoverability", "genome"),
        ("qscore", "Qscore", "genome"),
        ("informative_annotation_fraction", "Informative annotation fraction", "genome"),
        ("pathway_input_fraction", "Pathway-input fraction", "genome"),
        ("pathway_support_fraction", "Pathway-support fraction", "genome"),
        ("marker_supported_orfs", "Marker-supported ORFs", "genome"),
        ("reference_mode_supported_accessions", "Reference-supported accessions", "genome"),
    ]
    rows = []
    for metric, metric_label, scope in metric_specs:
        if scope == "sample":
            required = {sample_column, "genome_id"}
            if not required.issubset(set(working.columns)):
                continue
            metric_df = (
                working.groupby([sample_column, category_column])["genome_id"]
                .nunique()
                .reset_index(name="value")
            )
        else:
            if metric not in working.columns:
                continue
            metric_df = working[[category_column, metric]].copy().rename(columns={metric: "value"})
            metric_df["value"] = pd.to_numeric(metric_df["value"], errors="coerce")
            metric_df = metric_df.dropna(subset=["value"])
        if metric_df.empty:
            continue

        present_categories = [
            category
            for category in order
            if metric_df.loc[metric_df[category_column].astype(str).eq(category)].shape[0] > 0
        ]
        if len(present_categories) < 2:
            continue

        for group_a, group_b in itertools.combinations(present_categories, 2):
            values_a = metric_df.loc[metric_df[category_column].astype(str).eq(group_a), "value"].astype(float)
            values_b = metric_df.loc[metric_df[category_column].astype(str).eq(group_b), "value"].astype(float)
            if values_a.empty or values_b.empty:
                continue
            try:
                statistic, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            median_a = float(values_a.median())
            median_b = float(values_b.median())
            rows.append(
                {
                    "analysis_scope": f"{scope}_metric_pairwise",
                    "test": "Mann-Whitney U",
                    "metric": metric,
                    "metric_label": metric_label,
                    "group_column": category_column,
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "group_a_median": median_a,
                    "group_b_median": median_b,
                    "median_difference_a_minus_b": median_a - median_b,
                    "winner_by_median": group_a if median_a > median_b else group_b if median_b > median_a else "tie",
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

        grouped_values = [
            metric_df.loc[metric_df[category_column].astype(str).eq(category), "value"].astype(float).values
            for category in present_categories
        ]
        if len(grouped_values) >= 2:
            try:
                statistic, pvalue = stats.kruskal(*grouped_values)
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            rows.append(
                {
                    "analysis_scope": f"{scope}_metric_global",
                    "test": "Kruskal-Wallis",
                    "metric": metric,
                    "metric_label": metric_label,
                    "group_column": category_column,
                    "n_groups": int(len(present_categories)),
                    "groups": ";".join(map(str, present_categories)),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )

    if not rows:
        return pd.DataFrame()
    stats_df = pd.DataFrame(rows)
    stats_df["qvalue_bh"] = benjamini_hochberg_adjust(stats_df["pvalue"])
    stats_df["significant_p05"] = pd.to_numeric(stats_df["pvalue"], errors="coerce").lt(0.05)
    stats_df["significant_q05"] = pd.to_numeric(stats_df["qvalue_bh"], errors="coerce").lt(0.05)
    stats_df["significance_stars"] = stats_df["qvalue_bh"].map(significance_stars)
    return stats_df


def add_significance_brackets(ax, metric_name, grouped_values, order, stats_df):
    if stats_df is None or stats_df.empty:
        return
    subset = stats_df.loc[
        stats_df["metric"].astype(str).eq(str(metric_name))
        & stats_df["analysis_scope"].astype(str).str.endswith("_pairwise")
        & stats_df["significant_q05"].fillna(False)
    ].copy()
    if subset.empty:
        return

    finite_values = []
    for values in grouped_values:
        if len(values) == 0:
            continue
        numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().tolist()
        finite_values.extend(numeric)
    if not finite_values:
        return

    data_max = float(max(finite_values))
    data_min = float(min(finite_values))
    data_span = max(1e-9, data_max - data_min)
    level_height = data_span * 0.09
    line_height = data_span * 0.03
    top_padding = data_span * 0.14
    if data_max == data_min:
        level_height = max(0.5, abs(data_max) * 0.1 if data_max != 0 else 0.5)
        line_height = level_height * 0.35
        top_padding = level_height * 1.2

    position_map = {category: float(index + 1) for index, category in enumerate(order)}
    occupied = []
    brackets = []
    subset["span"] = subset.apply(
        lambda row: abs(position_map.get(str(row["group_b"]), 0.0) - position_map.get(str(row["group_a"]), 0.0)),
        axis=1,
    )
    subset = subset.sort_values(by=["span", "qvalue_bh"], ascending=[True, True], kind="mergesort")

    max_level = 0
    for row in subset.to_dict("records"):
        group_a = str(row.get("group_a", ""))
        group_b = str(row.get("group_b", ""))
        if group_a not in position_map or group_b not in position_map:
            continue
        x1 = position_map[group_a]
        x2 = position_map[group_b]
        if x1 == x2:
            continue
        if x1 > x2:
            x1, x2 = x2, x1
        level = 0
        while any(not (x2 < start or x1 > end) and used_level == level for start, end, used_level in occupied):
            level += 1
        occupied.append((x1, x2, level))
        max_level = max(max_level, level)
        y = data_max + top_padding + (level * level_height)
        star_label = str(row.get("significance_stars", "")).strip() or "*"
        brackets.append((x1, x2, y, star_label))

    for x1, x2, y, star_label in brackets:
        ax.plot([x1, x1, x2, x2], [y, y + line_height, y + line_height, y], color="black", linewidth=0.9, zorder=4)
        ax.text((x1 + x2) / 2.0, y + line_height, star_label, ha="center", va="bottom", fontsize=9, zorder=5)

    current_bottom, current_top = ax.get_ylim()
    required_top = data_max + top_padding + ((max_level + 1.9) * level_height)
    ax.set_ylim(current_bottom, max(current_top, required_top))


def compact_metric_grouped_values(genome_df, metric, order, category_column="category", sample_column="sample"):
    working = genome_df.copy()
    if metric == "genomes_per_sample":
        if sample_column not in working.columns or "genome_id" not in working.columns:
            return []
        samples = (
            working[sample_column]
            .astype(str)
            .str.strip()
            .loc[lambda series: series.ne("")]
            .unique()
            .tolist()
        )
        return [
            (
                working.loc[working[category_column].astype(str).eq(category)]
                .groupby(sample_column)["genome_id"]
                .nunique()
                .reindex(samples, fill_value=0)
                .values
                .astype(float)
            )
            for category in order
        ]
    if metric not in working.columns:
        return []
    return [
        (
            pd.to_numeric(
                working.loc[working[category_column].astype(str).eq(category), metric],
                errors="coerce",
            )
            .dropna()
            .values
            .astype(float)
        )
        for category in order
    ]


def draw_compact_metric_axis(ax, grouped_values, order, metric, title, stats_df=None, rng=None):
    if not any(len(values) for values in grouped_values):
        ax.axis("off")
        return False
    rng = rng or np.random.default_rng(42)
    box = ax.boxplot(grouped_values, patch_artist=True, labels=order, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#c0c0c0")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)
    for whisker in box["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.9)
    for cap in box["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.9)
    for idx, values in enumerate(grouped_values, start=1):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(
            np.full(len(values), float(idx)) + jitter,
            np.asarray(values, dtype=float),
            s=18,
            color="#c7c7c7",
            edgecolors="none",
            alpha=0.7,
            zorder=1,
        )
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xticklabels(order, rotation=90)
    ax.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.6)
    add_significance_brackets(ax, metric, grouped_values, order, stats_df)
    return True


def metapathways_single_metric_specs(genome_df):
    informative_column = (
        "informative_annotation_orfs"
        if "informative_annotation_orfs" in genome_df.columns
        else "annotated_orfs"
    )
    specs = [
        ("genomes_per_sample", "Genomes per sample"),
        ("total_orfs", "Total ORFs per genome"),
        (informative_column, "Informative ORFs per genome"),
        ("total_pathways", "Inferred pathways per genome"),
        ("mimag_quality_index", "MIMAG quality index"),
        ("integrity_score", "Integrity"),
        ("recoverability_score", "Recoverability"),
        ("qscore", "Qscore"),
        ("informative_annotation_fraction", "Informative annotation fraction"),
        ("pathway_input_fraction", "Pathway-input fraction"),
        ("pathway_support_fraction", "Pathway-support fraction"),
        ("marker_supported_orfs", "Marker-supported ORFs"),
        ("reference_mode_supported_accessions", "Reference-supported accessions"),
    ]
    seen = set()
    available = []
    for metric, title in specs:
        if metric in seen:
            continue
        seen.add(metric)
        if metric == "genomes_per_sample" or metric in genome_df.columns:
            available.append((metric, title))
    return available


def plot_single_compact_metric(genome_df, metric, title, output_base, category_column="category", sample_column="sample", stats_df=None):
    ensure_plotting()
    if genome_df.empty or category_column not in genome_df.columns:
        return False
    order = category_order(genome_df, category_column=category_column)
    if not order:
        return False
    grouped_values = compact_metric_grouped_values(
        genome_df,
        metric,
        order,
        category_column=category_column,
        sample_column=sample_column,
    )
    if not grouped_values or not any(len(values) for values in grouped_values):
        return False
    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(figsize=(max(7.5, len(order) * 1.45), 6.5))
    drew = draw_compact_metric_axis(
        ax,
        grouped_values,
        order,
        metric,
        title,
        stats_df=stats_df,
        rng=np.random.default_rng(42),
    )
    if not drew:
        plt_local.close(fig)
        return False
    if metric == "genomes_per_sample":
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_base)
    return True


def build_sample_category_summary(genome_df, category_column="category", sample_column="sample"):
    required = {category_column, sample_column, "genome_id"}
    if genome_df.empty or not required.issubset(set(genome_df.columns)):
        return pd.DataFrame()

    working = genome_df.copy()
    numeric_columns = [
        "total_orfs",
        "annotated_orfs",
        "annotation_fraction",
        "informative_annotation_orfs",
        "informative_annotation_fraction",
        "total_pathways",
        "pathway_support_fraction",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")

    agg_spec = {"n_genomes": ("genome_id", "nunique")}
    median_metrics = [
        "total_orfs",
        "annotated_orfs",
        "annotation_fraction",
        "informative_annotation_orfs",
        "informative_annotation_fraction",
        "total_pathways",
        "pathway_support_fraction",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
    ]
    for column in median_metrics:
        if column in working.columns:
            agg_spec[f"median_{column}"] = (column, "median")

    summary = (
        working.groupby([sample_column, category_column], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    summary[sample_column] = summary[sample_column].astype(str).str.strip()
    summary[category_column] = summary[category_column].astype(str).str.strip().map(canonical_method_label)
    summary["method_family"] = summary[category_column].map(method_family_label)
    summary["is_variant"] = summary[category_column].map(method_variant_flag)
    return summary


def detect_base_variant_method_pairs(sample_summary_df, category_column="category"):
    if sample_summary_df.empty or category_column not in sample_summary_df.columns:
        return pd.DataFrame()

    categories = ordered_methods(sample_summary_df[category_column].astype(str).unique().tolist())
    rows = []
    for family in ["SAG", "MAG"]:
        family_categories = [
            category
            for category in categories
            if method_family_label(category) == family
        ]
        if not family_categories:
            continue
        base_category = next((category for category in family_categories if not method_variant_flag(category)), "")
        variant_category = next((category for category in family_categories if method_variant_flag(category)), "")
        if not base_category or not variant_category:
            continue
        rows.append(
            {
                "method_family": family,
                "base_category": str(base_category),
                "variant_category": str(variant_category),
                "category_pair": f"{base_category} -> {variant_category}",
            }
        )
    return pd.DataFrame(rows)


def build_variant_improvement_tables(sample_summary_df, category_column="category", sample_column="sample"):
    if sample_summary_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pair_df = detect_base_variant_method_pairs(sample_summary_df, category_column=category_column)
    if pair_df.empty:
        return sample_summary_df.copy(), pd.DataFrame(), pd.DataFrame()

    metric_specs = [
        ("n_genomes", "Genome count", "int"),
        ("median_total_orfs", "Median total ORFs", "int"),
        ("median_annotated_orfs", "Median annotated ORFs", "int"),
        ("median_annotation_fraction", "Median annotation fraction", "float2"),
        ("median_informative_annotation_orfs", "Median informative ORFs", "int"),
        ("median_informative_annotation_fraction", "Median informative fraction", "float2"),
        ("median_total_pathways", "Median inferred pathways", "float1"),
        ("median_pathway_support_fraction", "Median pathway-support fraction", "float2"),
        ("median_marker_supported_orfs", "Median marker-supported ORFs", "int"),
        ("median_reference_mode_supported_accessions", "Median ref-supported accessions", "int"),
    ]
    available_specs = [
        (metric_id, metric_label, fmt)
        for metric_id, metric_label, fmt in metric_specs
        if metric_id in sample_summary_df.columns
    ]
    if not available_specs:
        return sample_summary_df.copy(), pd.DataFrame(), pd.DataFrame()

    rows = []
    for pair_row in pair_df.to_dict("records"):
        family = str(pair_row.get("method_family", "")).strip()
        base_category = str(pair_row.get("base_category", "")).strip()
        variant_category = str(pair_row.get("variant_category", "")).strip()
        if not family or not base_category or not variant_category:
            continue

        base_df = sample_summary_df.loc[
            sample_summary_df[category_column].astype(str).eq(base_category)
        ].copy()
        variant_df = sample_summary_df.loc[
            sample_summary_df[category_column].astype(str).eq(variant_category)
        ].copy()
        if base_df.empty or variant_df.empty:
            continue

        merged = base_df.merge(
            variant_df,
            on=sample_column,
            how="inner",
            suffixes=("_base", "_variant"),
        )
        if merged.empty:
            continue

        for merged_row in merged.to_dict("records"):
            sample_value = str(merged_row.get(sample_column, "")).strip()
            for metric_id, metric_label, fmt in available_specs:
                base_value = pd.to_numeric(pd.Series([merged_row.get(f"{metric_id}_base")]), errors="coerce").iat[0]
                variant_value = pd.to_numeric(pd.Series([merged_row.get(f"{metric_id}_variant")]), errors="coerce").iat[0]
                if pd.isna(base_value) or pd.isna(variant_value):
                    continue
                delta = float(variant_value) - float(base_value)
                pct_delta = np.nan
                if float(base_value) != 0.0:
                    pct_delta = (delta / float(base_value)) * 100.0
                rows.append(
                    {
                        "sample": sample_value,
                        "method_family": family,
                        "base_category": base_category,
                        "variant_category": variant_category,
                        "category_pair": f"{base_category} -> {variant_category}",
                        "metric_id": metric_id,
                        "metric_label": metric_label,
                        "metric_format": fmt,
                        "base_value": float(base_value),
                        "variant_value": float(variant_value),
                        "delta": delta,
                        "percent_delta": pct_delta,
                    }
                )

    by_sample_df = pd.DataFrame(rows)
    if by_sample_df.empty:
        return sample_summary_df.copy(), by_sample_df, pd.DataFrame()

    summary_df = (
        by_sample_df.groupby(
            ["method_family", "base_category", "variant_category", "category_pair", "metric_id", "metric_label", "metric_format"],
            dropna=False,
        )
        .agg(
            n_samples=("sample", "nunique"),
            median_base=("base_value", "median"),
            median_variant=("variant_value", "median"),
            median_delta=("delta", "median"),
            mean_delta=("delta", "mean"),
            positive_sample_fraction=("delta", lambda values: float((pd.Series(values) > 0).mean()) if len(values) else np.nan),
            nonnegative_sample_fraction=("delta", lambda values: float((pd.Series(values) >= 0).mean()) if len(values) else np.nan),
        )
        .reset_index()
    )
    return sample_summary_df.copy(), by_sample_df, summary_df


def _format_metric_value(value, fmt):
    if pd.isna(value):
        return "NA"
    numeric = float(value)
    if fmt == "float2":
        return f"{numeric:.2f}"
    if fmt == "float1":
        return f"{numeric:.1f}"
    return f"{int(round(numeric))}"


def _ordered_variant_improvement_entities(by_sample_df):
    sample_order = sorted(by_sample_df["sample"].astype(str).unique().tolist())
    family_order = ["SAG", "MAG"]
    rows = []
    for family in family_order:
        family_samples = [
            sample
            for sample in sample_order
            if by_sample_df.loc[
                by_sample_df["method_family"].astype(str).eq(family)
                & by_sample_df["sample"].astype(str).eq(sample)
            ].shape[0]
            > 0
        ]
        for sample in family_samples:
            rows.append((family, sample))
    return rows


def plot_variant_improvement_deltas(by_sample_df, output_base):
    ensure_plotting()
    if by_sample_df.empty:
        return False

    metric_order = by_sample_df[["metric_id", "metric_label", "metric_format"]].drop_duplicates().to_dict("records")
    if not metric_order:
        return False
    entity_order = _ordered_variant_improvement_entities(by_sample_df)
    if not entity_order:
        return False
    entity_labels = [f"{family} | {sample}" for family, sample in entity_order]
    entity_positions = {label: index for index, label in enumerate(entity_labels)}
    family_colors = {"SAG": "#4d4d4d", "MAG": "#8c8c8c"}

    n_cols = min(3, len(metric_order))
    n_rows = int(np.ceil(len(metric_order) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, n_cols * 5.0), max(6.5, n_rows * 3.8)),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, metric_row in zip(axes, metric_order):
        metric_id = str(metric_row["metric_id"])
        metric_label = str(metric_row["metric_label"])
        fmt = str(metric_row["metric_format"])
        subset = by_sample_df.loc[by_sample_df["metric_id"].astype(str).eq(metric_id)].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["entity_label"] = subset.apply(
            lambda row: f"{row['method_family']} | {row['sample']}",
            axis=1,
        )
        subset = subset.loc[subset["entity_label"].isin(entity_labels)].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["y_pos"] = subset["entity_label"].map(entity_positions)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.9)
        for family, family_df in subset.groupby("method_family", dropna=False):
            ax.scatter(
                family_df["delta"].astype(float).values,
                family_df["y_pos"].astype(float).values,
                s=40,
                color=family_colors.get(str(family), "#666666"),
                edgecolors="black",
                linewidths=0.35,
                alpha=0.9,
                zorder=3,
            )
        ax.set_yticks(np.arange(len(entity_labels)))
        ax.set_yticklabels(entity_labels, fontsize=8)
        ax.set_title(metric_label)
        ax.set_xlabel("Variant - base")
        ax.grid(axis="x", color="#dddddd", linestyle="-", linewidth=0.6)
        finite = subset["delta"].replace([np.inf, -np.inf], np.nan).dropna()
        if finite.empty:
            summary_label = "n=0"
        else:
            summary_label = (
                f"n={int(finite.shape[0])}; median={_format_metric_value(finite.median(), fmt)}; "
                f"positive={float((finite > 0).mean()):.2f}"
            )
        ax.text(
            0.02,
            0.02,
            summary_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85, "pad": 2.0},
        )

    for index in range(len(metric_order), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Base-to-variant MetaPathways improvement deltas by sample", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True


def plot_variant_improvement_before_after(by_sample_df, output_base):
    ensure_plotting()
    if by_sample_df.empty:
        return False

    metric_order = by_sample_df[["metric_id", "metric_label", "metric_format"]].drop_duplicates().to_dict("records")
    if not metric_order:
        return False
    entity_order = _ordered_variant_improvement_entities(by_sample_df)
    if not entity_order:
        return False
    entity_labels = [f"{family} | {sample}" for family, sample in entity_order]
    entity_positions = {label: index for index, label in enumerate(entity_labels)}
    base_color = "#bdbdbd"
    variant_color = "#525252"

    n_cols = min(3, len(metric_order))
    n_rows = int(np.ceil(len(metric_order) / float(n_cols)))
    plt_local = ensure_plotting()
    fig, axes = plt_local.subplots(
        n_rows,
        n_cols,
        figsize=(max(14, n_cols * 5.0), max(6.5, n_rows * 3.8)),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, metric_row in zip(axes, metric_order):
        metric_id = str(metric_row["metric_id"])
        metric_label = str(metric_row["metric_label"])
        fmt = str(metric_row["metric_format"])
        subset = by_sample_df.loc[by_sample_df["metric_id"].astype(str).eq(metric_id)].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["entity_label"] = subset.apply(
            lambda row: f"{row['method_family']} | {row['sample']}",
            axis=1,
        )
        subset = subset.loc[subset["entity_label"].isin(entity_labels)].copy()
        if subset.empty:
            ax.axis("off")
            continue
        subset["y_pos"] = subset["entity_label"].map(entity_positions)
        for row in subset.to_dict("records"):
            y_pos = float(row["y_pos"])
            base_value = float(row["base_value"])
            variant_value = float(row["variant_value"])
            ax.plot([base_value, variant_value], [y_pos, y_pos], color="#b3b3b3", linewidth=1.4, zorder=1)
            ax.scatter(base_value, y_pos, s=34, color=base_color, edgecolors="black", linewidths=0.3, zorder=2)
            ax.scatter(variant_value, y_pos, s=34, color=variant_color, edgecolors="black", linewidths=0.3, zorder=3)
        ax.set_yticks(np.arange(len(entity_labels)))
        ax.set_yticklabels(entity_labels, fontsize=8)
        ax.set_title(metric_label)
        ax.set_xlabel("Per-sample median" if metric_id != "n_genomes" else "Per-sample count")
        ax.grid(axis="x", color="#dddddd", linestyle="-", linewidth=0.6)
        median_base = subset["base_value"].median()
        median_variant = subset["variant_value"].median()
        ax.text(
            0.02,
            0.02,
            f"base={_format_metric_value(median_base, fmt)}; variant={_format_metric_value(median_variant, fmt)}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.85, "pad": 2.0},
        )

    for index in range(len(metric_order), len(axes)):
        axes[index].axis("off")

    fig.suptitle("Base-versus-variant MetaPathways sample summaries", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True


def summarize_selected_category_counts(selected_df, category_column="category"):
    if selected_df is None or selected_df.empty or category_column not in selected_df.columns:
        return pd.DataFrame()

    working = selected_df.copy()
    working[category_column] = working[category_column].astype(str).str.strip().map(canonical_method_label)
    if "mimag_tier" in working.columns:
        working["_tier_norm"] = working["mimag_tier"].astype(str).str.lower().str.strip()
    else:
        working["_tier_norm"] = ""
    count_column = "Genome_Id" if "Genome_Id" in working.columns else "genome_id" if "genome_id" in working.columns else category_column

    summary = (
        working.groupby(category_column, dropna=False)
        .agg(
            n_selected_genomes=(count_column, "nunique"),
            n_selected_hq=("_tier_norm", lambda values: int((pd.Series(values) == "high").sum())),
            n_selected_mq=("_tier_norm", lambda values: int((pd.Series(values) == "medium").sum())),
            n_selected_lq=("_tier_norm", lambda values: int((pd.Series(values) == "low").sum())),
        )
        .reset_index()
    )
    order = ordered_methods(summary[category_column].astype(str).tolist(), counts=dict(zip(summary[category_column], summary["n_selected_genomes"])))
    summary[category_column] = summary[category_column].astype(str)
    summary = summary.set_index(category_column).reindex(order).reset_index()
    return summary


def valid_path_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def best_set_fasta_source(row):
    for column in ["copied_fasta_path", "fasta_path", "mp_fasta_path", "ani_fasta_path"]:
        if column not in row.index:
            continue
        text = valid_path_text(row.get(column, ""))
        if not text:
            continue
        path = Path(text).expanduser()
        if path.exists():
            return column, path
    return "", None


def phylogeny_duplicate_source_key(row):
    for column in ["fasta_export_source_path", "fasta_path", "mp_fasta_path", "ani_fasta_path", "copied_fasta_path"]:
        if column not in row.index:
            continue
        text = valid_path_text(row.get(column, ""))
        if not text:
            continue
        path = Path(text).expanduser()
        return str(path.resolve()) if path.exists() else str(path)
    return ""


def deduplicate_phylogeny_rows(selected_df):
    if selected_df is None or selected_df.empty:
        return selected_df.copy() if isinstance(selected_df, pd.DataFrame) else selected_df

    working = selected_df.copy().reset_index(drop=True)
    dedupe_keys = []
    for _index, row in working.iterrows():
        source_key = phylogeny_duplicate_source_key(row)
        genome_id = str(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", "")))).strip()
        sample = str(row.get("sample", row.get("mp_sample", ""))).strip()
        category = canonical_method_label(str(row.get("category", row.get("mp_category", ""))).strip())
        dedupe_keys.append((sample, category, genome_id, source_key))
    working["_phylogeny_exact_duplicate_key"] = dedupe_keys
    working = (
        working.drop_duplicates(subset=["_phylogeny_exact_duplicate_key"], keep="first")
        .drop(columns=["_phylogeny_exact_duplicate_key"])
        .reset_index(drop=True)
    )
    return working


def export_selected_set_fastas(selected_df, selected_set_dir):
    selected_set_path = Path(selected_set_dir)
    fasta_dir = selected_set_path / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    selected_set_path.mkdir(parents=True, exist_ok=True)
    for existing_fasta in list(fasta_dir.iterdir()):
        if existing_fasta.is_file() and existing_fasta.name.lower().endswith(
            (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
        ):
            existing_fasta.unlink()

    working = deduplicate_phylogeny_rows(selected_df)

    copied_paths = []
    fasta_statuses = []
    fasta_source_columns = []
    fasta_source_paths = []
    audit_rows = []
    for index, row in working.iterrows():
        source_column, source = best_set_fasta_source(row)
        fallback_stem = source.stem if source is not None else f"row_{index}"
        genome_id = str(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", fallback_stem)))).strip()
        sample = str(row.get("sample", row.get("mp_sample", ""))).strip()
        category = canonical_method_label(str(row.get("category", row.get("mp_category", ""))).strip())

        if source is None:
            copied_paths.append("")
            fasta_statuses.append("missing")
            fasta_source_columns.append("")
            fasta_source_paths.append("")
            audit_rows.append(
                {
                    "row_index": int(index),
                    "sample": sample,
                    "category": category,
                    "Genome_Id": genome_id,
                    "status": "missing",
                    "source_column": "",
                    "source_path": "",
                    "copied_fasta_path": "",
                }
            )
            continue

        stem_parts = [part for part in [sample, category, genome_id] if part and part.lower() != "nan"]
        target_stem = sanitize_label("__".join(stem_parts) if stem_parts else source.stem)
        target_name = f"{target_stem}.fasta"
        destination = fasta_dir / target_name
        if destination.exists():
            source_hash = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:10]
            destination = fasta_dir / f"{target_stem}__source_{source_hash}.fasta"
            counter = 2
            while destination.exists():
                destination = fasta_dir / f"{target_stem}__source_{source_hash}_{counter}.fasta"
                counter += 1
        shutil.copy2(source, destination)
        copied_paths.append(str(destination))
        fasta_statuses.append("copied")
        fasta_source_columns.append(source_column)
        fasta_source_paths.append(str(source))
        audit_rows.append(
            {
                "row_index": int(index),
                "sample": sample,
                "category": category,
                "Genome_Id": genome_id,
                "status": "copied",
                "source_column": source_column,
                "source_path": str(source),
                "copied_fasta_path": str(destination),
            }
        )

    working["copied_fasta_path"] = copied_paths
    working["fasta_export_status"] = fasta_statuses
    working["fasta_export_source_column"] = fasta_source_columns
    working["fasta_export_source_path"] = fasta_source_paths
    master_path = selected_set_path / "master.tsv"
    audit_path = selected_set_path / "fasta_export_audit.tsv"
    working.to_csv(master_path, sep="\t", index=False)
    pd.DataFrame(audit_rows).to_csv(audit_path, sep="\t", index=False)
    copied_file_paths = [Path(path) for path in copied_paths if path]
    return working, [master_path, audit_path, fasta_dir] + copied_file_paths


def summarize_selection_preferences(review_df, category_column="category"):
    columns_by_category = [
        category_column,
        "n_components_present_total",
        "n_components_competing",
        "n_components_singleton",
        "n_components_won_when_competing",
        "win_fraction_when_competing",
    ]
    pairwise_columns = [
        "category_a",
        "category_b",
        "n_components_both_present",
        "n_decisive_components",
        "n_components_won_a",
        "n_components_won_b",
        "n_components_won_other",
        "win_fraction_a_over_b",
        "preferred_category",
        "pvalue_binom",
        "qvalue_bh",
        "significance",
    ]
    if review_df is None or review_df.empty or category_column not in review_df.columns or "component_id" not in review_df.columns:
        return pd.DataFrame(columns=columns_by_category), pd.DataFrame(columns=pairwise_columns)

    working = review_df.copy()
    working[category_column] = working[category_column].astype(str).str.strip().map(canonical_method_label)
    working["component_id"] = working["component_id"].astype(str).str.strip()
    working = working.loc[working[category_column].ne("") & working["component_id"].ne("")].copy()
    if working.empty:
        return pd.DataFrame(columns=columns_by_category), pd.DataFrame(columns=pairwise_columns)

    present_df = working.loc[:, ["component_id", category_column]].drop_duplicates()
    categories = ordered_methods(present_df[category_column].tolist())
    component_categories = (
        present_df.groupby("component_id")[category_column]
        .agg(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    selected_df = working.loc[working["selected"].fillna(False), ["component_id", category_column]].drop_duplicates(subset=["component_id"])
    winner_map = selected_df.set_index("component_id")[category_column].to_dict()

    overall_rows = []
    for category in categories:
        present_components = [component_id for component_id, values in component_categories.items() if category in values]
        competing_components = [component_id for component_id in present_components if len(component_categories.get(component_id, [])) > 1]
        singleton_components = [component_id for component_id in present_components if len(component_categories.get(component_id, [])) == 1]
        wins_competing = sum(1 for component_id in competing_components if winner_map.get(component_id) == category)
        overall_rows.append(
            {
                category_column: category,
                "n_components_present_total": int(len(present_components)),
                "n_components_competing": int(len(competing_components)),
                "n_components_singleton": int(len(singleton_components)),
                "n_components_won_when_competing": int(wins_competing),
                "win_fraction_when_competing": (
                    float(wins_competing) / float(len(competing_components))
                    if competing_components else float("nan")
                ),
            }
        )
    overall_df = pd.DataFrame(overall_rows)

    pairwise_rows = []
    pvalues = []
    for idx_a, category_a in enumerate(categories):
        for category_b in categories[idx_a + 1:]:
            both_components = [
                component_id
                for component_id, values in component_categories.items()
                if category_a in values and category_b in values
            ]
            wins_a = sum(1 for component_id in both_components if winner_map.get(component_id) == category_a)
            wins_b = sum(1 for component_id in both_components if winner_map.get(component_id) == category_b)
            wins_other = int(len(both_components) - wins_a - wins_b)
            decisive = int(wins_a + wins_b)
            pvalue = exact_binomial_test_two_sided(wins_a, decisive) if decisive > 0 else float("nan")
            pvalues.append(pvalue)
            preferred = ""
            if wins_a > wins_b:
                preferred = category_a
            elif wins_b > wins_a:
                preferred = category_b
            pairwise_rows.append(
                {
                    "category_a": category_a,
                    "category_b": category_b,
                    "n_components_both_present": int(len(both_components)),
                    "n_decisive_components": decisive,
                    "n_components_won_a": int(wins_a),
                    "n_components_won_b": int(wins_b),
                    "n_components_won_other": wins_other,
                    "win_fraction_a_over_b": (float(wins_a) / float(decisive)) if decisive > 0 else float("nan"),
                    "preferred_category": preferred,
                    "pvalue_binom": pvalue,
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["qvalue_bh"] = benjamini_hochberg_adjust(pairwise_df["pvalue_binom"]).to_numpy(dtype=float)
        pairwise_df["significance"] = pairwise_df["qvalue_bh"].map(significance_stars)
        pairwise_df = pairwise_df.sort_values(by=["category_a", "category_b"], kind="mergesort").reset_index(drop=True)
    else:
        pairwise_df = pd.DataFrame(columns=pairwise_columns)

    if not overall_df.empty:
        counts = dict(zip(overall_df[category_column].astype(str), overall_df["n_components_won_when_competing"]))
        order = ordered_methods(overall_df[category_column].astype(str).tolist(), counts=counts)
        overall_df[category_column] = overall_df[category_column].astype(str)
        overall_df = overall_df.set_index(category_column).reindex(order).reset_index()
    return overall_df, pairwise_df


def plot_selection_preference_summary(overall_df, pairwise_df, output_base, category_column="category"):
    ensure_plotting()
    if overall_df is None or overall_df.empty or category_column not in overall_df.columns:
        return False

    plt_local = ensure_plotting()
    order = overall_df[category_column].astype(str).tolist()
    fig, axes = plt_local.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.0, 1.15]})

    bar_df = overall_df.copy()
    x = np.arange(len(order))
    values = pd.to_numeric(bar_df["win_fraction_when_competing"], errors="coerce")
    axes[0].bar(x, values.fillna(0.0).to_numpy(dtype=float), color="#7f7f7f", edgecolor="black", linewidth=0.7)
    axes[0].set_ylim(0, 1.0)
    axes[0].axhline(0.5, color="#4d4d4d", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=90)
    axes[0].set_title("Win rate when categories compete")
    axes[0].set_xlabel(category_column)
    axes[0].set_ylabel("Selected fraction")
    axes[0].grid(axis="y", color="#dddddd", linestyle="-", linewidth=0.6)
    for index, category in enumerate(order):
        row = bar_df.loc[bar_df[category_column].astype(str).eq(category)].head(1)
        if row.empty:
            continue
        wins = int(row["n_components_won_when_competing"].iloc[0])
        competing = int(row["n_components_competing"].iloc[0])
        fraction = pd.to_numeric(row["win_fraction_when_competing"], errors="coerce").iloc[0]
        if pd.isna(fraction):
            continue
        axes[0].text(index, float(fraction), f"{wins}/{competing}", ha="center", va="bottom", fontsize=8)

    heatmap_matrix = pd.DataFrame(np.nan, index=order, columns=order)
    annotation_matrix = pd.DataFrame("", index=order, columns=order)
    if pairwise_df is not None and not pairwise_df.empty:
        for row in pairwise_df.to_dict("records"):
            category_a = str(row.get("category_a", "")).strip()
            category_b = str(row.get("category_b", "")).strip()
            if category_a not in heatmap_matrix.index or category_b not in heatmap_matrix.columns:
                continue
            wins_a = int(row.get("n_components_won_a", 0))
            wins_b = int(row.get("n_components_won_b", 0))
            decisive = int(row.get("n_decisive_components", 0))
            frac_a = row.get("win_fraction_a_over_b", float("nan"))
            frac_b = (float(wins_b) / float(decisive)) if decisive > 0 else float("nan")
            sig = str(row.get("significance", "") or "")
            heatmap_matrix.loc[category_a, category_b] = frac_a
            heatmap_matrix.loc[category_b, category_a] = frac_b
            annotation_matrix.loc[category_a, category_b] = f"{wins_a}-{wins_b}\n{sig}" if decisive > 0 else "0-0"
            annotation_matrix.loc[category_b, category_a] = f"{wins_b}-{wins_a}\n{sig}" if decisive > 0 else "0-0"
    image = axes[1].imshow(heatmap_matrix.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks(np.arange(len(order)))
    axes[1].set_xticklabels(order, rotation=90)
    axes[1].set_yticks(np.arange(len(order)))
    axes[1].set_yticklabels(order)
    axes[1].set_title("Pairwise preference when both categories are present")
    axes[1].set_xlabel("Compared against")
    axes[1].set_ylabel("Preferred row category")
    for row_index, row_category in enumerate(order):
        for col_index, col_category in enumerate(order):
            text = annotation_matrix.loc[row_category, col_category]
            value = heatmap_matrix.loc[row_category, col_category]
            if pd.isna(value):
                continue
            color = "white" if float(value) >= 0.55 else "black"
            axes[1].text(col_index, row_index, text, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_label("Row category win fraction")

    fig.suptitle("Category selection preference across competing components", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_base)
    return True


def plot_selected_category_counts(summary_df, output_base, category_column="category"):
    ensure_plotting()
    if summary_df is None or summary_df.empty or category_column not in summary_df.columns:
        return False

    plt_local = ensure_plotting()
    plot_df = summary_df.copy()
    order = plot_df[category_column].astype(str).tolist()
    metrics = [
        ("n_selected_genomes", "Total selected genomes"),
        ("n_selected_hq", "Selected HQ genomes"),
        ("n_selected_mq", "Selected MQ genomes"),
        ("n_selected_lq", "Selected LQ genomes"),
    ]
    fig, axes = plt_local.subplots(2, 2, figsize=(15, 11), sharex=False, squeeze=False)
    axes = axes.ravel()
    x = np.arange(len(order))
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in plot_df.columns:
            ax.axis("off")
            continue
        values = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, values, color="#7f7f7f", edgecolor="black", linewidth=0.6)
        ax.set_title(title)
        ax.set_xlabel(category_column)
        ax.set_ylabel("Genome count")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90)
        for index, value in enumerate(values.tolist()):
            ax.text(index, float(value), f"{int(round(value))}", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", color="#dddddd", linestyle="-", linewidth=0.6)
    fig.suptitle("Final selected genomes by category", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, output_base)
    return True


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
    stats_df=None,
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
    rng = np.random.default_rng(42)

    for ax, (metric, title) in zip(axes, metrics):
        if metric == "genomes_per_sample" and not samples:
            ax.axis("off")
            continue
        grouped_values = compact_metric_grouped_values(
            working,
            metric,
            order,
            category_column=category_column,
            sample_column=sample_column,
        )
        if not draw_compact_metric_axis(ax, grouped_values, order, metric, title, stats_df=stats_df, rng=rng):
            continue
        if metric == "genomes_per_sample":
            ax.set_ylim(bottom=0)

    n_samples = int(working[sample_column].astype(str).nunique()) if sample_column in working.columns else int(working.shape[0])
    dedup_note = "all genomes"
    fig.suptitle(
        f"MetaPathways compact summary ({dedup_note}; samples={n_samples})",
        fontsize=16,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, output_base)
    return True


def plot_combined_category_distributions(genome_df, output_base, category_column="category", stats_df=None):
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
        draw_compact_metric_axis(
            ax,
            grouped,
            order,
            metric,
            title,
            stats_df=stats_df,
            rng=np.random.default_rng(42),
        )
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
    fig, ax = plt_local.subplots(figsize=(max(9, matrix.shape[1] * 1.0), max(5.5, matrix.shape[0] * 0.65)))
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
        figsize=(max(11, len(panels) * 9.5), max(6.0, max(panel[1].shape[0] for panel in panels) * 0.62)),
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
        figsize=(max(11, len(top_pathways) * 0.36), max(5.5, fraction_matrix.shape[0] * 0.7))
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
        "informative_annotation_orfs",
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
        "informative_annotation_fraction",
        "informative_annotation_orfs",
        "marker_supported_orfs",
        "reference_mode_supported_accessions",
        "total_pathways",
        "qscore",
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


def species_lineage_label(row):
    lineage_values = []
    for rank in SPECIES_LINEAGE_RANKS:
        value = str(row.get(rank, "")).strip()
        if taxonomy_value_is_informative(value):
            lineage_values.append(value)
    if lineage_values:
        return "; ".join(lineage_values)
    for column in ["taxonomy_display_label", "taxonomy_display_value", "Species", "genome_display_label", "genome_id"]:
        value = str(row.get(column, "")).strip()
        if taxonomy_value_is_informative(value):
            return value
    return ""


def species_lineage_depth(row):
    depth = 0
    for rank in SPECIES_LINEAGE_RANKS:
        value = str(row.get(rank, "")).strip()
        if taxonomy_value_is_informative(value):
            depth += 1
    return depth


def prepare_species_comparison_frame(genome_df, category_column="category", sample_column="sample"):
    required = {category_column, sample_column, "genome_id"}
    if genome_df.empty or not required.issubset(set(genome_df.columns)):
        return pd.DataFrame()

    working = genome_df.copy()
    working[category_column] = working[category_column].astype(str).str.strip().map(canonical_method_label)
    working[sample_column] = working[sample_column].astype(str).str.strip()
    working["genome_id"] = working["genome_id"].astype(str).str.strip()
    working = working.loc[
        working[category_column].ne("")
        & working[sample_column].ne("")
        & working["genome_id"].ne("")
    ].copy()
    if working.empty:
        return pd.DataFrame()

    working["species_key"] = working.apply(dedup_species_key, axis=1)
    working["species_lineage_label"] = working.apply(species_lineage_label, axis=1)
    working["species_lineage_depth"] = working.apply(species_lineage_depth, axis=1)
    working = working.loc[working["species_key"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    sort_columns, ascending = representative_sort_spec(working)
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    return working.reset_index(drop=True)


def select_species_category_best_genomes(genome_df, category_column="category", sample_column="sample"):
    working = prepare_species_comparison_frame(
        genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )
    if working.empty:
        return pd.DataFrame()
    best = (
        working.groupby(["species_key", category_column], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def select_species_sample_category_best_genomes(genome_df, category_column="category", sample_column="sample"):
    working = prepare_species_comparison_frame(
        genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )
    if working.empty:
        return pd.DataFrame()
    best = (
        working.groupby(["species_key", sample_column, category_column], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )
    return best


def _species_label_lookup(frame):
    if frame.empty or "species_key" not in frame.columns:
        return pd.DataFrame(columns=["species_key", "species_lineage_label"])
    working = frame.copy()
    if "species_lineage_depth" not in working.columns:
        working["species_lineage_depth"] = working.apply(species_lineage_depth, axis=1)
    if "species_lineage_label" not in working.columns:
        working["species_lineage_label"] = working.apply(species_lineage_label, axis=1)
    working["__label_length"] = working["species_lineage_label"].astype(str).map(len)
    ranked = working.sort_values(
        by=["species_lineage_depth", "__label_length", "species_lineage_label"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    labels = (
        ranked.groupby("species_key", as_index=False, sort=False)
        .head(1)[["species_key", "species_lineage_label"]]
        .reset_index(drop=True)
    )
    return labels


def build_species_category_best_marker_table(genome_df, category_column="category", sample_column="sample"):
    best_df = select_species_category_best_genomes(
        genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )
    if best_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    label_df = _species_label_lookup(best_df)
    rows = []
    for row in best_df.to_dict("records"):
        species_key = str(row.get("species_key", "")).strip()
        species_label = str(row.get("species_lineage_label", "")).strip()
        category_value = str(row.get(category_column, "")).strip()
        for support_class in ["specific", "generic"]:
            observed_suffix = f"{support_class}_gene_count"
            possible_suffix = f"possible_{support_class}_gene_count"
            for mode_id in ELEMENTAL_MODE_ORDER:
                observed_column = f"marker_{mode_id}_{observed_suffix}"
                possible_column = f"marker_{mode_id}_{possible_suffix}"
                if observed_column not in best_df.columns:
                    continue
                observed = pd.to_numeric(pd.Series([row.get(observed_column)]), errors="coerce").iat[0]
                possible = pd.to_numeric(pd.Series([row.get(possible_column)]), errors="coerce").iat[0] if possible_column in best_df.columns else np.nan
                fraction = np.nan
                if pd.notna(observed) and pd.notna(possible) and float(possible) > 0.0:
                    fraction = float(observed) / float(possible)
                rows.append(
                    {
                        "species_key": species_key,
                        "species_lineage_label": species_label,
                        category_column: category_value,
                        "genome_id": str(row.get("genome_id", "")).strip(),
                        "mode_id": mode_id,
                        "mode_label": ELEMENTAL_MODE_LABELS.get(mode_id, mode_id),
                        "support_class": support_class,
                        "observed_count": float(observed) if pd.notna(observed) else np.nan,
                        "possible_count": float(possible) if pd.notna(possible) else np.nan,
                        "support_fraction": fraction,
                        "cell_text": (
                            f"{int(round(float(observed)))}/{int(round(float(possible)))}"
                            if pd.notna(observed) and pd.notna(possible) and float(possible) > 0.0
                            else ""
                        ),
                    }
                )
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return best_df, long_df
    long_df = long_df.merge(label_df, on="species_key", how="left", suffixes=("", "_best"))
    long_df["species_lineage_label"] = (
        long_df["species_lineage_label_best"].fillna(long_df["species_lineage_label"]).astype(str)
    )
    long_df = long_df.drop(columns=["species_lineage_label_best"], errors="ignore")
    return best_df, long_df


def build_species_category_breadth_marker_table(genome_df, category_column="category", sample_column="sample"):
    sample_best_df = select_species_sample_category_best_genomes(
        genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )
    if sample_best_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    label_df = _species_label_lookup(sample_best_df)
    group_columns = ["species_key", category_column]
    base_df = (
        sample_best_df.groupby(group_columns, dropna=False)
        .agg(samples_with_species=(sample_column, "nunique"))
        .reset_index()
    )
    base_df = base_df.merge(label_df, on="species_key", how="left")

    rows = []
    for base_row in base_df.to_dict("records"):
        species_key = str(base_row.get("species_key", "")).strip()
        species_label = str(base_row.get("species_lineage_label", "")).strip()
        category_value = str(base_row.get(category_column, "")).strip()
        species_samples = int(base_row.get("samples_with_species", 0) or 0)
        subset = sample_best_df.loc[
            sample_best_df["species_key"].astype(str).eq(species_key)
            & sample_best_df[category_column].astype(str).eq(category_value)
        ].copy()
        for support_class in ["specific", "generic"]:
            observed_suffix = f"{support_class}_gene_count"
            for mode_id in ELEMENTAL_MODE_ORDER:
                observed_column = f"marker_{mode_id}_{observed_suffix}"
                if observed_column not in subset.columns:
                    continue
                values = pd.to_numeric(subset[observed_column], errors="coerce").fillna(0.0)
                support_samples = int(values.gt(0).sum())
                fraction = np.nan
                if species_samples > 0:
                    fraction = float(support_samples) / float(species_samples)
                rows.append(
                    {
                        "species_key": species_key,
                        "species_lineage_label": species_label,
                        category_column: category_value,
                        "mode_id": mode_id,
                        "mode_label": ELEMENTAL_MODE_LABELS.get(mode_id, mode_id),
                        "support_class": support_class,
                        "samples_with_species": species_samples,
                        "samples_with_support": support_samples,
                        "support_fraction": fraction,
                        "cell_text": f"{support_samples}/{species_samples}" if species_samples > 0 else "",
                    }
                )
    long_df = pd.DataFrame(rows)
    return sample_best_df, long_df


def plot_species_category_marker_heatmap(
    summary_df,
    output_base,
    plot_title,
    colorbar_label,
    footer_note,
    category_column="category",
):
    ensure_plotting()
    required = {"species_key", "species_lineage_label", "mode_id", "support_fraction", "cell_text", category_column}
    if summary_df.empty or not required.issubset(set(summary_df.columns)):
        return False

    species_order = (
        summary_df[["species_key", "species_lineage_label"]]
        .drop_duplicates()
        .sort_values(by=["species_lineage_label", "species_key"], kind="mergesort")
    )
    ordered_species = species_order.to_dict("records")
    ordered_categories = ordered_methods(summary_df[category_column].astype(str).tolist())
    mode_ids = [
        mode_id
        for mode_id in ELEMENTAL_MODE_ORDER
        if summary_df["mode_id"].astype(str).eq(mode_id).any()
    ]
    if not ordered_species or not ordered_categories or not mode_ids:
        return False

    column_keys = [(mode_id, category) for mode_id in mode_ids for category in ordered_categories]
    value_matrix = np.full((len(ordered_species), len(column_keys)), np.nan, dtype=float)
    text_matrix = np.full((len(ordered_species), len(column_keys)), "", dtype=object)
    lookup = {}
    for row in summary_df.to_dict("records"):
        species_key = str(row.get("species_key", "")).strip()
        species_label = str(row.get("species_lineage_label", "")).strip()
        mode_id = str(row.get("mode_id", "")).strip()
        category_value = str(row.get(category_column, "")).strip()
        if not species_key or not species_label or not mode_id or not category_value:
            continue
        lookup[(species_key, mode_id, category_value)] = row

    for row_index, species_row in enumerate(ordered_species):
        species_key = str(species_row.get("species_key", "")).strip()
        for col_index, (mode_id, category_value) in enumerate(column_keys):
            row = lookup.get((species_key, mode_id, category_value))
            if row is None:
                continue
            value = pd.to_numeric(pd.Series([row.get("support_fraction")]), errors="coerce").iat[0]
            if pd.notna(value):
                value_matrix[row_index, col_index] = float(value)
            text_matrix[row_index, col_index] = str(row.get("cell_text", "")).strip()

    plt_local = ensure_plotting()
    fig, ax = plt_local.subplots(
        figsize=(max(16, len(column_keys) * 0.72), max(7.0, len(ordered_species) * 0.34))
    )
    if hasattr(plt_local, "colormaps"):
        cmap = plt_local.colormaps["Greys"].copy()
    else:
        cmap = plt_local.cm.get_cmap("Greys").copy()
    cmap.set_bad(color="#ffffff")
    image = ax.imshow(value_matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(column_keys)))
    ax.set_xticklabels(
        [f"{ELEMENTAL_MODE_LABELS.get(mode_id, mode_id)}\n{category}" for mode_id, category in column_keys],
        rotation=90,
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(ordered_species)))
    ax.set_yticklabels([str(row.get("species_lineage_label", "")).strip() for row in ordered_species], fontsize=7)
    ax.set_xlabel("Mode and category")
    ax.set_ylabel("Species lineage")
    ax.set_title(plot_title)

    for separator_index in range(1, len(mode_ids)):
        ax.axvline((separator_index * len(ordered_categories)) - 0.5, color="#9a9a9a", linewidth=0.8)

    for row_index in range(value_matrix.shape[0]):
        for col_index in range(value_matrix.shape[1]):
            if np.isnan(value_matrix[row_index, col_index]):
                continue
            value = float(value_matrix[row_index, col_index])
            text = str(text_matrix[row_index, col_index]).strip()
            ax.text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color=_heatmap_text_color(value, 1.0),
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.text(0.5, 0.012, footer_note, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_figure(fig, output_base)
    return True


def write_species_category_marker_comparisons(
    output_dir,
    prefix,
    combined_genome_df,
    category_column="category",
    sample_column="sample",
):
    output_dir = Path(output_dir)
    root_dir = output_dir / "species_category_comparison"
    best_dir = root_dir / "best_vs_best"
    breadth_dir = root_dir / "cross_sample_breadth"
    best_dir.mkdir(parents=True, exist_ok=True)
    breadth_dir.mkdir(parents=True, exist_ok=True)

    wrote_paths = []
    best_genomes_df, best_long_df = build_species_category_best_marker_table(
        combined_genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )
    breadth_genomes_df, breadth_long_df = build_species_category_breadth_marker_table(
        combined_genome_df,
        category_column=category_column,
        sample_column=sample_column,
    )

    outputs = [
        (best_dir / f"{sanitize_label(prefix)}_species_category_best_genomes.tsv", best_genomes_df),
        (best_dir / f"{sanitize_label(prefix)}_species_category_best_marker_long.tsv", best_long_df),
        (breadth_dir / f"{sanitize_label(prefix)}_species_category_sample_best_genomes.tsv", breadth_genomes_df),
        (breadth_dir / f"{sanitize_label(prefix)}_species_category_cross_sample_marker_long.tsv", breadth_long_df),
    ]
    for path, frame in outputs:
        frame.to_csv(path, sep="\t", index=False)
        wrote_paths.append(path)

    plot_specs = [
        (
            best_long_df.loc[best_long_df["support_class"].astype(str).eq("specific")].copy(),
            best_dir / f"{sanitize_label(prefix)}_species_category_best_marker_heatmap_specific",
            "Best-vs-best specific marker recovery by species and category",
            "Best representative fraction of possible specific markers recovered",
            "Cell text = observed specific markers / possible specific markers for the best representative in that species-category combination.",
        ),
        (
            best_long_df.loc[best_long_df["support_class"].astype(str).eq("generic")].copy(),
            best_dir / f"{sanitize_label(prefix)}_species_category_best_marker_heatmap_generic",
            "Best-vs-best generic marker recovery by species and category",
            "Best representative fraction of possible generic markers recovered",
            "Cell text = observed generic markers / possible generic markers for the best representative in that species-category combination.",
        ),
        (
            breadth_long_df.loc[breadth_long_df["support_class"].astype(str).eq("specific")].copy(),
            breadth_dir / f"{sanitize_label(prefix)}_species_category_cross_sample_marker_heatmap_specific",
            "Cross-sample breadth of specific marker recovery by species and category",
            "Fraction of species-positive samples with specific marker support",
            "Cell text = samples with specific support / samples where that species was recovered for the category.",
        ),
        (
            breadth_long_df.loc[breadth_long_df["support_class"].astype(str).eq("generic")].copy(),
            breadth_dir / f"{sanitize_label(prefix)}_species_category_cross_sample_marker_heatmap_generic",
            "Cross-sample breadth of generic marker recovery by species and category",
            "Fraction of species-positive samples with generic marker support",
            "Cell text = samples with generic support / samples where that species was recovered for the category.",
        ),
    ]
    for plot_df, plot_base, title, colorbar_label, footer_note in plot_specs:
        wrote = bool(
            plot_species_category_marker_heatmap(
                plot_df,
                str(plot_base),
                title,
                colorbar_label,
                footer_note,
                category_column=category_column,
            )
        )
        if wrote:
            wrote_paths.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])
    return wrote_paths


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
        figsize=(max(16, len(order) * 1.1), max(9.0, n_rows * 3.95)),
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
    compact_summary_df, _compact_informative_column = build_compact_category_summary(scored_df)
    method_summary_df = build_method_effectiveness_summary(scored_df, representative_df)
    sample_category_summary_df, variant_improvement_by_sample_df, variant_improvement_summary_df = build_variant_improvement_tables(scored_df)
    compact_summary_stats_df = build_compact_summary_stats(scored_df)
    representatives_out = Path(str(base) + "_sample_representatives.tsv")
    compact_set_out = Path(str(base) + "_compact_deduplicated_genomes.tsv")
    compact_summary_out = Path(str(base) + "_compact_summary_table.tsv")
    compact_summary_stats_out = Path(str(base) + "_compact_summary_stats.tsv")
    method_summary_out = Path(str(base) + "_method_effectiveness_summary.tsv")
    sample_category_summary_out = Path(str(base) + "_sample_category_summary.tsv")
    variant_improvement_by_sample_out = Path(str(base) + "_variant_improvement_by_sample.tsv")
    variant_improvement_summary_out = Path(str(base) + "_variant_improvement_summary.tsv")
    representative_df.to_csv(representatives_out, sep="\t", index=False)
    compact_df.to_csv(compact_set_out, sep="\t", index=False)
    compact_summary_df.to_csv(compact_summary_out, sep="\t", index=False)
    compact_summary_stats_df.to_csv(compact_summary_stats_out, sep="\t", index=False)
    method_summary_df.to_csv(method_summary_out, sep="\t", index=False)
    sample_category_summary_df.to_csv(sample_category_summary_out, sep="\t", index=False)
    variant_improvement_by_sample_df.to_csv(variant_improvement_by_sample_out, sep="\t", index=False)
    variant_improvement_summary_df.to_csv(variant_improvement_summary_out, sep="\t", index=False)
    wrote_paths.extend(
        [
            representatives_out,
            compact_set_out,
            compact_summary_out,
            compact_summary_stats_out,
            method_summary_out,
            sample_category_summary_out,
            variant_improvement_by_sample_out,
            variant_improvement_summary_out,
        ]
    )

    plot_specs = [
        (
            "compact_summary",
            lambda: plot_combined_category_compact_summary(
                scored_df,
                str(base) + "_compact_summary",
                stats_df=compact_summary_stats_df,
            ),
        ),
        ("distribution_facets", lambda: plot_combined_category_distributions(scored_df, str(base) + "_distribution_facets", stats_df=compact_summary_stats_df)),
        ("sample_count_heatmap", lambda: plot_sample_category_count_heatmap(scored_df, str(base) + "_sample_count_heatmap")),
        ("mode_support_heatmaps", lambda: plot_category_mode_support_heatmaps(scored_df, str(base) + "_mode_support_heatmaps")),
        ("pathway_presence_heatmap", lambda: plot_category_pathway_presence_heatmap(combined_pathway_df, scored_df, str(base) + "_pathway_presence_heatmap")),
        ("sample_representative_heatmap", lambda: plot_sample_representative_heatmap(representative_df, str(base) + "_sample_representative_heatmap")),
        ("method_effectiveness_panel", lambda: plot_method_effectiveness_panel(method_summary_df, str(base) + "_method_effectiveness_panel")),
        ("variant_improvement_before_after", lambda: plot_variant_improvement_before_after(variant_improvement_by_sample_df, str(base) + "_variant_improvement_before_after")),
        ("variant_improvement_deltas", lambda: plot_variant_improvement_deltas(variant_improvement_by_sample_df, str(base) + "_variant_improvement_deltas")),
    ]
    for label, plotter in plot_specs:
        wrote = bool(plotter())
        if wrote:
            plot_base = Path(str(base) + f"_{label}")
            wrote_paths.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])
    single_metric_dir = output_dir / "single_metric_plots"
    single_metric_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in metapathways_single_metric_specs(scored_df):
        plot_base = single_metric_dir / f"{sanitize_label(prefix)}_compare_category_{sanitize_label(metric)}"
        if plot_single_compact_metric(
            scored_df,
            metric,
            title,
            str(plot_base),
            stats_df=compact_summary_stats_df,
        ):
            wrote_paths.extend([Path(str(plot_base) + ".png"), Path(str(plot_base) + ".pdf")])
    wrote_paths.extend(
        write_species_category_marker_comparisons(
            output_dir,
            prefix,
            scored_df,
        )
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
    combined_annotation_audit_df=None,
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
    annotation_audit_out = output_dir / f"{prefix}_elemental_annotation_audit.tsv"
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
    if combined_annotation_audit_df is None:
        combined_annotation_audit_df = pd.DataFrame()
    combined_annotation_audit_df.to_csv(annotation_audit_out, sep="\t", index=False)
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
        annotation_audit_out,
        presence_out,
        score_out,
        coverage_out,
    ]


def main():
    args = build_parser().parse_args()
    python_exe = sys.executable or shutil.which("python3") or shutil.which("python")
    if not python_exe:
        raise RuntimeError("Could not determine a Python executable for running the phylogeny helper")
    manifest = read_manifest(args.manifest_tsv)
    manifest_path = Path(args.manifest_tsv).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else manifest_path.parent / "metapathways_batch_summary"
    )
    prefix = sanitize_label(args.prefix)
    organize_roots = {str(output_dir)}
    worker_count, prep_worker_count = effective_worker_settings(args)

    if (not args.disable_auto_atlas) and (not args.genome_atlas_dir):
        detected_atlas_dir = auto_detect_atlas_dir(manifest_path, output_dir, manifest)
        if detected_atlas_dir is not None:
            args.genome_atlas_dir = str(detected_atlas_dir)
            print(f"[done] auto-detected genome atlas dir: {detected_atlas_dir}")
        else:
            print(
                "[warn] no genome atlas directory auto-detected; MetaPathways best-of and "
                "best-ANI subset outputs require --genome-atlas-dir."
            )

    combined_genome = []
    combined_annotation = []
    combined_annotation_quality = []
    combined_annotation_audit = []
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
    skipped_inputs = []
    built_metapathways_best_sets = False
    marker_manifest_path = resolve_optional_path_arg(args.marker_manifest)
    marker_manifest = load_marker_manifest(marker_manifest_path) if marker_manifest_path else None

    print(f"[start] manifest rows: {len(manifest)}")
    for index, row in manifest.reset_index(drop=True).iterrows():
        sample = row["sample"]
        category = row["category"]
        input_paths = resolve_manifest_row_inputs(row)
        row_filter_context = build_row_filter_context(args, row)
        if row_filter_context["allowed_genomes"] is not None:
            print(
                f"[done] ({index + 1}/{len(manifest)}) loaded genome filter: "
                f"ids={len(row_filter_context['allowed_genomes']):,} via {row_filter_context['filter_tsv']}"
            )
        if row_filter_context["taxonomy_lookup"] is not None:
            print(
                f"[done] ({index + 1}/{len(manifest)}) loaded taxonomy labels: "
                f"aliases={len(row_filter_context['taxonomy_lookup']):,} "
                f"ambiguous={int(row_filter_context['taxonomy_ambiguous_alias_count']):,} "
                f"via {row_filter_context['taxonomy_tsv']}"
            )
        input_modes = []
        for input_path in input_paths:
            input_mode, _resolved_input = detect_input_mode(input_path)
            input_modes.append(input_mode)
        if len(set(input_modes)) != 1:
            raise ValueError(
                f"Manifest row mixes raw results and summary directories, which is not supported: {row['input_dir']}"
            )
        input_mode = input_modes[0]
        display_input = str(input_paths[0]) if len(input_paths) == 1 else f"{input_paths[0]} ... ({len(input_paths)} inputs)"
        print(
            f"[start] ({index + 1}/{len(manifest)}) sample '{sample}' "
            f"category '{category}' from {display_input} [{input_mode}]"
        )
        if input_mode == "results":
            group_output_dir = resolve_row_output_dir(row, input_paths, args)
            try:
                result_bundle = summarize_results_group(
                    input_dirs=input_paths,
                    group_output_dir=group_output_dir,
                    prefix=prefix,
                    row_filter_context=row_filter_context,
                    marker_manifest=marker_manifest,
                    args=args,
                )
            except (FileNotFoundError, ValueError) as exc:
                skipped_inputs.append(
                    {
                        "sample": sample,
                        "category": category,
                        "input_dir": str(row["input_dir"]),
                        "input_mode": input_mode,
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
                )
                print(
                    f"[warn] ({index + 1}/{len(manifest)}) skipping '{sample}' '{category}' from {row['input_dir']}: {exc}"
                )
                continue

            genome_summary = result_bundle["genome_summary"]
            annotation_summary = result_bundle["annotation_summary"]
            annotation_quality_summary = result_bundle["annotation_quality_summary"]
            marker_summary = result_bundle["marker_summary"]
            reference_mode_summary = result_bundle["reference_mode_summary"]
            elemental_annotation_summary = result_bundle["elemental_annotation_summary"]
            elemental_mode_annotation_summary = result_bundle["elemental_mode_annotation_summary"]
            elemental_pathway_support_summary = result_bundle["elemental_pathway_support_summary"]
            elemental_mode_pathway_support_summary = result_bundle["elemental_mode_pathway_support_summary"]
            elemental_pathway_summary = result_bundle["elemental_pathway_summary"]
            elemental_mode_pathway_summary = result_bundle["elemental_mode_pathway_summary"]
            pathway_long = result_bundle["pathway_long"]
            pathway_orf_long = result_bundle["pathway_orf_long"]
            annotation_audit_long = result_bundle["annotation_audit_long"]
            organize_roots.add(str(group_output_dir))
            for input_path in input_paths:
                organize_roots.add(str(input_path / args.individual_subdir))
            if result_bundle["skipped_inputs"]:
                for skipped in result_bundle["skipped_inputs"]:
                    skipped_inputs.append(
                        {
                            "sample": sample,
                            "category": category,
                            "input_dir": skipped["input_dir"],
                            "input_mode": input_mode,
                            "error_type": skipped["error_type"],
                            "error_message": skipped["error_message"],
                        }
                    )
        else:
            summary_parts = []
            for input_path in input_paths:
                summary_parts.append(load_existing_summary(input_path, marker_manifest=marker_manifest))
            (
                genome_parts,
                annotation_parts,
                annotation_quality_parts,
                marker_parts,
                reference_mode_parts,
                elemental_annotation_parts,
                elemental_mode_annotation_parts,
                elemental_pathway_support_parts,
                elemental_mode_pathway_support_parts,
                elemental_pathway_parts,
                elemental_mode_pathway_parts,
                pathway_parts,
                pathway_orf_parts,
                annotation_audit_parts,
            ) = map(list, zip(*summary_parts))
            genome_summary = pd.concat(genome_parts, ignore_index=True)
            annotation_summary = pd.concat(annotation_parts, ignore_index=True)
            annotation_quality_summary = pd.concat(annotation_quality_parts, ignore_index=True)
            marker_summary = pd.concat(marker_parts, ignore_index=True)
            reference_mode_summary = pd.concat(reference_mode_parts, ignore_index=True)
            elemental_annotation_summary = pd.concat(elemental_annotation_parts, ignore_index=True)
            elemental_mode_annotation_summary = pd.concat(elemental_mode_annotation_parts, ignore_index=True)
            elemental_pathway_support_summary = pd.concat(elemental_pathway_support_parts, ignore_index=True)
            elemental_mode_pathway_support_summary = pd.concat(elemental_mode_pathway_support_parts, ignore_index=True)
            elemental_pathway_summary = pd.concat(elemental_pathway_parts, ignore_index=True)
            elemental_mode_pathway_summary = pd.concat(elemental_mode_pathway_parts, ignore_index=True)
            pathway_long = pd.concat(pathway_parts, ignore_index=True)
            pathway_orf_long = pd.concat(pathway_orf_parts, ignore_index=True)
            annotation_audit_long = pd.concat(annotation_audit_parts, ignore_index=True)
            print(f"[done] ({index + 1}/{len(manifest)}) loaded existing summary inputs: {len(input_paths)}")

        input_dir_value = str(group_output_dir) if input_mode == "results" else str(row["input_dir"])
        combined_genome.append(add_context_columns(genome_summary, sample, category, input_dir_value))
        combined_annotation.append(add_context_columns(annotation_summary, sample, category, input_dir_value))
        combined_annotation_quality.append(add_context_columns(annotation_quality_summary, sample, category, input_dir_value))
        combined_annotation_audit.append(add_context_columns(annotation_audit_long, sample, category, input_dir_value))
        combined_marker.append(add_context_columns(marker_summary, sample, category, input_dir_value))
        combined_reference_mode.append(add_context_columns(reference_mode_summary, sample, category, input_dir_value))
        combined_elemental_annotation.append(add_context_columns(elemental_annotation_summary, sample, category, input_dir_value))
        combined_elemental_mode_annotation.append(add_context_columns(elemental_mode_annotation_summary, sample, category, input_dir_value))
        combined_elemental_pathway_support.append(add_context_columns(elemental_pathway_support_summary, sample, category, input_dir_value))
        combined_elemental_mode_pathway_support.append(add_context_columns(elemental_mode_pathway_support_summary, sample, category, input_dir_value))
        combined_elemental_pathway.append(add_context_columns(elemental_pathway_summary, sample, category, input_dir_value))
        combined_elemental_mode_pathway.append(add_context_columns(elemental_mode_pathway_summary, sample, category, input_dir_value))
        combined_pathway.append(add_context_columns(pathway_long, sample, category, input_dir_value))
        combined_pathway_orf.append(add_context_columns(pathway_orf_long, sample, category, input_dir_value))

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
    combined_annotation_audit_df = (
        pd.concat(combined_annotation_audit, ignore_index=True)
        if combined_annotation_audit
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
    written_paths = []
    if skipped_inputs:
        skipped_path = output_dir / f"{prefix}_skipped_inputs.tsv"
        pd.DataFrame(skipped_inputs).to_csv(skipped_path, sep="	", index=False)
        written_paths.append(str(skipped_path))
        print(f"[done] skipped inputs table: {skipped_path} (n={len(skipped_inputs)})")

    pathway_status_columns = [
        "sample",
        "category",
        "input_dir",
        "genome_id",
        "results_mode",
        "genome_dir",
        "broad_orf_source",
        "pathway_status",
        "pathway_missing_reason",
        "pathway_table_present",
        "pathway_orf_table_present",
    ]
    if not combined_genome_df.empty and "pathway_status" in combined_genome_df.columns:
        available_columns = [
            column for column in pathway_status_columns
            if column in combined_genome_df.columns
        ]
        pathway_status_path = output_dir / f"{prefix}_pathway_availability.tsv"
        combined_genome_df.loc[:, available_columns].to_csv(
            pathway_status_path,
            sep="\t",
            index=False,
        )
        written_paths.append(str(pathway_status_path))
        print(f"[done] pathway availability table: {pathway_status_path}")

    if combined_genome_df.empty:
        raise ValueError(
            "No valid MetaPathways summaries were produced from the manifest inputs. "
            "Check the skipped inputs table for failures."
        )

    written_paths.extend(write_combined_tables(
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
        combined_annotation_audit_df=combined_annotation_audit_df,
    ))

    print("[start] combined category summary/comparison plots")
    combined_plot_paths = run_combined_summary_plots(
        output_dir=output_dir,
        prefix=prefix,
        combined_genome_df=combined_genome_df,
        combined_pathway_df=combined_pathway_df,
    )
    written_paths.extend(combined_plot_paths)
    print(f"[done] combined category plots: {len(combined_plot_paths)} files")

    atlas_best_ready = has_atlas_component_members(args)
    metapathways_best_set_paths = []

    if atlas_best_ready:
        print("[start] MetaPathways best-of-sample / best-of-best selection")
        metapathways_best_set_paths = run_metapathways_best_sets(
            args=args,
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
            combined_annotation_audit_df=combined_annotation_audit_df,
        )
        written_paths.extend(metapathways_best_set_paths)
        built_metapathways_best_sets = True
        print(f"[done] MetaPathways best sets: {len(metapathways_best_set_paths)} files")
    else:
        print(
            "[warn] MetaPathways best-of-sample / best-of-best selection not run; "
            "no atlas component-member TSV was found. Provide --genome-atlas-dir containing "
            "tables/components/best_sets_review_component_members.tsv."
        )

    subset_path, subset_table = (None, pd.DataFrame())
    subset_prefix = sanitize_label(f"{prefix}_bestani")
    subset_genome_df = pd.DataFrame()
    subset_annotation_audit_df = pd.DataFrame()
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
        subset_annotation_audit_df = filter_frame_by_best_subset(combined_annotation_audit_df, subset_lookup)
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
            combined_annotation_audit_df=subset_annotation_audit_df,
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
    elif not args.skip_best_subset:
        print(
            "[warn] best-ANI subset comparisons not run; no best-set table was found. "
            "Provide --best-subset-tsv or --genome-atlas-dir containing best_sets_review_selected_genomes.tsv."
        )

    if args.skip_atlas_linked:
        print("[skip] atlas-linked category comparisons skipped by --skip-atlas-linked")
    elif args.genome_atlas_dir:
        from metapathways_atlas_linked_core import (
            has_atlas_annotated,
            has_atlas_shared_best,
            run_atlas_linked_comparisons,
        )

        atlas_ready = has_atlas_annotated(args) if args.atlas_linkage_mode == "species_all" else has_atlas_shared_best(args)
        if atlas_ready:
            atlas_args = build_atlas_linked_args(args, prefix)
            atlas_annotation_df = (
                pd.DataFrame()
                if args.skip_atlas_linked_annotation_presence
                else combined_annotation_audit_df
            )
            print(
                f"[start] atlas-linked category comparisons: mode={args.atlas_linkage_mode} "
                f"workers={worker_count}"
            )
            atlas_paths = run_atlas_linked_comparisons(
                args=atlas_args,
                output_dir=output_dir,
                combined_genome_df=combined_genome_df,
                combined_annotation_audit_df=atlas_annotation_df,
            )
            written_paths.extend(atlas_paths)
            print(f"[done] atlas-linked category comparisons: {len(atlas_paths)} files")

            if subset_path is not None and not subset_genome_df.empty:
                subset_atlas_prefix = sanitize_label(f"{subset_prefix}_atlas")
                subset_atlas_args = build_atlas_linked_args(args, subset_atlas_prefix)
                subset_atlas_annotation_df = (
                    pd.DataFrame()
                    if args.skip_atlas_linked_annotation_presence
                    else subset_annotation_audit_df
                )
                print(f"[start] atlas-linked best-ANI subset comparisons: prefix={subset_atlas_prefix}")
                subset_atlas_paths = run_atlas_linked_comparisons(
                    args=subset_atlas_args,
                    output_dir=output_dir,
                    combined_genome_df=subset_genome_df,
                    combined_annotation_audit_df=subset_atlas_annotation_df,
                    prefix_override=subset_atlas_prefix,
                )
                written_paths.extend(subset_atlas_paths)
                print(f"[done] atlas-linked best-ANI subset comparisons: {len(subset_atlas_paths)} files")
        else:
            needed = "*_annotated.tsv" if args.atlas_linkage_mode == "species_all" else "*_shared_best_genomes.tsv"
            print(
                "[warn] atlas-linked category comparisons not run; no compatible atlas file was found in "
                f"{args.genome_atlas_dir} for mode={args.atlas_linkage_mode} (need {needed})."
            )
    else:
        print(
            "[warn] atlas-linked category comparisons not run; provide --genome-atlas-dir "
            "or allow auto-detection of a compatible atlas directory."
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

    if args.run_denovo_phylogeny:
        if atlas_best_ready:
            print("[start] de novo phylogeny cohort export")
            denovo_set_paths = build_denovo_phylogeny_sets(
                args=args,
                output_dir=output_dir,
                combined_genome_df=combined_genome_df,
            )
            written_paths.extend(denovo_set_paths)
            print(f"[done] de novo phylogeny cohort export: {len(denovo_set_paths)} files")
            phylogeny_paths = run_denovo_phylogeny(
                python_exe=python_exe,
                output_dir=output_dir / "denovo_phylogeny",
                threads=args.denovo_phylogeny_threads,
                gtdbtk_data_path=args.denovo_gtdbtk_data_path,
            )
            written_paths.extend(phylogeny_paths)
        else:
            print(
                "[warn] de novo phylogeny requested, but no atlas component-member TSV was found. "
                "Provide --genome-atlas-dir containing tables/components/best_sets_review_component_members.tsv."
            )

    print(f"[done] combined outputs in: {output_dir}")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
