#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

pd = None

PLOT_EXTENSIONS = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".tif", ".tiff"}
TABLE_EXTENSIONS = {".tsv", ".csv", ".txt", ".json", ".yaml", ".yml", ".parquet"}
STYLE_KEYWORDS = [
    ("clustermap", "clustermap"),
    ("heatmap", "heatmap"),
    ("upset", "upset"),
    ("panel", "panel"),
    ("facet", "facet"),
    ("distribution", "distribution"),
    ("summary", "summary"),
    ("significance", "significance"),
    ("taxonomy", "taxonomy"),
    ("fastani", "fastani"),
    ("marker", "marker"),
    ("reference_mode", "reference_mode"),
    ("elemental", "elemental"),
    ("pathway", "pathway"),
    ("pair", "pairs"),
    ("component", "components"),
    ("selected", "selected"),
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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run genome_quality_atlas.py across multiple run directories, build a "
            "combined master table, and perform a global category comparison."
        )
    )
    parser.add_argument(
        "manifest_tsv",
        help=(
            "Preferred manifest TSV: sample, category, run_directory, bin_id_token_index. "
            "The fourth column is optional and is a 1-based period-delimited token index "
            "used to pre-deduplicate same-bin parameterization variants within each sample/category. "
            "A legacy two-column form (category, run_directory) is also accepted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory for the combined master and global atlas. Defaults "
            "to <manifest-dir>/genome_atlas_batch."
        ),
    )
    parser.add_argument(
        "--atlas-script",
        default=None,
        help="Path to genome_quality_atlas.py. Defaults to the copy next to this wrapper.",
    )
    parser.add_argument(
        "--sample-column",
        default="sample",
        help="Column name to add to the masters for the manifest sample. Default: sample",
    )
    parser.add_argument(
        "--category-column",
        default="category",
        help="Column name to add to the masters for the manifest category. Default: category",
    )
    parser.add_argument(
        "--group-column",
        default="Phylum",
        help="Group column passed through to genome_quality_atlas.py. Default: Phylum",
    )
    parser.add_argument(
        "--top-n-groups",
        type=int,
        default=12,
        help="Top-N groups passed through to genome_quality_atlas.py. Default: 12",
    )
    parser.add_argument(
        "--individual-output-name",
        default="genome_atlas",
        help="Per-directory atlas output folder name. Default: genome_atlas",
    )
    parser.add_argument(
        "--individual-prefix",
        default="genome_quality",
        help="Per-directory atlas file prefix. Default: genome_quality",
    )
    parser.add_argument(
        "--global-prefix",
        default="genome_quality",
        help="Combined atlas file prefix. Default: genome_quality",
    )
    parser.add_argument(
        "--individual-master-name",
        default="Master_genome_QC.atlas.tsv",
        help="Filename for the augmented per-directory master table. Default: Master_genome_QC.atlas.tsv",
    )
    parser.add_argument(
        "--combined-master-name",
        default="Master_genome_QC.combined.tsv",
        help="Filename for the combined master table. Default: Master_genome_QC.combined.tsv",
    )
    parser.add_argument(
        "--ani-threshold",
        type=float,
        default=95.0,
        help="ANI threshold passed through to genome_quality_atlas.py. Default: 95.0",
    )
    parser.add_argument(
        "--ani-af-threshold",
        type=float,
        default=0.5,
        help="Alignment-fraction threshold passed through to genome_quality_atlas.py. Default: 0.5",
    )
    parser.add_argument(
        "--ani-threads",
        type=int,
        default=1,
        help="FastANI thread count passed through to genome_quality_atlas.py. Default: 1",
    )
    parser.add_argument(
        "--ani-top-overlaps",
        type=int,
        default=15,
        help="Top overlap combinations passed through to genome_quality_atlas.py. Default: 15",
    )
    parser.add_argument(
        "--ani-results",
        default=None,
        help=(
            "Optional existing FastANI results file to reuse for the global "
            "category comparison."
        ),
    )
    parser.add_argument(
        "--matched-samples-only",
        action="store_true",
        help=(
            "Restrict global category-comparison outputs to samples present in "
            "every category. FastANI overlap/dedup is then limited to within-sample matches."
        ),
    )
    parser.add_argument(
        "--best-sample-dir-name",
        default="best_of_sample_sets",
        help="Directory name under the combined output for per-sample best deduplicated sets. Default: best_of_sample_sets",
    )
    parser.add_argument(
        "--best-global-dir-name",
        default="best_of_best",
        help="Directory name under the combined output for the global best deduplicated set. Default: best_of_best",
    )
    parser.add_argument(
        "--best-set-min-mimag-tier",
        default="low",
        choices=["low", "medium", "high"],
        help=(
            "Minimum MIMAG tier allowed into atlas best-of-sample / best-of-best "
            "selection. Default: low (keeps all genomes). Use 'medium' to exclude "
            "LQ genomes or 'high' to keep only HQ genomes."
        ),
    )
    parser.add_argument(
        "--skip-to-best",
        action="store_true",
        help=(
            "Skip manifest processing and atlas generation, and only export "
            "best-of-sample / best-of-best sets from existing combined output files."
        ),
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
        "--run-best-set-phylogeny",
        action="store_true",
        help=(
            "After best-of-sample / best-of-best export, run 16S and GTDB-marker "
            "phylogeny generation for every best-set directory."
        ),
    )
    parser.add_argument(
        "--best-set-phylogeny-threads",
        type=int,
        default=1,
        help=(
            "Threads passed to best_set_phylogeny.py for Barrnap, MAFFT, and "
            "GTDB-Tk. Default: 1"
        ),
    )
    parser.add_argument(
        "--best-set-gtdbtk-data-path",
        default=None,
        help=(
            "Optional GTDBTK_DATA_PATH override passed to best_set_phylogeny.py "
            "when building GTDB marker alignments/trees."
        ),
    )
    return parser


def ensure_pandas():
    global pd
    if pd is None:
        warnings.filterwarnings("ignore")
        import pandas as pd_mod

        pd = pd_mod
    return pd


def read_manifest(path):
    ensure_pandas()
    manifest = pd.read_csv(path, sep="\t", header=None, comment="#")
    if manifest.shape[1] < 2:
        raise ValueError("Manifest TSV must have at least two columns.")
    if manifest.shape[1] >= 4:
        manifest = manifest.iloc[:, :4].copy()
        manifest.columns = ["sample", "category", "dir_path", "bin_id_token_index"]
    elif manifest.shape[1] >= 3:
        manifest = manifest.iloc[:, :3].copy()
        manifest.columns = ["sample", "category", "dir_path"]
        manifest["bin_id_token_index"] = pd.NA
    else:
        manifest = manifest.iloc[:, :2].copy()
        manifest.columns = ["category", "dir_path"]
        manifest.insert(0, "sample", manifest["category"])
        manifest["bin_id_token_index"] = pd.NA
    manifest["category"] = manifest["category"].astype(str).str.strip()
    manifest["sample"] = manifest["sample"].astype(str).str.strip()
    manifest["dir_path"] = manifest["dir_path"].astype(str).str.strip()
    manifest["bin_id_token_index"] = (
        manifest["bin_id_token_index"]
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    parsed_index = pd.to_numeric(manifest["bin_id_token_index"], errors="coerce")
    invalid_mask = parsed_index.notna() & ((parsed_index < 1) | (parsed_index % 1 != 0))
    if invalid_mask.any():
        bad_values = ", ".join(manifest.loc[invalid_mask, "bin_id_token_index"].astype(str).tolist()[:5])
        raise ValueError(
            "Manifest bin_id_token_index values must be positive 1-based integers. "
            f"Examples of invalid values: {bad_values}"
        )
    manifest["bin_id_token_index"] = parsed_index.astype("Int64")
    manifest = manifest.loc[
        (manifest["sample"] != "") & (manifest["category"] != "") & (manifest["dir_path"] != "")
    ]
    if manifest.empty:
        raise ValueError("Manifest TSV did not contain any usable sample/category/directory rows.")
    return manifest


def atlas_script_path(user_value):
    if user_value:
        path = os.path.abspath(os.path.expanduser(user_value))
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genome_quality_atlas.py")
    if not os.path.exists(path):
        raise ValueError(f"Atlas script was not found: {path}")
    return path


def best_set_phylogeny_script_path():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_set_phylogeny.py")
    if not os.path.exists(path):
        raise ValueError(f"Best-set phylogeny script was not found: {path}")
    return path


def sanitize_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unknown"


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

    def sort_key(value):
        rank = method_rank(value)
        if rank is not None:
            return (0, rank, str(value).lower())
        if counts is not None:
            return (1, -int(counts.get(value, 0)), str(value).lower())
        return (1, 0, str(value).lower())

    return sorted(unique, key=sort_key)


def bh_adjust(pvalues):
    cleaned = []
    for index, value in enumerate(pvalues):
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        cleaned.append((index, numeric))
    valid = [(index, value) for index, value in cleaned if not math.isnan(value)]
    adjusted = [float("nan")] * len(cleaned)
    if not valid:
        return adjusted
    valid.sort(key=lambda item: item[1])
    total = len(valid)
    running = 1.0
    for rank in range(total, 0, -1):
        index, value = valid[rank - 1]
        scaled = value * total / float(rank)
        running = min(running, scaled)
        adjusted[index] = min(running, 1.0)
    return adjusted


def exact_binomial_test_two_sided(k, n):
    if n <= 0:
        return float("nan")
    pmf_obs = math.comb(int(n), int(k)) / (2.0 ** int(n))
    total = 0.0
    for x in range(int(n) + 1):
        pmf_x = math.comb(int(n), int(x)) / (2.0 ** int(n))
        if pmf_x <= pmf_obs + 1e-15:
            total += pmf_x
    return min(float(total), 1.0)


def significance_stars(pvalue):
    if pvalue is None or pd.isna(pvalue):
        return ""
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 5e-2:
        return "*"
    return ""


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


def organize_directory_outputs(directory):
    ensure_pandas()
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
        if len(entry.parts) >= 2 and entry.parent.name in {"plots", "tables"}:
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
    for current_dir, dirnames, _ in os.walk(root_path):
        dirnames[:] = [name for name in dirnames if name not in {"plots", "tables", "_organized", "selected_set", "phylogeny"}]
        count, index_path = organize_directory_outputs(current_dir)
        if count > 0:
            total_files += count
            if index_path is not None:
                index_paths.append(index_path)
    return total_files, index_paths


def resolve_output_file(base_dir, filename):
    base_path = Path(base_dir).resolve()
    direct = base_path / filename
    if direct.exists():
        return str(direct)

    matches = [path for path in base_path.rglob(filename) if path.is_file()]
    if not matches:
        return None

    matches.sort(
        key=lambda path: (
            len(path.parts),
            -path.stat().st_mtime,
            str(path),
        )
    )
    return str(matches[0].resolve())


def normalize_gunc_genome_id(value):
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    for suffix in [
        ".fasta",
        ".fa",
        ".fna",
        ".fasta.gz",
        ".fa.gz",
        ".fna.gz",
        ".gz",
    ]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def find_gunc_summary_path(run_dir):
    patterns = [
        os.path.join(run_dir, "gunc", "gunc", "GUNC.*.maxCSS_level.tsv"),
        os.path.join(run_dir, "gunc", "GUNC.*.maxCSS_level.tsv"),
    ]
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(pattern))
    if not matches:
        return None
    return sorted({os.path.abspath(path) for path in matches})[0]


def load_gunc_summary(run_dir):
    ensure_pandas()
    gunc_summary_path = find_gunc_summary_path(run_dir)
    if not gunc_summary_path:
        return None

    gunc_df = pd.read_csv(gunc_summary_path, sep="\t")
    if gunc_df.empty or "genome" not in gunc_df.columns:
        return None

    gunc_df = gunc_df.copy()
    gunc_df["__gunc_key"] = gunc_df["genome"].map(normalize_gunc_genome_id)
    gunc_df = gunc_df.dropna(subset=["__gunc_key"]).drop_duplicates("__gunc_key")
    rename_map = {
        "n_genes_called": "gunc_n_genes_called",
        "n_genes_mapped": "gunc_n_genes_mapped",
        "n_contigs": "gunc_n_contigs",
        "taxonomic_level": "gunc_taxonomic_level",
        "proportion_genes_retained_in_major_clades": "gunc_proportion_genes_retained_in_major_clades",
        "genes_retained_index": "gunc_genes_retained_index",
        "clade_separation_score": "gunc_clade_separation_score",
        "contamination_portion": "gunc_contamination_portion",
        "n_effective_surplus_clades": "gunc_n_effective_surplus_clades",
        "mean_hit_identity": "gunc_mean_hit_identity",
        "reference_representation_score": "gunc_reference_representation_score",
        "pass.GUNC": "gunc_pass",
    }
    keep_columns = ["__gunc_key"] + [column for column in rename_map if column in gunc_df.columns]
    return gunc_df.loc[:, keep_columns].rename(columns=rename_map)


def merge_gunc_summary(master, run_dir):
    ensure_pandas()
    if "gunc_clade_separation_score" in master.columns:
        return master

    gunc_df = load_gunc_summary(run_dir)
    if gunc_df is None:
        return master

    merge_candidates = []
    if "Bin Id" in master.columns:
        merge_candidates.append("Bin Id")
    if "Genome_Id" in master.columns:
        merge_candidates.append("Genome_Id")

    for merge_column in merge_candidates:
        working = master.copy()
        working["__gunc_key"] = working[merge_column].map(normalize_gunc_genome_id)
        merged = working.merge(gunc_df, on="__gunc_key", how="left").drop(columns=["__gunc_key"])
        if merged["gunc_clade_separation_score"].notna().any():
            return merged
    return master


def build_fasta_map(run_dir):
    fasta_dir = Path(run_dir) / "dedupe" / "fasta"
    fasta_files = sorted(str(path.resolve()) for path in fasta_dir.rglob("*.fasta") if path.is_file())
    if not fasta_files:
        raise ValueError(f"No FASTA files were found under {fasta_dir}")
    fasta_map = {}
    representative_stems = {}
    for path in fasta_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        full_path = os.path.abspath(path)
        representative_stems[stem] = full_path
        for alias in generate_id_aliases(stem):
            fasta_map.setdefault(alias, full_path)
    return fasta_map, representative_stems


def choose_pre_dedup_source_column(master):
    for column in ["Genome_Id", "Bin Id"]:
        if column in master.columns:
            return column
    return None


def extract_period_token(value, token_index):
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split(".")
    if token_index < 1 or token_index > len(parts):
        return None
    return parts[token_index - 1].strip() or None


def compute_pre_ani_quality_proxy(frame, atlas_module):
    working = frame.copy()
    numeric_defaults = {
        "qscore": float("-inf"),
        "N50": float("-inf"),
        "sum_len": float("-inf"),
    }
    for column, default in numeric_defaults.items():
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(default)
        else:
            working[column] = default
    return working


def pre_deduplicate_master(master, token_index, atlas_module=None):
    if token_index is None or pd.isna(token_index):
        return master.copy(), {
            "enabled": False,
            "input_rows": len(master),
            "output_rows": len(master),
            "removed_rows": 0,
            "source_column": None,
            "token_index": None,
        }

    source_column = choose_pre_dedup_source_column(master)
    if source_column is None:
        raise ValueError(
            "Pre-ANI manifest deduplication was requested, but neither 'Genome_Id' nor 'Bin Id' exists in the master table."
        )

    working = master.copy()
    working["pre_ani_bin_key"] = working[source_column].map(lambda value: extract_period_token(value, int(token_index)))
    invalid_mask = working["pre_ani_bin_key"].isna() | working["pre_ani_bin_key"].astype(str).eq("")
    if invalid_mask.any():
        examples = ", ".join(
            working.loc[invalid_mask, source_column].astype(str).drop_duplicates().tolist()[:5]
        )
        raise ValueError(
            f"Could not extract period-delimited token {int(token_index)} from column '{source_column}'. "
            f"Examples: {examples}"
        )

    if atlas_module is None:
        atlas_module = load_atlas_module()
    working = compute_pre_ani_quality_proxy(working, atlas_module)
    sort_specs = [
        ("qscore", False),
        ("N50", False),
        ("sum_len", False),
        ("Genome_Id", True),
        ("Bin Id", True),
    ]
    sort_columns = [column for column, _ in sort_specs if column in working.columns]
    sort_ascending = [ascending for column, ascending in sort_specs if column in working.columns]
    working = working.sort_values(by=sort_columns, ascending=sort_ascending, kind="mergesort")

    duplicate_counts = working.groupby("pre_ani_bin_key").size().rename("pre_ani_duplicate_count")
    selected = working.groupby("pre_ani_bin_key", as_index=False, sort=False).head(1).copy()
    selected["pre_ani_duplicate_count"] = selected["pre_ani_bin_key"].map(duplicate_counts).astype(int)
    selected["pre_ani_token_source_column"] = source_column
    selected["pre_ani_token_index"] = int(token_index)
    selected["pre_ani_best_selection_metric"] = "qscore>N50>sum_len"

    drop_columns = []
    selected = selected.drop(columns=drop_columns, errors="ignore")
    return selected, {
        "enabled": True,
        "input_rows": len(master),
        "output_rows": len(selected),
        "removed_rows": len(master) - len(selected),
        "source_column": source_column,
        "token_index": int(token_index),
    }


def generate_id_aliases(value):
    text = str(value).strip()
    aliases = []
    if not text:
        return aliases

    def add(candidate):
        candidate = str(candidate).strip()
        if candidate and candidate not in aliases:
            aliases.append(candidate)

    add(text)

    if "." in text:
        parts = text.split(".")
        for end in range(len(parts) - 1, 0, -1):
            add(".".join(parts[:end]))

    bin_match = re.match(r"^(bin_\d+)", text)
    if bin_match:
        add(bin_match.group(1))

    genome_match = re.match(r"^(.+?)(?:\.best_match.*|\.majority_rule.*|\.ocsvm.*|\.intersect.*|\.xPG.*)$", text)
    if genome_match:
        add(genome_match.group(1))

    return aliases


def find_embedded_representative(candidate, representative_stems):
    text = str(candidate).strip()
    if not text:
        return None

    matches = []
    for stem, path in representative_stems.items():
        pattern = rf"(^|[._-]){re.escape(stem)}($|[._-])"
        if re.search(pattern, text):
            matches.append((stem, path))

    if not matches:
        return None

    matches.sort(key=lambda item: len(item[0]), reverse=True)
    longest_length = len(matches[0][0])
    longest = {(stem, path) for stem, path in matches if len(stem) == longest_length}
    if len(longest) == 1:
        return next(iter(longest))[1]
    return None


def augment_master(master_path, run_dir, sample, sample_column, category, category_column, bin_id_token_index=None, atlas_module=None):
    ensure_pandas()
    master = pd.read_csv(master_path, sep="\t")
    master, dedup_summary = pre_deduplicate_master(master, bin_id_token_index)
    master = merge_gunc_summary(master, run_dir)
    fasta_map, representative_stems = build_fasta_map(run_dir)
    fasta_paths = []
    missing = []

    for _, row in master.iterrows():
        candidates = []
        if "Bin Id" in master.columns:
            candidates.extend(generate_id_aliases(row["Bin Id"]))
        if "Genome_Id" in master.columns:
            candidates.extend(generate_id_aliases(row["Genome_Id"]))
        found = None
        for candidate in candidates:
            if candidate in fasta_map:
                found = fasta_map[candidate]
                break
        if found is None:
            for candidate in candidates:
                found = find_embedded_representative(candidate, representative_stems)
                if found is not None:
                    break
        fasta_paths.append(found)
        if found is None:
            missing.append(candidates[0] if candidates else "unknown")

    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Could not resolve FASTA paths for {len(missing)} genomes in {run_dir}. Examples: {preview}"
        )

    augmented = master.copy()
    augmented[sample_column] = sample
    augmented[category_column] = category
    augmented["source_dir"] = os.path.abspath(run_dir)
    augmented["fasta_path"] = fasta_paths
    return augmented, dedup_summary


def export_pre_ani_representative_fastas(frame, output_dir):
    ensure_pandas()
    if "pre_ani_bin_key" not in frame.columns or "fasta_path" not in frame.columns:
        return []

    reps = frame.loc[
        frame["pre_ani_bin_key"].notna()
        & frame["fasta_path"].notna()
        & frame["pre_ani_bin_key"].astype(str).str.strip().ne("")
        & frame["fasta_path"].astype(str).str.strip().ne("")
    ].copy()
    if reps.empty:
        return []

    rep_fastas_dir = os.path.join(output_dir, "rep_fastas")
    os.makedirs(rep_fastas_dir, exist_ok=True)

    written = []
    for row in reps.itertuples(index=False):
        rep_key = str(row.pre_ani_bin_key).strip()
        source_path = os.path.abspath(os.path.expanduser(str(row.fasta_path).strip()))
        target_path = os.path.join(rep_fastas_dir, f"{rep_key}.fasta")
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Representative FASTA source was not found: {source_path}")
        shutil.copy2(source_path, target_path)
        written.append(target_path)

    return sorted(written)


def find_master_path(run_dir):
    preferred = os.path.join(run_dir, "Master_genome_QC.tsv")
    fallback = os.path.join(run_dir, "Master_genome_QC.representatives.tsv")
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(fallback):
        return fallback
    raise ValueError(
        "Master table was not found. Expected one of: "
        f"{preferred}, {fallback}"
    )


def log_step(message):
    print(message, flush=True)


def run_command(cmd, description):
    log_step(f"[start] {description}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    log_step(f"[done] {description}")


def run_best_set_phylogeny(python_exe, output_dir, threads, gtdbtk_data_path=None):
    phylogeny_script = best_set_phylogeny_script_path()
    cmd = [
        python_exe,
        phylogeny_script,
        output_dir,
        "--threads",
        str(max(1, int(threads))),
    ]
    if gtdbtk_data_path:
        cmd.extend(["--gtdbtk-data-path", os.path.abspath(os.path.expanduser(gtdbtk_data_path))])
    log_step(f"[start] best-set phylogeny: {output_dir}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    completed = subprocess.run(cmd, check=False, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"Best-set phylogeny failed with exit code {completed.returncode}: {' '.join(cmd)}"
        )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    log_step(f"[done] best-set phylogeny: {output_dir}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def maybe_add_group_column(cmd, group_column):
    if group_column:
        cmd.extend(["--group-column", group_column])


def run_individual_atlas(python_exe, atlas_script, master_path, output_dir, prefix, group_column, top_n_groups):
    cmd = [
        python_exe,
        atlas_script,
        master_path,
        "--output-dir",
        output_dir,
        "--prefix",
        prefix,
        "--top-n-groups",
        str(top_n_groups),
    ]
    maybe_add_group_column(cmd, group_column)
    run_command(cmd, f"individual atlas: {master_path}")


def run_selected_set_atlas(
    python_exe,
    atlas_script,
    master_path,
    output_dir,
    prefix,
    group_column,
    top_n_groups,
    category_column,
    sample_column,
):
    cmd = [
        python_exe,
        atlas_script,
        master_path,
        "--output-dir",
        output_dir,
        "--prefix",
        prefix,
        "--top-n-groups",
        str(top_n_groups),
        "--sample-column",
        sample_column,
        "--compare-column",
        category_column,
    ]
    maybe_add_group_column(cmd, group_column)
    run_command(cmd, f"selected-set atlas: {master_path}")


def run_global_atlas(
    python_exe,
    atlas_script,
    master_path,
    output_dir,
    prefix,
    group_column,
    sample_column,
    top_n_groups,
    category_column,
    ani_threshold,
    ani_af_threshold,
    ani_threads,
    ani_top_overlaps,
    ani_results=None,
    matched_samples_only=False,
):
    cmd = [
        python_exe,
        atlas_script,
        master_path,
        "--output-dir",
        output_dir,
        "--prefix",
        prefix,
        "--top-n-groups",
        str(top_n_groups),
        "--sample-column",
        sample_column,
        "--compare-column",
        category_column,
        "--ani-compare-column",
        category_column,
        "--ani-fasta-column",
        "fasta_path",
        "--ani-threshold",
        str(ani_threshold),
        "--ani-af-threshold",
        str(ani_af_threshold),
        "--ani-threads",
        str(ani_threads),
        "--ani-top-overlaps",
        str(ani_top_overlaps),
    ]
    maybe_add_group_column(cmd, group_column)
    if matched_samples_only:
        cmd.append("--matched-samples-only")
    if ani_results:
        cmd.extend(["--ani-results", os.path.abspath(os.path.expanduser(ani_results))])
    run_command(cmd, f"global atlas: {master_path}")




def classify_category_family(value):
    text = str(value).strip().lower()
    if "mag" in text and "sag" not in text:
        return "mag_family"
    if "sag" in text and "mag" not in text:
        return "sag_family"
    return None


def run_family_comparison_atlases(
    combined,
    output_dir,
    combined_master_name,
    python_exe,
    atlas_script,
    prefix,
    group_column,
    sample_column,
    top_n_groups,
    category_column,
    ani_threshold,
    ani_af_threshold,
    ani_threads,
    ani_top_overlaps,
    ani_results=None,
    matched_samples_only=False,
):
    ensure_pandas()
    written_paths = []
    organize_roots = []
    family_labels = {
        "mag_family": "MAG-family",
        "sag_family": "SAG-family",
    }

    for family_key, family_label in family_labels.items():
        subset = combined.loc[
            combined[category_column].map(classify_category_family) == family_key
        ].copy()
        if subset.empty:
            log_step(f"[info] skipping {family_label} comparison: no rows found")
            continue

        n_categories = subset[category_column].astype(str).nunique()
        if n_categories < 2:
            log_step(
                f"[info] skipping {family_label} comparison: only {n_categories} category present"
            )
            continue

        family_output_dir = os.path.join(output_dir, family_key)
        os.makedirs(family_output_dir, exist_ok=True)
        family_master_path = os.path.join(family_output_dir, combined_master_name)
        subset.to_csv(family_master_path, sep="	", index=False)
        written_paths.append(family_master_path)
        organize_roots.append(os.path.abspath(family_output_dir))
        log_step(
            f"[start] {family_label} atlas: {family_master_path} "
            f"({len(subset)} genomes; {n_categories} categories)"
        )
        run_global_atlas(
            python_exe=python_exe,
            atlas_script=atlas_script,
            master_path=family_master_path,
            output_dir=family_output_dir,
            prefix=prefix,
            group_column=group_column,
            sample_column=sample_column,
            top_n_groups=top_n_groups,
            category_column=category_column,
            ani_threshold=ani_threshold,
            ani_af_threshold=ani_af_threshold,
            ani_threads=ani_threads,
            ani_top_overlaps=ani_top_overlaps,
            ani_results=ani_results,
            matched_samples_only=matched_samples_only,
        )
        written_paths.append(family_output_dir)
        log_step(f"[done] {family_label} outputs in: {family_output_dir}")

    return written_paths, organize_roots


def resolve_fastani_raw_path(output_dir, global_prefix, category_column, ani_results=None):
    if ani_results:
        candidate = os.path.abspath(os.path.expanduser(ani_results))
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Provided --ani-results file was not found: {candidate}")
        return candidate

    safe_category = sanitize_token(category_column)
    preferred = os.path.join(
        output_dir,
        f"{global_prefix}_compare_{safe_category}_fastani_fastani_raw.tsv",
    )
    legacy = os.path.join(
        output_dir,
        f"{global_prefix}_compare_{safe_category}_fastani_raw.tsv",
    )
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(legacy):
        return legacy

    pattern = os.path.join(
        output_dir,
        f"{global_prefix}_compare_{safe_category}*fastani*raw*.tsv",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        matches = sorted(
            [
                str(path)
                for path in Path(output_dir).rglob(f"{global_prefix}_compare_{safe_category}*fastani*raw*.tsv")
                if path.is_file()
            ]
        )
    if len(matches) == 1:
        return os.path.abspath(matches[0])
    if len(matches) > 1:
        raise FileNotFoundError(
            "Multiple FastANI raw result files matched; please pass --ani-results explicitly. "
            f"Matches: {', '.join(matches)}"
        )
    raise FileNotFoundError(f"FastANI raw results were not found under pattern: {pattern}")


def load_atlas_module():
    import importlib.util

    module_path = Path(__file__).with_name("genome_quality_atlas.py")
    spec = importlib.util.spec_from_file_location("genome_quality_atlas", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load atlas module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_fastani_frame(annotated):
    frame = annotated.copy()
    if "fasta_path" not in frame.columns:
        raise ValueError("Annotated atlas table is missing fasta_path; cannot build deduplicated exports.")
    frame["ani_fasta_path"] = frame["fasta_path"].astype(str).map(lambda value: os.path.abspath(os.path.expanduser(value)))
    frame["ani_record_id"] = frame["ani_fasta_path"]
    return frame


def filter_pair_df_to_frame_records(pair_df, frame):
    record_set = set(frame["ani_record_id"].astype(str).tolist())
    filtered = pair_df.loc[
        pair_df["record_a"].astype(str).isin(record_set)
        & pair_df["record_b"].astype(str).isin(record_set)
    ].copy()
    return filtered


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


def select_best_per_component(frame, pair_summary, atlas_module, category_column, sample_column):
    working = frame.copy()
    working["ani_record_id"] = working["ani_record_id"].astype(str)
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    working["__mimag_rank"] = (
        working["mimag_tier"]
        .astype(str)
        .str.lower()
        .map(tier_rank)
        .fillna(3)
        .astype(int)
    )
    working["__integrity_bin"] = coarse_quality_bin(working.get("integrity_score"), 0.02)
    working["__recoverability_bin"] = coarse_quality_bin(working.get("recoverability_score"), 0.02)
    working["__qscore_bin"] = coarse_quality_bin(working.get("qscore"), 5.0)
    working["__has_16s"] = best_set_has_16s(working)
    if "gunc_strict_assessment" in working.columns:
        working["__gunc_strict_rank"] = gunc_assessment_rank(working["gunc_strict_assessment"])
    sort_specs = [
        ("__mimag_rank", True),
        ("__gunc_strict_rank", True),
        ("__integrity_bin", False),
        ("__recoverability_bin", False),
        ("__qscore_bin", False),
        ("__has_16s", False),
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
    if hasattr(atlas_module, "strict_component_map_from_pairs"):
        component_map = atlas_module.strict_component_map_from_pairs(
            working,
            pair_summary,
            order_by=sort_columns,
            ascending=sort_ascending,
        )
    else:
        component_map = atlas_module.component_map_from_pairs(working, pair_summary)
    working["component_id"] = working["ani_record_id"].map(component_map)
    working = working.dropna(subset=["component_id"]).copy()
    if working.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    component_categories = (
        working.groupby("component_id")[category_column]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .rename("component_categories")
    )
    component_samples = (
        working.groupby("component_id")[sample_column]
        .apply(lambda values: ";".join(sorted(set(values.astype(str)))))
        .rename("component_samples")
    )
    component_member_count = working.groupby("component_id").size().rename("component_member_count")
    if sort_columns:
        working = working.sort_values(by=sort_columns, ascending=sort_ascending, kind="mergesort")

    selected = (
        working.groupby("component_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected["component_member_count"] = selected["component_id"].map(component_member_count)
    selected["component_categories"] = selected["component_id"].map(component_categories)
    selected["component_samples"] = selected["component_id"].map(component_samples)
    selected["best_selection_metric"] = (
        "ani_strict_component>mimag_tier>"
        "gunc(class_gate_only)>"
        "coarse(integrity,recoverability,qscore)>16S_presence>"
        "integrity>recoverability>qscore>completeness>contamination"
    )

    selected_records = set(selected["ani_record_id"].astype(str).tolist())
    member_table = working.copy()
    member_table["component_member_count"] = member_table["component_id"].map(component_member_count)
    member_table["component_categories"] = member_table["component_id"].map(component_categories)
    member_table["component_samples"] = member_table["component_id"].map(component_samples)
    member_table["is_selected"] = member_table["ani_record_id"].astype(str).isin(selected_records)

    winner_columns = [
        "component_id",
        "ani_record_id",
        category_column,
        sample_column,
        "Genome_Id",
        "Bin Id",
        "mimag_tier",
        "integrity_score",
        "recoverability_score",
        "qscore",
        "Completeness",
        "Contamination",
        "gunc_assessment",
        "gunc_strict_assessment",
        "gunc_clade_separation_score",
        "gunc_reference_representation_score",
        "component_member_count",
        "component_categories",
        "component_samples",
    ]
    winner_columns = [column for column in winner_columns if column in selected.columns]
    winner_table = selected.loc[:, winner_columns].rename(
        columns={
            "ani_record_id": "winner_ani_record_id",
            category_column: "winner_category",
            sample_column: "winner_sample",
            "Genome_Id": "winner_Genome_Id",
            "Bin Id": "winner_Bin_Id",
            "mimag_tier": "winner_mimag_tier",
            "integrity_score": "winner_integrity_score",
            "recoverability_score": "winner_recoverability_score",
            "qscore": "winner_qscore",
            "Completeness": "winner_Completeness",
            "Contamination": "winner_Contamination",
            "gunc_assessment": "winner_gunc_assessment",
            "gunc_strict_assessment": "winner_gunc_strict_assessment",
            "gunc_clade_separation_score": "winner_gunc_clade_separation_score",
            "gunc_reference_representation_score": "winner_gunc_reference_representation_score",
            "component_member_count": "winner_component_member_count",
            "component_categories": "winner_component_categories",
            "component_samples": "winner_component_samples",
        }
    )
    member_table = member_table.merge(winner_table, on="component_id", how="left")
    member_table["winner_same_category"] = member_table[category_column].astype(str) == member_table["winner_category"].astype(str)
    member_table["winner_same_sample"] = member_table[sample_column].astype(str) == member_table["winner_sample"].astype(str)
    if "winner_qscore" in member_table.columns and "qscore" in member_table.columns:
        member_table["winner_qscore_delta"] = member_table["winner_qscore"] - member_table["qscore"]

    tier_counts = (
        working.assign(__tier=working["mimag_tier"].astype(str).str.lower())
        .groupby(["component_id", "__tier"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={"high": "n_hq_members", "medium": "n_mq_members", "low": "n_lq_members"})
        .reset_index()
    )

    component_summary = (
        working.groupby("component_id")
        .agg(
            n_genomes=("ani_record_id", "size"),
            n_categories=(category_column, pd.Series.nunique),
            n_samples=(sample_column, pd.Series.nunique),
            categories=(category_column, lambda values: ";".join(sorted(set(values.astype(str))))),
            samples=(sample_column, lambda values: ";".join(sorted(set(values.astype(str))))),
            max_qscore=("qscore", "max"),
            median_qscore=("qscore", "median"),
            median_completeness=("Completeness", "median"),
            median_contamination=("Contamination", "median"),
        )
        .reset_index()
    )
    component_summary = component_summary.merge(tier_counts, on="component_id", how="left")
    component_summary = component_summary.merge(winner_table, on="component_id", how="left")

    member_table = member_table.drop(
        columns=[
            "__mimag_rank",
            "__gunc_strict_rank",
            "__gunc_css",
            "__gunc_rrs",
            "__integrity_bin",
            "__recoverability_bin",
            "__qscore_bin",
            "__has_16s",
        ],
        errors="ignore",
    )
    selected = selected.drop(
        columns=[
            "__mimag_rank",
            "__gunc_strict_rank",
            "__gunc_css",
            "__gunc_rrs",
            "__integrity_bin",
            "__recoverability_bin",
            "__qscore_bin",
            "__has_16s",
        ],
        errors="ignore",
    )
    return selected, member_table, component_summary


def summarize_selected_set(selected, source_frame, sample_column, category_column):
    if selected.empty:
        return pd.DataFrame(
            [{
                "n_input_genomes": len(source_frame),
                "n_selected_genomes": 0,
                "n_components": 0,
                "n_samples": 0,
                "n_categories": 0,
                "median_qscore": float("nan"),
                "mean_qscore": float("nan"),
                "median_integrity": float("nan"),
                "median_recoverability": float("nan"),
                "hq_genomes": 0,
                "mq_genomes": 0,
                "lq_genomes": 0,
            }]
        )

    return pd.DataFrame(
        [{
            "n_input_genomes": len(source_frame),
            "n_selected_genomes": len(selected),
            "n_components": int(selected["component_id"].nunique()),
            "n_samples": int(selected[sample_column].astype(str).nunique()),
            "n_categories": int(selected[category_column].astype(str).nunique()),
            "median_qscore": float(selected["qscore"].median()),
            "mean_qscore": float(selected["qscore"].mean()),
            "median_integrity": float(selected["integrity_score"].median()),
            "median_recoverability": float(selected["recoverability_score"].median()),
            "hq_genomes": int((selected["mimag_tier"].astype(str) == "high").sum()),
            "mq_genomes": int((selected["mimag_tier"].astype(str) == "medium").sum()),
            "lq_genomes": int((selected["mimag_tier"].astype(str) == "low").sum()),
        }]
    )


def summarize_best_set_category_contributions(source_frame, selected, member_table, category_column):
    categories = sorted(source_frame[category_column].astype(str).dropna().unique().tolist())
    contribution = pd.DataFrame({category_column: categories})
    if contribution.empty:
        return contribution

    tier_map = {"high": "hq", "medium": "mq", "low": "lq"}

    input_counts = source_frame.groupby(category_column).size().rename("n_input_genomes")
    contribution = contribution.merge(input_counts, on=category_column, how="left")
    for tier_value, tier_label in tier_map.items():
        tier_counts = (
            source_frame.loc[source_frame["mimag_tier"].astype(str).str.lower() == tier_value]
            .groupby(category_column)
            .size()
            .rename(f"n_input_{tier_label}")
        )
        contribution = contribution.merge(tier_counts, on=category_column, how="left")

    selected_counts = selected.groupby(category_column).size().rename("n_selected_genomes") if not selected.empty else pd.Series(dtype=int)
    contribution = contribution.merge(selected_counts, on=category_column, how="left")
    for tier_value, tier_label in tier_map.items():
        tier_counts = (
            selected.loc[selected["mimag_tier"].astype(str).str.lower() == tier_value]
            .groupby(category_column)
            .size()
            .rename(f"n_selected_{tier_label}")
            if not selected.empty else pd.Series(dtype=int)
        )
        contribution = contribution.merge(tier_counts, on=category_column, how="left")

    component_presence = (
        member_table.groupby(category_column)["component_id"].nunique().rename("n_components_with_category_present")
        if not member_table.empty else pd.Series(dtype=int)
    )
    contribution = contribution.merge(component_presence, on=category_column, how="left")
    component_wins = (
        selected.groupby(category_column)["component_id"].nunique().rename("n_components_won")
        if not selected.empty else pd.Series(dtype=int)
    )
    contribution = contribution.merge(component_wins, on=category_column, how="left")

    numeric_columns = [column for column in contribution.columns if column != category_column]
    for column in numeric_columns:
        contribution[column] = contribution[column].fillna(0)
        if column.startswith("n_"):
            contribution[column] = contribution[column].astype(int)

    contribution["selected_fraction_of_input"] = contribution["n_selected_genomes"] / contribution["n_input_genomes"].replace(0, pd.NA)
    contribution["win_fraction_when_present"] = contribution["n_components_won"] / contribution["n_components_with_category_present"].replace(0, pd.NA)
    contribution["selected_share_of_best_set"] = contribution["n_selected_genomes"] / max(int(len(selected)), 1)
    category_counts = dict(zip(contribution[category_column].astype(str), contribution["n_selected_genomes"]))
    order = ordered_methods(contribution[category_column].astype(str).tolist(), counts=category_counts)
    contribution[category_column] = contribution[category_column].astype(str)
    return contribution.set_index(category_column).reindex(order).reset_index()


def summarize_best_set_sample_category_contributions(source_frame, selected, member_table, category_column, sample_column):
    base = (
        source_frame[[sample_column, category_column]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values([sample_column, category_column])
        .reset_index(drop=True)
    )
    if base.empty:
        return base

    input_counts = (
        source_frame.groupby([sample_column, category_column])
        .size()
        .rename("n_input_genomes")
        .reset_index()
    )
    selected_counts = (
        selected.groupby([sample_column, category_column])
        .size()
        .rename("n_selected_genomes")
        .reset_index()
        if not selected.empty else pd.DataFrame(columns=[sample_column, category_column, "n_selected_genomes"])
    )
    component_presence = (
        member_table.groupby([sample_column, category_column])["component_id"]
        .nunique()
        .rename("n_components_present")
        .reset_index()
        if not member_table.empty else pd.DataFrame(columns=[sample_column, category_column, "n_components_present"])
    )
    component_wins = (
        selected.groupby([sample_column, category_column])["component_id"]
        .nunique()
        .rename("n_components_won")
        .reset_index()
        if not selected.empty else pd.DataFrame(columns=[sample_column, category_column, "n_components_won"])
    )

    summary = base.merge(input_counts, on=[sample_column, category_column], how="left")
    summary = summary.merge(selected_counts, on=[sample_column, category_column], how="left")
    summary = summary.merge(component_presence, on=[sample_column, category_column], how="left")
    summary = summary.merge(component_wins, on=[sample_column, category_column], how="left")
    for column in ["n_input_genomes", "n_selected_genomes", "n_components_present", "n_components_won"]:
        summary[column] = summary[column].fillna(0).astype(int)
    summary["selected_fraction_of_input"] = summary["n_selected_genomes"] / summary["n_input_genomes"].replace(0, pd.NA)
    summary["win_fraction_when_present"] = summary["n_components_won"] / summary["n_components_present"].replace(0, pd.NA)
    category_counts = (
        summary.groupby(category_column)["n_selected_genomes"].sum().to_dict()
        if not summary.empty else {}
    )
    method_order = ordered_methods(summary[category_column].astype(str).tolist(), counts=category_counts)
    order_map = {category: index for index, category in enumerate(method_order)}
    summary[category_column] = summary[category_column].astype(str)
    summary["__method_order"] = summary[category_column].map(order_map).fillna(len(order_map)).astype(int)
    summary = summary.sort_values(
        by=[sample_column, "__method_order", category_column],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop(columns=["__method_order"])
    return summary.reset_index(drop=True)


def summarize_best_set_selection_preferences(member_table, selected, category_column):
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
    if member_table is None or member_table.empty:
        return pd.DataFrame(columns=columns_by_category), pd.DataFrame(columns=pairwise_columns)

    present_df = member_table.loc[:, ["component_id", category_column]].dropna().copy()
    present_df[category_column] = present_df[category_column].astype(str).str.strip()
    present_df = present_df.loc[present_df[category_column].ne("")].drop_duplicates()
    if present_df.empty:
        return pd.DataFrame(columns=columns_by_category), pd.DataFrame(columns=pairwise_columns)

    categories = ordered_methods(present_df[category_column].tolist())
    component_categories = (
        present_df.groupby("component_id")[category_column]
        .agg(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    winner_df = (
        selected.loc[:, ["component_id", category_column]]
        .dropna()
        .copy()
        if selected is not None and not selected.empty else pd.DataFrame(columns=["component_id", category_column])
    )
    winner_df[category_column] = winner_df[category_column].astype(str).str.strip()
    winner_map = (
        winner_df.drop_duplicates(subset=["component_id"])
        .set_index("component_id")[category_column]
        .to_dict()
    )

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
    pairwise_pvalues = []
    for index_a, category_a in enumerate(categories):
        for category_b in categories[index_a + 1:]:
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
            pairwise_pvalues.append(pvalue)
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
                    "win_fraction_a_over_b": (
                        float(wins_a) / float(decisive) if decisive > 0 else float("nan")
                    ),
                    "preferred_category": preferred,
                    "pvalue_binom": pvalue,
                }
            )
    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        pairwise_df["qvalue_bh"] = bh_adjust(pairwise_pvalues)
        pairwise_df["significance"] = pairwise_df["qvalue_bh"].map(significance_stars)
        pairwise_df = pairwise_df.sort_values(
            by=["category_a", "category_b"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        pairwise_df = pd.DataFrame(columns=pairwise_columns)

    if not overall_df.empty:
        counts = dict(zip(overall_df[category_column].astype(str), overall_df["n_components_won_when_competing"]))
        order = ordered_methods(overall_df[category_column].astype(str).tolist(), counts=counts)
        overall_df[category_column] = overall_df[category_column].astype(str)
        overall_df = overall_df.set_index(category_column).reindex(order).reset_index()
    return overall_df, pairwise_df


def plot_best_set_selection_preferences(overall_df, pairwise_df, destination_dir, label, category_column, atlas_module):
    if overall_df is None or overall_df.empty:
        return []

    plt, sns = atlas_module.ensure_plotting()
    category_counts = dict(zip(overall_df[category_column].astype(str), overall_df["n_components_won_when_competing"]))
    order = ordered_methods(overall_df[category_column].astype(str).tolist(), counts=category_counts)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.0, 1.15]})

    bar_df = overall_df.copy()
    bar_df[category_column] = bar_df[category_column].astype(str)
    sns.barplot(
        data=bar_df,
        x=category_column,
        y="win_fraction_when_competing",
        order=order,
        color="#7f7f7f",
        edgecolor="black",
        linewidth=0.7,
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Win rate when categories compete")
    axes[0].set_xlabel(category_column)
    axes[0].set_ylabel("Selected fraction")
    axes[0].tick_params(axis="x", rotation=90)
    axes[0].axhline(0.5, color="#4d4d4d", linestyle="--", linewidth=0.8, alpha=0.7)
    for index, category in enumerate(order):
        row = bar_df.loc[bar_df[category_column].astype(str).eq(category)].head(1)
        if row.empty:
            continue
        wins = int(row["n_components_won_when_competing"].iloc[0])
        competing = int(row["n_components_competing"].iloc[0])
        fraction = pd.to_numeric(row["win_fraction_when_competing"], errors="coerce").iloc[0]
        if pd.isna(fraction):
            continue
        axes[0].text(
            index,
            float(fraction),
            f"{wins}/{competing}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    heatmap_matrix = pd.DataFrame(float("nan"), index=order, columns=order)
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
    sns.heatmap(
        heatmap_matrix,
        cmap="Greys",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="#f0f0f0",
        annot=annotation_matrix,
        fmt="",
        cbar_kws={"label": "Row category win fraction"},
        ax=axes[1],
    )
    axes[1].set_title("Pairwise preference when both categories are present")
    axes[1].set_xlabel("Compared against")
    axes[1].set_ylabel("Preferred row category")

    fig.suptitle("Category selection preference across competing components", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    written = []
    for extension in ("png", "pdf"):
        out_path = os.path.join(destination_dir, f"{label}.selection_preference.{extension}")
        fig.savefig(out_path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def build_best_set_excluded_hqmq_table(source_frame, member_table, category_column, sample_column):
    columns = [
        sample_column,
        category_column,
        "component_id",
        "component_member_count",
        "component_categories",
        "component_samples",
        "Genome_Id",
        "Bin Id",
        "mimag_tier",
        "qscore",
        "integrity_score",
        "recoverability_score",
        "Completeness",
        "Contamination",
        "winner_category",
        "winner_sample",
        "winner_Genome_Id",
        "winner_Bin_Id",
        "winner_mimag_tier",
        "winner_qscore",
        "winner_integrity_score",
        "winner_recoverability_score",
        "winner_Completeness",
        "winner_Contamination",
        "winner_same_category",
        "winner_same_sample",
        "winner_qscore_delta",
    ]
    base_columns = [column for column in columns if column in member_table.columns or column in {sample_column, category_column}]
    excluded = member_table.loc[~member_table["is_selected"].fillna(False)].copy()
    if excluded.empty:
        return pd.DataFrame(columns=base_columns)

    excluded = excluded.loc[
        excluded["mimag_tier"].astype(str).str.lower().isin(["medium", "high"])
    ].copy()
    if excluded.empty:
        return pd.DataFrame(columns=base_columns)

    available_columns = [column for column in columns if column in excluded.columns]
    excluded = excluded.loc[:, available_columns].copy()
    return excluded.sort_values(
        by=[sample_column, "component_id", category_column, "qscore"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )


def plot_best_set_category_contributions(category_df, sample_df, destination_dir, label, category_column, sample_column, atlas_module):
    if category_df is None or category_df.empty:
        return []

    plt, sns = atlas_module.ensure_plotting()
    counts = dict(zip(category_df[category_column].astype(str), category_df["n_selected_genomes"]))
    order = ordered_methods(category_df[category_column].astype(str).tolist(), counts=counts)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    count_df = category_df.loc[:, [category_column, "n_input_genomes", "n_selected_genomes"]].melt(
        id_vars=category_column,
        var_name="metric",
        value_name="count",
    )
    sns.barplot(
        data=count_df,
        x=category_column,
        y="count",
        hue="metric",
        order=order,
        palette=["#bdbdbd", "#1a1a1a"],
        ax=axes[0],
    )
    axes[0].set_title("Input vs selected genomes")
    axes[0].set_xlabel(category_column)
    axes[0].set_ylabel("Genome count")
    axes[0].tick_params(axis="x", rotation=90)

    component_df = category_df.loc[:, [category_column, "n_components_with_category_present", "n_components_won"]].melt(
        id_vars=category_column,
        var_name="metric",
        value_name="count",
    )
    sns.barplot(
        data=component_df,
        x=category_column,
        y="count",
        hue="metric",
        order=order,
        palette=["#bdbdbd", "#1a1a1a"],
        ax=axes[1],
    )
    axes[1].set_title("Components present vs won")
    axes[1].set_xlabel(category_column)
    axes[1].set_ylabel("Component count")
    axes[1].tick_params(axis="x", rotation=90)

    rate_df = category_df.loc[:, [category_column, "selected_fraction_of_input", "win_fraction_when_present"]].melt(
        id_vars=category_column,
        var_name="metric",
        value_name="fraction",
    )
    sns.barplot(
        data=rate_df,
        x=category_column,
        y="fraction",
        hue="metric",
        order=order,
        palette=["#7f7f7f", "#1a1a1a"],
        ax=axes[2],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Retention and win rates")
    axes[2].set_xlabel(category_column)
    axes[2].set_ylabel("Fraction")
    axes[2].tick_params(axis="x", rotation=90)

    if sample_df is not None and not sample_df.empty:
        heat_df = sample_df.pivot(index=sample_column, columns=category_column, values="n_components_won").fillna(0)
        heat_df = heat_df.reindex(columns=order)
        sns.heatmap(
            heat_df,
            cmap="Greys",
            cbar_kws={"label": "Components won"},
            linewidths=0.5,
            ax=axes[3],
        )
        axes[3].set_title("Component wins by sample and category")
        axes[3].set_xlabel(category_column)
        axes[3].set_ylabel(sample_column)
    else:
        axes[3].axis("off")
        axes[3].text(0.5, 0.5, "No sample/category contribution data", ha="center", va="center")

    fig.suptitle(f"{label.replace('_', ' ')} category contributions", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    written = []
    for extension in ("png", "pdf"):
        out_path = os.path.join(destination_dir, f"{label}.category_contributions.{extension}")
        fig.savefig(out_path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def plot_best_set_selected_category_counts(category_df, destination_dir, label, category_column, atlas_module):
    if category_df is None or category_df.empty:
        return []

    plt, sns = atlas_module.ensure_plotting()
    plot_df = category_df.copy()
    counts = dict(zip(plot_df[category_column].astype(str), plot_df["n_selected_genomes"]))
    order = ordered_methods(plot_df[category_column].astype(str).tolist(), counts=counts)
    metrics = [
        ("n_selected_genomes", "Total selected genomes"),
        ("n_selected_hq", "Selected HQ genomes"),
        ("n_selected_mq", "Selected MQ genomes"),
        ("n_selected_lq", "Selected LQ genomes"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in plot_df.columns:
            ax.axis("off")
            continue
        sns.barplot(
            data=plot_df,
            x=category_column,
            y=metric,
            order=order,
            color="#7f7f7f",
            edgecolor="black",
            linewidth=0.6,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(category_column)
        ax.set_ylabel("Genome count")
        ax.tick_params(axis="x", rotation=90)
        values = pd.to_numeric(plot_df.set_index(category_column).reindex(order)[metric], errors="coerce").fillna(0)
        for index, value in enumerate(values.tolist()):
            ax.text(index, float(value), f"{int(value)}", ha="center", va="bottom", fontsize=8)
    fig.suptitle(f"{label.replace('_', ' ')} selected genomes by category", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    written = []
    for extension in ("png", "pdf"):
        out_path = os.path.join(destination_dir, f"{label}.selected_category_count_summary.{extension}")
        fig.savefig(out_path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        written.append(out_path)
    plt.close(fig)
    return written


def copy_selected_fastas(selected, destination_dir, include_sample=False):
    fasta_dir = os.path.join(destination_dir, "fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    for existing_name in os.listdir(fasta_dir):
        existing_path = os.path.join(fasta_dir, existing_name)
        if os.path.isfile(existing_path) and existing_name.lower().endswith(
            (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
        ):
            os.unlink(existing_path)
    copied = []
    for row in selected.itertuples(index=False):
        source = os.path.abspath(os.path.expanduser(str(getattr(row, "fasta_path"))))
        if not os.path.exists(source):
            raise FileNotFoundError(f"Selected FASTA is missing: {source}")
        base_name = os.path.basename(source)
        name_parts = []
        if include_sample and hasattr(row, "sample"):
            name_parts.append(sanitize_token(getattr(row, "sample")))
        if hasattr(row, "category"):
            name_parts.append(sanitize_token(getattr(row, "category")))
        name_parts.append(base_name)
        target_name = "__".join([part for part in name_parts if part])
        destination = os.path.join(fasta_dir, target_name)
        shutil.copy2(source, destination)
        copied.append({"ani_record_id": getattr(row, "ani_record_id"), "copied_fasta_path": destination})
    return pd.DataFrame(copied)


def export_deduplicated_set(
    set_frame,
    pair_df,
    destination_dir,
    label,
    python_exe,
    atlas_script,
    atlas_prefix,
    group_column,
    top_n_groups,
    category_column,
    sample_column,
    atlas_module,
    ani_threshold,
    ani_af_threshold,
    include_sample_in_fasta_name=False,
):
    os.makedirs(destination_dir, exist_ok=True)
    audit_dir = os.path.join(destination_dir, "audit")
    plots_dir = os.path.join(destination_dir, "plots")
    selected_dir = os.path.join(destination_dir, "selected_set")
    for directory in (audit_dir, plots_dir, selected_dir):
        os.makedirs(directory, exist_ok=True)

    compare_df, pair_summary, _, _ = atlas_module.summarize_fastani_matches(
        set_frame,
        pair_df,
        category_column,
        ani_threshold,
        ani_af_threshold,
    )
    selected, member_table, component_summary = select_best_per_component(
        compare_df,
        pair_summary,
        atlas_module,
        category_column,
        sample_column,
    )
    summary = summarize_selected_set(selected, compare_df, sample_column, category_column)
    category_contrib = summarize_best_set_category_contributions(compare_df, selected, member_table, category_column)
    sample_category_contrib = summarize_best_set_sample_category_contributions(
        compare_df,
        selected,
        member_table,
        category_column,
        sample_column,
    )
    selection_pref_df, selection_pairwise_df = summarize_best_set_selection_preferences(
        member_table=member_table,
        selected=selected,
        category_column=category_column,
    )
    excluded_hqmq = build_best_set_excluded_hqmq_table(compare_df, member_table, category_column, sample_column)

    copied_fasta_df = copy_selected_fastas(
        selected,
        selected_dir,
        include_sample=include_sample_in_fasta_name,
    )
    if not copied_fasta_df.empty:
        selected = selected.merge(copied_fasta_df, on="ani_record_id", how="left")
        member_table = member_table.merge(copied_fasta_df, on="ani_record_id", how="left")

    input_candidates_path = os.path.join(audit_dir, "input_candidates.tsv")
    selected_audit_path = os.path.join(audit_dir, "selected_genomes.tsv")
    member_path = os.path.join(audit_dir, "component_members.tsv")
    components_path = os.path.join(audit_dir, "components.tsv")
    pair_summary_path = os.path.join(audit_dir, "pair_summary.tsv")
    summary_path = os.path.join(audit_dir, "set_summary.tsv")
    category_contrib_path = os.path.join(audit_dir, "category_contributions.tsv")
    sample_contrib_path = os.path.join(audit_dir, "sample_category_contributions.tsv")
    excluded_hqmq_path = os.path.join(audit_dir, "excluded_hqmq.tsv")
    selected_category_count_path = os.path.join(audit_dir, "selected_category_count_summary.tsv")
    selection_pref_path = os.path.join(audit_dir, "selection_preference_by_category.tsv")
    selection_pairwise_path = os.path.join(audit_dir, "selection_preference_pairwise.tsv")
    selected_master_path = os.path.join(selected_dir, "master.tsv")

    compare_df.to_csv(input_candidates_path, sep="	", index=False)
    selected.to_csv(selected_audit_path, sep="	", index=False)
    selected.to_csv(selected_master_path, sep="	", index=False)
    member_table.to_csv(member_path, sep="	", index=False)
    component_summary.to_csv(components_path, sep="	", index=False)
    pair_summary.to_csv(pair_summary_path, sep="	", index=False)
    summary.to_csv(summary_path, sep="	", index=False)
    category_contrib.to_csv(category_contrib_path, sep="	", index=False)
    sample_category_contrib.to_csv(sample_contrib_path, sep="	", index=False)
    excluded_hqmq.to_csv(excluded_hqmq_path, sep="	", index=False)
    selection_pref_df.to_csv(selection_pref_path, sep="	", index=False)
    selection_pairwise_df.to_csv(selection_pairwise_path, sep="	", index=False)
    category_contrib.loc[
        :,
        [column for column in [category_column, "n_selected_genomes", "n_selected_hq", "n_selected_mq", "n_selected_lq"] if column in category_contrib.columns]
    ].to_csv(selected_category_count_path, sep="	", index=False)
    contribution_plots = plot_best_set_category_contributions(
        category_contrib,
        sample_category_contrib,
        plots_dir,
        "candidate_vs_selected_summary",
        category_column,
        sample_column,
        atlas_module,
    )
    selected_count_plots = plot_best_set_selected_category_counts(
        category_contrib,
        plots_dir,
        "final_selected",
        category_column,
        atlas_module,
    )
    selection_pref_plots = plot_best_set_selection_preferences(
        overall_df=selection_pref_df,
        pairwise_df=selection_pairwise_df,
        destination_dir=plots_dir,
        label="selection_preference_summary",
        category_column=category_column,
        atlas_module=atlas_module,
    )

    atlas_output_dir = os.path.join(selected_dir, "genome_atlas")
    os.makedirs(atlas_output_dir, exist_ok=True)
    run_selected_set_atlas(
        python_exe=python_exe,
        atlas_script=atlas_script,
        master_path=selected_master_path,
        output_dir=atlas_output_dir,
        prefix=atlas_prefix,
        group_column=group_column,
        top_n_groups=top_n_groups,
        category_column=category_column,
        sample_column=sample_column,
    )
    return [
        destination_dir,
        audit_dir,
        plots_dir,
        selected_dir,
        input_candidates_path,
        selected_audit_path,
        member_path,
        components_path,
        pair_summary_path,
        summary_path,
        category_contrib_path,
        sample_contrib_path,
        excluded_hqmq_path,
        selected_category_count_path,
        selection_pref_path,
        selection_pairwise_path,
        selected_master_path,
        *contribution_plots,
        *selected_count_plots,
        *selection_pref_plots,
        atlas_output_dir,
    ]


def write_best_set_review_tables(review_entries, output_dir):
    ensure_pandas()
    if not review_entries:
        return []

    input_frames = []
    selected_frames = []
    members_frames = []
    components_frames = []
    summary_frames = []
    category_contrib_frames = []
    sample_contrib_frames = []
    excluded_frames = []
    for entry in review_entries:
        input_df = pd.read_csv(entry["input_candidates_path"], sep="	")
        input_df["best_set_scope"] = entry["best_set_scope"]
        input_df["best_set_name"] = entry["best_set_name"]
        input_df["best_set_dir"] = entry["best_set_dir"]
        input_frames.append(input_df)

        selected_df = pd.read_csv(entry["selected_path"], sep="	")
        selected_df["best_set_scope"] = entry["best_set_scope"]
        selected_df["best_set_name"] = entry["best_set_name"]
        selected_df["best_set_dir"] = entry["best_set_dir"]
        selected_frames.append(selected_df)

        members_df = pd.read_csv(entry["members_path"], sep="	")
        members_df["best_set_scope"] = entry["best_set_scope"]
        members_df["best_set_name"] = entry["best_set_name"]
        members_df["best_set_dir"] = entry["best_set_dir"]
        members_frames.append(members_df)

        components_df = pd.read_csv(entry["components_path"], sep="	")
        components_df["best_set_scope"] = entry["best_set_scope"]
        components_df["best_set_name"] = entry["best_set_name"]
        components_df["best_set_dir"] = entry["best_set_dir"]
        components_frames.append(components_df)

        summary_df = pd.read_csv(entry["summary_path"], sep="	")
        summary_df["best_set_scope"] = entry["best_set_scope"]
        summary_df["best_set_name"] = entry["best_set_name"]
        summary_df["best_set_dir"] = entry["best_set_dir"]
        summary_frames.append(summary_df)

        category_df = pd.read_csv(entry["category_contrib_path"], sep="	")
        category_df["best_set_scope"] = entry["best_set_scope"]
        category_df["best_set_name"] = entry["best_set_name"]
        category_df["best_set_dir"] = entry["best_set_dir"]
        category_contrib_frames.append(category_df)

        sample_df = pd.read_csv(entry["sample_contrib_path"], sep="	")
        sample_df["best_set_scope"] = entry["best_set_scope"]
        sample_df["best_set_name"] = entry["best_set_name"]
        sample_df["best_set_dir"] = entry["best_set_dir"]
        sample_contrib_frames.append(sample_df)

        excluded_df = pd.read_csv(entry["excluded_hqmq_path"], sep="	")
        excluded_df["best_set_scope"] = entry["best_set_scope"]
        excluded_df["best_set_name"] = entry["best_set_name"]
        excluded_df["best_set_dir"] = entry["best_set_dir"]
        excluded_frames.append(excluded_df)

    input_combined = pd.concat(input_frames, ignore_index=True)
    selected_combined = pd.concat(selected_frames, ignore_index=True)
    members_combined = pd.concat(members_frames, ignore_index=True)
    components_combined = pd.concat(components_frames, ignore_index=True)
    summary_combined = pd.concat(summary_frames, ignore_index=True)
    category_contrib_combined = pd.concat(category_contrib_frames, ignore_index=True)
    sample_contrib_combined = pd.concat(sample_contrib_frames, ignore_index=True)
    excluded_combined = pd.concat(excluded_frames, ignore_index=True)

    input_out = os.path.join(output_dir, "best_sets_review_input_candidates.tsv")
    selected_out = os.path.join(output_dir, "best_sets_review_selected_genomes.tsv")
    members_out = os.path.join(output_dir, "best_sets_review_component_members.tsv")
    components_out = os.path.join(output_dir, "best_sets_review_components.tsv")
    summary_out = os.path.join(output_dir, "best_sets_review_set_summaries.tsv")
    category_contrib_out = os.path.join(output_dir, "best_sets_review_category_contributions.tsv")
    sample_contrib_out = os.path.join(output_dir, "best_sets_review_sample_category_contributions.tsv")
    excluded_out = os.path.join(output_dir, "best_sets_review_excluded_hqmq.tsv")
    input_combined.to_csv(input_out, sep="	", index=False)
    selected_combined.to_csv(selected_out, sep="	", index=False)
    members_combined.to_csv(members_out, sep="	", index=False)
    components_combined.to_csv(components_out, sep="	", index=False)
    summary_combined.to_csv(summary_out, sep="	", index=False)
    category_contrib_combined.to_csv(category_contrib_out, sep="	", index=False)
    sample_contrib_combined.to_csv(sample_contrib_out, sep="	", index=False)
    excluded_combined.to_csv(excluded_out, sep="	", index=False)
    return [
        input_out,
        selected_out,
        members_out,
        components_out,
        summary_out,
        category_contrib_out,
        sample_contrib_out,
        excluded_out,
    ]


def export_best_sets(
    output_dir,
    combined_master_path,
    global_prefix,
    category_column,
    sample_column,
    python_exe,
    atlas_script,
    group_column,
    top_n_groups,
    ani_threshold,
    ani_af_threshold,
    ani_results,
    best_sample_dir_name,
    best_global_dir_name,
    best_set_min_mimag_tier,
):
    ensure_pandas()
    atlas_module = load_atlas_module()
    annotated_name = f"{global_prefix}_annotated.tsv"
    annotated_path = resolve_output_file(output_dir, annotated_name)
    if annotated_path is None or not os.path.exists(annotated_path):
        raise FileNotFoundError(f"Global annotated atlas table was not found: {annotated_path}")

    annotated = pd.read_csv(annotated_path, sep="\t")
    fastani_raw = resolve_fastani_raw_path(
        output_dir=output_dir,
        global_prefix=global_prefix,
        category_column=category_column,
        ani_results=ani_results,
    )

    fastani_frame = prepare_fastani_frame(annotated)
    if "mimag_tier" not in fastani_frame.columns:
        raise ValueError(
            "Annotated atlas table is missing mimag_tier; cannot rank best-set genomes by quality tier."
        )
    candidate_frame = fastani_frame.copy()
    tier_order = {"low": 0, "medium": 1, "high": 2}
    min_tier_value = tier_order[str(best_set_min_mimag_tier).strip().lower()]
    candidate_frame["_mimag_tier_norm"] = candidate_frame["mimag_tier"].astype(str).str.lower().str.strip()
    candidate_frame["_mimag_tier_value"] = candidate_frame["_mimag_tier_norm"].map(tier_order)
    candidate_frame = candidate_frame.loc[candidate_frame["_mimag_tier_value"].fillna(-1).ge(min_tier_value)].copy()
    candidate_frame = candidate_frame.drop(columns=["_mimag_tier_norm", "_mimag_tier_value"], errors="ignore")
    if candidate_frame.empty:
        raise ValueError(
            "No genomes available for best-set selection after applying "
            f"--best-set-min-mimag-tier {best_set_min_mimag_tier}."
        )
    tier_counts = (
        candidate_frame["mimag_tier"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )
    log_step(
        f"[info] best-set candidate pool: {len(candidate_frame)} genomes "
        f"(HQ={tier_counts.get('high', 0)}, MQ={tier_counts.get('medium', 0)}, LQ={tier_counts.get('low', 0)}; "
        f"min_tier={best_set_min_mimag_tier})"
    )
    pair_df = atlas_module.load_fastani_pairs(fastani_raw, fastani_frame)
    pair_df = filter_pair_df_to_frame_records(pair_df, candidate_frame)

    written_paths = []
    review_entries = []
    best_sample_root = os.path.join(output_dir, best_sample_dir_name)
    os.makedirs(best_sample_root, exist_ok=True)
    sample_values = sorted(candidate_frame[sample_column].astype(str).dropna().unique().tolist())
    for index, sample_value in enumerate(sample_values, start=1):
        sample_frame = candidate_frame.loc[candidate_frame[sample_column].astype(str) == sample_value].copy()
        if sample_frame.empty:
            continue
        sample_pair_df = filter_pair_df_to_frame_records(pair_df, sample_frame)
        sample_dir = os.path.join(best_sample_root, sanitize_token(sample_value))
        log_step(
            f"[start] ({index}/{len(sample_values)}) best-of-sample export for '{sample_value}' "
            f"({len(sample_frame)} genomes)"
        )
        export_paths = export_deduplicated_set(
            set_frame=sample_frame,
            pair_df=sample_pair_df,
            destination_dir=sample_dir,
            label="best_of_sample",
            python_exe=python_exe,
            atlas_script=atlas_script,
            atlas_prefix="genome_quality",
            group_column=group_column,
            top_n_groups=top_n_groups,
            category_column=category_column,
            sample_column=sample_column,
            atlas_module=atlas_module,
            ani_threshold=ani_threshold,
            ani_af_threshold=ani_af_threshold,
            include_sample_in_fasta_name=False,
        )
        written_paths.extend(export_paths)
        review_entries.append(
            {
                "best_set_scope": "sample",
                "best_set_name": str(sample_value),
                "best_set_dir": sample_dir,
                "input_candidates_path": os.path.join(sample_dir, "audit", "input_candidates.tsv"),
                "selected_path": os.path.join(sample_dir, "audit", "selected_genomes.tsv"),
                "members_path": os.path.join(sample_dir, "audit", "component_members.tsv"),
                "components_path": os.path.join(sample_dir, "audit", "components.tsv"),
                "summary_path": os.path.join(sample_dir, "audit", "set_summary.tsv"),
                "category_contrib_path": os.path.join(sample_dir, "audit", "category_contributions.tsv"),
                "sample_contrib_path": os.path.join(sample_dir, "audit", "sample_category_contributions.tsv"),
                "excluded_hqmq_path": os.path.join(sample_dir, "audit", "excluded_hqmq.tsv"),
            }
        )
        log_step(f"[done] ({index}/{len(sample_values)}) best-of-sample export: {sample_dir}")

    best_global_root = os.path.join(output_dir, best_global_dir_name)
    os.makedirs(best_global_root, exist_ok=True)
    log_step(f"[start] best-of-best export ({len(candidate_frame)} candidate genomes)")
    global_paths = export_deduplicated_set(
        set_frame=candidate_frame,
        pair_df=pair_df,
        destination_dir=best_global_root,
        label="best_of_best",
        python_exe=python_exe,
        atlas_script=atlas_script,
        atlas_prefix="genome_quality",
        group_column=group_column,
        top_n_groups=top_n_groups,
        category_column=category_column,
        sample_column=sample_column,
        atlas_module=atlas_module,
        ani_threshold=ani_threshold,
        ani_af_threshold=ani_af_threshold,
        include_sample_in_fasta_name=True,
    )
    written_paths.extend(global_paths)
    review_entries.append(
        {
            "best_set_scope": "global",
            "best_set_name": "best_of_best",
            "best_set_dir": best_global_root,
            "input_candidates_path": os.path.join(best_global_root, "audit", "input_candidates.tsv"),
            "selected_path": os.path.join(best_global_root, "audit", "selected_genomes.tsv"),
            "members_path": os.path.join(best_global_root, "audit", "component_members.tsv"),
            "components_path": os.path.join(best_global_root, "audit", "components.tsv"),
            "summary_path": os.path.join(best_global_root, "audit", "set_summary.tsv"),
            "category_contrib_path": os.path.join(best_global_root, "audit", "category_contributions.tsv"),
            "sample_contrib_path": os.path.join(best_global_root, "audit", "sample_category_contributions.tsv"),
            "excluded_hqmq_path": os.path.join(best_global_root, "audit", "excluded_hqmq.tsv"),
        }
    )
    log_step(f"[done] best-of-best export: {best_global_root}")
    wrote_review_tables = write_best_set_review_tables(review_entries, output_dir)
    written_paths.extend(wrote_review_tables)
    log_step("[done] wrote consolidated best-set review tables")
    return written_paths


def main():
    args = build_parser().parse_args()
    warnings.filterwarnings("ignore")
    ensure_pandas()
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_tsv))
    atlas_script = atlas_script_path(args.atlas_script)
    atlas_module = load_atlas_module()
    python_exe = sys.executable or shutil.which("python3") or shutil.which("python")
    if not python_exe:
        raise RuntimeError("Could not determine a Python executable for running genome_quality_atlas.py")

    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(manifest_path), "genome_atlas_batch")
    else:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    log_step(f"[info] atlas script: {atlas_script}")
    log_step(f"[info] global output dir: {output_dir}")
    organize_roots = {os.path.abspath(output_dir)}

    if args.skip_to_best:
        combined_master_path = resolve_output_file(output_dir, args.combined_master_name)
        if combined_master_path is None or not os.path.exists(combined_master_path):
            raise FileNotFoundError(
                "--skip-to-best requires an existing combined master: "
                f"{os.path.join(output_dir, args.combined_master_name)}"
            )
        log_step("[start] exporting best-of-sample and best-of-best genome sets")
        best_set_paths = export_best_sets(
            output_dir=output_dir,
            combined_master_path=combined_master_path,
            global_prefix=args.global_prefix,
            category_column=args.category_column,
            sample_column=args.sample_column,
            python_exe=python_exe,
            atlas_script=atlas_script,
            group_column=args.group_column,
            top_n_groups=args.top_n_groups,
            ani_threshold=args.ani_threshold,
            ani_af_threshold=args.ani_af_threshold,
            ani_results=args.ani_results,
            best_sample_dir_name=args.best_sample_dir_name,
            best_global_dir_name=args.best_global_dir_name,
            best_set_min_mimag_tier=args.best_set_min_mimag_tier,
        )
        if not args.skip_organize_outputs:
            log_step("[start] organizing outputs by type/style")
            organized_files, index_paths = organize_output_tree(output_dir)
            best_set_paths.extend([str(path) for path in index_paths])
            log_step(
                f"[done] organized outputs under {output_dir}: "
                f"{organized_files} files ({len(index_paths)} index tables)"
            )
        if args.run_best_set_phylogeny:
            phylogeny_paths = run_best_set_phylogeny(
                python_exe=python_exe,
                output_dir=output_dir,
                threads=args.best_set_phylogeny_threads,
                gtdbtk_data_path=args.best_set_gtdbtk_data_path,
            )
            best_set_paths.extend(phylogeny_paths)
        log_step("[done] exported best-of-sample and best-of-best genome sets")
        print("Wrote:")
        for path in [combined_master_path] + best_set_paths:
            print(path)
        return

    log_step(f"[start] reading manifest: {manifest_path}")
    manifest = read_manifest(manifest_path)
    log_step(f"[done] manifest rows: {len(manifest)}")

    combined_frames = []
    written_paths = []

    total_runs = len(manifest)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        sample = row.sample
        category = row.category
        run_dir = os.path.abspath(os.path.expanduser(row.dir_path))
        master_path = find_master_path(run_dir)

        log_step(
            f"[start] ({index}/{total_runs}) preparing sample '{sample}' category '{category}' from {run_dir}"
        )
        augmented, dedup_summary = augment_master(
            master_path,
            run_dir,
            sample,
            args.sample_column,
            category,
            args.category_column,
            row.bin_id_token_index,
            atlas_module,
        )
        individual_master_path = os.path.join(run_dir, args.individual_master_name)
        augmented.to_csv(individual_master_path, sep="\t", index=False)
        written_paths.append(individual_master_path)
        combined_frames.append(augmented)
        dedup_note = ""
        if dedup_summary["enabled"]:
            dedup_note = (
                f"; pre-ANI token dedup removed {dedup_summary['removed_rows']} "
                f"using {dedup_summary['source_column']} token {dedup_summary['token_index']}"
            )
        log_step(
            f"[done] ({index}/{total_runs}) augmented master: {individual_master_path} "
            f"({len(augmented)} genomes{dedup_note})"
        )

        individual_output_dir = os.path.join(run_dir, args.individual_output_name)
        os.makedirs(individual_output_dir, exist_ok=True)
        rep_fasta_paths = export_pre_ani_representative_fastas(augmented, individual_output_dir)
        if rep_fasta_paths:
            written_paths.extend(rep_fasta_paths)
            log_step(
                f"[done] ({index}/{total_runs}) exported pre-ANI representative FASTAs: "
                f"{os.path.join(individual_output_dir, 'rep_fastas')} ({len(rep_fasta_paths)} FASTAs)"
            )
        run_individual_atlas(
            python_exe=python_exe,
            atlas_script=atlas_script,
            master_path=individual_master_path,
            output_dir=individual_output_dir,
            prefix=args.individual_prefix,
            group_column=args.group_column,
            top_n_groups=args.top_n_groups,
        )
        written_paths.append(individual_output_dir)
        organize_roots.add(os.path.abspath(individual_output_dir))
        log_step(f"[done] ({index}/{total_runs}) outputs in: {individual_output_dir}")

    log_step("[start] building combined master")
    combined = pd.concat(combined_frames, ignore_index=True)
    combined_master_path = os.path.join(output_dir, args.combined_master_name)
    combined.to_csv(combined_master_path, sep="\t", index=False)
    written_paths.append(combined_master_path)
    log_step(f"[done] combined master: {combined_master_path} ({len(combined)} genomes)")

    manifest_copy_path = os.path.join(output_dir, "category_directory_manifest.tsv")
    manifest.to_csv(manifest_copy_path, sep="\t", index=False)
    written_paths.append(manifest_copy_path)
    log_step(f"[done] copied manifest: {manifest_copy_path}")

    run_global_atlas(
        python_exe=python_exe,
        atlas_script=atlas_script,
        master_path=combined_master_path,
        output_dir=output_dir,
        prefix=args.global_prefix,
        group_column=args.group_column,
        sample_column=args.sample_column,
        top_n_groups=args.top_n_groups,
        category_column=args.category_column,
        ani_threshold=args.ani_threshold,
        ani_af_threshold=args.ani_af_threshold,
        ani_threads=args.ani_threads,
        ani_top_overlaps=args.ani_top_overlaps,
        ani_results=args.ani_results,
        matched_samples_only=args.matched_samples_only,
    )
    written_paths.append(output_dir)
    log_step(f"[done] global outputs in: {output_dir}")

    family_paths, family_roots = run_family_comparison_atlases(
        combined=combined,
        output_dir=output_dir,
        combined_master_name=args.combined_master_name,
        python_exe=python_exe,
        atlas_script=atlas_script,
        prefix=args.global_prefix,
        group_column=args.group_column,
        sample_column=args.sample_column,
        top_n_groups=args.top_n_groups,
        category_column=args.category_column,
        ani_threshold=args.ani_threshold,
        ani_af_threshold=args.ani_af_threshold,
        ani_threads=args.ani_threads,
        ani_top_overlaps=args.ani_top_overlaps,
        ani_results=args.ani_results,
        matched_samples_only=args.matched_samples_only,
    )
    written_paths.extend(family_paths)
    organize_roots.update(family_roots)

    log_step("[start] exporting best-of-sample and best-of-best genome sets")
    best_set_paths = export_best_sets(
        output_dir=output_dir,
        combined_master_path=combined_master_path,
        global_prefix=args.global_prefix,
        category_column=args.category_column,
        sample_column=args.sample_column,
        python_exe=python_exe,
        atlas_script=atlas_script,
        group_column=args.group_column,
        top_n_groups=args.top_n_groups,
        ani_threshold=args.ani_threshold,
        ani_af_threshold=args.ani_af_threshold,
        ani_results=args.ani_results,
        best_sample_dir_name=args.best_sample_dir_name,
        best_global_dir_name=args.best_global_dir_name,
        best_set_min_mimag_tier=args.best_set_min_mimag_tier,
    )
    written_paths.extend(best_set_paths)
    log_step("[done] exported best-of-sample and best-of-best genome sets")

    if not args.skip_organize_outputs:
        log_step("[start] organizing outputs by type/style")
        organized_files_total = 0
        organized_index_paths = []
        for root_dir in sorted(organize_roots):
            organized_files, index_paths = organize_output_tree(root_dir)
            organized_files_total += organized_files
            organized_index_paths.extend(index_paths)
        written_paths.extend([str(path) for path in organized_index_paths])
        log_step(
            f"[done] organized outputs: {organized_files_total} files "
            f"({len(organized_index_paths)} index tables)"
        )

    if args.run_best_set_phylogeny:
        phylogeny_paths = run_best_set_phylogeny(
            python_exe=python_exe,
            output_dir=output_dir,
            threads=args.best_set_phylogeny_threads,
            gtdbtk_data_path=args.best_set_gtdbtk_data_path,
        )
        written_paths.extend(phylogeny_paths)

    print("Wrote:")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
