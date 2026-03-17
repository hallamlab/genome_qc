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


def sanitize_token(value):
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = token.strip("._-")
    return token or "unknown"


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
        dirnames[:] = [name for name in dirnames if name not in {"plots", "tables", "_organized"}]
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


def compute_pre_ani_quality_proxy(frame):
    working = frame.copy()
    numeric_defaults = {
        "Completeness": 0.0,
        "Contamination": math.inf,
        "qscore": float("-inf"),
        "sum_len": 0.0,
        "16S_rRNA": 0.0,
        "23S_rRNA": 0.0,
        "5S_rRNA": 0.0,
        "trna_unique": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(default)
        else:
            working[column] = default

    working["pre_rrna_16S_score"] = working["16S_rRNA"].clip(lower=0, upper=1)
    working["pre_rrna_23S_score"] = working["23S_rRNA"].clip(lower=0, upper=1)
    working["pre_rrna_5S_score"] = working["5S_rRNA"].clip(lower=0, upper=1)
    working["pre_trna_ge_18"] = (working["trna_unique"] >= 18).astype(int)
    working["pre_mC"] = ((working["Completeness"] - 50.0) / 40.0).clip(lower=0, upper=1)
    working["pre_mK"] = ((10.0 - working["Contamination"]) / 5.0).clip(lower=0, upper=1)
    working["pre_integrity_score"] = working[["pre_mC", "pre_mK"]].min(axis=1)
    working["pre_recoverability_score"] = working[
        ["pre_rrna_16S_score", "pre_rrna_23S_score", "pre_rrna_5S_score", "pre_trna_ge_18"]
    ].mean(axis=1)
    working["pre_mimag_proxy"] = (working["pre_integrity_score"] * working["pre_recoverability_score"]).pow(0.5)
    working["pre_feature_count"] = working[
        ["pre_rrna_16S_score", "pre_rrna_23S_score", "pre_rrna_5S_score", "pre_trna_ge_18"]
    ].sum(axis=1)
    working["pre_mimag_tier_rank"] = 2
    medium_mask = (working["Completeness"] >= 50.0) & (working["Contamination"] < 10.0)
    high_mask = (
        (working["Completeness"] > 90.0)
        & (working["Contamination"] < 5.0)
        & (working["pre_rrna_16S_score"] >= 1.0)
        & (working["pre_rrna_23S_score"] >= 1.0)
        & (working["pre_rrna_5S_score"] >= 1.0)
        & (working["pre_trna_ge_18"] >= 1)
    )
    working.loc[medium_mask, "pre_mimag_tier_rank"] = 1
    working.loc[high_mask, "pre_mimag_tier_rank"] = 0
    return working


def pre_deduplicate_master(master, token_index):
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

    working = compute_pre_ani_quality_proxy(working)
    sort_specs = [
        ("pre_mimag_tier_rank", True),
        ("pre_mimag_proxy", False),
        ("qscore", False),
        ("Completeness", False),
        ("Contamination", True),
        ("pre_feature_count", False),
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
    selected["pre_ani_best_selection_metric"] = "mimag_proxy>qscore>completeness>contamination"

    drop_columns = [
        "pre_rrna_16S_score",
        "pre_rrna_23S_score",
        "pre_rrna_5S_score",
        "pre_trna_ge_18",
        "pre_mC",
        "pre_mK",
        "pre_integrity_score",
        "pre_recoverability_score",
        "pre_mimag_proxy",
        "pre_feature_count",
        "pre_mimag_tier_rank",
    ]
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


def augment_master(master_path, run_dir, sample, sample_column, category, category_column, bin_id_token_index=None):
    ensure_pandas()
    master = pd.read_csv(master_path, sep="\t")
    master, dedup_summary = pre_deduplicate_master(master, bin_id_token_index)
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


def select_best_per_component(frame, pair_summary, atlas_module, category_column, sample_column):
    working = frame.copy()
    working["ani_record_id"] = working["ani_record_id"].astype(str)
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
    tier_rank = {"high": 0, "medium": 1, "low": 2}
    working["__mimag_rank"] = (
        working["mimag_tier"]
        .astype(str)
        .str.lower()
        .map(tier_rank)
        .fillna(3)
        .astype(int)
    )
    sort_specs = [
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
        working.groupby("component_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected["component_member_count"] = selected["component_id"].map(component_member_count)
    selected["component_categories"] = selected["component_id"].map(component_categories)
    selected["component_samples"] = selected["component_id"].map(component_samples)
    selected["best_selection_metric"] = "mimag_tier>integrity>recoverability>qscore"

    member_table = working.copy()
    selected_records = set(selected["ani_record_id"].astype(str).tolist())
    member_table["component_member_count"] = member_table["component_id"].map(component_member_count)
    member_table["component_categories"] = member_table["component_id"].map(component_categories)
    member_table["component_samples"] = member_table["component_id"].map(component_samples)
    member_table["is_selected"] = member_table["ani_record_id"].astype(str).isin(selected_records)
    member_table = member_table.drop(columns=["__mimag_rank"], errors="ignore")
    selected = selected.drop(columns=["__mimag_rank"], errors="ignore")

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


def copy_selected_fastas(selected, destination_dir, include_sample=False):
    fasta_dir = os.path.join(destination_dir, "fasta")
    os.makedirs(fasta_dir, exist_ok=True)
    copied = []
    used_names = set()
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
        stem, ext = os.path.splitext(target_name)
        counter = 1
        while target_name in used_names:
            target_name = f"{stem}_{counter}{ext}"
            counter += 1
        used_names.add(target_name)
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
    copied_fasta_df = copy_selected_fastas(
        selected,
        destination_dir,
        include_sample=include_sample_in_fasta_name,
    )
    if not copied_fasta_df.empty:
        selected = selected.merge(copied_fasta_df, on="ani_record_id", how="left")
        member_table = member_table.merge(copied_fasta_df, on="ani_record_id", how="left")

    selected_master_path = os.path.join(destination_dir, f"{label}.selected_genomes.tsv")
    selected.to_csv(selected_master_path, sep="\t", index=False)
    member_table.to_csv(os.path.join(destination_dir, f"{label}.component_members.tsv"), sep="\t", index=False)
    component_summary.to_csv(os.path.join(destination_dir, f"{label}.components.tsv"), sep="\t", index=False)
    pair_summary.to_csv(os.path.join(destination_dir, f"{label}.pair_summary.tsv"), sep="\t", index=False)
    summary.to_csv(os.path.join(destination_dir, f"{label}.summary.tsv"), sep="\t", index=False)

    atlas_output_dir = os.path.join(destination_dir, "genome_atlas")
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
        selected_master_path,
        os.path.join(destination_dir, f"{label}.component_members.tsv"),
        os.path.join(destination_dir, f"{label}.components.tsv"),
        os.path.join(destination_dir, f"{label}.pair_summary.tsv"),
        os.path.join(destination_dir, f"{label}.summary.tsv"),
        atlas_output_dir,
    ]


def write_best_set_review_tables(review_entries, output_dir):
    ensure_pandas()
    if not review_entries:
        return []

    selected_frames = []
    members_frames = []
    summary_frames = []
    for entry in review_entries:
        selected_df = pd.read_csv(entry["selected_path"], sep="\t")
        selected_df["best_set_scope"] = entry["best_set_scope"]
        selected_df["best_set_name"] = entry["best_set_name"]
        selected_df["best_set_dir"] = entry["best_set_dir"]
        selected_frames.append(selected_df)

        members_df = pd.read_csv(entry["members_path"], sep="\t")
        members_df["best_set_scope"] = entry["best_set_scope"]
        members_df["best_set_name"] = entry["best_set_name"]
        members_df["best_set_dir"] = entry["best_set_dir"]
        members_frames.append(members_df)

        summary_df = pd.read_csv(entry["summary_path"], sep="\t")
        summary_df["best_set_scope"] = entry["best_set_scope"]
        summary_df["best_set_name"] = entry["best_set_name"]
        summary_df["best_set_dir"] = entry["best_set_dir"]
        summary_frames.append(summary_df)

    selected_combined = pd.concat(selected_frames, ignore_index=True)
    members_combined = pd.concat(members_frames, ignore_index=True)
    summary_combined = pd.concat(summary_frames, ignore_index=True)

    selected_out = os.path.join(output_dir, "best_sets_review_selected_genomes.tsv")
    members_out = os.path.join(output_dir, "best_sets_review_component_members.tsv")
    summary_out = os.path.join(output_dir, "best_sets_review_set_summaries.tsv")
    selected_combined.to_csv(selected_out, sep="\t", index=False)
    members_combined.to_csv(members_out, sep="\t", index=False)
    summary_combined.to_csv(summary_out, sep="\t", index=False)
    return [selected_out, members_out, summary_out]


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
            "Annotated atlas table is missing mimag_tier; cannot enforce MQ/HQ-only best-set selection."
        )
    candidate_frame = fastani_frame.loc[
        fastani_frame["mimag_tier"].astype(str).str.lower().isin(["medium", "high"])
    ].copy()
    if candidate_frame.empty:
        raise ValueError("No MQ/HQ genomes available for best-set selection after excluding LQ genomes.")
    excluded_lq = len(fastani_frame) - len(candidate_frame)
    log_step(
        f"[info] best-set candidate pool: {len(candidate_frame)} MQ/HQ genomes "
        f"(excluded {excluded_lq} LQ genomes)"
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
                "selected_path": os.path.join(sample_dir, "best_of_sample.selected_genomes.tsv"),
                "members_path": os.path.join(sample_dir, "best_of_sample.component_members.tsv"),
                "summary_path": os.path.join(sample_dir, "best_of_sample.summary.tsv"),
            }
        )
        log_step(f"[done] ({index}/{len(sample_values)}) best-of-sample export: {sample_dir}")

    best_global_root = os.path.join(output_dir, best_global_dir_name)
    os.makedirs(best_global_root, exist_ok=True)
    log_step(f"[start] best-of-best export ({len(candidate_frame)} MQ/HQ genomes)")
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
            "selected_path": os.path.join(best_global_root, "best_of_best.selected_genomes.tsv"),
            "members_path": os.path.join(best_global_root, "best_of_best.component_members.tsv"),
            "summary_path": os.path.join(best_global_root, "best_of_best.summary.tsv"),
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
        )
        if not args.skip_organize_outputs:
            log_step("[start] organizing outputs by type/style")
            organized_files, index_paths = organize_output_tree(output_dir)
            best_set_paths.extend([str(path) for path in index_paths])
            log_step(
                f"[done] organized outputs under {output_dir}: "
                f"{organized_files} files ({len(index_paths)} index tables)"
            )
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

    print("Wrote:")
    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
