#!/usr/bin/env python3
"""Build SAG-xPG module KO heatmaps beside de novo phylogenies."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from best_set_phylogeny import parse_newick_tip_labels  # noqa: E402
from kofam_cross_scope_module_panels import infer_ko_short_label  # noqa: E402
from kofam_nitrogen_module_ko_facets import (  # noqa: E402
    CYCLE_LABELS,
    DEFAULT_RESULTS_DIR,
    ELEMENTAL_CYCLE_MODULE_IDS,
    KEYWORD_CYCLE_PATTERNS,
    add_quality_tier,
    attach_ko_labels,
    load_ko_definition_map,
    load_ko_definition_map_rg,
    load_selected_ko_presence,
    load_taxonomy_table,
    prepare_cycle_modules,
    read_tsv,
)
from kofam_sag_xpg_species_clustermaps import has_complete_species  # noqa: E402


DEFAULT_RENDER_ENV = "genome_qc_ete_render"
DEFAULT_CYCLE = "nitrogen"
DEFAULT_MODULE_ID = "M00529"
ELEMENTAL_REPRESENTATIVE_CYCLES = (
    "nitrogen",
    "sulfur",
    "carbon",
    "phosphorus",
    "iron",
    "trace_metals",
)
ELEMENTAL_OVERRIDE_THRESHOLD = 0.05
TAXONOMY_COLUMNS = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
MQ_BAC120_PHYLUM_PALETTE = {
    "Actinomycetota": "#4E79A7",
    "Bacteroidota": "#F28E2B",
    "Campylobacterota": "#59A14F",
    "Chloroflexota": "#E15759",
    "Cloacimonadota": "#76B7B2",
    "Elusimicrobiota": "#EDC948",
    "Margulisbacteria": "#B07AA1",
    "Marinisomatota": "#FF9DA7",
    "Nitrospinota": "#9C755F",
    "Patescibacteriota": "#BAB0AC",
    "Planctomycetota": "#2F4B7C",
    "Pseudomonadota": "#A05195",
    "SAR324": "#D45087",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cycle", default=DEFAULT_CYCLE)
    parser.add_argument("--module-id", default=DEFAULT_MODULE_ID)
    parser.add_argument(
        "--selection-mode",
        choices=["fixed-elemental", "module-maximized", "filtered-cohort"],
        default="fixed-elemental",
        help=(
            "Representative selection strategy. 'fixed-elemental' preserves quality tiers and uses balanced "
            "all-elemental coverage; 'module-maximized' reproduces the original module-KO-first selection; "
            "'filtered-cohort' retains every bacterial MQ+ SAG-xPG with 16S >0.5."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--render-env", default=DEFAULT_RENDER_ENV)
    parser.add_argument("--skip-tree", action="store_true", help="Do not run best_set_phylogeny.py.")
    parser.add_argument("--skip-render", action="store_true", help="Write inputs/config but do not render.")
    parser.add_argument("--gtdbtk-data-path", default=None)
    return parser.parse_args()


def safe_filename(value: object, max_length: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return (text or "module")[:max_length]


def safe_tree_id(value: object, max_length: int = 220) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return (text.strip("._") or "genome")[:max_length]


def tree_id_from_row(row: pd.Series) -> str:
    parts = [row.get("sample_id", ""), row.get("category", ""), row.get("original_bin_id", "")]
    return safe_tree_id("__".join(str(part).strip() for part in parts if str(part).strip()))


def ko_tree_label(ko: str, ko_label: str) -> str:
    short, _source = infer_ko_short_label(ko, ko_label)
    if short:
        return f"{short} ({ko})"
    label = str(ko_label).strip()
    if label:
        label = re.sub(r"\s+", " ", label)
        label = re.sub(r"\s*\((K\d+)\)$", r" (\1)", label)
        return label
    return str(ko)


def has_complete_genus(value: object) -> bool:
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "na", "n/a", "unknown", "unclassified", "uncultured"}:
        return False
    return text.casefold() not in {"g__", "genus"}


def load_sag_xpg_genome_metadata(results_dir: Path, taxonomy_level: str = "species") -> pd.DataFrame:
    metadata = read_tsv(results_dir / "metadata.tsv")
    for column in ["bin_id", "original_bin_id", "category", "sample_id"]:
        if column not in metadata.columns:
            raise ValueError(f"metadata.tsv is missing required column: {column}")
        metadata[column] = metadata[column].astype(str)
    metadata = metadata.loc[metadata["category"].eq("xPG_SAGs")].copy()
    taxonomy = load_tree_metadata_table(metadata, results_dir)
    metadata = metadata.merge(taxonomy, on=["category", "sample_id", "original_bin_id"], how="left")
    domain = metadata["Domain"].fillna("").astype(str).str.lower()
    metadata = metadata.loc[domain.str.contains("bacteria", na=False)].copy()
    metadata = add_quality_tier(metadata)
    metadata = metadata.loc[metadata["quality_tier"].isin(["high", "medium"])].copy()
    metadata["Genus"] = metadata["Genus"].fillna("").astype(str).str.strip()
    metadata["Species"] = metadata["Species"].fillna("").astype(str).str.strip()
    if taxonomy_level == "species":
        metadata = metadata.loc[metadata["Species"].map(has_complete_species)].copy()
    elif taxonomy_level == "genus":
        metadata = metadata.loc[metadata["Genus"].map(has_complete_genus)].copy()
    elif taxonomy_level == "none":
        pass
    else:
        raise ValueError(f"Unsupported taxonomy level: {taxonomy_level}")
    metadata["Genome_Id"] = metadata["original_bin_id"].astype(str)
    metadata["genome_id"] = metadata["Genome_Id"]
    metadata["sample"] = metadata["sample_id"].astype(str)
    metadata["tree_id"] = metadata.apply(tree_id_from_row, axis=1)
    return metadata.drop_duplicates("tree_id").reset_index(drop=True)


def load_tree_metadata_table(metadata: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    value_columns = [
        "qscore",
        "Completeness",
        "Contamination",
        "N50",
        "sum_len",
        "contains_16S",
        "16S_rRNA",
        "23S_rRNA",
        "5S_rRNA",
        "trna_total",
        "trna_unique",
        "gunc_assessment",
        "gunc_strict_assessment",
        "gunc_clade_separation_score",
        "gunc_reference_representation_score",
        "gunc_pass",
        "fasta_path",
    ]
    for (category, sample_id), _group in metadata.groupby(["category", "sample_id"], dropna=False):
        atlas_path = results_dir / str(category) / str(sample_id) / "Master_genome_QC.atlas.tsv"
        if not atlas_path.is_file():
            atlas_path = results_dir / str(category) / str(sample_id) / "Master_genome_QC.tsv"
        if not atlas_path.is_file():
            continue
        table = read_tsv(atlas_path)
        required = {"Genome_Id", "Bin Id"}
        if not required.issubset(table.columns):
            continue
        optional = (
            [column for column in taxonomy_columns if column in table.columns]
            + [column for column in value_columns if column in table.columns]
            + (["pre_ani_bin_key"] if "pre_ani_bin_key" in table.columns else [])
        )
        for record in table.loc[:, ["Genome_Id", "Bin Id", *optional]].to_dict("records"):
            candidate_ids = {
                str(record.get("Genome_Id", "")).strip(),
                str(record.get("Bin Id", "")).strip(),
                strip_bin_size_suffix(record.get("Bin Id", "")),
                str(record.get("pre_ani_bin_key", "")).strip(),
            }
            base = {
                "category": str(category),
                "sample_id": str(sample_id),
                **{column: str(record.get(column, "")).strip() for column in taxonomy_columns},
                **{column: record.get(column, "") for column in value_columns},
            }
            for original_bin_id in candidate_ids:
                if not original_bin_id:
                    continue
                rows.append({"original_bin_id": original_bin_id, **base})
    if not rows:
        return load_taxonomy_table(metadata, results_dir)
    return pd.DataFrame(rows).drop_duplicates(["category", "sample_id", "original_bin_id"])


def strip_bin_size_suffix(value: object) -> str:
    text = str(value).strip()
    if "." not in text:
        return text
    prefix, suffix = text.rsplit(".", 1)
    return prefix if suffix.isdigit() else text


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


def selected_module_data(cycle_modules: pd.DataFrame, module_id: str) -> pd.DataFrame:
    module_ids = cycle_modules["module_id"].astype(str)
    module_data = cycle_modules.loc[
        module_ids.eq(module_id) | module_ids.str.startswith(f"{module_id}.")
    ].copy()
    if module_data.empty:
        raise ValueError(f"No module rows matched {module_id}")
    return module_data.sort_values(["module_order", "ko_order", "ko"]).reset_index(drop=True)


def write_selected_genomes(metadata: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = metadata.copy()
    selected["sample"] = selected["sample_id"].astype(str)
    selected["mimag_tier"] = selected["quality_tier"].astype(str)
    preferred = [
        "tree_id",
        "Genome_Id",
        "genome_id",
        "bin_id",
        "original_bin_id",
        "sample",
        "category",
        "quality_tier",
        "mimag_tier",
        "representative_selection",
        "representative_selection_reason",
        "elemental_override_threshold",
        "elemental_score_gain_vs_quality_best",
        "elemental_module_completeness_mean",
        "elemental_module_total",
        "elemental_unique_ko_count",
        "elemental_unique_ko_total",
        "elemental_unique_ko_fraction",
        "elemental_cycles_scored",
        "module_ko_count",
        "module_ko_total",
        "Completeness",
        "Contamination",
        "qscore",
        "contains_16S",
        "16S_rRNA",
        "23S_rRNA",
        "5S_rRNA",
        "trna_total",
        "trna_unique",
        *TAXONOMY_COLUMNS,
        "gunc_assessment",
        "gunc_strict_assessment",
        "gunc_clade_separation_score",
        "gunc_reference_representation_score",
        "gunc_pass",
        "fasta_path",
    ]
    columns = [column for column in preferred if column in selected.columns]
    path = out_dir / "selected_genomes.tsv"
    selected.loc[:, columns].to_csv(path, sep="\t", index=False)
    return path


def module_presence_for_metadata(metadata: pd.DataFrame, module_data: pd.DataFrame, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ko_rows = (
        module_data.drop_duplicates("ko")
        .loc[:, ["ko", "ko_label", "ko_order"]]
        .sort_values("ko_order")
        .reset_index(drop=True)
    )
    selected_kos = set(ko_rows["ko"].astype(str))
    selected_bins = set(metadata["bin_id"].astype(str))
    presence = load_selected_ko_presence(results_dir / "all_kofam.mapper.unique.tsv", selected_bins, selected_kos)
    if presence.empty:
        presence = pd.DataFrame(columns=["bin_id", "ko", "present"])
    return presence, ko_rows


def load_balanced_elemental_module_definitions(results_dir: Path) -> pd.DataFrame:
    module_path = results_dir / "SI_kofam.ko_to_module.tsv"
    frames = []
    for cycle in ELEMENTAL_REPRESENTATIVE_CYCLES:
        if cycle not in ELEMENTAL_CYCLE_MODULE_IDS:
            continue
        cycle_modules, _summary, _names = prepare_cycle_modules(results_dir, module_path, cycle)
        cycle_modules = cycle_modules.copy()
        source_column = "source_module_id" if "source_module_id" in cycle_modules.columns else "module_id"
        cycle_modules["elemental_cycle"] = cycle
        cycle_modules["elemental_module_key"] = (
            cycle_modules["elemental_cycle"].astype(str)
            + "::"
            + cycle_modules[source_column].astype(str)
        )
        frames.append(cycle_modules.loc[:, ["elemental_cycle", "elemental_module_key", "ko"]])
    if not frames:
        raise ValueError("No elemental module definitions were available for representative selection")
    return pd.concat(frames, ignore_index=True).drop_duplicates(["elemental_module_key", "ko"])


def annotate_balanced_elemental_scores(metadata: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    definitions = load_balanced_elemental_module_definitions(results_dir)
    selected_bins = set(metadata["bin_id"].astype(str))
    selected_kos = set(definitions["ko"].astype(str))
    presence = load_selected_ko_presence(
        results_dir / "all_kofam.mapper.unique.tsv",
        selected_bins,
        selected_kos,
    )
    present_pairs = presence.loc[:, ["bin_id", "ko"]].drop_duplicates() if not presence.empty else pd.DataFrame(
        columns=["bin_id", "ko"]
    )
    module_sizes = definitions.groupby("elemental_module_key")["ko"].nunique().astype(float)
    module_keys = module_sizes.index.astype(str).tolist()
    bin_ids = metadata["bin_id"].astype(str).drop_duplicates().tolist()
    score_grid = pd.MultiIndex.from_product(
        [bin_ids, module_keys], names=["bin_id", "elemental_module_key"]
    ).to_frame(index=False)
    if present_pairs.empty:
        module_counts = pd.DataFrame(columns=["bin_id", "elemental_module_key", "present_kos"])
    else:
        module_counts = (
            present_pairs.merge(definitions.loc[:, ["elemental_module_key", "ko"]], on="ko", how="inner")
            .drop_duplicates(["bin_id", "elemental_module_key", "ko"])
            .groupby(["bin_id", "elemental_module_key"])["ko"]
            .nunique()
            .rename("present_kos")
            .reset_index()
        )
    score_grid = score_grid.merge(module_counts, on=["bin_id", "elemental_module_key"], how="left")
    score_grid["present_kos"] = score_grid["present_kos"].fillna(0).astype(int)
    score_grid["module_ko_total"] = score_grid["elemental_module_key"].map(module_sizes)
    score_grid["module_completeness"] = score_grid["present_kos"] / score_grid["module_ko_total"]
    balanced_scores = score_grid.groupby("bin_id")["module_completeness"].mean()
    unique_counts = present_pairs.groupby("bin_id")["ko"].nunique() if not present_pairs.empty else pd.Series(dtype=float)

    annotated = metadata.copy()
    annotated["elemental_module_completeness_mean"] = (
        annotated["bin_id"].astype(str).map(balanced_scores).fillna(0.0)
    )
    annotated["elemental_unique_ko_count"] = (
        annotated["bin_id"].astype(str).map(unique_counts).fillna(0).astype(int)
    )
    annotated["elemental_unique_ko_total"] = int(definitions["ko"].nunique())
    annotated["elemental_unique_ko_fraction"] = (
        annotated["elemental_unique_ko_count"] / max(1, int(definitions["ko"].nunique()))
    )
    annotated["elemental_module_total"] = int(definitions["elemental_module_key"].nunique())
    annotated["elemental_cycles_scored"] = ",".join(ELEMENTAL_REPRESENTATIVE_CYCLES)
    return annotated


def select_fixed_species_representatives(
    metadata: pd.DataFrame,
    results_dir: Path,
    elemental_override_threshold: float = ELEMENTAL_OVERRIDE_THRESHOLD,
) -> pd.DataFrame:
    """Select one fixed representative using tier, elemental breadth, then quality."""
    selected = annotate_balanced_elemental_scores(metadata, results_dir)
    selected["quality_rank"] = selected["quality_tier"].map({"high": 0, "medium": 1}).fillna(9).astype(int)
    selected["Completeness_numeric"] = pd.to_numeric(selected.get("Completeness", np.nan), errors="coerce")
    selected["Contamination_numeric"] = pd.to_numeric(selected.get("Contamination", np.nan), errors="coerce")
    selected["qscore_numeric"] = pd.to_numeric(selected.get("qscore", np.nan), errors="coerce")
    selected["contains_16S_rank"] = (
        selected.get("contains_16S", "").astype(str).str.lower().isin({"true", "1", "yes", "y"}).astype(int)
    )
    representatives = []
    for _species, species_group in selected.groupby("Species", sort=True):
        best_tier_rank = int(species_group["quality_rank"].min())
        tier_group = species_group.loc[species_group["quality_rank"].eq(best_tier_rank)].copy()
        quality_sorted = tier_group.sort_values(
            ["qscore_numeric", "Completeness_numeric", "Contamination_numeric", "contains_16S_rank", "tree_id"],
            ascending=[False, False, True, False, True],
            na_position="last",
        )
        quality_best = quality_sorted.iloc[0]
        functional_sorted = tier_group.sort_values(
            [
                "elemental_module_completeness_mean",
                "elemental_unique_ko_count",
                "qscore_numeric",
                "Completeness_numeric",
                "Contamination_numeric",
                "contains_16S_rank",
                "tree_id",
            ],
            ascending=[False, False, False, False, True, False, True],
            na_position="last",
        )
        functional_best = functional_sorted.iloc[0]
        score_gain = float(functional_best["elemental_module_completeness_mean"]) - float(
            quality_best["elemental_module_completeness_mean"]
        )
        use_functional = score_gain >= float(elemental_override_threshold)
        representative = (functional_best if use_functional else quality_best).copy()
        representative["elemental_score_gain_vs_quality_best"] = score_gain
        representative["representative_selection_reason"] = (
            "elemental_override_within_quality_tier" if use_functional else "quality_best_within_quality_tier"
        )
        representatives.append(representative)
    selected = pd.DataFrame(representatives)
    selected["elemental_override_threshold"] = float(elemental_override_threshold)
    selected["representative_selection"] = (
        "fixed_by_species:quality_tier_then_balanced_elemental_modules_if_gain>=0.05_then_quality"
    )
    return selected.sort_values(["Phylum", "Family", "Genus", "Species", "tree_id"]).reset_index(drop=True)


def annotate_module_ko_counts(
    metadata: pd.DataFrame,
    module_data: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    presence, ko_rows = module_presence_for_metadata(metadata, module_data, results_dir)
    module_ko_count = (
        presence.drop_duplicates(["bin_id", "ko"])
        .groupby("bin_id")["ko"]
        .size()
        .rename("module_ko_count")
    )
    selected = metadata.copy()
    selected["module_ko_count"] = selected["bin_id"].astype(str).map(module_ko_count).fillna(0).astype(int)
    selected["module_ko_total"] = int(ko_rows["ko"].nunique())
    return selected


def select_module_maximized_species_representatives(
    metadata: pd.DataFrame,
    module_data: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    """Reproduce the original module-KO-first representative selection."""
    selected = annotate_module_ko_counts(metadata, module_data, results_dir)
    selected["quality_rank"] = selected["quality_tier"].map({"high": 0, "medium": 1}).fillna(9).astype(int)
    selected["Completeness_numeric"] = pd.to_numeric(selected.get("Completeness", np.nan), errors="coerce")
    selected["Contamination_numeric"] = pd.to_numeric(selected.get("Contamination", np.nan), errors="coerce")
    selected["qscore_numeric"] = pd.to_numeric(selected.get("qscore", np.nan), errors="coerce")
    selected["contains_16S_rank"] = (
        selected.get("contains_16S", "").astype(str).str.lower().isin({"true", "1", "yes", "y"}).astype(int)
    )
    selected = selected.sort_values(
        [
            "Species",
            "module_ko_count",
            "quality_rank",
            "qscore_numeric",
            "Completeness_numeric",
            "Contamination_numeric",
            "contains_16S_rank",
            "tree_id",
        ],
        ascending=[True, False, True, False, False, True, False, True],
        na_position="last",
    )
    selected = selected.drop_duplicates("Species", keep="first").copy()
    selected["representative_selection"] = "module_maximized:module_ko_count_then_quality"
    selected["representative_selection_reason"] = "maximum_module_ko_count_then_quality"
    return selected.sort_values(["Phylum", "Family", "Genus", "Species", "tree_id"]).reset_index(drop=True)


def build_module_metadata(metadata: pd.DataFrame, module_data: pd.DataFrame, results_dir: Path, out_path: Path) -> Path:
    presence, ko_rows = module_presence_for_metadata(metadata, module_data, results_dir)
    present_pairs = set(zip(presence["bin_id"].astype(str), presence["ko"].astype(str)))

    table = metadata.copy()
    table["tree_id"] = table["tree_id"].astype(str)
    table["Completeness_percent"] = pd.to_numeric(table.get("Completeness", np.nan), errors="coerce")
    table["Contamination_percent"] = pd.to_numeric(table.get("Contamination", np.nan), errors="coerce")
    table["Qscore_value"] = pd.to_numeric(table.get("qscore", np.nan), errors="coerce")
    table["16S_percent"] = pd.to_numeric(table.get("16S_rRNA", np.nan), errors="coerce") * 100.0
    for row in ko_rows.itertuples(index=False):
        column = f"ko_{row.ko}"
        table[column] = [
            1 if (str(bin_id), str(row.ko)) in present_pairs else 0
            for bin_id in table["bin_id"].astype(str)
        ]
    keep_columns = [
        "tree_id",
        "Completeness_percent",
        "Contamination_percent",
        "Qscore_value",
        "16S_percent",
        "contains_16S",
        *[f"ko_{ko}" for ko in ko_rows["ko"].astype(str)],
    ]
    table.loc[:, keep_columns].to_csv(out_path, sep="\t", index=False)
    return out_path


def write_tree_metadata_files(metadata: pd.DataFrame, tree_path: Path, module_data: pd.DataFrame, results_dir: Path, prefix: str) -> list[Path]:
    tips = parse_newick_tip_labels(tree_path)
    if not tips:
        raise ValueError(f"No tips parsed from tree: {tree_path}")
    table = metadata.copy()
    table["tree_id"] = table["tree_id"].astype(str)
    table = table.drop_duplicates("tree_id").set_index("tree_id").reindex(tips).reset_index()
    table["cohort"] = tree_path.parents[1].name
    table["tree_domain"] = tree_path.stem.split("_tree", 1)[0]
    table["tree_file"] = tree_path.name

    specs = {
        "core": ["tree_id", "Genome_Id", "sample", "category", "cohort", "tree_domain", "tree_file"],
        "taxonomy": ["tree_id", *TAXONOMY_COLUMNS],
        "quality": [
            "tree_id",
            "mimag_tier",
            "Completeness",
            "Contamination",
            "qscore",
            "N50",
            "sum_len",
            "contains_16S",
            "16S_rRNA",
            "23S_rRNA",
            "5S_rRNA",
            "trna_total",
            "trna_unique",
            "module_ko_count",
            "module_ko_total",
        ],
        "gunc": [
            "tree_id",
            "gunc_assessment",
            "gunc_strict_assessment",
            "gunc_clade_separation_score",
            "gunc_reference_representation_score",
            "gunc_pass",
        ],
    }
    paths = []
    for name, columns in specs.items():
        keep = [column for column in columns if column in table.columns]
        path = tree_path.with_name(f"{tree_path.stem}_{name}_metadata.tsv")
        table.loc[:, keep].to_csv(path, sep="\t", index=False)
        paths.append(path)
    module_metadata_path = tree_path.with_name(f"{tree_path.stem}_{prefix}_module_metadata.tsv")
    build_module_metadata(table, module_data, results_dir, module_metadata_path)
    paths.append(module_metadata_path)
    return paths


def yaml_quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def write_render_config(
    tree_path: Path,
    metadata_paths: list[Path],
    module_data: pd.DataFrame,
    output_prefix: Path,
    config_path: Path,
) -> Path:
    ko_rows = (
        module_data.drop_duplicates("ko")
        .loc[:, ["ko", "ko_label", "ko_order"]]
        .sort_values("ko_order")
        .reset_index(drop=True)
    )
    heatmap_lines = [
        "- column: Completeness_percent\n"
        "  label: Completeness\n"
        "  show: true\n"
        "  width: 7\n"
        "  height: 9\n"
        "  min_value: 0\n"
        "  max_value: 100\n"
        "  value_scale: 1\n"
        "  lightest_shade: 255\n"
        "  darkest_shade: 20\n"
        "  reverse: false\n"
        "  gap_after: 1\n"
        "  label_margin_right: 14",
        "- column: Contamination_percent\n"
        "  label: Contamination\n"
        "  show: true\n"
        "  width: 7\n"
        "  height: 9\n"
        "  min_value: 0\n"
        "  max_value: 100\n"
        "  value_scale: 1\n"
        "  lightest_shade: 255\n"
        "  darkest_shade: 20\n"
        "  reverse: false\n"
        "  gap_after: 1\n"
        "  label_margin_right: 14",
        "- column: Qscore_value\n"
        "  label: Qscore\n"
        "  show: true\n"
        "  width: 7\n"
        "  height: 9\n"
        "  min_value: 0\n"
        "  max_value: 100\n"
        "  value_scale: 1\n"
        "  lightest_shade: 255\n"
        "  darkest_shade: 20\n"
        "  reverse: false\n"
        "  gap_after: 1\n"
        "  label_margin_right: 14",
        "- column: 16S_percent\n"
        "  label: 16S\n"
        "  show: true\n"
        "  width: 7\n"
        "  height: 9\n"
        "  min_value: 0\n"
        "  max_value: 100\n"
        "  value_scale: 1\n"
        "  lightest_shade: 255\n"
        "  darkest_shade: 20\n"
        "  reverse: false\n"
        "  gap_after: 6\n"
        "  divider_after: true\n"
        "  divider_color: '#4A4A4A'\n"
        "  divider_width: 1.5\n"
        "  divider_dash: '6,4'\n"
        "  label_margin_right: 14",
    ]
    for row in ko_rows.itertuples(index=False):
        label = ko_tree_label(str(row.ko), str(row.ko_label))
        heatmap_lines.append(
            f"- column: ko_{row.ko}\n"
            f"  label: {yaml_quote(label)}\n"
            "  show: true\n"
            "  width: 6\n"
            "  height: 9\n"
            "  min_value: 0\n"
            "  max_value: 1\n"
            "  value_scale: 1\n"
            "  lightest_shade: 255\n"
            "  darkest_shade: 20\n"
            "  reverse: false\n"
            "  gap_after: 1\n"
            "  label_margin_right: 13"
        )
    metadata_yaml = "\n".join(f"- {path}" for path in metadata_paths)
    heatmap_yaml = "\n".join(heatmap_lines)
    phylum_palette_yaml = "\n".join(
        f"    {phylum}: {yaml_quote(color)}" for phylum, color in MQ_BAC120_PHYLUM_PALETTE.items()
    )
    config_text = f"""tree: {tree_path}
metadata:
{metadata_yaml}
id_column: tree_id
output_prefix: {output_prefix}
output_formats:
- svg
- pdf
- png
newick_format: 0
rooting:
  method: midpoint
  outgroup_id: ''
layout: rectangular
aligned_order: taxonomy_before_heatmap
canvas:
  width_mm: 390
  dpi: 300
  margin_top: 30
  margin_right: 260
  margin_bottom: 95
  margin_left: 40
  show_scale: true
  branch_vertical_margin: 1
ladderize: true
show_leaf_names: false
leaf_name_column: tree_id
leaf_name_font_size: 7
branch_width: 2
guiding_lines:
  show: true
  type: 1
  color: '#000000'
  width: 0.35
  dasharray: '0.8,1.1'
spacing:
  after_leaf_name: 6
  before_strips: 2
  between_strips: 3
  between_strips_and_text: 7
  between_text_columns: 6
  between_heatmap_and_text: 8
support:
  show: true
  min_value: 0.9
  font_size: 6
  scale: auto
  position: branch-top
clade_brackets:
- column: Phylum
  label: Phylum
  show: false
  font_size: 8
  position: branch-right
  min_tips: 2
  margin_left: 8
  color: '#111111'
branch_colors:
- column: Phylum
  label: Phylum branches
  show: true
  palette:
{phylum_palette_yaml}
  mixed_color: '#777777'
  color_mixed: false
metadata_strips: []
heatmap_columns:
{heatmap_yaml}
quality_category_matrix:
  show: false
best_representatives:
  show: false
text_columns:
- column: Family
  label: Family
  font_size: 6
  max_length: null
  width: auto
  align: left
- column: Species
  label: Species
  font_size: 6
  max_length: null
  width: auto
  align: left
numeric_bars: []
legend:
  show: true
  max_items_per_strip: 20
  title_font_size: 8
  item_font_size: 6
  swatch_size: 8
  right_padding: 8
  swatch_text_gap: 2
  heatmap_label_font_size: 5
  show_heatmap_column_labels: true
  heatmap_column_label: Module KO presence
  heatmap_colorbar_label: 'Quality metrics (0-100)'
  heatmap_colorbar_lightest_shade: 255
  heatmap_colorbar_darkest_shade: 20
  show_heatmap_colorbar: true
  heatmap_colorbar_orientation: vertical
  heatmap_colorbar_vertical_height: 90
  heatmap_colorbar_vertical_bar_width: 10
  heatmap_colorbar_tick_font_size: 6
  heatmap_colorbar_ticks:
  - '0'
  - '50'
  - '100'
  heatmap_colorbar_tick_positions:
  - 0
  - 0.5
  - 1
"""
    config_path.write_text(config_text)
    return config_path


def run_command(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    cycle = args.cycle
    cycle_modules = load_cycle_modules_with_labels(results_dir, cycle)
    module_data = selected_module_data(cycle_modules, args.module_id)
    module_id = str(module_data["module_id"].iat[0])
    module_name = str(module_data["module_name"].iat[0])
    prefix = f"{safe_filename(module_id)}_{safe_filename(module_name)}"
    output_name = prefix
    if args.selection_mode == "module-maximized":
        output_name = f"{prefix}_module_maximized_representatives"
    elif args.selection_mode == "filtered-cohort":
        output_name = f"{prefix}_filtered_cohort_bacterial_mq_16s_gt0.5"
    output_dir = results_dir / "elemental_cycles" / cycle / "sag_xpg_species_phylogenies" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_level = "none" if args.selection_mode == "filtered-cohort" else "species"
    metadata_all = load_sag_xpg_genome_metadata(results_dir, taxonomy_level=taxonomy_level)
    if args.selection_mode == "module-maximized":
        metadata = select_module_maximized_species_representatives(metadata_all, module_data, results_dir)
    elif args.selection_mode == "filtered-cohort":
        rrna_16s = pd.to_numeric(metadata_all.get("16S_rRNA", np.nan), errors="coerce")
        metadata = metadata_all.loc[rrna_16s.gt(0.5)].copy()
        metadata = annotate_module_ko_counts(metadata, module_data, results_dir)
        metadata["representative_selection"] = "none:filtered_genome_cohort"
        metadata["representative_selection_reason"] = "bacterial_mq_plus_16s_gt0.5"
        metadata = metadata.sort_values(
            ["Phylum", "Family", "Genus", "Species", "sample_id", "tree_id"]
        ).reset_index(drop=True)
    else:
        metadata = select_fixed_species_representatives(metadata_all, results_dir)
        metadata = annotate_module_ko_counts(metadata, module_data, results_dir)
    selected_path = write_selected_genomes(metadata, output_dir)
    print(
        "[done] selected genomes: "
        f"{metadata.shape[0]} selected genomes from {metadata_all.shape[0]} taxonomy-eligible MQ+ genomes -> "
        f"{selected_path}"
    )

    if not args.skip_tree:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "best_set_phylogeny.py"),
            str(output_dir),
            "--threads",
            str(max(1, int(args.threads))),
        ]
        if args.gtdbtk_data_path:
            cmd.extend(["--gtdbtk-data-path", str(Path(args.gtdbtk_data_path).expanduser().resolve())])
        if args.selection_mode == "filtered-cohort":
            cmd.extend(["--taxonomy-level", "none"])
        run_command(cmd)

    gtdb_dir = output_dir / "phylogeny" / "gtdb_markers"
    tree_path = gtdb_dir / "bac120_tree.nwk"
    if not tree_path.exists():
        tree_path = gtdb_dir / "ar53_tree.nwk"
    if not tree_path.exists():
        raise FileNotFoundError(f"No GTDB marker tree was built under {gtdb_dir}")

    metadata_paths = write_tree_metadata_files(metadata, tree_path, module_data, results_dir, prefix)
    missing = [path for path in metadata_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing tree metadata: {missing}")
    config_path = tree_path.with_name(f"{tree_path.stem}_{prefix}_ete_config.yaml")
    output_prefix = tree_path.with_name(f"{tree_path.stem}_{prefix}_ete")
    write_render_config(tree_path, metadata_paths, module_data, output_prefix, config_path)
    print(f"[done] render config: {config_path}")

    if not args.skip_render:
        run_command(
            [
                str(Path.home() / "mambaforge" / "bin" / "conda"),
                "run",
                "-n",
                args.render_env,
                "python",
                str(REPO_ROOT / "render_phylogeny_ete.py"),
                "render",
                "-c",
                str(config_path),
            ]
        )
    print(f"[done] output prefix: {output_prefix}")


if __name__ == "__main__":
    main()
