#!/usr/bin/env python3
"""Collapse a SAG-xPG module tree into exact species-taxonomy prevalence groups."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kofam_nitrogen_module_ko_facets import load_selected_ko_presence  # noqa: E402
from kofam_sag_xpg_module_phylogeny import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    TAXONOMY_COLUMNS,
    has_complete_species,
    load_cycle_modules_with_labels,
    selected_module_data,
    write_render_config,
)

DEFAULT_SOURCE_NAME = "M00529_Denitrification_nitrate_nitrogen_filtered_cohort_bacterial_mq_16s_gt0.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--cycle", default="nitrogen")
    parser.add_argument("--module-id", default="M00529")
    parser.add_argument("--source-name", default=DEFAULT_SOURCE_NAME)
    parser.add_argument("--render-env", default="genome_qc_ete_render")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def exact_taxonomy_groups(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = selected.loc[selected["Species"].map(has_complete_species)].copy()
    kept["taxonomy_key"] = kept[TAXONOMY_COLUMNS].fillna("").astype(str).agg(" | ".join, axis=1)
    keys = kept.loc[:, ["taxonomy_key", *TAXONOMY_COLUMNS]].drop_duplicates().sort_values(TAXONOMY_COLUMNS)
    keys = keys.reset_index(drop=True)
    keys["tree_id"] = [f"taxon_group_{index:03d}" for index in range(1, len(keys) + 1)]
    kept = kept.merge(keys.loc[:, ["taxonomy_key", "tree_id"]], on="taxonomy_key", how="left")
    return kept, keys


def build_group_distance_tree(source_tree: Path, members: pd.DataFrame, output_tree: Path) -> None:
    tree = Phylo.read(source_tree, "newick")
    tips = {tip.name: tip for tip in tree.get_terminals()}
    grouped = {
        group_id: [tips[name] for name in group["source_tree_id"] if name in tips]
        for group_id, group in members.groupby("tree_id", sort=True)
    }
    names = sorted(grouped)
    matrix = []
    for row_index, left_name in enumerate(names):
        row = []
        for right_name in names[: row_index + 1]:
            if left_name == right_name:
                row.append(0.0)
                continue
            distances = [
                tree.distance(left_tip, right_tip)
                for left_tip in grouped[left_name]
                for right_tip in grouped[right_name]
            ]
            row.append(float(np.mean(distances)))
        matrix.append(row)
    group_tree = DistanceTreeConstructor().nj(DistanceMatrix(names, matrix))
    output_tree.parent.mkdir(parents=True, exist_ok=True)
    Phylo.write(group_tree, output_tree, "newick")


def build_group_metadata(
    members: pd.DataFrame,
    taxonomy: pd.DataFrame,
    module_data: pd.DataFrame,
    results_dir: Path,
) -> pd.DataFrame:
    selected_bins = set(members["bin_id"].astype(str))
    selected_kos = set(module_data["ko"].astype(str))
    presence = load_selected_ko_presence(
        results_dir / "all_kofam.mapper.unique.tsv",
        selected_bins,
        selected_kos,
    )
    bin_to_group = members.set_index(members["bin_id"].astype(str))["tree_id"].to_dict()
    presence["tree_id"] = presence["bin_id"].astype(str).map(bin_to_group)
    presence_counts = presence.groupby(["tree_id", "ko"])["bin_id"].nunique()

    grouped = members.groupby("tree_id", sort=True)
    table = taxonomy.copy()
    table["n_genomes"] = table["tree_id"].map(grouped.size()).astype(int)
    table["Completeness_percent"] = table["tree_id"].map(grouped["Completeness"].median())
    table["Contamination_percent"] = table["tree_id"].map(grouped["Contamination"].median())
    table["Qscore_value"] = table["tree_id"].map(grouped["qscore"].median())
    table["16S_percent"] = table["tree_id"].map(grouped["16S_rRNA"].median()) * 100.0
    for ko in module_data.drop_duplicates("ko")["ko"].astype(str):
        table[f"ko_{ko}"] = [
            100.0 * float(presence_counts.get((group_id, ko), 0)) / float(n_genomes)
            for group_id, n_genomes in zip(table["tree_id"], table["n_genomes"])
        ]
    return table


def adapt_config_for_group_prevalence(config_path: Path, max_group_size: int) -> None:
    text = config_path.read_text()
    rounded_max_group_size = max(10, int(np.ceil(max_group_size / 10.0) * 10))
    value_columns = [
        "Completeness_percent",
        "Contamination_percent",
        "Qscore_value",
        "16S_percent",
    ]
    value_columns.extend(f"ko_{ko}" for ko in re.findall(r"^- column: ko_(K\d+)$", text, flags=re.MULTILINE))
    for column in value_columns:
        pattern = rf"(- column: {re.escape(column)}\n(?:.*?\n)*?  height:) 9\n"
        text = re.sub(
            pattern,
            rf"\g<1> 10\n"
            "  style: dot\n"
            "  color: '#FFFFFF'\n"
            "  border_color: '#111111'\n"
            "  min_diameter: 0\n"
            "  max_diameter: 9\n"
            "  border_width: 0.8\n"
            "  fill: true\n",
            text,
            count=1,
        )
    for ko in re.findall(r"^- column: ko_(K\d+)$", text, flags=re.MULTILINE):
        pattern = rf"(- column: ko_{ko}\n(?:.*\n)*?  max_value:) 1\n"
        text = re.sub(pattern, rf"\g<1> 100\n", text, count=1)
    text = text.replace("aligned_order: taxonomy_before_heatmap", "aligned_order: heatmap_before_taxonomy")
    text = text.replace(
        "numeric_bars: []",
        "numeric_bars: []\n"
        "numeric_dots:\n"
        "- column: n_genomes\n"
        "  label: Genomes per taxon\n"
        "  min_value: 0\n"
        f"  max_value: {max_group_size}\n"
        "  width: 12\n"
        "  height: 12\n"
        "  min_diameter: 3\n"
        "  max_diameter: 11\n"
        "  color: '#111111'\n"
        "  border_color: '#111111'\n"
        "  border_width: 0.8\n"
        "  fill: true\n"
        "  gap_after: 6\n"
        f"  legend_values: [1, 5, 25, {rounded_max_group_size}]",
    )
    text = text.replace(
        "heatmap_column_label: Module KO presence",
        "heatmap_column_label: Quality metrics / KO prevalence",
    )
    text = text.replace(
        "  show_heatmap_colorbar: true",
        "  show_heatmap_dot_legend: true\n"
        "  heatmap_dot_legend_label: Quality / prevalence (0-100)\n"
        "  heatmap_dot_legend_values: [10, 25, 50, 100]\n"
        "  heatmap_dot_legend_highlight_value: 100\n"
        "  heatmap_dot_legend_highlight_color: '#BDBDBD'\n"
        "  show_numeric_dot_legend: true\n"
        "  show_heatmap_colorbar: false",
    )
    config_path.write_text(text)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    source_dir = (
        results_dir
        / "elemental_cycles"
        / args.cycle
        / "sag_xpg_species_phylogenies"
        / args.source_name
    )
    selected = pd.read_csv(source_dir / "selected_genomes.tsv", sep="\t")
    selected = selected.rename(columns={"tree_id": "source_tree_id"})
    members, taxonomy = exact_taxonomy_groups(selected)
    module_data = selected_module_data(load_cycle_modules_with_labels(results_dir, args.cycle), args.module_id)

    output_dir = source_dir / "taxonomy_aggregated_species"
    source_tree = source_dir / "phylogeny" / "gtdb_markers" / "bac120_tree.nwk"
    group_tree = output_dir / "exact_species_taxonomy_group_tree.nwk"
    build_group_distance_tree(source_tree, members, group_tree)
    metadata = build_group_metadata(members, taxonomy, module_data, results_dir)

    members.to_csv(output_dir / "exact_species_taxonomy_group_members.tsv", sep="\t", index=False)
    metadata_path = output_dir / "exact_species_taxonomy_group_metadata.tsv"
    metadata.to_csv(metadata_path, sep="\t", index=False)
    summary = metadata.loc[:, ["tree_id", *TAXONOMY_COLUMNS, "n_genomes", *[f"ko_{ko}" for ko in module_data["ko"].drop_duplicates()]]]
    summary.to_csv(output_dir / "exact_species_taxonomy_ko_prevalence.tsv", sep="\t", index=False)

    config = output_dir / "exact_species_taxonomy_denitrification_ete_config.yaml"
    output_prefix = output_dir / "exact_species_taxonomy_denitrification_ete"
    write_render_config(group_tree, [metadata_path], module_data, output_prefix, config)
    adapt_config_for_group_prevalence(config, int(metadata["n_genomes"].max()))
    if not args.skip_render:
        subprocess.run(
            [
                str(Path.home() / "mambaforge" / "bin" / "conda"),
                "run",
                "-n",
                args.render_env,
                "python",
                str(REPO_ROOT / "render_phylogeny_ete.py"),
                "render",
                "-c",
                str(config),
            ],
            check=True,
        )
    print(f"[done] species-resolved genomes: {len(members)}")
    print(f"[done] exact taxonomy groups: {len(metadata)}")
    print(f"[done] output prefix: {output_prefix}")


if __name__ == "__main__":
    main()
