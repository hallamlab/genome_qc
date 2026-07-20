#!/usr/bin/env python3
"""Render every elemental module for individual and species-aggregated SAG-xPGs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kofam_nitrogen_module_ko_facets import load_selected_ko_presence  # noqa: E402
from kofam_sag_xpg_module_phylogeny import (  # noqa: E402
    DEFAULT_RENDER_ENV,
    DEFAULT_RESULTS_DIR,
    ELEMENTAL_REPRESENTATIVE_CYCLES,
    load_cycle_modules_with_labels,
    safe_filename,
    write_render_config,
)
from kofam_sag_xpg_taxonomy_aggregated_phylogeny import (  # noqa: E402
    DEFAULT_SOURCE_NAME,
    adapt_config_for_group_prevalence,
    build_group_distance_tree,
    exact_taxonomy_groups,
)

VARIANTS = ("individual", "species-aggregated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--render-env", default=DEFAULT_RENDER_ENV)
    parser.add_argument("--cycles", nargs="+", choices=ELEMENTAL_REPRESENTATIVE_CYCLES)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--module-id", help="Render one module ID, including all split parts.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def cohort_paths(results_dir: Path) -> tuple[Path, Path, list[Path]]:
    base = (
        results_dir
        / "elemental_cycles"
        / "nitrogen"
        / "sag_xpg_species_phylogenies"
        / DEFAULT_SOURCE_NAME
    )
    selected_path = base / "selected_genomes.tsv"
    tree = base / "phylogeny" / "gtdb_markers" / "bac120_tree.nwk"
    metadata_paths = [tree.with_name(f"{tree.stem}_{suffix}_metadata.tsv") for suffix in ["core", "taxonomy", "quality", "gunc"]]
    missing = [path for path in [selected_path, tree] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing 152-genome cohort inputs: {missing}")
    return selected_path, tree, [path for path in metadata_paths if path.is_file()]


def all_elemental_modules(results_dir: Path, cycles: list[str] | tuple[str, ...]) -> pd.DataFrame:
    frames = []
    for cycle in cycles:
        plotdata_path = (
            results_dir
            / "elemental_cycles"
            / cycle
            / "all"
            / f"kofam_{cycle}_modules.ko_delta_plotdata.tsv"
        )
        if plotdata_path.is_file():
            frame = pd.read_csv(plotdata_path, sep="\t")
            columns = [
                column
                for column in [
                    "module_id",
                    "module_name",
                    "source_module_id",
                    "source_module_name",
                    "ko",
                    "ko_name",
                    "ko_label",
                    "module_order",
                    "ko_order",
                ]
                if column in frame.columns
            ]
            frame = frame.loc[:, columns].drop_duplicates(["module_id", "ko"]).copy()
        else:
            frame = load_cycle_modules_with_labels(results_dir, cycle).copy()
        frame["elemental_cycle"] = cycle
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ordered_kos(module_data: pd.DataFrame) -> list[str]:
    return (
        module_data.drop_duplicates("ko")
        .sort_values(["ko_order", "ko"])["ko"]
        .astype(str)
        .tolist()
    )


def build_individual_metadata(
    selected: pd.DataFrame,
    module_data: pd.DataFrame,
    present_pairs: set[tuple[str, str]],
    output_path: Path,
) -> Path:
    table = selected.copy()
    table["Completeness_percent"] = pd.to_numeric(table.get("Completeness", np.nan), errors="coerce")
    table["Contamination_percent"] = pd.to_numeric(table.get("Contamination", np.nan), errors="coerce")
    table["Qscore_value"] = pd.to_numeric(table.get("qscore", np.nan), errors="coerce")
    table["16S_percent"] = pd.to_numeric(table.get("16S_rRNA", np.nan), errors="coerce") * 100.0
    kos = ordered_kos(module_data)
    for ko in kos:
        table[f"ko_{ko}"] = [int((str(bin_id), ko) in present_pairs) for bin_id in table["bin_id"].astype(str)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, ["tree_id", "Completeness_percent", "Contamination_percent", "Qscore_value", "16S_percent", *[f"ko_{ko}" for ko in kos]]].to_csv(
        output_path,
        sep="\t",
        index=False,
    )
    return output_path


def build_group_metadata(
    members: pd.DataFrame,
    taxonomy: pd.DataFrame,
    module_data: pd.DataFrame,
    present_pairs: set[tuple[str, str]],
    output_path: Path,
) -> Path:
    grouped = members.groupby("tree_id", sort=True)
    table = taxonomy.copy()
    table["n_genomes"] = table["tree_id"].map(grouped.size()).astype(int)
    table["Completeness_percent"] = table["tree_id"].map(grouped["Completeness"].median())
    table["Contamination_percent"] = table["tree_id"].map(grouped["Contamination"].median())
    table["Qscore_value"] = table["tree_id"].map(grouped["qscore"].median())
    table["16S_percent"] = table["tree_id"].map(grouped["16S_rRNA"].median()) * 100.0
    group_bins = {
        group_id: set(group["bin_id"].astype(str))
        for group_id, group in grouped
    }
    for ko in ordered_kos(module_data):
        present_bins = {bin_id for bin_id, pair_ko in present_pairs if pair_ko == ko}
        table[f"ko_{ko}"] = [
            100.0 * len(group_bins[group_id] & present_bins) / len(group_bins[group_id])
            for group_id in table["tree_id"]
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, sep="\t", index=False)
    return output_path


def render(config: Path, render_env: str) -> None:
    subprocess.run(
        [
            str(Path.home() / "mambaforge" / "bin" / "conda"),
            "run",
            "-n",
            render_env,
            "python",
            str(REPO_ROOT / "render_phylogeny_ete.py"),
            "render",
            "-c",
            str(config),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    selected_path, individual_tree, individual_shared_metadata = cohort_paths(results_dir)
    selected = pd.read_csv(selected_path, sep="\t")
    cycles = args.cycles or ELEMENTAL_REPRESENTATIVE_CYCLES
    modules = all_elemental_modules(results_dir, cycles)
    if args.module_id:
        module_ids = modules["module_id"].astype(str)
        modules = modules.loc[module_ids.eq(args.module_id) | module_ids.str.startswith(f"{args.module_id}.")].copy()
        if modules.empty:
            raise ValueError(f"No elemental module matched {args.module_id!r} in {list(cycles)}")

    selected_bins = set(selected["bin_id"].astype(str))
    selected_kos = set(modules["ko"].astype(str))
    presence = load_selected_ko_presence(results_dir / "all_kofam.mapper.unique.tsv", selected_bins, selected_kos)
    present_pairs = set(zip(presence["bin_id"].astype(str), presence["ko"].astype(str)))

    species_source = selected.rename(columns={"tree_id": "source_tree_id"})
    members, taxonomy = exact_taxonomy_groups(species_source)
    shared_group_dir = selected_path.parent / "taxonomy_aggregated_species"
    group_tree = shared_group_dir / "exact_species_taxonomy_group_tree.nwk"
    build_group_distance_tree(individual_tree, members, group_tree)
    members.to_csv(shared_group_dir / "exact_species_taxonomy_group_members.tsv", sep="\t", index=False)

    rows = []
    grouped_modules = modules.groupby(["elemental_cycle", "module_id", "module_name"], sort=False, dropna=False)
    total_renders = grouped_modules.ngroups * len(args.variants)
    render_index = 0
    for (cycle, module_id, module_name), module_data in grouped_modules:
        module_prefix = f"{safe_filename(module_id)}_{safe_filename(module_name)}"
        file_prefix = module_prefix.replace(".", "_")
        module_root = (
            results_dir
            / "elemental_cycles"
            / str(cycle)
            / "sag_xpg_mq_16s_gt0.5_module_phylogenies"
            / module_prefix
        )
        for variant in args.variants:
            render_index += 1
            out_dir = module_root / ("individual_genomes" if variant == "individual" else "species_aggregated")
            output_prefix = out_dir / f"{file_prefix}_{variant.replace('-', '_')}_ete"
            expected_pdf = output_prefix.with_suffix(".pdf")
            if args.skip_existing and expected_pdf.is_file():
                status = "skipped_existing"
            else:
                if variant == "individual":
                    metadata_path = build_individual_metadata(
                        selected,
                        module_data,
                        present_pairs,
                        out_dir / f"{file_prefix}_individual_metadata.tsv",
                    )
                    tree = individual_tree
                    metadata_paths = [*individual_shared_metadata, metadata_path]
                else:
                    metadata_path = build_group_metadata(
                        members,
                        taxonomy,
                        module_data,
                        present_pairs,
                        out_dir / f"{file_prefix}_species_aggregated_metadata.tsv",
                    )
                    tree = group_tree
                    metadata_paths = [metadata_path]
                    metadata_path.with_name(f"{file_prefix}_species_aggregated_prevalence.tsv").write_text(metadata_path.read_text())
                config = out_dir / f"{file_prefix}_{variant.replace('-', '_')}_ete_config.yaml"
                write_render_config(tree, metadata_paths, module_data, output_prefix, config)
                if variant == "species-aggregated":
                    adapt_config_for_group_prevalence(config, int(members.groupby("tree_id").size().max()))
                print(
                    f"[render {render_index}/{total_renders}] {variant} {cycle} {module_id} {module_name}",
                    flush=True,
                )
                render(config, args.render_env)
                status = "rendered"
            rows.append(
                {
                    "variant": variant,
                    "elemental_cycle": cycle,
                    "module_id": module_id,
                    "module_name": module_name,
                    "n_kos": int(module_data["ko"].nunique()),
                    "n_input_genomes": int(len(selected)),
                    "n_plotted_genomes": int(len(selected) if variant == "individual" else len(members)),
                    "n_tree_tips": int(len(selected) if variant == "individual" else len(taxonomy)),
                    "status": status,
                    "output_prefix": str(output_prefix),
                }
            )

    summary = pd.DataFrame(rows)
    manifest_name = "sag_xpg_mq_16s_gt0.5_module_phylogeny_manifest.tsv"
    if args.cycles or args.module_id or set(args.variants) != set(VARIANTS):
        manifest_name = "sag_xpg_mq_16s_gt0.5_module_phylogeny_subset_manifest.tsv"
    summary_path = results_dir / "elemental_cycles" / manifest_name
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[done] {len(summary)} module figures -> {summary_path}")


if __name__ == "__main__":
    main()
