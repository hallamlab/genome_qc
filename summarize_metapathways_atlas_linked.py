#!/usr/bin/env python3

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from metapathways_atlas_linked_core import run_atlas_linked_comparisons
from summarize_metapathways_wrapper import find_preferred_file, organize_output_tree, sanitize_label


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run atlas-linked MetaPathways genome/category comparison analyses from an "
            "existing summarize_metapathways_wrapper.py output directory using the "
            "Shaiber/Willis-style grouped logit/Rao enrichment workflow implemented "
            "locally in Python."
        )
    )
    parser.add_argument(
        "functional_dir",
        help=(
            "Existing MetaPathways wrapper output directory containing "
            "*_genome_summary.tsv and, when available, *_elemental_annotation_audit.tsv."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Directory for atlas-linked outputs. Default: write into functional_dir."
        ),
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Functional table prefix to read/write. Default: infer from the selected "
            "*_genome_summary.tsv, usually metapathways_batch."
        ),
    )
    parser.add_argument(
        "--genome-atlas-dir",
        required=True,
        help="Genome atlas combined output directory.",
    )
    parser.add_argument(
        "--atlas-annotated-tsv",
        default=None,
        help=(
            "Explicit atlas *_annotated.tsv path. Required for --atlas-linkage-mode "
            "species_all unless discoverable from --genome-atlas-dir."
        ),
    )
    parser.add_argument(
        "--atlas-shared-best-tsv",
        default=None,
        help=(
            "Explicit atlas *_shared_best_genomes.tsv path. Required for "
            "--atlas-linkage-mode shared_best unless discoverable from --genome-atlas-dir."
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
        help="Comparison/category column used in atlas outputs. Default: category",
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
        "--atlas-linkage-mode",
        choices=["species_all", "shared_best"],
        default="species_all",
        help=(
            "Atlas-linked comparison design. 'species_all' links all atlas-matched "
            "genomes with complete species-level GTDB taxonomy by full lineage and "
            "keeps all members. 'shared_best' uses atlas shared-best representatives. "
            "Default: species_all"
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
        "--atlas-disable-alias-fallback",
        action="store_true",
        help="Disable conservative ID alias fallback when joining atlas genomes to MetaPathways genome IDs.",
    )
    parser.add_argument(
        "--min-mimag-tier",
        default="low",
        choices=["low", "medium", "high"],
        help=(
            "Minimum MIMAG tier for atlas-linked comparisons. Default: low "
            "(keeps low+medium+high)."
        ),
    )
    parser.add_argument(
        "--no-organize-output",
        action="store_true",
        help="Do not move generated outputs into plots/ and tables/ subdirectories.",
    )
    parser.add_argument(
        "--skip-annotation-presence",
        action="store_true",
        help=(
            "Skip loading *_elemental_annotation_audit.tsv and skip linked annotation "
            "presence matrices. Useful for quick test runs."
        ),
    )
    parser.add_argument(
        "--max-heatmap-functions",
        type=int,
        default=250,
        help=(
            "Maximum functional annotations/features to draw in linked comparative "
            "heatmaps. Full matrices/tables are still written. Default: 250"
        ),
    )
    parser.add_argument(
        "--max-lineage-detail-plots",
        type=int,
        default=25,
        help=(
            "Maximum candidate lineages to draw individual annotation heatmaps for, "
            "ranked by category completeness and genome count. Tables are written "
            "for all selected candidate lineages. Default: 25"
        ),
    )
    return parser


def choose_functional_file(functional_dir, prefix, suffix, required=True):
    root = Path(functional_dir).expanduser().resolve()
    if prefix:
        return find_preferred_file(
            root,
            f"{sanitize_label(prefix)}_{suffix}",
            f"*_{suffix}",
            required=required,
        )

    candidates = sorted(path for path in root.rglob(f"*_{suffix}") if path.is_file())
    if not candidates:
        if required:
            raise FileNotFoundError(f"No '*_{suffix}' files found under: {root}")
        return None

    def score(path):
        parts = set(path.parts)
        is_main_batch = path.name == f"metapathways_batch_{suffix}"
        is_best_subset = bool({"best_of_sample", "best_of_best", "selected_set"} & parts)
        return (
            int(is_best_subset),
            int(not is_main_batch),
            len(path.parts),
            str(path),
        )

    return sorted(candidates, key=score)[0]


def infer_prefix_from_genome_path(path):
    name = Path(path).name
    suffix = "_genome_summary.tsv"
    if not name.endswith(suffix):
        raise ValueError(f"Cannot infer prefix from genome summary path: {path}")
    return name[: -len(suffix)]


def load_functional_tables(functional_dir, prefix=None, load_annotation_audit=True):
    genome_path = choose_functional_file(
        functional_dir=functional_dir,
        prefix=prefix,
        suffix="genome_summary.tsv",
        required=True,
    )
    inferred_prefix = sanitize_label(prefix) if prefix else infer_prefix_from_genome_path(genome_path)
    audit_path = None
    if load_annotation_audit:
        audit_path = choose_functional_file(
            functional_dir=functional_dir,
            prefix=inferred_prefix,
            suffix="elemental_annotation_audit.tsv",
            required=False,
        )

    genome_df = pd.read_csv(genome_path, sep="\t", low_memory=False)
    audit_df = pd.read_csv(audit_path, sep="\t", low_memory=False) if audit_path else pd.DataFrame()
    return inferred_prefix, genome_path, audit_path, genome_df, audit_df


def build_atlas_args(args, prefix):
    return SimpleNamespace(
        genome_atlas_dir=str(Path(args.genome_atlas_dir).expanduser().resolve()),
        atlas_shared_best_tsv=(
            str(Path(args.atlas_shared_best_tsv).expanduser().resolve())
            if args.atlas_shared_best_tsv else None
        ),
        atlas_annotated_tsv=(
            str(Path(args.atlas_annotated_tsv).expanduser().resolve())
            if args.atlas_annotated_tsv else None
        ),
        atlas_prefix=args.atlas_prefix,
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


def main():
    args = build_parser().parse_args()
    functional_dir = Path(args.functional_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else functional_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix, genome_path, audit_path, genome_df, audit_df = load_functional_tables(
        functional_dir=functional_dir,
        prefix=args.prefix,
        load_annotation_audit=not args.skip_annotation_presence,
    )
    print(f"[info] functional dir: {functional_dir}")
    print(f"[info] output dir: {output_dir}")
    print(f"[info] prefix: {prefix}")
    print(f"[info] genome summary: {genome_path} rows={len(genome_df)}")
    if args.skip_annotation_presence:
        print("[skip] annotation audit loading skipped by --skip-annotation-presence")
    elif audit_path:
        print(f"[info] annotation audit: {audit_path} rows={len(audit_df)}")
    else:
        print("[warn] annotation audit not found; annotation presence outputs will be empty.")

    print("[start] atlas-linked genome comparisons")
    wrote_paths = run_atlas_linked_comparisons(
        args=build_atlas_args(args, prefix),
        output_dir=output_dir,
        combined_genome_df=genome_df,
        combined_annotation_audit_df=audit_df,
    )
    print(f"[done] atlas-linked genome comparison outputs: {len(wrote_paths)} files")

    if not args.no_organize_output:
        print("[start] organizing atlas-linked outputs")
        organized_count, _index_paths = organize_output_tree(output_dir)
        print(f"[done] organized files: {organized_count}")


if __name__ == "__main__":
    main()
