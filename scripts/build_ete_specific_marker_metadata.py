#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from best_set_phylogeny import add_tree_ids, parse_newick_tip_labels
from summarize_metapathways_genomes import (
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    MARKER_MANIFEST_PATH,
    load_marker_manifest,
    marker_mode_possible_counts,
)


SHORT_LABELS = {
    "arsenic_detox_redox": "As detox",
    "carbon_fixation": "C fix",
    "co_formate": "CO/formate",
    "methylotrophy_methane": "C1/methane",
    "methanogenesis": "Methanogenesis",
    "hydrogen_metabolism": "H2",
    "iron_acquisition": "Fe uptake",
    "iron_redox": "Fe redox",
    "anammox": "Anammox",
    "denitrification": "Denitrif",
    "dnra": "DNRA",
    "nitrate_reduction": "NO3 red",
    "nitrification": "Nitrif",
    "nitrite_reduction": "NO2 red",
    "nitrogen_fixation": "N fix",
    "urea_cyanate": "Urea/cyan",
    "phosphate_scavenging": "P scav",
    "phosphonate_phosphite": "Phosphonate",
    "sulfate_reduction": "SO4 red",
    "sulfur_oxidation": "S oxid",
    "reverse_dsr_sulfur_oxidation": "rDSR S oxid",
    "thiosulfate_tetrathionate": "Thio/tetra",
    "organosulfur": "Org sulfur",
    "dmsp_dms": "DMSP/DMS",
    "reductive_tca": "rTCA",
}


def marker_column(mode_id):
    return f"mp_marker_{mode_id}_specific_gene_count"


def fraction_column(mode_id):
    return f"marker_specific_{mode_id}_fraction"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build compact ETE metadata for MetaPathways specific marker fractions."
    )
    parser.add_argument("--tree", required=True, help="Tree Newick used to order/filter output rows.")
    parser.add_argument("--selected", required=True, help="Selected-genomes TSV with mp_marker_* columns.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output TSV. Default: <tree_stem>_specific_marker_metadata.tsv beside the tree.",
    )
    parser.add_argument(
        "--marker-manifest",
        default=str(MARKER_MANIFEST_PATH),
        help="Marker manifest used to compute possible specific marker denominators.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tree_path = Path(args.tree).expanduser()
    selected_path = Path(args.selected).expanduser()
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else tree_path.with_name(f"{tree_path.stem}_specific_marker_metadata.tsv")
    )

    tips = parse_newick_tip_labels(tree_path)
    selected = pd.read_csv(selected_path, sep="\t", dtype=str).fillna("")
    selected = add_tree_ids(selected)
    marker_manifest = load_marker_manifest(Path(args.marker_manifest).expanduser())
    _, _, specific_possible_counts, _ = marker_mode_possible_counts(marker_manifest)

    rows = []
    for row in selected.to_dict("records"):
        out = {"tree_id": row.get("tree_id", "")}
        for mode_id in ELEMENTAL_MODE_ORDER:
            observed = pd.to_numeric(pd.Series([row.get(marker_column(mode_id), "0")]), errors="coerce").fillna(0).iat[0]
            possible = float(specific_possible_counts.get(mode_id, 0) or 0)
            out[fraction_column(mode_id)] = 0.0 if possible <= 0 else max(0.0, min(float(observed) / possible, 1.0))
        rows.append(out)

    metadata = pd.DataFrame(rows)
    if tips:
        metadata = metadata.drop_duplicates("tree_id").set_index("tree_id").reindex(tips).reset_index().fillna(0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, sep="\t", index=False)
    print(output_path)
    print("heatmap_columns:")
    for mode_id in ELEMENTAL_MODE_ORDER:
        label = SHORT_LABELS.get(mode_id, ELEMENTAL_MODE_LABELS.get(mode_id, mode_id))
        print(f"- column: {fraction_column(mode_id)}")
        print(f"  label: {label}")
        print("  show: true")
        print("  width: 5")
        print("  height: 9")
        print("  min_value: 0")
        print("  max_value: 1")
        print("  value_scale: 1")
        print("  reverse: false")
        print("  gap_after: 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
