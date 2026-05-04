#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "ete_pretty_tree.yaml"


def friendly_path(path_text):
    return Path(path_text).expanduser().absolute()


def default_metadata_paths(tree_path):
    stem = tree_path.stem
    parent = tree_path.parent
    return {
        "METADATA_CORE": parent / f"{stem}_core_metadata.tsv",
        "METADATA_GUNC": parent / f"{stem}_gunc_metadata.tsv",
        "METADATA_QUALITY": parent / f"{stem}_quality_metadata.tsv",
        "METADATA_TAXONOMY": parent / f"{stem}_taxonomy_metadata.tsv",
        "METADATA_SPECIFIC_MARKER": parent / f"{stem}_specific_marker_metadata.tsv",
    }


def default_selected_genomes_path(tree_path):
    for parent in tree_path.parents:
        candidate = parent / "selected_genomes.tsv"
        if candidate.exists():
            return candidate
    return tree_path.parents[2] / "selected_genomes.tsv" if len(tree_path.parents) > 2 else tree_path.parent / "selected_genomes.tsv"


def default_best_table_path(tree_path):
    for parent in tree_path.parents:
        candidate = parent / "best_of_best" / "selected_set" / "master.tsv"
        if candidate.exists():
            return candidate
    return Path("")


def build_specific_marker_metadata(tree_path, selected_path, output_path):
    script = Path(__file__).resolve().with_name("build_ete_specific_marker_metadata.py")
    if not script.exists():
        raise FileNotFoundError(f"Specific marker metadata builder not found: {script}")
    cmd = [
        sys.executable,
        str(script),
        "--tree",
        str(tree_path),
        "--selected",
        str(selected_path),
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"Failed to build specific marker metadata: {' '.join(cmd)}")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    return output_path


def replace_placeholders(template_text, values):
    rendered = template_text
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def unresolved_placeholders(text):
    tokens = []
    cursor = 0
    while True:
        start = text.find("{{", cursor)
        if start == -1:
            return tokens
        end = text.find("}}", start + 2)
        if end == -1:
            return tokens + [text[start:]]
        tokens.append(text[start : end + 2])
        cursor = end + 2


def build_parser():
    parser = argparse.ArgumentParser(
        description="Apply the reusable ETE pretty-tree YAML template to a specific tree."
    )
    parser.add_argument("--tree", required=True, help="Path to the *_tree.nwk file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output YAML path. Default: <tree_stem>_ete_config.yaml beside the tree.",
    )
    parser.add_argument(
        "--template",
        default=str(DEFAULT_TEMPLATE),
        help=f"Template YAML path. Default: {DEFAULT_TEMPLATE}",
    )
    parser.add_argument(
        "--best-table",
        default="",
        help="Path to best_of_best/selected_set/master.tsv for Species-label stars.",
    )
    parser.add_argument("--metadata-core", default=None, help="Override core metadata path.")
    parser.add_argument("--metadata-gunc", default=None, help="Override GUNC metadata path.")
    parser.add_argument("--metadata-quality", default=None, help="Override quality metadata path.")
    parser.add_argument("--metadata-taxonomy", default=None, help="Override taxonomy metadata path.")
    parser.add_argument("--metadata-specific-marker", default=None, help="Override functional marker heatmap metadata path.")
    parser.add_argument(
        "--selected",
        default=None,
        help="Selected-genomes TSV used to auto-build missing functional marker metadata. Default: nearest selected_genomes.tsv above the tree.",
    )
    parser.add_argument(
        "--build-missing-specific-marker-metadata",
        action="store_true",
        default=True,
        help="Auto-build missing *_specific_marker_metadata.tsv when the selected-genomes TSV is available. Default: on.",
    )
    parser.add_argument(
        "--no-build-missing-specific-marker-metadata",
        action="store_false",
        dest="build_missing_specific_marker_metadata",
        help="Do not auto-build missing functional marker metadata.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Rendered tree output prefix. Default: <tree_stem>_ete beside the tree.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write the YAML even if inferred input files do not exist.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    tree_path = friendly_path(args.tree)
    template_path = friendly_path(args.template)
    output_path = (
        friendly_path(args.output)
        if args.output
        else tree_path.with_name(f"{tree_path.stem}_ete_config.yaml")
    )
    output_prefix = (
        friendly_path(args.output_prefix)
        if args.output_prefix
        else tree_path.with_name(f"{tree_path.stem}_ete")
    )

    metadata = default_metadata_paths(tree_path)
    overrides = {
        "METADATA_CORE": args.metadata_core,
        "METADATA_GUNC": args.metadata_gunc,
        "METADATA_QUALITY": args.metadata_quality,
        "METADATA_TAXONOMY": args.metadata_taxonomy,
        "METADATA_SPECIFIC_MARKER": args.metadata_specific_marker,
    }
    for key, override in overrides.items():
        if override:
            metadata[key] = friendly_path(override)

    if (
        args.build_missing_specific_marker_metadata
        and not metadata["METADATA_SPECIFIC_MARKER"].exists()
    ):
        selected_path = friendly_path(args.selected) if args.selected else default_selected_genomes_path(tree_path)
        if selected_path.exists():
            try:
                built_path = build_specific_marker_metadata(
                    tree_path=tree_path,
                    selected_path=selected_path,
                    output_path=metadata["METADATA_SPECIFIC_MARKER"],
                )
                print(f"Built missing functional marker metadata: {built_path}", file=sys.stderr)
            except Exception as exc:
                print(f"Could not auto-build functional marker metadata: {exc}", file=sys.stderr)

    best_table_path = friendly_path(args.best_table) if args.best_table else default_best_table_path(tree_path)

    values = {
        "TREE": tree_path,
        "OUTPUT_PREFIX": output_prefix,
        "BEST_REPRESENTATIVE_TABLE": best_table_path,
        **metadata,
    }

    input_keys = {
        "TREE",
        "METADATA_CORE",
        "METADATA_GUNC",
        "METADATA_QUALITY",
        "METADATA_TAXONOMY",
        "METADATA_SPECIFIC_MARKER",
    }
    missing = [path for key, path in values.items() if key in input_keys and isinstance(path, Path) and not path.exists()]
    if args.best_table and not values["BEST_REPRESENTATIVE_TABLE"].exists():
        missing.append(values["BEST_REPRESENTATIVE_TABLE"])
    if missing and not args.allow_missing:
        print("Missing input file(s):", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 2

    template_text = template_path.read_text()
    rendered = replace_placeholders(template_text, values)
    unresolved = unresolved_placeholders(rendered)
    if unresolved:
        print(f"Unresolved template placeholders: {', '.join(unresolved)}", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
