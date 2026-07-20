#!/usr/bin/env python3
"""Audit publication labels for every KO in the elemental module outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kofam_cross_scope_module_panels import infer_ko_short_label  # noqa: E402
from kofam_nitrogen_module_ko_facets import DEFAULT_RESULTS_DIR  # noqa: E402

CYCLES = ("nitrogen", "sulfur", "carbon", "phosphorus", "iron", "trace_metals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    rows = []
    for cycle in CYCLES:
        path = (
            results_dir
            / "elemental_cycles"
            / cycle
            / "all"
            / f"kofam_{cycle}_modules.ko_delta_plotdata.tsv"
        )
        table = pd.read_csv(path, sep="\t", usecols=["module_id", "module_name", "ko", "ko_name"])
        table = table.drop_duplicates(["module_id", "ko"])
        table["elemental_cycle"] = cycle
        rows.append(table)

    combined = pd.concat(rows, ignore_index=True)
    audit_rows = []
    for ko, group in combined.groupby("ko", sort=True):
        names = [value for value in group["ko_name"].dropna().astype(str).unique() if value.strip()]
        ko_name = names[0] if names else ""
        short_label, source = infer_ko_short_label(ko, ko_name)
        audit_rows.append(
            {
                "ko": ko,
                "short_label": short_label or "",
                "display_label": f"{short_label} ({ko})" if short_label else f"{ko_name} ({ko})",
                "ko_name": ko_name,
                "label_source": source,
                "confidence": "high" if source != "description_fallback" else "fallback",
                "elemental_cycles": ";".join(sorted(group["elemental_cycle"].unique())),
                "module_ids": ";".join(sorted(group["module_id"].astype(str).unique())),
            }
        )

    audit = pd.DataFrame(audit_rows)
    output = args.output or results_dir / "elemental_cycles" / "elemental_ko_short_labels.audit.tsv"
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, sep="\t", index=False)
    counts = audit["label_source"].value_counts().to_dict()
    print(f"Wrote {len(audit)} KO labels to {output}")
    print("Label sources: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
