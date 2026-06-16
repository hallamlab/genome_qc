#!/usr/bin/env python3

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genome_quality_atlas import (  # noqa: E402
    atlas_single_metric_specs,
    compute_quality_index,
    ensure_plotting,
    export_compare_single_metric_plots,
    save_figure,
    sanitize_token,
)
from genome_quality_atlas_wrapper import organize_directory_outputs  # noqa: E402


DEFAULT_ATLAS_DIR = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_genome_atlas_gunc_98"
)
DEFAULT_PROJECT_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109")
DEFAULT_INPUT = DEFAULT_ATLAS_DIR / "tables" / "other" / "genome_quality_annotated.tsv"
DEFAULT_MANIFEST = DEFAULT_PROJECT_DIR / "qc_runs_mags_sags.csv"
DEFAULT_OUTPUT = DEFAULT_ATLAS_DIR / "genome_quality_compare_category_single_metrics" / "autoopt"
AUTOOPT_ORDER = ["algo_defaults", "best_match", "best_cluster", "majority_rule"]
PR_ORDER = ["Default", "very_strict", "strict", "relaxed", "very_relaxed"]
METRIC_SCALE_LIMITS = {
    "qscore": (0.0, 103.0, "0-100"),
    "Completeness": (0.0, 103.0, "0-100"),
    "Contamination": (0.0, 100.0, "0-100"),
    "integrity_score": (-0.03, 1.03, "0-1"),
    "recoverability_score": (-0.03, 1.03, "0-1"),
    "mimag_quality_index": (-0.03, 1.03, "0-1"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One-off SAG-xPG autoopt/P-R single-metric panels from "
            "genome_quality_annotated.tsv."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Annotated genome quality TSV.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="QC run manifest used by --source raw. Default: qc_runs_mags_sags.csv.",
    )
    parser.add_argument(
        "--source",
        choices=["annotated", "raw"],
        default="annotated",
        help="Use atlas annotated table or raw run Master_genome_QC.tsv files. Default: annotated.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT),
        help="Output directory. Default: final_genome_atlas_gunc_98/..._single_metrics/autoopt",
    )
    parser.add_argument(
        "--category",
        default="xPGs_SAGs",
        help="Category to keep before parsing Bin Id. Default: xPGs_SAGs.",
    )
    parser.add_argument(
        "--prefix",
        default="genome_quality_sag_xpg_autoopt",
        help="Filename prefix for generated plots and tables.",
    )
    return parser.parse_args()


def find_column(frame, candidates):
    normalized = {str(column).strip().lower().replace("_", " "): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def add_bin_id_groups(frame, bin_column):
    parsed = frame[bin_column].astype(str).str.strip().str.split(".", expand=True)
    frame = frame.copy()
    frame["autoopt"] = parsed[1] if 1 in parsed.columns else np.nan
    frame["P/R"] = parsed[2] if 2 in parsed.columns else np.nan
    for column in ["autoopt", "P/R"]:
        frame[column] = frame[column].astype(str).str.strip()
        frame.loc[frame[column].isin(["", "nan", "None"]), column] = np.nan
    return frame


def load_raw_manifest(manifest_path, category):
    manifest = pd.read_csv(
        manifest_path,
        sep=None,
        engine="python",
        header=None,
        names=["sample", "category", "source_dir", "enabled"],
        dtype=str,
    )
    manifest = manifest.loc[manifest["category"].astype(str).eq(str(category))].copy()
    if "enabled" in manifest.columns:
        enabled = manifest["enabled"].fillna("1").astype(str).str.strip().str.lower()
        enabled_numeric = pd.to_numeric(enabled, errors="coerce")
        manifest = manifest.loc[
            enabled.isin({"true", "yes", "y"}) | enabled_numeric.fillna(0).ne(0)
        ].copy()
    return manifest


def load_raw_qc_runs(manifest_path, category):
    manifest = load_raw_manifest(manifest_path, category)
    frames = []
    missing = []
    for row in manifest.to_dict("records"):
        source_dir = Path(str(row["source_dir"])).expanduser()
        master_path = source_dir / "Master_genome_QC.tsv"
        if not master_path.is_file():
            missing.append(str(master_path))
            continue
        run_frame = pd.read_csv(master_path, sep="\t", low_memory=False)
        run_frame["sample"] = row["sample"]
        run_frame["category"] = row["category"]
        run_frame["source_dir"] = str(source_dir)
        frames.append(run_frame)
    if missing:
        raise SystemExit("Missing raw Master_genome_QC.tsv files:\n" + "\n".join(missing))
    if not frames:
        raise SystemExit(f"No raw QC runs found for category={category!r}.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return compute_quality_index(combined)


def metric_summary(frame, group_column):
    rows = []
    for metric, label in atlas_single_metric_specs(frame):
        values = pd.to_numeric(frame[metric], errors="coerce")
        summary_frame = frame.assign(_metric_value=values).dropna(subset=[group_column, "_metric_value"])
        for group, subset in summary_frame.groupby(group_column, sort=True):
            group_values = subset["_metric_value"].to_numpy(dtype=float)
            if group_values.size == 0:
                continue
            rows.append(
                {
                    "comparison": group_column,
                    "group": group,
                    "metric": metric,
                    "metric_label": label,
                    "n": int(group_values.size),
                    "mean": float(np.mean(group_values)),
                    "sd": float(np.std(group_values, ddof=1)) if group_values.size > 1 else 0.0,
                    "median": float(np.median(group_values)),
                    "q1": float(np.percentile(group_values, 25)),
                    "q3": float(np.percentile(group_values, 75)),
                    "min": float(np.min(group_values)),
                    "max": float(np.max(group_values)),
                }
            )
    return pd.DataFrame(rows)


def write_summary_tables(frame, output_dir, prefix):
    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    wrote = []

    group_counts = []
    for group_column in ["autoopt", "P/R"]:
        counts = (
            frame[group_column]
            .value_counts(dropna=False)
            .rename_axis(group_column)
            .reset_index(name="n_genomes")
        )
        counts.insert(0, "comparison", group_column)
        group_counts.append(counts)
        summary = metric_summary(frame, group_column)
        summary_path = summary_dir / f"{prefix}_compare_{sanitize_token(group_column)}_metric_summary.tsv"
        summary.to_csv(summary_path, sep="\t", index=False)
        wrote.append(summary_path)

    counts_path = summary_dir / f"{prefix}_parsed_group_counts.tsv"
    pd.concat(group_counts, ignore_index=True).to_csv(counts_path, sep="\t", index=False)
    wrote.append(counts_path)
    return wrote


def write_metric_scale_audit(frame, output_dir, prefix):
    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metric, (axis_min, axis_max, scale) in METRIC_SCALE_LIMITS.items():
        if metric not in frame.columns:
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if values.empty:
            continue
        observed_min = float(values.min())
        observed_max = float(values.max())
        rows.append(
            {
                "metric": metric,
                "expected_scale": scale,
                "axis_min": axis_min,
                "axis_max": axis_max,
                "observed_min": observed_min,
                "observed_max": observed_max,
                "below_axis_min": int((values < axis_min).sum()),
                "above_axis_max": int((values > axis_max).sum()),
                "n": int(values.size),
            }
        )
    audit = pd.DataFrame(rows)
    audit_path = summary_dir / f"{prefix}_metric_scale_audit.tsv"
    audit.to_csv(audit_path, sep="\t", index=False)
    return audit_path


def ordered_present(values, preferred_order):
    values = [str(value) for value in pd.Series(values).dropna().unique()]
    preferred = [value for value in preferred_order if value in values]
    remaining = sorted(value for value in values if value not in set(preferred))
    return preferred + remaining


def write_qscore_heatmaps(frame, output_dir, prefix):
    ensure_plotting()
    import genome_quality_atlas as atlas

    heatmap_frame = frame.dropna(subset=["autoopt", "P/R"]).copy()
    heatmap_frame["qscore"] = pd.to_numeric(heatmap_frame.get("qscore"), errors="coerce")
    heatmap_frame = heatmap_frame.dropna(subset=["qscore"])
    if heatmap_frame.empty:
        return []

    row_order = ordered_present(heatmap_frame["autoopt"], AUTOOPT_ORDER)
    column_order = ordered_present(heatmap_frame["P/R"], PR_ORDER)
    grouped = heatmap_frame.groupby(["autoopt", "P/R"], dropna=False)["qscore"]
    summary = grouped.agg(n="count", mean="mean", median="median", sd="std").reset_index()
    summary_path = output_dir / f"{prefix}_qscore_heatmap_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    wrote = [summary_path]
    for mode in ["mean", "median"]:
        matrix = (
            summary.pivot(index="autoopt", columns="P/R", values=mode)
            .reindex(index=row_order, columns=column_order)
        )
        count_matrix = (
            summary.pivot(index="autoopt", columns="P/R", values="n")
            .reindex(index=row_order, columns=column_order)
        )
        matrix_path = output_dir / f"{prefix}_qscore_heatmap_{mode}_matrix.tsv"
        count_path = output_dir / f"{prefix}_qscore_heatmap_{mode}_counts.tsv"
        matrix.to_csv(matrix_path, sep="\t")
        count_matrix.to_csv(count_path, sep="\t")
        wrote.extend([matrix_path, count_path])

        fig_width = max(6.5, len(column_order) * 1.35)
        fig_height = max(4.8, len(row_order) * 0.9)
        fig, ax = atlas.plt.subplots(figsize=(fig_width, fig_height))
        valid_values = matrix.to_numpy(dtype=float)
        finite_values = valid_values[np.isfinite(valid_values)]
        vmin = float(np.nanmin(finite_values)) if finite_values.size else 0.0
        vmax = float(np.nanmax(finite_values)) if finite_values.size else 1.0
        if vmin == vmax:
            vmin -= 0.5
            vmax += 0.5
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Greys", vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(np.arange(len(column_order)))
        ax.set_xticklabels(column_order, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(row_order)))
        ax.set_yticklabels(row_order)
        ax.set_xlabel("P/R")
        ax.set_ylabel("autoopt")
        ax.set_title(f"SAG-xPG qscore by autoopt and P/R ({mode})", fontsize=12, fontweight="bold")

        for row_index, row_name in enumerate(row_order):
            for col_index, col_name in enumerate(column_order):
                value = matrix.loc[row_name, col_name]
                if pd.isna(value):
                    continue
                midpoint = (vmin + vmax) / 2.0
                color = "white" if float(value) > midpoint else "black"
                ax.text(col_index, row_index, f"{float(value):.1f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_xticks(np.arange(-0.5, len(column_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label(f"{mode} qscore")
        fig.tight_layout()
        out_base = output_dir / f"{prefix}_qscore_heatmap_{mode}"
        save_figure(fig, str(out_base))
        wrote.extend([Path(str(out_base) + ".png"), Path(str(out_base) + ".pdf")])
    return wrote


def move_generated_files(temp_dir, output_dir):
    moved = []
    for source in sorted(temp_dir.rglob("*")):
        if not source.is_file():
            continue
        target = output_dir / source.name
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.move(str(source), str(target))
        moved.append(target)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return moved


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "raw":
        frame = load_raw_qc_runs(manifest_path, args.category)
    else:
        frame = pd.read_csv(input_path, sep="\t", low_memory=False)
    bin_column = find_column(frame, ["bin_id", "Bin Id", "Bin_Id"])
    category_column = find_column(frame, ["category"])
    if bin_column is None:
        raise SystemExit("Could not find a bin_id/Bin Id column in the input table.")
    if category_column is None:
        raise SystemExit("Could not find a category column in the input table.")

    frame = frame.loc[frame[category_column].astype(str).eq(str(args.category))].copy()
    if frame.empty:
        raise SystemExit(f"No rows found for category={args.category!r}.")
    frame = add_bin_id_groups(frame, bin_column)
    frame = frame.dropna(subset=["autoopt", "P/R"]).copy()
    if frame.empty:
        raise SystemExit("No rows remained after parsing autoopt and P/R from Bin Id.")

    safe_prefix = sanitize_token(args.prefix)
    write_summary_tables(frame, output_dir, safe_prefix)
    write_metric_scale_audit(frame, output_dir, safe_prefix)

    temp_dir = output_dir / "_autoopt_build"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    written_plots = []
    for group_column in ["autoopt", "P/R"]:
        compare_base = temp_dir / f"{safe_prefix}_compare_{sanitize_token(group_column)}"
        written_plots.extend(export_compare_single_metric_plots(frame, group_column, str(compare_base)))
    heatmap_paths = write_qscore_heatmaps(frame, temp_dir, safe_prefix)

    moved = move_generated_files(temp_dir, output_dir)
    organize_directory_outputs(output_dir)

    print(f"input_rows={len(frame)}")
    print(f"summary_tables_dir={output_dir / 'tables' / 'summary'}")
    print(f"plot_files={len(written_plots)}")
    print(f"heatmap_files={len(heatmap_paths)}")
    print(f"moved_files={len(moved)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
