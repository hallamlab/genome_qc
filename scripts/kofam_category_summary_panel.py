#!/usr/bin/env python3
"""Build KOfam category summary panels from genome_qc_results tables."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summarize_metapathways_genomes import ensure_plotting, save_figure  # noqa: E402
from summarize_metapathways_wrapper import (  # noqa: E402
    add_benchmark_significance_brackets,
    grayscale_palette,
    metric_interval_summary,
    style_benchmark_metric_axis,
)


DEFAULT_RESULTS_DIR = Path(
    "/media/nfs/Ryan/SABer/SI_data/SI_METAGs/SABer_260109/genome_qc_results"
)
CATEGORY_ORDER = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]
NONINFORMATIVE_BRITE_A = {"not included in pathway or brite"}
NONINFORMATIVE_BRITE_B = {"poorly characterized", "not included in regular maps"}
NONINFORMATIVE_BRITE_C = {"function unknown", "general function prediction only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--prefix", default="kofam_category_summary")
    parser.add_argument("--q-threshold", type=float, default=0.05)
    return parser.parse_args()


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False, **kwargs)


def normalize_category_order(values: pd.Series) -> list[str]:
    present = [category for category in CATEGORY_ORDER if category in set(values.astype(str))]
    extras = sorted(set(values.astype(str)) - set(present))
    return present + extras


def resolve_existing_path(path_text: str, results_dir: Path) -> Path | None:
    path = Path(str(path_text))
    if path.is_file():
        return path

    parts = path.parts
    if "genome_qc_results" in parts:
        index = parts.index("genome_qc_results")
        relative = Path(*parts[index + 1 :])
        candidate = results_dir / relative
        if candidate.is_file():
            return candidate
        path = candidate

    if path.name.endswith(".kofamscan.mapper.tsv"):
        candidate = path.parent / "tables" / "other" / path.name
        if candidate.is_file():
            return candidate
        genome_name = path.name[: -len(".kofamscan.mapper.tsv")]
        candidate = path.parent / "tables" / "other" / f"{genome_name}.ORF_annotation_table.txt"
        if candidate.is_file():
            return candidate
    return None


def resolve_mapper_context(mapper_path_text: str, results_dir: Path) -> tuple[Path, str]:
    mapper_path = resolve_existing_path(mapper_path_text, results_dir)
    if mapper_path is None:
        path = Path(str(mapper_path_text))
        parts = path.parts
        if "genome_qc_results" in parts:
            index = parts.index("genome_qc_results")
            path = results_dir / Path(*parts[index + 1 :])
        mapper_path = path

    for suffix in [
        ".kofamscan.mapper.tsv",
        ".kofamscan.passed.tsv",
        ".kofamscan.detail.tsv",
    ]:
        if mapper_path.name.endswith(suffix):
            genome_name = mapper_path.name[: -len(suffix)]
            break
    else:
        genome_name = mapper_path.stem
    return mapper_path, genome_name


def resolve_orf_annotation_path(mapper_path_text: str, results_dir: Path) -> Path | None:
    mapper_path, genome_name = resolve_mapper_context(mapper_path_text, results_dir)

    candidates = [
        mapper_path.parent / "tables" / "other" / f"{genome_name}.ORF_annotation_table.txt",
        mapper_path.parent / f"{genome_name}.ORF_annotation_table.txt",
    ]
    if mapper_path.parent.name == "other" and mapper_path.parent.parent.name == "tables":
        candidates.append(mapper_path.parent / f"{genome_name}.ORF_annotation_table.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_raw_orf_fasta_path(mapper_path_text: str, results_dir: Path) -> Path | None:
    mapper_path, genome_name = resolve_mapper_context(mapper_path_text, results_dir)
    parents = list(mapper_path.parents)
    mpoutput_root = None
    for parent in parents:
        if parent.name == genome_name and parent.parent.name == "mpoutput":
            mpoutput_root = parent
            break
    if mpoutput_root is None:
        for parent in parents:
            candidate = parent / "orf_prediction"
            if candidate.is_dir():
                mpoutput_root = parent
                break
    if mpoutput_root is None:
        return None
    candidates = [
        mpoutput_root / "orf_prediction" / f"{genome_name}.faa",
        mpoutput_root / "orf_prediction" / f"{genome_name}.qced.faa",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def count_fasta_records(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    with path.open("rt", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.startswith(">"))


def build_informative_ko_set(brite_path: Path) -> tuple[set[str], pd.DataFrame]:
    brite = read_tsv(brite_path)
    required = {"ko", "brite_A", "brite_B", "brite_C"}
    missing = required - set(brite.columns)
    if missing:
        raise ValueError(f"{brite_path} is missing columns: {', '.join(sorted(missing))}")
    for column in ["ko", "brite_A", "brite_B", "brite_C"]:
        brite[column] = brite[column].fillna("").astype(str).str.strip()
    noninformative_row = (
        brite["brite_A"].str.casefold().isin(NONINFORMATIVE_BRITE_A)
        | brite["brite_B"].str.casefold().isin(NONINFORMATIVE_BRITE_B)
        | brite["brite_C"].str.casefold().isin(NONINFORMATIVE_BRITE_C)
    )
    brite["brite_row_is_informative"] = ~noninformative_row
    ko_summary = (
        brite.groupby("ko", as_index=False)
        .agg(
            n_brite_rows=("ko", "size"),
            n_informative_brite_rows=("brite_row_is_informative", "sum"),
        )
    )
    ko_summary["ko_is_informative"] = ko_summary["n_informative_brite_rows"].gt(0)
    informative_kos = set(ko_summary.loc[ko_summary["ko_is_informative"], "ko"].astype(str))
    return informative_kos, ko_summary


def parse_mapper_gene_id(value: str) -> tuple[str, str]:
    text = str(value).strip()
    if "||" in text:
        bin_id, gene_id = text.split("||", 1)
        return bin_id, gene_id
    return text, text


def count_informative_ko_orfs(mapper_unique_path: Path, informative_kos: set[str]) -> pd.DataFrame:
    records = []
    seen = set()
    with mapper_unique_path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if parts[1].strip() not in informative_kos:
                continue
            bin_id, gene_id = parse_mapper_gene_id(parts[0])
            key = (bin_id, gene_id)
            if key in seen:
                continue
            seen.add(key)
    counts = {}
    for bin_id, _gene_id in seen:
        counts[bin_id] = counts.get(bin_id, 0) + 1
    for bin_id, count in counts.items():
        records.append({"bin_id": bin_id, "informative_ko_orfs": count})
    return pd.DataFrame(records)


def build_enriched_ko_counts(enrichment_path: Path, q_threshold: float, categories: list[str]) -> pd.DataFrame:
    enrichment = read_tsv(enrichment_path, usecols=["item", "adjusted_q_value", "associated_groups"])
    enrichment["adjusted_q_value"] = pd.to_numeric(enrichment["adjusted_q_value"], errors="coerce")
    enriched = enrichment.loc[enrichment["adjusted_q_value"].le(q_threshold)].copy()
    rows = []
    for category in categories:
        mask = enriched["associated_groups"].fillna("").astype(str).str.split(",").map(
            lambda groups: category in {group.strip() for group in groups}
        )
        rows.append(
            {
                "category": category,
                "enriched_kos_q_le_threshold": int(enriched.loc[mask, "item"].astype(str).nunique()),
            }
        )
    return pd.DataFrame(rows)


def build_enriched_ko_prevalence_table(
    enrichment_path: Path,
    q_threshold: float,
    categories: list[str],
) -> pd.DataFrame:
    enrichment = read_tsv(enrichment_path)
    required = {"item", "adjusted_q_value", "associated_groups"}
    missing = required - set(enrichment.columns)
    if missing:
        raise ValueError(f"{enrichment_path} is missing columns: {', '.join(sorted(missing))}")
    enrichment["adjusted_q_value"] = pd.to_numeric(enrichment["adjusted_q_value"], errors="coerce")
    enriched = enrichment.loc[enrichment["adjusted_q_value"].le(q_threshold)].copy()
    rows = []
    for record in enriched.to_dict("records"):
        associated = {
            group.strip()
            for group in str(record.get("associated_groups", "")).split(",")
            if group.strip()
        }
        for category in categories:
            if category not in associated:
                continue
            prevalence_column = f"p_{category}"
            if prevalence_column not in enriched.columns:
                continue
            prevalence = pd.to_numeric(pd.Series([record.get(prevalence_column)]), errors="coerce").iat[0]
            if pd.isna(prevalence):
                continue
            rows.append(
                {
                    "category": category,
                    "ko": str(record.get("item", "")),
                    "adjusted_q_value": record.get("adjusted_q_value"),
                    "prevalence_fraction": float(prevalence),
                    "prevalence_percent": float(prevalence) * 100.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_per_bin(per_bin: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    metrics = [
        ("total_orfs", "raw_orfs"),
        ("unique_gene_count", "ko_annotated_orfs"),
        ("informative_ko_orfs", "informative_ko_orfs"),
    ]
    rows = []
    for category in categories:
        subset = per_bin.loc[per_bin["category"].astype(str).eq(category)].copy()
        row = {"category": category, "genomes": int(len(subset))}
        for source_column, output_column in metrics:
            values = pd.to_numeric(subset[source_column], errors="coerce").fillna(0.0)
            row[f"total_{output_column}"] = float(values.sum())
            row[f"mean_{output_column}_per_genome"] = float(values.mean()) if len(values) else np.nan
            row[f"median_{output_column}_per_genome"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_benchmark_metric_values(
    per_bin: pd.DataFrame,
    group_summary: pd.DataFrame,
    enriched_prevalence: pd.DataFrame,
    categories: list[str],
) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    rows = []
    if {"sample_id", "category", "bin_id"}.issubset(set(per_bin.columns)):
        sample_counts = (
            per_bin.groupby(["sample_id", "category"], as_index=False)["bin_id"]
            .nunique()
            .rename(columns={"bin_id": "value"})
        )
        for record in sample_counts.to_dict("records"):
            rows.append(
                {
                    "category": str(record["category"]),
                    "metric": "genomes_per_sample",
                    "value": float(record["value"]),
                }
            )
    else:
        for record in group_summary.to_dict("records"):
            rows.append(
                {
                    "category": str(record["category"]),
                    "metric": "genomes",
                    "value": float(record["genomes"]),
                }
            )

    per_genome_metrics = [
        ("total_orfs", "Total ORFs per genome"),
        ("unique_gene_count", "KO-annotated ORFs per genome"),
        ("informative_ko_orfs", "Informative KO ORFs per genome"),
        ("ko_annotation_fraction_percent", "KO annotation fraction (%)"),
    ]
    for metric, _title in per_genome_metrics:
        if metric not in per_bin.columns:
            continue
        values = pd.to_numeric(per_bin[metric], errors="coerce")
        for category, value in zip(per_bin["category"].astype(str), values):
            if pd.isna(value):
                continue
            rows.append({"category": category, "metric": metric, "value": float(value)})

    if not enriched_prevalence.empty:
        for record in enriched_prevalence.to_dict("records"):
            rows.append(
                {
                    "category": str(record["category"]),
                    "metric": "enriched_ko_prevalence_percent",
                    "value": float(record["prevalence_percent"]),
                }
            )

    metric_specs = [
        ("genomes_per_sample", "Genomes per sample"),
        ("total_orfs", "Total ORFs per genome"),
        ("unique_gene_count", "KO-annotated ORFs"),
        ("ko_annotation_fraction_percent", "KO annotation fraction (%)"),
        ("informative_ko_orfs", "Informative KO ORFs"),
        ("enriched_ko_prevalence_percent", "Enriched KO prevalence (%)"),
    ]
    if not any(row["metric"] == "genomes_per_sample" for row in rows):
        metric_specs[0] = ("genomes", "Number of genomes")
    return pd.DataFrame(rows), metric_specs


def benjamini_hochberg_adjust(pvalues: list[float]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(pvalues), errors="coerce")
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return adjusted
    order = valid.sort_values().index.tolist()
    ordered = valid.loc[order].to_numpy(dtype=float)
    running = 1.0
    values = np.empty(len(ordered), dtype=float)
    for reverse_index in range(len(ordered) - 1, -1, -1):
        rank = reverse_index + 1.0
        candidate = (ordered[reverse_index] * float(len(ordered))) / rank
        running = min(running, candidate)
        values[reverse_index] = min(1.0, running)
    adjusted.loc[order] = values
    return adjusted


def significance_stars(value: float) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iat[0]
    if pd.isna(numeric):
        return ""
    if numeric < 0.0001:
        return "****"
    if numeric < 0.001:
        return "***"
    if numeric < 0.01:
        return "**"
    if numeric < 0.05:
        return "*"
    return ""


def build_benchmark_stats(metric_values: pd.DataFrame, metric_specs: list[tuple[str, str]], categories: list[str]) -> pd.DataFrame:
    try:
        from scipy import stats
    except ImportError:
        return pd.DataFrame()

    rows = []
    labels = dict(metric_specs)
    for metric, _title in metric_specs:
        metric_df = metric_values.loc[metric_values["metric"].eq(metric)].copy()
        present = [
            category
            for category in categories
            if metric_df.loc[metric_df["category"].astype(str).eq(category), "value"].dropna().size > 0
        ]
        for group_a, group_b in itertools.combinations(present, 2):
            values_a = metric_df.loc[metric_df["category"].astype(str).eq(group_a), "value"].astype(float)
            values_b = metric_df.loc[metric_df["category"].astype(str).eq(group_b), "value"].astype(float)
            if values_a.empty or values_b.empty:
                continue
            try:
                statistic, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            rows.append(
                {
                    "analysis_scope": "benchmark_metric_pairwise",
                    "test": "Mann-Whitney U",
                    "metric": metric,
                    "metric_label": labels.get(metric, metric),
                    "group_a": group_a,
                    "group_b": group_b,
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "statistic": statistic,
                    "pvalue": pvalue,
                }
            )
    if not rows:
        return pd.DataFrame()
    stats_df = pd.DataFrame(rows)
    stats_df["qvalue_bh"] = benjamini_hochberg_adjust(stats_df["pvalue"].tolist())
    stats_df["significant_q05"] = pd.to_numeric(stats_df["qvalue_bh"], errors="coerce").lt(0.05)
    stats_df["significance_stars"] = stats_df["qvalue_bh"].map(significance_stars)
    return stats_df


def benchmark_grouped_values(metric_values: pd.DataFrame, metric: str, categories: list[str]) -> list[np.ndarray]:
    metric_df = metric_values.loc[metric_values["metric"].astype(str).eq(str(metric))].copy()
    return [
        pd.to_numeric(
            metric_df.loc[metric_df["category"].astype(str).eq(category), "value"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
        for category in categories
    ]


def write_benchmark_metric_panel(
    metric_values: pd.DataFrame,
    metric_specs: list[tuple[str, str]],
    stats_df: pd.DataFrame,
    output_base: Path,
    categories: list[str],
    summary_mode: str = "mean",
) -> list[Path]:
    ensure_plotting()
    if metric_values.empty or not metric_specs or not categories:
        return []

    n_cols = 3
    n_rows = int(math.ceil(len(metric_specs) / float(n_cols)))
    plt = ensure_plotting()
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(11.5, n_cols * 4.0), max(6.5, n_rows * 2.8)),
        squeeze=False,
    )
    axes = axes.ravel()
    y_positions = np.arange(len(categories), dtype=float)
    category_colors = dict(zip(categories, grayscale_palette(len(categories))))
    wrote_any = False
    if summary_mode == "mean":
        point_label = "mean"
        interval_label = "approximate 95% confidence intervals"
    else:
        point_label = "median"
        interval_label = "interquartile ranges"

    for ax, (metric, title) in zip(axes, metric_specs):
        grouped_values = benchmark_grouped_values(metric_values, metric, categories)
        summaries = [metric_interval_summary(values, summary_mode=summary_mode) for values in grouped_values]
        if not any(summary is not None for summary in summaries):
            ax.axis("off")
            continue

        for y_pos, category, summary in zip(y_positions, categories, summaries):
            if summary is None:
                continue
            color = category_colors[category]
            ax.hlines(
                y_pos,
                summary["low"],
                summary["high"],
                color=color,
                linewidth=3.0,
                zorder=2,
            )
            ax.scatter(
                summary["point"],
                y_pos,
                s=52,
                color=color,
                edgecolors="black",
                linewidths=0.75,
                zorder=3,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=1)
        ax.set_xlabel(title, fontsize=8.5)
        style_benchmark_metric_axis(ax, metric)
        ax.set_xlim(left=0)
        if metric in {"ko_annotation_fraction_percent", "enriched_ko_prevalence_percent"}:
            ax.set_xlim(0, 100)
        add_benchmark_significance_brackets(ax, metric, summaries, categories, stats_df)
        wrote_any = True

    for index in range(len(metric_specs), len(axes)):
        axes[index].axis("off")

    if not wrote_any:
        plt.close(fig)
        return []

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=2.3, h_pad=2.0)
    save_figure(fig, str(output_base))
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def long_group_metrics(group_summary: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("genomes", "Genomes", "count"),
        ("total_raw_orfs", "Total ORFs", "count"),
        ("total_ko_annotated_orfs", "KO-annotated ORFs", "count"),
        ("total_informative_ko_orfs", "Informative KO ORFs", "count"),
        ("enriched_kos_q_le_threshold", "Enriched KOs (q<=0.05)", "count"),
    ]
    rows = []
    for _, record in group_summary.iterrows():
        category = str(record["category"])
        for metric, label, unit in metric_specs:
            rows.append(
                {
                    "category": category,
                    "category_label": category,
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "value": pd.to_numeric(record.get(metric), errors="coerce"),
                }
            )
    return pd.DataFrame(rows)


def kofam_single_metric_specs() -> list[tuple[str, str, str]]:
    return [
        ("genomes", "Number of genomes", "category"),
        ("total_orfs", "Total ORFs per genome", "genome"),
        ("unique_gene_count", "KO-annotated ORFs per genome", "genome"),
        ("informative_ko_orfs", "Informative KO ORFs per genome", "genome"),
        ("enriched_kos_q_le_threshold", "Enriched KOs (q<=0.05)", "category"),
    ]


def single_metric_grouped_values(
    per_bin: pd.DataFrame,
    group_summary: pd.DataFrame,
    metric: str,
    scope: str,
    categories: list[str],
) -> list[np.ndarray]:
    if scope == "category":
        if metric not in group_summary.columns:
            return []
        values = pd.to_numeric(group_summary.set_index("category")[metric], errors="coerce")
        return [
            values.reindex([category]).dropna().to_numpy(dtype=float)
            for category in categories
        ]
    if metric not in per_bin.columns:
        return []
    return [
        pd.to_numeric(
            per_bin.loc[per_bin["category"].astype(str).eq(category), metric],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
        for category in categories
    ]


def draw_single_metric_axis(
    ax,
    grouped_values: list[np.ndarray],
    categories: list[str],
    title: str,
    rng: np.random.Generator | None = None,
) -> bool:
    if not any(len(values) for values in grouped_values):
        ax.axis("off")
        return False
    rng = rng or np.random.default_rng(42)
    box = ax.boxplot(grouped_values, patch_artist=True, labels=categories, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#c0c0c0")
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)
    for whisker in box["whiskers"]:
        whisker.set_color("black")
        whisker.set_linewidth(0.9)
    for cap in box["caps"]:
        cap.set_color("black")
        cap.set_linewidth(0.9)
    for index, values in enumerate(grouped_values, start=1):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        ax.scatter(
            np.full(len(values), float(index)) + jitter,
            np.asarray(values, dtype=float),
            s=18,
            color="#c7c7c7",
            edgecolors="none",
            alpha=0.7,
            zorder=1,
        )
    ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xticklabels(categories, rotation=90)
    ax.grid(axis="y", color="#d9d9d9", linestyle="-", linewidth=0.6)
    return True


def write_single_metric_plots(
    per_bin: pd.DataFrame,
    group_summary: pd.DataFrame,
    output_base: Path,
    categories: list[str],
) -> list[Path]:
    plt = ensure_plotting()
    wrote = []
    rng = np.random.default_rng(42)
    for metric, title, scope in kofam_single_metric_specs():
        grouped_values = single_metric_grouped_values(per_bin, group_summary, metric, scope, categories)
        if not grouped_values or not any(len(values) for values in grouped_values):
            continue
        fig, ax = plt.subplots(figsize=(max(7.5, len(categories) * 1.45), 6.5))
        if not draw_single_metric_axis(ax, grouped_values, categories, title, rng=rng):
            plt.close(fig)
            continue
        finite_values = np.concatenate([values[np.isfinite(values)] for values in grouped_values if len(values)])
        if finite_values.size:
            top = float(np.nanmax(finite_values))
            ax.set_ylim(bottom=0, top=max(1.0, top * 1.08))
        else:
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        metric_base = output_base.parent / f"{output_base.name}_{metric}"
        save_figure(fig, str(metric_base))
        wrote.extend([Path(str(metric_base) + ".png"), Path(str(metric_base) + ".pdf")])
    return wrote


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    metadata_path = results_dir / "metadata.tsv"
    per_bin_path = results_dir / "SI_kofam.total_KO.per_bin.tsv"
    brite_path = results_dir / "SI_kofam.ko_to_brite.tsv"
    mapper_unique_path = results_dir / "all_kofam.mapper.unique.tsv"
    enrichment_path = results_dir / "kofam_ko.enrichment_results.compact.tsv"

    metadata = read_tsv(metadata_path)
    per_bin = read_tsv(per_bin_path)
    categories = normalize_category_order(metadata["category"])

    print("[start] resolving raw ORF counts from ORF FASTA files")
    raw_counts = []
    for row in metadata.to_dict("records"):
        annotation_source_path = row.get("mapper_path", "") or row.get("passed_path", "")
        orf_path = resolve_orf_annotation_path(annotation_source_path, results_dir)
        fasta_path = resolve_raw_orf_fasta_path(annotation_source_path, results_dir)
        raw_counts.append(
            {
                "bin_id": row["bin_id"],
                "orf_annotation_path": str(orf_path) if orf_path else "",
                "raw_orf_fasta_path": str(fasta_path) if fasta_path else "",
                "total_orfs": count_fasta_records(fasta_path),
            }
        )
    raw_counts_df = pd.DataFrame(raw_counts)

    print("[start] classifying informative KOs from BRITE annotations")
    informative_kos, ko_brite_summary = build_informative_ko_set(brite_path)

    print("[start] counting informative KO ORFs from unique mapper hits")
    informative_counts = count_informative_ko_orfs(mapper_unique_path, informative_kos)

    print("[start] counting enriched KOs by associated category")
    enriched_counts = build_enriched_ko_counts(enrichment_path, args.q_threshold, categories)

    print("[start] collecting enriched KO prevalence values")
    enriched_prevalence = build_enriched_ko_prevalence_table(enrichment_path, args.q_threshold, categories)

    per_bin = per_bin.merge(raw_counts_df, on="bin_id", how="left")
    per_bin = per_bin.merge(informative_counts, on="bin_id", how="left")
    per_bin["total_orfs"] = pd.to_numeric(per_bin["total_orfs"], errors="coerce").fillna(0).astype(int)
    per_bin["informative_ko_orfs"] = (
        pd.to_numeric(per_bin["informative_ko_orfs"], errors="coerce").fillna(0).astype(int)
    )
    per_bin["unique_gene_count"] = pd.to_numeric(per_bin["unique_gene_count"], errors="coerce").fillna(0).astype(int)
    per_bin["ko_annotation_fraction_percent"] = (
        per_bin["unique_gene_count"] / per_bin["total_orfs"].replace(0, np.nan)
    ) * 100.0
    group_summary = summarize_per_bin(per_bin, categories).merge(enriched_counts, on="category", how="left")
    long_df = long_group_metrics(group_summary)
    benchmark_values, benchmark_specs = build_benchmark_metric_values(
        per_bin,
        group_summary,
        enriched_prevalence,
        categories,
    )
    benchmark_stats = build_benchmark_stats(benchmark_values, benchmark_specs, categories)

    per_bin_out = results_dir / f"{args.prefix}.per_bin_metrics.tsv"
    group_out = results_dir / f"{args.prefix}.group_summary.tsv"
    long_out = results_dir / f"{args.prefix}.plotdata.tsv"
    ko_out = results_dir / f"{args.prefix}.ko_brite_informative_summary.tsv"
    enriched_prevalence_out = results_dir / f"{args.prefix}.enriched_ko_prevalence.tsv"
    benchmark_values_out = results_dir / f"{args.prefix}.benchmark_metric_plotdata.tsv"
    benchmark_stats_out = results_dir / f"{args.prefix}.benchmark_metric_stats.tsv"
    plot_base = results_dir / args.prefix
    benchmark_mean_plot_base = results_dir / f"{args.prefix}_benchmark_metric_panel_mean_ci95_sig"
    benchmark_median_plot_base = results_dir / f"{args.prefix}_benchmark_metric_panel_median_iqr_sig"

    per_bin.to_csv(per_bin_out, sep="\t", index=False)
    group_summary.to_csv(group_out, sep="\t", index=False)
    long_df.to_csv(long_out, sep="\t", index=False)
    ko_brite_summary.to_csv(ko_out, sep="\t", index=False)
    enriched_prevalence.to_csv(enriched_prevalence_out, sep="\t", index=False)
    benchmark_values.to_csv(benchmark_values_out, sep="\t", index=False)
    benchmark_stats.to_csv(benchmark_stats_out, sep="\t", index=False)
    plot_paths = write_single_metric_plots(per_bin, group_summary, plot_base, categories)
    plot_paths.extend(
        write_benchmark_metric_panel(
            benchmark_values,
            benchmark_specs,
            benchmark_stats,
            benchmark_mean_plot_base,
            categories,
            summary_mode="mean",
        )
    )
    plot_paths.extend(
        write_benchmark_metric_panel(
            benchmark_values,
            benchmark_specs,
            benchmark_stats,
            benchmark_median_plot_base,
            categories,
            summary_mode="median",
        )
    )

    print(per_bin_out)
    print(group_out)
    print(long_out)
    print(ko_out)
    print(enriched_prevalence_out)
    print(benchmark_values_out)
    print(benchmark_stats_out)
    for path in plot_paths:
        print(path)


if __name__ == "__main__":
    main()
