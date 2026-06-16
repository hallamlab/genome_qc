#!/usr/bin/env python3
"""Module enrichment across all MP genomes by category."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_metapathways_wrapper import (  # noqa: E402
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_ORDER,
    ensure_plotting,
    sanitize_label,
    save_figure,
)
from summarize_metapathways_genomes import (  # noqa: E402
    ELEMENTAL_CYCLE_LABELS,
    ELEMENTAL_CYCLE_ORDER,
    ELEMENTAL_MODE_FAMILY,
)


DEFAULT_MP_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98")
DEFAULT_OUT_DIR = DEFAULT_MP_DIR / "all_genomes_category_module_enrichment"
CATEGORIES = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]
CATEGORY_DISPLAY = {
    "SAGs": "SAGs",
    "xPG_SAGs": "SAG-xPGs",
    "MAGs": "MAGs",
    "xPG_MAGs": "MAG-xPGs",
}
EVIDENCE_SPECS = [
    ("marker_specific", "specific marker genes", "marker", "specific_gene_count"),
    ("marker_generic", "generic marker genes", "marker", "generic_gene_count"),
    ("reference_mode", "reference-mode accessions", "reference_mode", "accession_count"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="metapathways_batch")
    parser.add_argument("--purge", action="store_true")
    return parser.parse_args()


def bh_adjust(pvalues: list[float]) -> list[float]:
    arr = np.asarray(pvalues, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if not valid.any():
        return out.tolist()
    vals = arr[valid]
    order = np.argsort(vals)
    adjusted = np.empty_like(vals)
    running = 1.0
    n = float(vals.size)
    for pos in range(vals.size - 1, -1, -1):
        idx = order[pos]
        rank = pos + 1
        running = min(running, vals[idx] * n / rank)
        adjusted[idx] = running
    out[valid] = adjusted
    return out.tolist()


def load_genomes(mp_dir: Path, prefix: str) -> pd.DataFrame:
    path = mp_dir / "tables" / "summary" / f"{sanitize_label(prefix)}_genome_summary.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["category"] = frame["category"].astype(str).str.strip()
    return frame


def build_long_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evidence_id, evidence_label, prefix, suffix in EVIDENCE_SPECS:
        for mode_id in ELEMENTAL_MODE_ORDER:
            column = f"{prefix}_{mode_id}_{suffix}"
            if column not in frame.columns:
                continue
            temp = frame[["genome_label", "sample", "category", "genome_id"]].copy()
            temp["evidence"] = evidence_id
            temp["evidence_class"] = evidence_label
            temp["module_id"] = mode_id
            temp["module_label"] = ELEMENTAL_MODE_LABELS.get(mode_id, mode_id)
            temp["value"] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            rows.append(temp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def enrichment_tables(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pair_rows = []
    enriched_rows = []
    for (evidence, module_id), subset in long_df.groupby(["evidence", "module_id"], dropna=False):
        evidence_class = str(subset["evidence_class"].iloc[0])
        module_label = str(subset["module_label"].iloc[0])
        grouped = {
            category: subset.loc[subset["category"].eq(category), "value"].astype(float).to_numpy()
            for category in CATEGORIES
        }
        if not all(values.size for values in grouped.values()):
            continue
        means = {category: float(np.nanmean(values)) for category, values in grouped.items()}
        medians = {category: float(np.nanmedian(values)) for category, values in grouped.items()}
        try:
            statistic, pvalue = stats.kruskal(*[grouped[category] for category in CATEGORIES])
        except Exception:
            statistic, pvalue = np.nan, np.nan
        summary_rows.append(
            {
                "evidence": evidence,
                "evidence_class": evidence_class,
                "module_id": module_id,
                "module_label": module_label,
                "kruskal_statistic": statistic,
                "kruskal_pvalue": pvalue,
                **{f"mean_{category}": means[category] for category in CATEGORIES},
                **{f"median_{category}": medians[category] for category in CATEGORIES},
                **{f"n_{category}": int(grouped[category].size) for category in CATEGORIES},
            }
        )
        for i, cat_a in enumerate(CATEGORIES):
            for cat_b in CATEGORIES[i + 1 :]:
                try:
                    pair_stat, pair_p = stats.mannwhitneyu(grouped[cat_b], grouped[cat_a], alternative="two-sided")
                except Exception:
                    pair_stat, pair_p = np.nan, np.nan
                pair_rows.append(
                    {
                        "evidence": evidence,
                        "evidence_class": evidence_class,
                        "module_id": module_id,
                        "module_label": module_label,
                        "contrast": f"{cat_b}_minus_{cat_a}",
                        "category_a": cat_a,
                        "category_b": cat_b,
                        "mean_difference_b_minus_a": means[cat_b] - means[cat_a],
                        "median_difference_b_minus_a": medians[cat_b] - medians[cat_a],
                        "mannwhitney_statistic": pair_stat,
                        "pvalue": pair_p,
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    pair_df = pd.DataFrame(pair_rows)
    if not summary_df.empty:
        summary_df["kruskal_qvalue_bh"] = bh_adjust(summary_df["kruskal_pvalue"].tolist())
    if not pair_df.empty:
        pair_df["qvalue_bh"] = bh_adjust(pair_df["pvalue"].tolist())

    for row in summary_df.to_dict("records"):
        if not np.isfinite(row.get("kruskal_qvalue_bh", np.nan)) or float(row["kruskal_qvalue_bh"]) >= 0.05:
            continue
        means = {category: row.get(f"mean_{category}", np.nan) for category in CATEGORIES}
        top = max(CATEGORIES, key=lambda category: means.get(category, -np.inf))
        subset = pair_df.loc[
            pair_df["evidence"].eq(row["evidence"])
            & pair_df["module_id"].eq(row["module_id"])
            & pair_df["qvalue_bh"].lt(0.05)
        ].copy()
        sig_vs = []
        for pair in subset.to_dict("records"):
            if pair["category_b"] == top and pair["mean_difference_b_minus_a"] > 0:
                sig_vs.append(CATEGORY_DISPLAY[pair["category_a"]])
            elif pair["category_a"] == top and pair["mean_difference_b_minus_a"] < 0:
                sig_vs.append(CATEGORY_DISPLAY[pair["category_b"]])
        if not sig_vs:
            continue
        enriched_rows.append(
            {
                "evidence_class": row["evidence_class"],
                "module_id": row["module_id"],
                "module_label": row["module_label"],
                "enriched_category": CATEGORY_DISPLAY[top],
                "kruskal_pvalue": row["kruskal_pvalue"],
                "kruskal_qvalue_bh": row["kruskal_qvalue_bh"],
                "mean_SAGs": means["SAGs"],
                "mean_SAG_xPGs": means["xPG_SAGs"],
                "mean_MAGs": means["MAGs"],
                "mean_MAG_xPGs": means["xPG_MAGs"],
                "significant_vs_q05": "; ".join(sorted(set(sig_vs))),
            }
        )
    enriched_df = pd.DataFrame(enriched_rows)
    if not enriched_df.empty:
        enriched_df = enriched_df.sort_values(
            ["evidence_class", "enriched_category", "module_label"],
            kind="mergesort",
        )
    return summary_df, pair_df, enriched_df


def plot_enrichment_counts(enriched_df: pd.DataFrame, output_base: Path) -> bool:
    if enriched_df.empty:
        return False
    plt = ensure_plotting()
    evidence_order = ["specific marker genes", "generic marker genes", "reference-mode accessions"]
    category_order = ["SAGs", "SAG-xPGs", "MAGs", "MAG-xPGs"]
    matrix = (
        enriched_df.pivot_table(
            index="evidence_class",
            columns="enriched_category",
            values="module_id",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(index=evidence_order, columns=category_order, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    vmax = max(1, int(matrix.to_numpy().max()))
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Enriched functional modules across all _98 MetaPathways genomes")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iat[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value / vmax > 0.55 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Enriched modules")
    fig.tight_layout()
    save_figure(fig, str(output_base))
    return True


def plot_enriched_module_means(enriched_df: pd.DataFrame, output_base: Path) -> bool:
    if enriched_df.empty:
        return False
    plt = ensure_plotting()
    categories = [
        ("mean_SAGs", "SAGs"),
        ("mean_SAG_xPGs", "SAG-xPGs"),
        ("mean_MAGs", "MAGs"),
        ("mean_MAG_xPGs", "MAG-xPGs"),
    ]
    family_rank = {family: index for index, family in enumerate(ELEMENTAL_CYCLE_ORDER)}
    mode_rank = {mode: index for index, mode in enumerate(ELEMENTAL_MODE_ORDER)}
    enriched_df = enriched_df.copy()
    enriched_df["family_id"] = enriched_df["module_id"].map(ELEMENTAL_MODE_FAMILY).fillna("")
    enriched_df["family_label"] = enriched_df["family_id"].map(ELEMENTAL_CYCLE_LABELS).fillna("")
    enriched_df["family_rank"] = enriched_df["family_id"].map(family_rank).fillna(999).astype(int)
    enriched_df["mode_rank"] = enriched_df["module_id"].map(mode_rank).fillna(999).astype(int)
    evidence_order = ["specific marker genes", "generic marker genes", "reference-mode accessions"]
    fig, axes = plt.subplots(
        1,
        len(evidence_order),
        figsize=(15.5, max(7.0, 0.34 * enriched_df.groupby("evidence_class").size().max() + 2.3)),
        squeeze=False,
        sharex=False,
    )
    axes = axes.ravel()
    wrote = False
    for ax, evidence_class in zip(axes, evidence_order):
        sub = enriched_df.loc[enriched_df["evidence_class"].eq(evidence_class)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values(["family_rank", "mode_rank", "module_label", "enriched_category"], kind="mergesort").reset_index(drop=True)
        y = np.arange(len(sub))
        all_values = []
        for col, label in categories:
            all_values.extend(pd.to_numeric(sub[col], errors="coerce").fillna(0).tolist())
        vmax = max(1.0, float(np.nanmax(all_values)))
        x_positions = np.array([0.0, 0.42, 0.84, 1.26])
        for x, (col, label) in zip(x_positions, categories):
            values = pd.to_numeric(sub[col], errors="coerce").fillna(0.0).to_numpy()
            sizes = 25 + 185 * (values / vmax)
            ax.scatter(
                np.full(len(sub), x),
                y,
                s=sizes,
                color="#2f2f2f",
                edgecolor="black",
                linewidth=0.35,
                alpha=0.88,
                label=label,
                zorder=3,
            )
        enriched_x = {label: x for x, (_, label) in zip(x_positions, categories)}
        for row_index, category in enumerate(sub["enriched_category"].astype(str)):
            ax.scatter(
                enriched_x.get(category, 0),
                row_index,
                s=18,
                color="white",
                edgecolor="black",
                linewidth=0.65,
                zorder=4,
            )
        family_blocks = (
            sub.reset_index()
            .groupby(["family_id", "family_label", "family_rank"], sort=False)["index"]
            .agg(["min", "max"])
            .reset_index()
        )
        for _, block in family_blocks.iterrows():
            family_label = str(block["family_label"]).strip()
            if not family_label:
                continue
            center = (float(block["min"]) + float(block["max"])) / 2.0
            ax.text(
                -0.42,
                center,
                family_label,
                ha="center",
                va="center",
                fontsize=7.2,
                fontweight="bold",
                rotation=45,
                clip_on=False,
            )
            ax.axhline(float(block["max"]) + 0.5, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, label in categories], rotation=45, ha="right")
        ax.set_xlim(x_positions[0] - 0.68, x_positions[-1] + 0.20)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["module_label"].astype(str), fontsize=7)
        ax.tick_params(axis="y", pad=10)
        ax.set_title(evidence_class)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        wrote = True
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:4], labels[:4], loc="center right", frameon=False, title="Category")
    fig.suptitle("Category mean support for enriched modules across all _98 MetaPathways genomes", y=0.995)
    fig.text(
        0.5,
        0.012,
        "Dot size reflects category mean support; white-centered dot marks the top-mean category with BH-significant pairwise enrichment.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.04, 0.04, 0.90, 0.94], w_pad=1.1)
    if not wrote:
        plt.close(fig)
        return False
    save_figure(fig, str(output_base))
    return True


def main() -> None:
    args = parse_args()
    if args.purge and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    table_dir = args.out_dir / "tables"
    plot_dir = args.out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = load_genomes(args.mp_dir, args.prefix)
    long_df = build_long_table(frame)
    summary_df, pair_df, enriched_df = enrichment_tables(long_df)
    long_df.to_csv(table_dir / "all_genomes_module_values_long.tsv", sep="\t", index=False)
    summary_df.to_csv(table_dir / "all_genomes_module_category_summary.tsv", sep="\t", index=False)
    pair_df.to_csv(table_dir / "all_genomes_module_pairwise_stats.tsv", sep="\t", index=False)
    enriched_df.to_csv(table_dir / "all_genomes_module_category_enrichment_summary.tsv", sep="\t", index=False)
    plot_enrichment_counts(enriched_df, plot_dir / "all_genomes_module_enrichment_counts")
    plot_enriched_module_means(enriched_df, plot_dir / "all_genomes_enriched_module_category_means")
    print(f"genomes={len(frame)}")
    print(frame["category"].value_counts().reindex(CATEGORIES).to_string())
    print(f"modules_tested={summary_df.shape[0]}")
    print(f"enriched_modules={enriched_df.shape[0]}")
    print(table_dir / "all_genomes_module_category_enrichment_summary.tsv")
    print(plot_dir / "all_genomes_module_enrichment_counts.png")
    print(plot_dir / "all_genomes_enriched_module_category_means.png")


if __name__ == "__main__":
    main()
