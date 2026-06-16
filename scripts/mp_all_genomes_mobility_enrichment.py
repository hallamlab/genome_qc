#!/usr/bin/env python3
"""Mobility annotation enrichment across all MP genomes by category."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_metapathways_wrapper import ensure_plotting, save_figure  # noqa: E402


DEFAULT_MP_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98")
DEFAULT_IN_DIR = DEFAULT_MP_DIR / "experimental_mobility" / "tables"
DEFAULT_OUT_DIR = DEFAULT_MP_DIR / "all_genomes_category_mobility_enrichment"
CATEGORIES = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]
CATEGORY_DISPLAY = {
    "SAGs": "SAGs",
    "xPG_SAGs": "SAG-xPGs",
    "MAGs": "MAGs",
    "xPG_MAGs": "MAG-xPGs",
}
MOBILITY_ORDER = [
    "plasmid_associated",
    "phage_prophage_associated",
    "generic_mge_associated",
    "genomic_island_associated",
]
MOBILITY_LABELS = {
    "plasmid_associated": "Plasmid",
    "phage_prophage_associated": "Phage/prophage",
    "generic_mge_associated": "Generic MGE",
    "genomic_island_associated": "Genomic island",
}
MOBILITY_SUBCATEGORY_SPECS = [
    ("plasmid_associated", "plasmid_replication", "Replication/maintenance", ["repa", "repb", "repc", "plasmid replication", "plasmid initiator"]),
    ("plasmid_associated", "plasmid_conjugation", "Conjugation/T4SS", ["relaxase", "type iv coupling protein", "t4cp", "vird4", "trb", "conjugation", "conjugal transfer", "mating pair formation", "mpf", "type iv secretion", "t4ss"]),
    ("plasmid_associated", "plasmid_partition_stability", "Partition/stability", ["partition protein", "plasmid partition", "stability protein"]),
    ("plasmid_associated", "plasmid_toxin_antitoxin", "Toxin-antitoxin", ["toxin antitoxin", "postsegregational killing"]),
    ("phage_prophage_associated", "phage_packaging", "Packaging/portal", ["terminase", "terminase large subunit", "terl", "portal protein", "dna packaging", "packaging atpase"]),
    ("phage_prophage_associated", "phage_capsid_head", "Capsid/head", ["major capsid", "capsid", "head protein", "prohead", "capsid maturation protease", "phage protease"]),
    ("phage_prophage_associated", "phage_tail_baseplate", "Tail/baseplate", ["tail protein", "tail fiber", "tail sheath", "tail tube", "baseplate", "tape measure protein"]),
    ("phage_prophage_associated", "phage_lysis", "Lysis", ["endolysin", "spanin"]),
    ("phage_prophage_associated", "phage_prophage_integration", "Prophage/integration", ["prophage", "phage integrase"]),
    ("generic_mge_associated", "mge_recombination_integration", "Integration/recombination", ["integrase", "site-specific recombinase", "tyrosine recombinase", "serine recombinase", "recombinase", "excisionase", "xis", "resolvase", "invertase"]),
    ("generic_mge_associated", "mge_transposition", "Transposition/IS", ["transposase", "insertion sequence", "is element"]),
    ("generic_mge_associated", "mge_conjugative_elements", "ICE/IME/conjugative elements", ["integrative conjugative element", "ice", "ime", "conjugative element", "mobilizable element", "conjugative transfer"]),
    ("generic_mge_associated", "mge_mobilization", "Mobilization/relaxosome", ["mobilization protein", "mobilization", "orit", "relaxosome", "type iv secretion", "t4ss"]),
    ("genomic_island_associated", "island_named", "Named islands", ["genomic island", "pathogenicity island", "pai", "symbiosis island", "metabolic island", "resistance island"]),
    ("genomic_island_associated", "island_integration", "Integration/excision", ["integrative element", "integrase", "recombinase", "excisionase", "direct repeat", "attl", "attr", "insertion hotspot", "integration hotspot"]),
    ("genomic_island_associated", "island_conjugation_cargo", "Conjugation/cargo", ["conjugation", "type iv secretion", "cargo gene cluster"]),
]
EVIDENCE_ORDER = ["candidate genome prevalence", "unique mobility-associated ORFs"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--input-dir", type=Path, default=None)
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


def mobility_metric_id(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("/", "_").replace("-", "_").replace(" ", "_")
    text = text.replace("mge", "mge")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def load_mobility_tables(input_dir: Path, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prevalence_path = input_dir / f"{prefix}_experimental_candidate_mobility_long.tsv"
    burden_path = input_dir / f"{prefix}_experimental_candidate_mobility_unique_orf_burden_per_genome.tsv"
    hits_path = input_dir / f"{prefix}_experimental_candidate_mobility_hits.tsv"
    if not prevalence_path.exists():
        raise FileNotFoundError(prevalence_path)
    if not burden_path.exists():
        raise FileNotFoundError(burden_path)
    if not hits_path.exists():
        raise FileNotFoundError(hits_path)
    prevalence = pd.read_csv(prevalence_path, sep="\t", low_memory=False)
    burden = pd.read_csv(burden_path, sep="\t", low_memory=False)
    hits = pd.read_csv(hits_path, sep="\t", low_memory=False)
    return prevalence, burden, hits


def build_long_table(prevalence: pd.DataFrame, burden: pd.DataFrame) -> pd.DataFrame:
    prev = prevalence.loc[prevalence["metric"].isin(MOBILITY_ORDER)].copy()
    prev["category"] = prev["category"].astype(str).str.strip()
    prev["evidence"] = "genome_prevalence"
    prev["evidence_class"] = "candidate genome prevalence"
    prev["module_id"] = prev["metric"].astype(str)
    prev["module_label"] = prev["metric_label"].astype(str)
    prev["value"] = pd.to_numeric(prev["mobility_positive"], errors="coerce").fillna(0.0)
    prev = prev[["sample", "category", "genome_id", "evidence", "evidence_class", "module_id", "module_label", "value"]]

    counts = burden.loc[burden["metric"].isin(MOBILITY_ORDER)].copy()
    counts["category"] = counts["category"].astype(str).str.strip()
    counts["evidence"] = "unique_orf_burden"
    counts["evidence_class"] = "unique mobility-associated ORFs"
    counts["module_id"] = counts["metric"].astype(str)
    counts["module_label"] = counts["metric_label"].astype(str)
    counts["value"] = pd.to_numeric(counts["unique_mobility_orfs"], errors="coerce").fillna(0.0)
    counts = counts[["sample", "category", "genome_id", "evidence", "evidence_class", "module_id", "module_label", "value"]]
    return pd.concat([prev, counts], ignore_index=True)


def build_subcategory_long_table(prevalence: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    base = (
        prevalence.loc[prevalence["metric"].isin(MOBILITY_ORDER), ["sample", "category", "genome_id"]]
        .drop_duplicates()
        .copy()
    )
    if base.empty:
        return pd.DataFrame()
    subcats = pd.DataFrame(
        [
            {
                "high_level_id": family_id,
                "high_level_label": MOBILITY_LABELS.get(family_id, family_id),
                "module_id": sub_id,
                "module_label": label,
            }
            for family_id, sub_id, label, _keywords in MOBILITY_SUBCATEGORY_SPECS
        ]
    )
    universe = base.merge(subcats, how="cross")

    keyword_lookup = {}
    for family_id, sub_id, label, keywords in MOBILITY_SUBCATEGORY_SPECS:
        for keyword in keywords:
            keyword_lookup[(family_id, str(keyword).strip().lower())] = (sub_id, label)

    rows = []
    if hits is not None and not hits.empty:
        for record in hits.to_dict("records"):
            family_id = mobility_metric_id(record.get("mobility_category", ""))
            if family_id not in MOBILITY_ORDER:
                continue
            keywords = [
                token.strip().lower()
                for token in str(record.get("matched_keywords", "")).split(";")
                if token.strip()
            ]
            for keyword in keywords:
                subcategory = keyword_lookup.get((family_id, keyword))
                if subcategory is None:
                    continue
                sub_id, label = subcategory
                rows.append(
                    {
                        "sample": str(record.get("sample", "")).strip(),
                        "category": str(record.get("category", "")).strip(),
                        "genome_id": str(record.get("genome_id", "")).strip(),
                        "orf_id": str(record.get("orf_id", "")).strip(),
                        "high_level_id": family_id,
                        "high_level_label": MOBILITY_LABELS.get(family_id, family_id),
                        "module_id": sub_id,
                        "module_label": label,
                    }
                )
    hit_subcats = pd.DataFrame(rows)
    if hit_subcats.empty:
        hit_subcats = pd.DataFrame(columns=["sample", "category", "genome_id", "orf_id", "module_id"])

    positive = (
        hit_subcats.drop_duplicates(["sample", "category", "genome_id", "module_id"])
        .loc[:, ["sample", "category", "genome_id", "module_id"]]
        .assign(present=1.0)
    )
    counts = (
        hit_subcats.loc[hit_subcats["orf_id"].astype(str).str.strip().ne("")]
        .drop_duplicates(["sample", "category", "genome_id", "module_id", "orf_id"])
        .groupby(["sample", "category", "genome_id", "module_id"], dropna=False)
        .size()
        .reset_index(name="unique_orfs")
    )

    prev = universe.merge(positive, on=["sample", "category", "genome_id", "module_id"], how="left")
    prev["evidence"] = "genome_prevalence"
    prev["evidence_class"] = "candidate genome prevalence"
    prev["value"] = pd.to_numeric(prev["present"], errors="coerce").fillna(0.0)
    prev = prev.drop(columns=["present"])

    burden = universe.merge(counts, on=["sample", "category", "genome_id", "module_id"], how="left")
    burden["evidence"] = "unique_orf_burden"
    burden["evidence_class"] = "unique mobility-associated ORFs"
    burden["value"] = pd.to_numeric(burden["unique_orfs"], errors="coerce").fillna(0.0)
    burden = burden.drop(columns=["unique_orfs"])

    columns = [
        "sample",
        "category",
        "genome_id",
        "evidence",
        "evidence_class",
        "high_level_id",
        "high_level_label",
        "module_id",
        "module_label",
        "value",
    ]
    return pd.concat([prev[columns], burden[columns]], ignore_index=True)


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
                "high_level_id": str(subset["high_level_id"].iloc[0]) if "high_level_id" in subset.columns else "",
                "high_level_label": str(subset["high_level_label"].iloc[0]) if "high_level_label" in subset.columns else "",
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
            for cat_b in CATEGORIES[i + 1:]:
                try:
                    pair_stat, pair_p = stats.mannwhitneyu(grouped[cat_b], grouped[cat_a], alternative="two-sided")
                except Exception:
                    pair_stat, pair_p = np.nan, np.nan
                pair_rows.append(
                    {
                        "evidence": evidence,
                        "evidence_class": evidence_class,
                        "high_level_id": str(subset["high_level_id"].iloc[0]) if "high_level_id" in subset.columns else "",
                        "high_level_label": str(subset["high_level_label"].iloc[0]) if "high_level_label" in subset.columns else "",
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
                "high_level_id": row.get("high_level_id", ""),
                "high_level_label": row.get("high_level_label", ""),
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
        enriched_df["module_rank"] = enriched_df["module_id"].map({item: index for index, item in enumerate(MOBILITY_ORDER)})
        enriched_df["evidence_rank"] = enriched_df["evidence_class"].map({item: index for index, item in enumerate(EVIDENCE_ORDER)})
        enriched_df = (
            enriched_df.sort_values(["evidence_rank", "module_rank", "enriched_category"], kind="mergesort")
            .drop(columns=["module_rank", "evidence_rank"])
            .reset_index(drop=True)
        )
    return summary_df, pair_df, enriched_df


def plot_enriched_mobility_subcategory_means(enriched_df: pd.DataFrame, output_base: Path) -> bool:
    if enriched_df.empty:
        return False
    plt = ensure_plotting()
    categories = [
        ("mean_SAGs", "SAGs"),
        ("mean_SAG_xPGs", "SAG-xPGs"),
        ("mean_MAGs", "MAGs"),
        ("mean_MAG_xPGs", "MAG-xPGs"),
    ]
    family_rank = {family_id: index for index, family_id in enumerate(MOBILITY_ORDER)}
    sub_rank = {sub_id: index for index, (_family, sub_id, _label, _keywords) in enumerate(MOBILITY_SUBCATEGORY_SPECS)}
    enriched_df = enriched_df.copy()
    enriched_df["high_level_rank"] = enriched_df["high_level_id"].map(family_rank).fillna(999).astype(int)
    enriched_df["module_rank"] = enriched_df["module_id"].map(sub_rank).fillna(999).astype(int)
    fig, axes = plt.subplots(
        1,
        len(EVIDENCE_ORDER),
        figsize=(13.6, max(6.2, 0.34 * enriched_df.groupby("evidence_class").size().max() + 2.1)),
        squeeze=False,
        sharex=False,
    )
    axes = axes.ravel()
    wrote = False
    for ax, evidence_class in zip(axes, EVIDENCE_ORDER):
        sub = enriched_df.loc[enriched_df["evidence_class"].eq(evidence_class)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values(["high_level_rank", "module_rank", "module_label", "enriched_category"], kind="mergesort").reset_index(drop=True)
        y = np.arange(len(sub))
        all_values = []
        for col, _label in categories:
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
            .groupby(["high_level_id", "high_level_label", "high_level_rank"], sort=False)["index"]
            .agg(["min", "max"])
            .reset_index()
        )
        for _, block in family_blocks.iterrows():
            family_label = str(block["high_level_label"]).strip()
            if not family_label:
                continue
            center = (float(block["min"]) + float(block["max"])) / 2.0
            ax.text(
                -0.48,
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
        ax.set_xlim(x_positions[0] - 0.72, x_positions[-1] + 0.20)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["module_label"].astype(str), fontsize=7.4)
        ax.tick_params(axis="y", pad=10)
        ax.set_title(evidence_class)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        wrote = True
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:4], labels[:4], loc="center right", frameon=False, title="Category")
    fig.suptitle("Category mean support for enriched candidate mobility subcategories across all _98 MetaPathways genomes", y=0.995)
    fig.text(
        0.5,
        0.014,
        "Rows are keyword subcategories grouped by high-level mobility class; dot size reflects category mean support and white-centered dots mark enriched categories.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.05, 0.06, 0.89, 0.93], w_pad=1.2)
    if not wrote:
        plt.close(fig)
        return False
    save_figure(fig, str(output_base))
    return True


def plot_enriched_mobility_means(enriched_df: pd.DataFrame, output_base: Path) -> bool:
    if enriched_df.empty:
        return False
    plt = ensure_plotting()
    categories = [
        ("mean_SAGs", "SAGs"),
        ("mean_SAG_xPGs", "SAG-xPGs"),
        ("mean_MAGs", "MAGs"),
        ("mean_MAG_xPGs", "MAG-xPGs"),
    ]
    module_rank = {module: index for index, module in enumerate(MOBILITY_ORDER)}
    enriched_df = enriched_df.copy()
    enriched_df["module_rank"] = enriched_df["module_id"].map(module_rank).fillna(999).astype(int)

    fig, axes = plt.subplots(
        1,
        len(EVIDENCE_ORDER),
        figsize=(11.2, 4.8),
        squeeze=False,
        sharex=False,
    )
    axes = axes.ravel()
    wrote = False
    for ax, evidence_class in zip(axes, EVIDENCE_ORDER):
        sub = enriched_df.loc[enriched_df["evidence_class"].eq(evidence_class)].copy()
        if sub.empty:
            ax.axis("off")
            continue
        sub = sub.sort_values(["module_rank", "module_label", "enriched_category"], kind="mergesort").reset_index(drop=True)
        y = np.arange(len(sub))
        all_values = []
        for col, _label in categories:
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
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, label in categories], rotation=45, ha="right")
        ax.set_xlim(x_positions[0] - 0.42, x_positions[-1] + 0.20)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["module_label"].astype(str), fontsize=8)
        ax.set_title(evidence_class)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        wrote = True
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:4], labels[:4], loc="center right", frameon=False, title="Category")
    fig.suptitle("Category mean support for enriched candidate mobility annotations across all _98 MetaPathways genomes", y=0.995)
    fig.text(
        0.5,
        0.014,
        "Dot size reflects category mean support; white-centered dot marks the top-mean category with BH-significant pairwise enrichment.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.04, 0.06, 0.88, 0.92], w_pad=1.2)
    if not wrote:
        plt.close(fig)
        return False
    save_figure(fig, str(output_base))
    return True


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir if args.input_dir is not None else args.mp_dir / "experimental_mobility" / "tables"
    if args.purge and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    table_dir = args.out_dir / "tables"
    plot_dir = args.out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    prevalence, burden, hits = load_mobility_tables(input_dir, args.prefix)
    long_df = build_long_table(prevalence, burden)
    summary_df, pair_df, enriched_df = enrichment_tables(long_df)
    subcategory_long_df = build_subcategory_long_table(prevalence, hits)
    subcategory_summary_df, subcategory_pair_df, subcategory_enriched_df = enrichment_tables(subcategory_long_df)
    long_df.to_csv(table_dir / "all_genomes_mobility_values_long.tsv", sep="\t", index=False)
    summary_df.to_csv(table_dir / "all_genomes_mobility_category_summary.tsv", sep="\t", index=False)
    pair_df.to_csv(table_dir / "all_genomes_mobility_pairwise_stats.tsv", sep="\t", index=False)
    enriched_df.to_csv(table_dir / "all_genomes_mobility_category_enrichment_summary.tsv", sep="\t", index=False)
    subcategory_long_df.to_csv(table_dir / "all_genomes_mobility_subcategory_values_long.tsv", sep="\t", index=False)
    subcategory_summary_df.to_csv(table_dir / "all_genomes_mobility_subcategory_summary.tsv", sep="\t", index=False)
    subcategory_pair_df.to_csv(table_dir / "all_genomes_mobility_subcategory_pairwise_stats.tsv", sep="\t", index=False)
    subcategory_enriched_df.to_csv(table_dir / "all_genomes_mobility_subcategory_enrichment_summary.tsv", sep="\t", index=False)
    plot_enriched_mobility_means(enriched_df, plot_dir / "all_genomes_enriched_mobility_category_means")
    plot_enriched_mobility_subcategory_means(
        subcategory_enriched_df,
        plot_dir / "all_genomes_enriched_mobility_subcategory_means",
    )

    genome_rows = long_df.loc[long_df["evidence"].eq("genome_prevalence"), ["sample", "category", "genome_id"]].drop_duplicates()
    print(f"genomes={len(genome_rows)}")
    print(genome_rows["category"].value_counts().reindex(CATEGORIES).to_string())
    print(f"mobility_features_tested={summary_df.shape[0]}")
    print(f"enriched_mobility_features={enriched_df.shape[0]}")
    print(f"mobility_subcategories_tested={subcategory_summary_df.shape[0]}")
    print(f"enriched_mobility_subcategories={subcategory_enriched_df.shape[0]}")
    print(table_dir / "all_genomes_mobility_category_enrichment_summary.tsv")
    print(table_dir / "all_genomes_mobility_subcategory_enrichment_summary.tsv")
    print(plot_dir / "all_genomes_enriched_mobility_category_means.png")
    print(plot_dir / "all_genomes_enriched_mobility_subcategory_means.png")


if __name__ == "__main__":
    main()
