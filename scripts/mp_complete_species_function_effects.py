#!/usr/bin/env python3
"""Species-blocked functional category effect plots for complete MP species sets."""

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
    ELEMENTAL_MODE_ORDER,
    ensure_plotting,
    matching_id_aliases,
    sanitize_label,
    save_figure,
)


DEFAULT_MP_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98")
DEFAULT_COMPONENT_MEMBERS = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_genome_atlas_gunc_98/"
    "tables/components/best_sets_review_component_members.tsv"
)
DEFAULT_OUT_DIR = DEFAULT_MP_DIR / "complete_species_category_function_effects"
REQUIRED_CATEGORIES = ["SAGs", "xPGs_SAGs", "MAGs", "xPGs_MAGs"]
DISPLAY_CATEGORIES = ["SAGs", "SAG-xPGs", "MAGs", "MAG-xPGs"]
CATEGORY_DISPLAY = {
    "MAGs": "MAGs",
    "xPGs_MAGs": "MAG-xPGs",
    "SAGs": "SAGs",
    "xPGs_SAGs": "SAG-xPGs",
}
MP_CATEGORY_MAP = {"xPG_MAGs": "xPGs_MAGs", "xPG_SAGs": "xPGs_SAGs"}
TIER_RANK = {"high": 3, "medium": 2, "low": 1, "failed": 0, "": -1}
EVIDENCE_SPECS = [
    ("marker_specific", "Specific marker genes", "marker", "specific_gene_count"),
    ("marker_generic", "Generic marker genes", "marker", "generic_gene_count"),
    ("reference_mode", "Reference accessions", "reference_mode", "accession_count"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--component-members", type=Path, default=DEFAULT_COMPONENT_MEMBERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="metapathways_batch")
    parser.add_argument("--require-high", action="store_true", help="Require at least one strict high genome per species.")
    parser.add_argument("--purge", action="store_true")
    return parser.parse_args()


def normalize_category(value: object) -> str:
    text = str(value).strip()
    return MP_CATEGORY_MAP.get(text, text)


def clean_taxon(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "na", "unknown", "unclassified"}:
        return ""
    return text


def load_mp_genomes(mp_dir: Path, prefix: str) -> pd.DataFrame:
    path = mp_dir / "tables" / "summary" / f"{sanitize_label(prefix)}_genome_summary.tsv"
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    frame["category"] = frame["category"].map(normalize_category)
    frame["sample"] = frame["sample"].astype(str).str.strip()
    frame["genome_id"] = frame["genome_id"].astype(str).str.strip()
    return frame


def load_complete_species(path: Path, require_high: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    usecols = [
        "best_set_scope",
        "best_set_name",
        "component_id",
        "category",
        "sample",
        "Genome_Id",
        "Bin Id",
        "pre_ani_bin_key",
        "Completeness",
        "Contamination",
        "qscore",
        "mimag_tier",
        "Family",
        "Genus",
        "Species",
    ]
    frame = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    frame = frame.loc[
        frame["best_set_scope"].astype(str).eq("global")
        & frame["best_set_name"].astype(str).eq("best_of_best")
    ].copy()
    frame["category"] = frame["category"].map(normalize_category)
    frame["sample"] = frame["sample"].astype(str).str.strip()
    frame["Genome_Id"] = frame["Genome_Id"].astype(str).str.strip()
    for column in ["Family", "Genus", "Species"]:
        frame[column] = frame[column].map(clean_taxon)
    frame = frame.loc[frame["Species"].ne("")].copy()
    species_summary = (
        frame.groupby("Species")
        .agg(
            n_genomes=("Genome_Id", "size"),
            categories=("category", lambda x: set(x.astype(str))),
            has_high=("mimag_tier", lambda x: any(str(value).strip().lower() == "high" for value in x)),
            family=("Family", lambda x: "; ".join(sorted({clean_taxon(v) for v in x if clean_taxon(v)}))),
            genus=("Genus", lambda x: "; ".join(sorted({clean_taxon(v) for v in x if clean_taxon(v)}))),
        )
        .reset_index()
    )
    keep = species_summary["categories"].map(lambda values: set(REQUIRED_CATEGORIES).issubset(values))
    if require_high:
        keep = keep & species_summary["has_high"].fillna(False)
    species_summary = species_summary.loc[keep].copy()
    complete = frame.loc[frame["Species"].isin(set(species_summary["Species"]))].copy()
    complete = complete.rename(columns={"component_id": "ani_component_id"})
    complete["component_id"] = complete["Species"].astype(str)
    return complete, species_summary.drop(columns=["categories"])


def merge_mp_function_data(members: pd.DataFrame, mp_genomes: pd.DataFrame) -> pd.DataFrame:
    mp_working = mp_genomes.reset_index(drop=False).rename(columns={"index": "_mp_index"}).copy()
    alias_map = {}
    for row in mp_working[["_mp_index", "sample", "category", "genome_id"]].to_dict("records"):
        group_key = (str(row["sample"]).strip(), normalize_category(row["category"]))
        for alias in matching_id_aliases(row["genome_id"]):
            alias_map.setdefault((group_key, alias), set()).add(int(row["_mp_index"]))
    matched = []
    for row in members.to_dict("records"):
        group_key = (str(row.get("sample", "")).strip(), normalize_category(row.get("category", "")))
        aliases = set()
        for column in ["Genome_Id", "Bin Id", "pre_ani_bin_key"]:
            aliases.update(matching_id_aliases(row.get(column, "")))
        candidates = set()
        for alias in aliases:
            candidates.update(alias_map.get((group_key, alias), set()))
        matched.append(sorted(candidates)[0] if candidates else np.nan)
    members = members.copy()
    members["_matched_mp_index"] = matched
    missing = int(members["_matched_mp_index"].isna().sum())
    if missing:
        raise ValueError(f"{missing} genomes could not be matched to MP genome summary")
    members["_matched_mp_index"] = members["_matched_mp_index"].astype(int)
    merged = members.merge(
        mp_working.drop(columns=["sample", "category"]),
        left_on="_matched_mp_index",
        right_on="_mp_index",
        how="left",
        suffixes=("", "_mp"),
    )
    for column in ["Completeness", "Contamination", "qscore"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["tier_rank"] = merged["mimag_tier"].astype(str).str.lower().map(TIER_RANK).fillna(-1).astype(float)
    return merged


def representative_rows(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["component_id", "category", "tier_rank", "qscore", "Completeness", "Contamination", "genome_id"],
        ascending=[True, True, False, False, False, True, True],
        kind="mergesort",
    )
    return ranked.groupby(["component_id", "category"], as_index=False, sort=False).head(1).reset_index(drop=True)


def evidence_total(frame: pd.DataFrame, evidence_id: str, value_prefix: str, value_suffix: str) -> pd.Series:
    columns = [f"{value_prefix}_{mode_id}_{value_suffix}" for mode_id in ELEMENTAL_MODE_ORDER]
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)


def build_effect_tables(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = {
        "representative": representative_rows(merged),
        "average": merged.copy(),
    }
    rows = []
    for selection, frame in selections.items():
        for evidence_id, evidence_label, prefix, suffix in EVIDENCE_SPECS:
            working = frame[["component_id", "Family", "Genus", "Species", "category", "Genome_Id", "mimag_tier"]].copy()
            working["value"] = evidence_total(frame, evidence_id, prefix, suffix)
            summary = (
                working.groupby(["component_id", "Family", "Genus", "Species", "category"], dropna=False)
                .agg(
                    value=("value", "mean"),
                    n_genomes=("Genome_Id", "nunique"),
                    strict_mimag_tiers=("mimag_tier", lambda x: ";".join(sorted(set(map(str, x))))),
                )
                .reset_index()
            )
            summary["selection"] = selection
            summary["evidence"] = evidence_id
            summary["evidence_label"] = evidence_label
            rows.append(summary)
    values = pd.concat(rows, ignore_index=True)
    values["category_display"] = values["category"].map(CATEGORY_DISPLAY).fillna(values["category"])
    values["species_mean"] = values.groupby(["selection", "evidence", "component_id"])["value"].transform("mean")
    values["species_centered_value"] = values["value"] - values["species_mean"]

    stats_rows = []
    for (selection, evidence), subset in values.groupby(["selection", "evidence"], dropna=False):
        wide = subset.pivot_table(index="component_id", columns="category", values="value")
        wide = wide.reindex(columns=REQUIRED_CATEGORIES).dropna()
        if wide.empty:
            continue
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*[wide[col].to_numpy() for col in REQUIRED_CATEGORIES])
        except Exception:
            friedman_stat, friedman_p = np.nan, np.nan
        stats_rows.append(
            {
                "selection": selection,
                "evidence": evidence,
                "contrast": "global_category_effect",
                "n_species": int(wide.shape[0]),
                "statistic": friedman_stat,
                "pvalue": friedman_p,
                "mean_difference": np.nan,
                "median_difference": np.nan,
            }
        )
        for cat_a in REQUIRED_CATEGORIES:
            for cat_b in REQUIRED_CATEGORIES:
                if REQUIRED_CATEGORIES.index(cat_b) <= REQUIRED_CATEGORIES.index(cat_a):
                    continue
                diff = wide[cat_b] - wide[cat_a]
                try:
                    pvalue = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else np.nan
                except Exception:
                    pvalue = np.nan
                stats_rows.append(
                    {
                        "selection": selection,
                        "evidence": evidence,
                        "contrast": f"{cat_b}_minus_{cat_a}",
                        "n_species": int(diff.shape[0]),
                        "statistic": np.nan,
                        "pvalue": pvalue,
                        "mean_difference": float(diff.mean()),
                        "median_difference": float(diff.median()),
                    }
                )
    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        stats_df["qvalue_bh"] = np.nan
        mask = stats_df["contrast"].ne("global_category_effect") & stats_df["pvalue"].notna()
        pvals = stats_df.loc[mask, "pvalue"].astype(float).to_numpy()
        if pvals.size:
            order = np.argsort(pvals)
            adjusted = np.empty_like(pvals)
            running = 1.0
            n = float(pvals.size)
            for rank_index in range(pvals.size - 1, -1, -1):
                idx = order[rank_index]
                rank = rank_index + 1
                running = min(running, pvals[idx] * n / rank)
                adjusted[idx] = running
            stats_df.loc[mask, "qvalue_bh"] = adjusted
    return values, stats_df


def ci95(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(arr.mean())
    delta = 0.0 if arr.size < 2 else 1.96 * float(arr.std(ddof=1)) / np.sqrt(float(arr.size))
    return mean, mean - delta, mean + delta


def significance_label(pvalue: object) -> str:
    try:
        p = float(pvalue)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def contrast_categories(contrast: str) -> tuple[str, str] | None:
    if "_minus_" not in str(contrast):
        return None
    right, left = str(contrast).split("_minus_", 1)
    if left not in REQUIRED_CATEGORIES or right not in REQUIRED_CATEGORIES:
        return None
    return left, right


def add_pairwise_brackets(ax, stats_subset: pd.DataFrame, x_map_raw: dict[str, int], y_values: pd.Series) -> None:
    if stats_subset.empty:
        return
    finite = pd.to_numeric(y_values, errors="coerce").dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return
    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))
    span = max(1.0, ymax - ymin)
    y = ymax + span * 0.16
    step = span * 0.14
    tick = span * 0.04
    plotted = 0
    for row in stats_subset.sort_values("qvalue_bh", kind="mergesort").to_dict("records"):
        cats = contrast_categories(row.get("contrast", ""))
        if cats is None:
            continue
        left, right = cats
        if left not in x_map_raw or right not in x_map_raw:
            continue
        label = significance_label(row.get("qvalue_bh"))
        if not label:
            continue
        x1 = x_map_raw[left]
        x2 = x_map_raw[right]
        if x1 > x2:
            x1, x2 = x2, x1
        yy = y + plotted * step
        ax.plot([x1, x1, x2, x2], [yy, yy + tick, yy + tick, yy], color="black", linewidth=0.75, clip_on=False)
        ax.text((x1 + x2) / 2.0, yy + tick, label, ha="center", va="bottom", fontsize=8, clip_on=False)
        plotted += 1
        if plotted >= 4:
            break
    if plotted:
        ax.set_ylim(bottom=ymin - span * 0.18, top=y + plotted * step + span * 0.08)


def plot_species_centered_effects(values: pd.DataFrame, stats_df: pd.DataFrame, output_base: Path) -> bool:
    plot_df = values.copy()
    if plot_df.empty:
        return False
    selections = ["representative", "average"]
    evidences = [spec[0] for spec in EVIDENCE_SPECS]
    evidence_labels = {spec[0]: spec[1] for spec in EVIDENCE_SPECS}
    x_labels = DISPLAY_CATEGORIES
    x_map_raw = {category: idx for idx, category in enumerate(REQUIRED_CATEGORIES)}
    rng = np.random.default_rng(4)
    plt = ensure_plotting()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=False)
    for row_idx, selection in enumerate(selections):
        for col_idx, evidence in enumerate(evidences):
            ax = axes[row_idx, col_idx]
            sub = plot_df[(plot_df["selection"].eq(selection)) & (plot_df["evidence"].eq(evidence))].copy()
            if sub.empty:
                ax.axis("off")
                continue
            for species, species_df in sub.groupby("component_id"):
                species_df = species_df.copy()
                species_df["__x"] = species_df["category"].map(x_map_raw)
                species_df = species_df.sort_values("__x")
                xs = species_df["__x"].astype(int).tolist()
                ys = species_df["species_centered_value"].astype(float).tolist()
                ax.plot(xs, ys, color="#c8c8c8", linewidth=0.8, alpha=0.65, zorder=1)
                ax.scatter(xs, ys, s=14, color="#b0b0b0", alpha=0.75, zorder=2)
            means = []
            lows = []
            highs = []
            for cat in REQUIRED_CATEGORIES:
                mean, low, high = ci95(sub.loc[sub["category"].eq(cat), "species_centered_value"])
                means.append(mean)
                lows.append(low)
                highs.append(high)
            xs = np.arange(4)
            ax.errorbar(
                xs,
                means,
                yerr=[np.asarray(means) - np.asarray(lows), np.asarray(highs) - np.asarray(means)],
                fmt="o-",
                color="black",
                linewidth=1.4,
                markersize=4.5,
                capsize=3,
                zorder=4,
            )
            ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--", zorder=0)
            ax.set_xticks(xs)
            ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_title(evidence_labels[evidence], fontsize=10)
            stats_subset = stats_df[
                stats_df["selection"].astype(str).eq(selection)
                & stats_df["evidence"].astype(str).eq(evidence)
            ].copy()
            pair_subset = stats_subset.loc[
                stats_subset["contrast"].astype(str).ne("global_category_effect")
                & pd.to_numeric(stats_subset["qvalue_bh"], errors="coerce").lt(0.05)
            ].copy()
            add_pairwise_brackets(ax, pair_subset, x_map_raw, sub["species_centered_value"])
            if col_idx == 0:
                ax.set_ylabel(f"{selection}\nspecies-centered total")
            ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Functional category effects within exact species represented by all four genome categories", y=0.995)
    fig.text(
        0.5,
        0.01,
        "Gray lines are exact species; black points show category mean +/- approximate 95% CI after centering each species to its own mean.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.96])
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

    mp_genomes = load_mp_genomes(args.mp_dir, args.prefix)
    complete, species_summary = load_complete_species(args.component_members, args.require_high)
    merged = merge_mp_function_data(complete, mp_genomes)
    values, stats_df = build_effect_tables(merged)

    species_summary.to_csv(table_dir / "complete_species_summary.tsv", sep="\t", index=False)
    merged.to_csv(table_dir / "complete_species_genomes.tsv", sep="\t", index=False)
    values.to_csv(table_dir / "species_category_function_totals.tsv", sep="\t", index=False)
    stats_df.to_csv(table_dir / "species_blocked_category_effect_stats.tsv", sep="\t", index=False)
    plot_species_centered_effects(values, stats_df, plot_dir / "species_centered_category_effects")

    print(f"mp_genome_rows={len(mp_genomes)}")
    print(f"complete_species={species_summary.shape[0]}")
    print(f"complete_species_genomes={len(merged)}")
    print(f"require_high={bool(args.require_high)}")
    print("strict_mimag_tiers")
    print(merged["mimag_tier"].value_counts().to_string())
    print(table_dir / "species_blocked_category_effect_stats.tsv")
    print(plot_dir / "species_centered_category_effects.png")


if __name__ == "__main__":
    main()
