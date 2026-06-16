#!/usr/bin/env python3
"""GO ontology enrichment across all MP genomes by category.

This is an ontology-first companion to mp_all_genomes_module_enrichment.py.
It uses unfiltered GOA annotations and GO ancestry instead of bespoke
metabolism modes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_metapathways_genomes import extract_uniprot_accessions  # noqa: E402
from summarize_metapathways_wrapper import ensure_plotting, sanitize_label, save_figure  # noqa: E402


DEFAULT_MP_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/final_mp_summary_gunc_98")
DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_mappings"
DEFAULT_OUT_DIR = DEFAULT_MP_DIR / "all_genomes_category_go_ontology_enrichment"
CATEGORIES = ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"]
CATEGORY_DISPLAY = {
    "SAGs": "SAGs",
    "xPG_SAGs": "SAG-xPGs",
    "MAGs": "MAGs",
    "xPG_MAGs": "MAG-xPGs",
}
GO_ROOTS = {
    "molecular_function": "GO:0003674",
    "biological_process": "GO:0008150",
    "cellular_component": "GO:0005575",
}
DEFAULT_NAMESPACES = ["molecular_function", "biological_process"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build unsupervised GO ontology term/ancestor summaries from MP annotation "
            "accessions and run category enrichment."
        )
    )
    parser.add_argument("--mp-dir", type=Path, default=DEFAULT_MP_DIR)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="metapathways_batch")
    parser.add_argument("--namespaces", default=",".join(DEFAULT_NAMESPACES))
    parser.add_argument("--levels", default="3,4,5", help="Comma-separated GO depth levels to roll direct hits up to.")
    parser.add_argument("--include-direct", action="store_true", help="Also test direct GO terms.")
    parser.add_argument("--include-all-ancestors", action="store_true", help="Also test every ancestor term hit by any accession.")
    parser.add_argument("--min-genomes", type=int, default=5, help="Minimum genomes with nonzero support for a term.")
    parser.add_argument("--audit-max-rows", type=int, default=500000, help="Maximum GO hit audit rows to write; 0 disables cap.")
    parser.add_argument("--chunk-size", type=int, default=250000)
    parser.add_argument("--skip-plots", action="store_true", help="Only write tables; skip optional plot generation.")
    parser.add_argument("--purge", action="store_true")
    return parser.parse_args()


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False, **kwargs).fillna("")


def annotation_audit_path(mp_dir: Path, prefix: str) -> Path:
    candidates = [
        mp_dir / "tables" / "elemental" / f"{sanitize_label(prefix)}_elemental_annotation_audit.tsv",
        mp_dir / "tables" / "summary" / f"{sanitize_label(prefix)}_elemental_annotation_audit.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find elemental annotation audit for prefix={prefix!r} in {mp_dir}")


def genome_summary_path(mp_dir: Path, prefix: str) -> Path:
    path = mp_dir / "tables" / "summary" / f"{sanitize_label(prefix)}_genome_summary.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def goa_path(reference_dir: Path) -> Path:
    normalized = reference_dir / "normalized"
    candidates = [
        normalized / "goa_uniprotkb_gaf_normalized.tsv",
        normalized / "goa_uniprotkb_gaf_normalized.tsv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find unfiltered normalized GOA table in {normalized}")


def go_obo_path(reference_dir: Path) -> Path:
    candidates = [
        reference_dir / "raw" / "go-basic.obo",
        reference_dir / "go-basic.obo",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find go-basic.obo in {reference_dir}")


def parse_go_obo(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, set[str]], dict[str, int]]:
    terms: dict[str, dict[str, str]] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current: dict[str, object] = {}

    def flush() -> None:
        go_id = str(current.get("id", "")).strip()
        if not go_id:
            return
        terms[go_id] = {
            "go_id": go_id,
            "go_name": str(current.get("name", "")).strip(),
            "go_namespace": str(current.get("namespace", "")).strip(),
        }
        for parent in current.get("parents", set()):
            parents[go_id].add(str(parent))

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                flush()
                current = {"parents": set()}
                continue
            if line == "[Typedef]":
                break
            if not line or line.startswith("!"):
                continue
            if not current:
                continue
            if line.startswith("id: "):
                current["id"] = line.split("id: ", 1)[1]
            elif line.startswith("name: "):
                current["name"] = line.split("name: ", 1)[1]
            elif line.startswith("namespace: "):
                current["namespace"] = line.split("namespace: ", 1)[1]
            elif line.startswith("is_a: "):
                current["parents"].add(line.split("is_a: ", 1)[1].split(" ! ", 1)[0].strip())
            elif line.startswith("relationship: part_of "):
                current["parents"].add(line.split("relationship: part_of ", 1)[1].split(" ! ", 1)[0].strip())
        flush()

    children: dict[str, set[str]] = defaultdict(set)
    for child, parent_ids in parents.items():
        for parent in parent_ids:
            children[parent].add(child)

    depths: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((root, 0) for root in GO_ROOTS.values())
    while queue:
        go_id, depth = queue.popleft()
        if go_id in depths and depths[go_id] <= depth:
            continue
        depths[go_id] = depth
        for child in children.get(go_id, set()):
            queue.append((child, depth + 1))

    return terms, parents, depths


def ancestors_inclusive(go_id: str, parents: dict[str, set[str]]) -> set[str]:
    seen = {go_id}
    stack = [go_id]
    while stack:
        current = stack.pop()
        for parent in parents.get(current, set()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def normalize_accession(token: object) -> str:
    text = str(token).strip().upper()
    if text.startswith("UNIREF") and "_" in text:
        text = text.split("_", 1)[1]
    if "_" in text:
        text = text.split("_", 1)[0]
    return text.split("-", 1)[0]


def collect_observed_accessions(audit_path: Path, chunk_size: int) -> tuple[pd.DataFrame, set[str]]:
    usecols = ["genome_label", "sample", "category", "input_dir", "genome_id", "orf_id", "source_db", "annotation_text"]
    rows = []
    accessions: set[str] = set()
    reader = pd.read_csv(audit_path, sep="\t", dtype=str, low_memory=False, chunksize=chunk_size, usecols=usecols)
    for chunk in reader:
        chunk = chunk.fillna("")
        for row in chunk.to_dict("records"):
            text = str(row.get("annotation_text", "")).strip()
            if not text:
                continue
            for accession in extract_uniprot_accessions(text):
                accession = normalize_accession(accession)
                if not accession:
                    continue
                accessions.add(accession)
                record = {column: row.get(column, "") for column in usecols}
                record["matched_accession"] = accession
                rows.append(record)
    if not rows:
        return pd.DataFrame(columns=usecols + ["matched_accession"]), accessions
    observed = pd.DataFrame(rows).drop_duplicates()
    return observed, accessions


def load_goa_for_accessions(path: Path, accessions: set[str], namespaces: set[str], chunk_size: int) -> pd.DataFrame:
    if not accessions:
        return pd.DataFrame()
    opener_path = str(path)
    compression = "gzip" if opener_path.endswith(".gz") else None
    usecols = [
        "accession",
        "qualifier",
        "go_id",
        "go_name",
        "go_namespace",
        "evidence_code",
        "with_from",
        "aspect",
        "assigned_by",
    ]
    rows = []
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
        chunksize=chunk_size,
        compression=compression,
        usecols=usecols,
    )
    for chunk in reader:
        chunk = chunk.fillna("")
        chunk["accession"] = chunk["accession"].map(normalize_accession)
        chunk = chunk.loc[chunk["accession"].isin(accessions)]
        if namespaces:
            chunk = chunk.loc[chunk["go_namespace"].isin(namespaces)]
        chunk = chunk.loc[~chunk["qualifier"].str.contains(r"\bNOT\b", case=False, regex=True)]
        if not chunk.empty:
            rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=usecols)
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def build_go_hit_audit(observed: pd.DataFrame, goa: pd.DataFrame) -> pd.DataFrame:
    if observed.empty or goa.empty:
        return pd.DataFrame()
    audit = observed.merge(goa, left_on="matched_accession", right_on="accession", how="inner")
    preferred = [
        "genome_label",
        "sample",
        "category",
        "input_dir",
        "genome_id",
        "orf_id",
        "source_db",
        "annotation_text",
        "matched_accession",
        "go_id",
        "go_name",
        "go_namespace",
        "qualifier",
        "evidence_code",
        "with_from",
        "aspect",
        "assigned_by",
    ]
    return audit[[column for column in preferred if column in audit.columns]].drop_duplicates()


def choose_level_terms(
    direct_go_id: str,
    terms: dict[str, dict[str, str]],
    parents: dict[str, set[str]],
    depths: dict[str, int],
    level: int,
) -> set[str]:
    candidates = ancestors_inclusive(direct_go_id, parents)
    exact = {go_id for go_id in candidates if depths.get(go_id) == level}
    if exact:
        return exact
    direct_depth = depths.get(direct_go_id)
    if direct_depth is not None and direct_depth < level:
        return {direct_go_id}
    lower = [go_id for go_id in candidates if depths.get(go_id, -1) < level]
    if lower:
        best_depth = max(depths.get(go_id, -1) for go_id in lower)
        return {go_id for go_id in lower if depths.get(go_id, -1) == best_depth}
    namespace = terms.get(direct_go_id, {}).get("go_namespace", "")
    root = GO_ROOTS.get(namespace, "")
    return {root} if root else set()


def build_term_values(
    hit_audit: pd.DataFrame,
    genome_df: pd.DataFrame,
    terms: dict[str, dict[str, str]],
    parents: dict[str, set[str]],
    depths: dict[str, int],
    levels: list[int],
    include_direct: bool,
    include_all_ancestors: bool,
    min_genomes: int,
) -> pd.DataFrame:
    context_cols = ["genome_label", "sample", "category", "genome_id"]
    genomes = genome_df[context_cols].drop_duplicates().copy()
    records = []
    if hit_audit.empty:
        return pd.DataFrame()

    for row in hit_audit.to_dict("records"):
        direct = str(row.get("go_id", "")).strip()
        if not direct or direct not in terms:
            continue
        modules: list[tuple[str, str]] = []
        if include_direct:
            modules.append(("go_direct_accessions", direct))
        if include_all_ancestors:
            modules.extend(("go_all_ancestor_accessions", go_id) for go_id in ancestors_inclusive(direct, parents))
        for level in levels:
            modules.extend((f"go_level_{level}_accessions", go_id) for go_id in choose_level_terms(direct, terms, parents, depths, level))
        for evidence, go_id in modules:
            info = terms.get(go_id, {})
            records.append(
                {
                    "genome_label": row.get("genome_label", ""),
                    "sample": row.get("sample", ""),
                    "category": row.get("category", ""),
                    "genome_id": row.get("genome_id", ""),
                    "evidence": evidence,
                    "evidence_class": evidence.replace("_", " "),
                    "module_id": go_id,
                    "module_label": info.get("go_name", go_id),
                    "go_namespace": info.get("go_namespace", ""),
                    "go_depth": depths.get(go_id, np.nan),
                    "matched_accession": row.get("matched_accession", ""),
                    "orf_id": row.get("orf_id", ""),
                }
            )
    if not records:
        return pd.DataFrame()

    hits = pd.DataFrame(records).drop_duplicates(
        ["genome_label", "sample", "category", "genome_id", "evidence", "module_id", "matched_accession"]
    )
    counts = (
        hits.groupby(
            ["genome_label", "sample", "category", "genome_id", "evidence", "evidence_class", "module_id", "module_label", "go_namespace", "go_depth"],
            dropna=False,
        )
        .agg(value=("matched_accession", "nunique"), supporting_orfs=("orf_id", "nunique"))
        .reset_index()
    )
    modules = counts[
        ["evidence", "evidence_class", "module_id", "module_label", "go_namespace", "go_depth"]
    ].drop_duplicates()
    universe = genomes.merge(modules, how="cross")
    values = universe.merge(
        counts,
        on=["genome_label", "sample", "category", "genome_id", "evidence", "evidence_class", "module_id", "module_label", "go_namespace", "go_depth"],
        how="left",
    )
    values["value"] = pd.to_numeric(values["value"], errors="coerce").fillna(0.0)
    values["supporting_orfs"] = pd.to_numeric(values["supporting_orfs"], errors="coerce").fillna(0).astype(int)
    if min_genomes > 0:
        keep = (
            values.loc[values["value"].gt(0)]
            .groupby(["evidence", "module_id"], dropna=False)["genome_id"]
            .nunique()
            .reset_index(name="nonzero_genomes")
        )
        keep = keep.loc[keep["nonzero_genomes"].ge(int(min_genomes)), ["evidence", "module_id"]]
        values = values.merge(keep, on=["evidence", "module_id"], how="inner")
    return values


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


def enrichment_tables(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    pair_rows = []
    enriched_rows = []
    for (evidence, module_id), subset in long_df.groupby(["evidence", "module_id"], dropna=False):
        evidence_class = str(subset["evidence_class"].iloc[0])
        module_label = str(subset["module_label"].iloc[0])
        namespace = str(subset["go_namespace"].iloc[0])
        depth = subset["go_depth"].iloc[0]
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
                "go_namespace": namespace,
                "go_depth": depth,
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
                        "go_namespace": namespace,
                        "go_depth": depth,
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
                "go_namespace": row["go_namespace"],
                "go_depth": row["go_depth"],
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
            ["evidence_class", "go_namespace", "enriched_category", "module_label"],
            kind="mergesort",
        )
    return summary_df, pair_df, enriched_df


def plot_enrichment_counts(enriched_df: pd.DataFrame, output_base: Path) -> bool:
    if enriched_df.empty:
        return False
    plt = ensure_plotting()
    category_order = ["SAGs", "SAG-xPGs", "MAGs", "MAG-xPGs"]
    evidence_order = sorted(enriched_df["evidence_class"].dropna().astype(str).unique().tolist())
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
    fig, ax = plt.subplots(figsize=(8.4, max(3.8, 0.45 * len(evidence_order) + 1.8)))
    vmax = max(1, int(matrix.to_numpy().max()))
    image = ax.imshow(matrix.values, cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist(), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_title("Enriched GO ontology terms across all MP genomes")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iat[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value / vmax > 0.55 else "black")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Enriched GO terms")
    fig.tight_layout()
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

    namespaces = {token.strip() for token in str(args.namespaces).split(",") if token.strip()}
    levels = [int(token.strip()) for token in str(args.levels).split(",") if token.strip()]

    genome_df = read_table(genome_summary_path(args.mp_dir, args.prefix))
    genome_df = genome_df.loc[genome_df["category"].astype(str).isin(CATEGORIES)].copy()
    terms, parents, depths = parse_go_obo(go_obo_path(args.reference_dir))
    observed, accessions = collect_observed_accessions(annotation_audit_path(args.mp_dir, args.prefix), args.chunk_size)
    goa = load_goa_for_accessions(goa_path(args.reference_dir), accessions, namespaces, args.chunk_size)
    hit_audit = build_go_hit_audit(observed, goa)
    long_df = build_term_values(
        hit_audit,
        genome_df,
        terms,
        parents,
        depths,
        levels,
        include_direct=args.include_direct,
        include_all_ancestors=args.include_all_ancestors,
        min_genomes=max(0, int(args.min_genomes)),
    )
    summary_df, pair_df, enriched_df = enrichment_tables(long_df)

    hit_audit_out = hit_audit
    if args.audit_max_rows > 0 and len(hit_audit_out) > args.audit_max_rows:
        hit_audit_out = hit_audit_out.head(int(args.audit_max_rows)).copy()
    observed.to_csv(table_dir / "all_genomes_go_observed_accessions.tsv", sep="\t", index=False)
    hit_audit_out.to_csv(table_dir / "all_genomes_go_hit_audit.tsv", sep="\t", index=False)
    long_df.to_csv(table_dir / "all_genomes_go_ontology_values_long.tsv", sep="\t", index=False)
    summary_df.to_csv(table_dir / "all_genomes_go_ontology_category_summary.tsv", sep="\t", index=False)
    pair_df.to_csv(table_dir / "all_genomes_go_ontology_pairwise_stats.tsv", sep="\t", index=False)
    enriched_df.to_csv(table_dir / "all_genomes_go_ontology_category_enrichment_summary.tsv", sep="\t", index=False)
    plot_path = plot_dir / "all_genomes_go_ontology_enrichment_counts.png"
    if not args.skip_plots:
        try:
            plot_enrichment_counts(enriched_df, plot_dir / "all_genomes_go_ontology_enrichment_counts")
        except Exception as exc:
            print(f"[warn] skipping GO ontology enrichment plot: {exc}", file=sys.stderr)

    print(f"genomes={len(genome_df)}")
    print(genome_df["category"].value_counts().reindex(CATEGORIES).to_string())
    print(f"observed_accessions={len(accessions)}")
    print(f"goa_accession_rows={len(goa)}")
    print(f"go_hit_rows={len(hit_audit)}")
    print(f"go_terms_tested={summary_df.shape[0]}")
    print(f"enriched_go_terms={enriched_df.shape[0]}")
    print(table_dir / "all_genomes_go_ontology_category_enrichment_summary.tsv")
    print(table_dir / "all_genomes_go_ontology_values_long.tsv")
    print(plot_path)


if __name__ == "__main__":
    main()
