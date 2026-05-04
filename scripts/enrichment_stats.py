#!/usr/bin/env python3

"""Pure-Python implementation of the Shaiber/Willis enrichment method used by anvi'o.

This script intentionally preserves the behavior of `anvi-script-enrichment-stats`
while removing the runtime dependency on R.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import chi2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Hypothesis testing for enrichment claims. Preserves the anvi'o "
            "Shaiber/Willis method in pure Python: grouped binomial GLM / Rao-score "
            "equivalent statistic with Storey-style q-values."
        )
    )
    parser.add_argument("--input", required=True, help="Input tab-delimited occurrence table.")
    parser.add_argument("--output", required=True, help="Output tab-delimited enrichment table.")
    parser.add_argument(
        "--compute-associated-groups-if-missing",
        action="store_true",
        help=(
            "If the input table has no `associated_groups` column, compute it using the "
            "same anvi'o rule: groups whose p_group exceeds the overall proportion."
        ),
    )
    parser.add_argument(
        "--allow-empty-associated-groups",
        action="store_true",
        help=(
            "Allow an existing `associated_groups` column to be empty. If not set, the "
            "script preserves anvi'o's error on a completely empty association column."
        ),
    )
    return parser


def fail(message: str) -> None:
    raise SystemExit(message)


def first_p_column_index(columns: list[str]) -> int:
    for index, column in enumerate(columns):
        if str(column).startswith("p_"):
            return index
    fail("Input table does not contain any `p_` columns.")


def discover_groups(columns: Iterable[str]) -> list[str]:
    groups: list[str] = []
    seen: set[str] = set()
    for column in columns:
        text = str(column)
        if text.startswith(("p_", "N_")):
            group = text.split("_", 1)[1]
            if group not in seen:
                seen.add(group)
                groups.append(group)
    return groups


def validate_group_columns(columns: list[str], dynamic_columns: list[str]) -> list[str]:
    groups = discover_groups(dynamic_columns)
    if not groups:
        fail("Input table does not contain any valid `p_`/`N_` group columns.")
    p_columns = {str(column)[2:] for column in columns if str(column).startswith("p_")}
    n_columns = {str(column)[2:] for column in columns if str(column).startswith("N_")}
    if p_columns != n_columns:
        missing_p = sorted(n_columns - p_columns)
        missing_n = sorted(p_columns - n_columns)
        fail(
            "Input table must contain matching `p_` and `N_` columns per group. "
            f"Missing p columns for: {missing_p or 'none'}. "
            f"Missing N columns for: {missing_n or 'none'}."
        )
    if len(dynamic_columns) != len(groups) * 2:
        fail("Input table does not contain an even number of `p_` and `N_` columns.")
    return groups


def overall_portion(props: np.ndarray, reps: np.ndarray) -> float:
    return float(np.sum(props * reps) / np.sum(reps))


def compute_associated_groups(props: np.ndarray, reps: np.ndarray, groups: list[str]) -> str:
    if not np.count_nonzero(props):
        return ""
    threshold = overall_portion(props, reps)
    return ",".join(group for group, value in zip(groups, props) if value > threshold)


def compute_rao_score(props: np.ndarray, reps: np.ndarray) -> tuple[float, float]:
    if not np.count_nonzero(props):
        return 0.0, 1.0
    x = np.rint(reps * props).astype(int)
    total_reps = int(np.sum(reps))
    total_x = int(np.sum(x))
    if total_reps <= 0:
        return 0.0, 1.0
    p0 = total_x / total_reps
    if p0 <= 0.0 or p0 >= 1.0:
        return 0.0, 1.0
    expected = reps * p0
    variance = reps * p0 * (1.0 - p0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.nansum(((x - expected) ** 2) / variance)
    score = float(score)
    if score < 0.0 and score > -1e-5:
        score = 0.0
    if score < -1e-5:
        fail(
            "A Rao test statistic was unexpectedly large and negative. "
            "This mirrors an anvi'o hard failure condition."
        )
    df = max(1, len(props) - 1)
    pvalue = float(chi2.sf(score, df))
    if math.isnan(pvalue):
        if score < 1e-3:
            pvalue = 1.0
        else:
            fail(
                "The score statistic was not small, but the p-value could not be computed."
            )
    return score, pvalue


def estimate_pi0(pvalues: np.ndarray, lambdas: np.ndarray) -> float:
    if len(lambdas) == 0:
        raise ValueError("No lambda values available for pi0 estimation.")
    pi0_values = np.array(
        [np.mean(pvalues >= lam) / (1.0 - lam) for lam in lambdas],
        dtype=float,
    )
    if len(lambdas) == 1:
        pi0 = float(pi0_values[0])
    else:
        spline_degree = min(3, len(lambdas) - 1)
        spline = UnivariateSpline(lambdas, pi0_values, k=spline_degree, s=None)
        pi0 = float(spline(np.max(lambdas)))
    return float(np.clip(pi0, 0.0, 1.0))


def compute_qvalues(pvalues: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.size == 0:
        return np.array([], dtype=float)
    pi0 = estimate_pi0(pvalues, lambdas)
    order = np.argsort(pvalues)
    ordered = pvalues[order]
    ranks = np.arange(1, len(ordered) + 1, dtype=float)
    qvalues = pi0 * len(ordered) * ordered / ranks
    qvalues = np.minimum.accumulate(qvalues[::-1])[::-1]
    qvalues = np.clip(qvalues, 0.0, 1.0)
    restored = np.empty_like(qvalues)
    restored[order] = qvalues
    return restored


def resolve_qvalue_lambdas(pvalues: np.ndarray) -> np.ndarray:
    max_lambda = min(0.95, float(np.max(pvalues)))
    if max_lambda < 0.05:
        return np.array([0.2], dtype=float)
    lambdas = np.arange(0.05, max_lambda + 1e-9, 0.05, dtype=float)
    try:
        estimate_pi0(pvalues, lambdas)
    except Exception:
        lambdas = np.array([0.2], dtype=float)
        estimate_pi0(pvalues, lambdas)
    return lambdas


def build_output_columns(input_columns: list[str], n_columns_before_data: int, num_groups: int) -> list[str]:
    columns = [
        input_columns[0],
        "enrichment_score",
        "unadjusted_p_value",
        "adjusted_q_value",
        "associated_groups",
        "accession",
    ]
    for index in [1, 2]:
        if index < len(input_columns):
            columns.append(input_columns[index])
    columns.extend(input_columns[n_columns_before_data : n_columns_before_data + (num_groups * 2)])
    deduped: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            seen.add(column)
            deduped.append(column)
    return deduped


def prepare_input_dataframe(
    df_in: pd.DataFrame,
    *,
    compute_associated_groups_if_missing: bool = False,
    allow_empty_associated_groups: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], int, list[str]]:
    if df_in.shape[0] < 2:
        fail("Your input file contains less than two lines :/ Not OK.")

    working = df_in.copy()
    input_columns = [str(column) for column in working.columns]
    n_columns_before_data = first_p_column_index(input_columns)
    dynamic_columns = input_columns[n_columns_before_data:]
    groups = validate_group_columns(input_columns, dynamic_columns)

    if "accession" not in working.columns:
        fail("The input table does not contain an essential column ('accession').")

    if "associated_groups" not in working.columns:
        if compute_associated_groups_if_missing:
            associated_values = []
            for _, row in working.iterrows():
                props = np.array([float(row[f"p_{group}"]) for group in groups], dtype=float)
                reps = np.array([float(row[f"N_{group}"]) for group in groups], dtype=float)
                associated_values.append(compute_associated_groups(props, reps, groups))
            working["associated_groups"] = associated_values
        else:
            fail("The input table does not contain an essential column ('associated_groups').")

    associated_nonempty = working["associated_groups"].astype(str).str.strip()
    if not allow_empty_associated_groups and not np.any(associated_nonempty.ne("")):
        fail(
            "You have no group associations in your data. Which means, nothing is "
            "differentially occurring across your groups :/"
        )

    return working, input_columns, groups, n_columns_before_data, dynamic_columns


def run_enrichment_dataframe(
    df_in: pd.DataFrame,
    *,
    compute_associated_groups_if_missing: bool = False,
    allow_empty_associated_groups: bool = False,
) -> pd.DataFrame:
    working, _input_columns, groups, _n_columns_before_data, dynamic_columns = prepare_input_dataframe(
        df_in,
        compute_associated_groups_if_missing=compute_associated_groups_if_missing,
        allow_empty_associated_groups=allow_empty_associated_groups,
    )

    df_unique = working.loc[:, dynamic_columns].drop_duplicates().reset_index(drop=True)
    df_unique.insert(0, "data_structure_index", np.arange(1, len(df_unique) + 1, dtype=int))

    rows = []
    for _, row in df_unique.iterrows():
        props = np.array([float(row[f"p_{group}"]) for group in groups], dtype=float)
        reps = np.array([float(row[f"N_{group}"]) for group in groups], dtype=float)
        enrichment_score, unadjusted_p_value = compute_rao_score(props, reps)
        record = {"data_structure_index": int(row["data_structure_index"])}
        for column in dynamic_columns:
            record[column] = row[column]
        record["unadjusted_p_value"] = unadjusted_p_value
        record["enrichment_score"] = enrichment_score
        rows.append(record)
    pvalues_df_unique = pd.DataFrame(rows)

    if float(pvalues_df_unique["enrichment_score"].min()) < -1e-5:
        fail("A Rao test statistic is large and negative, oh my!")

    pvalues_df = (
        df_unique.merge(
            pvalues_df_unique[["data_structure_index", "unadjusted_p_value", "enrichment_score"]],
            on="data_structure_index",
        )
        .drop(columns=["data_structure_index"])
        .merge(working, on=dynamic_columns, how="inner")
    )

    lambdas = resolve_qvalue_lambdas(pvalues_df["unadjusted_p_value"].to_numpy(dtype=float))
    pvalues_df["adjusted_q_value"] = compute_qvalues(
        pvalues_df["unadjusted_p_value"].to_numpy(dtype=float),
        lambdas,
    )
    pvalues_df["enrichment_score"] = np.maximum(0.0, pvalues_df["enrichment_score"].to_numpy(dtype=float))
    return pvalues_df.sort_values(by="enrichment_score", ascending=False, kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_file():
        fail(f"Specified input file '{input_path}' does not exist")
    if output_path.exists():
        fail(f"Output file '{output_path}' already exists")

    df_in = pd.read_csv(input_path, sep="\t", dtype=str, keep_default_na=False)
    prepared_df, input_columns, groups, n_columns_before_data, _dynamic_columns = prepare_input_dataframe(
        df_in,
        compute_associated_groups_if_missing=args.compute_associated_groups_if_missing,
        allow_empty_associated_groups=args.allow_empty_associated_groups,
    )
    num_groups = len(groups)
    pvalues_df = run_enrichment_dataframe(
        prepared_df,
        compute_associated_groups_if_missing=False,
        allow_empty_associated_groups=True,
    )
    output_columns = build_output_columns(input_columns, n_columns_before_data, num_groups)
    enrichment_output = pvalues_df.loc[:, output_columns].copy()
    enrichment_output.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
