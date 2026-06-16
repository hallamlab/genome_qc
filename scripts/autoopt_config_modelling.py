#!/usr/bin/env python3

import argparse
import math
import sys
import warnings
from pathlib import Path

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from summarize_metapathways_genomes import (  # noqa: E402
    apply_figure_typography,
    label_multi_panel_axes,
    plot_font_rc,
)


PROJECT_DIR = Path("/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109")
DEFAULT_MANIFEST = PROJECT_DIR / "qc_runs_mags_sags.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "autoopt_modelling_results"
CONFIG_FACTORS = ["autoopt_setting", "pr_balance", "filter"]
OUTPUT_FACTORS = ["algorithm"]
BLOCKING_FACTORS = ["sample_id"]
METRICS = ["Completeness", "Contamination", "qscore"]
MODEL_OUTCOMES = ["mq_hq", "is_hq"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Model SAG-xPG configuration effects on genome quality outcomes."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="QC runs manifest.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Output directory.")
    parser.add_argument("--category", default="xPGs_SAGs", help="Manifest category to model.")
    parser.add_argument("--min-config-n", type=int, default=30, help="Minimum n for top-config tables.")
    return parser.parse_args()


def enabled_mask(series):
    values = series.fillna("1").astype(str).str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    return values.isin({"true", "yes", "y"}) | numeric.fillna(0).ne(0)


def load_manifest(path, category):
    manifest = pd.read_csv(
        path,
        sep=None,
        engine="python",
        header=None,
        names=["sample", "category", "source_dir", "enabled"],
        dtype=str,
    )
    manifest = manifest.loc[manifest["category"].astype(str).eq(str(category))].copy()
    manifest = manifest.loc[enabled_mask(manifest["enabled"])].copy()
    if manifest.empty:
        raise SystemExit(f"No enabled manifest rows for category={category!r}.")
    return manifest


def load_raw_qc(manifest):
    frames = []
    missing = []
    for row in manifest.to_dict("records"):
        source_dir = Path(str(row["source_dir"])).expanduser()
        master_path = source_dir / "Master_genome_QC.tsv"
        if not master_path.is_file():
            missing.append(str(master_path))
            continue
        frame = pd.read_csv(master_path, sep="\t", low_memory=False)
        frame["manifest_sample"] = row["sample"]
        frame["manifest_category"] = row["category"]
        frame["source_dir"] = str(source_dir)
        frames.append(frame)
    if missing:
        raise SystemExit("Missing Master_genome_QC.tsv files:\n" + "\n".join(missing))
    if not frames:
        raise SystemExit("No raw QC frames loaded.")
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_bin_id(frame):
    parsed = frame["Bin Id"].astype(str).str.strip().str.split(".", expand=True)
    out = frame.copy()
    anatomy = {
        "sample_id": 0,
        "autoopt_setting": 1,
        "pr_balance": 2,
        "xpg_marker": 3,
        "sag_id": 4,
        "algorithm": 5,
        "mode": 6,
        "reassembly_marker": 7,
        "assembly_unit": 8,
        "filter": 9,
    }
    for column, index in anatomy.items():
        out[column] = parsed[index] if index in parsed.columns else np.nan
        out[column] = out[column].astype(str).str.strip()
        out.loc[out[column].isin(["", "nan", "None"]), column] = np.nan
    out["parse_ok"] = (
        out["xpg_marker"].eq("xpgs")
        & out["reassembly_marker"].eq("reasm")
        & out[CONFIG_FACTORS + ["sample_id", "sag_id", "mode"]].notna().all(axis=1)
    )
    return out


def classify_quality(frame):
    out = frame.copy()
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="coerce")
    hq = out["Completeness"].ge(90) & out["Contamination"].le(5) & out["qscore"].ge(50)
    mq_base = out["Completeness"].ge(50) & out["Contamination"].le(10)
    out["is_hq"] = hq.astype(int)
    out["is_mq"] = (mq_base & ~hq).astype(int)
    out["is_lq"] = (~mq_base).astype(int)
    out["mq_hq"] = mq_base.astype(int)
    out["quality_class"] = np.select([hq, mq_base], ["HQ", "MQ"], default="LQ")
    out["quality_class"] = pd.Categorical(out["quality_class"], categories=["LQ", "MQ", "HQ"], ordered=True)
    return out


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return center - half, center + half


def summarize_groups(frame, group_columns):
    grouped = frame.groupby(group_columns, dropna=False, observed=False)
    rows = []
    for keys, subset in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        n = int(len(subset))
        hq = int(subset["is_hq"].sum())
        mq = int(subset["is_mq"].sum())
        lq = int(subset["is_lq"].sum())
        mq_hq = int(subset["mq_hq"].sum())
        low, high = wilson_interval(mq_hq, n)
        record.update(
            {
                "n": n,
                "hq_n": hq,
                "mq_n": mq,
                "lq_n": lq,
                "mq_hq_n": mq_hq,
                "hq_rate": hq / n if n else np.nan,
                "mq_rate": mq / n if n else np.nan,
                "lq_rate": lq / n if n else np.nan,
                "mq_hq_rate": mq_hq / n if n else np.nan,
                "mq_hq_rate_ci_low": low,
                "mq_hq_rate_ci_high": high,
            }
        )
        for metric in METRICS:
            values = subset[metric].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            record[f"{metric}_median"] = float(values.median()) if not values.empty else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def write_core_tables(frame, table_dir, min_config_n):
    table_dir.mkdir(parents=True, exist_ok=True)
    model_input_path = table_dir / "sag_xpg_raw_model_input.tsv"
    frame.to_csv(model_input_path, sep="\t", index=False)

    factor_summaries = []
    for factor in CONFIG_FACTORS:
        summary = summarize_groups(frame, [factor]).sort_values(["mq_hq_rate", "n"], ascending=[False, False])
        summary.insert(0, "factor", factor)
        factor_summaries.append(summary)
        summary.to_csv(table_dir / f"factor_{factor}_outcome_summary.tsv", sep="\t", index=False)
    pd.concat(factor_summaries, ignore_index=True, sort=False).to_csv(
        table_dir / "factor_level_outcome_summary.tsv", sep="\t", index=False
    )

    output_algorithm_summary = summarize_groups(frame, OUTPUT_FACTORS)
    output_algorithm_summary.to_csv(table_dir / "output_algorithm_outcome_summary.tsv", sep="\t", index=False)

    config_summary = summarize_groups(frame, CONFIG_FACTORS)
    config_summary = config_summary.sort_values(["mq_hq_rate", "n"], ascending=[False, False])
    config_summary.to_csv(table_dir / "configuration_outcome_summary.tsv", sep="\t", index=False)
    config_summary.loc[config_summary["n"].ge(min_config_n)].to_csv(
        table_dir / "top_configurations_by_mq_hq_rate.tsv", sep="\t", index=False
    )

    balance = (
        frame.groupby(CONFIG_FACTORS + OUTPUT_FACTORS + BLOCKING_FACTORS, dropna=False, observed=False)
        .size()
        .reset_index(name="n")
        .sort_values(CONFIG_FACTORS + OUTPUT_FACTORS + BLOCKING_FACTORS)
    )
    balance.to_csv(table_dir / "design_balance_counts.tsv", sep="\t", index=False)
    return {
        "config_summary": config_summary,
        "factor_summary": pd.concat(factor_summaries, ignore_index=True, sort=False),
    }


def select_top_performers(frame):
    ranked = frame.copy()
    ranked["quality_rank"] = ranked["quality_class"].map({"LQ": 0, "MQ": 1, "HQ": 2}).astype(int)
    ranked["_top_tuple"] = list(
        zip(
            ranked["quality_rank"],
            ranked["qscore"],
            ranked["Completeness"],
            -ranked["Contamination"],
        )
    )
    best_by_sag = ranked.groupby("sag_id")["_top_tuple"].transform("max")
    top = ranked.loc[ranked["_top_tuple"].eq(best_by_sag)].copy()
    return top.drop(columns=["_top_tuple"])


def write_top_performer_tables(frame, table_dir):
    top = select_top_performers(frame)
    top.to_csv(table_dir / "top_performer_genomes_by_sag_id.tsv", sep="\t", index=False)
    total_sags = int(frame["sag_id"].nunique())
    top_counts_by_sag = top.groupby("sag_id", observed=False).size()
    unique_top_sag_ids = set(top_counts_by_sag.loc[top_counts_by_sag.eq(1)].index.astype(str))
    unique_top = top.loc[top["sag_id"].astype(str).isin(unique_top_sag_ids)].copy()
    rows = []
    for factor in CONFIG_FACTORS:
        for level, subset in top.groupby(factor, dropna=False, observed=False):
            unique_subset = unique_top.loc[unique_top[factor].astype(str).eq(str(level))]
            rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "top_row_n": int(len(subset)),
                    "top_sag_n": int(subset["sag_id"].nunique()),
                    "unique_top_sag_n": int(unique_subset["sag_id"].nunique()),
                    "total_sag_n": total_sags,
                    "share_of_top_rows": float(len(subset) / len(top)) if len(top) else np.nan,
                    "share_of_sags_with_top_level": float(subset["sag_id"].nunique() / total_sags) if total_sags else np.nan,
                    "share_of_sags_with_unique_top_level": float(unique_subset["sag_id"].nunique() / total_sags) if total_sags else np.nan,
                    "hq_top_row_n": int(subset["is_hq"].sum()),
                    "mq_top_row_n": int(subset["is_mq"].sum()),
                    "lq_top_row_n": int(subset["is_lq"].sum()),
                    "median_top_qscore": float(subset["qscore"].median()),
                    "median_top_completeness": float(subset["Completeness"].median()),
                    "median_top_contamination": float(subset["Contamination"].median()),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["factor", "top_sag_n"], ascending=[True, False])
    summary.to_csv(table_dir / "top_performer_parameter_counts.tsv", sep="\t", index=False)

    config_summary = summarize_groups(top, CONFIG_FACTORS)
    config_summary = config_summary.sort_values(["n", "mq_hq_rate"], ascending=[False, False])
    config_summary.to_csv(table_dir / "top_performer_configuration_counts.tsv", sep="\t", index=False)
    return top, summary


def treatment_term(factor, reference=None):
    if reference is None:
        return f"C({factor})"
    return f"C({factor}, Treatment(reference='{reference}'))"


def model_formula(response, include_factors):
    references = {
        "autoopt_setting": "algo_defaults",
        "pr_balance": "Default",
    }
    terms = [treatment_term(factor, references.get(factor)) for factor in include_factors]
    terms.extend([f"C({factor})" for factor in BLOCKING_FACTORS])
    return response + " ~ " + " + ".join(terms)


def fit_glm(formula, frame):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.glm(formula=formula, data=frame, family=sm.families.Binomial())
        result = model.fit(maxiter=200)
    try:
        return result.get_robustcov_results(cov_type="cluster", groups=frame["sag_id"])
    except Exception:
        return result


def coefficient_table(result):
    params = result.params
    conf = result.conf_int()
    pvalues = result.pvalues
    rows = []
    for term, estimate in params.items():
        if term == "Intercept":
            continue
        low, high = conf.loc[term]
        rows.append(
            {
                "term": term,
                "log_odds": float(estimate),
                "odds_ratio": float(np.exp(estimate)),
                "or_ci_low": float(np.exp(low)),
                "or_ci_high": float(np.exp(high)),
                "pvalue": float(pvalues.loc[term]),
            }
        )
    return pd.DataFrame(rows)


def model_importance(frame, response, full_result, table_dir):
    rows = []
    for factor in CONFIG_FACTORS:
        reduced_factors = [item for item in CONFIG_FACTORS if item != factor]
        reduced_formula = model_formula(response, reduced_factors)
        reduced_result = fit_glm(reduced_formula, frame)
        full_llf = float(full_result.llf)
        reduced_llf = float(reduced_result.llf)
        lr_stat = 2.0 * (full_llf - reduced_llf)
        df_diff = int(full_result.df_model - reduced_result.df_model)
        test_status = "ok"
        if (not np.isfinite(lr_stat)) or (not np.isfinite(reduced_llf)):
            lr_stat = np.nan
            pvalue = np.nan
            test_status = "unstable_reduced_model"
        elif df_diff > 0:
            from scipy.stats import chi2

            pvalue = float(chi2.sf(lr_stat, df_diff))
        else:
            pvalue = np.nan
            test_status = "no_df_difference"
        rows.append(
            {
                "factor": factor,
                "test_status": test_status,
                "full_llf": full_llf,
                "reduced_llf": reduced_llf,
                "lr_stat": lr_stat,
                "df": df_diff,
                "pvalue": pvalue,
                "delta_aic_reduced_minus_full": float(reduced_result.aic - full_result.aic),
                "delta_deviance": float(reduced_result.deviance - full_result.deviance),
            }
        )
    importance = pd.DataFrame(rows).sort_values("delta_deviance", ascending=False)
    importance.to_csv(table_dir / f"{response}_logistic_group_importance.tsv", sep="\t", index=False)
    return importance


def adjusted_config_predictions(frame, response, result, table_dir):
    combos = frame[CONFIG_FACTORS].drop_duplicates().sort_values(CONFIG_FACTORS)
    blocks = frame[BLOCKING_FACTORS].drop_duplicates().sort_values(BLOCKING_FACTORS)
    rows = []
    for combo in combos.to_dict("records"):
        pred_frame = blocks.copy()
        for factor, value in combo.items():
            pred_frame[factor] = value
        prediction = result.get_prediction(pred_frame).summary_frame()
        predicted = prediction["mean"].to_numpy(dtype=float)
        ci_low = prediction["mean_ci_lower"].to_numpy(dtype=float)
        ci_high = prediction["mean_ci_upper"].to_numpy(dtype=float)
        rows.append(
            {
                **combo,
                f"adjusted_predicted_{response}_rate": float(np.mean(predicted)),
                "adjusted_ci_low": float(np.mean(ci_low)),
                "adjusted_ci_high": float(np.mean(ci_high)),
                "prediction_min_over_sample_mode": float(np.min(predicted)),
                "prediction_max_over_sample_mode": float(np.max(predicted)),
            }
        )
    predictions = pd.DataFrame(rows).sort_values(f"adjusted_predicted_{response}_rate", ascending=False)
    predictions.to_csv(table_dir / f"adjusted_configuration_predictions_{response}.tsv", sep="\t", index=False)
    return predictions


def fit_model(frame, response, table_dir):
    full_formula = model_formula(response, CONFIG_FACTORS)
    full_result = fit_glm(full_formula, frame)
    coeff = coefficient_table(full_result)
    coeff.to_csv(table_dir / f"{response}_logistic_coefficients.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "model": f"{response}_logistic",
                "formula": full_formula,
                "n": int(full_result.nobs),
                "llf": float(full_result.llf),
                "aic": float(full_result.aic),
                "deviance": float(full_result.deviance),
                "df_model": float(full_result.df_model),
                "covariance_note": "cluster-robust by sag_id when available",
            }
        ]
    ).to_csv(table_dir / f"{response}_logistic_model_summary.tsv", sep="\t", index=False)
    importance = model_importance(frame, response, full_result, table_dir)
    predictions = adjusted_config_predictions(frame, response, full_result, table_dir)
    return {"coefficients": coeff, "importance": importance, "predictions": predictions}


def fit_models(frame, table_dir):
    return {response: fit_model(frame, response, table_dir) for response in MODEL_OUTCOMES}


def ensure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams.update(plot_font_rc())
    sns.set_theme(style="whitegrid", context="notebook", rc=plot_font_rc())
    return plt, sns


def savefig(fig, path_base):
    label_multi_panel_axes(fig)
    apply_figure_typography(fig)
    fig.savefig(str(path_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(path_base) + ".pdf", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_factor_summaries(factor_summary, plot_dir):
    plt, sns = ensure_plotting()
    plot_dir.mkdir(parents=True, exist_ok=True)
    for factor in CONFIG_FACTORS:
        subset = factor_summary.loc[factor_summary["factor"].eq(factor)].copy()
        subset = subset.sort_values("mq_hq_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(max(7, len(subset) * 1.0), 5.2))
        sns.barplot(data=subset, x=factor, y="mq_hq_rate", color="#8c8c8c", edgecolor="black", ax=ax)
        for index, row in enumerate(subset.to_dict("records")):
            ax.errorbar(
                index,
                row["mq_hq_rate"],
                yerr=[
                    [row["mq_hq_rate"] - row["mq_hq_rate_ci_low"]],
                    [row["mq_hq_rate_ci_high"] - row["mq_hq_rate"]],
                ],
                color="black",
                capsize=3,
                linewidth=0.9,
            )
            ax.text(index, row["mq_hq_rate"] + 0.025, f"n={int(row['n'])}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, min(1.05, max(0.2, subset["mq_hq_rate_ci_high"].max() + 0.1)))
        ax.set_ylabel("MQ/HQ rate")
        ax.set_xlabel(factor)
        ax.set_title(f"MQ/HQ rate by {factor}", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        savefig(fig, plot_dir / f"factor_{factor}_mq_hq_rate")


def plot_top_configs(config_summary, plot_dir, min_config_n):
    plt, sns = ensure_plotting()
    subset = config_summary.loc[config_summary["n"].ge(min_config_n)].head(20).copy()
    if subset.empty:
        return
    subset["config"] = subset[CONFIG_FACTORS].astype(str).agg(" | ".join, axis=1)
    subset = subset.sort_values("mq_hq_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(subset) * 0.35)))
    sns.barplot(data=subset, x="mq_hq_rate", y="config", color="#8c8c8c", edgecolor="black", ax=ax)
    ax.set_xlim(0, min(1.05, max(0.2, subset["mq_hq_rate"].max() + 0.1)))
    ax.set_xlabel("MQ/HQ rate")
    ax.set_ylabel("")
    ax.set_title(f"Top observed configurations (n >= {min_config_n})", fontweight="bold")
    fig.tight_layout()
    savefig(fig, plot_dir / "top_configurations_mq_hq_rate")


def plot_importance(importance, plot_dir, response):
    plt, sns = ensure_plotting()
    subset = importance.sort_values("delta_deviance", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sns.barplot(data=subset, x="delta_deviance", y="factor", color="#8c8c8c", edgecolor="black", ax=ax)
    ax.set_xlabel("Increase in deviance when factor is removed")
    ax.set_ylabel("")
    ax.set_title(f"Configuration group importance for {response}", fontweight="bold")
    fig.tight_layout()
    savefig(fig, plot_dir / f"{response}_logistic_group_importance")


def plot_coefficients(coefficients, plot_dir, response):
    plt, _ = ensure_plotting()
    subset = coefficients.loc[coefficients["term"].str.contains("|".join(CONFIG_FACTORS), regex=True)].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna(subset=["odds_ratio", "or_ci_low", "or_ci_high"])
    if subset.empty:
        return
    subset = subset.sort_values("odds_ratio")
    y = np.arange(len(subset), dtype=float)
    fig, ax = plt.subplots(figsize=(9, max(6, len(subset) * 0.28)))
    ax.errorbar(
        subset["odds_ratio"],
        y,
        xerr=[
            subset["odds_ratio"] - subset["or_ci_low"],
            subset["or_ci_high"] - subset["odds_ratio"],
        ],
        fmt="o",
        color="black",
        ecolor="#6f6f6f",
        capsize=2,
        markersize=4,
    )
    ax.axvline(1.0, color="#bdbdbd", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(subset["term"], fontsize=8)
    ax.set_xlabel(f"Odds ratio for {response}")
    ax.set_title("Adjusted configuration effects", fontweight="bold")
    fig.tight_layout()
    savefig(fig, plot_dir / f"{response}_logistic_config_odds_ratios")


def outcome_label(response):
    return {"mq_hq": "MQ/HQ", "is_hq": "HQ"}.get(response, response)


def write_summary_statement_table(model_objects, table_dir):
    rows = []
    for response, objects in model_objects.items():
        predictions = objects["predictions"].copy()
        importance = objects["importance"].copy()
        best = predictions.iloc[0].to_dict()
        probability_column = f"adjusted_predicted_{response}_rate"
        for rank, item in enumerate(importance.to_dict("records"), start=1):
            if pd.isna(item["pvalue"]):
                pvalue_display = "n/a*"
            elif item["pvalue"] < 0.001:
                pvalue_display = "p<0.001"
            else:
                pvalue_display = f"p={item['pvalue']:.3f}"
            rows.append(
                {
                    "outcome": response,
                    "outcome_label": outcome_label(response),
                    "best_configuration": " + ".join(str(best[factor]) for factor in CONFIG_FACTORS),
                    "best_autoopt_setting": best["autoopt_setting"],
                    "best_pr_balance": best["pr_balance"],
                    "best_filter": best["filter"],
                    "best_adjusted_probability": best[probability_column],
                    "best_adjusted_ci_low": best["adjusted_ci_low"],
                    "best_adjusted_ci_high": best["adjusted_ci_high"],
                    "factor": item["factor"],
                    "factor_importance_rank": rank,
                    "factor_test_status": item["test_status"],
                    "factor_lr_stat": item["lr_stat"],
                    "factor_df": item["df"],
                    "factor_pvalue": item["pvalue"],
                    "factor_pvalue_display": pvalue_display,
                    "factor_delta_deviance": item["delta_deviance"],
                    "factor_delta_aic_reduced_minus_full": item["delta_aic_reduced_minus_full"],
                    "factor_significant_p05": bool(pd.notna(item["pvalue"]) and item["pvalue"] < 0.05),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "summary_statement_support.tsv", sep="\t", index=False)
    return summary


def plot_summary_evidence_panel(model_objects, plot_dir):
    plt, sns = ensure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col_index, response in enumerate(MODEL_OUTCOMES):
        objects = model_objects[response]
        predictions = objects["predictions"].head(10).copy()
        probability_column = f"adjusted_predicted_{response}_rate"
        predictions["config"] = predictions[CONFIG_FACTORS].astype(str).agg(" + ".join, axis=1)
        predictions = predictions.sort_values(probability_column, ascending=True)

        ax = axes[0, col_index]
        y = np.arange(len(predictions), dtype=float)
        ax.errorbar(
            predictions[probability_column],
            y,
            xerr=[
                predictions[probability_column] - predictions["adjusted_ci_low"],
                predictions["adjusted_ci_high"] - predictions[probability_column],
            ],
            fmt="o",
            color="black",
            ecolor="#6f6f6f",
            capsize=3,
            markersize=4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(predictions["config"], fontsize=8)
        ax.set_xlim(0, min(1.0, max(0.1, float(predictions["adjusted_ci_high"].max()) + 0.08)))
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7)
        ax.set_xlabel(f"Adjusted probability of {outcome_label(response)}")
        ax.set_title(f"Top configurations for {outcome_label(response)}", fontweight="bold")

        importance = objects["importance"].copy().sort_values("delta_deviance", ascending=True)
        ax = axes[1, col_index]
        sns.barplot(
            data=importance,
            x="delta_deviance",
            y="factor",
            color="#8c8c8c",
            edgecolor="black",
            ax=ax,
        )
        for index, row in enumerate(importance.to_dict("records")):
            pvalue = row["pvalue"]
            if pd.isna(pvalue):
                label = "n/a*"
            elif pvalue < 0.001:
                label = "p<0.001"
            else:
                label = f"p={pvalue:.3f}"
            ax.text(row["delta_deviance"] + max(importance["delta_deviance"].max() * 0.015, 0.15), index, label, va="center", fontsize=8)
        ax.set_xlim(0, max(importance["delta_deviance"].max() * 1.18, 1.0))
        ax.set_xlabel("Increase in deviance when removed")
        ax.set_ylabel("")
        ax.set_title(f"Factor importance for {outcome_label(response)}", fontweight="bold")
        if importance["pvalue"].isna().any():
            ax.text(
                0.99,
                -0.18,
                "* reduced model unstable; p-value not reported",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#4d4d4d",
            )
    fig.tight_layout()
    savefig(fig, plot_dir / "summary_evidence_panel")


def mark_unique_top_performers(top_performers):
    counts = top_performers.groupby("sag_id", observed=False).size()
    unique_sags = set(counts.loc[counts.eq(1)].index.astype(str))
    marked = top_performers.copy()
    marked["top_type"] = np.where(marked["sag_id"].astype(str).isin(unique_sags), "unique top", "any top")
    return marked


def plot_top_performer_counts(top_summary, top_performers, plot_dir):
    plt, sns = ensure_plotting()
    marked_top = mark_unique_top_performers(top_performers)
    fig, axes = plt.subplots(2, len(CONFIG_FACTORS), figsize=(17, 9.5), squeeze=False)
    for ax, factor in zip(axes[0], CONFIG_FACTORS):
        subset = top_summary.loc[top_summary["factor"].eq(factor)].copy()
        subset = subset.sort_values("top_sag_n", ascending=False)
        x = np.arange(len(subset), dtype=float)
        ax.vlines(x, 0, subset["top_sag_n"], color="#6f6f6f", linewidth=2.0)
        ax.scatter(
            x,
            subset["unique_top_sag_n"],
            s=48,
            color="#d9d9d9",
            edgecolors="#5f5f5f",
            linewidths=0.6,
            zorder=4,
            label="unique top",
        )
        ax.scatter(x, subset["top_sag_n"], s=70, color="#2f2f2f", edgecolors="black", linewidths=0.7, zorder=3)
        for index, row in enumerate(subset.to_dict("records")):
            ax.text(
                index,
                row["top_sag_n"] + max(subset["top_sag_n"].max() * 0.015, 0.5),
                f"{int(row['top_sag_n'])}\n{row['share_of_sags_with_top_level']:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(subset["level"], rotation=35, ha="right")
        ax.set_xlabel(factor)
        ax.set_ylabel("SAGs with top performer" if ax is axes[0, 0] else "")
        ax.set_title(f"Top performers by {factor}", fontweight="bold")
        ax.set_ylim(0, subset["top_sag_n"].max() * 1.18 if not subset.empty else 1)
        ax.set_xlim(-0.5, len(subset) - 0.5)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)

    for ax, factor in zip(axes[1], CONFIG_FACTORS):
        levels = (
            top_summary.loc[top_summary["factor"].eq(factor)]
            .sort_values("top_sag_n", ascending=False)["level"]
            .astype(str)
            .tolist()
        )
        subset = marked_top.copy()
        subset[factor] = subset[factor].astype(str)
        sns.boxplot(
            data=subset,
            x=factor,
            y="qscore",
            order=levels,
            color="#6f6f6f",
            width=0.62,
            fliersize=1.5,
            linewidth=0.8,
            ax=ax,
        )
        unique_subset = subset.loc[subset["top_type"].eq("unique top")].copy()
        if not unique_subset.empty:
            sns.boxplot(
                data=unique_subset,
                x=factor,
                y="qscore",
                order=levels,
                color="#d9d9d9",
                width=0.34,
                fliersize=1.0,
                linewidth=0.8,
                ax=ax,
            )
        means_any = subset.groupby(factor, observed=False)["qscore"].mean().reindex(levels)
        means_unique = unique_subset.groupby(factor, observed=False)["qscore"].mean().reindex(levels)
        ax.scatter(np.arange(len(levels)) - 0.11, means_any, marker="D", s=30, color="black", zorder=5)
        ax.scatter(
            np.arange(len(levels)) + 0.11,
            means_unique,
            marker="D",
            s=30,
            color="#d9d9d9",
            edgecolors="#5f5f5f",
            linewidths=0.6,
            zorder=6,
        )
        ax.set_xlabel(factor)
        ax.set_ylabel("Top-performer Q-score" if ax is axes[1, 0] else "")
        ax.set_title(f"Q-score by {factor}", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7)
        ax.set_xlim(-0.5, len(levels) - 0.5)
        ax.set_ylim(bottom=min(0, float(subset["qscore"].min()) - 3), top=max(103, float(subset["qscore"].max()) + 3))
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=7,
            label="all genomes",
        ),
        mlines.Line2D(
            [],
            [],
            color="#5f5f5f",
            marker="o",
            markerfacecolor="#d9d9d9",
            linestyle="None",
            markersize=7,
            label="unique genomes",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="D",
            linestyle="None",
            markersize=5,
            label="diamond = mean",
        ),
    ]
    fig.legend(handles=legend_handles, frameon=False, loc="center right", bbox_to_anchor=(0.995, 0.5), fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    savefig(fig, plot_dir / "top_performer_parameter_counts")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(Path(args.manifest).expanduser(), args.category)
    raw = load_raw_qc(manifest)
    parsed = parse_bin_id(raw)
    parse_failures = parsed.loc[~parsed["parse_ok"]].copy()
    parse_failures.to_csv(table_dir / "bin_id_parse_failures.tsv", sep="\t", index=False)
    model_frame = parsed.loc[parsed["parse_ok"]].copy()
    model_frame = classify_quality(model_frame)
    model_frame = model_frame.dropna(subset=METRICS + CONFIG_FACTORS + BLOCKING_FACTORS + ["sag_id"]).copy()

    table_objects = write_core_tables(model_frame, table_dir, args.min_config_n)
    top_performers, top_summary = write_top_performer_tables(model_frame, table_dir)
    model_objects = fit_models(model_frame, table_dir)

    plot_factor_summaries(table_objects["factor_summary"], plot_dir)
    plot_top_configs(table_objects["config_summary"], plot_dir, args.min_config_n)
    plot_top_performer_counts(top_summary, top_performers, plot_dir)
    for response, objects in model_objects.items():
        plot_importance(objects["importance"], plot_dir, response)
        plot_coefficients(objects["coefficients"], plot_dir, response)
    write_summary_statement_table(model_objects, table_dir)
    plot_summary_evidence_panel(model_objects, plot_dir)

    print(f"input_rows={len(raw)}")
    print(f"model_rows={len(model_frame)}")
    print(f"top_performer_rows={len(top_performers)}")
    print(f"top_performer_sags={top_performers['sag_id'].nunique()}")
    print(f"parse_failures={len(parse_failures)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
