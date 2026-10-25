"""Sample-adjusted method mean comparisons with anchor-clustered GEE inference."""
import hashlib
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MIN_CLUSTERS = 10


def anchor_groups(frame, category_column, sample_column):
    """Keep all rows; join known cognate anchors, including orphan xPG siblings.

    Uses the same anchor parser as ANI provenance links. Sample and source family
    are part of every key. Unrecognized identities remain singleton groups and
    are marked unresolved, never silently claimed to be independent lineages.
    """
    from genome_quality_atlas import (category_family, base_anchor_aliases,
                                     xpg_genesis_anchor, base_family_from_xpg)
    out = frame.reset_index(drop=True).copy()
    parent = list(range(len(out)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    seen = {}
    statuses = []
    identities = []
    for i, row in out.iterrows():
        sample = str(row.get(sample_column, '')).strip()
        family = category_family(row.get('anchor_category', row.get('category', row.get(category_column, ''))))
        identity = next((str(row[c]).strip() for c in ['Genome_Id', 'genome_id', 'genome_label', 'Bin Id']
                         if c in row and pd.notna(row[c]) and str(row[c]).strip()), f'row:{i}')
        identities.append(identity)
        if family in ('sag', 'mag'):
            tokens = base_anchor_aliases(row)
            status = 'source_anchor' if tokens else 'unresolved_singleton'
        elif family in ('xpg_sag', 'xpg_mag'):
            token = xpg_genesis_anchor(row, family)
            if not token:
                # MP summary IDs are often already reduced to the source anchor.
                import re
                if family == 'xpg_mag' and re.fullmatch(r'bin_\d+', identity, flags=re.I):
                    token = identity
                elif family == 'xpg_sag' and re.fullmatch(r'AB-\d+_[A-Za-z0-9_-]+', identity):
                    token = identity
            tokens = [token] if token else []
            family = base_family_from_xpg(family)
            status = 'derived_anchor' if tokens else 'unresolved_xpg_anchor'
        else:
            tokens = []
            status = 'unresolved_singleton'
        for token in tokens:
            key = (sample, family, token)
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
        statuses.append(status)
    members = {}
    for i in range(len(out)):
        members.setdefault(find(i), []).append(i)
    labels = {}
    for indices in members.values():
        # Identity-based IDs are independent of metric values and row order.
        signature = sorted((str(out.iloc[i].get(sample_column, '')),
                            str(out.iloc[i].get(category_column, '')), identities[i]) for i in indices)
        label = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:20]
        for i in indices:
            labels[i] = label
    out['dependency_group'] = [labels[i] for i in range(len(out))]
    out['dependency_status'] = statuses
    out['dependency_genome_id'] = identities
    return out


def method_mean_tests(frame, category_column, sample_column, value_column, categories,
                      metric, metric_label, scope, dependency_groups_precomputed=False):
    """Gaussian identity GEE: sample-adjusted mean contrasts, robust Wald tests.

    No aggregation or matching restriction. Primary covariance groups anchors;
    sensitivity covariance groups each observation separately (HC0 equivalent).
    Both use method indicators plus sample fixed effects. Method contrasts
    assume an additive, common method effect across samples.
    """
    import statsmodels.api as sm
    required = {category_column, sample_column, value_column}
    if sample_column == category_column or not required.issubset(frame.columns):
        warnings.warn(f'Skipping {metric}: missing sample/category/metric columns.')
        return []
    if dependency_groups_precomputed:
        if not {'dependency_group', 'dependency_status'}.issubset(frame):
            raise ValueError('Precomputed dependency groups are missing.')
        frame = frame.copy()
    else:
        frame = anchor_groups(frame, category_column, sample_column)
    frame[value_column] = pd.to_numeric(frame[value_column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[sample_column, category_column, value_column])
    rows = []
    frame = frame.loc[frame[sample_column].astype(str).str.strip().ne('')]
    for sample, d in [('all_samples', frame)]:
        present = [c for c in categories if d[category_column].astype(str).eq(c).any()]
        if len(present) < 2:
            continue
        d = d.loc[d[category_column].astype(str).isin(present)].copy()
        y = d[value_column].to_numpy(float)
        x = np.column_stack([d[category_column].astype(str).eq(c).to_numpy(float) for c in present])
        samples = sorted(d[sample_column].astype(str).unique())
        sample_design = np.column_stack([d[sample_column].astype(str).eq(v).to_numpy(float)
                                        for v in samples[1:]]) if len(samples) > 1 else np.empty((len(d), 0))
        x = np.column_stack([x, sample_design])
        # Standardized means use the same observed sample distribution for every method.
        prediction = np.column_stack([np.eye(len(present)),
            np.tile(sample_design.mean(axis=0), (len(present), 1))])
        ng = d.dependency_group.nunique()
        base = dict(metric=metric, metric_label=metric_label, group_column=category_column,
                    sample_column=sample_column, sample=str(sample), blocking_unit='anchor',
                    aggregation='none', estimand='sample_adjusted_difference_in_means',
                    model='method + sample (additive fixed effects)', samples=';'.join(samples),
                    test='GEE Gaussian identity; sample-adjusted anchor-robust Wald', n_genomes=len(d),
                    n_clusters=ng, n_samples=len(samples), n_groups=len(present),
                    groups=';'.join(present), n_unresolved=int(d.dependency_status.isin(['unresolved_singleton','unresolved_xpg_anchor']).sum()),
                    covariance='robust', working_correlation='independence')
        status = 'ok'
        if ng < MIN_CLUSTERS:
            status = 'insufficient_clusters'
        elif d.dependency_status.eq('unresolved_xpg_anchor').any():
            status = 'unresolved_xpg_anchor'
        elif np.linalg.matrix_rank(x) < x.shape[1]:
            status = 'confounded_method_sample_design'
        elif np.ptp(y) == 0:
            status = 'constant_outcome'
        cov = independent = None
        means = np.full(x.shape[1], np.nan)
        if status == 'ok':
            try:
                fit = sm.GEE(y, x, groups=d.dependency_group.to_numpy(),
                             family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence()).fit()
                sensitivity = sm.GEE(y, x, groups=np.arange(len(d)),
                             family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence()).fit()
                if not fit.converged or not sensitivity.converged:
                    status = 'not_converged'
                else:
                    means = np.asarray(fit.params)
                    cov, independent = np.asarray(fit.cov_params()), np.asarray(sensitivity.cov_params())
                    if not np.isfinite(cov).all() or not np.isfinite(independent).all():
                        status = 'nonfinite_covariance'
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
                status = 'fit_failed: ' + str(exc)
        method_basis = np.eye(x.shape[1])[:len(present)]
        contrasts = [('global', None, None, method_basis[1:] - method_basis[0])]
        contrasts += [('pairwise', a, b, (method_basis[a]-method_basis[b])[None, :])
                      for a, b in itertools.combinations(range(len(present)), 2)]
        for kind, a, b, contrast in contrasts:
            row = dict(base, analysis_scope=scope+'_'+kind, status=status,
                       group_a='' if a is None else present[a], group_b='' if b is None else present[b],
                       statistic=np.nan, pvalue=np.nan, pvalue_independent=np.nan,
                       standard_error=np.nan, standard_error_independent=np.nan,
                       ci_low=np.nan, ci_high=np.nan, mean_difference_a_minus_b=np.nan)
            effect = contrast @ means
            if a is not None:
                aa, bb = d.loc[x[:, a] == 1, value_column], d.loc[x[:, b] == 1, value_column]
                row.update(n_a=len(aa), n_b=len(bb), group_a_mean=aa.mean(), group_b_mean=bb.mean(),
                           group_a_adjusted_mean=float(prediction[a] @ means),
                           group_b_adjusted_mean=float(prediction[b] @ means),
                           group_a_median=aa.median(), group_b_median=bb.median(),
                           mean_difference_a_minus_b=float(effect[0]))
            relevant = list(range(len(present))) if a is None else [a, b]
            counts = [d.loc[x[:, j] == 1, 'dependency_group'].nunique() for j in relevant]
            row['min_category_clusters'] = min(counts)
            if status == 'ok' and min(counts) < MIN_CLUSTERS:
                row['status'] = 'insufficient_category_clusters'
            if row['status'] == 'ok':
                v = contrast @ cov @ contrast.T
                vi = contrast @ independent @ contrast.T
                if np.linalg.matrix_rank(v) < len(contrast) or np.linalg.matrix_rank(vi) < len(contrast):
                    row['status'] = 'singular_contrast_covariance'
                else:
                    w = float(effect @ np.linalg.solve(v, effect))
                    wi = float(effect @ np.linalg.solve(vi, effect))
                    row.update(statistic=w, pvalue=float(stats.chi2.sf(w, len(contrast))),
                               pvalue_independent=float(stats.chi2.sf(wi, len(contrast))), df=len(contrast))
                    if a is not None:
                        se = float(np.sqrt(v[0, 0]))
                        row.update(standard_error=se, standard_error_independent=float(np.sqrt(vi[0, 0])),
                                   ci_low=float(effect[0]-stats.norm.ppf(.975)*se),
                                   ci_high=float(effect[0]+stats.norm.ppf(.975)*se))
            rows.append(row)
    return rows


def adjust_method_tests(rows):
    """BH separately per analysis scope (global/pairwise) and covariance analysis."""
    from statsmodels.stats.multitest import multipletests
    d = pd.DataFrame(rows)
    if d.empty:
        return pd.DataFrame(columns=['test','metric','analysis_scope','sample','status','pvalue',
                                     'qvalue_bh','significant_q05','significant_p05','group_a','group_b'])
    for pcol, qcol in [('pvalue', 'qvalue_bh'), ('pvalue_independent', 'qvalue_independent_bh')]:
        d[qcol] = np.nan
        for _, ix in d.groupby(['sample', 'analysis_scope']).groups.items():
            valid = d.loc[ix, pcol].dropna()
            if len(valid):
                d.loc[valid.index, qcol] = multipletests(valid, method='fdr_bh')[1]
    d['significant_independent_q05'] = d.qvalue_independent_bh.lt(.05)
    d['significant_p05'] = d.pvalue.lt(.05)
    d['significant_q05'] = d.qvalue_bh.lt(.05)
    d['significance_stars'] = d.qvalue_bh.map(lambda q: '***' if q < .001 else '**' if q < .01 else '*' if q < .05 else '')
    return d


def export_method_audit(frame, stats_df, category_column, sample_column, output_base):
    """Write membership audit and sample-adjusted effect plots."""
    if sample_column not in frame or category_column not in frame:
        return
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    grouped = anchor_groups(frame, category_column, sample_column)
    grouped[[sample_column, category_column, 'dependency_genome_id', 'dependency_group',
             'dependency_status']].to_csv(str(base)+'_dependency_groups.tsv', sep='\t', index=False)
    if stats_df is None or stats_df.empty:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = stats_df.loc[stats_df.analysis_scope.str.endswith('_pairwise') & stats_df.status.eq('ok')]
    for (sample, metric), d in valid.groupby(['sample','metric']):
        import re
        suffix = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{sample}_{metric}')
        fig, ax = plt.subplots(figsize=(8, max(3, .45*len(d)+1.5)))
        d = d.reset_index(drop=True)
        ax.errorbar(d.mean_difference_a_minus_b, d.index,
                    xerr=np.vstack([d.mean_difference_a_minus_b-d.ci_low, d.ci_high-d.mean_difference_a_minus_b]),
                    fmt='o', color='black', capsize=3)
        ax.set_yticks(d.index, [f'{r.group_a} − {r.group_b} {r.significance_stars}' for r in d.itertuples()])
        ax.axvline(0, color='gray', linestyle='--')
        ax.set_xlabel('Sample-adjusted mean difference (anchor-robust 95% CI; unadjusted intervals)')
        ax.set_title(f'{d.metric_label.iloc[0]}: method comparisons adjusted for sample\nStars: anchor-robust BH-adjusted pairwise p-values')
        fig.tight_layout()
        for ext in ('png','pdf'):
            fig.savefig(str(base)+'_'+suffix+'.'+ext, dpi=160)
        plt.close(fig)


def split_by_sample(function, argument_index=0, sample_columns=('source_sample', 'sample')):
    """Keep existing component-level comparisons, but report them per depth."""
    import functools
    import inspect
    signature = inspect.signature(function)
    parameter = list(signature.parameters)[argument_index]
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        frame = bound.arguments[parameter]
        column = next((c for c in sample_columns if isinstance(frame, pd.DataFrame) and c in frame), None)
        if column is None or frame.empty or frame[column].nunique() <= 1:
            result = function(*args, **kwargs)
            if column and not frame.empty:
                for table in result if isinstance(result, tuple) else [result]:
                    if isinstance(table, pd.DataFrame):
                        table['sample'] = str(frame[column].iloc[0])
                        _component_test_metadata(table)
            return result
        results = []
        original_arguments = dict(bound.arguments)
        for sample, subset in frame.groupby(column, sort=True):
            for name, value in original_arguments.items():
                if isinstance(value, pd.DataFrame) and column in value:
                    bound.arguments[name] = value.loc[value[column].eq(sample)].copy()
                else:
                    bound.arguments[name] = value
            bound.arguments[parameter] = subset.copy()
            result = function(*bound.args, **bound.kwargs)
            for table in result if isinstance(result, tuple) else [result]:
                if isinstance(table, pd.DataFrame):
                    table['sample'] = str(sample)
                    _component_test_metadata(table)
            results.append(result)
        if isinstance(results[0], tuple):
            return tuple(pd.concat([r[i] for r in results], ignore_index=True) for i in range(len(results[0])))
        return pd.concat(results, ignore_index=True)
    return wrapped


def _component_test_metadata(table):
    if 'pvalue_sign' in table or 'pvalue_binom' in table:
        table['test'] = 'two-sided exact binomial (component signs or decisive wins)'
        table['inference_unit'] = 'component within sample'
        table['null_hypothesis'] = 'equal probability of either direction among non-ties/decisive wins'
        table['independence_assumption'] = 'independent components; linked members collapsed within component'
