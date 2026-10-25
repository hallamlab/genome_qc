"""Strict GUNC detected-signal comparisons using the shared genome-level model."""
from pathlib import Path
import argparse
import json
import shlex
import sys

import numpy as np
import pandas as pd
from genome_set_stats import anchor_groups, method_mean_tests, adjust_method_tests, export_method_audit

STATES = {'credible_chimeric_signal', 'likely_not_chimeric', 'low_reference_uncertain', 'unscored'}


def gunc_tests(frame, category='category', sample='sample'):
    required = {category, sample, 'gunc_strict_assessment'}
    if not required.issubset(frame):
        raise ValueError('Missing GUNC comparison columns: ' + ', '.join(sorted(required-set(frame))))
    data = frame.copy()
    state = data.gunc_strict_assessment
    if state.isna().any() or not state.isin(STATES).all():
        raise ValueError('Missing or unknown strict GUNC classifications; do not silently count these as negative')
    data = anchor_groups(data, category, sample)
    state = data.gunc_strict_assessment
    data['gunc_detected_signal'] = state.eq('credible_chimeric_signal').astype(float)
    assessed = state.isin(['credible_chimeric_signal', 'likely_not_chimeric'])
    data['gunc_assessed_signal'] = data.gunc_detected_signal.where(assessed, np.nan)
    categories = sorted(data[category].dropna().astype(str).unique())
    rows = []
    for metric, label, scope in [
        ('gunc_detected_signal', 'Strict GUNC detected credible signal: all genomes', 'gunc_all_genomes'),
        ('gunc_assessed_signal', 'Strict GUNC credible signal: assessed genomes only', 'gunc_assessed_only')]:
        rows.extend(method_mean_tests(data, category, sample, metric, categories,
                    metric, label, scope, dependency_groups_precomputed=True))
    tests = adjust_method_tests(rows)
    counts = []
    for (sample_id, group), part in data.groupby([sample, category], dropna=False):
        n = int(part.gunc_assessed_signal.notna().sum())
        positive = int(part.gunc_detected_signal.sum())
        counts.append({sample: sample_id, category: group, 'n_all': len(part), 'n_assessed': n,
            'n_credible_signal': positive,
            'n_low_reference': int(part.gunc_strict_assessment.eq('low_reference_uncertain').sum()),
            'n_unscored': int(part.gunc_strict_assessment.eq('unscored').sum()),
            'fraction_detected_all': positive/len(part), 'fraction_chimeric_assessed': positive/n if n else np.nan})
    return data, tests, pd.DataFrame(counts)


def save_gunc_tests(frame, output_base, category='category', sample='sample', plots=True):
    data, tests, counts = gunc_tests(frame, category, sample)
    base = str(output_base)
    Path(base).parent.mkdir(parents=True, exist_ok=True)
    tests.to_csv(base+'_significance.tsv', sep='\t', index=False)
    counts.to_csv(base+'_denominators.tsv', sep='\t', index=False)
    data[[sample, category, 'dependency_genome_id', 'dependency_group', 'dependency_status',
          'gunc_strict_assessment', 'gunc_detected_signal', 'gunc_assessed_signal']].to_csv(
          base+'_genome_audit.tsv', sep='\t', index=False)
    if plots:
        export_method_audit(data, tests, category, sample, base)
    return tests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotated-table', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-column', default='sample')
    parser.add_argument('--category-column', default='category')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        parser.error('Use an empty output directory')
    import importlib.metadata
    settings = dict(input=str(args.annotated_table.resolve()), sample_column=args.sample_column,
        category_column=args.category_column, primary='Credible signal detected among all classified genomes; uncertain and unscored are not confirmed clean.',
        sensitivity='Credible signal among credible-chimeric or likely-not-chimeric genomes only; conditions on assessability.',
        model='Gaussian identity GEE, additive category and sample effects, anchor-clustered robust covariance',
        multiplicity='BH separately for omnibus and pairwise comparisons within each denominator analysis',
        limitations='Independent dependency groups assumed; no adjustment for all possible sequence overlap or ancestry; sparse events may limit asymptotic inference.',
        versions={p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels']})
    (args.output_dir/'analysis_settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    (args.output_dir/'command.txt').write_text(shlex.join([sys.executable, *sys.argv])+'\n')
    print('[start] Strict GUNC method comparisons', flush=True)
    tests = save_gunc_tests(pd.read_csv(args.annotated_table, sep='\t', low_memory=False),
                           args.output_dir/'gunc_strict', args.category_column, args.sample_column)
    print(f'[done] {len(tests)} tests; outputs: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
