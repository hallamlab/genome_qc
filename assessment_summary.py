"""Attach tool completion states without turning failed measurements into zeros."""
import json
from pathlib import Path
import pandas as pd

STAGES = ('checkm', 'gtdbtk', 'rrna', 'rrna_check', 'trna', 'gunc')


def attach(frame, root):
    root = Path(root)
    records = {}
    for file in sorted((root / 'assessments').glob('*.assessment.json')):
        record = json.loads(file.read_text())
        for genome in record['genomes']:
            records[(genome, record['stage'])] = record
    audit = []
    for index, row in frame.iterrows():
        variant = row['Bin Id']
        for stage in STAGES:
            record = records.get((variant, stage), {})
            status = record.get('status', 'not_completed')
            reason = record.get('reason', 'No completed task receipt was produced')
            if status == 'completed':
                metric = {'checkm': 'Completeness', 'gtdbtk': 'Domain', 'gunc': 'gunc_clade_separation_score'}.get(stage)
                if metric and (metric not in row or pd.isna(row[metric]) or str(row[metric]).strip() == ''):
                    status = 'not_reported'
                    reason = 'Tool completed but did not report this genome/metric'
            frame.loc[index, stage + '_status'] = status
            audit.append(dict(Genome_Id=row['Genome_Id'], variant=variant, stage=stage,
                              status=status, exit_code=record.get('exit_code'), reason=reason,
                              log=str(Path({'checkm':'checkm/individual','gtdbtk':'gtdbtk/individual','rrna':'barrnap','rrna_check':'barrnap','trna':'trnascan','gunc':'gunc'}[stage]) / record['log']) if record.get('log') else '', retry_token=record.get('retry_token', '')))
            if status in ('failed', 'blocked', 'not_completed'):
                columns = {'checkm': ['Completeness', 'Contamination', 'Strain heterogeneity', 'qscore'],
                           'rrna': ['5S_rRNA', '16S_rRNA', '23S_rRNA', 'contains_16S', 'pass_BARRNAP'],
                           'rrna_check': ['pass_BARRNAP'], 'trna': ['trna_total', 'trna_unique']}.get(stage, [])
                for column in columns:
                    if column in frame:
                        frame[column] = frame[column].astype(object)
                        frame.loc[index, column] = float('nan')
        frame.loc[index, 'sequence_stats_status'] = 'completed' if pd.notna(row.get('sum_len')) else 'not_completed'
        audit.append(dict(Genome_Id=row['Genome_Id'], variant=variant, stage='sequence_stats', status=frame.loc[index, 'sequence_stats_status'], exit_code=None, reason='' if pd.notna(row.get('sum_len')) else 'Sequence statistics unavailable; see SeqKit task logs', log='', retry_token=''))
        frame.loc[index, 'assessment_complete'] = pd.notna(row.get('sum_len')) and all(frame.loc[index, s + '_status'] == 'completed' for s in STAGES)
        frame.loc[index, 'rrna_domain_basis'] = 'GTDB-Tk' if str(row.get('Domain', '')).strip() else 'Bacteria fallback; taxonomy unavailable'
        frame.loc[index, 'trna_domain_basis'] = 'GTDB-Tk' if str(row.get('Domain', '')).strip() else 'general model; taxonomy unavailable'
    pd.DataFrame(audit, columns=['Genome_Id','variant','stage','status','exit_code','reason','log','retry_token']).to_csv(root/'Assessment_status.tsv', sep='\t', index=False)
    return frame
