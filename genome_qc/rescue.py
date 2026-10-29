"""Explicit recovery plans; never change analytical parameters or escalate resources."""
import csv
import json
import shutil
from pathlib import Path

FOLDERS = {'preprocess': 'genomes_pruned', 'sequence_stats': 'genomes_pruned',
           'rrna': 'barrnap', 'rrna_check': 'barrnap', 'trna': 'trnascan',
           'checkm': 'checkm/individual', 'gtdbtk': 'gtdbtk/individual', 'gunc': 'gunc'}


def records(output, work):
    """Read published receipts and legacy unpublished batch receipts, newest first."""
    candidates = []
    for folder in set(FOLDERS.values()):
        candidates.extend((output / folder).glob('*.assessment.json'))
    for pattern in ('*/*/*.assessment.json', '*/*/results/*.assessment.json'):
        candidates.extend(work.glob(pattern))
    result = {}
    for path in candidates:
        try:
            row = json.loads(path.read_text())
            if row['stage'] == 'preprocess':
                row['genomes'] = [path.name.removesuffix('.preprocess.assessment.json')]
            key = (row['stage'], tuple(sorted(row['genomes'])))
            if row['stage'] not in FOLDERS:
                continue
            if key not in result or row.get('finished_utc', '') > result[key][1].get('finished_utc', ''):
                result[key] = (path, row)
        except (ValueError, KeyError, OSError):
            continue
    return list(result.values())


def products(path, row):
    stem = path.name.removesuffix('.assessment.json')
    stage = row['stage']
    identity = row['genomes'][0]
    suffixes = {'preprocess': ['.fasta'], 'sequence_stats': ['.seqkit.tsv'],
                'rrna': ['.rRNA.gff', '.rRNA.fasta'], 'rrna_check': ['.blastout'],
                'trna': ['.gff']}
    if stage in suffixes:
        return [path.parent / (identity + s) for s in suffixes[stage]]
    if stage == 'checkm':
        return [path.parent / (stem + '.checkm.tsv')]
    if stage == 'gtdbtk':
        return [path.parent / stem, path.parent / stem.replace('gtdbtk_', 'marker_alignments_', 1)]
    return [path.parent / stem]


def normalize_settings(settings):
    return {k: v for k, v in settings.items()
            if k not in ('rescue_plan', 'small_batch_size', 'implementation_digest')
            and not k.endswith(('_batch_size', '_parallel_batches'))}


def prepare(output, work, destination):
    """Snapshot reusable products. Heavy tools checkpoint at their native batch boundary."""
    output, work, destination = map(Path, (output, work, destination))
    if not (output / 'Assessment_status.tsv').is_file():
        raise ValueError('--rescue requires a previous Assessment_status.tsv; use --resume for an interrupted initial workflow')
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output / 'Assessment_status.tsv', destination / 'previous-assessments.tsv')
    found = records(output, work)
    diagnostics = destination / 'previous-diagnostics'
    diagnostics.mkdir()
    for source, row in found:
        if row['status'] != 'completed':
            shutil.copy2(source, diagnostics / source.name)
            log = source.parent / row.get('log', '')
            if log.is_file(): shutil.copy2(log, diagnostics / log.name)
    good = [(p, r) for p, r in found if r['status'] == 'completed' and all(f.exists() for f in products(p, r))]
    pre = {g for _, r in good if r['stage'] == 'preprocess' for g in r['genomes']}
    stats = {g for _, r in good if r['stage'] == 'sequence_stats' for g in r['genomes']}
    ready_pre = pre & stats
    # Failed/missing taxonomy can change RNA kingdom choice on recovery.
    ready_tax = {g for _, r in good if r['stage'] == 'gtdbtk' for g in r['genomes']}
    plan = {s: {'ids': [], 'outputs': [], 'receipts': []} for s in FOLDERS}
    for path, row in good:
        stage, genomes = row['stage'], row['genomes']
        if not set(genomes) <= ready_pre:
            continue
        if stage in ('rrna', 'rrna_check', 'trna') and not set(genomes) <= ready_tax:
            continue
        if stage == 'rrna_check' and not any(r['stage'] == 'rrna' and r['genomes'] == genomes for _, r in good):
            continue
        folder = destination / stage
        folder.mkdir(exist_ok=True)
        copied = []
        for source in products(path, row):
            target = folder / source.name
            if target.exists():
                raise ValueError(f'Overlapping recovery products: {target}')
            # Retained work is required until rescue completes. Sources are not
            # mutated: completed heavy batches are excluded from re-execution.
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
            copied.append(str(target))
        receipt = folder / path.name
        shutil.copy2(path, receipt)
        log = path.parent / row.get('log', '')
        if log.is_file():
            shutil.copy2(log, folder / log.name)
        plan[stage]['ids'].extend(genomes)
        plan[stage]['outputs'].extend(copied)
        plan[stage]['receipts'].append(str(receipt))
    for stage in plan:
        if len(plan[stage]['ids']) != len(set(plan[stage]['ids'])):
            raise ValueError(f'Overlapping successful {stage} receipts; cannot safely choose recovery products')
    path = destination / 'plan.json'
    path.write_text(json.dumps(plan, indent=2) + '\n')
    return path


def failure_report(output, work, logs):
    """Publish diagnostic logs and an inspectable failure list without guessing causes."""
    output, work, logs = map(Path, (output, work, logs))
    retained = records(output, work)
    for path, row in retained:
        source = path.parent / row.get('log', '')
        target = output / FOLDERS[row['stage']] / source.name
        if source.is_file() and source.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    status = output / 'Assessment_status.tsv'
    if not status.exists():
        return
    with status.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter='\t') if r['status'] in ('failed', 'blocked', 'not_completed')]
    completed = {(g, r['stage']) for _, r in retained if r['status'] == 'completed' for g in r['genomes']}
    indexed = {(r['variant'], r['stage']): r for r in rows}
    for path, record in retained:
        if record['status'] == 'completed':
            continue
        for variant in record['genomes']:
            if (variant, record['stage']) in completed:
                continue
            row = indexed.get((variant, record['stage']))
            if row is None:
                row = dict(Genome_Id=variant.rsplit('.', 1)[0], variant=variant, stage=record['stage'], status=record['status'])
                rows.append(row); indexed[(variant, record['stage'])] = row
            if not row.get('log'):
                row.update(log=str(Path(FOLDERS[record['stage']]) / record.get('log', '')),
                           reason=record.get('reason', ''), exit_code=record.get('exit_code', ''))
    evidence = scheduler_evidence(output, work, logs)
    columns = ['Genome_Id', 'variant', 'stage', 'status', 'exit_code', 'reason', 'log', 'failure_class', 'job_id', 'slurm_state', 'slurm_exit', 'requested_cpus', 'requested_memory', 'requested_time', 'max_rss']
    for row in rows:
        row.update(evidence.get(Path(row.get('log', '')).name, {}))
        row['failure_class'] = 'upstream_failure' if row['status'] == 'blocked' else 'unknown'
        log = output / row.get('log', '')
        text = log.read_text(errors='replace').lower() if log.is_file() else ''
        if row.get('slurm_state') == 'OUT_OF_MEMORY':
            row['failure_class'] = 'out_of_memory'
        elif row.get('slurm_state') == 'TIMEOUT':
            row['failure_class'] = 'time_limit'
        elif row.get('slurm_state') in ('NODE_FAIL', 'PREEMPTED'):
            row['failure_class'] = row['slurm_state'].lower()
        elif any(s in text for s in ('out_of_memory', 'out of memory', 'oom-kill', 'oom_kill')):
            row['failure_class'] = 'out_of_memory'
        elif 'time limit' in text or 'timeout' in text:
            row['failure_class'] = 'time_limit'
    for target in (output / 'Assessment_failures.tsv', logs / 'Assessment_failures.tsv'):
        with target.open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter='\t', extrasaction='ignore')
            writer.writeheader(); writer.writerows(rows)


def scheduler_evidence(output, work, logs):
    """Join retained work logs to trace job IDs; accounting queries are read-only."""
    import subprocess
    import re
    traces = {}
    for trace in sorted((output / 'logs').glob('*/trace.tsv')):
        with trace.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row.get('hash'):
                    traces[row['hash']] = row
    work_logs = {}
    for pattern in ('*/*/*.assessment.log', '*/*/results/*.assessment.log'):
        for path in work.glob(pattern):
            relative = path.relative_to(work).parts
            key = relative[0] + '/' + relative[1][:6]
            if key in traces:
                previous = work_logs.get(path.name)
                if previous is None or path.stat().st_mtime > previous[0]:
                    work_logs[path.name] = (path.stat().st_mtime, traces[key])
    ids = sorted({r.get('native_id', '') for _, r in work_logs.values() if re.fullmatch(r'\d+', r.get('native_id', ''))})
    states = {}
    if ids and shutil.which('sacct'):
        try:
            command = ['sacct', '-n', '-P', '-j', ','.join(ids), '--format=JobID,State,ExitCode,ReqMem,MaxRSS,Elapsed']
            result = subprocess.run(command, text=True, capture_output=True, timeout=30)
            (logs / 'slurm-accounting.tsv').write_text(result.stdout)
            (logs / 'slurm-accounting.stderr.log').write_text(result.stderr)
            for line in result.stdout.splitlines():
                parts = line.split('|')
                if len(parts) >= 6:
                    job = parts[0].split('.')[0]
                    if job not in states or parts[1] in ('OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL', 'PREEMPTED'):
                        states[job] = dict(slurm_state=parts[1], slurm_exit=parts[2], max_rss=parts[4])
        except (OSError, subprocess.TimeoutExpired) as exc:
            (logs / 'slurm-accounting.stderr.log').write_text(str(exc))
    result = {}
    for name, (_, row) in work_logs.items():
        job = row.get('native_id', '')
        result[name] = dict(job_id=job, requested_cpus=row.get('cpus', ''), requested_memory=row.get('memory', ''),
                            requested_time=row.get('time', ''), **states.get(job, {}))
    return result
