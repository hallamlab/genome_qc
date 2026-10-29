"""Run one assessment, retaining explicit outcomes without blocking other tools."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

CHECKM_HEADER = ['Bin Id', 'Completeness', 'Contamination', 'Strain heterogeneity']


def run(stage, target, receipt, script, inputs, token='0', blocked=False):
    target, receipt = Path(target), Path(receipt)
    log = receipt.with_suffix('.log')
    code = None
    reason = 'Upstream rRNA prediction failed' if blocked else ''
    with log.open('w') as out:
        if not blocked:
            process = subprocess.Popen(['bash', '-euo', 'pipefail', script], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors='replace')
            for line in process.stdout:
                out.write(line); out.flush(); print(line, end='', flush=True)
            process.stdout.close()
            code = process.wait()
            if code:
                reason = f'Assessment command exited {code}; see {log.name}'
            elif not target.exists():
                reason = f'Assessment exited zero but did not create {target.name}'
            elif stage == 'rrna' and not target.with_suffix('.fasta').exists():
                reason = 'rRNA predictor did not create its FASTA output'
            elif stage == 'checkm':
                with target.open() as handle:
                    header = next(csv.reader(handle, delimiter='\t'), [])
                if not set(CHECKM_HEADER).issubset(header):
                    reason = 'CheckM output is missing required columns'
        if reason:
            out.write(reason + '\n')
            print(f'WARNING: {stage}: {reason}', file=sys.stderr, flush=True)
    status = 'blocked' if blocked else 'failed' if reason else 'completed'
    if status != 'completed':
        if target.exists():
            failed = target.with_name(target.name + '.failed')
            if failed.exists():
                shutil.rmtree(failed) if failed.is_dir() else failed.unlink()
            target.rename(failed)
        if stage in ('gtdbtk', 'gunc'):
            target.mkdir(parents=True, exist_ok=True)
            if stage == 'gtdbtk':
                (target / 'classify').mkdir()
        elif stage == 'checkm':
            target.write_text('\t'.join(CHECKM_HEADER) + '\n')
        else:
            target.touch()
        if stage == 'rrna':
            # The GFF target and FASTA must both exist for dependency bookkeeping.
            target.with_suffix('.fasta').write_text('')
    record = dict(stage=stage, status=status, exit_code=code, reason=reason,
                  genomes=[target.stem] if stage == 'preprocess' else [Path(f).name.removesuffix('.fasta') for f in inputs],
                  retry_token=token, finished_utc=datetime.now(timezone.utc).isoformat(),
                  log=log.name)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage'); p.add_argument('target'); p.add_argument('receipt'); p.add_argument('script')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--token', default='0')
    p.add_argument('--upstream')
    args = p.parse_args()
    blocked = args.upstream and json.loads(Path(args.upstream).read_text())['status'] != 'completed'
    run(args.stage, args.target, args.receipt, args.script, args.inputs, args.token, blocked)
