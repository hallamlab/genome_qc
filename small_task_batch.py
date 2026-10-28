"""Bounded batches for lightweight assessments; per-genome outcomes stay explicit."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import assessment_task


def command(parts):
    return shlex.join(map(str, parts))


def assess(stage, identity, target, lines, token, fasta, blocked=False):
    script = Path(identity + '.' + stage + '.sh')
    script.write_text('\n'.join(lines) + '\n')
    try:
        return assessment_task.run(stage, target, identity + '.' + stage + '.assessment.json',
                                   str(script), [fasta], token, blocked)
    finally:
        script.unlink(missing_ok=True)


def preprocess(record, cpus, compact):
    identity, size, fasta, token = record
    identity = identity + '.' + size
    target = Path(identity + '.fasta')
    status = assess('preprocess', identity, str(target),
                    [command(['seqkit', 'seq', '-j', '1', '-g', '-m', '0' if size == 'full' else size,
                              '-o', str(target), fasta])], token, fasta)
    if status['status'] != 'completed':
        target.unlink(missing_ok=True)
        return
    stats = Path(identity + '.seqkit.tsv')
    status = assess('sequence_stats', identity, str(stats),
                    [command(['seqkit', 'stat', '-j', '1', '-a', '-T', str(target)]) + ' > ' + shlex.quote(str(stats))], token, str(target))
    if status['status'] != 'completed':
        stats.unlink(missing_ok=True)
    # Empty assemblies retain their statistics but cannot enter biological tools.
    with target.open() as handle:
        has_sequence = any(line.strip() and not line.startswith('>') for line in handle)
    if not has_sequence:
        target.unlink()


def rrna(record, cpus, compact):
    identity, kingdom, fasta, token, check_token = record
    k = 'arc' if kingdom == 'Archaea' else 'bac'
    status = assess('rrna', identity, identity + '.rRNA.gff',
                    [command(['barrnap', '--kingdom', k, '--threads', '1', '--outseq',
                              identity + '.rRNA.fasta', fasta]) + ' > ' + shlex.quote(identity + '.rRNA.gff')], token, fasta)
    # Database index files are temporary even in keep-everything mode: final
    # self-comparison and predictor outputs remain available for inspection.
    with tempfile.TemporaryDirectory(prefix='.blast-') as tmp:
        db = str(Path(tmp) / 'rrna')
        lines = [f'if [ -s {shlex.quote(identity + ".rRNA.fasta")} ]; then',
                 command(['makeblastdb', '-in', identity + '.rRNA.fasta', '-dbtype', 'nucl', '-out', db]),
                 command(['blastn', '-query', identity + '.rRNA.fasta', '-db', db, '-outfmt', '6',
                          '-out', identity + '.blastout', '-max_target_seqs', '10']),
                 'else', ': > ' + shlex.quote(identity + '.blastout'), 'fi']
        assess('rrna_check', identity, identity + '.blastout', lines, check_token, fasta,
               status['status'] != 'completed')


def trna(record, cpus, compact):
    identity, kingdom, fasta, token = record
    flag = {'Archaea': '-A', 'Bacteria': '-B', 'Eukaryota': '-E'}.get(kingdom, '-G')
    # Run genomes sequentially within this batch; each tRNAscan gets the full
    # task CPU request. This avoids oversubscription of its own worker threads.
    assess('trna', identity, identity + '.gff',
           [command(['tRNAscan-SE', '-q', '-Q', flag, '-o', identity + '.output.txt',
                     '-m', identity + '.stats.txt', '-b', identity + '.bed', '-j', identity + '.gff',
                     '-a', identity + '.trna.fasta', '-l', identity + '.log', '--thread', cpus, fasta])], token, fasta)
    if compact:
        for suffix in ('.output.txt', '.stats.txt', '.bed', '.trna.fasta', '.log'):
            Path(identity + suffix).unlink(missing_ok=True)


def run(mode, records, cpus=1, compact=False):
    required = {'preprocess': ['seqkit'], 'rrna': ['barrnap', 'makeblastdb', 'blastn'], 'trna': ['tRNAscan-SE']}[mode]
    for name in required:
        if not shutil.which(name):
            raise RuntimeError(f'Missing required executable: {name}')
    Path('results').mkdir(exist_ok=True)
    # Outputs from all workers use unique, validated genome IDs. Avoid process-
    # global chdir from worker threads; only change directory before starting.
    input_index = 2
    records = [[*r[:input_index], str(Path(r[input_index]).resolve()), *r[input_index+1:]] for r in records]
    previous = Path.cwd()
    try:
        os.chdir('results')
        worker = {'preprocess': preprocess, 'rrna': rrna, 'trna': trna}[mode]
        with ThreadPoolExecutor(max_workers=1 if mode == 'trna' else cpus) as pool:
            list(pool.map(lambda r: worker(r, cpus, compact), records))
    finally:
        os.chdir(previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['preprocess', 'rrna', 'trna'])
    p.add_argument('manifest')
    p.add_argument('--cpus', type=int, default=1)
    p.add_argument('--compact', action='store_true')
    a = p.parse_args()
    if a.cpus < 1:
        p.error('--cpus must be positive')
    run(a.mode, json.loads(Path(a.manifest).read_text()), a.cpus, a.compact)
