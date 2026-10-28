"""Canonical genome identities and reproducible input staging."""
import csv
import gzip
import hashlib
import json
import re
import shutil
from pathlib import Path

SUFFIXES = ('.fasta.gz', '.fna.gz', '.fa.gz', '.fasta', '.fna', '.fa')


def discover(directory=None, manifest=None, recursive=True):
    rows = []
    if manifest:
        manifest = Path(manifest).resolve()
        with manifest.open(newline='') as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            if not {'id', 'fasta'} <= set(reader.fieldnames or []):
                raise ValueError('Manifest requires tab-separated id and fasta columns.')
            for row in reader:
                path = Path(row['fasta']).expanduser()
                rows.append({**row, 'fasta': str((manifest.parent / path).resolve())})
    else:
        root = Path(directory).expanduser().resolve()
        for path in sorted(root.rglob('*') if recursive else root.glob('*')):
            suffix = next((s for s in SUFFIXES if path.name.lower().endswith(s)), None)
            if path.is_file() and suffix:
                rows.append({'id': path.name[:-len(suffix)], 'fasta': str(path)})
    if not rows:
        raise ValueError('No genome FASTA inputs found.')
    seen = set()
    for row in rows:
        identity = row['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', identity):
            raise ValueError(f'Unsafe genome ID {identity!r}; use letters, digits, dot, underscore or dash.')
        if identity in seen:
            raise ValueError(f'Duplicate genome ID {identity!r}; assign unique IDs in a manifest.')
        seen.add(identity)
        path = Path(row['fasta'])
        if not path.is_file():
            raise ValueError(f'Missing FASTA: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        row['sha256'] = digest.hexdigest()
    return sorted(rows, key=lambda row: row['id'])


def stage(rows, output):
    output = Path(output)
    state = output / 'inputs.resolved.json'
    if state.exists() and json.loads(state.read_text()) != rows:
        raise ValueError('Inputs have changed; use a new output directory.')
    directory = output / '.genome_qc' / 'inputs'
    directory.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = directory / (row['id'] + '.fasta')
        if target.exists() and state.exists():
            continue
        temporary = target.with_suffix('.tmp')
        opener = gzip.open if row['fasta'].endswith('.gz') else open
        try:
            with opener(row['fasta'], 'rb') as source, temporary.open('wb') as dest:
                if source.read(1) != b'>':
                    raise ValueError(f"Not a FASTA file: {row['fasta']}")
                dest.write(b'>')
                shutil.copyfileobj(source, dest)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    pending_state = state.with_suffix('.json.tmp')
    pending_state.write_text(json.dumps(rows, indent=2) + '\n')
    pending_state.replace(state)
    return directory
