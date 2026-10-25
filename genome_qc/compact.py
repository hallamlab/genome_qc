"""Success-only cleanup with a verifiable completed-run checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

PRODUCT_SUFFIXES = {'.tsv', '.csv', '.json', '.pdf', '.png', '.svg', '.html', '.sqlite', '.fasta', '.fna', '.fa', '.gz', '.tree', '.nwk'}


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(rows, scientific, implementation):
    return hashlib.sha256(json.dumps([rows, scientific, implementation], sort_keys=True).encode()).hexdigest()


def completed(output, expected):
    marker = Path(output) / 'compact-results.json'
    if not marker.exists():
        return False
    state = json.loads(marker.read_text())
    if state['signature'] != expected:
        raise ValueError('Compacted results have different inputs, analysis settings or implementation; use a new output directory.')
    for relative, digest in state['products'].items():
        path = Path(output) / relative
        if not path.is_file() or checksum(path) != digest:
            raise ValueError(f'Compacted result missing or changed: {path}; use a new output directory to regenerate it.')
    return True


def remove(path):
    path = Path(path)
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def finish(output, expected, work_paths, cache, preserve_cache, inputs, database):
    output = Path(output).resolve()
    if not (output / 'reports/results.sqlite').is_file() or not (output / 'Master_genome_QC.tsv').is_file():
        raise ValueError('Cannot compact without completed master table and report index')
    status_table = output / 'Assessment_status.tsv'
    if status_table.exists():
        import csv
        with status_table.open() as handle:
            if any(r['status'] in ('failed', 'blocked', 'not_completed') for r in csv.DictReader(handle, delimiter='\t')):
                raise ValueError('Cannot finalize compact output with unresolved assessment failures; retain work for resume')
    targets = [output / name for name in ('genomes_filtered', 'genomes_pruned', 'genomes_subset')]
    targets += list(work_paths) + [output / '.genome_qc/inputs']
    if not preserve_cache:
        targets.append(cache)
    protected = [Path(row['fasta']).resolve() for row in inputs] + [Path(database).resolve()]
    if preserve_cache:
        protected.append(Path(cache).resolve())
    for target in targets:
        if any(path == Path(target).resolve() or path.is_relative_to(Path(target).resolve()) for path in protected):
            raise ValueError(f'Cleanup target contains input data or reference databases: {target}')
    # Keep every scientific table/figure and final selected genome. Work/cache
    # never enter the product inventory; hashes make later no-op resume explicit.
    products = {}
    for directory, dirs, files in os.walk(output):
        dirs[:] = [name for name in dirs if not name.startswith('.') and name not in ('logs', 'genomes_filtered', 'genomes_pruned', 'genomes_subset') and Path(directory, name).resolve() not in {Path(p).resolve() for p in [*targets, cache]}]
        for name in files:
            path = Path(directory) / name
            if name != 'compact-results.json' and path.suffix in PRODUCT_SUFFIXES:
                products[str(path.relative_to(output))] = checksum(path)
    archive = output / 'logs' / 'compact-task-logs.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as bundle:
            for i, work in enumerate(work_paths):
                work = Path(work)
                if work.is_dir():
                    for path in work.rglob('.command*'):
                        if path.is_file() and not path.is_symlink():
                            bundle.add(path, arcname=str(Path(str(i)) / path.relative_to(work)))
        temporary.replace(archive)
    marker = output / 'compact-results.json'
    pending = output / '.compact-results.json.tmp'
    pending.write_text(json.dumps(dict(signature=expected, products=products, cleanup_complete=False), indent=2))
    # Commit completion before removing checkpoints, so an interrupted cleanup
    # can be retried without rerunning the scientific workflow.
    pending.replace(marker)
    for target in targets:
        remove(target)
    state = json.loads(marker.read_text()); state['cleanup_complete'] = True
    pending.write_text(json.dumps(state, indent=2)); pending.replace(marker)
