"""Best-effort controller diagnostics; never change worker resource requests."""
import json
import os
from pathlib import Path
import resource
import sys


def controller_limits():
    limits = {}
    soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    if soft != resource.RLIM_INFINITY:
        limits['process_soft_limit'] = soft
    # Inspect the current cgroup and its ancestors: a parent can impose the
    # effective limit even when the leaf says 'max'. cgroup v1 is not assumed.
    root = Path('/sys/fs/cgroup')
    directories = [root]
    try:
        for entry in Path('/proc/self/cgroup').read_text().splitlines():
            if entry.startswith('0::'):
                leaf = root / entry[3:].lstrip('/')
                if '..' not in leaf.parts:
                    directories += [leaf, *[p for p in leaf.parents if p != root and root in p.parents]]
    except OSError:
        pass
    for filename in ('pids.max', 'memory.max'):
        values = []
        for directory in directories:
            try:
                value = (directory / filename).read_text().strip()
                if value != 'max':
                    values.append(int(value))
            except (OSError, ValueError):
                pass
        if values:
            limits[filename] = min(values)
    return limits


def preflight(output, database):
    limits = controller_limits()
    print('Slurm controller limits: ' + (json.dumps(limits) if limits else 'not detected'), flush=True)
    if not os.environ.get('SLURM_JOB_ID'):
        print('NOTICE: For large runs, submit the controller itself with sbatch; --max_tasks limits worker jobs, not controller threads or memory.', flush=True)
    if min(limits.get('pids.max', 1000000), limits.get('process_soft_limit', 1000000)) <= 1024:
        print('WARNING: Controller process/thread limit is restricted (<=1024); use a suitable controller allocation.', flush=True)
    print(f'Database preparation and Conda environment creation run on this controller host. Database directory: {database}. Missing resources require network access and storage.', flush=True)
    try:
        stat = os.statvfs(output)
        print(f'Shared output filesystem: {stat.f_bavail * stat.f_frsize / 1024**3:.1f} GiB available; {stat.f_favail} free inodes. These are filesystem totals, not your user/project quota; check site quota tools.', flush=True)
    except OSError:
        pass
    return limits


def save_packages(cache, destination):
    """A provenance error must never replace the workflow's original failure."""
    try:
        packages = {}
        for metadata_dir in cache.glob('*/conda-meta'):
            records = []
            for record in sorted(metadata_dir.glob('*.json')):
                installed = json.loads(record.read_text())
                records.append({key: installed.get(key) for key in ('name', 'version', 'build', 'channel', 'sha256')})
            packages[metadata_dir.parent.name] = records
        destination.write_text(json.dumps(packages, indent=2))
    except Exception as exc:
        print(f'WARNING: Could not save environment provenance to {destination}: {exc}', file=sys.stderr, flush=True)


def workload_estimate(genomes, variants, batch_size, max_batch_size, slots, submit_rate, environment_count):
    """Planning envelope; task count is an upper estimate before assessment losses."""
    import math
    n = genomes * variants
    def batches(count, cap, parallel):
        return min(count, max(math.ceil(count / cap), parallel)) if count else 0
    small = 3 * variants * batches(genomes, batch_size, slots['small'])
    large = sum(variants * batches(genomes, max_batch_size, slots[name])
                for name in ('checkm', 'gtdb', 'gunc'))
    # Merges, selection, atlas, ANI and phylogeny; optional branches may reduce it.
    tasks = small + large + 20
    return dict(genomes=genomes, variants=variants, task_upper_estimate=tasks,
                minimum_submission_minutes=round(tasks / submit_rate, 1),
                peak_files_low=40*n + 12*tasks + 10000*environment_count,
                peak_files_high=100*n + 20*tasks + 20000*environment_count,
                caveat='File envelope includes shared task inputs/outputs and environments; excludes database contents and other runs. It is not a quota guarantee.')


def describe_workload(estimate, executor, file_budget=None):
    print(f"Planning estimate: up to approximately {estimate['task_upper_estimate']:,} tasks; "
          f"peak files roughly {estimate['peak_files_low']:,}–{estimate['peak_files_high']:,} including environments.", flush=True)
    if executor == 'slurm':
        print(f"Submission-only lower bound at configured rate: {estimate['minimum_submission_minutes'] / 60:.1f} hours (queueing and execution add time).", flush=True)
    print(estimate['caveat'], flush=True)
    if file_budget and estimate['peak_files_high'] > file_budget:
        print(f'WARNING: Estimated peak files exceed your --file_budget of {file_budget:,}. Reduce shared intermediates or use an adequately sized allocation-local work directory; check cluster quota before proceeding.', flush=True)
