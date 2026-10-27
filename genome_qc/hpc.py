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
