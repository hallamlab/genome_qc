"""Local budgets and conservative Slurm submission controls."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def memory(value):
    if not re.fullmatch(r'\d+(?:\.\d+)?\s*(KB|MB|GB|TB)', value, re.I) or float(re.match(r'[\d.]+', value)[0]) <= 0:
        raise argparse.ArgumentTypeError('use positive memory, e.g. "32 GB"')
    return value.upper()


def memory_bytes(value):
    number, unit = re.fullmatch(r'([\d.]+)\s*(KB|MB|GB|TB)', memory(value)).groups()
    return int(float(number) * 1024 ** (('KB', 'MB', 'GB', 'TB').index(unit) + 1))


def token(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise argparse.ArgumentTypeError('use letters, digits, underscore, dot or hyphen')
    return value


def duration(value):
    if not re.fullmatch(r'[1-9][0-9]*(s|m|h|d)', value):
        raise argparse.ArgumentTypeError('use a positive duration, e.g. 24h')
    return value


def literal(value):
    return json.dumps(str(value)).replace('$', '\\$')


def capacity():
    """Usable hardware capacity, never a snapshot of currently free RAM.

    Nextflow accounts for running task reservations against this ceiling. Using
    MemAvailable (or subtracting memory.current) would permanently charge other
    applications and surviving tasks from an interrupted run to the new run.
    """
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    total = None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal:'):
                total = int(line.split()[1]) * 1024
        quota, period = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if quota != 'max':
            cpus = min(cpus, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            total = min(total, int(limit)) if total else int(limit)
    except (OSError, ValueError):
        pass
    # Reserve headroom once, across the entire run, rather than per task.
    return cpus, f'{max(1, int(total * 0.9) // 1024**2)} MB' if total else None


def add_options(parser, tools=None):
    parser.add_argument('--max_cpus', type=positive, help='Total CPU budget; local defaults to available CPUs, Slurm has no default aggregate cap')
    parser.add_argument('--max_memory', type=memory, help='Total memory budget; local defaults to 90%% of physical/cgroup capacity, not free RAM at startup')
    parser.add_argument('--memory', type=memory, help='Override memory requested per task unless a tool-specific override is supplied')
    for name in (('checkm', 'gtdb', 'gunc', 'trnascan', 'atlas') if tools is None else tools):
        parser.add_argument('--' + name + '_memory', type=memory,
                            help=f'Override {name} task memory independently of other tools')
    parser.add_argument('--max_tasks', type=positive, help='Queued plus running tasks [local: CPU budget; Slurm: 4]')
    parser.add_argument('--executor', choices=['local', 'slurm'], default='local')
    for key in ('account', 'partition', 'qos', 'reservation'):
        parser.add_argument('--' + key, type=token)
    parser.add_argument('--time_limit', type=duration, default='24h')
    parser.add_argument('--submit_rate', type=positive, default=6, help='Slurm submissions per minute [6]')
    parser.add_argument('--work_dir', help='Custom work parent; only this run’s private child is managed')
    parser.add_argument('--conda_cache', help='Shared environment cache; never deleted automatically')
    parser.add_argument('--keep_work', action='store_true', help='Retain work; incompatible with compact output')
    parser.add_argument('--compact_results', action='store_true', help='Use task scratch and compact successful full-workflow outputs')
    parser.add_argument('--scratch_dir', help='Worker-local scratch base; otherwise SLURM_TMPDIR on Slurm, TMPDIR locally')


def configuration(args, cache):
    local_cpus, local_memory = capacity() if args.executor == 'local' else (None, None)
    cpus = args.max_cpus or local_cpus
    budget = args.max_memory or local_memory
    threads = min(args.threads, cpus) if cpus else args.threads
    def request(default, tool=None):
        override = getattr(args, f'{tool}_memory', None) if tool else None
        if override or args.memory:
            return override or args.memory
        if budget:
            # capacity() already leaves headroom in the aggregate budget.
            limit = memory_bytes(budget)
            return f'{max(1, min(memory_bytes(default), limit) // 1024**2)} MB'
        return default
    # Full-tree CheckM needs ~40 GB. Reserve headroom, with more for larger
    # worker counts; do not give every tool GTDB-Tk's much larger reservation.
    task_memory = {name: request(default, name) for name, default in (
        ('checkm', '48 GB' if threads <= 8 else '64 GB'),
        ('gtdb', '128 GB'), ('gunc', '128 GB'),
        ('trnascan', '32 GB'), ('atlas', '64 GB'))}
    active = {'run': tuple(task_memory), 'quality': ('checkm',),
              'taxonomy': ('gtdb',), 'chimera': ('gunc',), 'trna': ('trnascan',)}
    requests = [request('8 GB' if args.command in ('run', 'quality', 'build_db') else '4 GB')]
    requests += [task_memory[name] for name in active.get(args.command, ())]
    largest = max(requests, key=memory_bytes)
    if budget and memory_bytes(largest) > memory_bytes(budget):
        raise ValueError(f'A task requests {largest}, exceeding the memory budget {budget}; adjust the tool memory override, --memory or --max_memory.')
    if args.keep_work and args.compact_results:
        raise ValueError('--compact_results and --keep_work cannot be combined')
    if args.compact_results and args.command != 'run':
        raise ValueError('--compact_results requires the full run so downstream products exist before cleanup')
    jobs = args.max_tasks or (cpus if args.executor == 'local' else 4)
    lines = [f'conda.cacheDir = {literal(cache)}', f'process.executor = {literal(args.executor)}',
             "process.errorStrategy = 'finish'", 'process.maxRetries = 0',
             f'process.time = {literal(args.time_limit)}', f'process.memory = {literal(request("4 GB"))}']
    for name, tool in [('CHECKM_SINGLE', 'checkm'), ('GTDBTK', 'gtdb'), ('GUNC', 'gunc'),
                       ('TRNASCAN', 'trnascan'), ('GENOME_QUALITY_ATLAS', 'atlas'), ('COHORT_ANI', 'atlas'), ('COHORT_PHYLOGENY', 'atlas')]:
        lines.append(f"process {{ withName: '{name}' {{ memory = '{task_memory[tool]}' }} }}")
    lines.append(f"process {{ withName: 'MERGE_MASTER|DEDUPE|QSCORE' {{ memory = '{request('8 GB')}' }} }}")
    if args.executor == 'local':
        if any(getattr(args, key) for key in ('account', 'partition', 'qos', 'reservation')):
            raise ValueError('Slurm options require --executor slurm')
        lines.append(f'executor.cpus = {cpus}')
        if budget:
            lines.append(f'executor.memory = {literal(budget)}')
    else:
        if not args.account:
            raise ValueError('Slurm requires --account')
        if cpus:
            jobs = min(jobs, cpus // threads)
        if budget:
            jobs = min(jobs, memory_bytes(budget) // memory_bytes(largest))
        options = ['--nodes=1'] + [f'--{key}={getattr(args,key)}' for key in ('account', 'qos', 'reservation') if getattr(args,key)]
        lines += [f'process.clusterOptions = {literal(" ".join(options))}',
                  f'executor.submitRateLimit = {literal(str(args.submit_rate) + "/1min")}',
                  "executor.queueStatInterval = '1min'", "executor.pollInterval = '10sec'"]
        if args.partition:
            lines.append(f'process.queue = {literal(args.partition)}')
    lines.append(f'executor.queueSize = {jobs}')
    if args.compact_results or args.scratch_dir:
        scratch = args.scratch_dir or ('$SLURM_TMPDIR' if args.executor == 'slurm' else '$TMPDIR')
        lines += [f'process.scratch = {literal(scratch)}', "process.stageInMode = 'copy'", "process.stageOutMode = 'copy'"]
        if args.executor == 'slurm' and not args.scratch_dir:
            guard = 'test -n "${SLURM_TMPDIR:-}" && test -d "$SLURM_TMPDIR" && test -w "$SLURM_TMPDIR" || { echo "Job-local SLURM_TMPDIR missing; provide --scratch_dir" >&2; exit 1; }'
            lines.append(f'process.beforeScript = {literal(guard)}')
    batch_slots = {}
    for tool in ('checkm', 'gtdb', 'gunc'):
        limits = [jobs]
        if cpus:
            limits.append(max(1, cpus // threads))
        if budget:
            limits.append(max(1, memory_bytes(budget) // memory_bytes(task_memory[tool])))
        batch_slots[tool] = max(1, min(limits))
    return '\n'.join(lines) + '\n', dict(cpus=cpus, memory=budget, threads=threads, max_tasks=jobs, task_memory=task_memory, batch_slots=batch_slots, gtdb_memory=task_memory['gtdb'],
        gtdb_disk_backed=memory_bytes(task_memory['gtdb']) < memory_bytes('128 GB'))
