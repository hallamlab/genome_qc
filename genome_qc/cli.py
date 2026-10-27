"""Genome QC controller. Scientific stages remain shared Nextflow processes."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import uuid

from . import __version__
from .inputs import discover, stage
from .runtime import runtime_root
from .execution import add_options, configuration, literal, memory_bytes, capacity
from . import compact
from .hpc import preflight, save_packages

ENTRIES = {'run': None, 'build_db': 'DATABASES', 'preprocess': 'PREPROCESS',
           'quality': 'CHECKM_ONLY', 'taxonomy': 'TAXONOMY_ONLY',
           'rrna': 'RRNA_ONLY', 'trna': 'TRNA_ONLY', 'chimera': 'GUNC_ONLY'}
SCRIPTS = {'atlas': 'genome_quality_atlas.py', 'compare': 'genome_quality_atlas_wrapper.py',
           'phylogeny': 'best_set_phylogeny.py', 'render': 'render_phylogeny_ete.py'}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def batch_limit(value):
    value = positive(value)
    if value > 1000:
        raise argparse.ArgumentTypeError('maximum batch size is 1000 genomes')
    return value


def finite(value):
    import math
    value = float(value)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError('must be a finite number')
    return value


def percentage(value):
    value = finite(value)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError('must be between 0 and 100')
    return value


def fraction(value):
    value = finite(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return value


def parser():
    p = argparse.ArgumentParser(prog='genome_qc', description='Genome quality assessment and genome atlas')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ENTRIES:
        s = sub.add_parser(name)
        if name != 'build_db':
            source = s.add_mutually_exclusive_group(required=True)
            source.add_argument('-i', '--input', help='Genome FASTA directory (searched recursively)')
            source.add_argument('--manifest', help='TSV with id and fasta; relative paths resolve beside the manifest')
        s.add_argument('-o', '--output', required=True)
        s.add_argument('-d', '--database', default='~/.local/share/genome_qc/databases')
        s.add_argument('--threads', type=positive, default=8)
        add_options(s.add_argument_group('Execution and resources'), tools=None if name == 'run' else {'quality': ('checkm',), 'taxonomy': ('gtdb',), 'trna': ('trnascan',), 'chimera': ('gunc',)}.get(name, ()))
        s.add_argument('--resume', action='store_true')
        s.add_argument('--plan', action='store_true', help='Validate inputs and save commands without executing')
        s.add_argument('--params', type=Path, help='Advanced Nextflow parameters as a JSON object')
        if name in ('run', 'quality', 'taxonomy', 'chimera'):
            s.add_argument('--max_batch_size', type=batch_limit, default=1000, help='Maximum genomes per balanced batch (default: 1000); smaller datasets use resource-aware batches')
        if name in ('run', 'preprocess'):
            g = s.add_argument_group('Preprocessing')
            g.add_argument('--filter_size', help='Contig-length variants; default: retain all contigs')
        if name == 'run':
            g = s.add_argument_group('Final selection (default: retain every assessed genome)')
            for flag, description in (
                ('min_genome_size', 'Minimum genome length (bp)'),
                ('max_genome_size', 'Maximum genome length (bp)'),
                ('completeness', 'Minimum CheckM completeness (%%)'),
                ('contamination', 'Maximum CheckM contamination (%%)'),
                ('qscore_min', 'Minimum Qscore')):
                g.add_argument('--' + flag, type=finite, help=description + '; default: unrestricted')
            g.add_argument('--require_rrna_consistency', action='store_true', default=None,
                           help='Require the rRNA consistency check to pass; default: disabled')
            g = s.add_argument_group('Atlas')
            g.add_argument('--ani_threshold', type=percentage, help='ANI grouping threshold (%%); default: 98')
            g.add_argument('--ani_af_threshold', type=fraction, help='ANI alignment fraction (0–1); default: 0.51')
            g.add_argument('--no_ani_containment', action='store_true', help='Disable containment rescue (default: smaller AF >=0.90 and larger AF >=0.10)')
            g.add_argument('--skip_ani', action='store_true', help='Skip cohort ANI grouping')
            g.add_argument('--skip_phylogeny', action='store_true', help='Skip GTDB-marker phylogeny and annotated trees')
        if name in ('rrna', 'trna'):
            s.add_argument('--kingdom', choices=['Bacteria', 'Archaea', 'Eukaryota'], default='Bacteria')
    for name in SCRIPTS:
        sub.add_parser(name, add_help=False, help=f'{name} analysis; use {name} --help for options')
    sub.add_parser('check', help='Check controller dependencies without downloading data')
    s = sub.add_parser('report', help='Index output tables and open the explorer')
    s.add_argument('-o', '--output', required=True)
    s.add_argument('--serve', action='store_true')
    s.add_argument('--port', type=positive, default=8765)
    return p


def execute(command, directory, logfile):
    print('Command:', shlex.join(command), flush=True)
    with logfile.open('w') as log:
        with subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            result = process.wait()
    if result:
        raise RuntimeError(f'Workflow exited {result}; see {logfile}. Reuse this output with --resume.')


def run(args):
    root = runtime_root()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    control = output / '.genome_qc'
    control.mkdir(exist_ok=True)
    with (control / 'controller.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another controller is using {output}') from None
        _run(args, root, output, control)


def _run(args, root, output, control):
    params = json.loads(args.params.read_text()) if args.params else {}
    if not isinstance(params, dict):
        raise ValueError('--params must contain a JSON object')
    for key in ('filter_size', 'min_genome_size', 'max_genome_size', 'completeness',
                'contamination', 'qscore_min', 'ani_threshold', 'ani_af_threshold', 'require_rrna_consistency'):
        value = getattr(args, key, None)
        if value is not None:
            params['atlas_' + key if key in ('ani_threshold', 'ani_af_threshold') else key] = value
    if args.command == 'run':
        if args.no_ani_containment: params['cohort_ani_containment'] = False
        if args.skip_ani: params['cohort_ani_enabled'] = False
        if args.skip_phylogeny: params['cohort_phylogeny_enabled'] = False
    cache = Path(args.conda_cache).expanduser().resolve() if args.conda_cache else control / 'conda'
    work = (Path(args.work_dir).expanduser().resolve() / ('genome-qc-' + hashlib.sha256(str(output).encode()).hexdigest()[:16])) if args.work_dir else control / 'work'
    locations = {'work': str(work), 'cache': str(cache)}
    locations_file = control / 'execution-paths.json'
    if locations_file.exists() and json.loads(locations_file.read_text()) != locations:
        raise ValueError('Work/cache locations changed; keep the original paths for resume or use a new output directory.')
    config, budgets = configuration(args, cache)
    if args.executor == 'local':
        source = 'explicit limit' if args.max_memory else '90% of physical/cgroup capacity'
        print(f"Local scheduling: CPU budget={budgets['cpus']}; memory budget={budgets['memory']} ({source}); jobs start as reservations become available", flush=True)
    print('Task memory reservations: ' + ', '.join(
        f'{name}={value}' for name, value in budgets['task_memory'].items()), flush=True)
    if args.command in ('run', 'taxonomy'):
        mode = f"disk-backed; pplacer uses {min(8, budgets['threads'])} CPUs" if budgets['gtdb_disk_backed'] else 'in-memory'
        print(f"GTDB placement: {mode}; task memory={budgets['gtdb_memory']}; other parallel tools use up to {budgets['threads']} CPUs", flush=True)
    threads = budgets['threads']
    locations_file.write_text(json.dumps(locations, indent=2))
    database = Path(args.database).expanduser().resolve()
    controller_limits = preflight(output, database) if args.executor == 'slurm' else {}
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    for asset in sorted([*root.glob('*.py'), root / 'main.nf', root / 'nextflow.config', *root.glob('envs/*.yaml')]):
        digest.update(str(asset.relative_to(root)).encode())
        digest.update(asset.read_bytes())
    params['implementation_digest'] = digest.hexdigest()
    for key in ('max_forks', 'checkm_max_forks', 'gtdbtk_max_forks', 'trnascan_max_forks'):
        params.setdefault(key, budgets['max_tasks'])
    for tool, prefix in (('checkm', 'checkm'), ('gtdb', 'gtdbtk'), ('gunc', 'gunc')):
        params[prefix + '_parallel_batches'] = budgets['batch_slots'][tool]
        cap = int(params.get(prefix + '_batch_size', getattr(args, 'max_batch_size', 1000)))
        if cap < 1:
            raise ValueError('Batch sizes must be positive')
        params[prefix + '_batch_size'] = min(cap, getattr(args, 'max_batch_size', 1000), 1000)
    if args.command in ('run', 'quality', 'taxonomy', 'chimera'):
        active = {'run': ('checkm', 'gtdb', 'gunc'), 'quality': ('checkm',), 'taxonomy': ('gtdb',), 'chimera': ('gunc',)}[args.command]
        print('Balanced batching (FASTA bytes; maximum 1000 genomes): ' + ', '.join(f"{tool} up to {budgets['batch_slots'][tool]} concurrent batches" for tool in active), flush=True)
    params['compact_results'] = args.compact_results
    params.update(stage=args.command, working_dir=str(output), ref_dir='', checkm_data_path=str(database / 'checkm_data'),
                  gtdbtk_data_path=str(database / 'gtdbtk_r232'),
                  gunc_db=str(database / 'gunc_db/gunc_db_progenomes2.1.dmnd'))
    for key in ('checkm_cpus', 'gtdbtk_cpus', 'gunc_cpus', 'trnascan_cpus', 'atlas_ani_threads', 'atlas_cpus'):
        params[key] = threads
    rows = []
    if args.command != 'build_db':
        rows = discover(args.input, args.manifest)
        params['fasta_dir'] = str(control / 'inputs')
        if args.command == 'rrna':
            kingdom_map = control / 'kingdoms.tsv'
            kingdom_map.write_text(''.join(f"{r['id']}\t{args.kingdom}\n" for r in rows))
            params['kingdom_map'] = str(kingdom_map)
        params['kingdom'] = getattr(args, 'kingdom', 'Bacteria')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    logs = output / 'logs' / stamp
    logs.mkdir(parents=True)
    cfg = logs / 'controller.config'
    cfg.write_text(config)
    parameters = logs / 'params.json'
    parameters.write_text(json.dumps(params, indent=2) + '\n')
    scientific = {key: value for key, value in params.items()
                  if key not in ('implementation_digest', 'compact_results', 'max_forks') and not key.endswith(('_cpus', '_threads', '_memory', '_time', '_max_forks', '_parallel_batches'))}
    identity = compact.signature(rows, scientific, params['implementation_digest'])
    cleanup_paths = [work, control / 'database-work', control / 'analysis-session', control / 'database-session']
    if compact.completed(output, identity):
        if not json.loads((output / 'compact-results.json').read_text())['cleanup_complete']:
            compact.finish(output, identity, cleanup_paths, cache, bool(args.conda_cache), rows, database)
        print(f'Completed compact results verified: {output / "reports"}')
        return
    if rows:
        stage(rows, output)
    settings_path = control / (args.command + '.settings.json')
    if settings_path.exists() and json.loads(settings_path.read_text()) != scientific:
        raise ValueError('Analysis settings changed for this output directory; use a new output directory to avoid mixing results.')
    settings_path.write_text(json.dumps(scientific, indent=2))
    executable = shutil.which('nextflow') or 'nextflow'
    command = [executable, '-c', str(cfg), '-log', str(logs / 'nextflow.log'),
               'run', str(root / 'main.nf'), '-params-file', str(parameters),
               '-work-dir', str(work), '-with-trace', str(logs / 'trace.tsv'),
               '-with-report', str(logs / 'execution.html'), '-with-timeline', str(logs / 'timeline.html')]
    if args.resume:
        command.append('-resume')
    preparation = None
    if args.command in ('run', 'build_db', 'quality', 'taxonomy', 'chimera'):
        dbparams = logs / 'databases.json'
        required = {'quality': ['checkm'], 'taxonomy': ['gtdbtk'], 'chimera': ['gunc']}.get(
            args.command, ['checkm', 'gtdbtk', 'gunc'])
        db_cpus = min(threads, capacity()[0])
        dbparams.write_text(json.dumps({**params, 'ref_dir': str(database), 'stage': 'build_db',
                                       'databases': required, 'reference_cpus': db_cpus}, indent=2))
        dbconfig = logs / 'databases.config'
        db_memory = f"{min(8192, memory_bytes(budgets['memory']) // 1024**2)} MB" if budgets['memory'] else '8 GB'
        dbconfig.write_text(f"process.executor = 'local'\nprocess.memory = {literal(db_memory)}\nexecutor.cpus = {db_cpus}\nexecutor.queueSize = 1\nconda.cacheDir = {literal(cache)}\n")
        preparation = [executable, '-c', str(dbconfig), '-log', str(logs / 'databases.log'),
                       'run', str(root / 'main.nf'),
                       '-params-file', str(dbparams), '-work-dir', str(control / 'database-work')]
    receipt = {'invocation': vars(args).copy(), 'resources': budgets, 'version': __version__, 'command': command, 'database_command': preparation, 'params': params,
               'status': 'planned', 'started_utc': stamp, 'controller_limits': controller_limits}
    receipt_path = logs / 'run.json'
    receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
    if args.plan:
        print(f'Plan saved: {receipt_path}')
        return
    try:
        if preparation:
            print(f'Preparing required databases in {database}; existing complete databases are reused.', flush=True)
            db_cwd = control / 'database-session'
            db_cwd.mkdir(exist_ok=True)
            execute(preparation, db_cwd, logs / 'database-console.log')
        if args.command != 'build_db':
            print(f'Running {args.command}: {len(rows)} genomes; {threads} threads per parallel tool; executor={args.executor}', flush=True)
            analysis_cwd = control / 'analysis-session'
            analysis_cwd.mkdir(exist_ok=True)
            execute(command, analysis_cwd, logs / 'console.log')
            if args.command == 'run':
                expected = [output / 'Master_genome_QC.tsv', output / 'genome_quality_atlas' / (params.get('atlas_prefix', 'genome_quality') + '_annotated.tsv')]
                missing = [str(path) for path in expected if not path.is_file()]
                if missing:
                    raise RuntimeError('Workflow did not produce required final outputs: ' + ', '.join(missing))
                write_inventory(rows, output)
            from .report import build
            build(output)
        receipt['status'] = 'complete'
        status_table = output / 'Assessment_status.tsv'
        if status_table.exists():
            with status_table.open() as handle:
                failures = [r for r in csv.DictReader(handle, delimiter='\t')
                            if r['status'] in ('failed', 'blocked', 'not_completed')]
            inventory = output / 'Genome_inventory.tsv'
            if inventory.exists():
                with inventory.open() as handle:
                    failures += [r for r in csv.DictReader(handle, delimiter='\t') if r['status'] == 'not_assessed']
            receipt['assessment_failures'] = len(failures)
            if failures:
                receipt['status'] = 'complete_with_assessment_failures'
                print(f'WARNING: {len(failures)} genome/tool assessments failed or could not complete; see {status_table}. All available results were retained.', flush=True)
    except BaseException:
        receipt['status'] = 'failed'
        raise
    finally:
        try:
            receipt_path.write_text(json.dumps(receipt, indent=2, default=str))
        except Exception as exc:
            print(f'WARNING: Could not update run receipt {receipt_path}: {exc}', file=sys.stderr)
        save_packages(cache, logs / 'environment-packages.json')

    if args.compact_results and receipt['status'] == 'complete_with_assessment_failures':
        print('Retaining work and cache for failed-assessment recovery; compact completion is deferred.')
    elif args.compact_results:
        compact.finish(output, identity, cleanup_paths, cache, bool(args.conda_cache), rows, database)
        print(f'Compact results saved: {output}; logs and final products retained.')


def write_inventory(inputs, output):
    """Keep every input visible and distinguish assessment from final selection."""
    master_path = output / 'Master_genome_QC.tsv'
    if master_path.exists():
        with master_path.open() as handle:
            reader = csv.DictReader(handle, delimiter='\t')
            new_selection = 'selected' in (reader.fieldnames or [])
            assessments = {row['Genome_Id']: row for row in reader}
        if new_selection:
            with (output / 'Genome_inventory.tsv').open('w', newline='') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['Genome_Id', 'sample', 'category', 'input_fasta', 'input_sha256', 'status', 'selection_reasons'])
                for row in inputs:
                    assessment = assessments.get(row['id'])
                    status = ('assessed_selected' if assessment['selected'].lower() == 'true' else 'assessed_not_selected') if assessment else 'not_assessed'
                    if assessment and assessment.get('assessment_complete', 'true').lower() == 'false':
                        status = 'assessed_incomplete'
                    reason = assessment['selection_reasons'] if assessment else 'No usable preprocessed variant; review preprocessing outputs and task logs'
                    writer.writerow([row['id'], row.get('sample', ''), row.get('category', ''), row['fasta'], row['sha256'], status, reason])
            return
    def ids(relative):
        path = output / relative
        if not path.exists():
            return set()
        with path.open() as handle:
            return {row['Genome_Id'] for row in csv.DictReader(handle, delimiter='\t')}
    qscore = ids('qscore/qscore_mqhp.tsv')
    final = ids('Master_genome_QC.tsv')
    with (output / 'Genome_inventory.tsv').open('w', newline='') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['Genome_Id', 'sample', 'category', 'input_fasta', 'input_sha256', 'status'])
        for row in inputs:
            identity = row['id']
            status = ('included_in_atlas' if identity in final else
                      'excluded_after_rRNA_check' if identity in qscore else 'excluded_by_size_or_quality')
            writer.writerow([identity, row.get('sample', ''), row.get('category', ''), row['fasta'], row['sha256'], status])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SCRIPTS:
        return subprocess.call([sys.executable, str(runtime_root() / SCRIPTS[argv[0]]), *argv[1:]])
    args = parser().parse_args(argv)
    try:
        if args.command == 'check':
            missing = []
            for name in ('nextflow', 'java', 'mamba'):
                path = shutil.which(name)
                print(f'{name}: {path or "MISSING"}')
                if not path:
                    missing.append(name)
            return int(bool(missing))
        if args.command == 'report':
            from .report import build, serve
            report_output = Path(args.output).expanduser().resolve()
            if not (report_output / 'compact-results.json').exists():
                build(report_output)
            if args.serve:
                serve(Path(args.output).expanduser().resolve(), args.port)
        else:
            run(args)
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        print(f'Genome QC: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
