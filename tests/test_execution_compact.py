import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from genome_qc.cli import parser
from genome_qc.execution import configuration
from genome_qc import compact


class ExecutionTests(unittest.TestCase):
    def test_capacity_does_not_lock_in_temporary_memory_usage(self):
        from genome_qc.execution import capacity, memory_bytes
        for available, used in ((70, 40), (105, 5)):
            files = {'/proc/meminfo': f'MemTotal: {120 * 1024**2} kB\nMemAvailable: {available * 1024**2} kB\n',
                     '/sys/fs/cgroup/cpu.max': 'max 100000',
                     '/sys/fs/cgroup/memory.max': 'max',
                     '/sys/fs/cgroup/memory.current': str(used * 1024**3)}
            with patch.object(Path, 'read_text', lambda p: files[str(p)]), \
                    patch('os.sched_getaffinity', return_value=set(range(32))):
                cpus, budget = capacity()
            self.assertEqual(cpus, 32)
            self.assertEqual(memory_bytes(budget), memory_bytes('108 GB'))

    def test_capacity_respects_cgroup_ceiling_and_headroom(self):
        from genome_qc.execution import capacity, memory_bytes
        files = {'/proc/meminfo': f'MemTotal: {120 * 1024**2} kB\n',
                 '/sys/fs/cgroup/cpu.max': '800000 100000',
                 '/sys/fs/cgroup/memory.max': str(40 * 1024**3)}
        with patch.object(Path, 'read_text', lambda p: files[str(p)]), \
                patch('os.sched_getaffinity', return_value=set(range(32))):
            cpus, budget = capacity()
        self.assertEqual(cpus, 8)
        self.assertEqual(memory_bytes(budget), memory_bytes('36 GB'))

    def args(self, *extra):
        return parser().parse_args(['run', '-i', '/inputs', '-o', '/output', *extra])

    def test_slurm_not_headnode_limited(self):
        with patch('genome_qc.execution.capacity', side_effect=AssertionError):
            config, resources = configuration(self.args('--executor', 'slurm', '--account', 'project',
                '--threads', '8', '--max_tasks', '100', '--memory', '64 GB', '--compact_results'), '/cache')
        self.assertEqual(resources['threads'], 8)
        self.assertEqual(resources['max_tasks'], 100)
        self.assertIsNone(resources['cpus'])
        self.assertIn('--nodes=1 --account=project', config)
        self.assertIn('SLURM_TMPDIR', config)
        self.assertIn('process.maxRetries = 0', config)

    def test_budgets_and_defaults(self):
        with patch('genome_qc.execution.capacity', return_value=(32, '256 GB')):
            _, resources = configuration(self.args('--threads', '8'), '/cache')
        self.assertEqual(resources['max_tasks'], 32)
        self.assertEqual(resources['threads'], 8)
        _, resources = configuration(self.args('--executor', 'slurm', '--account', 'x',
            '--max_tasks', '100', '--max_cpus', '32', '--max_memory', '256 GB'), '/cache')
        self.assertEqual(resources['max_tasks'], 2)

    def test_small_server_automatically_uses_disk_backed_placement(self):
        with patch('genome_qc.execution.capacity', return_value=(32, '108 GB')):
            config, resources = configuration(self.args('--threads', '32'), '/cache')
        self.assertTrue(resources['gtdb_disk_backed'])
        self.assertEqual(resources['threads'], 32)
        from genome_qc.execution import memory_bytes
        self.assertLessEqual(memory_bytes(resources['gtdb_memory']), memory_bytes('108 GB'))
        self.assertIn(resources['gtdb_memory'], config)

    def test_large_server_preserves_in_memory_mode(self):
        with patch('genome_qc.execution.capacity', return_value=(32, '256 GB')):
            _, resources = configuration(self.args(), '/cache')
        self.assertFalse(resources['gtdb_disk_backed'])

    def test_two_checkm_batches_fit_without_lowering_gtdb(self):
        from genome_qc.execution import memory_bytes
        with patch('genome_qc.execution.capacity', return_value=(32, '108 GB')):
            config, resources = configuration(self.args('--threads', '8'), '/cache')
        checkm = memory_bytes(resources['task_memory']['checkm'])
        self.assertEqual(checkm, memory_bytes('48 GB'))
        self.assertEqual(memory_bytes(resources['memory']) // checkm, 2)
        self.assertGreater(memory_bytes(resources['gtdb_memory']), checkm)
        self.assertIn("withName: 'CHECKM_SINGLE'", config)
        self.assertNotIn('CHECKM_SINGLE|GUNC', config)

    def test_tool_override_precedence_and_budget(self):
        with patch('genome_qc.execution.capacity', return_value=(32, '108 GB')):
            _, resources = configuration(self.args('--memory', '16 GB',
                '--checkm_memory', '48 GB', '--gtdb_memory', '96 GB'), '/cache')
            self.assertEqual(resources['task_memory']['checkm'], '48 GB')
            self.assertEqual(resources['task_memory']['gunc'], '16 GB')
            self.assertEqual(resources['gtdb_memory'], '96 GB')
            with self.assertRaises(ValueError):
                configuration(self.args('--checkm_memory', '200 GB'), '/cache')

    def test_checkm_high_threads_and_slurm_stage_budget(self):
        with patch('genome_qc.execution.capacity', return_value=(32, '108 GB')):
            _, resources = configuration(self.args('--threads', '32'), '/cache')
        from genome_qc.execution import memory_bytes
        self.assertEqual(memory_bytes(resources['task_memory']['checkm']), memory_bytes('64 GB'))
        args = parser().parse_args(['quality', '-i', '/inputs', '-o', '/output',
            '--executor', 'slurm', '--account', 'x', '--threads', '8',
            '--max_memory', '108 GB', '--max_tasks', '100'])
        _, resources = configuration(args, '/cache')
        self.assertEqual(resources['max_tasks'], 2)

    def test_slurm_small_job_request_uses_disk_backed_mode(self):
        _, resources = configuration(self.args('--executor', 'slurm', '--account', 'x',
                                             '--memory', '64 GB'), '/cache')
        self.assertTrue(resources['gtdb_disk_backed'])

    def test_reject_conflicts(self):
        with self.assertRaises(ValueError):
            configuration(self.args('--executor', 'slurm'), '/cache')
        with self.assertRaises(ValueError):
            configuration(self.args('--memory', '128 GB', '--max_memory', '32 GB'), '/cache')
        with self.assertRaises(ValueError):
            configuration(self.args('--compact_results', '--keep_work', '--max_memory', '256 GB'), '/cache')


class CompactTests(unittest.TestCase):
    def test_cleanup_and_verified_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'output'; output.mkdir()
            for name in ['reports', 'dedupe/fasta', 'genomes_filtered', '.genome_qc/inputs', '.genome_qc/work', '.genome_qc/conda']:
                (output / name).mkdir(parents=True)
            (output / 'reports/results.sqlite').write_text('index')
            master = output / 'Master_genome_QC.tsv'; master.write_text('table')
            (output / 'dedupe/fasta/g.fasta').write_text('>g\nACGT')
            (output / '.genome_qc/work/.command.log').write_text('task log')
            compact.finish(output, 'signature', [output / '.genome_qc/work'],
                           output / '.genome_qc/conda', False, [], root / 'db')
            self.assertTrue(compact.completed(output, 'signature'))
            self.assertFalse((output / '.genome_qc/work').exists())
            self.assertFalse((output / '.genome_qc/conda').exists())
            self.assertTrue((output / 'dedupe/fasta/g.fasta').exists())
            self.assertTrue((output / 'logs/compact-task-logs.tar.gz').exists())
            master.write_text('changed')
            with self.assertRaises(ValueError):
                compact.completed(output, 'signature')

    def test_sources_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'reports').mkdir()
            (output / 'reports/results.sqlite').touch()
            (output / 'Master_genome_QC.tsv').touch()
            protected = output / 'genomes_filtered/input.fa'
            protected.parent.mkdir(); protected.write_text('input')
            with self.assertRaises(ValueError):
                compact.finish(output, 'x', [], output / 'cache', True,
                               [{'fasta': str(protected)}], output / 'db')
            self.assertTrue(protected.exists())

class ControllerLifecycleTests(unittest.TestCase):
    def test_compact_noop_and_failed_run_retains_work(self):
        from genome_qc.cli import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'inputs'; inputs.mkdir()
            (inputs / 'g.fasta').write_text('>g\nACGT\n')
            output = root / 'output'
            args = ['run', '-i', str(inputs), '-o', str(output), '-d', str(root / 'db'),
                    '--max_memory', '256 GB', '--compact_results']
            def execute(command, directory, logfile):
                if logfile.name == 'console.log':
                    (output / 'Master_genome_QC.tsv').write_text('Genome_Id\n' + 'g\n')
                    atlas = output / 'genome_quality_atlas'; atlas.mkdir()
                    (atlas / 'genome_quality_annotated.tsv').write_text('Genome_Id\ng\n')
            with patch('genome_qc.cli.execute', side_effect=execute):
                self.assertEqual(main(args), 0)
            with patch('genome_qc.cli.execute', side_effect=AssertionError('should not execute')):
                self.assertEqual(main(args + ['--resume', '--max_tasks', '2']), 0)
            failed = root / 'failed'
            work = failed / '.genome_qc/work'; work.mkdir(parents=True)
            (work / 'checkpoint').touch()
            args[args.index('-o') + 1] = str(failed)
            with patch('genome_qc.cli.execute', side_effect=RuntimeError('test failure')):
                self.assertEqual(main(args), 1)
            self.assertTrue((work / 'checkpoint').exists())
            self.assertFalse((failed / 'compact-results.json').exists())
