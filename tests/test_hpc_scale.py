import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import database_install as db
from genome_qc.cli import main
from genome_qc.execution import capacity, memory_bytes
from genome_qc.hpc import workload_estimate
from genome_qc.inputs import discover


class HpcScaleTests(unittest.TestCase):
    def test_slurm_allocation_not_node_memory(self):
        files = {'/proc/meminfo': 'MemTotal: 786432000 kB\n',
                 '/sys/fs/cgroup/cpu.max': 'max 100000',
                 '/proc/self/cgroup': '0::/slurm/job42/step0',
                 '/sys/fs/cgroup/slurm/job42/memory.max': str(40 * 1024**3)}
        def read(p, *a, **k):
            if str(p) not in files: raise FileNotFoundError(str(p))
            return files[str(p)]
        with patch.object(Path, 'read_text', read), patch('os.sched_getaffinity', return_value=set(range(192))), \
                patch.dict(os.environ, {'SLURM_MEM_PER_NODE': '32768', 'SLURM_CPUS_PER_TASK': '16'}, clear=True):
            cpus, mem = capacity()
        self.assertEqual(cpus, 16)
        self.assertLessEqual(memory_bytes(mem), 32 * 1024**3 * .9)
        self.assertGreater(memory_bytes(mem), 28 * 1024**3)

    def test_slurm_per_cpu_and_zero_node_memory(self):
        with patch('genome_qc.hpc.controller_limits', return_value={}), \
                patch.dict(os.environ, {'SLURM_MEM_PER_NODE': '0', 'SLURM_MEM_PER_CPU': '2048', 'SLURM_CPUS_ON_NODE': '8'}, clear=True):
            cpus, mem = capacity()
        self.assertLessEqual(cpus, 8)
        self.assertLessEqual(memory_bytes(mem), 16 * 1024**3 * .9)

    def test_large_cohort_plan_is_batched(self):
        result = workload_estimate(10210, 1, 128, 1000, dict(small=4, checkm=4, gtdb=4, gunc=4), 6, 17)
        self.assertLess(result['task_upper_estimate'], 400)
        self.assertLess(result['minimum_submission_minutes'], 70)

    def test_nonrecursive_and_plan_do_not_stage_genomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = root / 'inputs'; inputs.mkdir()
            (inputs / 'a.fa').write_text('>a\nACGT\n')
            nested = inputs / 'excluded'; nested.mkdir(); (nested / 'b.fa').write_text('>b\nACGT\n')
            self.assertEqual(len(discover(inputs)), 2)
            self.assertEqual(len(discover(inputs, recursive=False)), 1)
            output = root / 'output'
            self.assertEqual(main(['run', '-i', str(inputs), '-o', str(output), '--no_recursive', '--plan']), 0)
            self.assertFalse((output / '.genome_qc/inputs').exists())

    def test_fresh_work_preserves_inputs_and_resume_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); inputs = root / 'inputs'; inputs.mkdir(); (inputs / 'a.fa').write_text('>a\nACGT\n')
            output = root / 'out'
            base = ['preprocess', '-i', str(inputs), '-o', str(output)]
            calls = []
            def execute(cmd, directory, logfile):
                calls.append((cmd, directory))
            with patch('genome_qc.cli.execute', execute), patch('genome_qc.report.build'):
                self.assertEqual(main(base + ['--work_dir', str(root / 'job1')]), 0)
                self.assertEqual(main(base + ['--work_dir', str(root / 'job2')]), 1)
                self.assertEqual(main(base + ['--work_dir', str(root / 'job2'), '--fresh_work']), 0)
                fresh = calls[-1]
                self.assertEqual(main(base + ['--work_dir', str(root / 'job2'), '--resume']), 0)
                self.assertEqual(fresh[1], calls[-1][1])
                self.assertEqual(fresh[0][fresh[0].index('-work-dir') + 1], calls[-1][0][calls[-1][0].index('-work-dir') + 1])
                (inputs / 'a.fa').write_text('>a\nAAAA\n')
                self.assertEqual(main(base + ['--work_dir', str(root / 'job3'), '--fresh_work']), 1)

    def test_gunc_supplied_archive_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root / 'gunc_db_progenomes2.1.dmnd.gz'
            data = b'diamond fixture\n'
            archive.write_bytes(gzip.compress(data))
            spec = ('https://example.invalid/' + archive.name, hashlib.md5(archive.read_bytes()).hexdigest())
            with patch.dict(db.SPECS, gunc=spec), patch.object(db, 'GUNC_DMND_MD5', hashlib.md5(data).hexdigest()), \
                    patch.object(db, 'download', side_effect=AssertionError('network')):
                db.install('gunc', root / 'gunc_db', extract_only=True)
            self.assertEqual((root / 'gunc_db/gunc_db_progenomes2.1.dmnd').read_bytes(), data)
            self.assertTrue(archive.exists())

    def test_download_only_and_missing_offline_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'no download attempted'):
                db.install('gunc', root / 'gunc_db', extract_only=True)
            data = b'archive'
            spec = ('https://example.invalid/gunc.gz', hashlib.md5(data).hexdigest())
            def download(url, archive, expected, attempts):
                Path(archive).parent.mkdir(parents=True, exist_ok=True)
                Path(archive).write_bytes(data)
                return db.hashes(archive)
            with patch.dict(db.SPECS, gunc=spec), patch.object(db, 'download', download):
                db.install('gunc', root / 'gunc_db', download_only=True, url='https://mirror.invalid/gunc.gz')
            self.assertFalse((root / 'gunc_db').exists())
            self.assertEqual((root / 'gunc.gz').read_bytes(), data)
            self.assertIn('mirror.invalid', (root / 'gunc.gz.provenance.json').read_text())

    def test_stale_cleanup_only_marked_matching_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dest = root / 'gunc_db'; dest.mkdir()
            (dest / 'gunc_db_progenomes2.1.dmnd').write_bytes(b'valid')
            owned = root / '.gunc-install-old'; owned.mkdir()
            (owned / '.genome-qc-install.json').write_text(json.dumps({'destination': str(dest)}))
            other = root / '.gunc-install-unknown'; other.mkdir()
            db.install('gunc', dest)
            self.assertFalse(owned.exists()); self.assertTrue(other.exists())
