import csv
import gzip
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import tarfile

import database_install

from genome_qc.inputs import discover, stage
from genome_qc.report import build, query
from genome_qc.cli import main


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'input'; self.inputs.mkdir()
        (self.inputs / 'a.fna').write_text('>contig\nACGT\n')
        self.output = self.root / 'out'

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_and_gzip_identity(self):
        with gzip.open(self.inputs / 'b.fa.gz', 'wt') as handle:
            handle.write('>x\nACGT\n')
        manifest = self.root / 'manifest.tsv'
        manifest.write_text('id\tfasta\tsample\ncustom\tinput/b.fa.gz\tS1\n')
        rows = discover(manifest=manifest)
        result = stage(rows, self.output)
        self.assertEqual((result / 'custom.fasta').read_text(), '>x\nACGT\n')
        self.assertEqual(rows[0]['sample'], 'S1')

    def test_duplicate_and_changed_inputs_rejected(self):
        nested = self.inputs / 'nested'; nested.mkdir()
        (nested / 'a.fasta').write_text('>x\nACGT\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            discover(self.inputs)
        (nested / 'a.fasta').unlink()
        stage(discover(self.inputs), self.output)
        (self.inputs / 'a.fna').write_text('>x\nAAAA\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            stage(discover(self.inputs), self.output)

    def test_plan_no_tools_or_database_download(self):
        result = main(['run', '-i', str(self.inputs), '-o', str(self.output),
                       '--plan', '--threads', '8', '--max_cpus', '2', '--max_memory', '256 GB'])
        self.assertEqual(result, 0)
        receipt = json.loads(next(self.output.glob('logs/*/run.json')).read_text())
        self.assertEqual(receipt['params']['checkm_cpus'], 2)
        self.assertEqual(receipt['status'], 'planned')
        self.assertIsNotNone(receipt['database_command'])

    def test_explorer_exact_rows_and_search(self):
        self.output.mkdir()
        (self.output / 'inputs.resolved.json').write_text(json.dumps(discover(self.inputs)))
        with (self.output / 'quality.tsv').open('w') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['genome', 'note']); writer.writerow(['a', '<script>'])
            writer.writerow(['b', '100% recovered'])
        hidden = self.output / '.work'; hidden.mkdir()
        (hidden / 'private.tsv').write_text('x\ny\n')
        build(self.output)
        with sqlite3.connect(self.output / 'reports/results.sqlite') as db:
            self.assertEqual(db.execute('SELECT count(*) FROM catalog').fetchone()[0], 1)
            cols, total, rows = query(db, 't0', '%')
            self.assertEqual(total, 1)
            self.assertEqual(rows.fetchone()[0], 'b')
            _, count, rows = query(db, 't0', genome='a')
            self.assertEqual(count, 1)
            self.assertEqual(rows.fetchone()[0], 'a')
            with self.assertRaises(ValueError):
                query(db, 't0; DROP TABLE catalog')

    def test_empty_rna_predictions(self):
        root = Path(__file__).resolve().parents[1]
        (self.root / 'a.rRNA.fasta').touch()
        (self.root / 'a.blastout').touch()
        (self.root / 'a.rRNA.gff').write_text('##gff-version 3\n')
        for script in ('compare_barrnap.py', 'trnascan_parse.py'):
            result = subprocess.run([sys.executable, str(root / script), str(self.root)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        with (self.root / 'barrnap_blast_pass.tsv').open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        self.assertEqual(rows[0]['Bin Id'], 'a')

    def test_database_atomic_install_and_checksum(self):
        source = self.root / 'reference'; source.mkdir()
        for name in ('hmms', 'genome_tree'):
            (source / name).mkdir()
        (source / 'hmms/checkm.hmm').write_text('fixture\n')
        (source / 'hmms/phylo.hmm').write_text('fixture\n')
        (source / 'taxon_marker_sets.tsv').write_text('fixture\n')
        archive = self.root / 'reference.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(source, arcname='reference')
        checksum = hashlib.md5(archive.read_bytes()).hexdigest()
        target = self.root / 'installed'
        with patch.dict(database_install.SPECS, checkm=(archive.as_uri(), 'bad')), patch('database_install.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                database_install.install('checkm', target)
            self.assertFalse(target.exists())
        with patch.dict(database_install.SPECS, checkm=(archive.as_uri(), checksum)):
            database_install.install('checkm', target)
        self.assertTrue(database_install.valid('checkm', target))
        with patch('database_install.urllib.request.urlopen', side_effect=AssertionError('Unexpected download')):
            database_install.install('checkm', target)

    def test_scoped_selection_flags(self):
        from genome_qc.cli import parser
        commands = parser()._subparsers._group_actions[0].choices
        for name in ('build_db', 'preprocess', 'quality', 'taxonomy', 'rrna', 'trna', 'chimera'):
            help_text = commands[name].format_help()
            self.assertNotIn('--completeness', help_text)
            self.assertNotIn('--ani_threshold', help_text)
        self.assertNotIn('--filter_size', commands['quality'].format_help())
        self.assertNotIn('--gtdb_memory', commands['quality'].format_help())
        self.assertIn('--checkm_memory', commands['quality'].format_help())
        self.assertIn('unrestricted', commands['run'].format_help())
        args=parser().parse_args(['run','-i','input','-o','output'])
        self.assertIsNone(args.completeness)
        self.assertIsNone(args.contamination)
        self.assertIsNone(args.require_rrna_consistency)


if __name__ == '__main__':
    unittest.main()
