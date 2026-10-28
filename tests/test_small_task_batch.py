import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import small_task_batch as batch


FIXTURE = '''#!PYTHON
import pathlib, sys
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
def opt(flag): return args[args.index(flag)+1]
if name == 'seqkit':
    src = pathlib.Path(args[-1])
    if args[0] == 'seq':
        if src.stem == 'badinput': sys.exit(2)
        pathlib.Path(opt('-o')).write_text(src.read_text())
    else:
        if src.name.startswith('badstats.'): sys.exit(2)
        print('file\\tnum_seqs\\tsum_len\\tmin_len\\tavg_len\\tmax_len\\tN50')
        print(str(src)+'\\t1\\t4\\t4\\t4\\t4\\t4')
elif name == 'barrnap':
    if pathlib.Path(args[-1]).stem == 'bad': sys.exit(2)
    pathlib.Path(opt('--outseq')).write_text('>rrna\\nACGT\\n')
    print('##gff-version 3')
elif name == 'blastn': pathlib.Path(opt('-out')).write_text('hit\\n')
elif name == 'makeblastdb': pathlib.Path(opt('-out')+'.nhr').touch()
elif name == 'tRNAscan-SE':
    pathlib.Path(opt('-j')).write_text('##gff-version 3\\n')
    pathlib.Path(opt('-o')).write_text('extra')
'''


class SmallBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.previous = Path.cwd(); os.chdir(self.root)
        bins = self.root / 'bin'; bins.mkdir()
        for name in ('seqkit', 'barrnap', 'blastn', 'makeblastdb', 'tRNAscan-SE'):
            p = bins / name; p.write_text(FIXTURE.replace('PYTHON', sys.executable, 1)); p.chmod(0o755)
        self.env = patch.dict(os.environ, {'PATH': str(bins) + os.pathsep + os.environ['PATH']}); self.env.start()

    def tearDown(self):
        self.env.stop(); os.chdir(self.previous); self.tmp.cleanup()

    def fasta(self, name):
        p = self.root / (name + '.fasta'); p.write_text('>x\nACGT\n'); return str(p)

    def test_preprocessing_keeps_other_genomes_and_failed_stats_fasta(self):
        rows = [[name, 'full', self.fasta(name), '0'] for name in ('good', 'badinput', 'badstats')]
        batch.run('preprocess', rows, cpus=2)
        out = self.root / 'results'
        self.assertTrue((out / 'good.full.fasta').exists())
        self.assertFalse((out / 'badinput.full.fasta').exists())
        self.assertTrue((out / 'badstats.full.fasta').exists())
        self.assertFalse((out / 'badstats.full.seqkit.tsv').exists())
        record = json.loads((out / 'badstats.full.sequence_stats.assessment.json').read_text())
        self.assertEqual(record['status'], 'failed')
        self.assertTrue((out / 'badstats.full.sequence_stats.assessment.log').exists())

    def test_rrna_failure_is_per_genome_and_blast_indexes_are_temporary(self):
        batch.run('rrna', [[name, 'Bacteria', self.fasta(name), '0', '0'] for name in ('good', 'bad')], cpus=2)
        out = self.root / 'results'
        self.assertEqual(json.loads((out / 'good.rrna_check.assessment.json').read_text())['status'], 'completed')
        self.assertEqual(json.loads((out / 'bad.rrna_check.assessment.json').read_text())['status'], 'blocked')
        self.assertEqual((out / 'good.blastout').read_text(), 'hit\n')
        self.assertFalse(list(out.rglob('*.nhr')))

    def test_compact_trna_retains_report_inputs_and_diagnostics(self):
        batch.run('trna', [['good', 'Bacteria', self.fasta('good'), '0']], cpus=2, compact=True)
        out = self.root / 'results'
        self.assertTrue((out / 'good.gff').exists())
        self.assertTrue((out / 'good.trna.assessment.json').exists())
        self.assertTrue((out / 'good.trna.assessment.log').exists())
        self.assertFalse((out / 'good.output.txt').exists())
