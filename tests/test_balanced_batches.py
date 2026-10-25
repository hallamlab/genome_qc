import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from genome_qc.cli import parser
from genome_qc.execution import configuration
from merge_gunc_batches import merge


class BalancedBatches(unittest.TestCase):
    def test_local_slots_respect_memory_and_threads(self):
        args=parser().parse_args(['run','-i','input','-o','out','--threads','8'])
        with patch('genome_qc.execution.capacity',return_value=(32,'108 GB')):
            _,budget=configuration(args,'cache')
        self.assertEqual(budget['batch_slots'],dict(checkm=2,gtdb=1,gunc=1))
        args.threads=16
        with patch('genome_qc.execution.capacity',return_value=(32,'108 GB')):
            _,budget=configuration(args,'cache')
        self.assertEqual(budget['batch_slots']['checkm'],1)

    def test_slurm_slots_respect_job_limit(self):
        args=parser().parse_args(['run','-i','input','-o','out','--executor','slurm','--account','test','--max_tasks','20'])
        _,budget=configuration(args,'cache')
        self.assertEqual(budget['batch_slots'],dict(checkm=20,gtdb=20,gunc=20))

    def test_merge_retains_all_batches_and_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for batch,id in [('one','a'),('two','b')]:
                (root/batch).mkdir()
                (root/batch/'GUNC.test.maxCSS_level.tsv').write_text('genome\tclade_separation_score\n'+id+'\t0.1\n')
            merge(root/'merged',[root/'one',root/'two'])
            with (root/'merged/GUNC.test.maxCSS_level.tsv').open() as f:
                rows=list(csv.DictReader(f,delimiter='\t'))
            self.assertEqual([r['genome'] for r in rows],['a','b'])
            with self.assertRaisesRegex(ValueError,'Duplicate genome'):
                merge(root/'duplicate',[root/'one',root/'one'])
