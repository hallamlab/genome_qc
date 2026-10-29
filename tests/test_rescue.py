import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from genome_qc.rescue import prepare, failure_report, normalize_settings


class RescueTests(unittest.TestCase):
    def test_legacy_unpublished_products_and_taxonomy_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); out = root / 'output'; out.mkdir()
            work = out / '.genome_qc/work'; products = work / 'aa/123456789/results'; products.mkdir(parents=True)
            (out / 'Assessment_status.tsv').write_text('Genome_Id\tstage\tstatus\n')
            def record(stage, suffixes, status='completed'):
                for suffix in suffixes: (products / ('g.full' + suffix)).write_text('data')
                p = products / ('g.full.' + stage + '.assessment.json')
                p.write_text(json.dumps(dict(stage=stage, genomes=['g' if stage == 'preprocess' else 'g.full'], status=status, log=p.with_suffix('.log').name)))
                p.with_suffix('.log').write_text('diagnostic')
            record('preprocess', ['.fasta']); record('sequence_stats', ['.seqkit.tsv'])
            record('rrna', ['.rRNA.gff', '.rRNA.fasta'])
            record('rrna_check', ['.blastout'])
            record('gtdbtk', [], 'failed')
            plan = json.loads(prepare(out, work, root / 'recovery').read_text())
            self.assertEqual(plan['preprocess']['ids'], ['g.full'])
            self.assertFalse(plan['rrna']['ids'])
            self.assertTrue((root / 'recovery/previous-diagnostics/g.full.gtdbtk.assessment.log').is_file())
            failure_report(out, work, out)
            self.assertTrue((out / 'barrnap/g.full.rrna.assessment.log').is_file())

    def test_execution_changes_do_not_change_science(self):
        self.assertEqual(normalize_settings({'checkm_batch_size': 20, 'small_batch_size': 10, 'filter_size': '500'}),
                         normalize_settings({'checkm_batch_size': 100, 'small_batch_size': 5, 'filter_size': '500'}))
        self.assertNotEqual(normalize_settings({'filter_size': '500'}), normalize_settings({'filter_size': '1000'}))

    def test_scheduler_oom_evidence_overrides_generic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp); work = out / 'work'; work.mkdir()
            (out / 'Assessment_status.tsv').write_text('Genome_Id\tvariant\tstage\tstatus\tlog\n' + 'g\tg.full\trrna\tfailed\tbarrnap/g.log\n')
            with patch('genome_qc.rescue.scheduler_evidence', return_value={'g.log': {'job_id':'123','slurm_state':'OUT_OF_MEMORY'}}):
                failure_report(out, work, out)
            self.assertIn('out_of_memory', (out / 'Assessment_failures.tsv').read_text())
