import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from assessment_task import run
from assessment_summary import attach
from genome_quality_atlas import compute_quality_index


class AssessmentFailures(unittest.TestCase):
    def test_failed_command_does_not_supply_partial_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            script=root/'run.sh'; target=root/'checkm.tsv'; receipt=root/'checkm.assessment.json'
            script.write_text(f"echo partial > {target}\necho 'tool failure' >&2\nexit 7\n")
            result=run('checkm',target,receipt,str(script),['a.full.fasta'])
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['exit_code'],7)
            self.assertEqual(len(target.read_text().splitlines()),1)
            self.assertIn('tool failure',receipt.with_suffix('.log').read_text())
            self.assertEqual(target.with_name(target.name+'.failed').read_text(),'partial\n')

    def test_blocked_comparison_is_not_no_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=run('rrna_check',root/'a.blastout',root/'receipt.json','does-not-exist',['a.full.fasta'],blocked=True)
            self.assertEqual(result['status'],'blocked')
            self.assertIsNone(result['exit_code'])

    def test_missing_assessments_stay_unknown_in_master_and_atlas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'assessments').mkdir()
            for stage in ('checkm','rrna','rrna_check','trna'):
                (root/'assessments'/f'{stage}.assessment.json').write_text(json.dumps(dict(stage=stage,status='failed',genomes=['a.full'],exit_code=7,reason='test')))
            frame=pd.DataFrame([dict(Genome_Id='a',**{'Bin Id':'a.full'}, Domain='',Completeness=90,Contamination=1,qscore=90,
                                     sum_len=1000,**{'16S_rRNA':0,'23S_rRNA':0,'5S_rRNA':0},trna_total=0,trna_unique=0,contains_16S=False,pass_BARRNAP=True)])
            annotated=attach(frame,root)
            self.assertTrue(pd.isna(annotated.loc[0,'16S_rRNA']))
            self.assertTrue(pd.isna(annotated.loc[0,'trna_total']))
            self.assertFalse(annotated.loc[0,'assessment_complete'])
            atlas=compute_quality_index(annotated)
            self.assertTrue(pd.isna(atlas.loc[0,'recoverability_score']))
            self.assertTrue(pd.isna(atlas.loc[0,'mimag_tier']))
            self.assertEqual(atlas.loc[0,'recovery_pattern'],'unavailable')

            from genome_quality_atlas import save_atlas, save_individual_plots
            save_atlas(atlas, str(root/'missing'), group_column='Domain')
            save_individual_plots(atlas, str(root/'missing'), group_column='Domain')
            self.assertTrue((root/'missing_atlas.pdf').is_file())

    def test_compaction_defers_cleanup_for_failed_assessment(self):
        from genome_qc.compact import finish
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'reports').mkdir();(root/'reports/results.sqlite').touch()
            (root/'Master_genome_QC.tsv').write_text('Genome_Id\na\n')
            (root/'Assessment_status.tsv').write_text('Genome_Id\tstatus\na\tfailed\n')
            work=root/'work';work.mkdir();(work/'checkpoint').touch()
            with self.assertRaisesRegex(ValueError,'unresolved assessment failures'):
                finish(root,'test',[work],root/'cache',False,[],root/'db')
            self.assertTrue((work/'checkpoint').exists())
            self.assertFalse((root/'compact-results.json').exists())
