"""Quality gates annotate results rather than discarding assessment rows."""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class FinalSelectionTests(unittest.TestCase):
    def test_failed_rrna_and_low_quality_retain_all_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def table(name, columns, rows):
                path=root/name; path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('w') as out:
                    writer=csv.writer(out, delimiter='\t'); writer.writerow(columns); writer.writerows(rows)
            table('qscore/qscore_all.tsv', ['Genome_Id','Bin Id','Completeness','Contamination','Strain heterogeneity','num_seqs','sum_len','min_len','avg_len','max_len','N50','qscore'],
                  [['a','a.full',90,1,0,1,120000,120000,120000,120000,120000,90],['b','b.full',20,25,0,1,80000,80000,80000,80000,80000,-100]])
            table('barrnap/barrnap_blast_pass.tsv',['Bin Id','pass_BARRNAP'],[['a.full',False],['b.full',1.0]])
            table('gtdbtk/classify/gtdbtk.bac120.summary.tsv',['user_genome','classification'],[['a.full','d__Bacteria;p__P'],['b.full','d__Bacteria;p__P']])
            (root/'dedupe').mkdir()
            table('barrnap/barrnap_subunit_counts.tsv',['genome_id','name','subunit_count'],[])
            table('trnascan/trnascan_trna_counts.tsv',['genome_id','trna_total','trna_unique'],[['a.full',20,19],['b.full',10,9]])
            table('gunc/GUNC.test.maxCSS_level.tsv',['genome','clade_separation_score','reference_representation_score','taxonomic_level'],[['a.full',0.1,0.9,'species'],['b.full',0.5,0.8,'genus']])
            env={**os.environ,'MPLBACKEND':'Agg','MPLCONFIGDIR':str(root/'mpl'),'OPENBLAS_NUM_THREADS':'1'}
            for script,args in [('dedupe_genomes.py',[]),('merge_master.py',['100000','15000000','50','10','0','true'])]:
                result=subprocess.run([sys.executable,str(ROOT/script),str(root),*args],env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            master=pd.read_csv(root/'Master_genome_QC.tsv',sep='\t').set_index('Genome_Id')
            self.assertEqual(set(master.index),{'a','b'})
            self.assertEqual(master.loc['a','selection_reasons'],'rRNA_check')
            self.assertEqual(master.loc['b','selection_reasons'],'genome_size;completeness;contamination;qscore')
            self.assertTrue(master['gunc_clade_separation_score'].notna().all())
            self.assertEqual(master.loc['b','trna_total'],10)
            self.assertTrue(pd.read_csv(root/'Selected_genome_QC.tsv',sep='\t').empty)

            result=subprocess.run([sys.executable,str(ROOT/'merge_master.py'),str(root)],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            master=pd.read_csv(root/'Master_genome_QC.tsv',sep='\t')
            self.assertTrue(master['selected'].all())
            self.assertEqual(len(pd.read_csv(root/'Selected_genome_QC.tsv',sep='\t')),2)
