import gzip
from pathlib import Path
import tempfile
import unittest
from cohort_phylogeny import ani_pairs, clusters, collect_alignments, stage_msa


class CohortTests(unittest.TestCase):
    def test_containment_label_and_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw.tsv'
            raw.write_text('a.fasta\tb.fasta\t99\t95\t100\nb.fasta\ta.fasta\t98\t95\t400\n')
            ids={'a':'A','b':'B'}; lengths={'a':100,'b':400}
            pair=ani_pairs(raw,ids,98,.51,lengths)[0]
            self.assertTrue(pair['passes']);self.assertEqual(pair['match_type'],'containment')
            self.assertFalse(ani_pairs(raw,ids,98,.51,lengths,False)[0]['passes'])
            self.assertFalse(ani_pairs(raw,ids,99,.51,lengths)[0]['passes'])
            raw.write_text('a.fasta\tb.fasta\t99\t95\t100\n')
            self.assertFalse(ani_pairs(raw,ids,98,.51,lengths)[0]['passes'])
            raw.write_text('a.fasta\tb.fasta\t98\t51\t100\nb.fasta\ta.fasta\t98\t204\t400\n')
            pair=ani_pairs(raw,ids,98,.51,lengths)[0]
            self.assertEqual(pair['match_type'],'standard')
            self.assertEqual(len({r['ani_cluster'] for r in clusters(['A','B','C'],[pair])}),2)

    def test_reuse_only_selected_variants_and_validate_batches(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'native';source.mkdir()
            (source/'gtdbtk.bac120.user_msa.fasta').write_text('>a.full\nAA-X\n>a.1000\nWWWW\n>b.full\n----\n')
            output=root/'msa';stage_msa(source,output)
            self.assertEqual(collect_alignments([output],'bac120',{'a.full':'A','b.full':'B'}),{'A':'AA-X'})
            other=root/'other';other.mkdir()
            with gzip.open(other/'bac120.fasta.gz','wt') as f:f.write('>a.full\nXXXXA\n')
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                collect_alignments([output,other],'bac120',{'a.full':'A'})
