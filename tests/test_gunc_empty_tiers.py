import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import genome_quality_atlas as atlas


class GuncEmptyTierTests(unittest.TestCase):
    def test_selected_sets_with_absent_quality_tiers(self):
        rows = []
        for tier, category in [('medium', 'MAGs'), ('high', 'SAGs')]:
            rows.append(dict(Genome_Id=tier, qscore=80, category=category, mimag_tier=tier,
                gunc_assessment='likely_not_chimeric',
                gunc_strict_assessment='likely_not_chimeric', gunc_strict_chimera=False,
                gunc_clade_separation_score=0.1, gunc_reference_representation_score=0.9,
                gunc_contamination_portion=0.0))
        for tiers in [('medium', 'high'), ('high',)]:
            with self.subTest(tiers=tiers):
                frame = pd.DataFrame(rows)
                frame = frame.loc[frame.mimag_tier.isin(tiers)]
                result = atlas.summarize_compare_gunc(frame, 'category')
                empty = result.loc[~result.quality_scope.isin(['all', *tiers])]
                self.assertFalse(empty.empty)
                self.assertTrue(empty.filter(regex='^n_').eq(0).all().all())
                self.assertTrue(empty.filter(regex='^fraction_').eq(0).all().all())
                self.assertTrue(empty.filter(regex='^(mean|median)_').isna().all().all())
                overall = result.loc[result.quality_scope.eq('all')]
                self.assertEqual(overall.n_genomes.sum(), len(frame))
                self.assertTrue(overall.fraction_gunc_likely_not_chimeric.eq(1).all())


if __name__ == '__main__':
    unittest.main()
