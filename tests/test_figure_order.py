import sqlite3
import unittest
from genome_qc.report import artifacts


class FigureOrderTests(unittest.TestCase):
    def test_existing_index_and_custom_prefix_follow_reading_order(self):
        names = [
            'genome_quality_atlas/my_study_atlas.png',
            'genome_quality_atlas/my_study_integrity_recoverability.png',
            'genome_quality_atlas/my_study_qscore_distribution.png',
            'genome_quality_atlas/my_study_hallmark_contributions.png',
            'genome_quality_atlas/my_study_recovery_patterns.png',
            'genome_quality_atlas/my_study_taxonomy_qscore.png',
            'genome_quality_atlas/my_study_threshold_landscape.png',
            'genome_quality_atlas/ani/ani_summary.png',
            'genome_quality_atlas/phylogeny/bac120_annotated_tree.png',
            'genome_quality_atlas/another_comparison.png',
            'other/report.html',
        ]
        db = sqlite3.connect(':memory:')
        try:
            db.execute('CREATE TABLE artifacts (id INTEGER PRIMARY KEY, path TEXT)')
            db.executemany('INSERT INTO artifacts(path) VALUES (?)', [(p,) for p in reversed(names)])
            rows = artifacts(db)
            self.assertEqual([r['path'] for r in rows], names)
            for r in rows:
                self.assertEqual(db.execute('SELECT path FROM artifacts WHERE id=?', (r['id'],)).fetchone()[0], r['path'])
        finally:
            db.close()
