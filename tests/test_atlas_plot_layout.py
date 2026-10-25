"""Regressions for clipped data and annotations in atlas exports."""
import unittest
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import genome_quality_atlas as atlas


class AtlasLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        atlas.ensure_plotting()

    def tearDown(self):
        atlas.plt.close('all')

    def test_qscore_is_not_a_percentage(self):
        fig, ax = atlas.plt.subplots()
        ax.scatter([0, 1], [95, 108])
        ax.set_ylabel('Qscore')
        atlas.standardize_figure_scales(fig)
        self.assertGreater(ax.get_ylim()[1], 108)

    def test_component_exact_positions_counts_and_mean_shading(self):
        fig, ax = atlas.plt.subplots()
        frame = pd.DataFrame({'integrity_score':[1., 1., .999],
                              'recoverability_score':[1., 1., 1.],
                              'qscore':[90., 110., 80.]})
        scatter = atlas.plot_component_scatter(ax, frame)
        self.assertEqual(len(scatter.get_offsets()), 2)
        self.assertEqual(list(scatter.get_array()), [80., 100.])
        self.assertGreater(scatter.get_sizes()[1], scatter.get_sizes()[0])
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ['1', '2'])

    def test_fraction_markers_keep_padding(self):
        fig, ax = atlas.plt.subplots()
        ax.scatter([0, 1], [0, 1])
        ax.set_xlabel('Integrity (I)')
        ax.set_ylabel('Recoverability (R)')
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.02)
        atlas.standardize_figure_scales(fig)
        self.assertLess(ax.get_xlim()[0], 0)
        self.assertGreater(ax.get_ylim()[1], 1)

    def test_bar_annotations_do_not_touch_title(self):
        fig, ax = atlas.plt.subplots(figsize=(6.5, 5.5))
        frame = pd.DataFrame({key:[1., 1.] for key in
            ['rrna_16S_score', 'rrna_23S_score', 'rrna_5S_score', 'trna_ge_18']})
        atlas.plot_feature_recovery(ax, frame)
        atlas.standardize_figure_scales(fig)
        atlas.apply_figure_typography(fig)
        atlas.apply_tight_layout(fig)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        title=ax.title.get_window_extent(renderer)
        for text in ax.texts:
            self.assertFalse(text.get_window_extent(renderer).overlaps(title))
            self.assertLess(text.get_window_extent(renderer).y1, ax.bbox.y1)

    def test_contamination_outliers_remain_visible(self):
        fig, ax = atlas.plt.subplots()
        ax.scatter([100], [45])
        atlas.add_quality_thresholds(ax)
        atlas.standardize_figure_scales(fig)
        self.assertGreater(ax.get_ylim()[1], 45)
        self.assertGreater(ax.get_xlim()[1], 100)

    def test_quality_tiers_share_color_scale(self):
        fig, ax = atlas.plt.subplots()
        frame = pd.DataFrame({'mimag_tier':['low', 'medium', 'high'],
                              'Completeness':[40, 80, 100], 'Contamination':[2, 3, 1],
                              'qscore':[30, 75, 108]})
        atlas.plot_completeness_contamination(ax, frame)
        self.assertEqual([c.get_clim() for c in ax.collections], [(30, 108)] * 3)
