"""Explicit workflow assets: never package local analyses or manuscript files."""
from pathlib import Path
from setuptools import setup

root = Path(__file__).parent
scripts = ['main.nf', 'nextflow.config', 'database_install.py', 'small_task_batch.py', 'compact_task.py', 'assessment_task.py', 'assessment_summary.py', 'merge_gunc_batches.py', 'calc_qscore.py',
           'compare_barrnap.py', 'dedupe_genomes.py', 'trnascan_parse.py',
           'merge_master.py', 'genome_quality_atlas.py', 'genome_quality_atlas_wrapper.py',
           'preserve_marker_alignments.py', 'cohort_phylogeny.py', 'best_set_phylogeny.py', 'render_phylogeny_ete.py', 'genome_set_stats.py',
           'gunc_method_stats.py', 'statistics_inventory.py']
setup(data_files=[('share/genome_qc', scripts),
                  ('share/genome_qc/envs', [str(p.relative_to(root)) for p in sorted((root / 'envs').glob('*.yaml'))])])
