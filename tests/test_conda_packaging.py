import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_recipe', ROOT/'scripts/prepare_conda_recipe.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)

class PackageSourceTests(unittest.TestCase):
    def test_source_allowlist_excludes_study_material(self):
        files = {str(p.relative_to(ROOT)) for p in recipe.source_files(ROOT)}
        self.assertIn('main.nf',files)
        self.assertIn('cohort_phylogeny.py',files)
        self.assertIn('envs/cohort_phylogeny.yaml',files)
        self.assertIn('genome_qc/explorer.py',files)
        self.assertIn('assessment_summary.py',files)
        self.assertIn('envs/gtdbtk.yaml',files)
        self.assertFalse(any(p.startswith(('docs/','preview/','tests/','scripts/')) for p in files))
        self.assertFalse(any(Path(p).suffix in ('.docx','.pdf','.tsv','.sif') for p in files))
        self.assertEqual(len(files),len(set(files)))
