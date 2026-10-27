import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from genome_qc.hpc import save_packages, controller_limits


class HpcSafeguardsTests(unittest.TestCase):
    def test_broken_metadata_does_not_mask_primary_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / 'env/conda-meta'; meta.mkdir(parents=True)
            (meta / 'broken.json').write_text('{')
            with self.assertRaisesRegex(RuntimeError, 'original workflow failure'):
                try:
                    raise RuntimeError('original workflow failure')
                finally:
                    save_packages(root, root / 'packages.json')

    def test_valid_metadata_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / 'env/conda-meta'; meta.mkdir(parents=True)
            (meta / 'tool.json').write_text(json.dumps({'name': 'tool', 'version': '1'}))
            save_packages(root, root / 'packages.json')
            self.assertEqual(json.loads((root / 'packages.json').read_text())['env'][0]['version'], '1')

    def test_parent_cgroup_limit_is_considered(self):
        files = {'/proc/self/cgroup': '0::/user.slice/job\n',
                 '/sys/fs/cgroup/user.slice/pids.max': '512',
                 '/sys/fs/cgroup/user.slice/job/pids.max': 'max',
                 '/sys/fs/cgroup/memory.max': '17179869184'}
        def read(path, *args, **kwargs):
            if str(path) not in files: raise FileNotFoundError(str(path))
            return files[str(path)]
        with patch.object(Path, 'read_text', read):
            result = controller_limits()
        self.assertEqual(result['pids.max'], 512)
        self.assertEqual(result['memory.max'], 17179869184)
