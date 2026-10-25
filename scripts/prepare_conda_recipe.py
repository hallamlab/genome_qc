#!/usr/bin/env python3
"""Create a checksummed, allow-listed source archive and buildable Conda recipe."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]

def source_files(root):
    tree = ast.parse((root / 'setup.py').read_text())
    assets = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'scripts' for t in n.targets))
    return sorted(set([root / n for n in ['pyproject.toml', 'setup.py', 'LICENSE', *assets]]
                      + list((root / 'genome_qc').glob('*.py'))
                      + list((root / 'envs').glob('*.yaml'))))

def prepare(output):
    import yaml
    version = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']['version']
    package_tree = ast.parse((ROOT / 'genome_qc/__init__.py').read_text())
    package_version = next(ast.literal_eval(n.value) for n in package_tree.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    if version != package_version:
        raise ValueError('Python metadata and runtime versions disagree')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'genome-qc-{version}.tar.gz'
    if archive.exists():
        raise ValueError(f'Refusing to replace existing source archive: {archive}')
    files = source_files(ROOT)
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Expected a regular source file: {path}')
            bundle.add(path, arcname=str(Path(f'genome-qc-{version}') / path.relative_to(ROOT)))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    recipe = output / 'recipe'; recipe.mkdir()
    deps = yaml.safe_load((ROOT / 'envs/controller.yaml').read_text())['dependencies']
    # pip is used by the build, never required as a user installation step.
    deps = [d for d in deps if d != 'pip']
    if not all(isinstance(d,str) for d in deps):
        raise ValueError('Controller dependencies must be Conda specifications')
    template = (ROOT / 'conda_recipe/meta_template.yaml').read_text()
    for key,value in {'VERSION':version, 'SOURCE':archive.as_uri(), 'SHA256':digest,
                      'DEPENDENCIES':'\n'.join('    - '+d for d in deps)}.items():
        template = template.replace('@'+key+'@',value)
    (recipe / 'meta.yaml').write_text(template)
    shutil.copyfile(ROOT / 'conda_recipe/run_test.py', recipe / 'run_test.py')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    (output / 'source-provenance.json').write_text(json.dumps(dict(version=version,revision=revision,
        dirty=dirty,source_sha256=digest,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
    print(recipe)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    prepare(p.parse_args().output)
