"""Installed-package checks, executed outside the checkout by conda-build."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from genome_qc.runtime import runtime_root
from genome_qc.report import build

root = runtime_root()
assert root == Path(sys.prefix) / 'share/genome_qc', root
for name in ('main.nf', 'nextflow.config', 'database_install.py', 'assessment_task.py',
             'small_task_batch.py', 'envs/rrna_batch.yaml', 'assessment_summary.py', 'merge_gunc_batches.py', 'genome_quality_atlas.py',
             'envs/checkm.yaml', 'envs/gtdbtk.yaml', 'envs/gunc.yaml', 'envs/reference_setup.yaml'):
    assert (root / name).is_file(), name
with tempfile.TemporaryDirectory() as temp:
    temp = Path(temp)
    inputs = temp / 'inputs'; inputs.mkdir()
    (inputs / 'example.fna').write_text('>contig\nACGTACGTACGT\n')
    output = temp / 'output'
    subprocess.run(['genome_qc', 'run', '-i', str(inputs), '-o', str(output),
                    '--plan', '--threads', '1', '--max_cpus', '1', '--max_memory', '4 GB'],
                   cwd=temp, check=True)
    receipt = json.loads(next(output.glob('logs/*/run.json')).read_text())
    assert str(root / 'main.nf') in receipt['command']
    (output / 'Master_genome_QC.tsv').write_text('Genome_Id\tCompleteness\nexample\t95\n')
    build(output)
    assert (output / 'reports/EDA_portal.html').is_file()
    assert (output / 'reports/results.sqlite').is_file()
print('Installed workflow assets, CLI planning and explorer verified.')
