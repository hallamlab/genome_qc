"""Keep helper runtimes explicit; infrastructure errors must not be ignored."""
from pathlib import Path
import os
import re
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get('GQC_TEST_RUNTIME', Path(__file__).resolve().parents[1]))

class RuntimeContracts(unittest.TestCase):
    def test_every_python_process_declares_python(self):
        source = (ROOT / 'main.nf').read_text()
        for name, block in re.findall(r'process (\w+) \{(.*?)(?=\nprocess |\nworkflow |\Z)', source, re.S):
            if not re.search(r'\bpython(?:3)?\s', block):
                continue
            match = re.search(r'conda "\$\{projectDir\}/(envs/[^" ]+)"', block)
            self.assertIsNotNone(match, name)
            self.assertRegex((ROOT / match[1]).read_text(), r'(?m)^  - python(?:[=<> ].*)?$', name)

    def test_missing_runtime_is_fatal_before_assessment(self):
        source = (ROOT / 'main.nf').read_text()
        for name in ('CHECKM_SINGLE', 'GTDBTK', 'BARRNAP', 'BLAST_RRNA', 'TRNASCAN', 'GUNC'):
            block = source.split('process ' + name + ' {', 1)[1].split('\nprocess ', 1)[0]
            hook = re.search(r"'-c', shellQuote\('([^']+)'", block)[1].split('; exec /bin/bash')[0]
            with tempfile.TemporaryDirectory() as empty:
                result = subprocess.run(['/bin/bash', '-c', hook], env={'PATH': empty}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 127, name)
            self.assertIn('Missing required executable: python', result.stderr)
            self.assertIn("? 'retry' : 'terminate'", block)

    @unittest.skipUnless(os.environ.get('GQC_NF_INTEGRATION') == '1', 'Opt-in Nextflow integration test')
    def test_launch_guard_preserves_successful_task_cache(self):
        with tempfile.TemporaryDirectory(prefix='gqc-runtime-cache-') as tmp:
            root = Path(tmp)
            workflow = root / 'main.nf'
            base = '\n'.join(("def shellQuote(value) { return \"'\" + value + \"'\" }", '''process ASSESSMENT {
%s
output:
path 'receipt.txt'
script:
"""
echo completed > receipt.txt
"""
}
workflow { ASSESSMENT() }
'''))
            workflow.write_text(base % "errorStrategy 'ignore'")
            cmd = ['nextflow', 'run', str(workflow), '-ansi-log', 'false']
            first = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            workflow.write_text(base % "shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('command -v python >/dev/null || exit 127; exec /bin/bash -euo pipefail \"$@\"'), '--'])\nerrorStrategy 'terminate'")
            second = subprocess.run(cmd + ['-resume'], cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn('Cached process > ASSESSMENT', second.stdout + second.stderr)
            # Force a fresh task to exercise the launch shell, not just hashing.
            third = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            workflow.write_text(workflow.read_text().replace('command -v python', 'command -v genome_qc_deliberately_missing_executable'))
            missing = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=90)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn('exit status (127)', missing.stdout + missing.stderr)
            self.assertNotIn('Error is ignored', missing.stdout + missing.stderr)
