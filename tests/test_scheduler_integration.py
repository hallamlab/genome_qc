"""Local scheduler regression: many small jobs followed by a full-budget job.

Opt in with GQC_NF_INTEGRATION=1. Commands are synthetic and allocate no large
memory buffers. Nextflow 26.04.6 can finish LIGHT but strand HEAVY indefinitely
with stale CPU credits (reproduced: 31 available CPUs out of 32, none running).
"""
import os
import csv
from unittest.mock import patch
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('GQC_NF_INTEGRATION') == '1',
                     'Opt-in Nextflow integration test')
class SchedulerIntegration(unittest.TestCase):
    def test_two_checkm_reservations_overlap(self):
        from genome_qc.cli import parser
        from genome_qc.execution import configuration
        args = parser().parse_args(['run', '-i', '/inputs', '-o', '/output', '--threads', '8'])
        with patch('genome_qc.execution.capacity', return_value=(32, '108 GB')):
            config, _ = configuration(args, '/unused')
        with tempfile.TemporaryDirectory(prefix='gqc-memory-') as tmp:
            root = Path(tmp)
            (root / 'nextflow.config').write_text(config + "\ntrace.raw = true\ntrace.fields = 'name,start,complete,cpus,memory'\n")
            (root / 'main.nf').write_text("""
process CHECKM_SINGLE {
    cpus 8
    input:
    val x
    output:
    val x
    script:
    'sleep 2'
}
process GTDBTK {
    cpus 8
    input:
    val completed
    script:
    'true'
}
workflow {
    CHECKM_SINGLE(Channel.of(1, 2))
    GTDBTK(CHECKM_SINGLE.out.collect())
}
""")
            result = subprocess.run(['nextflow', 'run', 'main.nf', '-ansi-log', 'false',
                                     '-with-trace', 'trace.tsv'], cwd=root, capture_output=True,
                                    text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (root / 'trace.tsv').open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            checkm = [r for r in rows if r['name'].startswith('CHECKM_SINGLE')]
            self.assertEqual(len(checkm), 2)
            self.assertLess(max(int(r['start']) for r in checkm),
                            min(int(r['complete']) for r in checkm))
            self.assertTrue(all(int(r['memory']) == 48 * 1024**3 for r in checkm))
            gtdb = next(r for r in rows if r['name'].startswith('GTDBTK'))
            self.assertGreater(int(gtdb['memory']), 48 * 1024**3)

    def test_full_budget_task_after_many_completions(self):
        with tempfile.TemporaryDirectory(prefix='gqc-scheduler-') as tmp:
            root = Path(tmp)
            (root / 'nextflow.config').write_text("""
manifest.nextflowVersion = '!>=26.10.3'
executor.cpus=32
executor.memory='108 GB'
executor.queueSize=32
process.executor='local'
""")
            (root / 'main.nf').write_text("""
process LIGHT {
    cpus 1
    memory '4 GB'
    input:
    val x
    output:
    val x
    script:
    'sleep 0.1'
}
process HEAVY {
    cpus 32
    memory '98 GB'
    input:
    val completed
    output:
    stdout
    script:
    'echo FULL_BUDGET_TASK_COMPLETED'
}
workflow {
    LIGHT(Channel.fromList((1..4200).toList()))
    HEAVY(LIGHT.out.collect()).view()
}
""")
            command = ['nextflow', 'run', 'main.nf', '-ansi-log', 'false']
            process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       start_new_session=True)
            try:
                output, _ = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                output, _ = process.communicate(timeout=30)
                self.fail('Local scheduler stalled:\n' + output[-8000:])
            self.assertEqual(process.returncode, 0, output[-8000:])
            self.assertIn('FULL_BUDGET_TASK_COMPLETED', output)


if __name__ == '__main__':
    unittest.main()
