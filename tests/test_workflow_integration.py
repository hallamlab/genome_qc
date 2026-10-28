"""Exercise real Nextflow and downstream analysis with explicit tool-contract fixtures.

Opt in with GQC_NF_INTEGRATION=1. This validates orchestration, not the biological
accuracy of the external tools. Requires Nextflow, Java, SeqKit and atlas Python dependencies.
"""
import csv
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

from genome_qc.inputs import discover, stage
from genome_qc.report import build


@unittest.skipUnless(os.environ.get('GQC_NF_INTEGRATION') == '1', 'Opt-in Nextflow integration test')
class WorkflowIntegration(unittest.TestCase):
    def test_full_chain_and_resume(self):
        self.exercise(False)

    def test_compact_scratch_and_completion(self):
        self.exercise(True)

    def test_assessment_failures_continue_and_resume(self):
        self.exercise(False, failures=('checkm', 'gtdbtk', 'barrnap', 'tRNAscan-SE'))

    def test_phylogeny_without_ani(self):
        self.exercise(True, skip_ani=True)

    def exercise(self, compact_mode, failures=(), skip_ani=False):
        root = Path(__file__).resolve().parents[1]
        runtime = Path(os.environ.get('GQC_TEST_RUNTIME', root))
        with tempfile.TemporaryDirectory(prefix='gqc-integration-') as tmp:
            work = Path(tmp)
            # Private runtime permits a helper-code invalidation test without
            # touching the repository or an installed/running workflow.
            private_runtime = work / 'runtime'; private_runtime.mkdir()
            for source in runtime.iterdir():
                if source.is_file() and (source.suffix in ('.py', '.nf') or source.name == 'nextflow.config'):
                    shutil.copy2(source, private_runtime / source.name)
            shutil.copytree(runtime / 'envs', private_runtime / 'envs')
            runtime = private_runtime
            bins = work / 'bin'; bins.mkdir()
            tool = bins / 'tool_fixture'
            shutil.copyfile(root / 'tests/tool_contract_fixture.py', tool)
            tool.chmod(0o755)
            for name in failures:
                (bins / ('fail-' + name)).touch()
            for name in ('checkm', 'gtdbtk', 'barrnap', 'tRNAscan-SE', 'gunc'):
                (bins / name).symlink_to(tool)
            inputs = work / 'inputs'; inputs.mkdir()
            for i in range(10):
                (inputs / f'g{i}.fasta').write_text('>contig\n' + 'ACGT' * (20000 if i == 3 else 30000) + '\n')
            checkm = work / 'db/checkm'; checkm.mkdir(parents=True)
            (checkm / 'fixture').touch()
            gtdb = work / 'db/gtdb/metadata'; gtdb.mkdir(parents=True)
            (gtdb / 'metadata.txt').write_text('VERSION_DATA=r232\n')
            (work / 'db/gunc.dmnd').write_text('fixture\n')
            output = work / 'output'
            params = dict(small_batch_size=3, small_batch_cpus=2, max_forks=2, compact_results=compact_mode, stage='run', filter_size='', working_dir=str(output),
                          fasta_dir=str(stage(discover(inputs), output)),
                          checkm_data_path=str(checkm), gtdbtk_data_path=str(gtdb.parent),
                          gunc_db=str(work / 'db/gunc.dmnd'), checkm_cpus=1, gtdbtk_cpus=32 if compact_mode else 1,
                          gunc_cpus=1, atlas_cpus=1, atlas_ani_threads=1, atlas_top_n_groups=12,
                          checkm_batch_size=3, gtdbtk_batch_size=3, gunc_batch_size=3, checkm_parallel_batches=2, gtdbtk_parallel_batches=2, gunc_parallel_batches=2, completeness=50, min_genome_size=100000, contamination=10)
            params['cohort_ani_enabled'] = not skip_ani
            (work / 'params.json').write_text(json.dumps(params))
            from genome_qc.cli import parser
            from genome_qc.execution import configuration
            options = ['run', '-i', str(inputs), '-o', str(output), '--threads', '1',
                       '--max_cpus', '32' if compact_mode else '4', '--memory', '2 GB', '--max_memory', '16 GB']
            if compact_mode:
                options.append('--compact_results')
            config, _ = configuration(parser().parse_args(options), work / 'conda')
            (work / 'test.config').write_text(config + "\nconda.enabled=false\n")
            env = {**os.environ, 'PATH': str(bins) + os.pathsep + os.environ['PATH'],
                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MPLCONFIGDIR': str(work / 'mpl')}
            command = ['nextflow', '-c', str(work / 'test.config'), 'run', str(runtime / 'main.nf'),
                       '-ansi-log', 'false',
                       '-params-file', str(work / 'params.json'), '-work-dir', str(work / 'work')]
            first = subprocess.run(command, cwd=work, env=env, text=True, capture_output=True, timeout=300)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr + (work / '.nextflow.log').read_text()[-12000:])
            annotated = output / 'genome_quality_atlas/genome_quality_annotated.tsv'
            with annotated.open() as handle:
                rows = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual({r['Genome_Id'] for r in rows}, {f'g{i}' for i in range(10)})
            if failures:
                self.assertTrue(all(r['checkm_status'] == 'failed' for r in rows))
                self.assertTrue(all(r['gtdbtk_status'] == 'failed' for r in rows))
                self.assertTrue(all(r['rrna_status'] == 'failed' for r in rows))
                self.assertTrue(all(r['trna_status'] == 'failed' for r in rows))
                self.assertTrue(all(r['gunc_status'] == 'completed' for r in rows))
                with (output / 'Master_genome_QC.tsv').open() as handle:
                    raw = list(csv.DictReader(handle, delimiter='\t'))
                self.assertTrue(all(r['Completeness'] == '' and r['trna_total'] == '' and r['16S_rRNA'] == '' for r in raw))
                self.assertTrue((output / 'Assessment_status.tsv').is_file())
                for name in failures:
                    (bins / ('fail-' + name)).unlink()
                prior_receipts = {str(p.relative_to(output)): json.loads(p.read_text()) for p in output.glob('checkm/individual/*.assessment.json')}
                self.assertTrue(prior_receipts)
                recovered = subprocess.run(command + ['-resume'], cwd=work, env=env, text=True, capture_output=True, timeout=300)
                self.assertEqual(recovered.returncode, 0, recovered.stdout + recovered.stderr)
                with (output / 'Master_genome_QC.tsv').open() as handle:
                    rows = list(csv.DictReader(handle, delimiter='\t'))
                self.assertTrue(all(r['checkm_status'] == 'completed' for r in rows), recovered.stdout + recovered.stderr + str(prior_receipts) + str({str(p):json.loads(p.read_text()) for p in output.glob('checkm/individual/*.assessment.json')}))
                self.assertTrue(all(r['trna_status'] == 'completed' for r in rows), rows)
                return
            self.assertEqual((output / 'genome_quality_atlas/ani/ani_clusters.tsv').is_file(), not skip_ani)
            tree = output / 'genome_quality_atlas/phylogeny/bac120.nwk'
            self.assertTrue(tree.is_file())
            self.assertTrue((tree.parent / 'bac120_annotated_tree.pdf').is_file())
            with (tree.parent / 'phylogeny_status.tsv').open() as handle:
                placed = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual({r['Genome_Id'] for r in placed if r['status']=='inferred'}, {f'g{i}' for i in range(10)})
            self.assertTrue(all(float(r['16S_rRNA']) == 0 for r in rows))
            self.assertTrue(all('gunc_clade_separation_score' in r for r in rows))
            self.assertTrue(all(r['gunc_clade_separation_score'] for r in rows))
            # Every genome appears once per tool batch set; no batch exceeds its cap.
            for directory in ('checkm/individual', 'gtdbtk/individual', 'gunc'):
                receipts=[json.loads(p.read_text()) for p in (output/directory).glob('*.assessment.json')]
                members=[g for r in receipts for g in r['genomes']]
                self.assertEqual(sorted(members), [f'g{i}.full' for i in range(10)])
                self.assertTrue(all(len(r['genomes']) <= 3 for r in receipts))
                loads=[sum(80000 if g=='g3.full' else 120000 for g in r['genomes']) for r in receipts]
                self.assertLessEqual(max(loads)-min(loads),120000)

            self.assertEqual({r['Genome_Id'] for r in rows if r['selected'] == 'False'}, {'g1', 'g2', 'g3'}, [(r['Genome_Id'], r['selection_reasons'], r.get('pass_BARRNAP')) for r in rows])
            with (output / 'Selected_genome_QC.tsv').open() as handle:
                selected = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(len(selected), 7)
            from genome_qc.cli import write_inventory
            write_inventory(discover(inputs), output)
            with (output / 'Genome_inventory.tsv').open() as handle:
                inventory = list(csv.DictReader(handle, delimiter='\t'))
            self.assertEqual(sum(r['status'] == 'assessed_not_selected' for r in inventory), 3)
            build(output)
            with sqlite3.connect(output / 'reports/results.sqlite') as db:
                self.assertEqual(db.execute('SELECT count(*) FROM genomes').fetchone()[0], 10)
                self.assertGreater(db.execute('SELECT count(*) FROM genome_rows').fetchone()[0], 100)
            # Controller updates and scheduling-only changes must not discard
            # completed biological work. The provenance digest is deliberately
            # different, just as after reinstalling a resource-policy fix.
            params['implementation_digest'] = 'controller-updated-after-first-run'
            (work / 'params.json').write_text(json.dumps(params))
            (work / 'test.config').write_text(config + "\nconda.enabled=false\nexecutor.queueSize=2\n")
            resumed = subprocess.run(command + ['-resume'], cwd=work, env=env, text=True, capture_output=True, timeout=300)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertIn('Cached process', resumed.stdout + resumed.stderr)
            self.assertNotIn('Submitted process', resumed.stdout + resumed.stderr)

            if not compact_mode:
                for criterion in ('completeness', 'contamination', 'min_genome_size'):
                    params.pop(criterion, None)
                (work / 'params.json').write_text(json.dumps(params))
                unrestricted = subprocess.run(command + ['-resume'], cwd=work, env=env,
                                              text=True, capture_output=True, timeout=300)
                self.assertEqual(unrestricted.returncode, 0, unrestricted.stdout + unrestricted.stderr)
                with (output / 'Selected_genome_QC.tsv').open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle, delimiter='\t'))), 10)
                params['completeness'] = 101
                (work / 'params.json').write_text(json.dumps(params))
                reselection = subprocess.run(command + ['-resume'], cwd=work, env=env,
                                            text=True, capture_output=True, timeout=300)
                self.assertEqual(reselection.returncode, 0, reselection.stdout + reselection.stderr)
                submitted = '\n'.join(line for line in (reselection.stdout + reselection.stderr).splitlines() if 'Submitted process' in line)
                self.assertIn('MERGE_MASTER', submitted)
                for name in ('CHECKM_SINGLE', 'GTDBTK', 'BARRNAP', 'TRNASCAN', 'GUNC'):
                    self.assertNotIn(name, submitted)
                with (output / 'Selected_genome_QC.tsv').open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle, delimiter='\t'))), 0)
                with annotated.open() as handle:
                    self.assertEqual(len(list(csv.DictReader(handle, delimiter='\t'))), 10)
                params['completeness'] = 50
                (work / 'params.json').write_text(json.dumps(params))
                helper = runtime / 'trnascan_parse.py'
                helper.write_text(helper.read_text() + '\n# Cache-invalidation regression fixture\n')
                changed = subprocess.run(command + ['-resume'], cwd=work, env=env,
                                         text=True, capture_output=True, timeout=300)
                messages = changed.stdout + changed.stderr
                self.assertEqual(changed.returncode, 0, messages)
                submitted = '\n'.join(line for line in messages.splitlines() if 'Submitted process' in line)
                self.assertIn('TRNASCAN_PARSE', submitted)
                self.assertNotIn('CHECKM_SINGLE', submitted)
                self.assertNotIn('FILTER_FASTA', submitted)

            if compact_mode:
                from genome_qc.compact import finish, completed
                finish(output, 'test', [work / 'work'], output / '.genome_qc/conda', False,
                       discover(inputs), work / 'db')
                self.assertTrue(completed(output, 'test'))
                self.assertFalse((work / 'work').exists())
                self.assertTrue(annotated.is_file())
