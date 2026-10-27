#!/usr/bin/env python3
"""Regenerate public CLI help; --help never runs biological analyses.

The standalone parsers require their Python import dependencies (NumPy, pandas,
SciPy, etc.) in the documentation-generation environment, but no reference DBs.
"""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genome_qc.cli import parser, ENTRIES, SCRIPTS
p = parser()
parts = ['''# CLI reference

Generated from the command parsers by `scripts/build_cli_reference.py`.
Every public subcommand is included, including the nested tree-render commands.
See [commands](commands.md) for choosing a module, [resources](resources.md) for
runtime resource defaults, [selection](selection.md) for acceptance criteria,
and the [output dictionary](output-columns.md) for the resulting measurements.

## Common option meanings

Options apply only where shown in the corresponding help block. The complete
workflow uses underscore-style flags; standalone atlas/comparison/render commands
use their own hyphen-style flags. These are not interchangeable aliases.

| Option | Meaning / default |
| --- | --- |
| `-i`, `--input` | Genome FASTA directory, searched recursively. Mutually exclusive with `--manifest`. |
| `--manifest` | TSV containing unique `id` and `fasta`; relative FASTA paths resolve beside the manifest. Extra fields are retained as metadata. |
| `-o`, `--output` | Required analysis output directory. Keep outside the recursively searched input tree. Standalone analyses may call this `--output-dir`; read their help. |
| `-d`, `--database` | Reference root; default `~/.local/share/genome_qc/databases`. Complete existing references are reused. |
| `--threads` | CPUs requested per parallel tool (default 8 for workflow stages); not an aggregate machine/job limit. See `--max_cpus`. |
| `--resume` | Ask Nextflow to reuse matching cached work. Requires the original inputs/settings and work/cache locations. Completed compact results are verified/reused without needing task caches. |
| `--plan` | Validate/stage inputs and save commands/settings without launching Nextflow or downloading references. Creates files: use a disposable planning output directory. |
| `--params` | Advanced Nextflow parameters in a JSON object. Explicit CLI selection/resource settings can override corresponding values; inspect the saved `params.json`. This is not a free-form shell config. |
| `--filter_size` | Contig minimum-length variants, e.g. `--filter_size "1000 2500"`; absent means all contigs (`full`). This creates variants; final acceptance thresholds are separate. |
| `--kingdom` | RNA-model choice for standalone RNA commands; default Bacteria. The complete workflow uses available GTDB taxonomy with recorded fallbacks. |
| `--serve` | Start the local report HTTP server; keep the terminal alive while browsing. |
| `--port` | Report port, default 8765. For remote use, match the SSH forwarding port. |
| `--version` | Print controller version and exit. |
| `-h`, `--help` | Print help for the current command and exit. |

`check` checks availability of Nextflow, Java and Mamba on PATH; it does not
download databases or validate the scientific tools. `--max_tasks` caps queued
and running worker tasks, not controller threads or heap size.

**Different ANI defaults:** `run` defaults to 98% ANI and larger-assembly AF 0.51
with labelled containment rescue. Standalone `atlas` / `compare` retain their
comparison defaults (95% ANI, AF 0.5) and their separate comparison logic. Do not
assume they reproduce `run` cohort groups without reviewing their configuration.

## Top-level command
''', '```text\n'+p.format_help()+'```\n']
sub = next(a for a in p._actions if getattr(a, 'choices', None))
for name in [*ENTRIES, 'report', 'check']:
    parts += [f'## {name}\n', '```text\n'+sub.choices[name].format_help()+'```\n']
for name, script in SCRIPTS.items():
    help_text = subprocess.check_output([sys.executable, str(ROOT/script), '--help'], text=True, cwd=ROOT)
    parts += [f'## {name}\n', '```text\n'+help_text.replace(script, 'genome_qc '+name)+'```\n']
    if name == 'render':
        for command in ('init-config', 'render', 'batch', 'convert-config'):
            text = subprocess.check_output([sys.executable, str(ROOT/script), command, '--help'], text=True, cwd=ROOT)
            parts += [f'### render {command}\n', '```text\n'+text.replace(script, 'genome_qc render')+'```\n']
(ROOT/'docs/user-guide/cli-reference.md').write_text('\n'.join(parts))
