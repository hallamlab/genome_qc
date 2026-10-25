#!/usr/bin/env python3
"""Regenerate controller help without loading the biological analysis environments."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genome_qc.cli import parser, ENTRIES
p = parser()
parts = ['# CLI reference\n\nGenerated from the controller parser by `scripts/build_cli_reference.py`.\nFor standalone atlas/comparison/tree commands, use their individual `--help`;\nsee [commands](commands.md). Resource defaults resolved at runtime are explained\nin [resources](resources.md).\n', '```text\n'+p.format_help()+'```\n']
sub = next(a for a in p._actions if getattr(a, 'choices', None))
for name in [*ENTRIES, 'report', 'check']:
    parts += [f'## {name}\n', '```text\n'+sub.choices[name].format_help()+'```\n']
(ROOT/'docs/user-guide/cli-reference.md').write_text('\n'.join(parts))
