"""Inventory inference tables and plots for review; no analysis is run."""
import argparse
import csv
from pathlib import Path

STAT_TOKENS = ('pvalue', 'p_value', 'qvalue', 'q_value', 'significance', 'significant', 'pval', 'qval')


def write_inventory(root, destination=None):
    root = Path(root).resolve()
    destination = Path(destination) if destination else root / 'statistics_review_inventory.tsv'
    rows = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or 'previous_statistics' in path.parts or path == destination:
            continue
        relative = str(path.relative_to(root))
        if path.suffix.lower() == '.tsv':
            try:
                with path.open() as handle:
                    header = next(csv.reader(handle, delimiter='\t'), [])
            except (OSError, UnicodeDecodeError):
                continue
            columns = [c for c in header if any(token in c.lower() for token in STAT_TOKENS)]
            if columns:
                rows.append((relative, 'statistics_table', ';'.join(columns), 'contains statistical columns'))
        elif path.suffix.lower() in ('.png', '.pdf', '.svg'):
            # Never claim that a filename alone proves whether stars/CI are drawn.
            effect = '_method_effects_' in path.name
            candidate = any(t in path.name.lower() for t in ('_sig', 'enrich', 'volcano', 'preference', 'delta', 'ci95', 'benchmark'))
            rows.append((relative, 'statistical_effect_plot' if effect else 'plot_review_candidate' if candidate else 'other_plot',
                         '', 'dedicated mean-effect CI plot' if effect else 'inspect for inference, CI, or significance-based selection'))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w') as handle:
        writer = csv.writer(handle, delimiter='\t')
        writer.writerow(['relative_path', 'review_type', 'statistical_columns', 'note'])
        writer.writerows(rows)
    return destination


def archive_previous_statistics(root):
    """Move old inference outputs aside on rerun; preserve raw data and summaries."""
    from datetime import datetime, timezone
    root = Path(root)
    if not root.exists():
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = root / 'previous_statistics' / stamp
    for p in list(root.rglob('*')):
        if not p.is_file() or 'previous_statistics' in p.parts:
            continue
        move = p.name == 'statistics_review_inventory.tsv'
        if p.suffix == '.tsv':
            try:
                with p.open() as handle:
                    header = next(csv.reader(handle, delimiter='\t'), [])
                move |= any(any(token in c.lower() for token in STAT_TOKENS) for c in header)
            except (OSError, UnicodeDecodeError):
                pass
        if p.suffix in ('.png', '.pdf', '.svg'):
            move |= '_sig' in p.stem or '_method_effects_' in p.stem
        if move:
            target = archive / p.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            p.rename(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    parser.add_argument('-o', '--output', default=None)
    args = parser.parse_args()
    print(write_inventory(args.output_dir, args.output))
