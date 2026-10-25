"""Merge GUNC batch summaries without overwriting genomes from earlier batches."""
import csv
from pathlib import Path
import sys


def merge(destination, directories):
    destination=Path(destination); destination.mkdir(parents=True, exist_ok=True)
    groups={}
    for directory in directories:
        for path in sorted(Path(directory).glob('GUNC.*.maxCSS_level.tsv')):
            groups.setdefault(path.name, []).append(path)
    for name, paths in groups.items():
        columns=[]; rows=[]; seen=set()
        for path in paths:
            with path.open() as handle:
                reader=csv.DictReader(handle, delimiter='\t')
                columns.extend(c for c in (reader.fieldnames or []) if c not in columns)
                for row in reader:
                    genome=row.get('genome')
                    if genome in seen:
                        raise ValueError(f'Duplicate genome in GUNC batches: {genome}')
                    seen.add(genome); rows.append(row)
        with (destination/name).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=columns,delimiter='\t')
            writer.writeheader(); writer.writerows(sorted(rows,key=lambda row:row.get('genome','')))


if __name__ == '__main__':
    merge(sys.argv[1],sys.argv[2:])
