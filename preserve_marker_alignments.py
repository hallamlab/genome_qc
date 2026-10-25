"""Preserve GTDB masked user marker alignments before compact cleanup."""
import gzip
from pathlib import Path


def fasta(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        name, seq = None, []
        for line in f:
            if line.startswith('>'):
                if name is not None:
                    yield name, ''.join(seq)
                name, seq = line[1:].split()[0], []
            else:
                seq.append(line.strip())
        if name is not None:
            yield name, ''.join(seq)


def stage_msa(source, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for domain in ('bac120', 'ar53'):
        candidates = sorted(Path(source).rglob(f'gtdbtk.{domain}.user_msa.fasta*'))
        if candidates:
            # The native align file and top-level symlink can both be present.
            content = dict(fasta(candidates[0]))
            for other in candidates[1:]:
                if dict(fasta(other)) != content:
                    raise ValueError(f'Conflicting {domain} user alignments in {source}')
            with gzip.open(output / f'{domain}.fasta.gz', 'wt') as handle:
                for name, seq in sorted(content.items()):
                    handle.write(f'>{name}\n{seq}\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('output')
    args = parser.parse_args()
    stage_msa(args.source, args.output)
