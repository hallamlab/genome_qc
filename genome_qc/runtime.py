from pathlib import Path
import sys


def runtime_root():
    source = Path(__file__).resolve().parents[1]
    if (source / 'main.nf').is_file():
        return source
    installed = Path(sys.prefix) / 'share/genome_qc'
    if (installed / 'main.nf').is_file():
        return installed
    raise RuntimeError('Genome QC workflow assets are missing; reinstall the package.')
