"""Remove disposable tool products before a compact task leaves worker scratch."""
from pathlib import Path
import os
import sys


def compact(directory):
    root = Path(directory)
    keep = {'.tsv', '.csv', '.log', '.txt', '.json', '.tree', '.nwk'}
    for directory, dirs, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if path.suffix not in keep:
                path.unlink()
        for name in dirs:
            path = Path(directory) / name
            if path.is_symlink():
                path.unlink()
            elif not any(path.iterdir()):
                path.rmdir()


if __name__ == '__main__':
    compact(sys.argv[1])
