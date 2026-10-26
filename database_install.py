"""Locked, atomic database installation; called in a tool-specific environment."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import urllib.error
import http.client
import re
import time

SPECS = {
    'checkm': ('https://zenodo.org/records/7401545/files/checkm_data_2015_01_16.tar.gz', '631012fa598c43fdeb88c619ad282c4d'),
    'gtdbtk': ('https://data.gtdb.ecogenomic.org/releases/release232/232.0/auxillary_files/gtdbtk_package/full_package/gtdbtk_r232_data.tar.gz', '25a59e0352b1fd150c589f56559767d4'),
}



CHUNK = 8 * 1024 * 1024


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b''):
            md5.update(chunk); sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def download(url, archive, expected, attempts=5, delay=5):
    """Caller holds the destination lock; pinned checksum guards resumed bytes."""
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        print(f'Checking retained download: {archive}', flush=True)
        retained = hashes(archive)
        if retained[0] == expected:
            print(f'Using verified download: {archive}', flush=True)
            return retained
    for attempt in range(1, attempts + 1):
        offset = archive.stat().st_size if archive.exists() else 0
        print(f'Download attempt {attempt}/{attempts}: {url}; resuming at {offset / 1024**3:.2f} GiB', flush=True)
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'GenomeQC-database-installer'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        start = last = time.monotonic()
        received = 0
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                status = getattr(response, 'status', None)
                length = response.headers.get('Content-Length')
                length = int(length) if length is not None else None
                total = None
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if not match or int(match[1]) != offset or int(match[2]) < offset or int(match[3]) <= int(match[2]):
                        raise RuntimeError('Invalid download Content-Range; refusing to append')
                    total = int(match[3])
                    span = int(match[2]) - offset + 1
                    if length is not None and length != span:
                        raise RuntimeError('Inconsistent download response length')
                    length = span
                else:
                    # A server may ignore Range; never append a full response.
                    if offset:
                        print('Server did not resume; restarting this archive safely.', flush=True)
                    offset = 0
                    total = length
                with archive.open('ab' if offset else 'wb') as out:
                    while True:
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        out.write(chunk); received += len(chunk)
                        now = time.monotonic()
                        if now - last >= 10:
                            done = offset + received
                            fraction = f' / {total / 1024**3:.2f} GiB ({100 * done / total:.1f}%)' if total else ' GiB'
                            rate = received / max(now - start, .001) / 1024**2
                            print(f'Download: {done / 1024**3:.2f}{fraction}; {rate:.1f} MiB/s', flush=True)
                            last = now
                if length is not None and received != length:
                    raise ConnectionError(f'Truncated response: received {received} of {length} bytes')
                if total is not None and archive.stat().st_size < total:
                    raise ConnectionError('Partial response complete; more archive bytes remain')
            print('Download complete; verifying checksum...', flush=True)
            result = hashes(archive)
            if result[0] != expected:
                archive.unlink()
                raise ConnectionError('Archive checksum mismatch; discarded corrupt download')
            return result
        except urllib.error.HTTPError as error:
            if error.code == 416:
                # Existing complete-but-corrupt or stale partial: restart safely.
                archive.unlink(missing_ok=True)
            elif error.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f'Download HTTP {error.code}: {url}; partial retained at {archive}') from error
            failure = str(error)
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as error:
            failure = str(error)
        if attempt == attempts:
            raise RuntimeError(f'Download failed after {attempts} attempts: {failure}. Rerun to resume {archive}')
        pause = min(delay * 2 ** (attempt - 1), 60)
        print(f'Download interrupted: {failure}; retrying in {pause}s. Partial file retained.', flush=True)
        time.sleep(pause)

def valid(name, root):
    if name not in ('checkm', 'gtdbtk', 'gunc'):
        raise ValueError(f'Unknown reference database: {name}')
    root = Path(root)
    if name == 'checkm':
        return all((root / p).exists() for p in ('hmms/checkm.hmm', 'hmms/phylo.hmm', 'genome_tree', 'taxon_marker_sets.tsv'))
    if name == 'gtdbtk':
        metadata = root / 'metadata/metadata.txt'
        if not metadata.is_file():
            return False
        release = re.search(r'^VERSION_DATA\s*=\s*[\'"]?(r[0-9]+)', metadata.read_text(), re.M)
        return bool(release and release[1] == 'r232') and all(
            (root / p).is_dir() for p in ('markers', 'masks', 'pplacer', 'metadata', 'radii', 'taxonomy', 'mrca_red', 'skani', 'split'))
    return (root / 'gunc_db_progenomes2.1.dmnd').is_file() and (root / 'gunc_db_progenomes2.1.dmnd').stat().st_size > 0



def extract_archive(archive, payload, threads):
    threads = int(threads)
    if threads < 1:
        raise ValueError('Extraction threads must be positive')
    if threads == 1:
        with tarfile.open(archive) as tar:
            tar.extractall(payload, filter='data')
        return
    workers = threads - 1
    print(f'Extracting with rapidgzip: {workers} decompression workers + tar extraction', flush=True)
    with subprocess.Popen(['rapidgzip', '-d', '-c', '-P', str(workers), str(archive)],
                          stdout=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
                tar.extractall(payload, filter='data')
            # Drain padding so decompressor integrity checks finish before publication.
            while process.stdout.read(CHUNK):
                pass
            if process.wait():
                raise RuntimeError('rapidgzip failed; extraction not published')
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()

def install(name, destination, threads=1):
    if name not in ('checkm', 'gtdbtk', 'gunc'):
        raise ValueError(f'Unknown reference database: {name}')
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / ('.' + dest.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if valid(name, dest):
            print(f'{name}: using {dest}', flush=True)
            return
        if dest.exists():
            raise RuntimeError(f'{dest} exists but is incomplete. Preserve/remove it explicitly or choose a new database directory.')
        with tempfile.TemporaryDirectory(prefix='.' + name + '-install-', dir=dest.parent) as tmp:
            work = Path(tmp); payload = work / 'payload'; payload.mkdir()
            receipt = {'database': name}
            if name in SPECS:
                url, expected = SPECS[name]
                key = hashlib.sha256((url + expected).encode()).hexdigest()[:24]
                supplied = dest.parent / Path(urllib.parse.urlparse(url).path).name
                managed_archive = not supplied.is_file()
                archive = (dest.parent / '.downloads' / (dest.name + '-' + key + '.tar.gz.part')) if managed_archive else supplied
                if managed_archive:
                    md5, sha = download(url, archive, expected)
                else:
                    print(f'{name}: verifying existing archive {archive}; no download needed', flush=True)
                    md5, sha = hashes(archive)
                    if md5 != expected:
                        raise RuntimeError(f'{name}: checksum mismatch for supplied archive {archive}; original preserved, installation not published')
                    receipt['source_archive'] = str(archive)
                print(f'{name}: extracting verified archive; temporary extraction is in {work}', flush=True)
                extract_archive(archive, payload, threads)
                candidates = [payload] + [p for p in payload.iterdir() if p.is_dir()]
                payload = next((p for p in candidates if valid(name, p)), payload)
                receipt.update(source_url=url, md5=md5, sha256=sha)
            else:
                subprocess.run(['gunc', 'download_db', str(payload), '--database', 'progenomes_2.1'], check=True)
                db = payload / 'gunc_db_progenomes2.1.dmnd'
                sha = hashlib.sha256()
                with db.open('rb') as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): sha.update(chunk)
                receipt.update(source='gunc download_db --database progenomes_2.1', sha256=sha.hexdigest())
            if not valid(name, payload):
                raise RuntimeError(f'{name}: extracted database is incomplete')
            receipt['files'] = {str(p.relative_to(payload)): p.stat().st_size for p in payload.rglob('*') if p.is_file()}
            (payload / 'genome_qc_database.json').write_text(json.dumps(receipt, indent=2) + '\n')
            os.replace(payload, dest)
            if name in SPECS and managed_archive:
                archive.unlink(missing_ok=True)
            print(f'{name}: installed {dest}', flush=True)


if __name__ == '__main__':
    install(*sys.argv[1:])
