"""Locked, atomic database installation; called in a tool-specific environment."""
import argparse
import gzip
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

# GUNC pins match gunc.gunc_database DB_DOWNLOADS['progenomes_2.1'].
GUNC_DMND_MD5 = '447c9330056b02f29f30fe81fe4af4eb'
SPECS = {
    'gunc': ('https://zenodo.org/records/19631736/files/gunc_db_progenomes2.1.dmnd.gz', 'bc93a855e0760aad5c4e5f2d0e26da46'),
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

# Conservative planning allowances, not user/project quota measurements.
SPACE_GIB = {'checkm': 3, 'gtdbtk': 180, 'gunc': 20}


def space_check(name, directory, download_only=False, retained_bytes=0):
    free = shutil.disk_usage(directory).free
    estimate = SPACE_GIB[name] * 1024**3
    if download_only:
        estimate = {'checkm': 1, 'gtdbtk': 65, 'gunc': 8}[name] * 1024**3
    estimate = max(0, estimate - retained_bytes)
    print(f'{name}: filesystem free={free / 1024**3:.1f} GiB; planning allowance={estimate / 1024**3:.0f} GiB. User/project quotas may be smaller.', flush=True)
    if free < estimate:
        raise RuntimeError(f'{name}: insufficient filesystem space for preparation; choose a larger database location with -d')


def install(name, destination, threads=1, download_only=False, extract_only=False, attempts=20, url=None):
    if name not in SPECS:
        raise ValueError(f'Unknown reference database: {name}')
    if download_only and extract_only:
        raise ValueError('download_only and extract_only are mutually exclusive')
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / ('.' + dest.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # Only remove our marked staging directories for this exact destination.
        # The destination lock proves no other installer for it is active.
        for stale in dest.parent.glob('.' + name + '-install-*'):
            marker = stale / '.genome-qc-install.json'
            try:
                owned = not stale.is_symlink() and stale.stat().st_uid == os.getuid()
                if owned and json.loads(marker.read_text()).get('destination') == str(dest):
                    shutil.rmtree(stale)
            except (OSError, ValueError):
                pass
        if valid(name, dest) and not download_only:
            print(f'{name}: using {dest}', flush=True)
            return
        if dest.exists() and not download_only:
            raise RuntimeError(f'{dest} exists but is incomplete. Preserve/remove it explicitly or choose a new database directory.')
        canonical_url, expected = SPECS[name]
        source_url = url or canonical_url
        basename = Path(urllib.parse.urlparse(canonical_url).path).name
        supplied = dest.parent / basename
        key = hashlib.sha256((canonical_url + expected).encode()).hexdigest()[:24]
        managed = dest.parent / '.downloads' / (dest.name + '-' + key + '.tar.gz.part')
        archive = supplied if supplied.is_file() else managed
        if extract_only and not archive.is_file():
            raise RuntimeError(f'{name}: --extract_only requires {supplied} (or a retained verified download); no download attempted')
        if supplied.is_file() or extract_only:
            md5, sha = hashes(archive)
            if md5 != expected:
                raise RuntimeError(f'{name}: checksum mismatch for supplied archive {archive}; original preserved, installation not published')
            if not download_only:
                space_check(name, dest.parent, retained_bytes=archive.stat().st_size)
        else:
            space_check(name, dest.parent, download_only)
            md5, sha = download(source_url, archive, expected, attempts=attempts)
        if download_only:
            if archive != supplied:
                archive.replace(supplied)
            supplied.with_name(supplied.name + '.provenance.json').write_text(json.dumps(
                dict(source_url=source_url, md5=md5, sha256=sha, bytes=supplied.stat().st_size), indent=2) + '\n')
            print(f'{name}: verified archive ready: {supplied}', flush=True)
            return
        with tempfile.TemporaryDirectory(prefix='.' + name + '-install-', dir=dest.parent) as tmp:
            work = Path(tmp)
            (work / '.genome-qc-install.json').write_text(json.dumps({'destination': str(dest)}))
            payload = work / 'payload'; payload.mkdir()
            print(f'{name}: extracting verified archive; temporary extraction is in {work}', flush=True)
            if name == 'gunc':
                target = payload / 'gunc_db_progenomes2.1.dmnd'
                with gzip.open(archive, 'rb') as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output, CHUNK)
                if hashes(target)[0] != GUNC_DMND_MD5:
                    raise RuntimeError('GUNC extracted database checksum mismatch')
            else:
                extract_archive(archive, payload, threads)
                candidates = [payload] + [p for p in payload.iterdir() if p.is_dir()]
                payload = next((p for p in candidates if valid(name, p)), payload)
            if not valid(name, payload):
                raise RuntimeError(f'{name}: extracted database is incomplete; installation not published')
            receipt = dict(database=name, source_url=source_url, md5=md5, sha256=sha,
                           source_archive=str(archive), archive_bytes=archive.stat().st_size)
            receipt['files'] = {str(p.relative_to(payload)): p.stat().st_size for p in payload.rglob('*') if p.is_file()}
            (payload / 'genome_qc_database.json').write_text(json.dumps(receipt, indent=2) + '\n')
            os.replace(payload, dest)
            if archive == managed:
                archive.unlink(missing_ok=True)
            print(f'{name}: installed {dest}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name', choices=sorted(SPECS)); p.add_argument('destination')
    p.add_argument('threads', type=int, nargs='?', default=1)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true')
    mode.add_argument('--extract-only', action='store_true')
    p.add_argument('--attempts', type=int, default=20)
    p.add_argument('--url')
    a = p.parse_args()
    if a.threads < 1 or a.attempts < 1:
        p.error('threads and attempts must be positive')
    install(a.name, a.destination, a.threads, a.download_only, a.extract_only, a.attempts, a.url)
