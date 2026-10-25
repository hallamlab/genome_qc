import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import database_install as installer


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = Path(self.temp.name) / 'archive.part'
        self.data = b'0123456789abcdef' * 2048
        self.expected = hashlib.md5(self.data).hexdigest()
        self.requests = []
        self.mode = 'range'
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                requested = self.headers.get('Range')
                owner.requests.append(requested)
                mode = owner.mode
                if mode == '404':
                    self.send_error(404); return
                offset = int(requested.split('=')[1].split('-')[0]) if requested else 0
                if mode == 'ignore':
                    offset = 0
                data = owner.data[offset:]
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header('Content-Range', f'bytes {offset}-{len(owner.data)-1}/{len(owner.data)}')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                if mode == 'broken' and len(owner.requests) == 1:
                    self.wfile.write(data[:8192])
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    self.wfile.write(data)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/archive'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_resume_and_verified_reuse(self):
        self.archive.write_bytes(self.data[:4096])
        installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(self.requests, ['bytes=4096-'])
        self.assertEqual(self.archive.read_bytes(), self.data)
        installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(len(self.requests), 1)

    def test_server_ignores_range(self):
        self.mode = 'ignore'
        self.archive.write_bytes(self.data[:4096])
        installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(self.archive.read_bytes(), self.data)

    def test_broken_transfer_retries_remaining_bytes(self):
        self.mode = 'broken'
        with patch.object(installer, 'CHUNK', 1024):
            installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(self.requests, [None, 'bytes=8192-'])
        self.assertEqual(self.archive.read_bytes(), self.data)

    def test_corrupt_prefix_restarts(self):
        self.archive.write_bytes(b'x' * 4096)
        installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(self.requests, ['bytes=4096-', None])
        self.assertEqual(self.archive.read_bytes(), self.data)

    def test_failure_retains_partial_for_next_invocation(self):
        self.mode = 'broken'
        with patch.object(installer, 'CHUNK', 1024):
            with self.assertRaisesRegex(RuntimeError, 'Rerun to resume'):
                installer.download(self.url, self.archive, self.expected, attempts=1, delay=0)
        self.assertEqual(self.archive.stat().st_size, 8192)
        installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(self.archive.read_bytes(), self.data)

    def test_permanent_http_error_not_retried(self):
        self.mode = '404'
        self.archive.write_bytes(b'valid partial')
        with self.assertRaisesRegex(RuntimeError, 'HTTP 404'):
            installer.download(self.url, self.archive, self.expected, delay=0)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.archive.read_bytes(), b'valid partial')


class InstallRecoveryTests(unittest.TestCase):
    def test_verified_archive_survives_extraction_failure(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            for directory in ('hmms', 'genome_tree'):
                (source / directory).mkdir(parents=True)
            for filename in ('hmms/checkm.hmm', 'hmms/phylo.hmm', 'taxon_marker_sets.tsv'):
                (source / filename).write_text('fixture')
            archive = root / 'source.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                handle.add(source, arcname='source')
            expected = hashlib.md5(archive.read_bytes()).hexdigest()
            destination = root / 'db/checkm_data'
            with patch.dict(installer.SPECS, checkm=(archive.as_uri(), expected)):
                with patch.object(tarfile.TarFile, 'extractall', side_effect=OSError('disk full')):
                    with self.assertRaisesRegex(OSError, 'disk full'):
                        installer.install('checkm', destination)
                self.assertFalse(destination.exists())
                self.assertEqual(len(list((root / 'db/.downloads').glob('*.part'))), 1)
                with patch('database_install.urllib.request.urlopen', side_effect=AssertionError('must reuse archive')):
                    installer.install('checkm', destination)
            self.assertTrue(installer.valid('checkm', destination))
            self.assertEqual(list((root / 'db/.downloads').glob('*.part')), [])

class ReleaseValidationTests(unittest.TestCase):
    def test_gtdb_release_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('markers', 'masks', 'pplacer', 'metadata', 'radii', 'taxonomy', 'mrca_red', 'skani', 'split'):
                (root / name).mkdir()
            metadata = root / 'metadata/metadata.txt'
            metadata.write_text('VERSION_DATA=r220\n')
            self.assertFalse(installer.valid('gtdbtk', root))
            metadata.write_text('VERSION_DATA=r226\n')
            self.assertTrue(installer.valid('gtdbtk', root))
            metadata.unlink()
            self.assertFalse(installer.valid('gtdbtk', root))

class SuppliedArchiveTests(unittest.TestCase):
    def test_local_archive_verified_extracted_and_preserved(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            for directory in ('hmms', 'genome_tree'):
                (source / directory).mkdir(parents=True)
            for filename in ('hmms/checkm.hmm', 'hmms/phylo.hmm', 'taxon_marker_sets.tsv'):
                (source / filename).write_text('fixture')
            archive = root / 'checkm_data.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                handle.add(source, arcname='reference')
            checksum = hashlib.md5(archive.read_bytes()).hexdigest()
            target = root / 'checkm_data'
            url = 'https://example.invalid/checkm_data.tar.gz'
            with patch.dict(installer.SPECS, checkm=(url, checksum)), patch(
                    'database_install.urllib.request.urlopen', side_effect=AssertionError('no network')):
                installer.install('checkm', target)
            self.assertTrue(installer.valid('checkm', target))
            self.assertTrue(archive.is_file())
            self.assertEqual(hashlib.md5(archive.read_bytes()).hexdigest(), checksum)

    def test_bad_local_archive_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'gtdbtk_r226_data.tar.gz'
            archive.write_bytes(b'bad archive')
            with patch('database_install.urllib.request.urlopen', side_effect=AssertionError('no network')):
                with self.assertRaisesRegex(RuntimeError, 'original preserved'):
                    installer.install('gtdbtk', root / 'gtdbtk_r226')
            self.assertEqual(archive.read_bytes(), b'bad archive')
            self.assertFalse((root / 'gtdbtk_r226').exists())


@unittest.skipUnless(__import__('shutil').which('rapidgzip'), 'rapidgzip integration dependency')
class ParallelExtractionTests(unittest.TestCase):
    def test_parallel_and_serial_contents_identical(self):
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'; source.mkdir()
            (source / 'data').write_bytes(b'ACGT' * 3000000)
            archive = root / 'data.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                handle.add(source, arcname='source')
            for threads in (1, 4):
                output = root / str(threads); output.mkdir()
                installer.extract_archive(archive, output, threads)
                self.assertEqual(installer.hashes(output / 'source/data'), installer.hashes(source / 'data'))

    def test_parallel_rejects_path_escape(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                info = tarfile.TarInfo('../escaped'); info.size = 4
                handle.addfile(info, io.BytesIO(b'nope'))
            output = root / 'output'; output.mkdir()
            with self.assertRaises(tarfile.FilterError):
                installer.extract_archive(archive, output, 4)
            self.assertFalse((root / 'escaped').exists())
