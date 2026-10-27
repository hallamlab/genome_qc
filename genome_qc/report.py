"""Read-only, local table explorer backed by an output-local SQLite index."""
import csv
import io
import json
import mimetypes
import os
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlparse

from .explorer import HTML
import math
import re


# Order the gallery by interpretation, not filename or directory traversal.
# Match suffixes so custom atlas prefixes retain the same presentation.
FIGURE_SEQUENCE = (
    '_atlas', '_integrity_recoverability', '_qscore_distribution',
    '_hallmark_contributions', '_recovery_patterns', '_taxonomy_qscore',
    '_threshold_landscape',
)


def artifact_order(path):
    path = Path(path)
    if path.parts and path.parts[0] == 'genome_quality_atlas':
        if len(path.parts) == 2:
            for rank, suffix in enumerate(FIGURE_SEQUENCE):
                if path.stem.endswith(suffix):
                    return (rank, path.as_posix())
        if 'ani' in path.parts:
            return (7, path.as_posix())
        if 'phylogeny' in path.parts:
            return (8, path.as_posix())
        return (9, path.as_posix())
    return (10, path.as_posix())


def artifacts(db):
    return sorted((dict(id=i, path=path) for i, path in
                   db.execute('SELECT id,path FROM artifacts')),
                  key=lambda item: artifact_order(item['path']))


def build(output):
    output = Path(output).resolve()
    reports = output / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    temporary = reports / 'results.build.sqlite'
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as db:
        db.execute('CREATE TABLE catalog (id TEXT PRIMARY KEY, source TEXT, rows INTEGER, columns TEXT)')
        db.execute('CREATE TABLE genomes (genome_id TEXT PRIMARY KEY, fasta TEXT, sha256 TEXT, metadata TEXT)')
        db.execute('CREATE TABLE artifacts (id INTEGER PRIMARY KEY, path TEXT)')
        db.execute('CREATE TABLE genome_rows (genome_id TEXT, table_id TEXT, row_id INTEGER)')
        db.execute('CREATE INDEX genome_rows_lookup ON genome_rows(genome_id,table_id,row_id)')
        manifest = output / 'inputs.resolved.json'
        genomes = json.loads(manifest.read_text()) if manifest.exists() else []
        aliases = {}
        for genome in genomes:
            identity = genome['id']
            db.execute('INSERT INTO genomes VALUES (?,?,?,?)',
                       (identity, genome['fasta'], genome['sha256'], json.dumps(genome)))
            aliases[identity] = identity
        # Derive aliases from actual preprocessing outputs, not arbitrary dot splitting.
        for fasta in (output / 'genomes_filtered').glob('*.fasta'):
            variant = fasta.stem
            identity, _, threshold = variant.rpartition('.')
            if identity in aliases and (threshold == 'full' or threshold.isdigit()):
                aliases[variant] = identity
        # Work directories and reports are excluded; published scientific tables remain authoritative.
        paths = []
        for directory, dirs, files in os.walk(output):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('logs', 'reports')]
            paths.extend(Path(directory) / name for name in files if Path(name).suffix in ('.tsv', '.csv'))
            for name in sorted(files):
                artifact = Path(directory) / name
                if artifact.suffix in ('.pdf', '.png', '.svg', '.html'):
                    db.execute('INSERT INTO artifacts(path) VALUES (?)', (str(artifact.relative_to(output)),))
        paths.sort()
        for path in paths:
            with path.open(newline='', encoding='utf-8-sig') as handle:
                reader = csv.reader(handle, delimiter='\t' if path.suffix == '.tsv' else ',')
                columns = next(reader, [])
                if not columns:
                    continue
                table = f't{len(db.execute("SELECT id FROM catalog").fetchall())}'
                db.execute(f'CREATE TABLE {table} (' + ','.join(f'c{i} TEXT' for i in range(len(columns))) + ')')
                count = 0
                id_columns = [i for i, name in enumerate(columns)
                              if name.lower().replace(' ', '_') in ('genome_id', 'genome', 'genome_a', 'genome_b', 'bin_id', 'user_genome', 'file')]
                for row in reader:
                    if not row:
                        continue
                    if len(row) != len(columns):
                        raise ValueError(f'Inconsistent table row in {path}: row {count + 2}')
                    db.execute(f'INSERT INTO {table} VALUES (' + ','.join('?' for _ in row) + ')', row)
                    count += 1
                    linked = set()
                    for i in id_columns:
                        value = row[i]
                        if columns[i].lower() == 'file':
                            value = Path(value).name.removesuffix('.fasta')
                        if value in aliases:
                            linked.add(aliases[value])
                    for identity in linked:
                        db.execute('INSERT INTO genome_rows VALUES (?,?,?)', (identity, table, count))
                db.execute('INSERT INTO catalog VALUES (?,?,?,?)',
                           (table, str(path.relative_to(output)), count, json.dumps(columns)))
    db.close()
    temporary.replace(reports / 'results.sqlite')
    (reports / 'EDA_portal.html').write_text(HTML)
    print(f'Explorer: {reports / "EDA_portal.html"}; serve with genome_qc report -o {output} --serve')


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def columns_for(db, table):
    if not re.fullmatch(r't[0-9]+', table):
        raise ValueError('Unknown table')
    meta = db.execute('SELECT columns FROM catalog WHERE id=?', (table,)).fetchone()
    if not meta:
        raise ValueError('Unknown table')
    return json.loads(meta[0])


def predicate(db, table, search='', filters=None, alias='r'):
    columns = columns_for(db, table)
    terms, values = ['1=1'], []
    if search:
        terms.append('(' + ' OR '.join(f'instr(lower({alias}.c{i}), lower(?)) > 0' for i in range(len(columns))) + ')')
        values.extend([search] * len(columns))
    if not isinstance(filters or [], list) or len(filters or []) > 30:
        raise ValueError('Use at most 30 filters')
    for rule in filters or []:
        if not isinstance(rule, dict) or rule.get('column') not in columns:
            raise ValueError('Unknown filter column')
        col = f'{alias}.c{columns.index(rule["column"])}'
        op = rule.get('op')
        if op == 'range':
            bounds = [(key, sign) for key, sign in [('min', '>='), ('max', '<=')] if rule.get(key) not in (None, '')]
            if not bounds:
                raise ValueError('A range needs a minimum or maximum')
            for key, sign in bounds:
                val = number(rule[key])
                if val is None:
                    raise ValueError('Range bounds must be finite numbers')
                terms.append(f'finite_number({col}) {sign} ?'); values.append(val)
            if len(bounds) == 2 and number(rule['min']) > number(rule['max']):
                raise ValueError('Minimum exceeds maximum')
        elif op == 'eq':
            terms.append(f'{col} = ?'); values.append(str(rule.get('value', '')))
        elif op == 'contains':
            terms.append(f'instr(lower({col}), lower(?)) > 0'); values.append(str(rule.get('value', '')))
        elif op == 'missing':
            terms.append(f"({col} IS NULL OR lower(trim({col})) IN ('', 'na', 'n/a', 'nan', 'none', 'null'))")
        else:
            raise ValueError('Unknown filter operation')
    return ' AND '.join(terms), values


def schema(db, table):
    columns = columns_for(db, table)
    result = []
    taxonomy = {'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'}
    for i, name in enumerate(columns):
        vals = [r[0] for r in db.execute(f'SELECT DISTINCT c{i} FROM {table}')]
        present = [v for v in vals if v is not None and v.strip().lower() not in ('', 'na', 'n/a', 'nan', 'none', 'null')]
        nums = [number(v) for v in present]
        numeric = bool(present) and all(v is not None for v in nums)
        result.append(dict(name=name, numeric=numeric, minimum=min(nums) if numeric else None,
                           maximum=max(nums) if numeric else None,
                           choices=sorted(present) if not numeric and (len(present) <= 60 or name in taxonomy) else []))
    return result


def query(db, table, search='', page=0, export=False, genome='', filters=None,
          sort='', direction='asc', selected=None, cohort=None):
    if not db.execute("SELECT 1 FROM pragma_function_list WHERE name='finite_number'").fetchone():
        db.create_function('finite_number', 1, number, deterministic=True)
    columns = columns_for(db, table)
    where, values = predicate(db, table, search, filters)
    if genome:
        where += ' AND r.rowid IN (SELECT row_id FROM genome_rows WHERE genome_id=? AND table_id=?)'
        values += [genome, table]
    if cohort:
        if not isinstance(cohort, dict):
            raise ValueError('Invalid genome subset')
        source = cohort.get('table', '')
        condition, cv = predicate(db, source, cohort.get('q', ''), cohort.get('filters'), 's')
        where += f""" AND r.rowid IN (SELECT dest.row_id FROM genome_rows dest
            JOIN genome_rows origin ON origin.genome_id=dest.genome_id
            JOIN {source} s ON s.rowid=origin.row_id
            WHERE dest.table_id=? AND origin.table_id=? AND {condition})"""
        values += [table, source, *cv]
    total = db.execute(f'SELECT count(*) FROM {table} r WHERE {where}', values).fetchone()[0]
    if selected is None:
        selected = columns
    if not isinstance(selected, list) or not selected or any(c not in columns for c in selected):
        raise ValueError('Choose valid output columns')
    sql = 'SELECT ' + ','.join(f'r.c{columns.index(c)}' for c in selected) + f' FROM {table} r WHERE {where}'
    if sort:
        if sort not in columns or direction not in ('asc', 'desc'):
            raise ValueError('Invalid sort')
        col = f'r.c{columns.index(sort)}'
        numeric = next(x['numeric'] for x in schema(db, table) if x['name'] == sort)
        expr = f'finite_number({col})' if numeric else col
        sql += f' ORDER BY ({expr} IS NULL), {expr} {direction}, r.rowid'
    else:
        sql += ' ORDER BY r.rowid'
    if not export:
        sql += ' LIMIT 100 OFFSET ?'
        values.append(max(0, page) * 100)
    return selected, total, db.execute(sql, values)


def selection_receipt(db, table, filters=None, search=''):
    """Fingerprint the indexed data actually queried, independent of table IDs/paths."""
    import hashlib
    from datetime import datetime, timezone
    from . import __version__
    columns = columns_for(db, table)
    if 'Genome_Id' not in columns:
        raise ValueError('Selection requires a genome master table with Genome_Id')
    digest = hashlib.sha256()
    for row in [columns]:
        digest.update((json.dumps(row, ensure_ascii=False) + '\n').encode())
    for row in db.execute(f'SELECT * FROM {table} ORDER BY rowid'):
        digest.update((json.dumps(row, ensure_ascii=False) + '\n').encode())
    _, total, rows = query(db, table, search=search, filters=filters, export=True, selected=['Genome_Id'])
    ids = sorted({row[0] for row in rows})
    return dict(format='genome-qc-selection', schema_version=1, version=__version__,
                created_utc=datetime.now(timezone.utc).isoformat(),
                source=db.execute('SELECT source FROM catalog WHERE id=?', (table,)).fetchone()[0],
                indexed_table_sha256=digest.hexdigest(), filters=filters or [], search=search,
                semantics='AND; inclusive numeric bounds; active numeric filters exclude missing values',
                matching_rows=total, genome_ids=ids)


def serve(output, port):
    index = Path(output) / 'reports/results.sqlite'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            p = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            try:
                with closing(sqlite3.connect(index.resolve().as_uri() + '?mode=ro', uri=True)) as db:
                    if parsed.path == '/api/artifacts':
                        content = json.dumps(artifacts(db)).encode()
                        mime = 'application/json'
                    elif parsed.path == '/artifact':
                        record = db.execute('SELECT path FROM artifacts WHERE id=?', (p.get('id', ''),)).fetchone()
                        if record is None:
                            self.send_error(404); return
                        artifact = (Path(output) / record[0]).resolve()
                        if not artifact.is_relative_to(Path(output).resolve()):
                            self.send_error(403); return
                        content = artifact.read_bytes()
                        mime = mimetypes.guess_type(artifact.name)[0] or 'application/octet-stream'
                    elif parsed.path == '/api/genomes':
                        content = json.dumps([row[0] for row in db.execute('SELECT genome_id FROM genomes ORDER BY genome_id')]).encode()
                        mime = 'application/json'
                    elif parsed.path == '/api/tables':
                        content = json.dumps([dict(id=a, source=b, rows=c) for a, b, c in
                                              db.execute('SELECT id,source,rows FROM catalog')]).encode()
                        mime = 'application/json'
                    elif parsed.path == '/api/selection':
                        content = json.dumps(selection_receipt(db, p.get('table', ''), json.loads(p.get('filters', '[]')), p.get('q', ''))).encode()
                        mime = 'application/json'
                    elif parsed.path == '/api/schema':
                        content = json.dumps(schema(db, p.get('table', ''))).encode()
                        mime = 'application/json'
                    elif parsed.path == '/api/summary':
                        from . import __version__
                        counts = {}
                        meta = db.execute("SELECT id,columns FROM catalog WHERE source='Genome_inventory.tsv'").fetchone()
                        if meta:
                            names = json.loads(meta[1])
                            if 'status' in names:
                                c = names.index('status')
                                counts = dict(db.execute(f'SELECT c{c}, count(*) FROM {meta[0]} GROUP BY c{c}'))
                        receipts = sorted((Path(output) / 'logs').glob('*/run.json'))
                        run = json.loads(receipts[-1].read_text()) if receipts else {}
                        content = json.dumps(dict(version=run.get('version', __version__),
                            status=run.get('status', 'Imported results'), started=run.get('started_utc', ''),
                            genomes=db.execute('SELECT count(*) FROM genomes').fetchone()[0],
                            counts=counts, output=str(Path(output).resolve()))).encode()
                        mime = 'application/json'
                    elif parsed.path in ('/api/rows', '/api/export'):
                        export = parsed.path.endswith('export')
                        cols, total, rows = query(db, p.get('table', ''), p.get('q', ''), int(p.get('page', 0)), export, p.get('genome', ''),
                            filters=json.loads(p.get('filters', '[]')), sort=p.get('sort', ''), direction=p.get('direction', 'asc'),
                            selected=json.loads(p['columns']) if 'columns' in p else None,
                            cohort=json.loads(p['cohort']) if 'cohort' in p else None)
                        if export:
                            buf = io.StringIO(); writer = csv.writer(buf); writer.writerow(cols); writer.writerows(rows)
                            content = buf.getvalue().encode(); mime = 'text/csv'
                        else:
                            content = json.dumps(dict(columns=cols, rows=rows.fetchall(), total=total)).encode(); mime = 'application/json'
                    elif parsed.path in ('/', '/EDA_portal.html'):
                        content = HTML.encode(); mime = 'text/html'
                    else:
                        self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                if mime == 'text/csv':
                    self.send_header('Content-Disposition', 'attachment; filename="genome_qc_subset.csv"')
                self.end_headers(); self.wfile.write(content)
            except (ValueError, TypeError, sqlite3.Error, OSError) as error:
                self.send_error(400, str(error))
    print(f'Explorer: http://127.0.0.1:{port} (Ctrl-C to stop)', flush=True)
    with ThreadingHTTPServer(('127.0.0.1', port), Handler) as server:
        server.serve_forever()
