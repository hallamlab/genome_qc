import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from genome_qc.report import build, query, schema, selection_receipt


class ExplorerFilters(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root/'inputs.resolved.json').write_text(json.dumps([
            dict(id=x, fasta=x+'.fa', sha256=x) for x in ['a','b','c','d']]))
        (self.root/'quality.tsv').write_text('Genome_Id\tPhylum\tCompleteness\tContamination\n'
            'a\tp__A\t95\t1\nb\tp__A\t80\t2\nc\tp__B\t100\t0\nd\tp__A\tNA\tnan\n')
        (self.root/'markers.tsv').write_text('genome\tmarker\na\tM1\na\tM2\nb\tM3\nc\tM4\n')
        (self.root/'unlinked.tsv').write_text('total\n4\n')
        build(self.root)
        self.db = sqlite3.connect(self.root/'reports/results.sqlite')
        self.tables = {source: t for t,source in self.db.execute('SELECT id,source FROM catalog')}

    def tearDown(self):
        self.db.close();self.tmp.cleanup()

    def test_numeric_taxonomy_missing_and_sort(self):
        t=self.tables['quality.tsv']
        cols,n,rows=query(self.db,t,filters=[dict(column='Phylum',op='eq',value='p__A'),
            dict(column='Completeness',op='range',min=90,max=100)],selected=['Genome_Id','Completeness'])
        self.assertEqual(n,1);self.assertEqual(rows.fetchall(),[('a','95')])
        _,_,rows=query(self.db,t,sort='Completeness',direction='desc')
        self.assertEqual([r[0] for r in rows],['c','a','b','d'])
        _,n,_=query(self.db,t,filters=[dict(column='Completeness',op='range',max=90)])
        self.assertEqual(n,1)  # NA is never CAST to a misleading zero.
        _,n,_=query(self.db,t,filters=[dict(column='Completeness',op='missing')])
        self.assertEqual(n,1)
        fields={f['name']:f for f in schema(self.db,t)}
        self.assertTrue(fields['Completeness']['numeric'])
        self.assertEqual(fields['Phylum']['choices'],['p__A','p__B'])

    def test_cross_table_genome_subset(self):
        cohort=dict(table=self.tables['quality.tsv'],filters=[dict(column='Completeness',op='range',min=90)])
        _,n,rows=query(self.db,self.tables['markers.tsv'],cohort=cohort,export=True)
        self.assertEqual(n,3);self.assertEqual([r[0] for r in rows],['a','a','c'])
        _,n,_=query(self.db,self.tables['unlinked.tsv'],cohort=cohort)
        self.assertEqual(n,0)

    def test_reject_invalid_filters_and_identifiers(self):
        t=self.tables['quality.tsv']
        for options in [dict(sort='x;DROP TABLE genomes'),dict(filters=[dict(column='bad',op='eq')]),
                        dict(filters=[dict(column='Completeness',op='range',min='nan')]),
                        dict(filters=[dict(column='Completeness',op='range',min=99,max=1)]),
                        dict(cohort=dict(table='t0;DROP TABLE genomes')),dict(selected=[])]:
            with self.subTest(options=options), self.assertRaises(ValueError):query(self.db,t,**options)
        _,n,_=query(self.db,t,filters=[dict(column='Phylum',op='eq',value="' OR 1=1 --")])
        self.assertEqual(n,0)

    def test_export_has_all_matching_rows_not_page(self):
        t=self.tables['markers.tsv']
        self.db.executemany(f'INSERT INTO {t} VALUES (?,?)',[('a',str(i)) for i in range(120)])
        _,n,page=query(self.db,t)
        self.assertEqual(n,124);self.assertEqual(len(page.fetchall()),100)
        _,_,rows=query(self.db,t,export=True,selected=['marker'])
        self.assertEqual(len(rows.fetchall()),124)

    def test_selection_receipt_replays_and_fingerprints_source(self):
        t=self.tables['quality.tsv']
        filters=[dict(column='Completeness',op='range',min=90)]
        receipt=selection_receipt(self.db,t,filters)
        self.assertEqual(receipt['genome_ids'],['a','c'])
        self.assertEqual(receipt['matching_rows'],2)
        replay=selection_receipt(self.db,t,receipt['filters'],receipt['search'])
        self.assertEqual(replay['indexed_table_sha256'],receipt['indexed_table_sha256'])
        self.assertEqual(replay['genome_ids'],receipt['genome_ids'])
        self.assertEqual(selection_receipt(self.db,t)['genome_ids'],['a','b','c','d'])
        self.db.execute(f"UPDATE {t} SET c2='98' WHERE c0='a'")
        self.assertNotEqual(selection_receipt(self.db,t)['indexed_table_sha256'],receipt['indexed_table_sha256'])
