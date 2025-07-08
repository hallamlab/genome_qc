import pandas as pd
pd.set_option('display.max_columns', None)
import os
import glob
import numpy as np
import sys



trna_dir = sys.argv[1]

trna_gffs = [x for x in glob.glob(os.path.join(trna_dir, "*.gff"))]

gff_list = []
for gff in trna_gffs:
	
	empty = os.stat(gff).st_size == 0
	if not empty:
		gff_df = pd.read_csv(gff, sep='\t', header=None, comment='#')
		gff_df.columns = ['seqid', 'source', 'type', 'start', 'end',
						  'score', 'strand', 'phase', 'attributes'
						  ]
		trna_df = gff_df.query("type == 'tRNA'")
		trna_df['genome_id'] = os.path.basename(gff).rsplit('.', 1)[0]
		trna_df['isotype'] = [x.split('isotype=')[1].split(';')[0] for x in trna_df['attributes']]
		gff_list.append(trna_df)

gff_cat_df = pd.concat(gff_list)
gff_cat_df.to_csv(os.path.join(trna_dir, 'trnascan_gff_table.tsv'), sep='\t', index=False)

gff_cnt_df = gff_cat_df.groupby(['genome_id'])['isotype'].size().reset_index()
gff_cnt_df.columns = ['genome_id', 'trna_total']
gff_uniq_df = gff_cat_df[['genome_id', 'isotype']].drop_duplicates(
						 ).groupby(['genome_id'])['isotype'].size().reset_index()
gff_uniq_df.columns = ['genome_id', 'trna_unique']

gff_merge_df = gff_cnt_df.merge(gff_uniq_df, on='genome_id')
gff_merge_df.to_csv(os.path.join(trna_dir, 'trnascan_trna_counts.tsv'), sep='\t', index=False)
