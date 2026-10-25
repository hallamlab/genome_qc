import pandas as pd
pd.set_option('display.max_columns', None)
import math
import sys
import numpy as np
import os
import glob


working_dir = sys.argv[1]
qscore_file = os.path.join(working_dir, "qscore/qscore_all.tsv")  # MQHP genomes list
barrnap_file = os.path.join(working_dir, "barrnap/barrnap_blast_pass.tsv")  # barrnap blast results
gtdbtk_files = glob.glob(os.path.join(working_dir, "gtdbtk/classify/gtdbtk.*.summary.tsv"))  # bac and arc gtdbtk results
qscore_df = pd.read_csv(qscore_file, header=0, sep='\t')
barrnap_df = pd.read_csv(barrnap_file, header=0, sep='\t')
cat_list = []
for g in gtdbtk_files:
	g_df = pd.read_csv(g, header=0, sep='\t')
	g_df['Bin Id'] = [x for x in g_df['user_genome']]
	cat_list.append(g_df)
if len(cat_list) > 1:
	gtdbtk_df = pd.concat(cat_list)
else:
	gtdbtk_df = cat_list[0] if cat_list else pd.DataFrame(columns=['Bin Id', 'classification'])
merge_list = ['Bin Id']
qscore_df = qscore_df.merge(barrnap_df, on=merge_list, how="left")
qscore_df = qscore_df.merge(gtdbtk_df, on=merge_list, how="left")
print(qscore_df.head())

qscore_df.to_csv(os.path.join(working_dir, "dedupe/dedupe_all.tsv"), index=False, sep='\t')

sorted_df = qscore_df.sort_values(by=['qscore', 'N50', 'sum_len'],
				ascending=[False, False, False]
				)
keep_cols = ['Genome_Id', 'Bin Id', 'classification', 'Completeness', 'Contamination', 'Strain heterogeneity',
		'num_seqs', 'sum_len', 'min_len', 'avg_len', 'max_len', 'N50', 'qscore', 'pass_BARRNAP']
# Select a representative variant without applying quality acceptance criteria.
# Every input genome retains its highest-ranked assessed variant.
assessed = sorted_df[keep_cols].drop_duplicates('Genome_Id')
assessed.to_csv(os.path.join(working_dir, 'dedupe/dedupe_assessed.tsv'), index=False, sep='\t')
