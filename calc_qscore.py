import pandas as pd
pd.set_option('display.max_columns', None)
import math
import sys
import numpy as np
import os
import glob


def calc_qscore(compl, contam, strhet, n50, size):
	A = 1
	B = 5
	C = 1
	D = 0.5
	E = 0
	qscore = \
		(A * compl) - \
		(B * contam) + \
		(C * (contam * (strhet / 100))) + \
		(D * math.log(n50)) + \
		(E * math.log(size))
	return qscore


working_dir = sys.argv[1]

checkm_file = os.path.join(working_dir, "checkm/checkm_output.tsv")  # checkm output tsv
seqkit_file = os.path.join(working_dir, "seqkit/seqkit_stats.tsv")  # seqkit output tsv
checkm_df = pd.read_csv(checkm_file, header=0, sep='\t')
seqkit_df = pd.read_csv(seqkit_file, header=0, sep='\t')

def safe_bin_id(val):
	if pd.isna(val):
		return ""
	base = os.path.basename(str(val))
	return base.rsplit('.', 1)[0] if '.' in base else base

seqkit_df['Bin Id'] = [safe_bin_id(x) for x in seqkit_df['file']]
if len(sys.argv) > 2:
	expected = pd.DataFrame({'Bin Id': [safe_bin_id(f) for f in sys.argv[2:]]})
	seqkit_df = expected.merge(seqkit_df, on='Bin Id', how='left')
merge_list = ['Bin Id']
qscore_df = seqkit_df.merge(checkm_df, on=merge_list, how="left")
print(qscore_df.head())
qscore_df['qscore'] = qscore_df.apply(lambda row: calc_qscore(row['Completeness'],
								row['Contamination'],
								row['Strain heterogeneity'],
								row['N50'],
								row['sum_len']
								), axis=1
						)

qscore_df['Genome_Id'] = qscore_df['Bin Id'].str.rsplit('.', n=1).str[0]
qscore_df.to_csv(os.path.join(working_dir, 'qscore/qscore_all.tsv'), index=False, sep='\t')
