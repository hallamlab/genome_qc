import os
import pandas as pd
import subprocess
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from Bio.Seq import Seq
from pygenomeviz import Genbank, GenomeViz


# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)




#############################################################################################
# Magic Values
#############################################################################################

#tag_ref = 'xPG_coasm'
#fa_ref = '/home/ryan/Desktop/SI_GAPP/test_genomeviz/gbk_inputs/AB-754_D02_AB-908_coasm.fasta'
#gbk_ref = '/home/ryan/Desktop/SI_GAPP/test_genomeviz/gbk_inputs/AB-754_D02_AB-908_coasm.gbk'

tag_ref = 'xPG'
fa_ref = '/home/ryan/Desktop/SI_GAPP/GAPP_xPGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/preprocessed/AB-754_D02_AB-908.fasta'
gbk_ref = '/home/ryan/Desktop/SI_GAPP/GAPP_xPGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/genbank/AB-754_D02_AB-908.gbk'




tag_que = 'SAG'
fa_que = '/home/ryan/Desktop/SI_GAPP/GAPP_SAGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/preprocessed/AB-754_D02_AB-908.fasta'
gbk_que = '/home/ryan/Desktop/SI_GAPP/GAPP_SAGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/genbank/AB-754_D02_AB-908.gbk'

#tag_que = 'MAG'
#fa_que = '/home/ryan/Desktop/SI_GAPP/GAPP_SI060_MAGs/GAPP_output/GAPP-SI060_200m_bin_15/SI060_200m_bin_15/preprocessed/SI060_200m_bin_15.fasta'
#gbk_que = '/home/ryan/Desktop/SI_GAPP/GAPP_SI060_MAGs/GAPP_output/GAPP-SI060_200m_bin_15/SI060_200m_bin_15/genbank/SI060_200m_bin_15.gbk'

#tag_que = 'best_MAG'
#fa_que = '/home/ryan/Desktop/SI_GAPP/test_genomeviz/gbk_inputs/SI106_200m_bin_26.fasta'
#gbk_que = '/home/ryan/Desktop/SI_GAPP/test_genomeviz/gbk_inputs/SI106_200m_bin_26.gbk'

#tag_que = 'xPG'
#fa_que = '/home/ryan/Desktop/SI_GAPP/GAPP_xPGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/preprocessed/AB-754_D02_AB-908.fasta'
#gbk_que = '/home/ryan/Desktop/SI_GAPP/GAPP_xPGs/GAPP_output/GAPP-AB-754_D02_AB-908/AB-754_D02_AB-908/genbank/AB-754_D02_AB-908.gbk'





output_dir = '/home/ryan/Desktop/SI_GAPP/test_genomeviz/MUMMER/'

isExist = os.path.exists(output_dir)
if not isExist:
	os.makedirs(output_dir)

#############################################################################################

#############################################################################################
# Run Mummer to compare genomes and create a contip-to-contig map
#############################################################################################

print('Running promer...')
mum_prefix = os.path.join(output_dir, 'mumout')
promer_std = subprocess.run(['promer', '--mum', fa_ref, fa_que,
							 '--prefix=' + mum_prefix],
							 stdout=subprocess.PIPE).stdout.decode('utf-8')
print('Running delta-filter...')
df_outfile = mum_prefix + '.filter.delta'
df_stdout = subprocess.run(['delta-filter', '-m', mum_prefix + '.delta'],
						   stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(df_outfile, 'w') as df_out:
	df_out.write(df_stdout)
print('Running show-coords...')
show_outfile = mum_prefix + '.coords.tsv'
show_stdout = subprocess.run(['show-coords', '-B', '-T', '-r', '-k',
							  df_outfile],
							  stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(show_outfile, 'w') as show_out:
	show_out.write(show_stdout)

#############################################################################################

#############################################################################################
# Reorder and reorient query contigs to maximize the agreement with reference
#############################################################################################

show_df = pd.read_csv(show_outfile, sep='\t', header=0)
show_df.columns = ['qsid', 'date', 'qslen', 'aligntype', 'reffile',
				   'rsid', 'qsstart', 'qsend', 'rsstart', 'rsend',
				   'pid', 'psim', 'alignlen', 'c1', 'c2', 'c3', 'c4',
				   'qstrand', 'rslen', 'c5', 'c6'
				   ]

show_df['rs_cid'] = [int(x.rsplit('_', 1)[1]) for x in show_df['rsid']]
show_df['qs_cid'] = [int(x.rsplit('_', 1)[1]) for x in show_df['qsid']]
show_df['rstag'] = tag_ref
show_df['qstag'] = tag_que
cmap_df = show_df[['rsid', 'rs_cid', 'rstag', 'qsid', 'qs_cid', 'qstag',
				   'rslen', 'qslen', 'pid', 'psim', 'alignlen', 'qstrand',
				   'rsstart', 'rsend', 'qsstart', 'qsend'
				   ]]

# HACK to fix MP genbank format, need to fix in MP eventually...
with open(gbk_ref) as input_handle:
	dat = input_handle.readlines()
filter_list = ['REFERENCE', 'AUTHORS', 'CONSRTM', 'TITLE',
			   'JOURNAL', 'PUBMED', 'REMARK', 'COMMENT', 'XXXXX'
			   ]
gbk_ref_lines = []
for line in dat:
	res = bool([ele for ele in filter_list if(ele in line)])
	if not res:
		gbk_ref_lines.append(line)
fix_gbk_ref = os.path.join(output_dir, tag_ref + '.' + os.path.basename(gbk_ref))
with open(fix_gbk_ref, 'w') as output_handle:
	output_handle.write(''.join(gbk_ref_lines))

with open(gbk_que) as input_handle:
	dat = input_handle.readlines()
gbk_que_lines = []
for line in dat:
	res = bool([ele for ele in filter_list if(ele in line)])
	if not res:
		gbk_que_lines.append(line)
fix_gbk_que = os.path.join(output_dir, tag_que + '.' + os.path.basename(gbk_que))
with open(fix_gbk_que, 'w') as output_handle:
	output_handle.write(''.join(gbk_que_lines))

# Now load the fixed ones
with open(fix_gbk_ref) as input_handle:
	ref_recs = {rec.name: rec for rec in SeqIO.parse(input_handle, "genbank")}
with open(fix_gbk_que) as input_handle:
	que_recs = {rec.name: rec for rec in SeqIO.parse(input_handle, "genbank")}

print('Loaded', len(ref_recs), 'reference records...')
print('Loaded', len(que_recs), 'query records...')


print('Summing reference contigs...')
rslen_dict = {}
cumsum = 0
for rec in ref_recs:
	rs_cid = int(rec.rsplit('_', 1)[1])
	rec_dat = ref_recs[rec]
	rec_len = len(rec_dat.seq)
	cumsum += rec_len
	rslen_dict[rs_cid] = rec_len, cumsum

cmap_df['rsdiff'] = cmap_df['rsend'] - cmap_df['rsstart']
cmap_df['qsdiff'] = cmap_df['qsend'] - cmap_df['qsstart']

new_strd = []
new_rsstart = []
new_rsend = []
new_qsstart = []
new_qsend = []
for i,row in cmap_df.iterrows():
	if ((row.rsdiff < 0) & (row.qsdiff < 0)):
		new_strd.append('Plus')
		new_rsstart.append(row.rsend)
		new_rsend.append(row.rsstart)
		new_qsstart.append(row.qsend)
		new_qsend.append(row.qsstart)
	elif ((row.rsdiff > 0) & (row.qsdiff < 0)):
		new_strd.append(row.qstrand)
		new_rsstart.append(row.rsstart)
		new_rsend.append(row.rsend)
		new_qsstart.append(row.qsend)
		new_qsend.append(row.qsstart)
	elif ((row.rsdiff < 0) & (row.qsdiff > 0)):
		new_strd.append(row.qstrand)
		new_rsstart.append(row.rsend)
		new_rsend.append(row.rsstart)
		new_qsstart.append(row.qsstart)
		new_qsend.append(row.qsend)
	elif ((row.rsdiff > 0) & (row.qsdiff > 0)):
		new_strd.append(row.qstrand)
		new_rsstart.append(row.rsstart)
		new_rsend.append(row.rsend)
		new_qsstart.append(row.qsstart)
		new_qsend.append(row.qsend)

cmap_df['new_strand'] = new_strd
cmap_df['new_rsstart'] = new_rsstart
cmap_df['new_rsend'] = new_rsend
cmap_df['new_qsstart'] = new_qsstart
cmap_df['new_qsend'] = new_qsend

cmap_dedup_df = cmap_df.sort_values(by=['alignlen'],
									ascending=[False], inplace=False
									)
cmap_dedup_df = cmap_dedup_df.drop_duplicates('qsid')
cmap_dict = {x[0]: x[1] for x in
			 zip(cmap_dedup_df['qsid'], cmap_dedup_df['qstrand'])
			 }
cmap_dedup_df.sort_values(by=['rs_cid', 'new_rsstart'],
						  ascending=[True, True],
						  inplace=True
						  )

cmap_list = list(cmap_dedup_df['qsid'])

print('Reordering and reorienting', tag_que, '...')
rc_records = {}
qslen_dict = {}
cumsum = 0
for rec in que_recs:
	qs_cid = int(rec.rsplit('_', 1)[1])
	rec_dat = que_recs[rec]
	rec_len = len(rec_dat.seq)
	cumsum += rec_len
	qslen_dict[qs_cid] = rec_len, cumsum

	if rec in cmap_dict:
		strand = cmap_dict[rec]
		if strand == 'Minus':
			rc_rec = rec_dat.reverse_complement(id=rec_dat.name,
											name=rec_dat.name,
											description='',
											annotations=rec_dat.annotations
											)
			rc_records[rec] = rc_rec
			cmap_dict[rec] = 'Plus'
		else:
			rec_dat.id = rec_dat.name
			rc_records[rec] = rec_dat
	else:
		rec_dat.id = rec_dat.name
		rc_records[rec] = rec_dat

mapped_recs = []
for c_id in cmap_list:
	mapped_recs.append(rc_records[c_id])
	del rc_records[c_id]
unmapped_recs = list(rc_records.values())

final_recs = mapped_recs + unmapped_recs

output_gbk = os.path.join(output_dir, tag_que + '.reorder.' + os.path.basename(gbk_que))
output_fa = os.path.join(output_dir, tag_que + '.reorder.' + os.path.basename(gbk_que) + '.fasta')

SeqIO.write(final_recs, output_gbk, 'genbank')
SeqIO.write(final_recs, output_fa, 'fasta')

# Get cumsum for SAGs
qslen_dict = {}
convert_qs_cids = {}
map_id = 1
cumsum = 0
for rec in final_recs:
	qs_cid = int(rec.name.rsplit('_', 1)[1])
	rec_dat = rec
	rec_len = len(rec_dat.seq)
	cumsum += rec_len
	qslen_dict[map_id] = rec_len, cumsum
	convert_qs_cids[qs_cid] = map_id
	map_id += 1


# Get contigs regions
rs_regions = [((rslen_dict[r][1] - rslen_dict[r][0]) + 1, rslen_dict[r][1]) for r in rslen_dict]
qs_regions = [((qslen_dict[r][1] - qslen_dict[r][0]) + 1, qslen_dict[r][1]) for r in qslen_dict]

# Setting final strand direction, and calcing cumsum of start/ends
cmap_df['final_strand'] = [cmap_dict[x] for x in cmap_df['qsid']]
cu_rsstart = []
cu_rsend = []
cu_qsstart = []
cu_qsend = []
for i,row in cmap_df.iterrows():
	rskey = int(row.rs_cid)
	if rskey > 1:
		rs_cum_val = rslen_dict[rskey - 1][1]
	else:
		rs_cum_val = 0
	qskey = int(convert_qs_cids[row.qs_cid])
	if qskey > 1:
		qs_cum_val = qslen_dict[qskey - 1][1]
	else:
		qs_cum_val = 0
	cu_rss = row.new_rsstart + rs_cum_val
	cu_rse = row.new_rsend + rs_cum_val
	cu_qss = row.new_qsstart + qs_cum_val
	cu_qse = row.new_qsend + qs_cum_val
	cu_rsstart.append(cu_rss)
	cu_rsend.append(cu_rse)
	cu_qsstart.append(cu_qss)
	cu_qsend.append(cu_qse)

cmap_df['cu_rsstart'] = cu_rsstart
cmap_df['cu_rsend'] = cu_rsend
cmap_df['cu_qsstart'] = cu_qsstart
cmap_df['cu_qsend'] = cu_qsend

cmap_df.sort_values(by=['rs_cid', 'cu_rsstart'], ascending=[True, True],
					inplace=True)
cmap_df.to_csv(os.path.join(output_dir, tag_ref + '_' + tag_que + '_' + 'cmap.tsv'), sep='\t', index=False)

#############################################################################################

#############################################################################################
# Viz it
#############################################################################################

gbk_files = [output_gbk, fix_gbk_ref]
gv = GenomeViz(
    fig_width=12,
    fig_track_height=0.7,
    feature_track_ratio=0.3,
    tick_track_ratio=0.3,
    tick_style="bar",
    #tick_style="axis",
    tick_labelsize=10,
    align_type="center"
)

gbk_dict = {Genbank(gbk_file).name: Genbank(gbk_file) for gbk_file in gbk_files}
gbk_order = [os.path.basename(x).rsplit('.', 1)[0] for x in gbk_files]
for i, gbk_name in enumerate(gbk_order):
    gbk = gbk_dict[gbk_name]
    track = gv.add_feature_track(gbk_name, gbk.range_size, labelsize=15)
    #track.add_genbank_features(gbk,
                               #label_type="product", # if idx == 0 else None,  # Labeling only top track
                               #label_handle_func=lambda s: "" if s.startswith("hypothetical") else s,
                               # If label startswith `hypothetical`, facecolor="grey"; if not facecolor="orange"
                               #facecolor_handle_func=lambda f: "royalblue" if 'nitr' in f.qualifiers.get("product", [""])[0].lower() else "grey",
                               #labelsize=12,
                               #labelvpos="top",
                               #plotstyle="rbox",
     #                          facecolor="orange"
     #                          )

normal_color, inverted_color, alpha = "royalblue", "darkorange", 0.75
for i,link in cmap_df.iterrows():
    link_data1 = (gbk_order[0], link.cu_qsstart, link.cu_qsend)
    link_data2 = (gbk_order[1], link.cu_rsstart, link.cu_rsend)
    gv.add_link(link_data1, link_data2, normal_color, inverted_color, alpha,
                v=link.pid, curve=True)


'''
# Add subtracks to top track for plotting 'GC content' & 'GC skew'
gv.top_track.add_subtrack(ratio=0.7, name="gc_content")
gv.top_track.add_subtrack(ratio=0.7, name="gc_skew")
'''

fig = gv.plotfig(dpi=600)

# Add box annotation to query track
c = 1
for reg in qs_regions:
	target_track = gv.get_track(gbk_order[0])
	box_xmin, box_xmax = reg
	box_xmin += target_track.offset  # Offset is required if align_type is not 'left'
	box_xmax += target_track.offset
	x, y = (box_xmin, box_xmin, box_xmax, box_xmax), (1, -1, -1, 1)
	if (c % 2) != 0:
		fillcol = 'black'
	else:
		fillcol = 'orange'
	target_track.ax.fill(x, y, fc=fillcol, linewidth=0, alpha=0.5, zorder=-10)
	c += 1

# Add box annotation to ref track
c = 1
for reg in rs_regions:
	target_track = gv.get_track(gbk_order[1])
	box_xmin, box_xmax = reg
	box_xmin += target_track.offset  # Offset is required if align_type is not 'left'
	box_xmax += target_track.offset
	x, y = (box_xmin, box_xmin, box_xmax, box_xmax), (1, -1, -1, 1)
	if (c % 2) != 0:
		fillcol = 'black'
	else:
		fillcol = 'orange'
	target_track.ax.fill(x, y, fc=fillcol, linewidth=0, alpha=0.5, zorder=-10)
	c += 1

'''
# Plot GC content for top track
pos_list, gc_content_list = gbk_dict[gbk_order[0]].calc_gc_content()
pos_list += gv.top_track.offset  # Offset is required if align_type is not 'left'
gc_content_ax = gv.top_track.subtracks[0].ax
gc_content_ax.set_ylim(bottom=0, top=max(gc_content_list))
gc_content_ax.fill_between(pos_list, gc_content_list, alpha=0.2, color="blue")
gc_content_ax.text(gv.top_track.offset, max(gc_content_list) / 2, "GC(%) ", ha="right", va="center", color="blue")

# Plot GC skew for top track
pos_list, gc_skew_list = gbk_dict[gbk_order[0]].calc_gc_skew()
pos_list += gv.top_track.offset  # Offset is required if align_type is not 'left'
gc_skew_abs_max = max(abs(gc_skew_list))
gc_skew_ax = gv.top_track.subtracks[1].ax
gc_skew_ax.set_ylim(bottom=-gc_skew_abs_max, top=gc_skew_abs_max)
gc_skew_ax.fill_between(pos_list, gc_skew_list, alpha=0.2, color="red")
gc_skew_ax.text(gv.top_track.offset, 0, "GC skew ", ha="right", va="center", color="red")
'''
fig.savefig(os.path.join(output_dir, tag_ref + '_' + tag_que + '.png'))
fig.savefig(os.path.join(output_dir, tag_ref + '_' + tag_que + '.svg'))
fig.savefig(os.path.join(output_dir, tag_ref + '_' + tag_que + '.pdf'))

#############################################################################################

#############################################################################################
#
#############################################################################################
flurp
with open(fix_gbk_ref) as input_handle:
	ref_recs = {rec.name: rec for rec in SeqIO.parse(input_handle, "genbank")}
with open(output_gbk) as input_handle:
	que_recs = {rec.name: rec for rec in SeqIO.parse(input_handle, "genbank")}

ref_cid_list = [1, 2, 3, 4]
sub_cmap_df = cmap_df.query('rs_cid in @ref_cid_list')
rsid_keep_list = list(set(sub_cmap_df['rsid']))
qsid_keep_list = list(set(sub_cmap_df['qsid']))

sub_ref_list = []
for rec in ref_recs:
	rec_dat = ref_recs[rec]
	r_name = rec_dat.name
	if r_name in rsid_keep_list:
		sub_ref_list.append(rec_dat)

sub_que_list = []
for rec in que_recs:
	rec_dat = que_recs[rec]
	r_name = rec_dat.name
	if r_name in qsid_keep_list:
		sub_que_list.append(rec_dat)

print('Subset', len(sub_ref_list), 'reference records...')
print('Subset', len(sub_que_list), 'query records...')

output_gbk = os.path.join(output_dir, tag_ref + '.subset.' + os.path.basename(gbk_ref))
SeqIO.write(sub_ref_list, output_gbk, 'genbank')

output_gbk = os.path.join(output_dir, tag_que + '.subset.' + os.path.basename(gbk_que))
SeqIO.write(sub_que_list, output_gbk, 'genbank')


















'''
#############################################################################################

#############################################################################################
# Run Mummer to compare genomes and create a contip-to-contig map
#############################################################################################

print('Running promer...')
mum_prefix = os.path.join(output_dir, 'mumout2')
promer_std = subprocess.run(['promer', '--mum', fa_ref, output_fa,
							 '--prefix=' + mum_prefix],
							 stdout=subprocess.PIPE).stdout.decode('utf-8')
print('Running delta-filter...')
df_outfile = mum_prefix + '.filter2.delta'
df_stdout = subprocess.run(['delta-filter', '-m', mum_prefix + '.delta'],
						   stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(df_outfile, 'w') as df_out:
	df_out.write(df_stdout)
print('Running show-coords...')
show_outfile = mum_prefix + '.coords2.tsv'
show_stdout = subprocess.run(['show-coords', '-B', '-T', '-r', '-k',
							  df_outfile],
							  stdout=subprocess.PIPE).stdout.decode('utf-8')
with open(show_outfile, 'w') as show_out:
	show_out.write(show_stdout)

show_df = pd.read_csv(show_outfile, sep='\t', header=0)
show_df.columns = ['qsid', 'date', 'qslen', 'aligntype', 'reffile',
				   'rsid', 'qsstart', 'qsend', 'rsstart', 'rsend',
				   'pid', 'psim', 'alignlen', 'c1', 'c2', 'c3', 'c4',
				   'qstrand', 'rslen', 'c5', 'c6'
				   ]

show_df['rs_cid'] = [int(x.rsplit('_', 1)[1]) for x in show_df['rsid']]
show_df['qs_cid'] = [int(x.rsplit('_', 1)[1]) for x in show_df['qsid']]

cmap_df = show_df[['rsid', 'rs_cid', 'qsid', 'qs_cid', 'rslen', 'qslen',
				   'pid', 'psim', 'alignlen', 'qstrand',
				   'rsstart', 'rsend', 'qsstart', 'qsend'
				   ]]
print(cmap_df.head(10))
cmap_df.to_csv(os.path.join(output_dir, 'cmap2.tsv'), sep='\t', index=False)

#############################################################################################
'''