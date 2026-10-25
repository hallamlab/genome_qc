import os
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from genome_quality_atlas import apply_figure_typography, plot_font_rc, maybe_attach_gunc_summary


def apply_plot_style():
    plt.rcParams.update(plot_font_rc())
    sns.set_theme(style="whitegrid", context="notebook", rc=plot_font_rc())


def save_current_plot(pdf_path, png_path):
    apply_plot_style()
    apply_figure_typography(plt.gcf())
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    plt.savefig(png_path, bbox_inches='tight', format='png', dpi=300)
    plt.close()


def create_Contamination_Completeness_plot(mag_data, output_path):
    apply_plot_style()

    group_data = mag_data[['Genome_Id', 'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species', 'Completeness', 'Contamination', 'qscore', '16S_rRNA']]

    group_data['quality'] = 'low'
    group_data['quality'] = group_data.apply(lambda row: 'medium' if row['Completeness'] >= 50 and row['Contamination'] <= 10 else row['quality'], axis=1)
    group_data['quality'] = group_data.apply(lambda row: 'high' if row['Completeness'] >= 90 and row['Contamination'] <= 5 else row['quality'], axis=1)

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=group_data,
        x='Completeness',
        y='Contamination',
        hue='quality',
        hue_order=['low', 'medium', 'high'],
        palette={'high': 'black', 'medium': 'grey', 'low': 'white'},
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=50, 
        ymin=-5, 
        ymax=10, 
        colors='grey', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=10, 
        xmin=50, 
        xmax=105, 
        colors='grey', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(-0.5, 101)
    plt.ylim(-0.5, 17)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by Quality')

    # Adjust the legend
    leg = plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legend_handles:
        handle.set_color('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_current_plot(output_path + '_All.pdf', output_path + '_All.png')


    # Filter data based on specified criteria
    filtered_data = group_data.loc[group_data['quality'].isin(['high', 'medium'])]
    filtered_data['Has_16S'] = ['True' if x != 0 else 'False' for x in filtered_data['16S_rRNA']]

    '''
    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='Completeness',
        y='Contamination',
        size='log(RPM)',
        hue='quality',
        hue_order=['medium', 'high'],
        palette={'high': 'black', 'medium': 'grey'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legend_handles:
        handle.set_color('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_current_plot(output_path + '_MQHQ_RPM.pdf', output_path + '_MQHQ_RPM.png')
    '''

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='Completeness',
        y='Contamination',
        #size='log(RPM)',
        hue='Has_16S',
        hue_order=['False', 'True'],
        palette={'True': 'black', 'False': 'white'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination')

    # Adjust the legend
    leg = plt.legend(title='16S Present', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legend_handles:
        handle.set_color('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_current_plot(output_path + '_MQHQ_16S.pdf', output_path + '_MQHQ_16S.png')

    '''
    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    print(filtered_data.head())
    sns.scatterplot(
        data=filtered_data,
        x='Completeness',
        y='Contamination',
        size='log(RPM)',
        hue='ASV_PAIR',
        hue_order=['False', 'True'],
        palette={'True': 'black', 'False': 'white'},
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by DataType')

    # Adjust the legend
    leg = plt.legend(title='DataType', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legend_handles:
        handle.set_color('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_current_plot(output_path + '_MQHQ_ASV_PAIR.pdf', output_path + '_MQHQ_ASV_PAIR.png')
    '''

    # Create a scatterplot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=filtered_data,
        x='Completeness',
        y='Contamination',
        #size='log(RPM)',
        hue='Phylum',
        sizes=(5, 400),
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add the dashed red lines for the top and left borders of the region
    # Left side line (Vertical line from (90, 0) to (90, 5))
    plt.vlines(
        x=90, 
        ymin=-5, 
        ymax=5, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Top side line (Horizontal line from (90, 5) to (100, 5))
    plt.hlines(
        y=5, 
        xmin=90, 
        xmax=105, 
        colors='black', 
        linestyles='dashed', 
        linewidth=1
    )

    # Set the x and y axis limits
    plt.xlim(49, 101)
    plt.ylim(-0.5, 10.5)

    # Adjust plot aesthetics
    plt.xlabel('Completeness (%)')
    plt.ylabel('Contamination (%)')
    plt.title('Scatter Plot of Completeness vs Contamination by Phylum')

    # Adjust the legend
    leg = plt.legend(title='Phylum', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon=False)

    # Manually set edge colors for the legend markers
    for handle in leg.legend_handles:
        handle.set_color('black')
        handle.set_linewidth(0.5)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save the plot
    save_current_plot(output_path + '_MQHQ_Phylum.pdf', output_path + '_MQHQ_Phylum.png')







###################################################################################################################################################


parent_dir = sys.argv[1]
file_1 = 'barrnap_subunit_counts.tsv'
file_2 = 'dedupe_assessed.tsv'
file_3 = 'trnascan_trna_counts.tsv'
out_final = 'Master_genome_QC.tsv'
master = pd.DataFrame()

# loop over all directories in the parent directory
out_path = os.path.join(parent_dir, out_final)

#check if the directory is a folder
first_file_path = os.path.join(parent_dir, "barrnap", file_1)
second_file_path = os.path.join(parent_dir, "dedupe", file_2)
third_file_path = os.path.join(parent_dir, "trnascan", file_3)

#read the file into a pandas dataframe
bar_df = pd.read_csv(first_file_path,sep='\t')

#create a new dataframe with the ID as index
bar_piv_df = bar_df.pivot(index='genome_id', columns='name', values='subunit_count').reset_index().fillna(0)

#Read the dedupe_dataframe 
dedupe_df = pd.read_csv(second_file_path,sep='\t')

#Merge the two dataframes
merged_df = pd.merge(dedupe_df, bar_piv_df, left_on='Bin Id', right_on='genome_id', how='left').drop('genome_id', axis=1)
# Only absent RNA detections are zero; unknown taxonomy is not a numeric zero.
for column in ('5S_rRNA', '16S_rRNA', '23S_rRNA'):
    if column not in merged_df:
        merged_df[column] = 0
    merged_df[column] = merged_df[column].fillna(0)
#read the file into a pandas dataframe
trna_df = pd.read_csv(third_file_path,sep='\t')
merge2_df = pd.merge(merged_df, trna_df, left_on='Bin Id', right_on='genome_id', how='left').drop('genome_id', axis=1)

#Create the final master table
master = pd.concat([master, merge2_df])
ranks = master['classification'].fillna('').astype(str).str.split(';', expand=True).reindex(columns=range(7)).fillna('')
master[['Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']] = ranks
for column in ('trna_total', 'trna_unique'):
    master[column] = master[column].fillna(0)

# remove prefix from each column
master['Domain'] = master['Domain'].str.replace('d__', '')
master['Phylum'] = master['Phylum'].str.replace('p__', '')
master['Class'] = master['Class'].str.replace('c__', '')
master['Order'] = master['Order'].str.replace('o__', '')
master['Family'] = master['Family'].str.replace('f__', '')
master['Genus'] = master['Genus'].str.replace('g__', '')
master['Species'] = master['Species'].str.replace('s__', '')

final = master.drop('classification', axis=1)

if '16S_rRNA' in final.columns:
    final['contains_16S'] = [False if x == 0 else True for x in final['16S_rRNA']]
else:
    final['contains_16S'] = [False for x in final.index]

print(final.head())
print(final.columns)

# Quality selection is recorded only after every assessment has completed.
final = maybe_attach_gunc_summary(final, master_path=out_path)
if os.path.isdir(os.path.join(parent_dir, 'assessments')):
    from assessment_summary import attach
    final = attach(final, parent_dir)
else:
    pd.DataFrame(columns=['Genome_Id','variant','stage','status','exit_code','reason','log','retry_token']).to_csv(os.path.join(parent_dir, 'Assessment_status.tsv'), sep='\t', index=False)
bounds = [None if x.lower() in ('null', 'none', '') else float(x) for x in sys.argv[2:7]]
minimum, maximum, comp, cont, qmin = bounds if len(bounds) == 5 else [None] * 5
require_rrna = len(sys.argv) > 7 and sys.argv[7].lower() == 'true'
def bound(column, value, lower=True):
    if value is None:
        return pd.Series(True, index=final.index)
    numbers = pd.to_numeric(final[column], errors='coerce')
    return numbers.ge(value) if lower else numbers.le(value)
checks = {
    'pass_genome_size': bound('sum_len', minimum) & bound('sum_len', maximum, False),
    'pass_completeness': bound('Completeness', comp),
    'pass_contamination': bound('Contamination', cont, False),
    'pass_qscore': bound('qscore', qmin),
    'pass_rRNA_check': final['pass_BARRNAP'].astype(str).str.strip().str.lower().isin(['true', '1', '1.0']) if require_rrna else pd.Series(True, index=final.index),
}
for name, values in checks.items():
    final[name] = values.fillna(False)
final['selected'] = final[list(checks)].all(axis=1)
final['selection_reasons'] = final.apply(lambda row: ';'.join(name.removeprefix('pass_') for name in checks if not row[name]) or 'passed', axis=1)
final.loc[final['selected']].to_csv(os.path.join(parent_dir, 'Selected_genome_QC.tsv'), sep='\t', index=False)

#Save the master table
final.to_csv(out_path, sep='\t', index=False)

#Plot summaries
create_Contamination_Completeness_plot(final, parent_dir + '/Master_genome_QC')
