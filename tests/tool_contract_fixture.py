#!/usr/bin/env python
# Synthetic external-tool contracts only, not biological inference.
from pathlib import Path
import sys,csv
name=Path(sys.argv[0]).name;a=sys.argv[1:]
def opt(x): return a[a.index(x)+1]
def table(path,cols,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(cols);w.writerows(rows)
# Explicit failure injection for dependency/resume regression tests.
if (Path(__file__).parent / ('fail-' + name)).exists():
 print('Injected assessment failure: ' + name, file=sys.stderr)
 raise SystemExit(7)
if name=='checkm':
 cols=['Bin Id','Marker lineage','# genomes','# markers','# marker sets','0','1','2','3','4','5+','Completeness','Contamination','Strain heterogeneity']
 table(opt('-f'),cols,[[p.stem,'root',1,100,10,0,100,0,0,0,0,20 if p.stem.startswith('g2.') else 90+i,25 if p.stem.startswith('g1.') else 1,0] for i,p in enumerate(sorted(Path(a[1]).glob('*.fasta')))])
elif name=='gtdbtk':
 assert int(opt('--pplacer_cpus')) == min(8, int(opt('--cpus'))), 'Placement must use up to eight allocated CPUs'
 scratch=Path(opt('--scratch_dir')); assert scratch.is_dir()
 (scratch/'placement.cache').write_text('temporary placement data')
 rows=[[p.stem, 'Unclassified Bacteria' if p.stem.startswith('g0.') else 'd__Bacteria;p__P;c__C;o__O;f__F;g__G;s__S'] for p in sorted(Path(opt('--genome_dir')).glob('*.fasta'))]
 table(Path(opt('--out_dir'))/'classify/gtdbtk.bac120.summary.tsv',['user_genome','classification'],rows)
elif name=='barrnap':
 Path(opt('--outseq')).write_text('')
 print('##gff-version 3')
elif name=='tRNAscan-SE':
 p=Path(opt('-j'));p.write_text('##gff-version 3\n' if p.name.startswith('g0.') else 'contig\tfixture\ttRNA\t1\t70\t100\t+\t.\tID=1;isotype=Ala\n')
elif name=='gunc':
 table(Path(opt('-o'))/'GUNC.progenomes_2.1.maxCSS_level.tsv',['genome','clade_separation_score','reference_representation_score','taxonomic_level'],[[p.stem,0.1,0.9,'species'] for p in Path(opt('-d')).glob('*.fasta')])
else: raise SystemExit('Unexpected tool: '+name)
