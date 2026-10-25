#!/usr/bin/env python3
"""Deterministic public workflow SVG/PDFs using the shared MP nodal grammar."""
from pathlib import Path
from html import escape
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/user-guide/assets'
BRIEF = [
 ('Preprocessing', 'Genome FASTAs', [('Prepare inputs', 'Genome QC'), ('Assembly statistics', 'SeqKit')], 'Contig variants'),
 ('Quality & taxonomy', 'Contig variants', [('Completeness / quality', 'CheckM, Qscore'), ('Taxonomic assignment', 'GTDB-Tk')], 'Scored genomes'),
 ('rRNA consistency', 'Classified variants', [('Predict rRNAs', 'barrnap'), ('Compare rRNAs', 'BLAST+, Genome QC')], 'rRNA evidence'),
 ('Genome assessment', 'Assessed variants', [('Choose representative', 'Qscore, N50, length'), ('tRNAs / chimerism', 'tRNAscan-SE, GUNC')], 'Combined evidence'),
 ('Final integration', 'All representatives', [('Join measurements', 'Genome QC'), ('Apply selection', 'User criteria')], 'Master + selected'),
 ('Atlas & explorer', 'Master QC table', [('Build genome atlas', 'Python plotting'), ('Explore / select / export', 'Genome QC Explorer')], 'Figures + subsets'),
]
DETAILED = [
 ('Preprocessing', 'Genome FASTAs', [('Stage / contig variants', 'Genome QC, SeqKit'), ('Assembly statistics', 'SeqKit')], 'Nonempty variants'),
 ('Quality assessment', 'Nonempty variants', [('Marker assessment', 'CheckM lineage_wf'), ('Calculate Qscore', 'Genome QC')], 'Quality metrics'),
 ('Taxonomic context', 'Assessed variants', [('Classify genomes', 'GTDB-Tk, R226'), ('Extract lineage / domain', 'Genome QC')], 'Taxonomy table'),
 ('rRNA evidence', 'Variants + domain', [('Predict rRNA genes', 'barrnap'), ('Self-BLAST / consistency', 'BLAST+, Genome QC')], 'RNA consistency'),
 ('Representative', 'Variant evidence', [('Choose variant', 'Qscore, N50, length'), ('Predict / assess', 'parallel')], 'tRNA + GUNC'),
 ('Final selection', 'All representatives', [('Join metrics + status', 'Genome QC'), ('Apply requested bounds', 'Default: unrestricted')], 'Master + selected'),
 ('Genome atlas', 'Complete master', [('Annotate quality metrics', 'pandas, NumPy'), ('Summarize / plot', 'Matplotlib, seaborn')], 'Atlas tables + plots'),
 ('Explorer', 'Tables + atlas', [('Index linked results', 'Genome QC'), ('Filter / document subsets', 'Genome QC Explorer')], 'CSV + JSON record'),
]

def figure(rows, detailed=False):
    w=1240; gap=165; top=210
    # Leave space for the explicitly parallel RNA/chimerism branch.
    ys=[]; y=top
    for i in range(len(rows)):
        ys.append(y); y += 215 if detailed and i==4 else gap
    h=ys[-1]+100
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" data-workflow-style="mp-nodal-v1" role="img" aria-labelledby="title desc">',
           '<title id="title">Genome QC '+('detailed workflow' if detailed else 'workflow overview')+'</title>',
           '<desc id="desc">Genome FASTAs to quality, taxonomy, RNA and GUNC measurements, final selection, genome atlas and explorer. All assessed genomes remain in the master table.</desc>',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M 0 0 L 10 5 L 0 10 Z" fill="#111111"/></marker></defs>',
           '<rect width="100%" height="100%" fill="#FFFFFF"/>',
           '<g font-family="Times New Roman, Times, serif" fill="#111111">']
    def text(x,y,label,size=22,anchor='middle'):
        parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(label)}</text>')
    def edge(points,arrow=True):
        p=' '.join(f'{x},{y}' for x,y in points)
        parts.append(f'<polyline points="{p}" fill="none" stroke="#111111" stroke-width="1.5"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    def circle(x,y,kind='mid',radius=18):
        color,stroke={'input':('#DAE8FC','#6C8EBF'),'mid':('#F5F5F5','#666666'),'output':('#D5E8D4','#82B366')}[kind]
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="3"/>')
    def diamond(x,y,r=20):
        parts.append(f'<path d="M {x} {y-r} L {x+r} {y} L {x} {y+r} L {x-r} {y} Z" fill="#F5F5F5" stroke="#666666" stroke-width="3"/>')
    parts.append(f'<rect x="25" y="20" width="{w-50}" height="50" fill="#CCCCCC" stroke="#666666" stroke-width="1.5"/>')
    text(w/2,54,'Genome QC · '+('assessment to exploration' if detailed else 'genomes to atlas'),28)
    text(50,110,'Inputs: genome FASTAs + reference databases',22,'start')
    for x,kind,label in [(620,'input','Input'),(780,'compute','Compute'),(945,'mid','Data'),(1080,'output','Output')]:
        if kind=='compute': diamond(x,104,12)
        else: circle(x,104,kind,11)
        text(x+22,111,label,20,'start')
    # Connected module squares group operations, not a serial execution promise.
    for a,b in zip(ys,ys[1:]): edge([(240,a+19),(240,b-20)])
    for i,((module,source,steps,result),y) in enumerate(zip(rows,ys),1):
        parts.append(f'<rect x="222" y="{y-18}" width="36" height="36" fill="#F5F5F5" stroke="#111111" stroke-width="3"/>')
        text(240,y+8,str(i),24)
        # Split long module labels at a word boundary, maintaining a left-aligned spine.
        words=module.split(); lines=[module]
        if len(module)>14:
            split=max(1,len(words)//2); lines=[' '.join(words[:split]),' '.join(words[split:])]
        for j,line in enumerate(lines): text(203,y+7+(j-(len(lines)-1)/2)*28,line,26,'end')
        circle(370,y,'input' if i==1 else 'mid')
        text(370,y-34,source,21)
        edge([(389,y),(570,y)])
        diamond(592,y); text(592,y-35,steps[0][0],21); text(592,y+51,steps[0][1],19)
        if detailed and i==5:
            # Parallel assessments of the same representative, not a tRNA-to-GUNC dependency.
            edge([(614,y),(700,y),(700,y-43),(805,y-43)])
            edge([(700,y),(700,y+70),(805,y+70)])
            for cy,label,tool in [(y-43,'Predict tRNAs','tRNAscan-SE'),(y+70,'Assess chimerism','GUNC')]:
                diamond(827,cy); text(827,cy-32,label,21); text(827,cy+45,tool,19)
                edge([(849,cy),(970,cy),(970,y)],False)
            circle(700,y,'mid',3); circle(970,y,'mid',3)
            edge([(974,y),(1090,y)])
        else:
            edge([(614,y),(805,y)]); diamond(827,y)
            text(827,y-35,steps[1][0],21); text(827,y+51,steps[1][1],19)
            edge([(849,y),(1090,y)])
        circle(1110,y,'output'); text(1110,y-34,result,21)
    parts+=['</g></svg>']
    return '\n'.join(parts)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--svg-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name,rows,detail in [('workflow-brief',BRIEF,False),('workflow-detailed',DETAILED,True)]:
        svg=OUT/(name+'.svg');svg.write_text(figure(rows,detail))
        if not args.svg_only:
            import cairosvg
            cairosvg.svg2pdf(url=str(svg),write_to=str(OUT/(name+'.pdf')))
        print(svg)
if __name__=='__main__': main()
