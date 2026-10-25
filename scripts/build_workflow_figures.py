#!/usr/bin/env python3
"""Render Genome QC using the approved current MP / ASPIRE connected-node style."""
from pathlib import Path
from html import escape
import argparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/user-guide/assets'

def figure(detailed=False):
    width=960; cx=440
    nodes=[]; edges=[]; dots=[]
    def node(key,x,y,kind,lines,side='left'):
        nodes.append((key,x,y,kind,lines,side))
    def edge(points,arrow=True): edges.append((points,arrow))
    def down(a,b): edge([(cx,a+21),(cx,b-24)])
    node('genomes',250,45,'input',['Genome FASTAs'])
    node('references',635,45,'input',['Reference databases','(CheckM, GTDB R226, GUNC)'],'right')
    edge([(250,65),(250,105),(cx,105)],False)
    edge([(635,65),(635,105),(cx,105)],False);dots.append((cx,105))
    upper=[('prepare','Preprocessing','(SeqKit)'),
           ('quality','Completeness / contamination','(CheckM, Qscore)'),
           ('taxonomy','Taxonomic classification','(GTDB-Tk)'),
           ('rrna','rRNA prediction / consistency','(barrnap, BLAST+)'),
           ('representative','Representative per genome','(Qscore, N50, length)')]
    if detailed:
        upper=[('prepare','Stage inputs / contig variants','(Genome QC, SeqKit)'),
               ('statistics','Assembly statistics','(SeqKit)'),
               ('quality','Completeness / contamination','(CheckM lineage_wf)'),
               ('qscore','Composite quality score','(Genome QC)'),
               ('taxonomy','Taxonomic classification','(GTDB-Tk, R226)'),
               ('rrna','Predict rRNA genes','(barrnap)'),
               ('consistency','Within-genome rRNA consistency','(BLAST+, Genome QC)'),
               ('representative','Choose one contig-length variant','(Qscore, N50, length)')]
    first=165; pitch=95
    edge([(cx,109),(cx,first-24)])
    previous=None
    for i,(key,title,software) in enumerate(upper):
        y=first+i*pitch
        node(key,cx,y,'process',[title,software])
        if previous is not None: down(previous,y)
        previous=y
    split=previous+65; branch=split+65; join=branch+70
    edge([(cx,previous+23),(cx,split)],False); dots.append((cx,split))
    for x,key,label,software,side in [(285,'trna','tRNA prediction','(tRNAscan-SE)','left'),(640,'gunc','Chimerism assessment','(GUNC)','right')]:
        edge([(cx,split),(x,split),(x,branch-24)])
        node(key,x,branch,'process',[label,software],side)
        edge([(x,branch+23),(x,join),(cx,join)],False)
    dots.append((cx,join))
    merge=join+60; master=merge+85; atlas=master+95; explorer=atlas+95; end=explorer+85
    edge([(cx,join+4),(cx,merge-24)])
    node('merge',cx,merge,'process',['Integrate measurements / status','(Genome QC)'])
    edge([(cx,merge+23),(cx,master-21)])
    node('master',cx,master,'output',['All assessed genomes','Master QC table'])
    edge([(cx,master+21),(cx,atlas-24)])
    node('atlas',cx,atlas,'process',['Genome atlas','(pandas, Matplotlib, seaborn)'])
    down(atlas,explorer)
    node('explorer',cx,explorer,'process',['Explore / select / export','Genome QC Explorer'])
    edge([(cx,explorer+23),(cx,end-21)])
    node('outputs',cx,end,'output',['Atlas + linked tables','CSV / selection records'],'right')
    # Selection is a side product of the master, never a gate to the atlas.
    sx=715
    edge([(cx+21,master),(sx,master),(sx,atlas-24)])
    node('selection',sx,atlas,'process',['Final selection','[optional criteria]'],'right')
    edge([(sx,atlas+23),(sx,explorer-21)])
    node('selected',sx,explorer,'output',['Selected subset'],'right')
    edge([(sx-21,explorer),(cx+24,explorer)])
    height=end+65
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-workflow-style="genome-qc-connected-nodal" role="img" aria-labelledby="title desc">',
      '<title id="title">Genome QC '+('detailed workflow' if detailed else 'brief workflow')+'</title>',
      '<desc id="desc">Continuous node flow from genome FASTAs and references to assessments, parallel tRNA and GUNC checks, a complete master table, optional selection, atlas and explorer.</desc>',
      '<rect width="100%" height="100%" fill="#FFFFFF"/>',
      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#111111"/></marker></defs>',
      '<g font-family="Trebuchet MS, sans-serif" fill="#111111">']
    for pts,arrow in edges:
        parts.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in pts)+'" fill="none" stroke="#111111" stroke-width="1.6"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
    for x,y in dots: parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="#111111" data-junction="true"/>')
    def shape(x,y,kind,r=18):
        if kind=='process':
            parts.append(f'<path d="M{x} {y-21} L{x+21} {y} L{x} {y+21} L{x-21} {y} Z" fill="#FFFFFF" stroke="#111111" stroke-width="1.2"/>')
        else:
            fill,stroke=('#DAE8FC','#6C8EBF') if kind=='input' else ('#D5E8D4','#82B366')
            parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
    def label(x,y,lines,side):
        anchor='end' if side=='left' else 'start';tx=x-36 if side=='left' else x+36
        for i,s in enumerate(lines):
            size=17 if s.startswith(('(','[')) else 20
            yy=y+7+(i-(len(lines)-1)/2)*25
            parts.append(f'<text x="{tx}" y="{yy}" font-size="{size}" text-anchor="{anchor}">{escape(s)}</text>')
    for key,x,y,kind,lines,side in nodes:
        parts.append(f'<g id="node-{key}" data-node-kind="{kind}">');shape(x,y,kind);label(x,y,lines,side);parts.append('</g>')
    for y,kind,text in [(150,'input','Input'),(205,'process','Process'),(260,'output','Output')]:
        shape(820,y,kind,15);label(820,y,[text],'left')
    parts.append('</g></svg>')
    return '\n'.join(parts)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--svg-only',action='store_true');args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    for name,detail in [('workflow-brief',False),('workflow-detailed',True)]:
        svg=OUT/(name+'.svg');svg.write_text(figure(detail))
        if not args.svg_only:
            import cairosvg
            cairosvg.svg2pdf(url=str(svg),write_to=str(OUT/(name+'.pdf')))
        print(svg)
if __name__=='__main__': main()
