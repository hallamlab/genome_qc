"""Cohort ANI groups and GTDB-marker trees; no quality or taxonomy-name exclusion."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from preserve_marker_alignments import fasta, stage_msa


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_table(path, rows, columns):
    with open(path, 'w') as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def run_logged(command, output, env=None):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with open(output, 'w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def ani_pairs(raw, ids, threshold, af, lengths, containment=True):
    directed = {}
    for line in Path(raw).read_text().splitlines():
        q, r, identity, mapped, total = line.split('\t')
        q, r = Path(q).stem, Path(r).stem
        if q not in ids or r not in ids:
            raise ValueError('FastANI returned an unknown genome')
        if q == r:
            continue
        identity, mapped, total = float(identity), int(mapped), int(total)
        if not math.isfinite(identity) or not 0 <= identity <= 100 or not 0 <= mapped <= total or total <= 0:
            raise ValueError('Invalid FastANI measurement')
        if (q, r) in directed:
            raise ValueError('Duplicate FastANI measurement')
        directed[q, r] = (identity, mapped / total)
    pairs = []
    for a, b in sorted({tuple(sorted(key)) for key in directed}):
        ab, ba = directed.get((a, b)), directed.get((b, a))
        both = ab is not None and ba is not None
        identity = min(ab[0], ba[0]) if both else None
        fraction = min(ab[1], ba[1]) if both else None
        larger, smaller = ((ab, ba) if lengths[a] >= lengths[b] else (ba, ab))
        af_large, af_small = (larger[1] if larger else None), (smaller[1] if smaller else None)
        standard = both and af_large >= af
        rescue = both and containment and af_small >= .90 and af_large >= .10
        passed = both and identity >= threshold and (standard or rescue)
        pairs.append(dict(genome_a=ids[a], genome_b=ids[b], ani_min=identity, af_a=ab[1] if ab else None,
                          af_b=ba[1] if ba else None, af_min=fraction,
                          passes=passed, af_larger=af_large, af_smaller=af_small,
                          match_type=('standard' if standard else 'containment') if passed else 'unmatched',
                          status='reciprocal' if both else 'one_direction_only'))
    return pairs


def clusters(genomes, pairs):
    parent = {g: g for g in genomes}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]; g = parent[g]
        return g
    for pair in pairs:
        if pair['passes']:
            a, b = find(pair['genome_a']), find(pair['genome_b'])
            parent[max(a,b)] = min(a,b)
    groups = {}
    for g in sorted(genomes):
        groups.setdefault(find(g), []).append(g)
    return [dict(Genome_Id=g, ani_cluster=f'ANI_{i:04d}', cluster_size=len(members))
            for i, members in enumerate(groups.values(), 1) for g in members]


def run_ani(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = read_table(args.master)
    ids = {r['Bin Id']: r['Genome_Id'] for r in rows}
    paths = {p.stem: p.resolve() for p in Path(args.genomes).glob('*.fasta')}
    if set(ids) - set(paths):
        raise ValueError('Representative FASTAs missing from ANI inputs')
    sequences = {g: [len(seq) for _,seq in fasta(paths[g])] for g in ids}
    lengths = {g: sum(sizes) for g,sizes in sequences.items()}
    eligible = {g for g,sizes in sequences.items() if any(n >= 3000 for n in sizes)}
    listing = out / 'genomes.txt'
    listing.write_text(''.join(str(paths[g]) + '\n' for g in sorted(eligible)))
    raw = out / 'fastani.raw.tsv'
    if len(eligible) >= 2:
        tool = shutil.which('fastANI') or shutil.which('fastani')
        if not tool:
            raise RuntimeError('FastANI is required for the default ANI stage')
        run_logged([tool, '--ql', str(listing), '--rl', str(listing), '-t', str(args.threads), '-o', str(raw)], out / 'fastani.log')
    else:
        raw.touch()
    pairs = ani_pairs(raw, ids, args.ani, args.af, lengths, not args.no_containment)
    write_table(out / 'ani_pairs.tsv', pairs, ['genome_a','genome_b','ani_min','af_a','af_b','af_min','af_larger','af_smaller','passes','match_type','status'])
    write_table(out / 'ani_status.tsv', [dict(Genome_Id=ids[g], status='compared' if g in eligible and len(eligible)>=2 else 'not_compared',
        reason='' if g in eligible and len(eligible)>=2 else ('no_contig_at_least_3000_bp' if g not in eligible else 'fewer_than_two_eligible_genomes'))
        for g in sorted(ids)], ['Genome_Id','status','reason'])
    group_rows = clusters(ids.values(), pairs)
    write_table(out / 'ani_clusters.tsv', group_rows, ['Genome_Id','ani_cluster','cluster_size'])
    plot_ani(pairs, group_rows, out / 'ani_summary', args.ani, args.af)
    (out / 'provenance.json').write_text(json.dumps(dict(ani_threshold=args.ani, af_threshold=args.af,
        rule='minimum reciprocal ANI; larger-assembly AF threshold, or labelled smaller-assembly containment',
        containment_enabled=not args.no_containment, containment_smaller_af=.90, containment_larger_af_floor=.10,
        clustering='connected components; not necessarily all-pairs cliques',
        missing_pairs='unreported, not zero ANI', genomes=len(ids)), indent=2))


def plot_ani(pairs, groups, output, ani, af):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for mode, marker, color in [('unmatched','o','#bbb'),('standard','o','#245d79'),('containment','^','#a06733')]:
        subset=[r for r in pairs if r['match_type']==mode and r['ani_min'] is not None]
        if subset: axes[0].scatter([r['ani_min'] for r in subset],[r['af_larger'] for r in subset],s=20,marker=marker,color=color,label=mode,alpha=.7)
    axes[0].axvline(ani,color='#555',ls='--',lw=1); axes[0].axhline(af,color='#555',ls='--',lw=1)
    axes[0].set(xlabel='Minimum reciprocal ANI (%)',ylabel='Larger-assembly alignment fraction',ylim=(0,1.02))
    if axes[0].get_legend_handles_labels()[0]: axes[0].legend(frameon=False,fontsize=8)
    sizes=Counter({r['ani_cluster']:r['cluster_size'] for r in groups}.values())
    axes[1].bar([str(n) for n in sorted(sizes)],[sizes[n] for n in sorted(sizes)],color='#6C8EBF')
    axes[1].set(xlabel='Genomes per connected ANI group',ylabel='Number of groups')
    from matplotlib.ticker import MaxNLocator
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    for ext in ('png','pdf','svg'): fig.savefig(str(output)+'.'+ext,dpi=160,bbox_inches='tight')
    plt.close(fig)


def collect_alignments(directories, domain, ids):
    result = {}
    for directory in directories:
        for path in sorted(Path(directory).glob(f'{domain}.fasta.gz')):
            for variant, seq in fasta(path):
                if variant not in ids:
                    continue
                name = ids[variant]
                if name in result and result[name] != seq:
                    raise ValueError(f'Conflicting marker sequences for {name}')
                if any(c.isalpha() and c.upper() != 'X' for c in seq):
                    result[name] = seq
    if len({len(seq) for seq in result.values()}) > 1:
        raise ValueError(f'Incompatible {domain} alignment lengths across GTDB batches')
    return result


def draw_tree(tree_path, rows, cluster_map, output):
    from Bio import Phylo
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    tree = Phylo.read(tree_path, 'newick')
    tree.ladderize()
    leaves = tree.get_terminals()
    metadata = {r['Genome_Id']: r for r in rows}
    if len(leaves) != len({n.name for n in leaves}) or any(n.name not in metadata for n in leaves):
        raise ValueError('Tree contains duplicate or unknown tips')
    # Display root is arbitrary: neither midpoint nor outgroup implies ancestry.
    y = {n: i for i, n in enumerate(leaves)}
    x = tree.depths()
    scale = max(x.values()) or 1.0
    def pos(n):
        if n not in y:
            y[n] = sum(pos(c) for c in n.clades) / len(n.clades)
        return y[n]
    pos(tree.root)
    phyla = sorted({metadata[n.name].get('Phylum') or 'Unclassified' for n in leaves})
    cmap = plt.get_cmap('tab20')
    colors = {p: cmap((2*i) % 20) for i,p in enumerate(phyla)}
    max_label = max(len(n.name) + len(metadata[n.name].get('Species', '')) for n in leaves)
    label_width = max(3.5, min(14, max_label * .065))
    legend_rows = math.ceil(len(phyla)/4)
    fig = plt.figure(figsize=(8.0 + label_width, max(4.5, len(leaves)*.23 + 2 + legend_rows*.22)))
    grid = fig.add_gridspec(1, 4, width_ratios=[4, label_width, 1.6, 1.8], wspace=.06)
    axes = [fig.add_subplot(grid[0,i]) for i in range(4)]
    ax, labels, heat, groups = axes
    for n in tree.find_clades():
        if n.clades:
            ax.plot([x[n],x[n]], [min(y[c] for c in n.clades),max(y[c] for c in n.clades)],color='#333',lw=.7)
            for c in n.clades:
                ax.plot([x[n],x[c]], [y[c],y[c]],color='#333',lw=.7)
            if n.confidence is not None and n.confidence >= .95:
                ax.plot(x[n],y[n],'o',color='#333',ms=2)
    import numpy as np
    values=[]
    for n in leaves:
        row=metadata[n.name]
        species=row.get('Species') or 'Unclassified'
        labels.text(0,y[n], f'{n.name} | {species}', va='center',fontsize=8,
                    color=colors[row.get('Phylum') or 'Unclassified'])
        def metric(key):
            try: return float(row.get(key,''))
            except (ValueError,TypeError): return float('nan')
        values.append([metric('Completeness'),metric('Contamination')])
        groups.text(0,y[n],cluster_map.get(n.name,'—'),va='center',fontsize=8)
    heat.imshow(np.ma.masked_invalid(values),aspect='auto',origin='lower',extent=(-.5,1.5,-.5,len(leaves)-.5),
                cmap='Greys',norm=Normalize(0,100))
    for i,row in enumerate(values):
        for j,v in enumerate(row):
            heat.text(j,i, f'{v:.1f}' if math.isfinite(v) else 'NA',ha='center',va='center',fontsize=7,
                      color='white' if math.isfinite(v) and v>60 else 'black')
    heat.set_xticks([0,1],['Complete\n(%)','Contam.\n(%)'],fontsize=8)
    heat.xaxis.tick_top()
    labels.set_title('Genome | GTDB species',loc='left',fontsize=10)
    groups.set_title('ANI group',loc='left',fontsize=10)
    ax.set_xlim(-scale*.02,scale*1.04); ax.set_xlabel('Substitutions / site',fontsize=9)
    ax.set_yticks([])
    for a in axes:
        a.set_ylim(len(leaves)-.5,-.5)
        if a is not ax:
            a.set_yticks([])
            if a is not heat: a.set_axis_off()
        for spine in a.spines.values(): spine.set_visible(False)
    fig.suptitle(Path(tree_path).stem + ' — GTDB marker phylogeny',fontsize=13,y=1-.1/fig.get_figheight())
    fig.legend(handles=[Patch(color=colors[p],label=p) for p in phyla],loc='lower center',ncol=4,fontsize=8,frameon=False)
    fig.subplots_adjust(top=1-.8/fig.get_figheight(),bottom=(.65+legend_rows*.22)/fig.get_figheight(),left=.04,right=.98)
    for ext in ('pdf','svg','png'):
        fig.savefig(str(output)+'.'+ext,bbox_inches='tight',dpi=min(160,60000/fig.get_figheight()))
    plt.close(fig)
    write_table(str(output)+'.tips.tsv',[{**metadata[n.name], 'ani_cluster':cluster_map.get(n.name,'')} for n in leaves],list(rows[0])+['ani_cluster'])


def run_phylogeny(args):
    from Bio import Phylo
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=read_table(args.master)
    ids={r['Bin Id']:r['Genome_Id'] for r in rows}
    groups={r['Genome_Id']:r['ani_cluster'] for r in read_table(args.clusters)} if args.clusters else {}
    statuses=[]
    placed=set()
    for domain in ('bac120','ar53'):
        sequences=collect_alignments(args.alignments,domain,ids)
        if len(sequences)<3:
            statuses += [dict(Genome_Id=g,domain=domain,status='not_inferred',reason='fewer_than_three_usable_aligned_genomes') for g in sequences]
            placed.update(sequences)
            continue
        alignment=out/f'{domain}.fasta'
        # Opaque tip names keep arbitrary input identifiers out of Newick syntax.
        tokens={f'tip{i:06d}':g for i,g in enumerate(sorted(sequences))}
        alignment.write_text(''.join(f'>{token}\n{sequences[g]}\n' for token,g in tokens.items()))
        write_table(out/f'{domain}.tip_ids.tsv',[dict(tip=t,Genome_Id=g) for t,g in tokens.items()],['tip','Genome_Id'])
        tool=shutil.which('FastTreeMP') or shutil.which('FastTree') or shutil.which('fasttree')
        if not tool: raise RuntimeError('FastTree is required for marker phylogeny')
        command=[tool,'-lg','-gamma','-seed','42',str(alignment)]
        (out/f'{domain}.command.json').write_text(json.dumps(command))
        raw=out/f'{domain}.raw.nwk'
        with raw.open('w') as result, (out/f'{domain}.log').open('w') as log:
            subprocess.run(command,stdout=result,stderr=log,check=True,env={**os.environ,'OMP_NUM_THREADS':str(args.threads)})
        tree=Phylo.read(raw,'newick')
        if len(tree.get_terminals()) != len(tokens) or {n.name for n in tree.get_terminals()} != set(tokens): raise ValueError('Tree tip set differs from alignment')
        for n in tree.get_terminals(): n.name=tokens[n.name]
        tree_path=out/f'{domain}.nwk'; Phylo.write(tree,tree_path,'newick',format_confidence='%1.8f',format_branch_length='%1.10f')
        draw_tree(tree_path,rows,groups,out/f'{domain}_annotated_tree')
        statuses += [dict(Genome_Id=g,domain=domain,status='inferred',reason='') for g in sequences]
        placed.update(sequences)
    statuses += [dict(Genome_Id=g,domain='',status='not_inferred',reason='no_usable_GTDB_marker_alignment') for g in ids.values() if g not in placed]
    write_table(out/'phylogeny_status.tsv',statuses,['Genome_Id','domain','status','reason'])
    (out/'methods.json').write_text(json.dumps(dict(method='FastTree approximate maximum likelihood; LG, CAT topology search and gamma branch-length rescaling',
        support='SH-like local support; dots >=0.95, not bootstrap probabilities', root='arbitrary display root; no outgroup',
        alignment='GTDB masked user marker alignments, reused across batches; Bacteria and Archaea separate',seed=42),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('stage-msa');s.add_argument('source');s.add_argument('output')
    for mode in ('ani','phylogeny'):
        s=sub.add_parser(mode);s.add_argument('--master',required=True);s.add_argument('--output',required=True);s.add_argument('--threads',type=int,default=1)
        if mode=='ani':
            s.add_argument('--genomes',required=True);s.add_argument('--ani',type=float,default=98);s.add_argument('--af',type=float,default=.51);s.add_argument('--no-containment',action='store_true')
        else:
            s.add_argument('--alignments',nargs='+',required=True);s.add_argument('--clusters')
    a=p.parse_args()
    if a.mode=='stage-msa': stage_msa(a.source,a.output)
    elif a.mode=='ani':
        if not 0<=a.ani<=100 or not 0<=a.af<=1: p.error('ANI must be 0–100 and AF 0–1')
        run_ani(a)
    else: run_phylogeny(a)


if __name__=='__main__': main()
