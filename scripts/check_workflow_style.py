#!/usr/bin/env python3
"""Check the shared MP nodal workflow contract without third-party dependencies."""
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NS = '{http://www.w3.org/2000/svg}'

def check(path):
    root = ET.parse(path).getroot()
    source = path.read_text()
    errors = []
    if root.get('data-workflow-style') == 'genome-qc-connected-nodal':
        if 'Trebuchet MS' not in source or list(root.iter(NS+'foreignObject')):
            errors.append('retain approved sans-serif labels as native SVG text')
        groups = {n.get('id') for n in root.iter(NS+'g')}
        for name in ('genomes','references','prepare','quality','taxonomy','rrna',
                     'representative','trna','gunc','merge','master','atlas',
                     'selection','selected','explorer','outputs'):
            if 'node-'+name not in groups:
                errors.append('missing connected workflow node: '+name)
        for color in ('#DAE8FC', '#D5E8D4'):
            if sum(n.get('fill') == color for n in root.iter(NS+'circle')) < 2:
                errors.append('retain blue inputs and green outputs')
        if any(re.fullmatch(r'\d+', (n.text or '').strip()) for n in root.iter(NS+'text')):
            errors.append('omit the superseded numbered module spine')
        if not list(root.iter(NS+'marker')) or not any(n.get('data-junction') == 'true' for n in root.iter(NS+'circle')):
            errors.append('retain directional connectors and junction dots')
        return errors
    if root.get('data-workflow-style') == 'aspire-v1-nodal':
        if 'Trebuchet MS' not in source or list(root.iter(NS+'foreignObject')):
            errors.append('use approved typography and native SVG text')
        for color in ('#DAE8FC', '#D5E8D4'):
            if sum(n.get('fill') == color for n in root.iter(NS+'circle')) < 2:
                errors.append(f'missing input/output circles: {color}')
        groups = {n.get('id') for n in root.iter(NS+'g')}
        for name in ('s1_0','s2_0','s3_0','community','population','report','end'):
            if f'node-{name}' not in groups:
                errors.append(f'missing workflow node: {name}')
        if not any(n.get('data-junction') == 'true' for n in root.iter(NS+'circle')):
            errors.append('retain black junction dots')
        if 'stroke-width="5.5"' not in source:
            errors.append('retain the heavy main path')
        if 'stroke-dasharray="2 4"' in source:
            errors.append('omit dotted text guide lines')
        return errors
    if root.get('data-workflow-style') == 'standard-flowchart':
        kinds = {n.get('data-node-kind') for n in root.iter()}
        for kind in ('input', 'output', 'process', 'decision', 'db', 'end'):
            if kind not in kinds:
                errors.append(f'missing standard flowchart shape: {kind}')
        for color in ('#DAE8FC', '#D5E8D4', '#F5F5F5'):
            if color not in source:
                errors.append(f'missing MP palette color {color}')
        if 'Trebuchet MS' not in source or not list(root.iter(NS+'marker')):
            errors.append('retain approved typography and directional connectors')
        if list(root.iter(NS+'foreignObject')):
            errors.append('labels must be native SVG text')
        return errors
    aspire_style = root.get('data-workflow-style') == 'aspire-v1-main-chain'
    if aspire_style:
        if 'Trebuchet MS' not in source:
            errors.append('use the approved ASPIRE-v1 Trebuchet MS typography')
    else:
        if 'Times New Roman' not in source or 'serif' not in source:
            errors.append('use the shared Times serif typography')
        if any(font in source for font in ('Arial', 'Helvetica', 'sans-serif')):
            errors.append('workflow labels must retain serif typography')
        if not any(n.get('fill', '').upper() == '#CCCCCC' for n in root.iter(NS+'rect')):
            errors.append('keep the restrained gray header / legend')
    circles = list(root.iter(NS + 'circle'))
    for color, role in [('#DAE8FC', 'input'), ('#D5E8D4', 'output')]:
        count = sum(n.get('fill', '').upper() == color for n in circles)
        if count < 2:
            errors.append(f'keep circular {role} nodes beyond the legend')
    squares = []
    for n in root.iter(NS + 'rect'):
        try:
            w, h = float(n.get('width', 0)), float(n.get('height', 0))
            if 18 <= w <= 40 and abs(w-h) < 0.1 and float(n.get('rx', 0)) == 0:
                squares.append(n)
        except ValueError:
            pass
    if len(squares) < 3:
        errors.append('keep the numbered square module spine')
    numbered = [n for n in root.iter() if n.tag in (NS+'text', NS+'tspan')
                and re.fullmatch(r'\d+', (n.text or '').strip())]
    if len(numbered) < 2:
        errors.append('keep module numbers')
    diamonds = [n for n in root.iter(NS+'path')
                if len(re.findall('[Ll]', n.get('d', ''))) >= 3
                and n.get('fill', '').upper() == '#F5F5F5']
    if len(diamonds) < 3:
        errors.append('keep diamond compute nodes beyond the legend')
    if not list(root.iter(NS+'marker')):
        errors.append('keep directional connectors')
    return errors

def main():
    paths = [ROOT / 'docs/user-guide/assets' / name for name in
             ('workflow.svg', 'workflow-main.svg', 'workflow-brief.svg', 'workflow-detailed.svg', 'workflow-brief-nodal.svg')]
    paths = [p for p in paths if p.exists()]
    if not paths:
        print('No primary workflow SVGs in this checkout; apply docs/WORKFLOW_STYLE.md to new figures.')
        return 0
    failed = False
    for path in paths:
        errors = check(path)
        failed |= bool(errors)
        print(('FAIL' if errors else 'PASS') + ' ' + str(path.relative_to(ROOT)))
        for error in errors:
            print('  ' + error)
    return int(failed)

if __name__ == '__main__':
    sys.exit(main())
