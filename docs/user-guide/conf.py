"""Public guide only: never crawl the adjacent local research documents."""
import os
project = 'Genome QC'
author = 'Genome QC contributors'
copyright = '2026, Genome QC contributors'
extensions = ['myst_parser', 'sphinxcontrib.mermaid']
source_suffix = {'.md': 'markdown'}
root_doc = 'index'
include_patterns = ['index.md', 'installation.md', 'quickstart.md', 'test.md',
                    'inputs.md', 'workflow.md', 'commands.md', 'databases.md',
                    'resources.md', 'outputs.md', 'selection.md', 'recovery.md',
                    'interpretation.md', 'cli-reference.md', 'citations.md', 'development.md']
myst_heading_anchors = 4
myst_fence_as_directive = ['mermaid']
html_theme = 'sphinx_rtd_theme'
html_theme_options = {'collapse_navigation': False, 'navigation_depth': 2}
html_title = project
html_baseurl = os.environ.get('READTHEDOCS_CANONICAL_URL', '')
html_static_path = ['assets']
html_css_files = ['docs.css']
mermaid_version = '11.12.1'
mermaid_init_config = {'startOnLoad': False, 'theme': 'neutral', 'fontFamily': 'Times New Roman, Times, serif', 'flowchart': {'htmlLabels': False}}
mermaid_fullscreen = True
mermaid_light_theme = 'neutral'
mermaid_dark_theme = 'neutral'
