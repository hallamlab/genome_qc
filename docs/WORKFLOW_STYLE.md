# Genome QC workflow style: current MP / ASPIRE nodal

The approved references are MetaPathways `docs/assets/workflow-brief-nodal.svg`
and ASPIRE `docs/assets/workflow-brief.svg`. The user explicitly requested this
current connected-node style for Genome QC. It supersedes the older inherited
numbered-module / horizontal-row specification for these figures.

- Continuous top-to-bottom connected flow, with true parallel branches and joins.
- Blue input circles (`#DAE8FC`, outline `#6C8EBF`), white process diamonds with
  thin black outlines, green output circles (`#D5E8D4`, outline `#82B366`).
- External labels vertically centered beside nodes; smaller parenthesized software
  names. Trebuchet MS or a close sans-serif fallback, matching MP/ASPIRE.
- Thin black orthogonal connectors, small arrowheads separated from junctions,
  and black dots for real joins/splits. No arrows through labels.
- White background, compact vertical legend, no title banner, numbered square
  spine or horizontal module grid. Explain scientific details in the caption/text.
- Brief and detailed figures use the same organization and grammar. The detailed
  figure expands real operations; it does not replace them with paragraph boxes.
- Preserve biological truth: every assessed genome remains in the master; final
  selection is a side product; tRNA and GUNC are parallel assessments.

Edit `scripts/build_workflow_figures.py`, rebuild SVG/PDF exports, run
`python scripts/check_workflow_style.py`, inspect labels and connector clearance,
and build the public guide with Sphinx warnings treated as errors.
Keep the brief on the README and guide landing page and the detailed figure on
its workflow page. Supporting architecture/data-flow diagrams remain secondary.
Biological plot styles are separate from these workflow figures.
