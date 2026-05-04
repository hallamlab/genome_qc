`enrichment_stats.py` is the repo-local pure-Python implementation of the Shaiber/Willis enrichment method used by anvi'o.

Preserved method:
- grouped binomial test of `cbind(x, N - x) ~ group`
- enrichment score = Rao score test equivalent statistic for the group effect
- unadjusted p-value from the chi-square reference distribution
- adjusted q-values using a Storey-style `qvalue` workflow with the same lambda fallback idea as anvi'o
- `associated_groups` semantics preserved:
  groups whose `p_group` exceeds the overall proportion

Expected input schema:
- first column: entity label
- `accession`
- one arbitrary ID/list column
- `associated_groups`
- one `p_<group>` column per group
- one `N_<group>` column per group

Example:

```bash
python scripts/enrichment_stats.py \
  --input test/data/enrichment_stats_example.tsv \
  --output /tmp/enrichment_stats_example.out.tsv
```

Workflow integration:
- `summarize_metapathways_atlas_linked.py` already accepts the two high-level wrapper directories you care about:
  the MetaPathways wrapper output dir plus `--genome-atlas-dir`
- the main `summarize_metapathways_wrapper.py` atlas-linked step also calls this same method through `metapathways_atlas_linked_core.py`
- the linked comparison tables and plots keep their existing filenames and general shape, but their enrichment statistics now come from this Python implementation instead of the old Fisher/BH path

Optional behavior:
- `--compute-associated-groups-if-missing`
  lets the script reconstruct `associated_groups` using the preserved anvi'o rule
- `--allow-empty-associated-groups`
  disables the default hard failure for a completely empty `associated_groups` column
