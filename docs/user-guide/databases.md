# Databases

The database root selected with `-d` contains `checkm_data/`,
`gtdbtk_r232/` and `gunc_db/`. Without `-d`, it defaults to
`~/.local/share/genome_qc/databases`. Reuse this root between runs.

GTDB R232 is paired with GTDB-Tk 2.8.0. R220/R226 databases are not reused
under this configuration. Classification uses `--place_species`: the initial
skani ANI screen runs, and genomes assigned there are also processed for marker
alignment and reference-tree placement. This preserves the alignments needed
for the downstream cohort phylogeny.
See the [official release compatibility table](https://ecogenomics.github.io/GTDBTk/installing/index.html).

GTDB-Tk R232 and CheckM archives are downloaded over HTTPS with pinned checksum
verification. The installer reports transferred GiB, percentage when the server
provides a size, and transfer speed about every ten seconds. Transient failures
retry up to five attempts with bounded exponential backoff. HTTP range requests
resume partial downloads; servers that ignore range requests trigger a safe
restart instead. Corrupt downloads are discarded and retried.

If `DATABASE_ROOT/gtdbtk_r232_data.tar.gz` already exists, it is verified
against the pinned archive checksum and extracted automatically into
`DATABASE_ROOT/gtdbtk_r232/`. No manual extraction or archive copy is needed.
Your supplied archive is preserved after installation; a checksum mismatch
stops installation without deleting it or silently downloading a replacement.
The same lookup applies to the official CheckM archive filename.

Partial archives persist under `DATABASE_ROOT/.downloads/`. If retries are
exhausted or the process is interrupted, rerun the same command (with `--resume`
for an interrupted workflow). Retained archives are checked and resumed without
an extra flag. A verified archive also survives extraction failure, so retrying
extraction does not require downloading it again. Permanent HTTP errors and
filesystem errors are reported rather than retried indefinitely.

Extraction uses parallel gzip decompression (rapidgzip) when more than one CPU
is requested, controlled by the existing `--threads` and `--max_cpus` flags.
One CPU is reserved for tar extraction; the remaining requested CPUs decompress.
Database preparation is capped by CPU availability on the controller host.
Checksum verification and file writes can still limit speed.
Extraction uses temporary space under the database root. Allow space for the
compressed archive **plus** the extracted database. Only a verified and
structurally checked extraction is moved into its final location; the archive
is deleted after successful installation. Each installation records source,
checksums and file sizes in `genome_qc_database.json`. A per-database lock
prevents duplicate concurrent installation. Existing complete installations
are reused; existing incomplete destination directories require explicit
attention rather than being overwritten.

GUNC currently uses its own `gunc download_db` command; the resumable HTTP
downloader described above applies to GTDB-Tk and CheckM. Reference databases
and retained download archives are not removed by compact-results cleanup.

## Recognize the reference layout

After setup, the root supplied with `-d` has this layout:

```text
genome-qc-databases/
├── checkm_data/                  # CheckM reference files
├── gtdbtk_r232/                  # Extracted GTDB R232 package
└── gunc_db/
    └── gunc_db_progenomes2.1.dmnd
```

Pass the parent `genome-qc-databases` directory to `-d`, not the GTDB directory
or the GUNC file itself. If you already have this layout, point at it directly;
there is no need to copy the references into each analysis output.

Use a persistent location with enough free space and permission to write during
setup. Setup receipts document downloaded references. Structural checks on an
existing directory establish whether it is usable by the installer; they do not
reconstruct a missing historical download provenance. Retain the original
release information when bringing an existing database installation.

The first setup needs internet. For a cluster with offline compute nodes,
prepare references and the required task environments on an approved host with
network access before submitting work that depends on them. Merely having the
controller installed does not mean that all tool environments are already cached.

## Prepare references before an analysis

```bash
genome_qc build_db -o ~/genome-qc-database-setup \
  -d ~/genome-qc-databases --threads 8 --max_cpus 8
```

A full `run` performs this preparation automatically. Standalone `quality`,
`taxonomy` and `chimera` prepare the reference they require. Files under the
reference root are shared, not bundled into a software package or container.
Do not rename a database from a different release to make it appear compatible.

## Updating from R226

The default database directory is now `gtdbtk_r232/`. An existing `gtdbtk_r226/`
directory is left untouched; it is not renamed or used with GTDB-Tk 2.8.0.
Run `build_db` against your existing database root to install R232 alongside it.
Existing compatible CheckM and GUNC references are reused.

Use a **new analysis output directory** for the R232 analysis. The completed
300-genome development test used R226 and GTDB-Tk 2.6.1; it does not validate
R232 classifications or runtime. Retain its original provenance when comparing
results across releases. Taxonomic assignments may change because both the
reference release and GTDB-Tk classification behavior changed.
