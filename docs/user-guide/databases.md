# Databases

The database root selected with `-d` contains `checkm_data/`,
`gtdbtk_r226/` and `gunc_db/`. Without `-d`, it defaults to
`~/.local/share/genome_qc/databases`. Reuse this root between runs.

GTDB R226 is paired with GTDB-Tk 2.6.1. R220/R232 databases are not reused
under this configuration. Classification uses `--skip_ani_screen` to retain
the pre-R232 workflow that skips the initial ANI screen.
See the [official release compatibility table](https://ecogenomics.github.io/GTDBTk/installing/index.html).

GTDB-Tk R226 and CheckM archives are downloaded over HTTPS with pinned checksum
verification. The installer reports transferred GiB, percentage when the server
provides a size, and transfer speed about every ten seconds. Transient failures
retry up to five attempts with bounded exponential backoff. HTTP range requests
resume partial downloads; servers that ignore range requests trigger a safe
restart instead. Corrupt downloads are discarded and retried.

If `DATABASE_ROOT/gtdbtk_r226_data.tar.gz` already exists, it is verified
against the pinned archive checksum and extracted automatically into
`DATABASE_ROOT/gtdbtk_r226/`. No manual extraction or archive copy is needed.
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

## Prepare references before an analysis

```bash
genome_qc build_db -o ~/genome-qc-database-setup \
  -d ~/genome-qc-databases --threads 8 --max_cpus 8
```

A full `run` performs this preparation automatically. Standalone `quality`,
`taxonomy` and `chimera` prepare the reference they require. Files under the
reference root are shared, not bundled into a software package or container.
Do not rename a database from a different release to make it appear compatible.
