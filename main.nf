nextflow.enable.dsl=2
params.rescue_plan = params.rescue_plan ?: ''
// Recovery products were validated by the controller before scheduling.
def rescueStage(stage) {
    if (!params.rescue_plan) return [ids: [], outputs: [], receipts: []]
    return new groovy.json.JsonSlurper().parse(new File(params.rescue_plan.toString()))[stage]
}
def rescueReady(stage, id) { rescueStage(stage).ids.contains(id.toString()) }
def rescuePaths(stage, suffix = '') {
    rescueStage(stage).outputs.findAll { !suffix || it.endsWith(suffix) }.collect { file(it) }
}
def rescueReceipts(stage) { rescueStage(stage).receipts.collect { file(it) } }

params.reference_cpus = params.reference_cpus ?: 1
params.compact_results = params.compact_results ?: false

params.fasta_dir        = params.fasta_dir        ?: "${projectDir}/test/fasta"
params.working_dir      = params.working_dir      ?: "${projectDir}/results_nextflow"
params.filter_size      = params.filter_size      ?: ""
// Final acceptance criteria are opt-in (null defaults in nextflow.config).
params.checkm_cpus      = params.checkm_cpus      ?: 1
params.checkm_batch_size = params.checkm_batch_size ?: 1
params.checkm_max_forks = params.checkm_max_forks ?: 1
params.trnascan_cpus    = params.trnascan_cpus    ?: 1
params.trnascan_max_forks = params.trnascan_max_forks ?: 24
params.gtdbtk_cpus      = params.gtdbtk_cpus      ?: 1
params.gtdbtk_batch_size = params.gtdbtk_batch_size ?: 50
params.gtdbtk_max_forks = params.gtdbtk_max_forks ?: 2
params.gunc_cpus        = params.gunc_cpus        ?: 1
params.atlas_enabled    = (params.atlas_enabled == null) ? true : params.atlas_enabled
params.atlas_prefix     = params.atlas_prefix     ?: "genome_quality"
params.atlas_group_column = params.atlas_group_column ?: "Phylum"
params.atlas_sample_column = params.atlas_sample_column ?: ""
params.atlas_matched_samples_only = params.atlas_matched_samples_only ?: false
params.atlas_top_n_groups = params.atlas_top_n_groups ?: 12
params.atlas_compare_columns = params.atlas_compare_columns ?: ""
params.atlas_compare_map = params.atlas_compare_map ?: ""
params.atlas_compare_map_key = params.atlas_compare_map_key ?: ""
params.atlas_ani_compare_columns = params.atlas_ani_compare_columns ?: ""
params.atlas_ani_results = params.atlas_ani_results ?: ""
params.atlas_ani_genome_dir = params.atlas_ani_genome_dir ?: ""
params.atlas_ani_fasta_column = params.atlas_ani_fasta_column ?: ""
params.atlas_ani_fasta_exts = params.atlas_ani_fasta_exts ?: ".fasta,.fa,.fna,.fasta.gz,.fa.gz,.fna.gz"
params.atlas_ani_threshold = params.atlas_ani_threshold ?: 98.0
params.atlas_ani_af_threshold = (params.atlas_ani_af_threshold == null) ? 0.51 : params.atlas_ani_af_threshold
params.atlas_ani_threads = params.atlas_ani_threads ?: 1
params.atlas_ani_top_overlaps = params.atlas_ani_top_overlaps ?: 15
params.atlas_cpus       = params.atlas_cpus       ?: params.atlas_ani_threads
params.atlas_memory     = params.atlas_memory     ?: "8 GB"
params.atlas_time       = params.atlas_time       ?: "12h"
params.max_forks        = params.max_forks        ?: 64
params.ref_dir          = params.ref_dir          ?: ""
params.checkm_data_path = params.checkm_data_path ?: ""
params.gtdbtk_data_path = params.gtdbtk_data_path ?: ""
params.gunc_db          = params.gunc_db          ?: ""
params.gtdbtk_package_url = params.gtdbtk_package_url ?: "https://data.gtdb.ecogenomic.org/releases/release232/232.0/auxillary_files/gtdbtk_package/full_package/gtdbtk_r232_data.tar.gz"
params.debug            = params.debug            ?: false


// Nextflow hashes task scripts, inputs and environments. Also hash external
// Python helpers used by a task, including their local imports. Controller-only
// changes must not invalidate unrelated biological tasks.
def helperDigest(names) {
    def digest = java.security.MessageDigest.getInstance('SHA-256')
    names.toSorted().each { name ->
        digest.update(name.getBytes('UTF-8'))
        def helper = new File(projectDir.toString(), name)
        digest.update(helper.exists() ? helper.bytes : 'absent'.getBytes('UTF-8'))
    }
    return digest.digest().encodeHex().toString()
}

def resolveAbsolutePath(pathString) {
    if (pathString == null) {
        return ""
    }
    def raw = pathString.toString().trim()
    if (!raw) {
        return ""
    }
    def asFile = new File(raw)
    if (!asFile.isAbsolute()) {
        asFile = new File(projectDir.toString(), raw)
    }
    return asFile.getAbsolutePath()
}

def gtdbtkDbLooksValid(dbPath) {
    if (!dbPath) {
        return false
    }
    def root = new File(dbPath)
    if (!root.exists() || !root.isDirectory()) {
        return false
    }
    def metadata = new File(root, "metadata/metadata.txt")
    if (!metadata.isFile()) return false
    def release = metadata.text =~ /(?m)^VERSION_DATA\s*=\s*['"]?(r[0-9]+)/
    return release.find() && release.group(1) == 'r232'
}

def guncDbLooksValid(dbPath) {
    if (!dbPath) {
        return false
    }
    def fileObj = new File(dbPath)
    return fileObj.exists() && fileObj.isFile() && fileObj.length() > 0
}

def checkmDataLooksValid(dataPath) {
    if (!dataPath) {
        return false
    }
    def root = new File(dataPath)
    if (!root.exists() || !root.isDirectory()) {
        return false
    }
    def children = root.listFiles()
    return children != null && children.length > 0
}

def normalizeToList(value) {
    if (value == null) {
        return []
    }
    if (value instanceof Collection) {
        return value as List
    }
    return [value]
}

def normalizeDelimitedList(value) {
    if (value == null) {
        return []
    }
    if (value instanceof Collection) {
        return value.collect { it.toString().trim() }.findAll { it }
    }
    def raw = value.toString().trim()
    if (!raw) {
        return []
    }
    return raw.split(/\s*,\s*/).collect { it.toString().trim() }.findAll { it }
}

def shellQuote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

def genomeId(path) {
    def name
    if (path instanceof java.nio.file.Path) {
        name = path.getFileName().toString()
    } else {
        name = path.getName()
    }
    return name.replaceFirst(/(\.fa|\.fna|\.fasta)(\.gz)?$/, "")
}

def filterSizes() {
    if (params.filter_size instanceof List) {
        def sizes = params.filter_size.collect { it.toString().trim() }.findAll { it }
        return sizes ? sizes : ["full"]
    }
    def fs = (params.filter_size == null) ? '' : params.filter_size.toString().trim()
    if (!fs) {
        return ["full"]
    }
    return fs.tokenize(' ')
}

params.helper_qscore = helperDigest(['calc_qscore.py'])
params.helper_compare_barrnap = helperDigest(['compare_barrnap.py'])
params.helper_dedupe = helperDigest(['dedupe_genomes.py'])
params.helper_trnascan_parse = helperDigest(['trnascan_parse.py'])
params.helper_setup_gtdbtk_db = helperDigest(['database_install.py'])
params.helper_setup_gunc_db = helperDigest(['database_install.py'])
params.helper_setup_checkm_db = helperDigest(['database_install.py'])
params.helper_merge_master = helperDigest(['merge_master.py', 'assessment_summary.py', 'merge_gunc_batches.py', 'genome_quality_atlas.py', 'genome_set_stats.py', 'gunc_method_stats.py'])
params.helper_genome_quality_atlas = helperDigest(['genome_quality_atlas.py', 'genome_set_stats.py', 'gunc_method_stats.py', 'scripts/genome_quality_category_lollipop_panel.py'])
params.helper_compact = params.compact_results ? helperDigest(['compact_task.py']) : 'unused'

params.helper_cohort = helperDigest(['cohort_phylogeny.py', 'preserve_marker_alignments.py'])
params.helper_marker_alignments = helperDigest(['preserve_marker_alignments.py'])
params.helper_assessment = helperDigest(['assessment_task.py'])
params.helper_merge_gunc = helperDigest(['merge_gunc_batches.py'])
// Reuse successful tool tasks; retry only recorded failed assessments on resume.
def assessmentToken(folder, name) {
    def receipt = new File("${params.working_dir}/${folder}/${name}.assessment.json")
    if (!receipt.exists()) return '0'
    def prior = new groovy.json.JsonSlurper().parse(receipt)
    def token = prior.retry_token ?: '0'
    return prior.status == 'completed' ? token : (token.toInteger() + 1).toString()
}

// Longest-input-first scheduling balances estimated sequence work, not just counts.
// Every group is capped; stable ordering makes membership reproducible on resume.
def balancedBatches(inputs, maximum, parallel) {
    def files = normalizeToList(inputs)
    def weights = [:]
    files.each { weights[it] = it.size() }
    files = files.toSorted { a, b -> (weights[b] <=> weights[a]) ?: (a.name <=> b.name) }
    if (!files) return []
    def cap = Math.max(1, Math.min(1000, maximum as Integer))
    def count = Math.min(files.size(), Math.max((Math.ceil(files.size() / (cap as Double)) as Integer), parallel as Integer))
    def bins = (0..<count).collect { [index: it, bytes: 0L, files: []] }
    files.each { f ->
        def bin = bins.findAll { it.files.size() < cap }.min { a, b -> (a.bytes <=> b.bytes) ?: (a.files.size() <=> b.files.size()) ?: (a.index <=> b.index) }
        bin.files.add(f)
        bin.bytes += weights[f]
    }
    return bins.findAll { it.files }.collect { it.files.toSorted { a, b -> a.name <=> b.name } }
}
def batchId(files) {
    return java.security.MessageDigest.getInstance('MD5').digest(files.collect { it.name }.join('|').getBytes('UTF-8')).encodeHex().toString().substring(0, 12)
}

// Batch membership and order are deterministic; per-genome receipts are kept.
def smallBatches(records, fastaIndex) {
    def byPath = records.collectEntries { [(it[fastaIndex].toString()): it] }
    return balancedBatches(records.collect { it[fastaIndex] }, params.small_batch_size, params.max_forks)
        .collect { files -> files.collect { byPath[it.toString()] } }
}

process PREPROCESS_BATCH {
    tag "${batch_id}"
    cpus params.small_batch_cpus
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/genomes_pruned", mode: 'copy', overwrite: true, pattern: 'results/*', saveAs: { filename -> new File(filename).getName() }
    input:
    tuple val(batch_id), val(records), path(fastas)
    output:
    path('results/*'), emit: results
    script:
    def manifest = groovy.json.JsonOutput.toJson(records)
    def helper = helperDigest(['small_task_batch.py', 'assessment_task.py'])
    """
    # Batch helper: ${helper}
    cat > batch.json <<'BATCH_JSON'
    ${manifest}
BATCH_JSON
    python ${projectDir}/small_task_batch.py preprocess batch.json --cpus ${task.cpus} --checkpoint-dir ${params.working_dir}/genomes_pruned ${params.compact_results ? '--compact' : ''}
    """
}

process RRNA_BATCH {
    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1
    tag "${batch_id}"
    cpus params.small_batch_cpus
    maxForks params.max_forks
    conda "${projectDir}/envs/rrna_batch.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true, pattern: 'results/*', saveAs: { filename -> new File(filename).getName() }
    input:
    tuple val(batch_id), val(records), path(fastas)
    output:
    path('results/*'), emit: results
    script:
    def manifest = groovy.json.JsonOutput.toJson(records)
    def helper = helperDigest(['small_task_batch.py', 'assessment_task.py'])
    """
    # Batch helper: ${helper}
    cat > batch.json <<'BATCH_JSON'
    ${manifest}
BATCH_JSON
    python ${projectDir}/small_task_batch.py rrna batch.json --cpus ${task.cpus} --checkpoint-dir ${params.working_dir}/barrnap ${params.rescue_plan ? '--reuse-rrna ' + new File(params.rescue_plan.toString()).parent + '/rrna' : ''} ${params.compact_results ? '--compact' : ''}
    """
}

process TRNA_BATCH {
    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1
    tag "${batch_id}"
    cpus params.trnascan_cpus
    maxForks params.max_forks
    conda "${projectDir}/envs/trnascan.yaml"
    publishDir "${params.working_dir}/trnascan", mode: 'copy', overwrite: true, pattern: 'results/*', saveAs: { filename -> new File(filename).getName() }
    input:
    tuple val(batch_id), val(records), path(fastas)
    output:
    path('results/*'), emit: results
    script:
    def manifest = groovy.json.JsonOutput.toJson(records)
    def helper = helperDigest(['small_task_batch.py', 'assessment_task.py'])
    """
    # Batch helper: ${helper}
    cat > batch.json <<'BATCH_JSON'
    ${manifest}
BATCH_JSON
    python ${projectDir}/small_task_batch.py trna batch.json --cpus ${task.cpus} --checkpoint-dir ${params.working_dir}/trnascan ${params.compact_results ? '--compact' : ''}
    """
}

process FILTER_FASTA {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/genomes_filtered", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [126, 127] ? 'terminate' : 'ignore' }

    input:
    tuple val(id), val(size), path(fasta)

    output:
    tuple val(id), val(size), path("${id}.${size}.fasta"), emit: fasta

    script:
    def min_len = (size == 'full') ? 0 : size
    """
    seqkit seq -j ${task.cpus} -g -m ${min_len} -o ${id}.${size}.fasta ${fasta}
    """
}

process SEQKIT_STAT {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/seqkit/individual", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [126, 127] ? 'terminate' : 'ignore' }

    input:
    tuple val(id), val(size), path(fasta)

    output:
    tuple val(id), val(size), path(fasta), path("${id}.${size}.seqkit.tsv"), emit: stats

    script:
    """
    seqkit stat -j ${task.cpus} -a -T -o ${id}.${size}.seqkit.tsv ${fasta}
    """
}

process PRUNE_FASTA {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/genomes_pruned", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(size), path(fasta, stageAs: "input.fasta")

    output:
    tuple val(id), val(size), path("${id}.${size}.fasta"), optional: true, emit: pruned

    script:
    """
    sum_len=\$(awk '!/^>/ {gsub(/[[:space:]]/, ""); n+=length(\$0)} END {print n+0}' ${fasta})
    if [ "\$sum_len" -gt 0 ]; then
      if [ "\$(readlink -f ${fasta})" != "\$(readlink -f ${id}.${size}.fasta)" ]; then
        cp -f ${fasta} ${id}.${size}.fasta
      fi
    fi
    """
}

process COMBINE_SEQKIT_STATS {
    cpus 1
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/seqkit", mode: 'copy', overwrite: true

    input:
    path(stats)

    output:
    path("seqkit_stats.tsv"), emit: stats

    script:
    """
    if [ -z "${stats}" ]; then
      printf 'file\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\tN50\n' > seqkit_stats.tsv
      exit 0
    fi
    head -n1 ${stats ? stats[0] : '/dev/null'} > seqkit_stats.tsv
    for f in ${stats}; do
      tail -n +2 \$f >> seqkit_stats.tsv
    done
    """
}

// Assessment launch shells check executables after Conda activation.
// Keep these guards outside scientific scripts so successful task caches remain valid.
// beforeScript runs before environment activation and cannot perform this check.
process CHECKM_SINGLE {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python checkm; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    tag "${size}.batch${batch_id}"
    cpus params.checkm_cpus
    maxForks params.checkm_max_forks
    conda "${projectDir}/envs/checkm.yaml"
    publishDir "${params.working_dir}/checkm/individual", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(size), val(batch_id), path(fastas), val(checkm_data_path), val(retryToken)

    output:
    path("${size}.batch${batch_id}.checkm.tsv"), emit: tsv

    path("${size}.batch${batch_id}.assessment.json"), emit: assessment
    path("${size}.batch${batch_id}.assessment.log"), emit: assessment_log

    script:
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    mkdir -p bins out
    for f in ${fastas}; do
      cp "\$f" bins/
    done
    export MPLCONFIGDIR="\$PWD/.mplconfig"
    mkdir -p "\$MPLCONFIGDIR"
    if [ -n "${checkm_data_path}" ]; then
      export CHECKM_DATA_PATH="${checkm_data_path}"
    fi
    checkm lineage_wf bins out -f ${size}.batch${batch_id}.checkm.tsv --tab_table -x fasta -t ${task.cpus} --pplacer_threads ${task.cpus}
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py checkm ${size}.batch${batch_id}.checkm.tsv ${size}.batch${batch_id}.assessment.json assessment.sh --token ${retryToken} --inputs ${fastas}

    """
}

process COMBINE_CHECKM {
    cpus 1
    conda "${projectDir}/envs/checkm.yaml"
    publishDir "${params.working_dir}/checkm", mode: 'copy', overwrite: true

    input:
    path(tsvs)

    output:
    path("checkm_output.tsv"), emit: full

    script:
    """
    python - ${tsvs} <<'PYTHON'
import csv, sys
columns = ['Bin Id', 'Completeness', 'Contamination', 'Strain heterogeneity']
rows = []
for name in sys.argv[1:]:
    with open(name) as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        columns.extend(c for c in (reader.fieldnames or []) if c not in columns)
        rows.extend(reader)
with open('checkm_output.tsv', 'w') as out:
    writer = csv.DictWriter(out, fieldnames=columns, delimiter='\t')
    writer.writeheader(); writer.writerows(rows)
PYTHON
    """
}

process QSCORE {
    cpus 1
    conda "${projectDir}/envs/python_calc_qscore.yaml"
    publishDir "${params.working_dir}/qscore", mode: 'copy', overwrite: true

    input:
    path(checkm_full)
    path(seqkit_stats)
    path(expected_fastas)

    output:
    path("qscore_all.tsv"), emit: all

    script:
    """
    # Helper implementation: ${params.helper_qscore}
    mkdir -p workdir/checkm workdir/seqkit workdir/qscore
    cp ${checkm_full} workdir/checkm/checkm_output.tsv
    cp ${seqkit_stats} workdir/seqkit/seqkit_stats.tsv
    python ${projectDir}/calc_qscore.py workdir ${expected_fastas}
    cp workdir/qscore/qscore_all.tsv .
    """
}

process PREP_GTDBTK_BATCHES {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/manifests", mode: 'copy', overwrite: true

    input:
    tuple path(subset_fastas), val(batch_size)

    output:
    path("gtdbtk_batch_*.list"), emit: manifests

    script:
    def groups = balancedBatches(subset_fastas, batch_size, params.gtdbtk_parallel_batches)
    def commands = groups.collect { fs -> "printf '%s\\n' " + fs.collect { shellQuote(it.name) }.join(' ') + " > gtdbtk_batch_${batchId(fs)}.list" }.join('\n')
    """
    ${commands}
    """
}

process GTDBTK {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python gtdbtk; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    tag "batch${batch_id}"
    cpus params.gtdbtk_cpus
    maxForks params.gtdbtk_max_forks
    conda "${projectDir}/envs/gtdbtk.yaml"
    publishDir "${params.working_dir}/gtdbtk/individual", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(batch_id), path(fasta_manifest), path(batch_fastas), val(gtdbtk_data_path), val(retryToken)

    output:
    path("gtdbtk_${batch_id}"), emit: outdir
    path("marker_alignments_${batch_id}"), emit: alignments

    path("gtdbtk_${batch_id}.assessment.json"), emit: assessment
    path("gtdbtk_${batch_id}.assessment.log"), emit: assessment_log

    script:
    def diskBacked = task.memory && task.memory.toMega() < 163840
    def placementCpus = Math.min(8, task.cpus as Integer)
    def placementArgs = diskBacked ? "--scratch_dir .gtdb-pplacer --pplacer_cpus ${placementCpus}" : ''
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    # Compact helper: ${params.helper_compact}
    mkdir -p genomes
    while IFS= read -r f; do
      [ -n "\$f" ] || continue
      cp "\$f" genomes/
    done < ${fasta_manifest}
    if [ -n "${gtdbtk_data_path}" ]; then
      export GTDBTK_DATA_PATH="${gtdbtk_data_path}"
    fi
    echo "GTDB placement: ${diskBacked ? 'disk-backed, pplacer CPUs=' + placementCpus : 'in-memory'}; memory=${task.memory}"
    mkdir -p .gtdb-pplacer
    trap 'rm -rf -- .gtdb-pplacer' EXIT
    gtdbtk classify_wf ${placementArgs} --place_species --genome_dir genomes --out_dir gtdbtk_${batch_id} --cpus ${task.cpus} -x fasta
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py gtdbtk gtdbtk_${batch_id} gtdbtk_${batch_id}.assessment.json assessment.sh --token ${retryToken} --inputs ${batch_fastas}
    # Preserve user alignments before cleanup: ${params.helper_marker_alignments}
    python ${projectDir}/preserve_marker_alignments.py gtdbtk_${batch_id} marker_alignments_${batch_id}
    ${params.compact_results ? "python " + projectDir + "/compact_task.py gtdbtk_" + batch_id : "true"}
    """
}

process COMBINE_GTDBTK {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/gtdbtk", mode: 'copy', overwrite: true

    input:
    path(gtdbtk_dirs)

    output:
    path("gtdbtk_merged"), emit: outdir

    script:
    """
    mkdir -p gtdbtk_merged/classify

    write_merged_summary() {
      target="\$1"
      shift
      wrote=0
      for f in "\$@"; do
        [ -s "\$f" ] || continue
        if [ "\$wrote" -eq 0 ]; then
          cat "\$f" > "\$target"
          wrote=1
        else
          tail -n +2 "\$f" >> "\$target"
        fi
      done
    }

    bac_files=()
    arc_files=()
    for d in ${gtdbtk_dirs}; do
      [ -f "\$d/classify/gtdbtk.bac120.summary.tsv" ] && bac_files+=("\$d/classify/gtdbtk.bac120.summary.tsv")
      [ -f "\$d/classify/gtdbtk.ar53.summary.tsv" ] && arc_files+=("\$d/classify/gtdbtk.ar53.summary.tsv")
    done

    if [ "\${#bac_files[@]}" -gt 0 ]; then
      write_merged_summary gtdbtk_merged/classify/gtdbtk.bac120.summary.tsv "\${bac_files[@]}"
    fi
    if [ "\${#arc_files[@]}" -gt 0 ]; then
      write_merged_summary gtdbtk_merged/classify/gtdbtk.ar53.summary.tsv "\${arc_files[@]}"
    fi
    """
}

process PARSE_GTDBTK {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"

    input:
    path(gtdbtk_dir)

    output:
    path("gtdbtk_genome_kingdom.tsv"), emit: mapping

    script:
    """
    {
      echo -e "genome_id\tkingdom"
      if [ -f ${gtdbtk_dir}/classify/gtdbtk.bac120.summary.tsv ]; then
        cut -f1 ${gtdbtk_dir}/classify/gtdbtk.bac120.summary.tsv | tail -n +2 | while read -r id; do
          id="\${id%.fasta}"
          printf "%s\tBacteria\n" "\$id"
        done
      fi
      if [ -f ${gtdbtk_dir}/classify/gtdbtk.ar53.summary.tsv ]; then
        cut -f1 ${gtdbtk_dir}/classify/gtdbtk.ar53.summary.tsv | tail -n +2 | while read -r id; do
          id="\${id%.fasta}"
          printf "%s\tArchaea\n" "\$id"
        done
      fi
    } > gtdbtk_genome_kingdom.tsv
    """
}

process BARRNAP {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python barrnap; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    tag "${id}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/barrnap.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(id), path(fasta), path(kingdom_map), val(retryToken)

    output:
    tuple val(id), path("${id}.rRNA.fasta"), path("${id}.rRNA.gff"), path("${id}.rrna.assessment.json"), emit: rrna

    path("${id}.rrna.assessment.json"), emit: assessment
    path("${id}.rrna.assessment.log"), emit: assessment_log

    script:
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    kingdom=\$(awk -v id="${id}" '(\$1==id){print \$2; exit}' ${kingdom_map})
    if [ -z "\$kingdom" ]; then
      kingdom="Bacteria"
    fi
    k=\$([ "\$kingdom" = "Archaea" ] && echo "arc" || echo "bac")
    barrnap --kingdom \${k} --threads ${task.cpus} --outseq ${id}.rRNA.fasta ${fasta} > ${id}.rRNA.gff
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py rrna ${id}.rRNA.gff ${id}.rrna.assessment.json assessment.sh --token ${retryToken} --inputs ${fasta}

    """
}

process BLAST_RRNA {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python makeblastdb blastn; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    tag "${id}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/blast.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(id), path(rrna_fasta), path(rrna_gff), path(rrna_status), val(retryToken)

    output:
    path("${id}.blastout"), emit: blast

    path("${id}.rrna_check.assessment.json"), emit: assessment
    path("${id}.rrna_check.assessment.log"), emit: assessment_log

    script:
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    if [ -s ${rrna_fasta} ]; then
      makeblastdb -in ${rrna_fasta} -dbtype nucl -out ${id}
      blastn -query ${rrna_fasta} -db ${id} -outfmt 6 -out ${id}.blastout -max_target_seqs 10
    else
      : > ${id}.blastout
    fi
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py rrna_check ${id}.blastout ${id}.rrna_check.assessment.json assessment.sh --token ${retryToken} --inputs ${id}.fasta --upstream ${rrna_status}

    """
}

process COMPARE_BARRNAP {
    cpus 1
    conda "${projectDir}/envs/python_compare_barrnap.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true

    input:
    path(blastouts)
    path(rrna_fastas)
    path(rrna_gffs)

    output:
    path("barrnap_blast_pass.tsv"), emit: pass
    path("barrnap_subunit_counts.tsv"), emit: subunits
    path("barrnap_blast_raw.tsv"), emit: raw
    path("barrnap_gff_table.tsv"), emit: gff

    script:
    """
    # Helper implementation: ${params.helper_compare_barrnap}
    files=( ${blastouts} )
    if [ "\${#files[@]}" -eq 0 ]; then
      echo -e "Bin Id\tpident\tbitscore\tpass_BARRNAP" > barrnap_blast_pass.tsv
      echo -e "genome_id\tname\tsubunit_count" > barrnap_subunit_counts.tsv
      echo -e "Bin Id\tqaccver\tsaccver\tpident\tlength\tmismatch\tgapopen\tqstart\tqend\tsstart\tsend\tevalue\tbitscore" > barrnap_blast_raw.tsv
      echo -e "seqid\tsource\ttype\tstart\tend\tscore\tstrand\tphase\tattributes\tgenome_id\tname\tproduct\tprod_mod" > barrnap_gff_table.tsv
      exit 0
    fi
    python ${projectDir}/compare_barrnap.py .
    """
}

process DEDUPE {
    cpus 1
    conda "${projectDir}/envs/python_dedupe.yaml"
    publishDir "${params.working_dir}/dedupe", mode: 'copy', overwrite: true

    input:
    path(qscore_all)
    path(barrnap_pass)
    path(gtdbtk_dir)

    output:
    path("dedupe_assessed.tsv"), emit: assessed
    path("dedupe_all.tsv"), emit: all

    script:
    """
    # Helper implementation: ${params.helper_dedupe}
    mkdir -p workdir/qscore workdir/barrnap workdir/gtdbtk/classify workdir/dedupe
    cp ${qscore_all} workdir/qscore/qscore_all.tsv
    cp ${barrnap_pass} workdir/barrnap/barrnap_blast_pass.tsv
    cp -r ${gtdbtk_dir}/classify/*.summary.tsv workdir/gtdbtk/classify/ 2>/dev/null || true
    python ${projectDir}/dedupe_genomes.py workdir
    cp workdir/dedupe/dedupe_assessed.tsv .
    cp workdir/dedupe/dedupe_all.tsv .
    """
}

process CP_DEDUPE {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/dedupe/fasta", mode: 'copy', overwrite: true

    input:
    path(dedupe_assessed)
    path(subset_fastas, stageAs: "subset/*")

    output:
    path("*.fasta"), optional: true, emit: fasta
    path("cp_dedupe_audit.tsv"), emit: audit

    script:
    """
    printf "requested_id\\tmatched_fasta\\tstatus\\n" > cp_dedupe_audit.tsv
    requested=0
    copied=0
    declare -A subset_map
    while IFS= read -r f; do
      [ -n "\$f" ] || continue
      base=\$(basename "\$f" .fasta)
      subset_map["\$base"]="\$f"
    done < <(printf "%s\\n" ${subset_fastas})
    while read -r g; do
      [ -z "\$g" ] && continue
      requested=\$((requested + 1))
      found="\${subset_map[\$g]:-}"
      dest="\${g}.fasta"
      if [ -n "\$found" ]; then
        cp "\$found" "\$dest"
        copied=\$((copied + 1))
        printf "%s\\t%s\\tcopied\\n" "\$g" "\$found" >> cp_dedupe_audit.tsv
      else
        printf "%s\\t\\tmissing\\n" "\$g" >> cp_dedupe_audit.tsv
      fi
    done < <(tail -n +2 ${dedupe_assessed} | cut -f2)
    if [ "\$requested" -gt 0 ] && [ "\$copied" -ne "\$requested" ]; then
      echo "CP_DEDUPE could not resolve every requested genome" >&2
      exit 1
    fi
    """
}

process PARSE_DEDUPE_KINGDOM {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"

    input:
    path(dedupe_assessed)

    output:
    path("dedupe_kingdom.tsv"), emit: mapping

    script:
    """
    awk -F '\t' 'NR==1{next} { if (\$3 ~ /Archaea/) print \$2"\tArchaea"; else if (\$3 ~ /Bacteria/) print \$2"\tBacteria"; else print \$2"\tUnknown"; }' ${dedupe_assessed} > dedupe_kingdom.tsv
    """
}

process TRNASCAN {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python tRNAscan-SE; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    tag "${id}"
    cpus params.trnascan_cpus
    maxForks params.trnascan_max_forks
    conda "${projectDir}/envs/trnascan.yaml"
    publishDir "${params.working_dir}/trnascan", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(id), val(kingdom), path(fasta), val(retryToken)

    output:
    path("${id}.gff"), emit: gff

    path("${id}.trna.assessment.json"), emit: assessment
    path("${id}.trna.assessment.log"), emit: assessment_log

    script:
    def flag = kingdom == 'Archaea' ? '-A' : (kingdom == 'Bacteria' ? '-B' : (kingdom == 'Eukaryota' ? '-E' : '-G'))
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    tRNAscan-SE -q -Q ${flag} \
      -o ${id}.output.txt \
      -m ${id}.stats.txt \
      -b ${id}.bed \
      -j ${id}.gff \
      -a ${id}.trna.fasta \
      -l ${id}.log \
      --thread ${task.cpus} ${fasta}
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py trna ${id}.gff ${id}.trna.assessment.json assessment.sh --token ${retryToken} --inputs ${fasta}

    """
}

process TRNASCAN_PARSE {
    cpus 1
    conda "${projectDir}/envs/python_trnascan_parse.yaml"
    publishDir "${params.working_dir}/trnascan", mode: 'copy', overwrite: true

    input:
    path(gffs)

    output:
    path("trnascan_trna_counts.tsv"), emit: counts
    path("trnascan_gff_table.tsv"), emit: table

    script:
    """
    # Helper implementation: ${params.helper_trnascan_parse}
    files=( ${gffs} )
    if [ "\${#files[@]}" -eq 0 ]; then
      echo -e "genome_id\ttrna_total\ttrna_unique" > trnascan_trna_counts.tsv
      echo -e "seqid\tsource\ttype\tstart\tend\tscore\tstrand\tphase\tattributes\tgenome_id\tisotype" > trnascan_gff_table.tsv
      exit 0
    fi
    python ${projectDir}/trnascan_parse.py .
    """
}

process GUNC {
    shell(['/bin/bash', '-euo', 'pipefail', '-c', shellQuote('for executable in python gunc; do command -v "$executable" >/dev/null 2>&1 || { echo "Missing required executable: $executable (task environment)" >&2; exit 127; }; done; exec /bin/bash -euo pipefail "$@"'), '--'])
    cpus params.gunc_cpus
    conda "${projectDir}/envs/gunc.yaml"
    publishDir "${params.working_dir}/gunc", mode: 'copy', overwrite: true

    errorStrategy { task.exitStatus in [137, 143] && task.attempt <= 1 ? 'retry' : 'terminate' }
    maxRetries 1

    input:
    tuple val(batch_id), path(selected_fastas), val(gunc_db), val(retryToken)

    output:
    path("gunc_${batch_id}"), emit: outdir

    path("gunc_${batch_id}.assessment.json"), emit: assessment
    path("gunc_${batch_id}.assessment.log"), emit: assessment_log

    script:
    """
    # Assessment runner: ${params.helper_assessment}; retry token: ${retryToken}
    cat > assessment.sh <<'ASSESSMENT_SCRIPT'

    # Compact helper: ${params.helper_compact}
    if [ -z "${gunc_db}" ]; then
      echo "params.gunc_db is required" >&2
      exit 1
    fi
    mkdir -p gunc_${batch_id} dedupe_fasta
    fastas=( ${selected_fastas} )
    if [ "\${#fastas[@]}" -eq 0 ]; then
      echo "No selected FASTA files supplied; skipping."
      exit 0
    fi
    for f in "\${fastas[@]}"; do
      cp "\$f" dedupe_fasta/
    done
    gunc run -r ${gunc_db} -d dedupe_fasta -o gunc_${batch_id} -e .fasta -t ${task.cpus}
ASSESSMENT_SCRIPT
    python ${projectDir}/assessment_task.py gunc gunc_${batch_id} gunc_${batch_id}.assessment.json assessment.sh --token ${retryToken} --inputs ${selected_fastas}
    ${params.compact_results ? "python " + projectDir + "/compact_task.py gunc_" + batch_id : "true"}
    """
}

process COMBINE_GUNC {
    cpus 1
    conda "${projectDir}/envs/python_utils.yaml"
    publishDir "${params.working_dir}/gunc", mode: 'copy', overwrite: true
    input:
    path(directories)
    output:
    path("gunc"), emit: outdir
    script:
    """
    # Merge helper: ${params.helper_merge_gunc}
    python ${projectDir}/merge_gunc_batches.py gunc ${directories}
    """
}

def databaseOptions(name) {
    def flags = " --attempts ${params.db_attempts}"
    if (params.db_download_only) flags += ' --download-only'
    if (params.db_extract_only) flags += ' --extract-only'
    if (params.db_urls[name]) flags += ' --url ' + shellQuote(params.db_urls[name])
    return flags
}

process SETUP_GTDBTK_DB {
    cpus params.reference_cpus
    conda "${projectDir}/envs/reference_setup.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(gtdbtk_data_path), val(gtdbtk_package_url), val(allow_bootstrap)

    output:
    path("gtdbtk_reference_ready.tsv"), emit: ready

    script:
    """
    # Helper implementation: ${params.helper_setup_gtdbtk_db}
    python ${projectDir}/database_install.py gtdbtk "${gtdbtk_data_path}" ${task.cpus} ${databaseOptions('gtdbtk')}
    printf 'reference\\tpath\\tstatus\\ngtdbtk\\t${gtdbtk_data_path}\\t${params.db_download_only ? 'downloaded' : 'ready'}\\n' > gtdbtk_reference_ready.tsv
    """
}

process SETUP_GUNC_DB {
    cpus params.reference_cpus
    conda "${projectDir}/envs/reference_setup.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(gunc_db), val(allow_bootstrap)

    output:
    path("gunc_reference_ready.tsv"), emit: ready

    script:
    """
    # Helper implementation: ${params.helper_setup_gunc_db}
    python ${projectDir}/database_install.py gunc "${new File(gunc_db.toString()).getParent()}" ${task.cpus} ${databaseOptions('gunc')}
    printf 'reference\\tpath\\tstatus\\ngunc\\t${new File(gunc_db.toString()).getParent()}\\t${params.db_download_only ? 'downloaded' : 'ready'}\\n' > gunc_reference_ready.tsv
    """
}

process SETUP_CHECKM_DB {
    cpus params.reference_cpus
    conda "${projectDir}/envs/reference_setup.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(checkm_data_path), val(allow_bootstrap)

    output:
    path("checkm_reference_ready.tsv"), emit: ready

    script:
    """
    # Helper implementation: ${params.helper_setup_checkm_db}
    python ${projectDir}/database_install.py checkm "${checkm_data_path}" ${task.cpus} ${databaseOptions('checkm')}
    printf 'reference\\tpath\\tstatus\\ncheckm\\t${checkm_data_path}\\t${params.db_download_only ? 'downloaded' : 'ready'}\\n' > checkm_reference_ready.tsv
    """
}

process MERGE_MASTER {
    cpus 1
    conda "${projectDir}/envs/merge_master.yaml"
    publishDir "${params.working_dir}", mode: 'copy', overwrite: true

    input:
    path(dedupe_assessed)
    path(barrnap_subunits)
    path(trna_counts)
    path(gunc_dir)
    path(assessment_receipts)

    output:
    path("Master_genome_QC.tsv"), emit: master
    path("Selected_genome_QC.tsv"), emit: selected
    path("Assessment_status.tsv"), emit: assessment_status
    path("merge_work/gunc"), emit: gunc_data
    path("Master_genome_QC_All.pdf"), emit: plot1
    path("Master_genome_QC_All.png"), emit: plot2
    path("Master_genome_QC_MQHQ_16S.pdf"), emit: plot3
    path("Master_genome_QC_MQHQ_16S.png"), emit: plot4
    path("Master_genome_QC_MQHQ_Phylum.pdf"), emit: plot5
    path("Master_genome_QC_MQHQ_Phylum.png"), emit: plot6

    script:
    """
    # Helper implementation: ${params.helper_merge_master}
    mkdir -p merge_work/barrnap merge_work/dedupe merge_work/trnascan merge_work/assessments
    for f in ${assessment_receipts}; do cp "\$f" merge_work/assessments/; done
    cp ${barrnap_subunits} merge_work/barrnap/barrnap_subunit_counts.tsv
    cp ${dedupe_assessed} merge_work/dedupe/dedupe_assessed.tsv
    cp ${trna_counts} merge_work/trnascan/trnascan_trna_counts.tsv
    mkdir -p merge_work/gunc
    python ${projectDir}/merge_gunc_batches.py merge_work/gunc ${gunc_dir}
    python ${projectDir}/merge_master.py merge_work ${params.min_genome_size} ${params.max_genome_size} ${params.completeness} ${params.contamination} ${params.qscore_min} ${params.require_rrna_consistency}
    cp merge_work/Master_genome_QC.tsv .
    cp merge_work/Selected_genome_QC.tsv .
    cp merge_work/Assessment_status.tsv .
    cp merge_work/Master_genome_QC_All.pdf .
    cp merge_work/Master_genome_QC_All.png .
    cp merge_work/Master_genome_QC_MQHQ_16S.pdf .
    cp merge_work/Master_genome_QC_MQHQ_16S.png .
    cp merge_work/Master_genome_QC_MQHQ_Phylum.pdf .
    cp merge_work/Master_genome_QC_MQHQ_Phylum.png .
    """
}


process GENOME_QUALITY_ATLAS {
    cpus params.atlas_cpus
    conda "${projectDir}/envs/genome_quality_atlas.yaml"
    publishDir "${params.working_dir}", mode: 'copy', overwrite: true

    input:
    path(master_tsv)
    path(gunc_dir)
    path(atlas_fastas)
    path(ani_results)
    path(tree_results)

    output:
    path("genome_quality_atlas"), emit: outdir

    script:
    def atlasPrefix = params.atlas_prefix?.toString()?.trim()
    def atlasGroupColumn = params.atlas_group_column?.toString()?.trim()
    def atlasSampleColumn = params.atlas_sample_column?.toString()?.trim()
    def atlasMatchedSamplesOnly = params.atlas_matched_samples_only ? true : false
    def atlasTopNGroups = params.atlas_top_n_groups as Integer
    def atlasCompareColumns = normalizeDelimitedList(params.atlas_compare_columns)
    def atlasCompareMap = resolveAbsolutePath(params.atlas_compare_map)
    def atlasCompareMapKey = params.atlas_compare_map_key?.toString()?.trim()
    def atlasAniCompareColumns = normalizeDelimitedList(params.atlas_ani_compare_columns)
    def atlasAniResults = resolveAbsolutePath(params.atlas_ani_results)
    def atlasAniGenomeDir = resolveAbsolutePath(params.atlas_ani_genome_dir)
    if (!atlasAniGenomeDir) {
        atlasAniGenomeDir = "atlas_genomes"
    }
    def atlasAniFastaColumn = params.atlas_ani_fasta_column?.toString()?.trim()
    def atlasAniFastaExts = params.atlas_ani_fasta_exts?.toString()?.trim()
    def atlasAniThreshold = params.atlas_ani_threshold
    def atlasAniAfThreshold = params.atlas_ani_af_threshold
    def atlasAniThreads = Math.max(1, Math.min(task.cpus as Integer, params.atlas_ani_threads as Integer))
    def atlasAniTopOverlaps = params.atlas_ani_top_overlaps as Integer
    def compareArgLines = atlasCompareColumns.collect { "    cmd+=( --compare-column ${shellQuote(it)} )" }.join("\n")
    def aniCompareArgLines = atlasAniCompareColumns.collect { "    cmd+=( --ani-compare-column ${shellQuote(it)} )" }.join("\n")
    """
    # Helper implementation: ${params.helper_genome_quality_atlas}
    test -d ${gunc_dir}
    mkdir -p atlas_genomes
    for f in ${atlas_fastas}; do cp "\$f" atlas_genomes/; done
    cmd=(
      python ${projectDir}/genome_quality_atlas.py
      ${master_tsv}
      -o genome_quality_atlas
      --top-n-groups ${atlasTopNGroups}
      --ani-threshold ${atlasAniThreshold}
      --ani-af-threshold ${atlasAniAfThreshold}
      --ani-threads ${atlasAniThreads}
      --ani-top-overlaps ${atlasAniTopOverlaps}
    )
    ${atlasPrefix ? "cmd+=( --prefix ${shellQuote(atlasPrefix)} )" : ""}
    ${atlasGroupColumn ? "cmd+=( --group-column ${shellQuote(atlasGroupColumn)} )" : ""}
    ${atlasSampleColumn ? "cmd+=( --sample-column ${shellQuote(atlasSampleColumn)} )" : ""}
    ${atlasMatchedSamplesOnly ? "cmd+=( --matched-samples-only )" : ""}
    ${atlasCompareMap ? "cmd+=( --compare-map ${shellQuote(atlasCompareMap)} )" : ""}
    ${atlasCompareMapKey ? "cmd+=( --compare-map-key ${shellQuote(atlasCompareMapKey)} )" : ""}
    ${atlasAniResults ? "cmd+=( --ani-results ${shellQuote(atlasAniResults)} )" : ""}
    ${atlasAniGenomeDir ? "cmd+=( --ani-genome-dir ${shellQuote(atlasAniGenomeDir)} )" : ""}
    ${atlasAniFastaColumn ? "cmd+=( --ani-fasta-column ${shellQuote(atlasAniFastaColumn)} )" : ""}
    ${atlasAniFastaExts ? "cmd+=( --ani-fasta-exts ${shellQuote(atlasAniFastaExts)} )" : ""}
${compareArgLines}
${aniCompareArgLines}
    printf 'Running:'
    printf ' %q' "\${cmd[@]}"
    printf '\\n'
    "\${cmd[@]}"
    ${params.cohort_ani_enabled ? "cp -RL " + ani_results + " genome_quality_atlas/ani" : "true"}
    ${params.cohort_phylogeny_enabled ? "cp -RL " + tree_results + " genome_quality_atlas/phylogeny" : "true"}
    """
}

process COHORT_ANI {
    cpus params.atlas_cpus
    conda "${projectDir}/envs/genome_quality_atlas.yaml"
    input:
    path(master)
    path(fastas)
    output:
    path("ani"), emit: results
    script:
    """
    # Implementation: ${params.helper_cohort}
    mkdir genomes
    for f in ${fastas}; do cp "\$f" genomes/; done
    python ${projectDir}/cohort_phylogeny.py ani --master ${master} --genomes genomes \
      --output ani --threads ${task.cpus} --ani ${params.atlas_ani_threshold} --af ${params.atlas_ani_af_threshold} ${params.cohort_ani_containment ? "" : "--no-containment"}
    """
}

process COHORT_PHYLOGENY {
    cpus params.atlas_cpus
    conda "${projectDir}/envs/cohort_phylogeny.yaml"
    input:
    path(master)
    path(alignments)
    path(ani)
    output:
    path("phylogeny"), emit: results
    script:
    def clusterArg = params.cohort_ani_enabled ? "--clusters ${ani}/ani_clusters.tsv" : ''
    """
    # Implementation: ${params.helper_cohort}
    python ${projectDir}/cohort_phylogeny.py phylogeny --master ${master} \
      --alignments ${alignments} ${clusterArg} --output phylogeny --threads ${task.cpus}
    """
}

workflow COMPLETE {
    main:
        def resolvedRefDir = resolveAbsolutePath(params.ref_dir)
        def resolvedCheckmData = resolveAbsolutePath(params.checkm_data_path)
        def resolvedGTDBTK = resolveAbsolutePath(params.gtdbtk_data_path)
        def resolvedGuncDb = resolveAbsolutePath(params.gunc_db)
        def bootstrapReferences = resolvedRefDir ? true : false

        if (resolvedRefDir) {
            if (!resolvedCheckmData) {
                resolvedCheckmData = new File(resolvedRefDir, "checkm_data").getAbsolutePath()
            }
            if (!resolvedGTDBTK) {
                resolvedGTDBTK = new File(resolvedRefDir, "gtdbtk_r232").getAbsolutePath()
            }
            if (!resolvedGuncDb) {
                resolvedGuncDb = new File(new File(resolvedRefDir, "gunc_db"), "gunc_db_progenomes2.1.dmnd").getAbsolutePath()
            }
        }

        if (!resolvedGTDBTK) {
            throw new IllegalArgumentException("GTDB-Tk database path is required. Provide --ref_dir or --gtdbtk_data_path.")
        }
        if (!resolvedGuncDb) {
            throw new IllegalArgumentException("GUNC database path is required. Provide --ref_dir or --gunc_db.")
        }
        if (!bootstrapReferences) {
            if (resolvedCheckmData && !checkmDataLooksValid(resolvedCheckmData)) {
                throw new IllegalArgumentException("CheckM data path is missing or empty: ${resolvedCheckmData}")
            }
            if (!gtdbtkDbLooksValid(resolvedGTDBTK)) {
                throw new IllegalArgumentException("GTDB-Tk requires a complete R232 data path; missing, incompatible or incomplete: ${resolvedGTDBTK}")
            }
            if (!guncDbLooksValid(resolvedGuncDb)) {
                throw new IllegalArgumentException("GUNC database file is missing or empty: ${resolvedGuncDb}")
            }
        }

        if (params.debug) {
            println "[DEBUG] resolved ref_dir=${resolvedRefDir}"
            println "[DEBUG] resolved checkm_data_path=${resolvedCheckmData}"
            println "[DEBUG] resolved gtdbtk_data_path=${resolvedGTDBTK}"
            println "[DEBUG] resolved gunc_db=${resolvedGuncDb}"
            println "[DEBUG] bootstrap_references=${bootstrapReferences}"
        }

        def fastaDir = new File(params.fasta_dir.toString())
        if (!fastaDir.isAbsolute()) {
            fastaDir = new File(projectDir.toString(), params.fasta_dir.toString())
        }
        def fastaFiles = []
        try {
            def entries = java.nio.file.Files.newDirectoryStream(fastaDir.toPath())
            entries.each { p ->
                def name = p.getFileName().toString()
                def lower = name.toLowerCase()
                def isMatch = lower.endsWith('.fa') || lower.endsWith('.fna') || lower.endsWith('.fasta') ||
                              lower.endsWith('.fa.gz') || lower.endsWith('.fna.gz') || lower.endsWith('.fasta.gz')
                if (params.debug) {
                    println "[DEBUG] dir entry: ${p} match=${isMatch}"
                }
            if (isMatch) {
                fastaFiles << file(p.toString())
            }
        }
        } catch (Exception e) {
            throw new IllegalStateException("Failed to list ${fastaDir}: ${e.class.name} ${e.message}")
        }
        if (!fastaFiles) {
            throw new IllegalStateException("No FASTA files found in ${fastaDir}")
        }
        fastaFiles = fastaFiles.sort { it.toString() }
        if (params.debug) {
            println "[DEBUG] projectDir=${projectDir}"
            println "[DEBUG] fasta_dir param=${params.fasta_dir}"
            println "[DEBUG] resolved fastaDir=${fastaDir}"
            println "[DEBUG] filter_size values=${filterSizes()}"
            fastaFiles.each { println "[DEBUG] genome file: ${it}" }
        }

        def raw_list = []
        fastaFiles.each { f ->
            def nfFile = file(f.toString())
            def id = genomeId(nfFile)
            filterSizes().each { size ->
                raw_list << tuple(id, size.toString(), nfFile)
            }
        }
        raw_genomes = Channel.fromList(raw_list)
        if (params.debug) {
            raw_genomes = raw_genomes.view { "[DEBUG] raw tuple: ${it}" }
        }

        def checkm_ready = null
        if (bootstrapReferences && resolvedCheckmData) {
            def checkmRefDir = resolvedRefDir ?: new File(resolvedCheckmData).getParent()
            checkm_ready = SETUP_CHECKM_DB(
                Channel.of([checkmRefDir, resolvedCheckmData, bootstrapReferences])
            ).ready
        }
        def gtdb_ready = bootstrapReferences ?
            SETUP_GTDBTK_DB(
                Channel.of([resolvedRefDir ?: new File(resolvedGTDBTK).getParent(), resolvedGTDBTK, params.gtdbtk_package_url.toString(), bootstrapReferences])
            ).ready :
            Channel.of("validated")
        def gunc_ready = bootstrapReferences ?
            SETUP_GUNC_DB(
                Channel.of([resolvedRefDir ?: new File(resolvedGuncDb).getParent(), resolvedGuncDb, bootstrapReferences])
            ).ready :
            Channel.of("validated")

        preprocess_input = raw_genomes.filter { id, size, fasta -> !rescueReady('preprocess', id + '.' + size) }.toSortedList().flatMap { rows ->
            rows.groupBy { it[1] }.values().collectMany { group ->
                smallBatches(group, 2).collect { batch ->
                    tuple(batchId(batch.collect { it[2] }) + '-' + batch[0][1],
                          batch.collect { [it[0], it[1], it[2].name, Math.max(assessmentToken('genomes_pruned', "${it[0]}.${it[1]}.preprocess").toInteger(), assessmentToken('genomes_pruned', "${it[0]}.${it[1]}.sequence_stats").toInteger()).toString()] }, batch.collect { it[2] })
                }
            }
        }
        preprocessed = PREPROCESS_BATCH(preprocess_input).results.map { fs -> normalizeToList(fs)[0].parent }
        pruned = preprocessed.flatMap { dir ->
            file("${dir}/*.fasta").with { value -> normalizeToList(value) }.findAll { it.exists() }.collect { f ->
                def stem = f.baseName
                def at = stem.lastIndexOf('.')
                tuple(stem.substring(0, at), stem.substring(at + 1), f)
            }
        }
        pruned = pruned.mix(Channel.fromList(rescuePaths('preprocess')).map { f ->
            def at = f.baseName.lastIndexOf('.')
            tuple(f.baseName.substring(0, at), f.baseName.substring(at + 1), f)
        })
        stats = preprocessed.flatMap { dir -> normalizeToList(file("${dir}/*.seqkit.tsv")).findAll { it.exists() } }
        seqkit_combined = COMBINE_SEQKIT_STATS(stats.mix(Channel.fromList(rescuePaths('sequence_stats'))).toSortedList()).stats

        def checkmBatchSize = Math.max(1, params.checkm_batch_size as Integer)
        checkm_batches = pruned.filter { id, size, fasta -> !rescueReady('checkm', fasta.baseName) }
            .map { id, size, fasta -> tuple(size, tuple(id, size, fasta)) }
            .groupTuple()
            .flatMap { size, records ->
                def out = []
                def grouped = balancedBatches(records.collect { it[2] }, checkmBatchSize, params.checkm_parallel_batches)
                grouped.each { fastas -> out << tuple(size, batchId(fastas), fastas) }
                return out
            }

        checkm_input = checkm_ready ?
            checkm_batches.combine(checkm_ready).map { size, batch_id, fastas, ready -> tuple(size, batch_id, fastas, resolvedCheckmData) } :
            checkm_batches.map { size, batch_id, fastas -> tuple(size, batch_id, fastas, resolvedCheckmData) }

        checkm = CHECKM_SINGLE(checkm_input.map { row -> new ArrayList(row) + [assessmentToken('checkm/individual', "${row[0]}.batch${row[1]}")] }).tsv
        checkm_combined = COMBINE_CHECKM(checkm.mix(Channel.fromList(rescuePaths('checkm'))).toSortedList())

        qscore = QSCORE(checkm_combined.full, seqkit_combined, pruned.map { it[2] }.toSortedList())
        qscore_for_dedupe = qscore.all.map { it }
        // All nonempty genome variants reach taxonomy and RNA assessment.
        subset = pruned.map { id, size, fasta -> fasta }
        subset_for_gtdbtk = subset.map { it }
        subset_for_ids = subset.map { it }
        subset_for_dedupe = subset.map { it }
        def gtdbtkBatchSize = Math.max(1, params.gtdbtk_batch_size as Integer)
        selected_files = subset_for_gtdbtk.toSortedList()
        gtdbtk_manifest_input = selected_files.map { files -> tuple(files.findAll { !rescueReady('gtdbtk', it.baseName) }, gtdbtkBatchSize) }.filter { fs, size -> !fs.isEmpty() }
        gtdbtk_manifests = PREP_GTDBTK_BATCHES(gtdbtk_manifest_input).manifests
            .flatMap { manifests -> normalizeToList(manifests) }

        gtdbtk_input = bootstrapReferences ?
            gtdb_ready
                .combine(gtdbtk_manifests)
                .map { ready, fasta_manifest ->
                    def batchId = fasta_manifest.baseName.replaceFirst(/^gtdbtk_batch_/, "")
                    tuple(batchId, fasta_manifest, resolvedGTDBTK)
                } :
            gtdbtk_manifests.map { fasta_manifest ->
                def batchId = fasta_manifest.baseName.replaceFirst(/^gtdbtk_batch_/, "")
                tuple(batchId, fasta_manifest, resolvedGTDBTK)
            }

        gtdbtk_staged = gtdbtk_input.combine(selected_files.map { [it] }).map { batchId, manifest, db, fastas ->
            def names = manifest.toFile().readLines().toSet()
            tuple(batchId, manifest, fastas.findAll { names.contains(it.name) }, db)
        }
        gtdbtk_batches_out = GTDBTK(gtdbtk_staged.map { row -> new ArrayList(row) + [assessmentToken('gtdbtk/individual', "gtdbtk_${row[0]}")] }).outdir
        gtdbtk = COMBINE_GTDBTK(gtdbtk_batches_out.mix(Channel.fromList(rescuePaths('gtdbtk').findAll { it.name.startsWith('gtdbtk_') })).toSortedList()).outdir
        gtdbtk_map = PARSE_GTDBTK(gtdbtk).mapping

        subset_ids = subset_for_ids.map { f -> tuple(f.baseName, f) }
        barrnap_in = subset_ids.combine(gtdbtk_map)

        rrna_input = barrnap_in.filter { id, fasta, mapping -> !(rescueReady('rrna', id) && rescueReady('rrna_check', id)) }.toSortedList().flatMap { rows ->
            smallBatches(rows, 1).collect { batch ->
                def kingdoms = batch[0][2].toFile().readLines().findAll { it.trim() }.collectEntries { line ->
                    def fields = line.split('\t'); [(fields[0]): fields[1]]
                }
                tuple(batchId(batch.collect { it[1] }), batch.collect { row ->
                    [row[0], kingdoms.get(row[0], 'Bacteria'), row[1].name,
                     assessmentToken('barrnap', "${row[0]}.rrna"), assessmentToken('barrnap', "${row[0]}.rrna_check")]
                }, batch.collect { it[1] })
            }
        }
        rrna = RRNA_BATCH(rrna_input).results.map { fs -> normalizeToList(fs)[0].parent }
        rrna_fastas = rrna.flatMap { dir -> normalizeToList(file("${dir}/*.rRNA.fasta")) }.mix(Channel.fromList(rescuePaths('rrna', '.rRNA.fasta').findAll { rescueReady('rrna', it.name.replace('.rRNA.fasta', '')) && rescueReady('rrna_check', it.name.replace('.rRNA.fasta', '')) }))
        rrna_gffs = rrna.flatMap { dir -> normalizeToList(file("${dir}/*.rRNA.gff")) }.mix(Channel.fromList(rescuePaths('rrna', '.rRNA.gff').findAll { rescueReady('rrna', it.name.replace('.rRNA.gff', '')) && rescueReady('rrna_check', it.name.replace('.rRNA.gff', '')) }))
        blast = rrna.flatMap { dir -> normalizeToList(file("${dir}/*.blastout")) }.mix(Channel.fromList(rescuePaths('rrna_check', '.blastout').findAll { rescueReady('rrna', it.name.replace('.blastout', '')) && rescueReady('rrna_check', it.name.replace('.blastout', '')) }))
        barblast = COMPARE_BARRNAP(blast.toSortedList(), rrna_fastas.toSortedList(), rrna_gffs.toSortedList())

        dedupe = DEDUPE(qscore_for_dedupe, barblast.pass, gtdbtk)
        dedupe_for_cp = dedupe.assessed.map { it }
        dedupe_for_parse = dedupe.assessed.map { it }
        dedupe_for_merge = dedupe.assessed.map { it }

        dedupe_fasta = CP_DEDUPE(dedupe_for_cp, selected_files).fasta.flatMap { normalizeToList(it) }

        dedupe_map = PARSE_DEDUPE_KINGDOM(dedupe_for_parse).mapping
        dedupe_kingdom = dedupe_map.flatMap { f ->
            def out = []
            f.toFile().readLines().findAll { it?.trim() }.each { line ->
                def parts = line.split('\t')
                out << tuple(parts[0], parts[1])
            }
            out
        }

        trna_in = subset_ids.join(dedupe_kingdom)
            .map { row ->
                def id = row[0]
                def fasta = row[1]
                def kingdom = (row.size() > 2 && row[2]) ? row[2] : "Bacteria"
                tuple(id, kingdom, fasta)
            }

        trna_input = trna_in.filter { id, kingdom, fasta -> !rescueReady('trna', id) }.toSortedList().flatMap { rows -> smallBatches(rows, 2).collect { batch ->
            tuple(batchId(batch.collect { it[2] }), batch.collect { row ->
                [row[0], row[1], row[2].name, assessmentToken('trnascan', "${row[0]}.trna")]
            }, batch.collect { it[2] })
        } }
        trna_results = TRNA_BATCH(trna_input).results.map { fs -> normalizeToList(fs)[0].parent }
        trna_gffs = trna_results.flatMap { dir -> normalizeToList(file("${dir}/*.gff")) }

        trna_counts = TRNASCAN_PARSE(trna_gffs.mix(Channel.fromList(rescuePaths('trna'))).toSortedList())

        gunc_input = dedupe_fasta.filter { !rescueReady('gunc', it.baseName) }.toSortedList().flatMap { fs -> balancedBatches(fs, params.gunc_batch_size, params.gunc_parallel_batches).collect { group -> tuple(batchId(group), group) } }.combine(gunc_ready).map { id, fastas, ready -> tuple(id, fastas, resolvedGuncDb) }
        gunc_batches = GUNC(gunc_input.map { row -> new ArrayList(row) + [assessmentToken('gunc', "gunc_${row[0]}")] }).outdir.toSortedList()
        gunc = COMBINE_GUNC(gunc_batches.map { it + rescuePaths('gunc') }).outdir

        rna_receipts = rrna.mix(trna_results).flatMap { dir -> normalizeToList(file("${dir}/*.assessment.json")) }
        reused_receipts = ['checkm', 'gtdbtk', 'trna', 'gunc'].collectMany { rescueReceipts(it) }
        reused_receipts += ['rrna', 'rrna_check'].collectMany { stage -> rescueReceipts(stage).findAll { f ->
            def r = new groovy.json.JsonSlurper().parse(f.toFile())
            rescueReady('rrna', r.genomes[0]) && rescueReady('rrna_check', r.genomes[0])
        } }
        receipts = CHECKM_SINGLE.out.assessment.mix(GTDBTK.out.assessment, rna_receipts, GUNC.out.assessment, Channel.fromList(reused_receipts)).toSortedList()
        master = MERGE_MASTER(dedupe_for_merge, barblast.subunits, trna_counts.counts, gunc, receipts)
        ani = params.cohort_ani_enabled ? COHORT_ANI(master.master, dedupe_fasta.toSortedList()).results : Channel.value([])
        phylogeny = params.cohort_phylogeny_enabled ?
            COHORT_PHYLOGENY(master.master, GTDBTK.out.alignments.mix(Channel.fromList(rescuePaths('gtdbtk').findAll { it.name.startsWith('marker_alignments_') })).toSortedList(), ani).results : Channel.value([])
        atlas = params.atlas_enabled ?
            GENOME_QUALITY_ATLAS(master.master, master.gunc_data, dedupe_fasta.toSortedList(), ani, phylogeny).outdir :
            Channel.empty()

    emit:
        master = master.master
        atlas = atlas
}

// Database installation is a separate local preparation phase in the CLI.
workflow DATABASES {
    if (params.databases.contains('checkm')) {
        SETUP_CHECKM_DB(Channel.of([params.ref_dir, "${params.ref_dir}/checkm_data", true]))
    }
    if (params.databases.contains('gtdbtk')) {
        SETUP_GTDBTK_DB(Channel.of([params.ref_dir, "${params.ref_dir}/gtdbtk_r232", params.gtdbtk_package_url, true]))
    }
    if (params.databases.contains('gunc')) {
        SETUP_GUNC_DB(Channel.of([params.ref_dir, "${params.ref_dir}/gunc_db/gunc_db_progenomes2.1.dmnd", true]))
    }
}

// Standalone stages use the same processes and file contracts as the full run.
workflow PREPROCESS {
    inputs = Channel.fromPath("${params.fasta_dir}/*.fasta", checkIfExists: true)
        .flatMap { f -> filterSizes().collect { size -> tuple(genomeId(f), size.toString(), f) } }
    f = FILTER_FASTA(inputs).fasta
    s = SEQKIT_STAT(f).stats
    COMBINE_SEQKIT_STATS(s.map { it[3] }.toSortedList())
}
workflow CHECKM_ONLY {
    files = Channel.fromPath("${params.fasta_dir}/*.fasta").toSortedList()
    batches = files.flatMap { fs -> balancedBatches(fs, params.checkm_batch_size, params.checkm_parallel_batches).collect { group -> tuple('full', batchId(group), group, params.checkm_data_path) } }
    COMBINE_CHECKM(CHECKM_SINGLE(batches.map { row -> new ArrayList(row) + [assessmentToken('checkm/individual', "${row[0]}.batch${row[1]}")] }).tsv.toSortedList())
}
workflow TAXONOMY_ONLY {
    files = Channel.fromPath("${params.fasta_dir}/*.fasta").toSortedList()
    manifests = PREP_GTDBTK_BATCHES(files.map { fs -> tuple(fs, params.gtdbtk_batch_size) }).manifests.flatMap { normalizeToList(it) }
    inputs = manifests.combine(files.map { [it] }).map { m, fs ->
        def names = m.toFile().readLines().toSet()
        tuple(m.baseName, m, fs.findAll { names.contains(it.name) }, params.gtdbtk_data_path)
    }
    COMBINE_GTDBTK(GTDBTK(inputs.map { row -> new ArrayList(row) + [assessmentToken('gtdbtk/individual', "gtdbtk_${row[0]}")] }).outdir.toSortedList())
}
workflow RRNA_ONLY {
    inputs = Channel.fromPath("${params.fasta_dir}/*.fasta").map { f -> tuple(genomeId(f), f, file(params.kingdom_map)) }
    rrna = BARRNAP(inputs.map { row -> new ArrayList(row) + [assessmentToken('barrnap', "${row[0]}.rrna")] }).rrna
    blast = BLAST_RRNA(rrna.map { row -> new ArrayList(row) + [assessmentToken('barrnap', "${row[0]}.rrna_check")] }).blast
    COMPARE_BARRNAP(blast.toSortedList(), rrna.map { it[1] }.toSortedList(), rrna.map { it[2] }.toSortedList())
}
workflow TRNA_ONLY {
    inputs = Channel.fromPath("${params.fasta_dir}/*.fasta").map { f -> tuple(genomeId(f), params.kingdom, f) }
    TRNASCAN_PARSE(TRNASCAN(inputs.map { row -> new ArrayList(row) + [assessmentToken('trnascan', "${row[0]}.trna")] }).gff.toSortedList())
}
workflow GUNC_ONLY {
    files = Channel.fromPath("${params.fasta_dir}/*.fasta").toSortedList()
    batches = GUNC(files.flatMap { fs -> balancedBatches(fs, params.gunc_batch_size, params.gunc_parallel_batches).collect { group -> tuple(batchId(group), group, params.gunc_db) } }.map { row -> new ArrayList(row) + [assessmentToken('gunc', "gunc_${row[0]}")] }).outdir.toSortedList()
    COMBINE_GUNC(batches)
}

workflow {
    if (params.stage == 'run') { COMPLETE() }
    else if (params.stage == 'build_db') { DATABASES() }
    else if (params.stage == 'preprocess') { PREPROCESS() }
    else if (params.stage == 'quality') { CHECKM_ONLY() }
    else if (params.stage == 'taxonomy') { TAXONOMY_ONLY() }
    else if (params.stage == 'rrna') { RRNA_ONLY() }
    else if (params.stage == 'trna') { TRNA_ONLY() }
    else if (params.stage == 'chimera') { GUNC_ONLY() }
    else { error "Unknown stage: ${params.stage}" }
}
