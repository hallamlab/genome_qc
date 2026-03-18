nextflow.enable.dsl=2

params.fasta_dir        = params.fasta_dir        ?: "${projectDir}/test/fasta"
params.working_dir      = params.working_dir      ?: "${projectDir}/results_nextflow"
params.filter_size      = params.filter_size      ?: ""
params.min_genome_size  = params.min_genome_size  ?: 0
params.max_genome_size  = params.max_genome_size  ?: 999999999
params.completeness     = params.completeness     ?: 50
params.contamination    = params.contamination    ?: 5
params.qscore_min       = params.qscore_min       ?: 50
params.checkm_cpus      = params.checkm_cpus      ?: 1
params.checkm_batch_size = params.checkm_batch_size ?: 1
params.checkm_max_forks = params.checkm_max_forks ?: 1
params.trnascan_cpus    = params.trnascan_cpus    ?: 1
params.trnascan_max_forks = params.trnascan_max_forks ?: 24
params.gtdbtk_cpus      = params.gtdbtk_cpus      ?: 1
params.gtdbtk_batch_size = params.gtdbtk_batch_size ?: 50
params.gtdbtk_max_forks = params.gtdbtk_max_forks ?: 2
params.gunc_cpus        = params.gunc_cpus        ?: 1
params.max_forks        = params.max_forks        ?: 64
params.ref_dir          = params.ref_dir          ?: ""
params.checkm_data_path = params.checkm_data_path ?: ""
params.gtdbtk_data_path = params.gtdbtk_data_path ?: ""
params.gunc_db          = params.gunc_db          ?: ""
params.gtdbtk_package_url = params.gtdbtk_package_url ?: "https://data.gtdb.ecogenomic.org/releases/release220/220.0/auxillary_files/gtdbtk_package/full_package/gtdbtk_r220_data.tar.gz"
params.debug            = params.debug            ?: false


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
    return new File(root, "VERSION_CONFIG").exists() ||
           new File(root, "metadata/metadata.txt").exists() ||
           new File(root, "mrca_red/gtdbtk_r220_bac120.tsv").exists() ||
           new File(root, "mrca_red/gtdbtk_r226_bac120.tsv").exists()
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

def shellQuote(value) {
    return "'" + value.toString().replace("'", "'\"'\"'") + "'"
}

def writeManifestFile(pathString, values) {
    def outFile = new File(pathString)
    outFile.parentFile?.mkdirs()
    def lines = normalizeToList(values).collect { it.toString() }
    outFile.text = lines ? (lines.join(System.lineSeparator()) + System.lineSeparator()) : ""
    return file(outFile.getAbsolutePath())
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

process FILTER_FASTA {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/genomes_filtered", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(size), path(fasta)

    output:
    tuple val(id), val(size), path("${id}.${size}.fasta"), emit: fasta

    script:
    def min_len = (size == 'full') ? 0 : size
    """
    seqkit seq -g -m ${min_len} -o ${id}.${size}.fasta ${fasta}
    """
}

process SEQKIT_STAT {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/seqkit/individual", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(size), path(fasta)

    output:
    tuple val(id), val(size), path(fasta), path("${id}.${size}.seqkit.tsv"), emit: stats

    script:
    """
    seqkit stat -a -T -o ${id}.${size}.seqkit.tsv ${fasta}
    """
}

process PRUNE_FASTA {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/seqkit.yaml"
    publishDir "${params.working_dir}/genomes_pruned", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(size), path(fasta), path(stat)

    output:
    tuple val(id), val(size), path("${id}.${size}.fasta"), optional: true, emit: pruned

    script:
    """
    sum_col=\$(head -n1 ${stat} | tr '\t' '\n' | nl -ba | awk '\$2=="sum_len"{print \$1}')
    if [ -z "\$sum_col" ]; then
      echo "sum_len column not found in ${stat}" >&2
      exit 1
    fi
    sum_len=\$(awk -v c=\$sum_col 'NR==2{print \$c}' ${stat})
    if [ -n "\$sum_len" ] && [ "\$sum_len" -ge ${params.min_genome_size} ] && [ "\$sum_len" -le ${params.max_genome_size} ]; then
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
    head -n1 ${stats[0]} > seqkit_stats.tsv
    for f in ${stats}; do
      tail -n +2 \$f >> seqkit_stats.tsv
    done
    """
}

process CHECKM_SINGLE {
    tag "${size}.batch${batch_id}"
    cpus params.checkm_cpus
    maxForks params.checkm_max_forks
    conda "${projectDir}/envs/checkm.yaml"
    publishDir "${params.working_dir}/checkm/individual", mode: 'copy', overwrite: true

    input:
    tuple val(size), val(batch_id), path(fastas), val(checkm_data_path)

    output:
    path("${size}.batch${batch_id}.checkm.tsv"), emit: tsv

    script:
    """
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
    path("checkm_output_HPMQ.tsv"), emit: hpmq

    script:
    """
    head -n1 ${tsvs[0]} > checkm_output.tsv
    for f in ${tsvs}; do
      tail -n +2 \$f >> checkm_output.tsv
    done
    awk -F '\t' 'NR==1 || (\$12 >= ${params.completeness} && \$13 <= ${params.contamination})' checkm_output.tsv > checkm_output_HPMQ.tsv
    """
}

process QSCORE {
    cpus 1
    conda "${projectDir}/envs/python_calc_qscore.yaml"
    publishDir "${params.working_dir}/qscore", mode: 'copy', overwrite: true

    input:
    path(checkm_hpmq)
    path(seqkit_stats)

    output:
    path("qscore_mqhp.tsv"), emit: mqhp
    path("qscore_all.tsv"), emit: all

    script:
    """
    mkdir -p workdir/checkm workdir/seqkit workdir/qscore
    cp ${checkm_hpmq} workdir/checkm/checkm_output_HPMQ.tsv
    cp ${seqkit_stats} workdir/seqkit/seqkit_stats.tsv
    python ${projectDir}/calc_qscore.py workdir ${params.completeness} ${params.contamination} ${params.qscore_min}
    cp workdir/qscore/qscore_mqhp.tsv .
    cp workdir/qscore/qscore_all.tsv .
    """
}

process SUBSET_GENOMES {
    tag "${id}.${size}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/genomes_subset", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(size), path(fasta, stageAs: 'input.fasta'), path(qscore_mqhp)

    output:
    path("${id}.${size}.fasta"), optional: true, emit: subset

    script:
    """
    target="${id}.${size}"
    if awk -F '\t' -v target="\$target" 'NR > 1 && \$2 == target { found=1; exit } END { exit(found ? 0 : 1) }' ${qscore_mqhp}; then
      cp ${fasta} ${id}.${size}.fasta
    fi
    """
}

process GTDBTK {
    tag "batch${batch_id}"
    cpus params.gtdbtk_cpus
    maxForks params.gtdbtk_max_forks
    conda "${projectDir}/envs/gtdbtk.yaml"
    publishDir "${params.working_dir}/gtdbtk/individual", mode: 'copy', overwrite: true

    input:
    tuple val(batch_id), path(fasta_manifest), val(gtdbtk_data_path)

    output:
    path("gtdbtk_${batch_id}"), emit: outdir

    script:
    """
    mkdir -p genomes
    while IFS= read -r f; do
      [ -n "\$f" ] || continue
      cp "\$f" genomes/
    done < ${fasta_manifest}
    if [ -n "${gtdbtk_data_path}" ]; then
      export GTDBTK_DATA_PATH="${gtdbtk_data_path}"
    fi
    gtdbtk classify_wf --skip_ani_screen --genome_dir genomes --out_dir gtdbtk_${batch_id} --cpus ${task.cpus} -x fasta
    """
}

process COMBINE_GTDBTK {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/gtdbtk", mode: 'copy', overwrite: true

    input:
    path(gtdbtk_dir_manifest)

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
    while IFS= read -r d; do
      [ -n "\$d" ] || continue
      [ -f "\$d/classify/gtdbtk.bac120.summary.tsv" ] && bac_files+=("\$d/classify/gtdbtk.bac120.summary.tsv")
      [ -f "\$d/classify/gtdbtk.ar53.summary.tsv" ] && arc_files+=("\$d/classify/gtdbtk.ar53.summary.tsv")
    done < ${gtdbtk_dir_manifest}

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
    tag "${id}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/barrnap.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true

    input:
    tuple val(id), path(fasta), path(kingdom_map)

    output:
    tuple val(id), path("${id}.rRNA.fasta"), path("${id}.rRNA.gff"), emit: rrna

    script:
    """
    kingdom=\$(awk -v id="${id}" '(\$1==id){print \$2; exit}' ${kingdom_map})
    if [ -z "\$kingdom" ]; then
      kingdom="Bacteria"
    fi
    k=\$([ "\$kingdom" = "Archaea" ] && echo "arc" || echo "bac")
    barrnap --kingdom \${k} --threads ${task.cpus} --outseq ${id}.rRNA.fasta ${fasta} > ${id}.rRNA.gff
    """
}

process BLAST_RRNA {
    tag "${id}"
    cpus 1
    maxForks params.max_forks
    conda "${projectDir}/envs/blast.yaml"
    publishDir "${params.working_dir}/barrnap", mode: 'copy', overwrite: true

    input:
    tuple val(id), path(rrna_fasta), path(rrna_gff)

    output:
    path("${id}.blastout"), emit: blast

    script:
    """
    if [ -s ${rrna_fasta} ]; then
      makeblastdb -in ${rrna_fasta} -dbtype nucl -out ${id}
      blastn -query ${rrna_fasta} -db ${id} -outfmt 6 -out ${id}.blastout -max_target_seqs 10
    else
      : > ${id}.blastout
    fi
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
    path(qscore_mqhp)
    path(barrnap_pass)
    path(gtdbtk_dir)

    output:
    path("dedupe_mqhp.tsv"), emit: mqhp
    path("dedupe_all.tsv"), emit: all
    path("total_mqhp.tsv"), emit: total

    script:
    """
    if [ \$(tail -n +2 ${qscore_mqhp} | wc -l) -eq 0 ]; then
      echo -e "Genome_Id\tBin Id\tclassification\tCompleteness\tContamination\tStrain heterogeneity\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\tN50\tqscore\tpass_BARRNAP" > dedupe_mqhp.tsv
      echo -e "Genome_Id\tBin Id\tclassification\tCompleteness\tContamination\tStrain heterogeneity\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\tN50\tqscore\tpass_BARRNAP" > dedupe_all.tsv
      echo -e "Genome_Id\tBin Id\tclassification\tCompleteness\tContamination\tStrain heterogeneity\tnum_seqs\tsum_len\tmin_len\tavg_len\tmax_len\tN50\tqscore\tpass_BARRNAP" > total_mqhp.tsv
      exit 0
    fi
    if ! ls ${gtdbtk_dir}/classify/*.summary.tsv >/dev/null 2>&1; then
      echo "No GTDB-Tk summary files found in ${gtdbtk_dir}/classify" >&2
      exit 1
    fi
    mkdir -p workdir/qscore workdir/barrnap workdir/gtdbtk/classify workdir/dedupe
    cp ${qscore_mqhp} workdir/qscore/qscore_mqhp.tsv
    cp ${barrnap_pass} workdir/barrnap/barrnap_blast_pass.tsv
    cp -r ${gtdbtk_dir}/classify/*.summary.tsv workdir/gtdbtk/classify/ 2>/dev/null || true
    python ${projectDir}/dedupe_genomes.py workdir ${params.completeness} ${params.contamination} ${params.qscore_min}
    cp workdir/dedupe/dedupe_mqhp.tsv .
    cp workdir/dedupe/dedupe_all.tsv .
    cp workdir/dedupe/total_mqhp.tsv .
    """
}

process CP_DEDUPE {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"
    publishDir "${params.working_dir}/dedupe/fasta", mode: 'copy', overwrite: true

    input:
    path(dedupe_mqhp)
    path(subset_fastas)

    output:
    path("*.fasta"), optional: true, emit: fasta

    script:
    """
    while read -r g; do
      [ -z "\$g" ] && continue
      found=""
      dest="\${g}.fasta"
      for f in ${subset_fastas}; do
        base=\$(basename "\$f" .fasta)
        if [ "\$base" = "\$g" ]; then
          found="\$f"
          break
        fi
      done
      if [ -n "\$found" ]; then
        if [ "\$(readlink -f "\$found")" != "\$(readlink -f "\$dest")" ]; then
          cp "\$found" "\$dest"
        fi
      fi
    done < <(tail -n +2 ${dedupe_mqhp} | cut -f2)
    """
}

process PARSE_DEDUPE_KINGDOM {
    cpus 1
    conda "${projectDir}/envs/utils.yaml"

    input:
    path(dedupe_mqhp)

    output:
    path("dedupe_kingdom.tsv"), emit: mapping

    script:
    """
    awk -F '\t' 'NR==1{next} { if (\$3 ~ /Archaea/) print \$2"\tArchaea"; else if (\$3 ~ /Bacteria/) print \$2"\tBacteria"; }' ${dedupe_mqhp} > dedupe_kingdom.tsv
    """
}

process TRNASCAN {
    tag "${id}"
    cpus params.trnascan_cpus
    maxForks params.trnascan_max_forks
    conda "${projectDir}/envs/trnascan.yaml"
    publishDir "${params.working_dir}/trnascan", mode: 'copy', overwrite: true

    input:
    tuple val(id), val(kingdom), path(fasta)

    output:
    path("${id}.gff"), emit: gff

    script:
    def flag = (kingdom == 'Archaea') ? '-A' : '-B'
    """
    tRNAscan-SE -q -Q ${flag} \
      -o ${id}.output.txt \
      -m ${id}.stats.txt \
      -b ${id}.bed \
      -j ${id}.gff \
      -a ${id}.trna.fasta \
      -l ${id}.log \
      --thread ${task.cpus} ${fasta}
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
    cpus params.gunc_cpus
    conda "${projectDir}/envs/gunc.yaml"
    publishDir "${params.working_dir}/gunc", mode: 'copy', overwrite: true

    input:
    tuple path(dedupe_fasta_manifest), val(gunc_db)

    output:
    path("gunc"), emit: outdir

    script:
    """
    if [ -z "${gunc_db}" ]; then
      echo "params.gunc_db is required" >&2
      exit 1
    fi
    mkdir -p gunc dedupe_fasta
    if [ ! -s ${dedupe_fasta_manifest} ]; then
      echo "No deduped FASTA files provided to GUNC; skipping."
      exit 0
    fi
    while IFS= read -r f; do
      [ -n "\$f" ] || continue
      cp "\$f" dedupe_fasta/
    done < ${dedupe_fasta_manifest}
    gunc run -r ${gunc_db} -d dedupe_fasta -o gunc -e .fasta -t ${task.cpus}
    """
}

process SETUP_GTDBTK_DB {
    cpus 1
    conda "${projectDir}/envs/reference_setup.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(gtdbtk_data_path), val(gtdbtk_package_url), val(allow_bootstrap)

    output:
    path("gtdbtk_reference_ready.tsv"), emit: ready

    script:
    """
    if [ -n "${ref_dir}" ]; then
      mkdir -p "${ref_dir}"
    fi
    if [ ! -s "${gtdbtk_data_path}/mrca_red/gtdbtk_r220_bac120.tsv" ] && [ ! -s "${gtdbtk_data_path}/VERSION_CONFIG" ] && [ ! -s "${gtdbtk_data_path}/metadata/metadata.txt" ]; then
      if [ "${allow_bootstrap}" = "true" ]; then
        archive="${ref_dir}/gtdbtk_package.tar.gz"
        tmpdir=\$(mktemp -d)
        if command -v wget >/dev/null 2>&1; then
          wget --no-check-certificate -O "${archive}" "${gtdbtk_package_url}"
        else
          curl -L -o "${archive}" "${gtdbtk_package_url}"
        fi
        tar -xzf "${archive}" -C "${tmpdir}"
        if [ -f "${tmpdir}/release220/VERSION_CONFIG" ]; then
          rm -rf "${gtdbtk_data_path}"
          mkdir -p "${gtdbtk_data_path}"
          cp -r "${tmpdir}/release220/." "${gtdbtk_data_path}/"
        else
          rm -rf "${gtdbtk_data_path}"
          mkdir -p "${gtdbtk_data_path}"
          tar -xzf "${archive}" -C "${gtdbtk_data_path}" --strip-components 1
        fi
        rm -rf "${tmpdir}"
      else
        echo "GTDB-Tk data path is missing or incomplete: ${gtdbtk_data_path}" >&2
        exit 1
      fi
    fi
    if [ ! -f "${gtdbtk_data_path}/VERSION_CONFIG" ] && [ ! -f "${gtdbtk_data_path}/metadata/metadata.txt" ] && [ ! -f "${gtdbtk_data_path}/mrca_red/gtdbtk_r220_bac120.tsv" ]; then
      echo "GTDB-Tk data were not installed correctly under ${gtdbtk_data_path}" >&2
      exit 1
    fi
    {
      echo -e "reference\tpath\tstatus"
      echo -e "gtdbtk\t${gtdbtk_data_path}\tready"
      echo -e "source_url\t${gtdbtk_package_url}\tused"
    } > gtdbtk_reference_ready.tsv
    """
}

process SETUP_GUNC_DB {
    cpus 1
    conda "${projectDir}/envs/gunc.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(gunc_db), val(allow_bootstrap)

    output:
    path("gunc_reference_ready.tsv"), emit: ready

    script:
    """
    if [ -n "${ref_dir}" ]; then
      mkdir -p "${ref_dir}"
    fi
    if [ ! -s "${gunc_db}" ]; then
      if [ "${allow_bootstrap}" = "true" ]; then
        mkdir -p "\$(dirname "${gunc_db}")"
        gunc download_db "\$(dirname "${gunc_db}")"
      else
        echo "GUNC database file is missing or empty: ${gunc_db}" >&2
        exit 1
      fi
    fi
    if [ ! -s "${gunc_db}" ]; then
      echo "GUNC database was not installed correctly at ${gunc_db}" >&2
      exit 1
    fi
    {
      echo -e "reference\tpath\tstatus"
      echo -e "gunc\t${gunc_db}\tready"
    } > gunc_reference_ready.tsv
    """
}

process SETUP_CHECKM_DB {
    cpus 1
    conda "${projectDir}/envs/checkm.yaml"
    publishDir "${params.working_dir}/references", mode: 'copy', overwrite: true

    input:
    tuple val(ref_dir), val(checkm_data_path), val(allow_bootstrap)

    output:
    path("checkm_reference_ready.tsv"), emit: ready

    script:
    """
    if [ -n "${ref_dir}" ]; then
      mkdir -p "${ref_dir}"
    fi
    export MPLCONFIGDIR="\$PWD/.mplconfig"
    mkdir -p "\$MPLCONFIGDIR"
    export CHECKM_DATA_PATH="${checkm_data_path}"
    if [ ! -d "${checkm_data_path}" ] || [ -z "\$(find "${checkm_data_path}" -mindepth 1 -maxdepth 1 \\( -type d -o -type f \\) 2>/dev/null | head -n 1)" ]; then
      if [ "${allow_bootstrap}" = "true" ]; then
        mkdir -p "${checkm_data_path}"
        checkm data download -p "${checkm_data_path}"
      else
        echo "CheckM data path is missing or empty: ${checkm_data_path}" >&2
        exit 1
      fi
    fi
    if [ -z "\$(find "${checkm_data_path}" -mindepth 1 -maxdepth 1 \\( -type d -o -type f \\) 2>/dev/null | head -n 1)" ]; then
      echo "CheckM data were not installed correctly under ${checkm_data_path}" >&2
      exit 1
    fi
    {
      echo -e "reference\tpath\tstatus"
      echo -e "checkm\t${checkm_data_path}\tready"
    } > checkm_reference_ready.tsv
    """
}

process MERGE_MASTER {
    cpus 1
    conda "${projectDir}/envs/merge_master.yaml"
    publishDir "${params.working_dir}", mode: 'copy', overwrite: true

    input:
    path(dedupe_mqhp)
    path(barrnap_subunits)
    path(trna_counts)

    output:
    path("Master_genome_QC.tsv"), emit: master
    path("Master_genome_QC_All.pdf"), emit: plot1
    path("Master_genome_QC_All.png"), emit: plot2
    path("Master_genome_QC_MQHQ_16S.pdf"), emit: plot3
    path("Master_genome_QC_MQHQ_16S.png"), emit: plot4
    path("Master_genome_QC_MQHQ_Phylum.pdf"), emit: plot5
    path("Master_genome_QC_MQHQ_Phylum.png"), emit: plot6

    script:
    """
    mkdir -p merge_work/barrnap merge_work/dedupe merge_work/trnascan
    cp ${barrnap_subunits} merge_work/barrnap/barrnap_subunit_counts.tsv
    cp ${dedupe_mqhp} merge_work/dedupe/dedupe_mqhp.tsv
    cp ${trna_counts} merge_work/trnascan/trnascan_trna_counts.tsv
    python ${projectDir}/merge_master.py merge_work
    cp merge_work/Master_genome_QC.tsv .
    cp merge_work/Master_genome_QC_All.pdf .
    cp merge_work/Master_genome_QC_All.png .
    cp merge_work/Master_genome_QC_MQHQ_16S.pdf .
    cp merge_work/Master_genome_QC_MQHQ_16S.png .
    cp merge_work/Master_genome_QC_MQHQ_Phylum.pdf .
    cp merge_work/Master_genome_QC_MQHQ_Phylum.png .
    """
}

workflow {
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
                resolvedGTDBTK = new File(resolvedRefDir, "gtdbtk_r220").getAbsolutePath()
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
                throw new IllegalArgumentException("GTDB-Tk data path is missing or incomplete: ${resolvedGTDBTK}")
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

        filtered = FILTER_FASTA(raw_genomes).fasta
        filtered_for_stats = filtered.map { it }
        filtered_for_subset = filtered.map { it }
        filtered_fastas = filtered_for_subset.map { it[2] }

        stats = SEQKIT_STAT(filtered_for_stats).stats
        stats_for_prune = stats.map { it }
        stats_for_combine = stats.map { it }

        seqkit_stats_files = stats_for_combine.map { it[3] }.collect()
        seqkit_combined = COMBINE_SEQKIT_STATS(seqkit_stats_files).stats

        pruned = PRUNE_FASTA(stats_for_prune).pruned

        def checkmBatchSize = Math.max(1, params.checkm_batch_size as Integer)
        checkm_batches = pruned
            .map { id, size, fasta -> tuple(size, tuple(id, size, fasta)) }
            .groupTuple()
            .flatMap { size, records ->
                def out = []
                def ordered = records.sort { a, b -> a[0].toString() <=> b[0].toString() }
                def grouped = ordered.collate(checkmBatchSize)
                for (group in grouped) {
                    def fastas = group.collect { it[2] }
                    def batchSeed = group.collect { it[0].toString() }.join("|")
                    def batchId = java.security.MessageDigest
                        .getInstance("MD5")
                        .digest(batchSeed.getBytes("UTF-8"))
                        .encodeHex()
                        .toString()
                        .substring(0, 12)
                    out << tuple(size, batchId, fastas)
                }
                return out
            }

        checkm_input = checkm_ready ?
            checkm_batches.combine(checkm_ready).map { size, batch_id, fastas, ready -> tuple(size, batch_id, fastas, resolvedCheckmData) } :
            checkm_batches.map { size, batch_id, fastas -> tuple(size, batch_id, fastas, resolvedCheckmData) }

        checkm = CHECKM_SINGLE(checkm_input).tsv
        checkm_combined = COMBINE_CHECKM(checkm.collect())

        qscore = QSCORE(checkm_combined.hpmq, seqkit_combined)
        qscore_for_subset = qscore.mqhp.map { it }
        qscore_for_dedupe = qscore.mqhp.map { it }

        subset_input = filtered_for_subset
            .combine(qscore_for_subset)
            .map { id, size, fasta, qscore_file -> tuple(id, size, fasta, qscore_file) }

        subset = SUBSET_GENOMES(subset_input).subset
        subset_for_gtdbtk = subset.map { it }
        subset_for_ids = subset.map { it }
        subset_for_dedupe = subset.map { it }
        def gtdbtkBatchSize = Math.max(1, params.gtdbtk_batch_size as Integer)
        gtdbtk_batches = subset_for_gtdbtk
            .map { fasta -> tuple(fasta.baseName, fasta.toString()) }
            .collect()
            .flatMap { records ->
                def out = []
                def ordered = records.sort { a, b -> a[0].toString() <=> b[0].toString() }
                def grouped = ordered.collate(gtdbtkBatchSize)
                for (group in grouped) {
                    def fastas = group.collect { it[1] }
                    def batchSeed = group.collect { it[0].toString() }.join("|")
                    def batchId = java.security.MessageDigest
                        .getInstance("MD5")
                        .digest(batchSeed.getBytes("UTF-8"))
                        .encodeHex()
                        .toString()
                        .substring(0, 12)
                    def manifestPath = new File(params.working_dir.toString(), "manifests/gtdbtk_batch_${batchId}.list").getAbsolutePath()
                    out << tuple(batchId, writeManifestFile(manifestPath, fastas))
                }
                return out
            }

        gtdbtk_input = bootstrapReferences ?
            gtdb_ready
                .combine(gtdbtk_batches)
                .map { ready, batch_id, fasta_manifest -> tuple(batch_id, fasta_manifest, resolvedGTDBTK) } :
            gtdbtk_batches.map { batch_id, fasta_manifest -> tuple(batch_id, fasta_manifest, resolvedGTDBTK) }

        gtdbtk_batches_out = GTDBTK(gtdbtk_input).outdir
        gtdbtk_dir_manifest = gtdbtk_batches_out
            .map { it.toString() }
            .collect()
            .map { dirs ->
                def manifestPath = new File(params.working_dir.toString(), "manifests/gtdbtk_merged_dirs.list").getAbsolutePath()
                writeManifestFile(manifestPath, dirs)
            }
        gtdbtk = COMBINE_GTDBTK(gtdbtk_dir_manifest).outdir
        gtdbtk_map = PARSE_GTDBTK(gtdbtk).mapping

        subset_ids = subset_for_ids.map { f -> tuple(f.baseName, f) }
        barrnap_in = subset_ids.combine(gtdbtk_map)

        rrna = BARRNAP(barrnap_in).rrna
        rrna_for_blast = rrna.map { it }
        rrna_for_files = rrna.map { it }
        rrna_fastas = rrna_for_files.map { it[1] }
        rrna_gffs = rrna_for_files.map { it[2] }

        blast = BLAST_RRNA(rrna_for_blast).blast

        barblast = COMPARE_BARRNAP(blast.collect(), rrna_fastas.collect(), rrna_gffs.collect())

        dedupe = DEDUPE(qscore_for_dedupe, barblast.pass, gtdbtk)
        dedupe_for_cp = dedupe.mqhp.map { it }
        dedupe_for_parse = dedupe.mqhp.map { it }
        dedupe_for_merge = dedupe.mqhp.map { it }

        dedupe_fasta = CP_DEDUPE(dedupe_for_cp, subset_for_dedupe.collect()).fasta

        dedupe_map = PARSE_DEDUPE_KINGDOM(dedupe_for_parse).mapping
        dedupe_kingdom = dedupe_map.map { f ->
            def out = []
            f.toFile().readLines().findAll { it?.trim() }.each { line ->
                def parts = line.split('\t')
                out << tuple(parts[0], parts[1])
            }
            out
        }.flatten()

        trna_in = subset_ids.join(dedupe_kingdom)
            .map { row ->
                def id = row[0]
                def fasta = row[1]
                def kingdom = (row.size() > 2 && row[2]) ? row[2] : "Bacteria"
                tuple(id, kingdom, fasta)
            }

        trna_gffs = TRNASCAN(trna_in).gff

        trna_counts = TRNASCAN_PARSE(trna_gffs.collect())

        gunc_input = bootstrapReferences ?
            gunc_ready
                .combine(dedupe_fasta.map { it.toString() }.collect().map { fastas ->
                    def manifestPath = new File(params.working_dir.toString(), "manifests/gunc_dedupe_fastas.list").getAbsolutePath()
                    tuple(writeManifestFile(manifestPath, fastas))
                })
                .map { ready, fastas_manifest -> tuple(fastas_manifest, resolvedGuncDb) } :
            dedupe_fasta.map { it.toString() }.collect().map { fastas ->
                def manifestPath = new File(params.working_dir.toString(), "manifests/gunc_dedupe_fastas.list").getAbsolutePath()
                tuple(writeManifestFile(manifestPath, fastas), resolvedGuncDb)
            }
        gunc = GUNC(gunc_input).outdir

        master = MERGE_MASTER(dedupe_for_merge, barblast.subunits, trna_counts.counts)

    emit:
        master.master
}
