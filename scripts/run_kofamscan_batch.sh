#!/usr/bin/env bash
set -euo pipefail

# run_kofamscan_all_bins.sh
#
# Batch-run KofamScan across many nested bin directories.
#
# Expected input pattern:
#   SEARCH_ROOT/.../bin_10/orf_prediction/bin_10.qced.faa
#
# Expected output:
#   SEARCH_ROOT/.../bin_10/results/annotation_table/bin_10.kofamscan.detail.tsv
#   SEARCH_ROOT/.../bin_10/results/annotation_table/bin_10.kofamscan.passed.tsv
#   SEARCH_ROOT/.../bin_10/results/annotation_table/bin_10.kofamscan.mapper.tsv
#
# Usage:
#   ./run_kofamscan_all_bins.sh /path/to/search/root

SEARCH_ROOT="${1:?ERROR: provide search root}"

CONFIG_FILE="${HOME}/.config/kofamscan/config.yml"

# Run 16 KofamScan jobs at once, each using 1 CPU.
GENOMES_AT_ONCE=16
CPUS_PER_GENOME=1

RUN_LIST="kofamscan_faa_files.list"

command -v exec_annotation >/dev/null 2>&1 || {
    echo "ERROR: exec_annotation not found. Activate the kofamscan conda env."
    exit 1
}

command -v parallel >/dev/null 2>&1 || {
    echo "ERROR: GNU parallel not found."
    exit 1
}

[[ -d "${SEARCH_ROOT}" ]] || {
    echo "ERROR: search root not found: ${SEARCH_ROOT}"
    exit 1
}

[[ -f "${CONFIG_FILE}" ]] || {
    echo "ERROR: KofamScan config not found: ${CONFIG_FILE}"
    exit 1
}

find "${SEARCH_ROOT}" -type f -path "*/orf_prediction/*.qced.faa" | sort > "${RUN_LIST}"

if [[ ! -s "${RUN_LIST}" ]]; then
    echo "ERROR: no *.qced.faa files found under ${SEARCH_ROOT}"
    exit 1
fi

echo "Found $(wc -l < "${RUN_LIST}") FAA files."
echo "Running ${GENOMES_AT_ONCE} jobs at once, ${CPUS_PER_GENOME} CPU per job."
echo "Using config: ${CONFIG_FILE}"

export CONFIG_FILE CPUS_PER_GENOME

parallel -j "${GENOMES_AT_ONCE}" --halt soon,fail=1 '
    faa={}

    # Example input:
    # .../bin_10/orf_prediction/bin_10.qced.faa

    orf_dir=$(dirname "$faa")
    bin_dir=$(dirname "$orf_dir")
    bin_id=$(basename "$faa" .qced.faa)

    nfs_out_dir="${bin_dir}/results/annotation_table"

    nfs_detail_out="${nfs_out_dir}/${bin_id}.kofamscan.detail.tsv"
    nfs_passed_out="${nfs_out_dir}/${bin_id}.kofamscan.passed.tsv"
    nfs_mapper_out="${nfs_out_dir}/${bin_id}.kofamscan.mapper.tsv"

    if [[ -s "$nfs_detail_out" ]]; then
        echo "SKIP existing: $nfs_detail_out"
        exit 0
    fi

    tmp_dir="/tmp/kofamscan_${bin_id}_${RANDOM}_$$"
    mkdir -p "$tmp_dir"
    trap "rm -rf \"$tmp_dir\"" EXIT

    local_detail_out="${tmp_dir}/${bin_id}.kofamscan.detail.tsv"
    local_passed_out="${tmp_dir}/${bin_id}.kofamscan.passed.tsv"
    local_mapper_out="${tmp_dir}/${bin_id}.kofamscan.mapper.tsv"

    echo "Running KofamScan: ${bin_id}"
    echo "Input: $faa"
    echo "Local tmp: $tmp_dir"

    exec_annotation \
        -c "${CONFIG_FILE}" \
        --cpu "${CPUS_PER_GENOME}" \
        --tmp-dir "$tmp_dir" \
        -f detail-tsv \
        -o "$local_detail_out" \
        "$faa"

    grep -E '^\*' "$local_detail_out" > "$local_passed_out" || true

    if [[ -s "$local_passed_out" ]]; then
        cut -f2,3 "$local_passed_out" > "$local_mapper_out"
    else
        : > "$local_mapper_out"
    fi
    mkdir -p "$nfs_out_dir"

    cp "$local_detail_out" "$nfs_detail_out"
    cp "$local_passed_out" "$nfs_passed_out"
    cp "$local_mapper_out" "$nfs_mapper_out"

    echo "Finished: ${bin_id}"
' :::: "${RUN_LIST}"

echo "Done."
echo "Outputs written under each bin directory:"
echo "  [bin]/results/annotation_table/"
