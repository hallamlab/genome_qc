#!/usr/bin/env python3

import argparse
import gzip
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def log(message):
    print(message, file=sys.stderr, flush=True)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def command_exists(name):
    return shutil.which(name) is not None


def resolve_fasttree():
    for candidate in ["FastTree", "FastTreeMP"]:
        if command_exists(candidate):
            return candidate
    return None


def read_tsv(path):
    return pd.read_csv(path, sep="\t")


def write_status(path, rows):
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def parse_fasta(path):
    records = []
    header = None
    seq_chunks = []
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))
    return records


_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def reverse_complement(seq):
    return seq.translate(_COMPLEMENT)[::-1]


def extract_subsequence(seq, start, end, strand):
    subseq = seq[int(start) - 1:int(end)]
    if strand == "-":
        return reverse_complement(subseq)
    return subseq


def discover_phylogeny_set_dirs(root_dir):
    root = Path(root_dir).expanduser().absolute()
    candidates = []
    direct_markers = [root / "selected_genomes.tsv", root / "audit" / "selected_genomes.tsv", root / "selected_set" / "master.tsv"]
    if any(path.exists() for path in direct_markers):
        candidates.append(root)
    for child in sorted(root.iterdir() if root.exists() and root.is_dir() else []):
        if child.is_dir():
            markers = [child / "selected_genomes.tsv", child / "audit" / "selected_genomes.tsv", child / "selected_set" / "master.tsv"]
            if any(path.exists() for path in markers):
                candidates.append(child)
    for parent_name in ["denovo_phylogeny"]:
        parent = root / parent_name
        if parent.exists():
            for child in sorted(parent.iterdir()):
                if child.is_dir():
                    markers = [child / "selected_genomes.tsv", child / "audit" / "selected_genomes.tsv", child / "selected_set" / "master.tsv"]
                    if any(path.exists() for path in markers):
                        candidates.append(child)
    unique = []
    seen = set()
    for path in candidates:
        path_text = str(path)
        if path_text in seen:
            continue
        seen.add(path_text)
        unique.append(path)
    return unique


def resolve_selected_table(set_dir):
    for candidate in [
        set_dir / "selected_genomes.tsv",
        set_dir / "audit" / "selected_genomes.tsv",
        set_dir / "selected_set" / "master.tsv",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No selected-genome table found under {set_dir}")


def choose_fasta_column(frame):
    for column in ["copied_fasta_path", "fasta_path", "mp_fasta_path", "ani_fasta_path"]:
        if column in frame.columns:
            series = frame[column].astype(str).str.strip()
            if series.ne("").any():
                return column
    return None


def sanitize_token(value):
    text = str(value).strip()
    if not text:
        return "item"
    safe = []
    for ch in text:
        safe.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "item"


def fasta_suffixless_name(path_text):
    name = Path(str(path_text)).name
    for suffix in [".fasta.gz", ".fa.gz", ".fna.gz", ".fasta", ".fa", ".fna"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def parse_newick_tip_labels(path):
    text = Path(path).read_text().strip()
    labels = []
    index = 0
    previous = ""
    while index < len(text):
        character = text[index]
        if character in "(),:;":
            previous = character
            index += 1
            continue
        if character.isspace():
            index += 1
            continue

        if character == "'":
            index += 1
            chunks = []
            while index < len(text):
                if text[index] == "'":
                    if index + 1 < len(text) and text[index + 1] == "'":
                        chunks.append("'")
                        index += 2
                        continue
                    index += 1
                    break
                chunks.append(text[index])
                index += 1
            label = "".join(chunks)
        else:
            start = index
            while index < len(text) and text[index] not in "(),:;" and not text[index].isspace():
                index += 1
            label = text[start:index]

        lookahead = index
        while lookahead < len(text) and text[lookahead].isspace():
            lookahead += 1
        if previous in {"(", ","} and lookahead < len(text) and text[lookahead] == ":":
            labels.append(label)
        previous = "label"
        index = lookahead
    return labels


def add_tree_ids(frame):
    working = frame.copy()
    tree_ids = []
    for row in working.to_dict("records"):
        path_text = ""
        for column in ["copied_fasta_path", "fasta_path", "mp_fasta_path", "ani_fasta_path"]:
            value = str(row.get(column, "")).strip()
            if value and value.lower() not in {"nan", "none", "null"}:
                path_text = value
                break
        if path_text:
            tree_ids.append(fasta_suffixless_name(path_text))
        else:
            genome_id = str(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", "")))).strip()
            sample = str(row.get("sample", "")).strip()
            category = str(row.get("category", "")).strip()
            tree_ids.append(sanitize_token("__".join(part for part in [sample, category, genome_id] if part)))
    working["tree_id"] = tree_ids
    return working


TREE_METADATA_TABLES = {
    "core": [
        "tree_id",
        "Genome_Id",
        "sample",
        "category",
        "best_representative",
        "cohort",
        "tree_domain",
        "tree_file",
    ],
    "taxonomy": [
        "tree_id",
        "Domain",
        "Phylum",
        "Class",
        "Order",
        "Family",
        "Genus",
        "Species",
    ],
    "quality": [
        "tree_id",
        "mimag_tier",
        "Completeness",
        "Contamination",
        "qscore",
        "N50",
        "sum_len",
        "integrity_score",
        "recoverability_score",
        "mimag_quality_index",
        "contains_16S",
        "16S_rRNA",
        "trna_total",
        "trna_unique",
    ],
    "gunc": [
        "tree_id",
        "gunc_assessment",
        "gunc_strict_assessment",
        "gunc_clade_separation_score",
        "gunc_reference_representation_score",
        "gunc_contamination_portion",
        "gunc_taxonomic_level",
        "gunc_pass",
    ],
}


def write_tree_metadata(tree_path, selected_df, cohort_name="", domain_label=""):
    tips = parse_newick_tip_labels(tree_path)
    if not tips:
        return []
    metadata = add_tree_ids(selected_df)
    metadata = metadata.loc[metadata["tree_id"].isin(set(tips))].copy()
    metadata["cohort"] = cohort_name
    metadata["tree_domain"] = domain_label
    metadata["tree_file"] = Path(tree_path).name
    metadata = metadata.drop_duplicates("tree_id").set_index("tree_id").reindex(tips).reset_index()
    wrote = []
    for table_name, requested_columns in TREE_METADATA_TABLES.items():
        columns = [column for column in requested_columns if column in metadata.columns]
        if columns == ["tree_id"]:
            continue
        out_path = Path(tree_path).with_name(Path(tree_path).stem + f"_{table_name}_metadata.tsv")
        metadata.loc[:, columns].to_csv(out_path, sep="\t", index=False)
        wrote.append(out_path)
    return wrote


def ensure_selected_fastas(set_dir, selected_df):
    fasta_dir = set_dir / "selected_set" / "fasta"
    ensure_dir(fasta_dir)
    copied_column = "copied_fasta_path" if "copied_fasta_path" in selected_df.columns else None
    if copied_column:
        copied_paths = [
            Path(str(value).strip()).expanduser()
            for value in selected_df[copied_column].tolist()
            if str(value).strip() and str(value).strip().lower() not in {"nan", "none", "null"}
        ]
        if copied_paths and all(path.exists() for path in copied_paths):
            return fasta_dir

    for existing_fasta in list(fasta_dir.iterdir()):
        if existing_fasta.is_file() and existing_fasta.name.lower().endswith(
            (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz")
        ):
            existing_fasta.unlink()

    fasta_column = choose_fasta_column(selected_df)
    if fasta_column is None:
        raise ValueError(f"No usable FASTA path column found in selected table for {set_dir}")

    copied = 0
    for row in selected_df.to_dict("records"):
        source = Path(str(row.get(fasta_column, "")).strip()).expanduser()
        if not source.exists():
            continue
        genome_id = str(row.get("Genome_Id", row.get("genome_id", source.stem))).strip()
        category = str(row.get("category", "")).strip()
        sample = str(row.get("sample", "")).strip()
        stem_parts = [part for part in [sample, category, genome_id] if part]
        target_name = sanitize_token("__".join(stem_parts) if stem_parts else source.stem) + ".fasta"
        shutil.copy2(str(source), str(fasta_dir / target_name))
        copied += 1
    if copied == 0:
        raise ValueError(f"Could not materialize selected FASTAs for {set_dir}")
    return fasta_dir


def build_selected_manifest(selected_df, fasta_dir, out_path):
    rows = []
    for row in selected_df.to_dict("records"):
        path_text = str(row.get("copied_fasta_path", "")).strip()
        if not path_text or path_text.lower() in {"nan", "none", "null"}:
            genome_id = str(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", "")))).strip()
            sample = str(row.get("sample", "")).strip()
            category = str(row.get("category", "")).strip()
            fallback_stem = sanitize_token("__".join(part for part in [sample, category, genome_id] if part))
            path = fasta_dir / f"{fallback_stem}.fasta"
        else:
            path = Path(path_text).expanduser()
        if not path.exists():
            continue
        rows.append(
            {
                "fasta_path": str(path),
                "genome_id": str(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", path.stem)))),
                "sample": str(row.get("sample", "")),
                "category": str(row.get("category", "")),
                "Domain": str(row.get("Domain", row.get("mp_Domain", ""))),
            }
        )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(out_path, sep="\t", index=False)
    return manifest_df


def choose_barrnap_kingdom(domain_value):
    text = str(domain_value).strip().lower()
    if "archaea" in text or text.startswith("d__archaea"):
        return "arc"
    return "bac"


def run_command(command, cwd=None, env=None, stdout_path=None):
    if stdout_path is None:
        completed = subprocess.run(command, cwd=cwd, env=env, check=True, text=True, capture_output=True)
        return completed.stdout
    with open(stdout_path, "w") as handle:
        subprocess.run(command, cwd=cwd, env=env, check=True, text=True, stdout=handle)
    return ""


def run_16s_phylogeny(set_dir, manifest_df, threads):
    phylo_dir = set_dir / "phylogeny" / "rrna_16s"
    ensure_dir(phylo_dir)
    status_rows = []
    if not command_exists("barrnap"):
        status_rows.append({"status": "skipped", "reason": "barrnap_not_found"})
        write_status(phylo_dir / "status.tsv", status_rows)
        return [phylo_dir / "status.tsv"]

    extracted_records = []
    for row in manifest_df.to_dict("records"):
        fasta_path = Path(str(row.get("fasta_path", "")).strip())
        if not fasta_path.exists():
            status_rows.append({"genome_id": row.get("genome_id", fasta_path.stem), "status": "missing_fasta", "reason": str(fasta_path)})
            continue
        genome_id = str(row.get("genome_id", fasta_path.stem)).strip() or fasta_path.stem
        genome_dir = phylo_dir / sanitize_token(genome_id)
        ensure_dir(genome_dir)
        gff_path = genome_dir / f"{sanitize_token(genome_id)}.barrnap.gff"
        kingdom = choose_barrnap_kingdom(row.get("Domain", ""))
        command = [
            "barrnap",
            "--kingdom", kingdom,
            "--threads", str(max(1, int(threads))),
            str(fasta_path),
        ]
        try:
            run_command(command, stdout_path=str(gff_path))
        except subprocess.CalledProcessError as exc:
            status_rows.append({"genome_id": genome_id, "status": "barrnap_failed", "reason": str(exc)})
            continue
        contigs = {header: seq for header, seq in parse_fasta(str(fasta_path))}
        best_sequence = ""
        best_record = None
        with open(gff_path, "r") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                seqid, source, feature_type, start, end, score, strand, phase, attributes = parts
                attr_text = attributes.lower()
                if "16s" not in attr_text:
                    continue
                if seqid not in contigs:
                    continue
                sequence = extract_subsequence(contigs[seqid], start, end, strand)
                if len(sequence) > len(best_sequence):
                    best_sequence = sequence
                    best_record = {
                        "genome_id": genome_id,
                        "seqid": seqid,
                        "start": int(start),
                        "end": int(end),
                        "strand": strand,
                        "length": int(len(sequence)),
                        "kingdom": kingdom,
                    }
        if best_record is None:
            status_rows.append({"genome_id": genome_id, "status": "no_16s_found", "reason": kingdom})
            continue
        extracted_records.append((best_record, best_sequence))
        status_rows.append({"genome_id": genome_id, "status": "16s_found", "length": best_record["length"], "kingdom": kingdom})

    status_path = phylo_dir / "status.tsv"
    write_status(status_path, status_rows)
    wrote = [status_path]
    if not extracted_records:
        return wrote

    seq_table = pd.DataFrame([record for record, _seq in extracted_records])
    seq_table_path = phylo_dir / "16s_summary.tsv"
    seq_table.to_csv(seq_table_path, sep="\t", index=False)
    wrote.append(seq_table_path)

    fasta_out = phylo_dir / "16s_representatives.fasta"
    with open(fasta_out, "w") as handle:
        for record, sequence in extracted_records:
            handle.write(f">{sanitize_token(record['genome_id'])}\n")
            for idx in range(0, len(sequence), 80):
                handle.write(sequence[idx:idx + 80] + "\n")
    wrote.append(fasta_out)

    if len(extracted_records) < 2:
        return wrote
    if not command_exists("mafft"):
        fail_path = phylo_dir / "tree_failure.tsv"
        write_status(fail_path, [{"status": "skipped", "reason": "mafft_not_found"}])
        wrote.append(fail_path)
        return wrote
    fasttree_bin = resolve_fasttree()
    if fasttree_bin is None:
        fail_path = phylo_dir / "tree_failure.tsv"
        write_status(fail_path, [{"status": "skipped", "reason": "fasttree_not_found"}])
        wrote.append(fail_path)
        return wrote

    aln_path = phylo_dir / "16s_alignment.fasta"
    tree_path = phylo_dir / "16s_tree.nwk"
    try:
        run_command(["mafft", "--auto", "--thread", str(max(1, int(threads))), str(fasta_out)], stdout_path=str(aln_path))
        run_command([fasttree_bin, "-nt", str(aln_path)], stdout_path=str(tree_path))
        wrote.extend([aln_path, tree_path])
    except subprocess.CalledProcessError as exc:
        fail_path = phylo_dir / "tree_failure.tsv"
        write_status(fail_path, [{"status": "failed", "reason": str(exc)}])
        wrote.append(fail_path)
    return wrote


def read_fasta_count(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    count = 0
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def maybe_copy_gzip_fasta(path, destination):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as src, open(destination, "w") as dst:
            shutil.copyfileobj(src, dst)
        return destination
    shutil.copy2(path, destination)
    return destination


def find_gtdb_user_msa(align_dir, domain_label):
    candidates = [
        align_dir / f"gtdbtk.{domain_label}.user_msa.fasta.gz",
        align_dir / f"gtdbtk.{domain_label}.user_msa.fasta",
        align_dir / "align" / f"gtdbtk.{domain_label}.user_msa.fasta.gz",
        align_dir / "align" / f"gtdbtk.{domain_label}.user_msa.fasta",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(align_dir.rglob(f"gtdbtk.{domain_label}.user_msa.fasta*")) if align_dir.exists() else []
    return matches[0] if matches else None


def run_gtdb_marker_phylogeny(set_dir, fasta_dir, threads, gtdbtk_data_path=None, selected_df=None):
    phylo_dir = set_dir / "phylogeny" / "gtdb_markers"
    ensure_dir(phylo_dir)
    status_rows = []
    wrote = []
    fasttree_bin = resolve_fasttree()
    if fasttree_bin is None:
        status_rows.append({"status": "skipped", "reason": "fasttree_not_found"})
        status_path = phylo_dir / "status.tsv"
        write_status(status_path, status_rows)
        return [status_path]
    fasta_files = sorted([path for path in fasta_dir.iterdir() if path.is_file()])
    if len(fasta_files) < 2:
        status_rows.append({"status": "skipped", "reason": "fewer_than_two_genomes"})
        status_path = phylo_dir / "status.tsv"
        write_status(status_path, status_rows)
        return [status_path]

    identify_dir = phylo_dir / "identify"
    align_dir = phylo_dir / "align"
    env = os.environ.copy()
    if gtdbtk_data_path:
        env["GTDBTK_DATA_PATH"] = str(Path(gtdbtk_data_path).expanduser().resolve())
    existing_msa = any(find_gtdb_user_msa(align_dir, domain_label) is not None for domain_label in ["bac120", "ar53"])
    if not existing_msa:
        if not command_exists("gtdbtk"):
            status_rows.append({"status": "skipped", "reason": "gtdbtk_not_found"})
            status_path = phylo_dir / "status.tsv"
            write_status(status_path, status_rows)
            return [status_path]
        try:
            run_command([
                "gtdbtk", "identify",
                "--genome_dir", str(fasta_dir),
                "--out_dir", str(identify_dir),
                "--cpus", str(max(1, int(threads))),
                "-x", "fasta",
            ], env=env)
            run_command([
                "gtdbtk", "align",
                "--identify_dir", str(identify_dir),
                "--out_dir", str(align_dir),
                "--cpus", str(max(1, int(threads))),
            ], env=env)
        except subprocess.CalledProcessError as exc:
            status_path = phylo_dir / "status.tsv"
            write_status(status_path, [{"status": "failed", "reason": str(exc)}])
            return [status_path]

    domain_specs = [
        (find_gtdb_user_msa(align_dir, "bac120"), phylo_dir / "bac120_alignment.fasta", phylo_dir / "bac120_tree.nwk", "bac120"),
        (find_gtdb_user_msa(align_dir, "ar53"), phylo_dir / "ar53_alignment.fasta", phylo_dir / "ar53_tree.nwk", "ar53"),
    ]
    seen_domains = set()
    for msa_path, align_copy_path, tree_path, domain_label in domain_specs:
        if domain_label in seen_domains or msa_path is None or not msa_path.exists():
            continue
        seen_domains.add(domain_label)
        n_sequences = read_fasta_count(msa_path)
        if n_sequences < 2:
            status_rows.append({"domain": domain_label, "status": "skipped", "reason": "fewer_than_two_aligned_genomes"})
            continue
        maybe_copy_gzip_fasta(msa_path, align_copy_path)
        try:
            run_command([fasttree_bin, str(align_copy_path)], stdout_path=str(tree_path))
            status_rows.append({"domain": domain_label, "status": "tree_built", "n_sequences": n_sequences})
            wrote.extend([align_copy_path, tree_path])
            if selected_df is not None:
                metadata_paths = write_tree_metadata(
                    tree_path,
                    selected_df,
                    cohort_name=Path(set_dir).name,
                    domain_label=domain_label,
                )
                wrote.extend(metadata_paths)
        except subprocess.CalledProcessError as exc:
            status_rows.append({"domain": domain_label, "status": "failed", "reason": str(exc)})
    if not status_rows:
        status_rows.append({"status": "skipped", "reason": "no_gtdb_user_msa_found"})
    status_path = phylo_dir / "status.tsv"
    write_status(status_path, status_rows)
    wrote.append(status_path)
    return wrote


def process_phylogeny_set(set_dir, threads, gtdbtk_data_path=None):
    set_dir = Path(set_dir).expanduser().absolute()
    selected_table = resolve_selected_table(set_dir)
    selected_df = read_tsv(selected_table)
    phylo_dir = set_dir / "phylogeny"
    ensure_dir(phylo_dir)
    if selected_df.empty:
        status_path = phylo_dir / "status.tsv"
        write_status(status_path, [{"status": "skipped", "reason": "no_selected_genomes"}])
        return [status_path]
    fasta_dir = ensure_selected_fastas(set_dir, selected_df)
    manifest_path = phylo_dir / "selected_manifest.tsv"
    manifest_df = build_selected_manifest(selected_df, fasta_dir, manifest_path)
    wrote = [manifest_path]
    wrote.extend(run_16s_phylogeny(set_dir, manifest_df, threads=threads))
    wrote.extend(run_gtdb_marker_phylogeny(
        set_dir,
        fasta_dir,
        threads=threads,
        gtdbtk_data_path=gtdbtk_data_path,
        selected_df=selected_df,
    ))
    return wrote


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build 16S and GTDB-marker phylogenies for selected genome-set directories under a wrapper output root."
    )
    parser.add_argument("root_dir", help="Wrapper output directory, denovo_phylogeny directory, or a specific selected-set directory.")
    parser.add_argument("--threads", type=int, default=1, help="Threads for MAFFT, Barrnap, and GTDB-Tk. Default: 1")
    parser.add_argument("--gtdbtk-data-path", default=None, help="Optional GTDBTK_DATA_PATH override for GTDB marker alignments.")
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(args.root_dir).expanduser().absolute()
    set_dirs = discover_phylogeny_set_dirs(root)
    if not set_dirs:
        raise FileNotFoundError(f"No selected-set directories found under {root}")
    log(f"[start] phylogeny set discovery: {len(set_dirs)} set(s)")
    wrote = []
    for index, set_dir in enumerate(set_dirs, start=1):
        log(f"[start] ({index}/{len(set_dirs)}) phylogeny for {set_dir}")
        paths = process_phylogeny_set(set_dir, threads=args.threads, gtdbtk_data_path=args.gtdbtk_data_path)
        wrote.extend([str(path) for path in paths])
        log(f"[done] ({index}/{len(set_dirs)}) phylogeny outputs: {set_dir / 'phylogeny'}")
    for path in wrote:
        print(path)


if __name__ == "__main__":
    main()
