#!/usr/bin/env python3

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import ssl
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES_MANIFEST = REPO_ROOT / "config" / "external_mapping_sources.tsv"
DEFAULT_METABOLISM_MANIFEST = REPO_ROOT / "config" / "metabolism_keyword_manifest.tsv"
HTTP_HEADERS = {
    "User-Agent": "genome_qc-metabolism-mapper/1.0 (+https://github.com/openai/codex)",
    "Accept": "*/*",
}
GOA_CLASSIFIED_COLUMNS = [
    "source_id",
    "accession",
    "symbol",
    "qualifier",
    "go_id",
    "go_name",
    "go_namespace",
    "evidence_code",
    "with_from",
    "aspect",
    "assigned_by",
    "gene_product_form_id",
    "direct_family_ids",
    "direct_family_labels",
    "direct_family_keywords",
    "mode_ids",
    "mode_labels",
    "mode_keywords",
    "resolved_family_ids",
    "resolved_family_labels",
    "metabolism_match",
]
GOA_PROGRESS_EVERY = 500000
GOA_NORMALIZED_BASENAME = "goa_uniprotkb_gaf_normalized.tsv.gz"
GOA_FILTERED_BASENAME = "goa_uniprotkb_metabolism_filtered.tsv.gz"
DEFAULT_GOA_BATCH_SIZE = 50000
DEFAULT_GOA_MAX_INFLIGHT = 4
_GOA_WORKER_CONFIG = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download official GO/GOA/UniProt mapping resources and format normalized "
            "metabolism-aware reference tables for reproducible annotation auditing."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(REPO_ROOT / "reference_mappings"),
        help="Output directory. Default: <repo>/reference_mappings",
    )
    parser.add_argument(
        "--sources-manifest",
        default=str(DEFAULT_SOURCES_MANIFEST),
        help="TSV listing mapping resources to download. Default: config/external_mapping_sources.tsv",
    )
    parser.add_argument(
        "--metabolism-manifest",
        default=str(DEFAULT_METABOLISM_MANIFEST),
        help="TSV listing metabolism families, modes, and keywords. Default: config/metabolism_keyword_manifest.tsv",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also download and process sources that are disabled by default, such as GOA UniProtKB GAF.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download. Reuse existing files in <output-dir>/raw and only rebuild formatted outputs.",
    )
    parser.add_argument(
        "--goa-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes for large GOA classification. "
            "Use >1 to parallelize the --include-large GOA pass. Default: 1"
        ),
    )
    parser.add_argument(
        "--goa-batch-size",
        type=int,
        default=DEFAULT_GOA_BATCH_SIZE,
        help=f"Rows per GOA classification batch. Default: {DEFAULT_GOA_BATCH_SIZE}",
    )
    parser.add_argument(
        "--goa-max-inflight",
        type=int,
        default=DEFAULT_GOA_MAX_INFLIGHT,
        help=f"Maximum in-flight GOA worker batches. Default: {DEFAULT_GOA_MAX_INFLIGHT}",
    )
    return parser.parse_args()


def progress(stage, message):
    print(f"[{stage}] {message}", flush=True)


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str).fillna("")


def load_sources_manifest(path, include_large=False):
    frame = read_tsv(path)
    required = {
        "source_id",
        "resource_type",
        "enabled_default",
        "downloadable",
        "url",
        "local_name",
        "parser",
        "description",
        "docs_url",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sources manifest missing columns: {', '.join(sorted(missing))}")

    frame["enabled_default"] = frame["enabled_default"].astype(str)
    frame["downloadable"] = frame["downloadable"].astype(str)
    if include_large:
        return frame
    return frame.loc[frame["enabled_default"].eq("1")].copy()


def load_metabolism_manifest(path):
    frame = read_tsv(path)
    required = {"family_id", "family_label", "mode_id", "mode_label", "keyword_scope", "keyword"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metabolism manifest missing columns: {', '.join(sorted(missing))}")

    family_order = []
    family_labels = {}
    family_keywords = {}
    mode_order = []
    mode_labels = {}
    mode_family = {}
    mode_keywords = {}

    for row in frame.to_dict("records"):
        family_id = row["family_id"].strip()
        family_label = row["family_label"].strip()
        mode_id = row["mode_id"].strip()
        mode_label = row["mode_label"].strip()
        scope = row["keyword_scope"].strip().lower()
        keyword = row["keyword"].strip().lower()
        if not family_id or not scope or not keyword:
            continue

        if family_id not in family_order:
            family_order.append(family_id)
        family_labels.setdefault(family_id, family_label)
        family_keywords.setdefault(family_id, [])

        if scope == "family":
            if keyword not in family_keywords[family_id]:
                family_keywords[family_id].append(keyword)
            continue

        if scope != "mode":
            raise ValueError(f"Unsupported keyword_scope '{scope}' in metabolism manifest")
        if not mode_id or not mode_label:
            raise ValueError("Mode rows in metabolism manifest require mode_id and mode_label")
        if mode_id not in mode_order:
            mode_order.append(mode_id)
        mode_labels.setdefault(mode_id, mode_label)
        mode_family.setdefault(mode_id, family_id)
        mode_keywords.setdefault(mode_id, [])
        if keyword not in mode_keywords[mode_id]:
            mode_keywords[mode_id].append(keyword)

    return family_order, family_labels, family_keywords, mode_order, mode_labels, mode_family, mode_keywords


def sha256sum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HTTP_HEADERS)
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, context=context) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)


def parse_go_obo(path):
    terms = {}
    current = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                if current.get("id"):
                    terms[current["id"]] = {
                        "go_name": current.get("name", ""),
                        "namespace": current.get("namespace", ""),
                    }
                current = {}
                continue
            if not line or line.startswith("!"):
                continue
            if line == "[Typedef]":
                break
            if ": " in line:
                key, value = line.split(": ", 1)
                if key in {"id", "name", "namespace"}:
                    current[key] = value
        if current.get("id"):
            terms[current["id"]] = {
                "go_name": current.get("name", ""),
                "namespace": current.get("namespace", ""),
            }
    return terms


def parse_external2go(path, source_id, go_lookup):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("!"):
                continue
            if "> GO:" not in line or " ; GO:" not in line:
                continue
            left, right = line.split("> GO:", 1)
            go_name, go_id = right.rsplit("; ", 1)
            left = left.strip()
            go_id = go_id.strip()
            xref_db = ""
            xref_id = ""
            xref_label = ""
            if ":" in left:
                xref_db, payload = left.split(":", 1)
                payload = payload.strip()
                if " " in payload:
                    xref_id, xref_label = payload.split(" ", 1)
                else:
                    xref_id = payload
            go_info = go_lookup.get(go_id, {})
            rows.append(
                {
                    "source_id": source_id,
                    "xref_db": xref_db.strip(),
                    "xref_id": xref_id.strip(),
                    "xref_label": xref_label.strip(),
                    "go_id": go_id,
                    "go_name": go_info.get("go_name", go_name.strip()),
                    "go_namespace": go_info.get("namespace", ""),
                    "mapping_text": line,
                }
            )
    return pd.DataFrame(rows)


def parse_gaf(path, go_lookup):
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("!"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            if parts[0] != "UniProtKB":
                continue
            go_id = parts[4].strip()
            go_info = go_lookup.get(go_id, {})
            rows.append(
                {
                    "source_id": "goa_uniprot_all_gaf",
                    "accession": parts[1].strip(),
                    "symbol": parts[2].strip(),
                    "qualifier": parts[3].strip(),
                    "go_id": go_id,
                    "go_name": go_info.get("go_name", ""),
                    "go_namespace": go_info.get("namespace", ""),
                    "evidence_code": parts[6].strip(),
                    "with_from": parts[7].strip(),
                    "aspect": parts[8].strip(),
                    "assigned_by": parts[14].strip(),
                    "gene_product_form_id": parts[16].strip(),
                }
            )
    return pd.DataFrame(rows)


def keyword_match_map(text, ordered_ids, keyword_map):
    normalized = str(text).strip().lower()
    if not normalized:
        return {}
    matches = {}
    for item_id in ordered_ids:
        hits = [keyword for keyword in keyword_map[item_id] if keyword in normalized]
        if hits:
            matches[item_id] = hits
    return matches


def elemental_families_from_modes(mode_ids, mode_family, family_order):
    families = {mode_family[mode_id] for mode_id in mode_ids if mode_id in mode_family}
    return [family_id for family_id in family_order if family_id in families]


def classify_record_to_metabolism(row, text_columns, family_order, family_labels, family_keywords, mode_order, mode_labels, mode_family, mode_keywords):
    family_hits = {}
    mode_hits = {}
    for column in text_columns:
        text = str(row.get(column, "")).strip()
        if not text:
            continue
        for family_id, keywords in keyword_match_map(text, family_order, family_keywords).items():
            family_hits.setdefault(family_id, [])
            family_hits[family_id].extend(keyword for keyword in keywords if keyword not in family_hits[family_id])
        for mode_id, keywords in keyword_match_map(text, mode_order, mode_keywords).items():
            mode_hits.setdefault(mode_id, [])
            mode_hits[mode_id].extend(keyword for keyword in keywords if keyword not in mode_hits[mode_id])

    mode_ids = [mode_id for mode_id in mode_order if mode_id in mode_hits]
    direct_family_ids = [family_id for family_id in family_order if family_id in family_hits]
    resolved_family_ids = []
    for family_id in direct_family_ids + elemental_families_from_modes(mode_ids, mode_family, family_order):
        if family_id not in resolved_family_ids:
            resolved_family_ids.append(family_id)

    return {
        **row,
        "direct_family_ids": ";".join(direct_family_ids),
        "direct_family_labels": ";".join(family_labels[family_id] for family_id in direct_family_ids),
        "direct_family_keywords": ";".join(
            keyword for family_id in direct_family_ids for keyword in family_hits.get(family_id, [])
        ),
        "mode_ids": ";".join(mode_ids),
        "mode_labels": ";".join(mode_labels[mode_id] for mode_id in mode_ids),
        "mode_keywords": ";".join(
            keyword for mode_id in mode_ids for keyword in mode_hits.get(mode_id, [])
        ),
        "resolved_family_ids": ";".join(resolved_family_ids),
        "resolved_family_labels": ";".join(family_labels[family_id] for family_id in resolved_family_ids),
        "metabolism_match": bool(resolved_family_ids or mode_ids),
    }


def classify_texts_to_metabolism(frame, text_columns, family_order, family_labels, family_keywords, mode_order, mode_labels, mode_family, mode_keywords):
    classified_rows = [
        classify_record_to_metabolism(
            row,
            text_columns=text_columns,
            family_order=family_order,
            family_labels=family_labels,
            family_keywords=family_keywords,
            mode_order=mode_order,
            mode_labels=mode_labels,
            mode_family=mode_family,
            mode_keywords=mode_keywords,
        )
        for row in frame.to_dict("records")
    ]
    return pd.DataFrame(classified_rows)


def init_goa_worker(family_order, family_labels, family_keywords, mode_order, mode_labels, mode_family, mode_keywords):
    global _GOA_WORKER_CONFIG
    _GOA_WORKER_CONFIG = {
        "family_order": family_order,
        "family_labels": family_labels,
        "family_keywords": family_keywords,
        "mode_order": mode_order,
        "mode_labels": mode_labels,
        "mode_family": mode_family,
        "mode_keywords": mode_keywords,
    }


def classify_goa_batch(rows):
    config = _GOA_WORKER_CONFIG
    classified_rows = [
        classify_record_to_metabolism(
            row,
            text_columns=["go_name"],
            family_order=config["family_order"],
            family_labels=config["family_labels"],
            family_keywords=config["family_keywords"],
            mode_order=config["mode_order"],
            mode_labels=config["mode_labels"],
            mode_family=config["mode_family"],
            mode_keywords=config["mode_keywords"],
        )
        for row in rows
    ]
    matched_rows = sum(1 for row in classified_rows if row["metabolism_match"])
    return classified_rows, matched_rows


def write_classified_rows(all_writer, filtered_writer, classified_rows):
    matched_rows = 0
    for row in classified_rows:
        all_writer.writerow(row)
        if row["metabolism_match"]:
            filtered_writer.writerow(row)
            matched_rows += 1
    return matched_rows


def build_gaf_row(parts, go_lookup):
    go_id = parts[4].strip()
    go_info = go_lookup.get(go_id, {})
    return {
        "source_id": "goa_uniprot_all_gaf",
        "accession": parts[1].strip(),
        "symbol": parts[2].strip(),
        "qualifier": parts[3].strip(),
        "go_id": go_id,
        "go_name": go_info.get("go_name", ""),
        "go_namespace": go_info.get("namespace", ""),
        "evidence_code": parts[6].strip(),
        "with_from": parts[7].strip(),
        "aspect": parts[8].strip(),
        "assigned_by": parts[14].strip(),
        "gene_product_form_id": parts[16].strip(),
    }


def stream_classify_gaf(path, go_lookup, output_all_path, output_filtered_path, family_order, family_labels, family_keywords, mode_order, mode_labels, mode_family, mode_keywords, workers=1, batch_size=DEFAULT_GOA_BATCH_SIZE, max_inflight=DEFAULT_GOA_MAX_INFLIGHT):
    progress("start", f"streaming GOA GAF from {path}")
    opener = gzip.open if str(path).endswith(".gz") else open
    with gzip.open(output_all_path, "wt", encoding="utf-8", newline="") as all_handle, gzip.open(
        output_filtered_path, "wt", encoding="utf-8", newline=""
    ) as filtered_handle:
        all_writer = csv.DictWriter(all_handle, fieldnames=GOA_CLASSIFIED_COLUMNS, delimiter="\t", extrasaction="ignore")
        filtered_writer = csv.DictWriter(filtered_handle, fieldnames=GOA_CLASSIFIED_COLUMNS, delimiter="\t", extrasaction="ignore")
        all_writer.writeheader()
        filtered_writer.writeheader()

        total_rows = 0
        written_rows = 0
        matched_rows = 0
        next_progress = GOA_PROGRESS_EVERY
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            if workers <= 1:
                for raw_line in handle:
                    if not raw_line or raw_line.startswith("!"):
                        continue
                    parts = raw_line.rstrip("\n").split("\t")
                    if len(parts) < 17 or parts[0] != "UniProtKB":
                        continue
                    row = build_gaf_row(parts, go_lookup)
                    classified = classify_record_to_metabolism(
                        row,
                        text_columns=["go_name"],
                        family_order=family_order,
                        family_labels=family_labels,
                        family_keywords=family_keywords,
                        mode_order=mode_order,
                        mode_labels=mode_labels,
                        mode_family=mode_family,
                        mode_keywords=mode_keywords,
                    )
                    all_writer.writerow(classified)
                    total_rows += 1
                    written_rows += 1
                    if classified["metabolism_match"]:
                        filtered_writer.writerow(classified)
                        matched_rows += 1
                    if written_rows >= next_progress:
                        progress(
                            "progress",
                            f"GOA rows written: {written_rows:,}; metabolism-matching rows: {matched_rows:,}",
                        )
                        next_progress += GOA_PROGRESS_EVERY
            else:
                progress(
                    "start",
                    f"using {workers} GOA worker processes with batch_size={batch_size:,} and max_inflight={max_inflight}",
                )
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=init_goa_worker,
                    initargs=(
                        family_order,
                        family_labels,
                        family_keywords,
                        mode_order,
                        mode_labels,
                        mode_family,
                        mode_keywords,
                    ),
                ) as executor:
                    inflight = {}
                    batch_rows = []
                    for raw_line in handle:
                        if not raw_line or raw_line.startswith("!"):
                            continue
                        parts = raw_line.rstrip("\n").split("\t")
                        if len(parts) < 17 or parts[0] != "UniProtKB":
                            continue
                        batch_rows.append(build_gaf_row(parts, go_lookup))
                        if len(batch_rows) >= batch_size:
                            future = executor.submit(classify_goa_batch, batch_rows)
                            inflight[future] = len(batch_rows)
                            total_rows += len(batch_rows)
                            batch_rows = []
                        while len(inflight) >= max_inflight:
                            done, _ = concurrent.futures.wait(
                                inflight.keys(),
                                return_when=concurrent.futures.FIRST_COMPLETED,
                            )
                            for future in done:
                                classified_rows, matched_count = future.result()
                                written_rows += len(classified_rows)
                                matched_rows += write_classified_rows(all_writer, filtered_writer, classified_rows)
                                inflight.pop(future, None)
                                while written_rows >= next_progress:
                                    progress(
                                        "progress",
                                        f"GOA rows written: {written_rows:,}; metabolism-matching rows: {matched_rows:,}",
                                    )
                                    next_progress += GOA_PROGRESS_EVERY
                    if batch_rows:
                        future = executor.submit(classify_goa_batch, batch_rows)
                        inflight[future] = len(batch_rows)
                        total_rows += len(batch_rows)
                    for future in concurrent.futures.as_completed(list(inflight.keys())):
                        classified_rows, matched_count = future.result()
                        written_rows += len(classified_rows)
                        matched_rows += write_classified_rows(all_writer, filtered_writer, classified_rows)
                        while written_rows >= next_progress:
                            progress(
                                "progress",
                                f"GOA rows written: {written_rows:,}; metabolism-matching rows: {matched_rows:,}",
                            )
                            next_progress += GOA_PROGRESS_EVERY
    progress(
        "done",
        f"GOA streaming complete: {written_rows:,} rows written; {matched_rows:,} metabolism-matching rows",
    )
    return {"total_rows": written_rows, "matched_rows": matched_rows}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_dir = output_dir / "raw"
    normalized_dir = output_dir / "normalized"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources_manifest(Path(args.sources_manifest).expanduser().resolve(), include_large=args.include_large)
    progress("start", f"loaded sources manifest with {len(sources)} candidate resources")
    (
        family_order,
        family_labels,
        family_keywords,
        mode_order,
        mode_labels,
        mode_family,
        mode_keywords,
    ) = load_metabolism_manifest(Path(args.metabolism_manifest).expanduser().resolve())
    progress(
        "done",
        f"loaded metabolism manifest with {len(family_order)} families and {len(mode_order)} modes",
    )

    downloaded_rows = []
    raw_paths = {}
    downloadable_rows = [row for row in sources.to_dict("records") if row["downloadable"] == "1"]
    progress("start", f"preparing {len(downloadable_rows)} downloadable resources")
    for index, row in enumerate(downloadable_rows, start=1):
        if row["downloadable"] != "1":
            continue
        destination = raw_dir / row["local_name"]
        raw_paths[row["source_id"]] = destination
        if not args.skip_download:
            progress(
                "start",
                f"({index}/{len(downloadable_rows)}) downloading {row['source_id']} -> {destination.name}",
            )
            try:
                download_file(row["url"], destination)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed downloading source '{row['source_id']}' from {row['url']}"
                ) from exc
            progress("done", f"downloaded {row['source_id']}")
        else:
            progress(
                "skip",
                f"({index}/{len(downloadable_rows)}) reusing existing {row['source_id']} from {destination}",
            )
        if not destination.exists():
            raise FileNotFoundError(f"Required raw file is missing: {destination}")
        progress("start", f"hashing {row['source_id']}")
        downloaded_rows.append(
            {
                **row,
                "local_path": str(destination),
                "sha256": sha256sum(destination),
                "size_bytes": destination.stat().st_size,
            }
        )
        progress(
            "done",
            f"{row['source_id']} size={destination.stat().st_size:,} bytes sha256={downloaded_rows[-1]['sha256'][:12]}...",
        )

    downloads_manifest = pd.DataFrame(downloaded_rows)
    downloads_manifest.to_csv(output_dir / "download_manifest.tsv", sep="\t", index=False)
    progress("done", f"wrote download manifest to {output_dir / 'download_manifest.tsv'}")

    progress("start", f"parsing GO ontology from {raw_paths['go_basic_obo']}")
    go_lookup = parse_go_obo(raw_paths["go_basic_obo"])
    progress("done", f"parsed {len(go_lookup):,} GO terms")

    external_frames = []
    external_rows = [row for row in downloadable_rows if row["parser"] == "external2go"]
    progress("start", f"parsing {len(external_rows)} external2go resources")
    for index, row in enumerate(external_rows, start=1):
        if row["downloadable"] != "1":
            continue
        if row["parser"] != "external2go":
            continue
        progress("start", f"({index}/{len(external_rows)}) parsing {row['source_id']}")
        external_frames.append(parse_external2go(raw_paths[row["source_id"]], row["source_id"], go_lookup))
        progress("done", f"{row['source_id']} rows={len(external_frames[-1]):,}")

    external2go = pd.concat(external_frames, ignore_index=True) if external_frames else pd.DataFrame()
    progress("done", f"combined external2go rows={len(external2go):,}")
    progress("start", "classifying external2go metabolism mappings")
    classified_external2go = classify_texts_to_metabolism(
        external2go,
        text_columns=["xref_label", "go_name"],
        family_order=family_order,
        family_labels=family_labels,
        family_keywords=family_keywords,
        mode_order=mode_order,
        mode_labels=mode_labels,
        mode_family=mode_family,
        mode_keywords=mode_keywords,
    )
    progress(
        "done",
        "classified external2go rows="
        f"{len(classified_external2go):,}; matches={int(classified_external2go['metabolism_match'].sum()):,}",
    )
    classified_external2go.to_csv(normalized_dir / "external2go_normalized.tsv", sep="\t", index=False)
    classified_external2go.loc[classified_external2go["metabolism_match"]].to_csv(
        normalized_dir / "external2go_metabolism_filtered.tsv",
        sep="\t",
        index=False,
    )
    progress("done", f"wrote {normalized_dir / 'external2go_normalized.tsv'}")
    progress("done", f"wrote {normalized_dir / 'external2go_metabolism_filtered.tsv'}")

    go_term_classification = (
        classified_external2go[["go_id", "go_name", "go_namespace", "direct_family_ids", "direct_family_labels", "direct_family_keywords", "mode_ids", "mode_labels", "mode_keywords", "resolved_family_ids", "resolved_family_labels", "metabolism_match"]]
        .drop_duplicates()
        .sort_values(["metabolism_match", "go_id"], ascending=[False, True])
    )
    go_term_classification.to_csv(normalized_dir / "go_term_metabolism_classification.tsv", sep="\t", index=False)
    progress("done", f"wrote {normalized_dir / 'go_term_metabolism_classification.tsv'}")

    if "goa_uniprot_all_gaf" in raw_paths:
        progress("start", "classifying large GOA UniProtKB GAF in streaming mode")
        stream_classify_gaf(
            raw_paths["goa_uniprot_all_gaf"],
            go_lookup,
            normalized_dir / GOA_NORMALIZED_BASENAME,
            normalized_dir / GOA_FILTERED_BASENAME,
            family_order=family_order,
            family_labels=family_labels,
            family_keywords=family_keywords,
            mode_order=mode_order,
            mode_labels=mode_labels,
            mode_family=mode_family,
            mode_keywords=mode_keywords,
            workers=max(1, int(args.goa_workers)),
            batch_size=max(1000, int(args.goa_batch_size)),
            max_inflight=max(1, int(args.goa_max_inflight)),
        )
        progress("done", f"wrote {normalized_dir / GOA_NORMALIZED_BASENAME}")
        progress("done", f"wrote {normalized_dir / GOA_FILTERED_BASENAME}")

    print(output_dir / "download_manifest.tsv")
    print(normalized_dir / "external2go_normalized.tsv")
    print(normalized_dir / "external2go_metabolism_filtered.tsv")
    print(normalized_dir / "go_term_metabolism_classification.tsv")
    if (normalized_dir / GOA_NORMALIZED_BASENAME).exists():
        print(normalized_dir / GOA_NORMALIZED_BASENAME)
        print(normalized_dir / GOA_FILTERED_BASENAME)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
