#!/usr/bin/env python3

import argparse
import csv
import hashlib
import glob
import os
import pickle
import re
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

plt = None


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_repo_asset_path(*parts):
    cwd_candidate = Path.cwd().joinpath(*parts)
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return SCRIPT_DIR.joinpath(*parts).resolve()


def resolve_optional_path_arg(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "false", "no", "disable", "disabled"}:
        return None
    return Path(text).expanduser().resolve()


ANNOTATED_PRODUCT_TOKENS = {
    "",
    "<unannotated protein>",
    "uncharacterized protein",
    "hypothetical protein",
}
FUNCTIONAL_SOURCE_COLUMNS = ["metacyc", "swissprot", "uniref50"]
FUNCTIONAL_TARGET_COLUMNS = [f"{column}_target" for column in FUNCTIONAL_SOURCE_COLUMNS]
FUNCTIONAL_SOURCE_DB = {"metacyc", "swissprot", "uniref50"}
ANNOTATION_CATEGORY_ORDER = [
    "informative",
    "hypothetical",
    "uncharacterized",
    "domain_family",
    "qualifier",
    "fragment",
    "other_uncertain",
    "unannotated",
]
ANNOTATION_CATEGORY_LABELS = {
    "informative": "Informative",
    "hypothetical": "Hypothetical",
    "uncharacterized": "Uncharacterized/unknown",
    "domain_family": "Domain/family/DUF",
    "qualifier": "Provisional annotation",
    "fragment": "Fragment",
    "other_uncertain": "Other uncertain",
    "unannotated": "Unannotated",
}
ANNOTATION_CATEGORY_COLORS = {
    "informative": "#1a1a1a",
    "hypothetical": "#404040",
    "uncharacterized": "#666666",
    "domain_family": "#8c8c8c",
    "qualifier": "#a6a6a6",
    "fragment": "#c0c0c0",
    "other_uncertain": "#d9d9d9",
    "unannotated": "#efefef",
}
UNCERTAIN_ANNOTATION_PATTERNS = {
    "hypothetical": ["hypothetical protein"],
    "uncharacterized": [
        "uncharacterized protein",
        "unknown protein",
        "protein of unknown function",
        "unknown function",
    ],
    "domain_family": [
        "domain-containing protein",
        "domain containing protein",
        "family protein",
        "duf",
    ],
    "qualifier": [
        "putative",
        "probable",
        "predicted",
        "possible",
        "like protein",
    ],
    "fragment": ["fragment"],
}
COMPACT_METRICS = [
    ("total_orfs", "Total ORFs"),
    ("annotated_orfs", "Functionally annotated ORFs"),
    ("pathway_support_orfs", "Pathway-supporting ORFs"),
    ("total_pathways", "Inferred pathways"),
]
TAXONOMY_RANK_ORDER = ["Species", "Genus", "Family", "Order", "Class", "Phylum", "Domain"]
TAXONOMY_FALLBACK_RANKS = ["Family", "Order", "Class", "Phylum", "Domain"]
TAXONOMY_MISSING_TOKENS = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "nan",
    "unknown",
    "unclassified",
    "unassigned",
    "uncultured",
    "unidentified",
}
SWISSPROT_TRAILING_FIELD_RE = re.compile(r"\s(?:OS|OX|GN|PE|SV)\s")
UNIREF_TRAILING_FIELD_RE = re.compile(r"\s(?:n|Tax|TaxID|RepID)=")
ELEMENTAL_CYCLE_ORDER = [
    "carbon_c1",
    "nitrogen",
    "sulfur",
    "phosphorus",
    "hydrogen",
    "iron",
    "arsenic",
]
ELEMENTAL_CYCLE_LABELS = {
    "carbon_c1": "Carbon / C1",
    "nitrogen": "Nitrogen",
    "sulfur": "Sulfur",
    "phosphorus": "Phosphorus",
    "hydrogen": "Hydrogen",
    "iron": "Iron",
    "arsenic": "Arsenic",
}
ELEMENTAL_CYCLE_COLORS = {
    "carbon_c1": "#1a1a1a",
    "nitrogen": "#3d3d3d",
    "sulfur": "#5c5c5c",
    "phosphorus": "#7a7a7a",
    "hydrogen": "#969696",
    "iron": "#b3b3b3",
    "arsenic": "#d0d0d0",
}
ELEMENTAL_MODE_ORDER = [
    "carbon_fixation",
    "methylotrophy_methane",
    "co_formate",
    "nitrogen_fixation",
    "nitrification",
    "nitrate_reduction",
    "nitrite_reduction",
    "denitrification",
    "dnra",
    "urea_cyanate",
    "sulfate_reduction",
    "sulfur_oxidation",
    "thiosulfate_tetrathionate",
    "organosulfur",
    "phosphate_scavenging",
    "phosphonate_phosphite",
    "hydrogen_metabolism",
    "iron_redox",
    "iron_acquisition",
    "arsenic_detox_redox",
]
ELEMENTAL_MODE_LABELS = {
    "carbon_fixation": "Carbon fixation",
    "methylotrophy_methane": "Methane/methanol/C1 use",
    "co_formate": "CO/formate metabolism",
    "nitrogen_fixation": "Nitrogen fixation",
    "nitrification": "Nitrification",
    "nitrate_reduction": "Nitrate reduction",
    "nitrite_reduction": "Nitrite reduction",
    "denitrification": "Denitrification",
    "dnra": "DNRA",
    "urea_cyanate": "Urea/cyanate use",
    "sulfate_reduction": "Sulfate reduction",
    "sulfur_oxidation": "Sulfur oxidation",
    "thiosulfate_tetrathionate": "Thiosulfate/tetrathionate",
    "organosulfur": "Organosulfur use",
    "phosphate_scavenging": "Phosphate scavenging",
    "phosphonate_phosphite": "Phosphonate/phosphite",
    "hydrogen_metabolism": "Hydrogen metabolism",
    "iron_redox": "Iron redox",
    "iron_acquisition": "Iron acquisition",
    "arsenic_detox_redox": "Arsenic detox/redox",
}
ELEMENTAL_MODE_FAMILY = {
    "carbon_fixation": "carbon_c1",
    "methylotrophy_methane": "carbon_c1",
    "co_formate": "carbon_c1",
    "nitrogen_fixation": "nitrogen",
    "nitrification": "nitrogen",
    "nitrate_reduction": "nitrogen",
    "nitrite_reduction": "nitrogen",
    "denitrification": "nitrogen",
    "dnra": "nitrogen",
    "urea_cyanate": "nitrogen",
    "sulfate_reduction": "sulfur",
    "sulfur_oxidation": "sulfur",
    "thiosulfate_tetrathionate": "sulfur",
    "organosulfur": "sulfur",
    "phosphate_scavenging": "phosphorus",
    "phosphonate_phosphite": "phosphorus",
    "hydrogen_metabolism": "hydrogen",
    "iron_redox": "iron",
    "iron_acquisition": "iron",
    "arsenic_detox_redox": "arsenic",
}
ELEMENTAL_MODE_KEYWORDS = {
    "carbon_fixation": [
        "rubisco",
        "ribulose-bisphosphate carboxylase",
        "carbon fixation",
        "calvin",
        "wood-ljungdahl",
        "acetyl-coa pathway",
        "reverse tca",
        "reductive citric acid cycle",
        "co dehydrogenase/acetyl-coa synthase",
        "acetyl-coa synthase co dehydrogenase",
    ],
    "methylotrophy_methane": [
        "methane monooxygenase",
        "methanol dehydrogenase",
        "methylamine dehydrogenase",
        "methylotroph",
        "methylotrophy",
        "methanotroph",
        "methanogenesis",
        "methane oxidation",
        "methanol",
        "methylotrophic",
    ],
    "co_formate": [
        "carbon monoxide dehydrogenase",
        "co dehydrogenase",
        "formate dehydrogenase",
        "formate oxidation",
    ],
    "nitrogen_fixation": [
        "nitrogenase",
        "nifh",
        "nifd",
        "nifk",
        "dinitrogenase",
    ],
    "nitrification": [
        "ammonia monooxygenase",
        "hydroxylamine oxidoreductase",
        "nitrite oxidoreductase",
        "amoa",
        "amob",
        "amoc",
        "hao",
        "nxra",
        "nxrb",
        "nitrif",
    ],
    "nitrate_reduction": [
        "nitrate reductase",
        "nargh",
        "napa",
        "nitrate reduction",
    ],
    "nitrite_reduction": [
        "nitrite reductase",
        "nirbd",
        "nrfa",
        "nirk",
        "nirs",
        "nitrite reduction",
    ],
    "denitrification": [
        "nitric oxide reductase",
        "nitrous-oxide reductase",
        "nitrous oxide reductase",
        "denitr",
        "norb",
        "norc",
        "nosz",
    ],
    "dnra": [
        "dissimilatory nitrate reduction to ammonium",
        "dnra",
        "nrfa",
        "nirb",
        "nir d",
    ],
    "urea_cyanate": [
        "urease",
        "urea carboxylase",
        "urea amidolyase",
        "urea transport",
        "cyanase",
        "cyanate",
    ],
    "sulfate_reduction": [
        "sulfate adenylyltransferase",
        "adenylylsulfate reductase",
        "dissimilatory sulfite reductase",
        "sat",
        "apra",
        "aprb",
        "dsrc",
        "dsra",
        "dsrb",
        "sulfate reduction",
    ],
    "sulfur_oxidation": [
        "sulfide:quinone oxidoreductase",
        "sulfide quinone oxidoreductase",
        "sulfur oxidation",
        "sulfur dioxygenase",
        "sulfite oxidase",
        "sulfite dehydrogenase",
        "sqr",
        "soxe",
        "soxf",
        "soxy",
        "soxz",
    ],
    "thiosulfate_tetrathionate": [
        "thiosulfate",
        "tetrathionate",
        "soxa",
        "soxb",
        "soxc",
        "soxd",
        "tsd",
    ],
    "organosulfur": [
        "sulfonate",
        "taurine",
        "dimethylsulf",
        "dmsp",
        "sulfoquinovose",
    ],
    "phosphate_scavenging": [
        "phosphate transport",
        "phosphate regulon",
        "alkaline phosphatase",
        "pho regulon",
        "phoa",
        "phod",
        "phox",
        "psts",
        "phosphate uptake",
    ],
    "phosphonate_phosphite": [
        "phosphonate",
        "phosphite",
        "2-aminoethylphosphonate",
        "phn",
        "ptx",
        "htx",
    ],
    "hydrogen_metabolism": [
        "hydrogenase",
        "hydrogen oxidation",
        "hydrogen production",
        "[fefe] hydrogenase",
        "[nife] hydrogenase",
        "[nife]-hydrogenase",
        "[nife]-hydrogenase",
        "hya",
        "hyb",
        "hyda",
        "hydb",
        "hox",
    ],
    "iron_redox": [
        "iron oxidation",
        "iron reduction",
        "ferric reductase",
        "ferrous iron",
        "cyc2",
        "mtra",
        "mtrb",
        "mtrc",
    ],
    "iron_acquisition": [
        "siderophore",
        "heme uptake",
        "heme transport",
        "iron transport",
        "ferric uptake",
        "fec",
        "tonb-dependent iron",
    ],
    "arsenic_detox_redox": [
        "arsenate reductase",
        "arsenite oxidase",
        "arsenic resistance",
        "arsenic detoxification",
        "arsm",
        "arsa",
        "arsb",
        "arsc",
        "aioa",
        "aiob",
        "arr",
    ],
}

# Experimental mobility screen note:
# This is intentionally kept separate from the reusable reference_mappings workflow.
# It currently runs as a lightweight annotation-text screen during write_outputs().
# If it proves useful, promote it into a maintained reusable resource/config path;
# otherwise it can be removed without changing the core metabolism reference system.
EXPERIMENTAL_MOBILITY_CATEGORY_ORDER = [
    "plasmid-associated",
    "phage/prophage-associated",
    "generic MGE-associated",
    "genomic-island-associated",
]
EXPERIMENTAL_MOBILITY_CATEGORY_COLUMNS = {
    "plasmid-associated": "plasmid_associated",
    "phage/prophage-associated": "phage_prophage_associated",
    "generic MGE-associated": "generic_mge_associated",
    "genomic-island-associated": "genomic_island_associated",
}
EXPERIMENTAL_MOBILITY_KEYWORDS = {
    "plasmid-associated": [
        "repa",
        "repb",
        "repc",
        "plasmid replication",
        "plasmid initiator",
        "relaxase",
        "type iv coupling protein",
        "t4cp",
        "vird4",
        "trb",
        "conjugation",
        "conjugal transfer",
        "mating pair formation",
        "mpf",
        "type iv secretion",
        "t4ss",
        "partition protein",
        "plasmid partition",
        "stability protein",
        "toxin antitoxin",
        "postsegregational killing",
    ],
    "phage/prophage-associated": [
        "terminase",
        "terminase large subunit",
        "terl",
        "portal protein",
        "major capsid",
        "capsid",
        "head protein",
        "prohead",
        "tail protein",
        "tail fiber",
        "tail sheath",
        "tail tube",
        "baseplate",
        "tape measure protein",
        "endolysin",
        "spanin",
        "prophage",
        "phage integrase",
        "capsid maturation protease",
        "phage protease",
        "dna packaging",
        "packaging atpase",
    ],
    "generic MGE-associated": [
        "integrase",
        "site-specific recombinase",
        "tyrosine recombinase",
        "serine recombinase",
        "recombinase",
        "excisionase",
        "xis",
        "transposase",
        "insertion sequence",
        "is element",
        "resolvase",
        "invertase",
        "conjugative transfer",
        "mobilization protein",
        "mobilization",
        "orit",
        "relaxosome",
        "type iv secretion",
        "t4ss",
        "integrative conjugative element",
        "ice",
        "ime",
        "conjugative element",
        "mobilizable element",
    ],
    "genomic-island-associated": [
        "genomic island",
        "pathogenicity island",
        "pai",
        "symbiosis island",
        "metabolic island",
        "resistance island",
        "integrative element",
        "integrase",
        "recombinase",
        "excisionase",
        "direct repeat",
        "attl",
        "attr",
        "insertion hotspot",
        "integration hotspot",
        "conjugation",
        "type iv secretion",
        "cargo gene cluster",
    ],
}
EXPERIMENTAL_MOBILITY_OPTIONAL_BROAD_KEYWORDS = [
    "mobile element protein",
    "hypothetical phage protein",
    "plasmid protein",
    "conjugation protein",
    "phage protein",
    "virus-like",
    "viral protein",
    "temperate phage",
    "lysogeny",
    "lysogenic",
]
EXPERIMENTAL_SHORT_TOKEN_KEYWORDS = {
    "rep",
    "tra",
    "ice",
    "ime",
    "pai",
    "xis",
    "attl",
    "attr",
    "orit",
    "mpf",
    "t4ss",
    "t4cp",
    "terl",
    "lysin",
    "holin",
    "repa",
    "repb",
    "repc",
}
ELEMENTAL_CYCLE_KEYWORDS = {
    "carbon_c1": [
        "carbon fixation",
        "calvin",
        "rubisco",
        "ribulose-bisphosphate carboxylase",
        "wood-ljungdahl",
        "acetyl-coa pathway",
        "reductive citric acid cycle",
        "reverse tca",
        "methane",
        "methanotroph",
        "methanogenesis",
        "methanol",
        "methylotroph",
        "methylotrophy",
        "carbon monoxide dehydrogenase",
        "formaldehyde",
    ],
    "nitrogen": [
        "nitrogen",
        "nitrate",
        "nitrite",
        "nitric oxide",
        "nitrous oxide",
        "ammonia",
        "ammonium",
        "hydroxylamine",
        "denitr",
        "nitrif",
        "anammox",
        "cyanate",
        "urea",
        "urease",
        "nitrogenase",
    ],
    "sulfur": [
        "sulfur",
        "sulfur",
        "sulfate",
        "sulfite",
        "sulfide",
        "thiosulfate",
        "tetrathionate",
        "sulfonate",
        "sulfoquinovose",
        "dimethylsulf",
        "taurine",
    ],
    "phosphorus": [
        "phosphate",
        "phosphonate",
        "phosphite",
        "polyphosphate",
        "phosphorus",
    ],
    "hydrogen": [
        "hydrogenase",
        "molecular hydrogen",
        "hydrogen oxidation",
        "hydrogen production",
    ],
    "iron": [
        "iron",
        "ferric",
        "ferrous",
        "iron oxidation",
        "iron reduction",
        "siderophore",
        "heme uptake",
    ],
    "arsenic": [
        "arsenic",
        "arsenate",
        "arsenite",
        "arsenic detoxification",
    ],
}


METABOLISM_MANIFEST_PATH = resolve_repo_asset_path("config", "metabolism_keyword_manifest.tsv")
MARKER_MANIFEST_PATH = resolve_repo_asset_path("config", "metabolism_marker_manifest.tsv")
DEFAULT_REFERENCE_MAPPINGS_DIR = resolve_repo_asset_path("reference_mappings")
UNIPROT_ACCESSION_REGEX = re.compile(
    r"(?<![A-Z0-9])(?:"
    r"UniRef\d+_([A-Z0-9]+)"
    r"|"
    r"([OPQ][0-9][A-Z0-9]{3}[0-9](?:-\d+)?)"
    r"|"
    r"([A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}(?:-\d+)?)"
    r")(?:_[A-Z0-9]+)?(?=$|[^A-Z0-9])"
)


def load_metabolism_manifest(manifest_path):
    family_order = []
    family_labels = {}
    family_colors = {}
    family_keywords = {}
    mode_order = []
    mode_labels = {}
    mode_family = {}
    mode_keywords = {}

    if not manifest_path.exists():
        raise FileNotFoundError(f"Metabolism keyword manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"family_id", "family_label", "family_color", "mode_id", "mode_label", "keyword_scope", "keyword"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Metabolism keyword manifest is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            family_id = str(row["family_id"]).strip()
            family_label = str(row["family_label"]).strip()
            family_color = str(row["family_color"]).strip()
            mode_id = str(row["mode_id"]).strip()
            mode_label = str(row["mode_label"]).strip()
            keyword_scope = str(row["keyword_scope"]).strip().lower()
            keyword = str(row["keyword"]).strip().lower()

            if not family_id or not family_label or not family_color or not keyword_scope or not keyword:
                continue

            if family_id not in family_order:
                family_order.append(family_id)
            family_labels.setdefault(family_id, family_label)
            family_colors.setdefault(family_id, family_color)
            family_keywords.setdefault(family_id, [])

            if keyword_scope == "family":
                if keyword not in family_keywords[family_id]:
                    family_keywords[family_id].append(keyword)
                continue

            if keyword_scope != "mode":
                raise ValueError(
                    f"Unsupported keyword_scope '{keyword_scope}' in metabolism keyword manifest: {manifest_path}"
                )

            if not mode_id or not mode_label:
                raise ValueError(
                    f"Mode rows must define mode_id and mode_label in metabolism keyword manifest: {manifest_path}"
                )

            if mode_id not in mode_order:
                mode_order.append(mode_id)
            mode_labels.setdefault(mode_id, mode_label)
            mode_family.setdefault(mode_id, family_id)
            mode_keywords.setdefault(mode_id, [])
            if keyword not in mode_keywords[mode_id]:
                mode_keywords[mode_id].append(keyword)

    for mode_id, family_id in mode_family.items():
        if family_id not in family_labels:
            raise ValueError(
                f"Mode '{mode_id}' refers to unknown family '{family_id}' in metabolism keyword manifest: {manifest_path}"
            )

    if not family_order or not mode_order:
        raise ValueError(f"Metabolism keyword manifest is empty or incomplete: {manifest_path}")

    return (
        family_order,
        family_labels,
        family_colors,
        family_keywords,
        mode_order,
        mode_labels,
        mode_family,
        mode_keywords,
    )


(
    ELEMENTAL_CYCLE_ORDER,
    ELEMENTAL_CYCLE_LABELS,
    ELEMENTAL_CYCLE_COLORS,
    ELEMENTAL_CYCLE_KEYWORDS,
    ELEMENTAL_MODE_ORDER,
    ELEMENTAL_MODE_LABELS,
    ELEMENTAL_MODE_FAMILY,
    ELEMENTAL_MODE_KEYWORDS,
) = load_metabolism_manifest(METABOLISM_MANIFEST_PATH)


def load_marker_manifest(manifest_path):
    marker_rows = read_table(manifest_path)
    required = {"family_id", "mode_id", "marker_id", "marker_label", "is_core", "alias"}
    missing = required.difference(marker_rows.columns)
    if missing:
        raise ValueError(f"Marker manifest missing columns: {', '.join(sorted(missing))}")

    marker_rows["family_id"] = normalize_text(marker_rows["family_id"]).str.lower()
    marker_rows["mode_id"] = normalize_text(marker_rows["mode_id"]).str.lower()
    marker_rows["marker_id"] = normalize_text(marker_rows["marker_id"])
    marker_rows["marker_label"] = normalize_text(marker_rows["marker_label"])
    marker_rows["alias"] = normalize_text(marker_rows["alias"]).str.lower()
    marker_rows["is_core"] = marker_rows["is_core"].astype(str).str.strip().isin({"1", "true", "True"})
    marker_rows = marker_rows.loc[
        marker_rows["family_id"].ne("")
        & marker_rows["mode_id"].ne("")
        & marker_rows["marker_id"].ne("")
        & marker_rows["alias"].ne("")
    ].copy()
    return marker_rows


def prepare_marker_matcher(marker_manifest):
    alias_to_rows = {}
    if marker_manifest is None or marker_manifest.empty:
        return alias_to_rows, None
    specificity_map = classify_marker_manifest_specificity(marker_manifest)
    marker_rows = marker_manifest.to_dict("records")
    marker_alias_sets = {}
    marker_label_text = {}
    for row in marker_rows:
        marker_key = (
            str(row.get("family_id", "")).strip().lower(),
            str(row.get("mode_id", "")).strip().lower(),
            str(row.get("marker_id", "")).strip().lower(),
        )
        alias = str(row.get("alias", "")).strip().lower()
        label_text = re.sub(r"\s+", " ", str(row.get("marker_label", "")).strip().lower())
        marker_label_text[marker_key] = label_text
        marker_alias_sets.setdefault(marker_key, set())
        if alias:
            marker_alias_sets[marker_key].add(alias)
        if label_text:
            marker_alias_sets[marker_key].add(label_text)

    for row in marker_rows:
        marker_key = (
            str(row.get("family_id", "")).strip().lower(),
            str(row.get("mode_id", "")).strip().lower(),
            str(row.get("marker_id", "")).strip().lower(),
        )
        alias = str(row.get("alias", "")).strip().lower()
        row["_marker_key"] = marker_key
        row["_marker_label_text"] = marker_label_text.get(marker_key, "")
        row["_specificity_class"] = specificity_map.get(
            (str(row.get("mode_id", "")).strip().lower(), str(row.get("marker_id", "")).strip()),
            "specific",
        )
        row["_descriptive_aliases"] = tuple(
            sorted(
                candidate
                for candidate in marker_alias_sets.get(marker_key, set())
                if is_descriptive_marker_alias(candidate)
            )
        )
        aliases_for_row = set()
        if alias:
            aliases_for_row.add(alias)
        if row["_marker_label_text"]:
            aliases_for_row.add(row["_marker_label_text"])
        for candidate_alias in aliases_for_row:
            alias_to_rows.setdefault(candidate_alias, []).append(row)
    if not alias_to_rows:
        return alias_to_rows, None
    # Marker aliases are gene/protein tokens, not arbitrary substrings.
    # Require non-alphanumeric boundaries so short aliases like `ccs` and
    # `ccl` do not fire inside tokens such as `CCS1` or `Ccl2`.
    escaped_aliases = [re.escape(alias) for alias in sorted(alias_to_rows.keys(), key=len, reverse=True)]
    alias_regex = re.compile(
        r"(?<![a-z0-9_])(?:"
        + "|".join(escaped_aliases)
        + r")(?![a-z0-9_])"
    )
    return alias_to_rows, alias_regex


def is_short_symbol_marker_alias(alias):
    token = str(alias or "").strip().lower()
    return bool(token) and len(token) <= 4 and re.fullmatch(r"[a-z0-9]+", token) is not None


def is_descriptive_marker_alias(alias):
    return not is_short_symbol_marker_alias(alias)


def is_symbolic_marker_id(marker_id):
    token = str(marker_id or "").strip()
    if not token:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", token)
    if not compact:
        return False
    return re.search(r"[A-Z]", token) is not None or is_short_symbol_marker_alias(compact.lower())


@lru_cache(maxsize=200000)
def contains_boundary_phrase(text, phrase):
    normalized_text = str(text or "").strip().lower()
    normalized_phrase = str(phrase or "").strip().lower()
    if not normalized_text or not normalized_phrase:
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9_]){re.escape(normalized_phrase)}(?![a-z0-9_])",
            normalized_text,
        )
    )


@lru_cache(maxsize=200000)
def short_alias_in_strong_context(text, alias):
    normalized_text = str(text or "").strip().lower()
    normalized_alias = str(alias or "").strip().lower()
    if not normalized_text or not normalized_alias:
        return False

    patterns = [
        rf"^{re.escape(normalized_alias)}$",
        rf"[\(\[]\s*{re.escape(normalized_alias)}\s*[\)\]]",
        rf"(?<![a-z0-9_]){re.escape(normalized_alias)}(?=\s*$)",
        rf"(?:(?<=^)|(?<=[/|,;+]))\s*{re.escape(normalized_alias)}\s*(?=$|[/|,;+])",
        rf"(?<![a-z0-9_]){re.escape(normalized_alias)}(?=\s+(protein|subunit|chain|enzyme|family|like)\b)",
        rf"\bgn\s*=?\s*{re.escape(normalized_alias)}\b",
        rf"\bgene(?:_name)?\s*[:=]?\s*{re.escape(normalized_alias)}\b",
        rf"\bgene\s+symbol\s*[:=]?\s*{re.escape(normalized_alias)}\b",
    ]
    return any(re.search(pattern, normalized_text) for pattern in patterns)


def marker_row_match_evidence(normalized_text, alias, marker_row):
    if not normalized_text or not alias or not marker_row:
        return ""
    if not is_short_symbol_marker_alias(alias):
        return "descriptive_alias"

    descriptive_aliases = marker_row.get("_descriptive_aliases", ())
    label_text = str(marker_row.get("_marker_label_text", "")).strip().lower()

    if label_text and contains_boundary_phrase(normalized_text, label_text):
        return "marker_label"
    if any(contains_boundary_phrase(normalized_text, candidate) for candidate in descriptive_aliases):
        return "descriptive_alias"
    if short_alias_in_strong_context(normalized_text, alias):
        return "short_alias"
    return ""


def marker_row_supported_by_annotation(normalized_text, alias, marker_row):
    return bool(marker_row_match_evidence(normalized_text, alias, marker_row))


def marker_specificity_rank(marker_id, marker_label, evidence_type):
    marker_token = str(marker_id or "").strip()
    label_text = str(marker_label or "").strip().lower()
    evidence = str(evidence_type or "").strip().lower()

    rank = 0
    if evidence == "short_alias":
        rank += 40
    elif evidence == "marker_label":
        rank += 30
    elif evidence == "descriptive_alias":
        rank += 20

    if label_text:
        for token in [
            "subunit",
            "alpha",
            "beta",
            "gamma",
            "delta",
            "epsilon",
            "zeta",
            "large subunit",
            "small subunit",
            "catalytic subunit",
            "chain",
            "component",
        ]:
            if token in label_text:
                rank += 8
        if "maturation" in label_text or "assembly" in label_text or "accessory" in label_text:
            rank -= 10

    normalized_id = re.sub(r"[^a-z0-9]", "", marker_token.lower())
    if is_symbolic_marker_id(marker_token):
        rank += 6
    rank += min(len(normalized_id), 12)
    return rank


def marker_row_more_specific(candidate_row, other_row):
    if not candidate_row or not other_row:
        return False
    if str(candidate_row.get("mode_id", "")) != str(other_row.get("mode_id", "")):
        return False
    if str(candidate_row.get("family_id", "")) != str(other_row.get("family_id", "")):
        return False
    if str(candidate_row.get("marker_id", "")) == str(other_row.get("marker_id", "")):
        return False

    candidate_id = re.sub(r"[^a-z0-9]", "", str(candidate_row.get("marker_id", "")).lower())
    other_id = re.sub(r"[^a-z0-9]", "", str(other_row.get("marker_id", "")).lower())
    candidate_label = re.sub(r"\s+", " ", str(candidate_row.get("marker_label", "")).strip().lower())
    other_label = re.sub(r"\s+", " ", str(other_row.get("marker_label", "")).strip().lower())
    candidate_evidence = str(candidate_row.get("match_evidence_type", "")).strip().lower()
    other_evidence = str(other_row.get("match_evidence_type", "")).strip().lower()

    if other_label and candidate_label and other_label != candidate_label and contains_boundary_phrase(candidate_label, other_label):
        return True
    if candidate_label and other_label and candidate_label == other_label:
        if (
            candidate_id
            and other_id
            and candidate_id.startswith(other_id)
            and len(candidate_id) > len(other_id)
            and candidate_evidence == "short_alias"
        ):
            return True
        if (
            is_symbolic_marker_id(candidate_row.get("marker_id", ""))
            and not is_symbolic_marker_id(other_row.get("marker_id", ""))
            and candidate_evidence == "short_alias"
        ):
            return True
        if candidate_evidence == "short_alias" and other_evidence in {"marker_label", "descriptive_alias"}:
            return True
        return False
    if candidate_id and other_id and candidate_id.startswith(other_id) and len(candidate_id) > len(other_id):
        return True

    candidate_rank = marker_specificity_rank(
        candidate_row.get("marker_id", ""),
        candidate_row.get("marker_label", ""),
        candidate_evidence,
    )
    other_rank = marker_specificity_rank(
        other_row.get("marker_id", ""),
        other_row.get("marker_label", ""),
        other_evidence,
    )
    return candidate_rank > other_rank and (
        candidate_rank - other_rank >= 12
        or (candidate_evidence == "short_alias" and other_evidence != "short_alias")
    )


def choose_generic_marker_row(marker_rows):
    if not marker_rows:
        return None
    return sorted(
        marker_rows,
        key=lambda row: (
            is_symbolic_marker_id(row.get("marker_id", "")),
            len(re.sub(r"[^a-z0-9]", "", str(row.get("marker_id", "")).lower())),
            marker_specificity_rank(
                row.get("marker_id", ""),
                row.get("marker_label", ""),
                row.get("match_evidence_type", ""),
            ),
            str(row.get("marker_id", "")),
        ),
    )[0]


def manifest_marker_more_specific(candidate_row, other_row):
    if not candidate_row or not other_row:
        return False
    if str(candidate_row.get("mode_id", "")) != str(other_row.get("mode_id", "")):
        return False
    if str(candidate_row.get("marker_id", "")) == str(other_row.get("marker_id", "")):
        return False

    candidate_id = re.sub(r"[^a-z0-9]", "", str(candidate_row.get("marker_id", "")).lower())
    other_id = re.sub(r"[^a-z0-9]", "", str(other_row.get("marker_id", "")).lower())
    candidate_label = re.sub(r"\s+", " ", str(candidate_row.get("marker_label", "")).strip().lower())
    other_label = re.sub(r"\s+", " ", str(other_row.get("marker_label", "")).strip().lower())

    if other_label and candidate_label and other_label != candidate_label and contains_boundary_phrase(candidate_label, other_label):
        return True
    if candidate_label and other_label and candidate_label == other_label:
        if candidate_id and other_id and candidate_id.startswith(other_id) and len(candidate_id) > len(other_id):
            return True
        if is_symbolic_marker_id(candidate_row.get("marker_id", "")) and not is_symbolic_marker_id(other_row.get("marker_id", "")):
            return True
        return False
    if candidate_id and other_id and candidate_id.startswith(other_id) and len(candidate_id) > len(other_id):
        return True
    return False


def classify_marker_manifest_specificity(marker_manifest):
    specificity = {}
    if marker_manifest is None or marker_manifest.empty:
        return specificity

    records = marker_manifest[["mode_id", "marker_id", "marker_label"]].drop_duplicates().to_dict("records")
    for row in records:
        key = (str(row.get("mode_id", "")).strip().lower(), str(row.get("marker_id", "")).strip())
        specificity[key] = "specific"

    for row in records:
        key = (str(row.get("mode_id", "")).strip().lower(), str(row.get("marker_id", "")).strip())
        if any(manifest_marker_more_specific(other, row) for other in records):
            specificity[key] = "generic"
    return specificity


def resolve_marker_specificity(marker_rows):
    if not marker_rows:
        return []

    ambiguous_same_label_rows = set()
    grouped_by_label = {}
    for row in marker_rows:
        label_key = re.sub(r"\s+", " ", str(row.get("marker_label", "")).strip().lower())
        grouped_by_label.setdefault(label_key, []).append(row)

    for label_rows in grouped_by_label.values():
        unique_markers = {
            str(row.get("marker_id", "")).strip().lower()
            for row in label_rows
            if str(row.get("marker_id", "")).strip()
        }
        if len(unique_markers) <= 1:
            continue
        if any(str(row.get("match_evidence_type", "")).strip().lower() == "short_alias" for row in label_rows):
            continue
        non_symbolic_rows = [
            row for row in label_rows
            if not is_symbolic_marker_id(row.get("marker_id", ""))
        ]
        if non_symbolic_rows:
            generic_row = choose_generic_marker_row(non_symbolic_rows)
            for row in label_rows:
                if row is not generic_row:
                    ambiguous_same_label_rows.add(id(row))
            continue
        marker_ids = [
            re.sub(r"[^a-z0-9]", "", str(row.get("marker_id", "")).lower())
            for row in label_rows
        ]
        prefix_generic_rows = []
        for row, marker_id in zip(label_rows, marker_ids):
            if marker_id and all(other_id.startswith(marker_id) for other_id in marker_ids if other_id and other_id != marker_id):
                prefix_generic_rows.append(row)
        if len(prefix_generic_rows) == 1:
            generic_row = prefix_generic_rows[0]
            for row in label_rows:
                if row is not generic_row:
                    ambiguous_same_label_rows.add(id(row))
            continue
        for row in label_rows:
            ambiguous_same_label_rows.add(id(row))

    resolved_rows = []
    for row in marker_rows:
        if id(row) in ambiguous_same_label_rows:
            row_copy = row.copy()
            row_copy["match_resolution"] = "suppressed_same_label_ambiguous"
            resolved_rows.append(row_copy)
            continue
        keep = True
        for other in marker_rows:
            if row is other:
                continue
            if id(other) in ambiguous_same_label_rows:
                continue
            if marker_row_more_specific(other, row):
                keep = False
                break
        row_copy = row.copy()
        row_copy["match_resolution"] = "retained" if keep else "suppressed_by_specific_match"
        resolved_rows.append(row_copy)
    return [row for row in resolved_rows if row.get("match_resolution") == "retained"]


def load_reference_term_maps(reference_mappings_dir):
    family_terms = {family_id: [] for family_id in ELEMENTAL_CYCLE_ORDER}
    mode_terms = {mode_id: [] for mode_id in ELEMENTAL_MODE_ORDER}
    if reference_mappings_dir is None:
        return family_terms, mode_terms

    reference_dir = Path(reference_mappings_dir).expanduser().resolve()
    go_terms_path = reference_dir / "normalized" / "go_term_metabolism_classification.tsv"
    external2go_path = reference_dir / "normalized" / "external2go_metabolism_filtered.tsv"
    frames = []
    if go_terms_path.exists():
        frames.append(read_table(go_terms_path))
    if external2go_path.exists():
        frames.append(read_table(external2go_path))
    if not frames:
        return family_terms, mode_terms

    combined = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    text_columns = [column for column in ["go_name", "xref_label"] if column in combined.columns]
    for row in combined.to_dict("records"):
        texts = [str(row.get(column, "")).strip().lower() for column in text_columns if str(row.get(column, "")).strip()]
        if not texts:
            continue
        family_ids = [token for token in str(row.get("resolved_family_ids", "")).split(";") if token]
        mode_ids = [token for token in str(row.get("mode_ids", "")).split(";") if token]
        for family_id in family_ids:
            if family_id in family_terms:
                for text in texts:
                    if text and text not in family_terms[family_id]:
                        family_terms[family_id].append(text)
        for mode_id in mode_ids:
            if mode_id in mode_terms:
                for text in texts:
                    if text and text not in mode_terms[mode_id]:
                        mode_terms[mode_id].append(text)
    return family_terms, mode_terms


def merged_term_map(base_map, extra_map):
    merged = {key: list(values) for key, values in base_map.items()}
    for key, values in extra_map.items():
        merged.setdefault(key, [])
        for value in values:
            if value not in merged[key]:
                merged[key].append(value)
    return merged


ACTIVE_ELEMENTAL_CYCLE_TERMS = {key: list(values) for key, values in ELEMENTAL_CYCLE_KEYWORDS.items()}
ACTIVE_ELEMENTAL_MODE_TERMS = {key: list(values) for key, values in ELEMENTAL_MODE_KEYWORDS.items()}


def configure_reference_term_maps(reference_mappings_dir=None):
    global ACTIVE_ELEMENTAL_CYCLE_TERMS, ACTIVE_ELEMENTAL_MODE_TERMS
    family_terms, mode_terms = load_reference_term_maps(reference_mappings_dir)
    ACTIVE_ELEMENTAL_CYCLE_TERMS = merged_term_map(ELEMENTAL_CYCLE_KEYWORDS, family_terms)
    ACTIVE_ELEMENTAL_MODE_TERMS = merged_term_map(ELEMENTAL_MODE_KEYWORDS, mode_terms)
    classify_elemental_cycles.cache_clear()
    classify_elemental_modes.cache_clear()


def normalize_uniprot_accession(accession):
    token = str(accession).strip().upper()
    if not token:
        return ""
    if token.startswith("UNIREF") and "_" in token:
        token = token.split("_", 1)[1]
    if "_" in token:
        token = token.split("_", 1)[0]
    return token.split("-", 1)[0]


@lru_cache(maxsize=200000)
def _extract_uniprot_accessions_cached(text):
    normalized = str(text).strip()
    if not normalized:
        return tuple()
    accessions = []
    seen = set()
    for match in UNIPROT_ACCESSION_REGEX.finditer(normalized):
        token = next((group for group in match.groups() if group), "")
        accession = normalize_uniprot_accession(token)
        if accession and accession not in seen:
            seen.add(accession)
            accessions.append(accession)
    return tuple(accessions)


def extract_uniprot_accessions(text):
    return list(_extract_uniprot_accessions_cached(str(text)))


def accession_set_hash(accessions):
    digest = hashlib.sha1()
    for accession in sorted({str(token).strip().upper() for token in accessions if str(token).strip()}):
        digest.update(accession.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def collect_target_accessions_from_orf_lookup(
    orf_annotation_lookup,
    progress=False,
    progress_interval=100000,
    workers=1,
):
    if orf_annotation_lookup is None or orf_annotation_lookup.empty:
        return set()

    targets = set()
    scan_interval = max(1, int(progress_interval))
    worker_count = max(1, int(workers))

    def extract_chunk(texts):
        chunk_hits = set()
        for text in texts:
            for accession in extract_uniprot_accessions(text):
                if accession:
                    chunk_hits.add(accession)
        return chunk_hits

    columns_to_scan = list(FUNCTIONAL_SOURCE_COLUMNS) + list(FUNCTIONAL_TARGET_COLUMNS)
    for column in columns_to_scan:
        if column not in orf_annotation_lookup.columns:
            continue
        values = normalize_text(orf_annotation_lookup[column])
        unique_values = [str(text) for text in pd.unique(values[values.ne("")].astype(str))]
        if not unique_values:
            continue
        if worker_count <= 1 or len(unique_values) < 5000:
            scanned = 0
            for text in unique_values:
                scanned += 1
                for accession in extract_uniprot_accessions(text):
                    if accession:
                        targets.add(accession)
                if scanned % scan_interval == 0:
                    progress_log(
                        f"[progress] scanning ORF annotations for accessions [{column}]: "
                        f"texts={scanned:,}; unique_accessions={len(targets):,}",
                        enabled=progress,
                    )
        else:
            chunk_size = max(1000, int(np.ceil(len(unique_values) / float(worker_count * 8))))
            chunks = [unique_values[index:index + chunk_size] for index in range(0, len(unique_values), chunk_size)]
            stage = StageProgress(
                label=f"scanning ORF annotations for accessions [{column}]",
                total=len(chunks),
                workers=worker_count,
                interval_seconds=10,
                enabled=progress,
            )
            stage.start_stage()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(extract_chunk, chunk) for chunk in chunks]
                done = 0
                for future in as_completed(futures):
                    targets.update(future.result())
                    done += 1
                    stage.update(done)
            stage.done()
    return targets


def load_reference_accession_modes(
    reference_mappings_dir,
    progress=False,
    progress_interval_rows=2_000_000,
    chunk_size=500_000,
    target_accessions=None,
    workers=1,
):
    if reference_mappings_dir is None:
        return {}
    reference_dir = Path(reference_mappings_dir).expanduser().resolve()
    normalized_dir = reference_dir / "normalized" if (reference_dir / "normalized").exists() else reference_dir
    goa_candidates = [
        normalized_dir / "goa_uniprotkb_metabolism_filtered.tsv.gz",
        normalized_dir / "goa_uniprotkb_metabolism_filtered.tsv",
    ]
    goa_path = next((path for path in goa_candidates if path.exists()), None)
    if goa_path is None:
        return {}
    full_cache_path = goa_path.parent / f"{goa_path.stem}.accession_mode_cache.pkl"
    target_set = None
    subset_cache_path = None
    if target_accessions is not None:
        target_set = {normalize_uniprot_accession(token) for token in target_accessions if str(token).strip()}
        target_set = {token for token in target_set if token}
        target_hash = accession_set_hash(target_set) if target_set else "empty"
        subset_cache_path = goa_path.parent / f"{goa_path.stem}.accession_mode_cache.subset_{target_hash}.pkl"
        progress_log(
            f"[start] target accession mode enabled: targets={len(target_set):,}",
            enabled=progress,
        )
        if not target_set:
            progress_log("[done] no target accessions found; skipping GOA load", enabled=progress)
            return {}

    def _load_cache(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    def _cache_fresh(path):
        return path.exists() and path.stat().st_mtime >= goa_path.stat().st_mtime

    def _write_cache_atomic(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    try:
        if subset_cache_path is not None and _cache_fresh(subset_cache_path):
            progress_log(f"[start] loading targeted accession lookup cache: {subset_cache_path}", enabled=progress)
            cached = _load_cache(subset_cache_path)
            progress_log(
                f"[done] loaded targeted accession lookup cache: accessions={len(cached):,}",
                enabled=progress,
            )
            return cached
        if _cache_fresh(full_cache_path):
            progress_log(f"[start] loading accession lookup cache: {full_cache_path}", enabled=progress)
            cached = _load_cache(full_cache_path)
            if target_set is None:
                progress_log(
                    f"[done] loaded accession lookup cache: accessions={len(cached):,}",
                    enabled=progress,
                )
                return cached
            subset = {accession: cached[accession] for accession in target_set if accession in cached}
            progress_log(
                f"[done] subset from full accession cache: targets={len(target_set):,}; "
                f"matched={len(subset):,}",
                enabled=progress,
            )
            try:
                if subset_cache_path is not None:
                    _write_cache_atomic(subset_cache_path, subset)
                    progress_log(f"[done] wrote targeted accession cache: {subset_cache_path}", enabled=progress)
            except Exception as exc:
                progress_log(
                    f"[warn] unable to write targeted accession cache ({subset_cache_path}): {exc}",
                    enabled=progress,
                )
            return subset
    except Exception as exc:
        progress_log("[warn] ignoring stale/invalid accession cache; rebuilding.", enabled=progress)
        progress_log(f"[warn] cache error detail: {exc}", enabled=progress)
        for candidate in [subset_cache_path, full_cache_path]:
            if candidate is None:
                continue
            try:
                if candidate.exists():
                    candidate.unlink()
                    progress_log(f"[warn] removed invalid cache file: {candidate}", enabled=progress)
            except Exception as remove_exc:
                progress_log(
                    f"[warn] unable to remove invalid cache file ({candidate}): {remove_exc}",
                    enabled=progress,
                )

    progress_log(
        f"[start] streaming accession-mode reference table: {goa_path} "
        f"(chunk_size={int(chunk_size):,})",
        enabled=progress,
    )
    required_columns = {"accession", "qualifier", "mode_ids", "resolved_family_ids", "go_id", "go_name"}
    reader = pd.read_csv(
        goa_path,
        sep="\t",
        dtype=str,
        low_memory=False,
        chunksize=max(1, int(chunk_size)),
        usecols=lambda column: column in required_columns,
    )

    accession_lookup = {}
    worker_count = max(1, int(workers))
    rows_processed = 0
    rows_retained = 0
    next_report = max(1, int(progress_interval_rows))

    def parse_rows_to_partial(accessions, mode_vals, family_vals, go_id_vals, go_name_vals):
        partial = {}
        for accession, mode_raw, family_raw, go_id_raw, go_name_raw in zip(
            accessions,
            mode_vals,
            family_vals,
            go_id_vals,
            go_name_vals,
        ):
            entry = partial.get(accession)
            if entry is None:
                entry = {
                    "mode_ids": set(),
                    "family_ids": set(),
                    "go_ids": set(),
                    "go_names": set(),
                }
                partial[accession] = entry
            for token in mode_raw.split(";"):
                token = token.strip()
                if token:
                    entry["mode_ids"].add(token)
            for token in family_raw.split(";"):
                token = token.strip()
                if token:
                    entry["family_ids"].add(token)
            go_id = go_id_raw.strip()
            if go_id:
                entry["go_ids"].add(go_id)
            go_name = go_name_raw.strip()
            if go_name:
                entry["go_names"].add(go_name)
        return partial

    def merge_partial(partial):
        for accession, entry in partial.items():
            current = accession_lookup.get(accession)
            if current is None:
                accession_lookup[accession] = entry
                continue
            current["mode_ids"].update(entry["mode_ids"])
            current["family_ids"].update(entry["family_ids"])
            current["go_ids"].update(entry["go_ids"])
            current["go_names"].update(entry["go_names"])

    for chunk in reader:
        if chunk.empty:
            continue
        rows_processed += len(chunk)
        for column in required_columns:
            if column not in chunk.columns:
                chunk[column] = ""
        if "qualifier" in chunk.columns:
            chunk = chunk.loc[
                ~chunk["qualifier"].fillna("").astype(str).str.contains(r"\bNOT\b", case=False, regex=True)
            ].copy()
        if chunk.empty:
            continue
        chunk["accession"] = chunk["accession"].fillna("").astype(str).map(normalize_uniprot_accession)
        chunk = chunk.loc[chunk["accession"].ne("")]
        if target_set is not None:
            chunk = chunk.loc[chunk["accession"].isin(target_set)]
        if chunk.empty:
            continue
        rows_retained += len(chunk)

        accessions = chunk["accession"].astype(str).tolist()
        mode_vals = chunk["mode_ids"].fillna("").astype(str).tolist()
        family_vals = chunk["resolved_family_ids"].fillna("").astype(str).tolist()
        go_id_vals = chunk["go_id"].fillna("").astype(str).tolist()
        go_name_vals = chunk["go_name"].fillna("").astype(str).tolist()
        if worker_count <= 1 or len(accessions) < 50000:
            merge_partial(parse_rows_to_partial(accessions, mode_vals, family_vals, go_id_vals, go_name_vals))
        else:
            sub_chunk_size = max(10000, int(np.ceil(len(accessions) / float(worker_count * 6))))
            ranges = [(start, min(start + sub_chunk_size, len(accessions))) for start in range(0, len(accessions), sub_chunk_size)]
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        parse_rows_to_partial,
                        accessions[start:end],
                        mode_vals[start:end],
                        family_vals[start:end],
                        go_id_vals[start:end],
                        go_name_vals[start:end],
                    )
                    for start, end in ranges
                ]
                for future in as_completed(futures):
                    merge_partial(future.result())

        if rows_processed >= next_report:
            matched_accessions = len(accession_lookup)
            progress_log(
                f"[progress] accession-mode reference rows processed: {rows_processed:,}; "
                f"retained_rows: {rows_retained:,}; unique accessions: {matched_accessions:,}",
                enabled=progress,
            )
            next_report += max(1, int(progress_interval_rows))

    finalized = {}
    for accession, entry in accession_lookup.items():
        finalized[accession] = {
            "mode_ids": sorted(entry["mode_ids"]),
            "family_ids": sorted(entry["family_ids"]),
            "go_ids": sorted(entry["go_ids"]),
            "go_names": sorted(entry["go_names"]),
        }
    progress_log(
        f"[done] accession-mode reference loaded: rows={rows_processed:,}; retained_rows={rows_retained:,}; "
        f"unique accessions={len(finalized):,}",
        enabled=progress,
    )
    try:
        cache_path = subset_cache_path if subset_cache_path is not None else full_cache_path
        _write_cache_atomic(cache_path, finalized)
        progress_log(f"[done] wrote accession lookup cache: {cache_path}", enabled=progress)
    except Exception as exc:
        cache_path = subset_cache_path if subset_cache_path is not None else full_cache_path
        progress_log(f"[warn] unable to write accession lookup cache ({cache_path}): {exc}", enabled=progress)
    return finalized


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize MetaPathways genome or MAG outputs into per-genome ORF, "
            "annotation, and pathway summary tables."
        )
    )
    parser.add_argument(
        "results_dirs",
        nargs="*",
        help=(
            "One or more MetaPathways results directories. "
            "Shell-expanded globs are supported, e.g. path/*/results."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to <results_dir>/genome_function_summary for one input, "
            "or <common-parent>/genome_function_summary for multiple inputs."
        ),
    )
    parser.add_argument(
        "--also-write-individual-outputs",
        action="store_true",
        help=(
            "When multiple results directories are provided, also write a full per-input summary "
            "set to each input results directory."
        ),
    )
    parser.add_argument(
        "--individual-output-subdir",
        default="genome_function_summary",
        help=(
            "Subdirectory under each input results directory used with "
            "--also-write-individual-outputs. Default: genome_function_summary"
        ),
    )
    parser.add_argument(
        "--prefix",
        default="metapathways_genomes",
        help="Output file prefix. Default: metapathways_genomes",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Pathway score threshold used for high-confidence summaries. Default: 0.8",
    )
    parser.add_argument(
        "--top-n-pathways",
        type=int,
        default=20,
        help="Maximum number of pathways to show in the pathway heatmap. Default: 20",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Thread count for per-genome summarization work. "
            "Use 1 for fully serial execution; 2-8 is typically a good low-memory range. Default: 1"
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "Global thread count override for all major threaded stages. "
            "If >0, this value is used for both --workers and --prep-workers."
        ),
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        default=0,
        help=(
            "Worker threads for heavy preprocessing stages (ORF annotation lookup and accession extraction). "
            "Use 0 to reuse --workers. Default: 0"
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Progress update interval in seconds for long stages. Default: 10",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress/status logging and print only output file paths.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=60,
        help="Emit a 'still running' update for long steps at this interval (seconds). Default: 60",
    )
    parser.add_argument(
        "--genome-filter-tsv",
        default=None,
        help=(
            "Optional atlas/master-style TSV used to restrict which genomes are summarized. "
            "Useful with genome_quality_annotated.tsv."
        ),
    )
    parser.add_argument(
        "--filter-id-column",
        default=None,
        help=(
            "Optional genome ID column in --genome-filter-tsv. "
            "Defaults to auto-detecting Bin Id, genome_id, or Genome_Id."
        ),
    )
    parser.add_argument(
        "--filter-tier-column",
        default="mimag_tier",
        help="Tier column in --genome-filter-tsv. Default: mimag_tier",
    )
    parser.add_argument(
        "--include-tiers",
        default="medium,high",
        help="Comma-separated tiers to keep from --genome-filter-tsv. Default: medium,high",
    )
    parser.add_argument(
        "--taxonomy-label-tsv",
        default=None,
        help=(
            "Optional atlas/master-style TSV used to replace genome IDs in plot labels with taxonomy labels. "
            "If not set, --genome-filter-tsv is used when available."
        ),
    )
    parser.add_argument(
        "--taxonomy-id-column",
        default=None,
        help=(
            "Optional genome ID column in --taxonomy-label-tsv. "
            "Defaults to --filter-id-column, then auto-detect Bin Id/genome_id/Genome_Id."
        ),
    )
    parser.add_argument(
        "--marker-manifest",
        default=str(MARKER_MANIFEST_PATH),
        help="TSV listing marker genes and aliases per metabolism mode. Default: config/metabolism_marker_manifest.tsv",
    )
    parser.add_argument(
        "--reference-mappings-dir",
        default=str(DEFAULT_REFERENCE_MAPPINGS_DIR),
        help=(
            "Optional directory containing normalized official mapping tables built by "
            "scripts/build_metabolism_reference_mappings.py. Default: <repo>/reference_mappings"
        ),
    )
    parser.add_argument(
        "--reference-chunk-size",
        type=int,
        default=500000,
        help="Rows per chunk when streaming GOA accession reference mappings. Default: 500000",
    )
    parser.add_argument(
        "--reference-progress-rows",
        type=int,
        default=2000000,
        help="Progress update interval (rows) while loading GOA accession mappings. Default: 2000000",
    )
    parser.add_argument(
        "--reference-force-full-index",
        action="store_true",
        help=(
            "Force loading/building the full accession reference lookup (ignore input-derived target accession subsetting)."
        ),
    )
    parser.add_argument(
        "--reference-index-only",
        action="store_true",
        help=(
            "Build/load accession reference index cache(s) and exit without processing MetaPathways results. "
            "Useful to pre-index before batch runs."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-screen",
        action="store_true",
        help=(
            "Run a separate experimental candidate mobility marker screen from annotation text "
            "and write dedicated tables/plots."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-tsv",
        default=None,
        help=(
            "Optional metadata TSV used to resolve category labels for the "
            "experimental mobility screen."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-column",
        default=None,
        help=(
            "Column in --experimental-mobility-genome-type-tsv containing category labels. "
            "If omitted, common column names are tried."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-genome-type-id-column",
        default=None,
        help=(
            "Genome ID column in --experimental-mobility-genome-type-tsv. "
            "If omitted, common ID columns are tried."
        ),
    )
    parser.add_argument(
        "--experimental-mobility-include-broad-screen",
        action="store_true",
        help=(
            "Also write separate optional broad-screen mobility tables using the broad keyword list. "
            "These are kept out of the main prevalence table and figure."
        ),
    )
    return parser


def read_table(path):
    return pd.read_csv(path, sep="\t", low_memory=False)


def ensure_plotting():
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Plotting dependencies could not be imported. Install matplotlib in the runtime environment."
        ) from exc
    plt = plt_mod
    return plt


def sanitize_label(value):
    token = str(value).strip()
    allowed = []
    for char in token:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "unknown"


def normalize_text(series):
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
    )


def normalize_annotation_defline(text, source_db):
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"\s+", " ", value)
    source = str(source_db or "").strip().lower()

    if source == "swissprot":
        gene_name_match = re.search(r"\bGN\s*=?\s*([A-Za-z0-9_.-]+)\b", value, flags=re.IGNORECASE)
        gene_name = gene_name_match.group(1).strip() if gene_name_match else ""
        match = re.search(r"\s(?:OS|OX|PE|SV)\s", value)
        if match:
            value = value[: match.start()].strip()
        if gene_name and not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(gene_name)}(?![A-Za-z0-9_])",
            value,
            flags=re.IGNORECASE,
        ):
            value = f"{value} GN {gene_name}".strip()
        value = re.sub(r"\s+", " ", value)
        return value

    if source == "uniref50":
        value = re.sub(r"^UniRef\d+_[^\s]+\s+", "", value)
        value = re.sub(r"^(Cluster:|Multispecies:)\s*", "", value, flags=re.IGNORECASE)
        match = UNIREF_TRAILING_FIELD_RE.search(value)
        if match:
            value = value[: match.start()].strip()
        return value

    if source == "metacyc":
        value = re.sub(r"\s+\[EC[:=][^\]]+\]\s*$", "", value)
        value = re.sub(r"\s+\|.*$", "", value)
        return value.strip()

    return value


def infer_functional_source_db(target_value):
    target = str(target_value or "").strip()
    if not target:
        return ""
    lowered = target.lower()
    if lowered.startswith(("sp|", "tr|")):
        return "swissprot"
    if lowered.startswith("uniref"):
        return "uniref50"
    if lowered.startswith("gnl|meta|"):
        return "metacyc"
    return ""


def split_csv_arg(value):
    return [item.strip() for item in str(value).split(",") if item.strip()]


def expand_results_dirs(raw_inputs):
    expanded = []
    for raw in raw_inputs:
        token = str(raw).strip()
        if not token:
            continue
        path = Path(token).expanduser()
        if path.exists():
            expanded.append(path.resolve())
            continue

        matches = sorted(glob.glob(str(path), recursive=True))
        if not matches:
            raise FileNotFoundError(f"Input path/pattern did not match any directories: {token}")
        for match in matches:
            resolved = Path(match).expanduser().resolve()
            if not resolved.exists():
                continue
            expanded.append(resolved)

    unique = []
    seen = set()
    for path in expanded:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def default_output_dir_for_results(results_dirs):
    if not results_dirs:
        raise ValueError("No results directories provided.")
    if len(results_dirs) == 1:
        return results_dirs[0] / "genome_function_summary"
    common_root = Path(os.path.commonpath([str(path) for path in results_dirs]))
    return common_root / "genome_function_summary"


def progress_log(message, enabled=True):
    if enabled:
        print(message, flush=True)


def run_with_heartbeat(
    label,
    func,
    enabled=True,
    heartbeat_seconds=60,
    emit_start=True,
    emit_done=True,
):
    if not enabled:
        return func()
    heartbeat_seconds = max(1, int(heartbeat_seconds))
    if emit_start:
        progress_log(f"[start] {label}", enabled=True)
    start = time.time()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        while True:
            try:
                result = future.result(timeout=heartbeat_seconds)
                elapsed = time.time() - start
                if emit_done:
                    progress_log(f"[done] {label} in {elapsed/60:.2f}m", enabled=True)
                return result
            except FuturesTimeoutError:
                elapsed = time.time() - start
                progress_log(
                    f"[progress] {label}: still running ({elapsed/60:.1f}m elapsed)",
                    enabled=True,
                )


class StageProgress:
    def __init__(self, label, total, workers=1, interval_seconds=10, enabled=True):
        self.label = str(label)
        self.total = max(0, int(total))
        self.workers = max(1, int(workers))
        self.interval_seconds = max(1, int(interval_seconds))
        self.enabled = enabled
        self.start = time.time()
        self.last_emit = self.start

    def start_stage(self):
        progress_log(
            f"[start] {self.label}: total={self.total:,} workers={self.workers}",
            enabled=self.enabled,
        )

    def update(self, done):
        if not self.enabled:
            return
        done = max(0, min(int(done), self.total if self.total > 0 else int(done)))
        now = time.time()
        if done < self.total and (now - self.last_emit) < self.interval_seconds:
            return
        elapsed = max(1e-9, now - self.start)
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - done)
        eta_seconds = (remaining / rate) if rate > 0 else float("inf")
        eta_text = f"{eta_seconds/60:.1f}m" if eta_seconds != float("inf") else "NA"
        pct = (100.0 * done / self.total) if self.total > 0 else 100.0
        progress_log(
            f"[progress] {self.label}: {done:,}/{self.total:,} ({pct:.1f}%) "
            f"rate={rate:.2f}/s eta={eta_text} workers={self.workers}",
            enabled=self.enabled,
        )
        self.last_emit = now

    def done(self):
        if not self.enabled:
            return
        elapsed = max(1e-9, time.time() - self.start)
        rate = (self.total / elapsed) if self.total > 0 else 0.0
        progress_log(
            f"[done] {self.label}: {self.total:,}/{self.total:,} in {elapsed/60:.2f}m "
            f"(avg {rate:.2f}/s; workers={self.workers})",
            enabled=self.enabled,
        )


def id_aliases(value):
    cleaned = str(value).strip()
    if not cleaned:
        return set()

    aliases = {cleaned}
    stem = cleaned
    for suffix in [".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            aliases.add(stem)
            break

    if "." in stem:
        parts = stem.split(".")
        for index in range(len(parts) - 1, 0, -1):
            aliases.add(".".join(parts[:index]))

    match = re.match(r"^(bin_\d+)\b", stem)
    if match:
        aliases.add(match.group(1))

    return {alias for alias in aliases if alias}


def find_result_files(base_dir, pattern):
    base = Path(base_dir)
    direct = sorted(base.glob(pattern))
    if direct:
        return [path for path in direct if path.is_file()]
    recursive = sorted(base.rglob(pattern))
    return [path for path in recursive if path.is_file()]


def resolve_pgdb_genome_dir(path_obj):
    path_obj = Path(path_obj)
    parts = path_obj.parts
    if "tables" in parts and "other" in parts:
        tables_index = parts.index("tables")
        if tables_index > 0:
            return Path(*parts[:tables_index])
    return path_obj.parent


def parse_genes_dat(path):
    gene_ids = []
    current_gene = None
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "//":
                current_gene = None
                continue
            if line.startswith("UNIQUE-ID - "):
                current_gene = line.split(" - ", 1)[1].strip()
            elif line.startswith("ACCESSION-1 - ") and current_gene is None:
                current_gene = line.split(" - ", 1)[1].strip()
            if current_gene is not None:
                gene_ids.append(current_gene)
                current_gene = None
    return sorted(set(gene_ids))


def _parallel_map_unique_texts(
    unique_texts,
    mapper_fn,
    workers=1,
    progress=False,
    label="mapping",
):
    unique_texts = [str(text) for text in unique_texts if str(text)]
    if not unique_texts:
        return {}
    worker_count = max(1, int(workers))
    if worker_count <= 1 or len(unique_texts) < 1000:
        stage = StageProgress(label=label, total=len(unique_texts), workers=1, interval_seconds=10, enabled=progress)
        stage.start_stage()
        mapped = {}
        for index, text in enumerate(unique_texts, start=1):
            mapped[text] = mapper_fn(text)
            stage.update(index)
        stage.done()
        return mapped

    chunk_size = max(1000, int(np.ceil(len(unique_texts) / float(worker_count * 8))))
    chunks = [unique_texts[index:index + chunk_size] for index in range(0, len(unique_texts), chunk_size)]
    stage = StageProgress(
        label=label,
        total=len(chunks),
        workers=worker_count,
        interval_seconds=10,
        enabled=progress,
    )
    stage.start_stage()

    def map_chunk(text_chunk):
        return {text: mapper_fn(text) for text in text_chunk}

    mapped = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(map_chunk, chunk) for chunk in chunks]
        done = 0
        for future in as_completed(futures):
            mapped.update(future.result())
            done += 1
            stage.update(done)
    stage.done()
    return mapped


def _tuple_ids_to_mask(ids_tuple, bit_lookup):
    mask = np.uint64(0)
    for token in ids_tuple:
        bit = bit_lookup.get(token)
        if bit is not None:
            mask |= bit
    return mask


def _labels_from_mask(mask_value, ordered_items, bit_values):
    value = int(mask_value)
    if value == 0:
        return ""
    labels = []
    for item, bit in zip(ordered_items, bit_values):
        if value & bit:
            labels.append(item)
    return ";".join(labels)


def build_orf_annotation_lookup(results_dir, workers=1, progress=False):
    annotation_dir = results_dir / "annotation_table"
    functional_frame = None
    annotation_frame = None

    functional_files = find_result_files(annotation_dir, "*.functional_and_taxonomic_table.txt")
    for functional_file in functional_files:
        try:
            candidate = pd.read_csv(functional_file, sep="\t", low_memory=False).copy()
        except pd.errors.EmptyDataError:
            continue
        id_column = "# ORF_ID" if "# ORF_ID" in candidate.columns else "ORF_ID"
        if id_column not in candidate.columns or "product" not in candidate.columns:
            continue
        candidate["target"] = normalize_text(candidate.get("target", pd.Series("", index=candidate.index)))
        candidate["product"] = normalize_text(candidate.get("product", pd.Series("", index=candidate.index)))
        candidate["selected_source_db"] = candidate["target"].map(infer_functional_source_db)
        for column in FUNCTIONAL_SOURCE_COLUMNS:
            candidate[column] = ""
            candidate[f"{column}_target"] = ""
        for column in FUNCTIONAL_SOURCE_COLUMNS:
            source_mask = candidate["selected_source_db"].eq(column)
            if source_mask.any():
                candidate.loc[source_mask, column] = candidate.loc[source_mask, "product"].map(
                    lambda text: normalize_annotation_defline(text, column)
                )
                candidate.loc[source_mask, f"{column}_target"] = candidate.loc[source_mask, "target"]
        candidate["orf_id"] = normalize_text(candidate[id_column])
        functional_frame = candidate
        break

    annotation_files = find_result_files(annotation_dir, "*.ORF_annotation_table.txt")
    for annotation_file in annotation_files:
        try:
            candidate = pd.read_csv(annotation_file, sep="\t", low_memory=False).copy()
        except pd.errors.EmptyDataError:
            continue
        id_column = "# ORF_ID" if "# ORF_ID" in candidate.columns else "ORF_ID"
        if id_column not in candidate.columns:
            continue
        candidate["orf_id"] = normalize_text(candidate[id_column])
        annotation_frame = candidate
        break

    if annotation_frame is not None:
        frame = annotation_frame.copy()
        for column in FUNCTIONAL_SOURCE_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
            target_column = f"{column}_target"
            if target_column not in frame.columns:
                frame[target_column] = ""
            frame[column] = normalize_text(frame[column]).map(
                lambda text: normalize_annotation_defline(text, column)
            )
            frame[target_column] = normalize_text(frame[target_column])

        if functional_frame is not None:
            for column in FUNCTIONAL_SOURCE_COLUMNS:
                source_rows = functional_frame.loc[
                    functional_frame["selected_source_db"].eq(column),
                    ["orf_id", column, f"{column}_target"],
                ].drop_duplicates("orf_id")
                if source_rows.empty:
                    continue
                source_rows = source_rows.set_index("orf_id")
                source_values = source_rows[column].to_dict()
                target_values = source_rows[f"{column}_target"].to_dict()
                missing_source_mask = frame[column].eq("")
                if missing_source_mask.any():
                    frame.loc[missing_source_mask, column] = (
                        frame.loc[missing_source_mask, "orf_id"].map(source_values).fillna("")
                    )
                missing_target_mask = frame[target_column].eq("")
                if missing_target_mask.any():
                    frame.loc[missing_target_mask, target_column] = (
                        frame.loc[missing_target_mask, "orf_id"].map(target_values).fillna("")
                    )
    elif functional_frame is not None:
        frame = functional_frame.copy()
    else:
        return None

    worker_count = max(1, int(workers))
    source_category = {}
    source_mode_mask = {}
    source_cycle_mask = {}
    mode_bits = {mode: np.uint64(1 << index) for index, mode in enumerate(ELEMENTAL_MODE_ORDER)}
    cycle_bits = {cycle: np.uint64(1 << index) for index, cycle in enumerate(ELEMENTAL_CYCLE_ORDER)}

    for column in FUNCTIONAL_SOURCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        target_column = f"{column}_target"
        if target_column not in frame.columns:
            frame[target_column] = ""
        frame[column] = normalize_text(frame[column]).map(lambda text: normalize_annotation_defline(text, column))
        frame[target_column] = normalize_text(frame[target_column])
        frame[f"has_{column}"] = frame[column].ne("")
        unique_texts = pd.unique(frame.loc[frame[column].ne(""), column].astype(str))
        category_map = _parallel_map_unique_texts(
            unique_texts,
            classify_annotation_text,
            workers=worker_count,
            progress=progress,
            label=f"annotation category mapping [{column}]",
        )
        source_category[column] = frame[column].map(category_map).fillna("unannotated")

        mode_tuple_map = _parallel_map_unique_texts(
            unique_texts,
            classify_elemental_modes,
            workers=worker_count,
            progress=progress,
            label=f"elemental mode mapping [{column}]",
        )
        mode_mask_map = {
            text: _tuple_ids_to_mask(mode_tuple_map.get(text, ()), mode_bits)
            for text in unique_texts
        }
        source_mode_mask[column] = frame[column].map(mode_mask_map).fillna(np.uint64(0)).astype(np.uint64)

        cycle_tuple_map = _parallel_map_unique_texts(
            unique_texts,
            classify_elemental_cycles,
            workers=worker_count,
            progress=progress,
            label=f"elemental cycle mapping [{column}]",
        )
        cycle_mask_map = {
            text: _tuple_ids_to_mask(cycle_tuple_map.get(text, ()), cycle_bits)
            for text in unique_texts
        }
        source_cycle_mask[column] = frame[column].map(cycle_mask_map).fillna(np.uint64(0)).astype(np.uint64)

    frame["annotated_orf"] = frame[[f"has_{column}" for column in FUNCTIONAL_SOURCE_COLUMNS]].any(axis=1)
    frame["unannotated_orf"] = ~frame["annotated_orf"]
    frame["source_count"] = frame[[f"has_{column}" for column in FUNCTIONAL_SOURCE_COLUMNS]].sum(axis=1)

    annotated_mask = frame["annotated_orf"].to_numpy(dtype=bool)
    annotation_category = np.full(len(frame), "unannotated", dtype=object)
    annotation_category[annotated_mask] = "other_uncertain"

    informative_mask = np.zeros(len(frame), dtype=bool)
    for column in FUNCTIONAL_SOURCE_COLUMNS:
        informative_mask |= source_category[column].to_numpy(dtype=object) == "informative"
    annotation_category[informative_mask] = "informative"
    for category in ["hypothetical", "uncharacterized", "domain_family", "qualifier", "fragment"]:
        category_mask = np.zeros(len(frame), dtype=bool)
        for column in FUNCTIONAL_SOURCE_COLUMNS:
            category_mask |= source_category[column].to_numpy(dtype=object) == category
        assign_mask = annotated_mask & ~informative_mask & category_mask
        annotation_category[assign_mask] = category

    frame["annotation_category"] = annotation_category
    frame["annotation_is_informative"] = frame["annotation_category"].eq("informative")
    frame["annotation_is_uncertain"] = frame["annotated_orf"] & ~frame["annotation_is_informative"]

    mode_mask = np.zeros(len(frame), dtype=np.uint64)
    cycle_mask = np.zeros(len(frame), dtype=np.uint64)
    for column in FUNCTIONAL_SOURCE_COLUMNS:
        mode_mask |= source_mode_mask[column].to_numpy(dtype=np.uint64)
        cycle_mask |= source_cycle_mask[column].to_numpy(dtype=np.uint64)

    family_mode_bits = {}
    for family_id in ELEMENTAL_CYCLE_ORDER:
        bit_sum = np.uint64(0)
        for mode_id, mapped_family in ELEMENTAL_MODE_FAMILY.items():
            if mapped_family == family_id and mode_id in mode_bits:
                bit_sum |= mode_bits[mode_id]
        family_mode_bits[family_id] = bit_sum
    for family_id in ELEMENTAL_CYCLE_ORDER:
        mapped_mode_bits = family_mode_bits.get(family_id, np.uint64(0))
        if mapped_mode_bits == 0:
            continue
        cycle_mask[(mode_mask & mapped_mode_bits) != 0] |= cycle_bits[family_id]

    mode_series = pd.Series(mode_mask, index=frame.index)
    cycle_series = pd.Series(cycle_mask, index=frame.index)
    mode_bit_values = [int(mode_bits[mode]) for mode in ELEMENTAL_MODE_ORDER]
    cycle_bit_values = [int(cycle_bits[cycle]) for cycle in ELEMENTAL_CYCLE_ORDER]
    mode_label_lookup = {
        int(mask_value): _labels_from_mask(mask_value, ELEMENTAL_MODE_ORDER, mode_bit_values)
        for mask_value in pd.unique(mode_series)
    }
    cycle_label_lookup = {
        int(mask_value): _labels_from_mask(mask_value, ELEMENTAL_CYCLE_ORDER, cycle_bit_values)
        for mask_value in pd.unique(cycle_series)
    }
    frame["elemental_mode_labels"] = mode_series.map(lambda value: mode_label_lookup.get(int(value), ""))
    frame["elemental_cycle_labels"] = cycle_series.map(lambda value: cycle_label_lookup.get(int(value), ""))

    for mode_id in ELEMENTAL_MODE_ORDER:
        frame[f"elemental_mode_{mode_id}"] = (mode_series.to_numpy(dtype=np.uint64) & mode_bits[mode_id]) != 0
    for cycle_id in ELEMENTAL_CYCLE_ORDER:
        frame[f"elemental_{cycle_id}"] = (cycle_series.to_numpy(dtype=np.uint64) & cycle_bits[cycle_id]) != 0
    return frame.drop_duplicates("orf_id").set_index("orf_id")


def build_ptinput_lookup(results_dir):
    annotation_dir = results_dir / "annotation_table"
    ptinput_files = find_result_files(annotation_dir, "*.ptinput.tsv")
    if not ptinput_files:
        return None

    ptinput = None
    for ptinput_file in ptinput_files:
        try:
            candidate = read_table(ptinput_file).copy()
        except pd.errors.EmptyDataError:
            continue
        if "orf_id" not in candidate.columns:
            continue
        ptinput = candidate
        break
    if ptinput is None:
        return None

    ptinput["orf_id"] = normalize_text(ptinput["orf_id"])
    ptinput["sourcedb"] = normalize_text(ptinput.get("sourcedb", pd.Series("", index=ptinput.index))).str.lower()
    ptinput["product"] = normalize_text(ptinput.get("product", pd.Series("", index=ptinput.index)))
    ptinput["target"] = normalize_text(ptinput.get("target", pd.Series("", index=ptinput.index)))
    ptinput["ec"] = normalize_text(ptinput.get("ec", pd.Series("", index=ptinput.index)))
    ptinput["taxon"] = normalize_text(ptinput.get("taxon", pd.Series("", index=ptinput.index)))
    ptinput["feature"] = normalize_text(ptinput.get("feature", pd.Series("", index=ptinput.index))).str.lower()
    ptinput["annotated_product"] = ~ptinput["product"].str.lower().isin(ANNOTATED_PRODUCT_TOKENS)
    ptinput["has_target"] = ptinput["target"].ne("") & ptinput["target"].ne("<no accession>")
    ptinput["has_ec"] = ptinput["ec"].ne("")
    ptinput["has_taxon"] = ptinput["taxon"].ne("")
    ptinput["is_annotated"] = ptinput["annotated_product"] | ptinput["has_target"] | ptinput["has_ec"]
    ptinput["is_unannotated"] = ~ptinput["is_annotated"]
    ptinput["is_partial"] = normalize_text(ptinput.get("partial", pd.Series("", index=ptinput.index))).ne("")
    ptinput["is_cds"] = ptinput["feature"].eq("cds")
    ptinput["has_functional_source"] = ptinput["sourcedb"].isin(FUNCTIONAL_SOURCE_DB)
    ptinput["is_nonfunctional_feature"] = ~ptinput["is_cds"]
    return ptinput.drop_duplicates("orf_id").set_index("orf_id")


def infer_results_mode(path_obj):
    return "metagenome" if "MAGs" in Path(path_obj).parts else "genome"


def infer_genome_id_from_results_dir(results_dir):
    annotation_dir = results_dir / "annotation_table"
    functional_files = find_result_files(annotation_dir, "*.functional_and_taxonomic_table.txt")
    if functional_files:
        return functional_files[0].name[: -len(".functional_and_taxonomic_table.txt")]
    annotation_files = find_result_files(annotation_dir, "*.ORF_annotation_table.txt")
    if annotation_files:
        return annotation_files[0].name[: -len(".ORF_annotation_table.txt")]

    ptinput_files = find_result_files(annotation_dir, "*.ptinput.tsv")
    if ptinput_files:
        return ptinput_files[0].name[: -len(".ptinput.tsv")]

    results_dir = Path(results_dir)
    if results_dir.name == "results" and results_dir.parent.name:
        return results_dir.parent.name
    return results_dir.name


def build_empty_pathway_table(genome_id):
    columns = [
        "genome_id",
        "PWY_NAME",
        "PWY_COMMON_NAME",
        "PWY_SCORE",
        "NUM_REACTIONS",
        "NUM_COVERED_REACTIONS",
        "ORF_COUNT",
        "reaction_coverage_fraction",
        "high_confidence_pathway",
        "complete_pathway",
        "elemental_mode_labels",
        "elemental_cycle_labels",
    ]
    for mode in ELEMENTAL_MODE_ORDER:
        columns.append(f"elemental_mode_{mode}")
    for cycle in ELEMENTAL_CYCLE_ORDER:
        columns.append(f"elemental_{cycle}")
    return pd.DataFrame(columns=columns).assign(genome_id=pd.Series(dtype=str))


def build_genome_record(
    genome_id,
    mode,
    genome_dir,
    pwy_path=None,
    pwy2orf_path=None,
    genes_dat_path=None,
    pathway_status="present",
    pathway_missing_reason="",
):
    return {
        "genome_id": genome_id,
        "mode": mode,
        "genome_dir": Path(genome_dir),
        "pwy_path": Path(pwy_path) if pwy_path else None,
        "pwy2orf_path": Path(pwy2orf_path) if pwy2orf_path else None,
        "genes_dat_path": Path(genes_dat_path) if genes_dat_path else None,
        "pathway_status": pathway_status,
        "pathway_missing_reason": pathway_missing_reason,
    }


def find_genome_records(results_dir):
    pgdb_dir = results_dir / "pgdb"
    records_by_id = {}
    explicit_pathway_record_count = 0

    def upsert_record(record):
        genome_id = str(record["genome_id"])
        existing = records_by_id.get(genome_id)
        if existing is None:
            records_by_id[genome_id] = record
            return

        for key in ["pwy_path", "pwy2orf_path", "genes_dat_path"]:
            if existing.get(key) is None and record.get(key) is not None:
                existing[key] = record[key]
        if existing.get("genome_dir") is None and record.get("genome_dir") is not None:
            existing["genome_dir"] = record["genome_dir"]
        if existing.get("mode") in {None, ""} and record.get("mode"):
            existing["mode"] = record["mode"]
        if existing.get("pathway_status") != "present" and record.get("pathway_status") == "present":
            existing["pathway_status"] = "present"
            existing["pathway_missing_reason"] = ""
        elif not existing.get("pathway_missing_reason") and record.get("pathway_missing_reason"):
            existing["pathway_missing_reason"] = record["pathway_missing_reason"]

    if pgdb_dir.exists():
        pathway_files = sorted(pgdb_dir.glob("**/*_pwy.tsv"))
        for pwy_path in pathway_files:
            if "community" in pwy_path.parts:
                continue
            genome_id = pwy_path.name[: -len("_pwy.tsv")]
            genome_dir = resolve_pgdb_genome_dir(pwy_path)
            pwy2orf_candidates = [
                genome_dir / f"{genome_id}_pwy2orf.tsv",
                genome_dir / "tables" / "other" / f"{genome_id}_pwy2orf.tsv",
            ]
            pwy2orf_path = next((path for path in pwy2orf_candidates if path.exists()), None)
            genes_dat_path = genome_dir / "1.0" / "data" / "genes.dat"
            upsert_record(
                build_genome_record(
                    genome_id=genome_id,
                    mode=infer_results_mode(pwy_path),
                    genome_dir=genome_dir,
                    pwy_path=pwy_path,
                    pwy2orf_path=pwy2orf_path,
                    genes_dat_path=genes_dat_path if genes_dat_path.exists() else None,
                    pathway_status="present",
                )
            )
            explicit_pathway_record_count += 1

        for pwy2orf_path in sorted(pgdb_dir.glob("**/*_pwy2orf.tsv")):
            if "community" in pwy2orf_path.parts:
                continue
            genome_id = pwy2orf_path.name[: -len("_pwy2orf.tsv")]
            genome_dir = resolve_pgdb_genome_dir(pwy2orf_path)
            genes_dat_path = genome_dir / "1.0" / "data" / "genes.dat"
            upsert_record(
                build_genome_record(
                    genome_id=genome_id,
                    mode=infer_results_mode(pwy2orf_path),
                    genome_dir=genome_dir,
                    pwy2orf_path=pwy2orf_path,
                    genes_dat_path=genes_dat_path if genes_dat_path.exists() else None,
                    pathway_status="missing_pathway_table",
                    pathway_missing_reason="missing *_pwy.tsv; using non-pathway summaries only",
                )
            )

        for genes_dat_path in sorted(pgdb_dir.glob("**/1.0/data/genes.dat")):
            genome_dir = genes_dat_path.parents[2]
            if genome_dir.name == "community":
                continue
            genome_id = genome_dir.name
            pwy2orf_candidates = [
                genome_dir / f"{genome_id}_pwy2orf.tsv",
                genome_dir / "tables" / "other" / f"{genome_id}_pwy2orf.tsv",
            ]
            pwy2orf_path = next((path for path in pwy2orf_candidates if path.exists()), None)
            upsert_record(
                build_genome_record(
                    genome_id=genome_id,
                    mode=infer_results_mode(genes_dat_path),
                    genome_dir=genome_dir,
                    pwy2orf_path=pwy2orf_path,
                    genes_dat_path=genes_dat_path,
                    pathway_status="missing_pathway_table",
                    pathway_missing_reason="missing *_pwy.tsv; using non-pathway summaries only",
                )
            )

        community_pwy = sorted((pgdb_dir / "community").glob("**/*_pwy.tsv"))
        if community_pwy and explicit_pathway_record_count == 0:
            pwy_path = community_pwy[0]
            genome_id = pwy_path.name[: -len("_pwy.tsv")]
            genome_dir = resolve_pgdb_genome_dir(pwy_path)
            pwy2orf_candidates = [
                genome_dir / f"{genome_id}_pwy2orf.tsv",
                genome_dir / "tables" / "other" / f"{genome_id}_pwy2orf.tsv",
            ]
            pwy2orf_path = next((path for path in pwy2orf_candidates if path.exists()), None)
            genes_dat_path = genome_dir / "1.0" / "data" / "genes.dat"
            records_by_id.pop("community", None)
            upsert_record(
                build_genome_record(
                    genome_id=genome_id,
                    mode=infer_results_mode(results_dir),
                    genome_dir=genome_dir,
                    pwy_path=pwy_path,
                    pwy2orf_path=pwy2orf_path,
                    genes_dat_path=genes_dat_path if genes_dat_path.exists() else None,
                    pathway_status="present",
                    pathway_missing_reason="",
                )
            )

    if not records_by_id:
        annotation_dir = results_dir / "annotation_table"
        has_annotation = annotation_dir.exists() and any(annotation_dir.iterdir())
        if has_annotation:
            genome_id = infer_genome_id_from_results_dir(results_dir)
            upsert_record(
                build_genome_record(
                    genome_id=genome_id,
                    mode=infer_results_mode(results_dir),
                    genome_dir=results_dir,
                    pathway_status="missing_pgdb_or_pathway_tables",
                    pathway_missing_reason="missing pgdb/pathway tables; using annotation-derived summaries only",
                )
            )

    if not records_by_id:
        if not pgdb_dir.exists():
            raise FileNotFoundError(
                f"MetaPathways pgdb directory not found and no usable annotation tables were found: {pgdb_dir}"
            )
        raise FileNotFoundError(
            f"No MetaPathways pathway summary files or usable annotation tables found under: {results_dir}"
        )

    return (
        pd.DataFrame(records_by_id.values())
        .sort_values("genome_id")
        .reset_index(drop=True)
    )


def load_allowed_genomes(filter_tsv, filter_id_column=None, filter_tier_column="mimag_tier", include_tiers=None):
    filter_df = read_table(filter_tsv).copy()
    possible_id_columns = [filter_id_column] if filter_id_column else ["Bin Id", "genome_id", "Genome_Id"]
    selected_id_column = None
    for column in possible_id_columns:
        if column and column in filter_df.columns:
            selected_id_column = column
            break
    if selected_id_column is None:
        raise ValueError(
            "--genome-filter-tsv does not contain a usable genome ID column. "
            "Tried: " + ", ".join([column for column in possible_id_columns if column])
        )

    filter_df[selected_id_column] = normalize_text(filter_df[selected_id_column])
    filter_df = filter_df.loc[filter_df[selected_id_column].ne("")].copy()
    if filter_tier_column and filter_tier_column in filter_df.columns and include_tiers:
        allowed_tiers = {tier.lower() for tier in include_tiers}
        filter_df[filter_tier_column] = normalize_text(filter_df[filter_tier_column]).str.lower()
        filter_df = filter_df.loc[filter_df[filter_tier_column].isin(allowed_tiers)].copy()
    allowed = set()
    alias_columns = [selected_id_column]
    for optional_column in ["Bin Id", "Genome_Id", "genome_id", "fasta_path"]:
        if optional_column in filter_df.columns and optional_column not in alias_columns:
            alias_columns.append(optional_column)

    for column in alias_columns:
        values = normalize_text(filter_df[column])
        if column == "fasta_path":
            values = values.map(lambda value: Path(value).name if value else "")
        for value in values.astype(str).tolist():
            allowed.update(id_aliases(value))
    return allowed, selected_id_column


def taxonomy_value_is_informative(value):
    text = normalize_text(pd.Series([value])).iat[0]
    lowered = text.lower()
    if lowered in TAXONOMY_MISSING_TOKENS:
        return False
    if lowered.startswith("unclassified"):
        return False
    if lowered.startswith("unknown"):
        return False
    return bool(text)


def build_taxonomy_label_entry(row):
    values = {rank: normalize_text(pd.Series([row.get(rank, "")])).iat[0] for rank in TAXONOMY_RANK_ORDER}
    if taxonomy_value_is_informative(values.get("Species", "")):
        return {
            "taxonomy_display_label": values["Species"],
            "taxonomy_display_rank": "Species",
            "taxonomy_display_value": values["Species"],
            "taxonomy_display_status": "classified",
            "taxonomy_label_score": 7,
        }
    if taxonomy_value_is_informative(values.get("Genus", "")):
        return {
            "taxonomy_display_label": values["Genus"],
            "taxonomy_display_rank": "Genus",
            "taxonomy_display_value": values["Genus"],
            "taxonomy_display_status": "classified",
            "taxonomy_label_score": 6,
        }
    for score, rank in zip([5, 4, 3, 2, 1], TAXONOMY_FALLBACK_RANKS):
        if taxonomy_value_is_informative(values.get(rank, "")):
            value = values[rank]
            return {
                "taxonomy_display_label": f"Unclassified_{rank}_{value}",
                "taxonomy_display_rank": rank,
                "taxonomy_display_value": value,
                "taxonomy_display_status": "unclassified_fallback",
                "taxonomy_label_score": score,
            }
    return {
        "taxonomy_display_label": "Unclassified_unknown",
        "taxonomy_display_rank": "unknown",
        "taxonomy_display_value": "",
        "taxonomy_display_status": "unknown",
        "taxonomy_label_score": 0,
    }


def load_taxonomy_label_lookup(taxonomy_tsv, taxonomy_id_column=None):
    taxonomy_df = read_table(taxonomy_tsv).copy()
    possible_id_columns = (
        [taxonomy_id_column] if taxonomy_id_column else ["SAG_ID", "Bin Id", "genome_id", "Genome_Id"]
    )
    selected_id_column = None
    for column in possible_id_columns:
        if column and column in taxonomy_df.columns:
            selected_id_column = column
            break
    if selected_id_column is None:
        raise ValueError(
            "--taxonomy-label-tsv does not contain a usable genome ID column. "
            "Tried: " + ", ".join([column for column in possible_id_columns if column])
        )

    rank_columns_present = [rank for rank in TAXONOMY_RANK_ORDER if rank in taxonomy_df.columns]
    if not rank_columns_present:
        raise ValueError(
            "--taxonomy-label-tsv does not contain taxonomy rank columns. "
            f"Expected at least one of: {', '.join(TAXONOMY_RANK_ORDER)}"
        )

    alias_columns = [selected_id_column]
    for optional_column in ["SAG_ID", "Bin Id", "Genome_Id", "genome_id", "fasta_path"]:
        if optional_column in taxonomy_df.columns and optional_column not in alias_columns:
            alias_columns.append(optional_column)

    alias_candidates = {}
    for row in taxonomy_df.to_dict("records"):
        label_entry = build_taxonomy_label_entry(row)
        primary_id = str(row.get(selected_id_column, "")).strip()
        label_entry["taxonomy_source_id"] = primary_id
        aliases = set()
        for column in alias_columns:
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            if column == "fasta_path":
                value = Path(value).name
            aliases.update(id_aliases(value))
        for alias in aliases:
            alias_candidates.setdefault(alias, []).append(label_entry.copy())

    alias_lookup = {}
    ambiguous_alias_count = 0
    for alias, entries in alias_candidates.items():
        if not entries:
            continue
        labels = sorted({entry.get("taxonomy_display_label", "") for entry in entries})
        if len(labels) > 1:
            ambiguous_alias_count += 1
            continue
        best_entry = sorted(
            entries,
            key=lambda entry: (
                int(entry.get("taxonomy_label_score", 0)),
                str(entry.get("taxonomy_source_id", "")),
            ),
            reverse=True,
        )[0]
        alias_lookup[alias] = best_entry
    return alias_lookup, selected_id_column, ambiguous_alias_count


def apply_taxonomy_labels(genome_summary, taxonomy_lookup):
    frame = genome_summary.copy()
    if frame.empty:
        frame["genome_display_label"] = []
        return frame

    labels = []
    ranks = []
    values = []
    statuses = []
    methods = []
    for genome_id in frame["genome_id"].astype(str).tolist():
        aliases = id_aliases(genome_id)
        candidates = []
        if genome_id in taxonomy_lookup:
            candidates.append((taxonomy_lookup[genome_id], "exact"))
        for alias in sorted(aliases):
            if alias == genome_id:
                continue
            if alias in taxonomy_lookup:
                candidates.append((taxonomy_lookup[alias], "alias"))

        if not candidates:
            labels.append(genome_id)
            ranks.append("")
            values.append("")
            statuses.append("unmatched")
            methods.append("none")
            continue

        candidates.sort(
            key=lambda item: (
                int(item[0].get("taxonomy_label_score", 0)),
                item[0].get("taxonomy_display_label", ""),
            ),
            reverse=True,
        )
        best_entry, best_method = candidates[0]
        labels.append(best_entry.get("taxonomy_display_label", genome_id))
        ranks.append(best_entry.get("taxonomy_display_rank", ""))
        values.append(best_entry.get("taxonomy_display_value", ""))
        statuses.append(best_entry.get("taxonomy_display_status", ""))
        methods.append(best_method)

    frame["genome_display_label"] = labels
    frame["taxonomy_display_rank"] = ranks
    frame["taxonomy_display_value"] = values
    frame["taxonomy_display_status"] = statuses
    frame["taxonomy_match_method"] = methods
    return frame


def summarize_annotation_lookup(orf_ids, orf_annotation_lookup):
    if orf_annotation_lookup is None or not orf_ids:
        return {
            "total_orfs": len(orf_ids),
            "annotated_orfs": np.nan,
            "annotation_fraction": np.nan,
            "informative_annotation_orfs": np.nan,
            "informative_annotation_fraction": np.nan,
            "uncertain_annotation_orfs": np.nan,
            "uncertain_annotation_fraction": np.nan,
            "unannotated_orfs": np.nan,
            "hypothetical_annotation_orfs": np.nan,
            "uncharacterized_annotation_orfs": np.nan,
            "domain_family_annotation_orfs": np.nan,
            "qualifier_annotation_orfs": np.nan,
            "fragment_annotation_orfs": np.nan,
            "other_uncertain_annotation_orfs": np.nan,
            "metacyc_orfs": np.nan,
            "swissprot_orfs": np.nan,
            "uniref50_orfs": np.nan,
            "multi_source_orfs": np.nan,
        }

    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    annotated = int(lookup_hits["annotated_orf"].fillna(False).sum())
    total = len(orf_ids)
    summary = {
        "total_orfs": total,
        "annotated_orfs": annotated,
        "annotation_fraction": annotated / total if total else np.nan,
        "informative_annotation_orfs": int(lookup_hits["annotation_is_informative"].fillna(False).sum()),
        "informative_annotation_fraction": (
            float(lookup_hits["annotation_is_informative"].fillna(False).sum()) / total
            if total else np.nan
        ),
        "uncertain_annotation_orfs": int(lookup_hits["annotation_is_uncertain"].fillna(False).sum()),
        "uncertain_annotation_fraction": (
            float(lookup_hits["annotation_is_uncertain"].fillna(False).sum()) / total
            if total else np.nan
        ),
        "unannotated_orfs": int(lookup_hits["unannotated_orf"].fillna(False).sum()),
        "metacyc_orfs": int(lookup_hits["has_metacyc"].fillna(False).sum()),
        "swissprot_orfs": int(lookup_hits["has_swissprot"].fillna(False).sum()),
        "uniref50_orfs": int(lookup_hits["has_uniref50"].fillna(False).sum()),
        "multi_source_orfs": int(lookup_hits["source_count"].fillna(0).ge(2).sum()),
    }
    for category in ANNOTATION_CATEGORY_ORDER:
        if category in {"informative", "unannotated"}:
            continue
        summary[f"{category}_annotation_orfs"] = int(
            lookup_hits["annotation_category"].fillna("").eq(category).sum()
        )
    return summary


@lru_cache(maxsize=200000)
def classify_annotation_text(text):
    normalized = str(text).strip().lower()
    if not normalized:
        return "unannotated"
    for category in ["hypothetical", "uncharacterized", "domain_family", "qualifier", "fragment"]:
        for pattern in UNCERTAIN_ANNOTATION_PATTERNS[category]:
            if pattern in normalized:
                return category
    return "informative"


def classify_orf_annotation_row(row):
    source_texts = []
    source_categories = []
    for column in FUNCTIONAL_SOURCE_COLUMNS:
        value = str(row.get(column, "")).strip()
        if not value:
            continue
        source_texts.append(value)
        source_categories.append(classify_annotation_text(value))

    if not source_texts:
        return {
            "annotation_category": "unannotated",
            "annotation_is_informative": False,
            "annotation_is_uncertain": False,
        }

    if any(category == "informative" for category in source_categories):
        return {
            "annotation_category": "informative",
            "annotation_is_informative": True,
            "annotation_is_uncertain": False,
        }

    for category in ["hypothetical", "uncharacterized", "domain_family", "qualifier", "fragment"]:
        if category in source_categories:
            return {
                "annotation_category": category,
                "annotation_is_informative": False,
                "annotation_is_uncertain": True,
            }

    return {
        "annotation_category": "other_uncertain",
        "annotation_is_informative": False,
        "annotation_is_uncertain": True,
    }


@lru_cache(maxsize=500000)
def mobility_keyword_matches(text, keyword):
    normalized_text = str(text).strip().lower()
    normalized_keyword = str(keyword).strip().lower()
    if not normalized_text or not normalized_keyword:
        return False
    if normalized_keyword in EXPERIMENTAL_SHORT_TOKEN_KEYWORDS:
        pattern = r"(?<![a-z0-9])" + re.escape(normalized_keyword) + r"(?![a-z0-9])"
        return bool(re.search(pattern, normalized_text))
    return normalized_keyword in normalized_text


def detect_experimental_mobility_hits(annotation_text, keyword_map):
    normalized_text = str(annotation_text).strip().lower()
    if not normalized_text:
        return {}
    matches = {}
    for category, keywords in keyword_map.items():
        matched_keywords = [keyword for keyword in keywords if mobility_keyword_matches(normalized_text, keyword)]
        if matched_keywords:
            matches[category] = matched_keywords
    return matches


def clean_experimental_category_label(value):
    text = str(value).strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def infer_experimental_category_from_path(value):
    text = str(value).strip()
    if not text:
        return ""
    parts = [part for part in Path(text).parts if str(part).strip()]
    for part in parts:
        label = clean_experimental_category_label(part)
        token = "".join(character for character in label.lower() if character.isalnum())
        if "sag" in token or "mag" in token:
            return label
    text_label = clean_experimental_category_label(text)
    text_token = "".join(character for character in text_label.lower() if character.isalnum())
    if text_token == "genome":
        return "SAGs"
    if text_token == "metagenome":
        return "MAGs"
    if "sag" in text_token or "mag" in text_token:
        return text_label
    return ""


def normalize_experimental_genome_type(value):
    return clean_experimental_category_label(value)


def infer_experimental_genome_type_from_row(row):
    direct_label = clean_experimental_category_label(row.get("genome_type", ""))
    if direct_label:
        return direct_label, "genome_type"

    for column in ["genome_dir", "genome_id", "results_mode"]:
        value = row.get(column, "")
        normalized = infer_experimental_category_from_path(value)
        if normalized:
            return normalized, column
    return "", ""


def resolve_experimental_genome_type_lookup(genome_summary, genome_type_tsv=None, genome_type_column=None, genome_type_id_column=None):
    rows = []
    for row in genome_summary.to_dict("records"):
        genome_id = str(row.get("genome_id", "")).strip()
        if not genome_id:
            continue
        genome_type, source_column = infer_experimental_genome_type_from_row(row)
        rows.append(
            {
                "genome_id": genome_id,
                "genome_type": genome_type,
                "genome_type_source": source_column or "",
            }
        )
    genome_type_df = pd.DataFrame(rows).drop_duplicates("genome_id")

    if genome_type_tsv:
        metadata_df = read_table(Path(genome_type_tsv).expanduser().resolve()).copy()
        candidate_id_columns = (
            [genome_type_id_column] if genome_type_id_column
            else ["genome_id", "Genome_Id", "Bin Id", "SAG_ID", "pre_ani_bin_key", "fasta_path"]
        )
        selected_id_column = None
        for column in candidate_id_columns:
            if column and column in metadata_df.columns:
                selected_id_column = column
                break
        if selected_id_column is None:
            raise ValueError(
                "Experimental mobility genome-type metadata does not contain a usable genome ID column. "
                "Provide --experimental-mobility-genome-type-id-column."
            )

        candidate_type_columns = (
            [genome_type_column] if genome_type_column
            else ["genome_type", "category", "method", "type"]
        )
        selected_type_column = None
        for column in candidate_type_columns:
            if column and column in metadata_df.columns:
                selected_type_column = column
                break
        if selected_type_column is None:
            raise ValueError(
                "Experimental mobility genome-type metadata does not contain a usable genome-type column. "
                "Provide --experimental-mobility-genome-type-column."
            )

        alias_lookup = {}
        for record in metadata_df.to_dict("records"):
            raw_id = str(record.get(selected_id_column, "")).strip()
            if not raw_id:
                continue
            if selected_id_column == "fasta_path":
                raw_id = Path(raw_id).name
            normalized_type = normalize_experimental_genome_type(record.get(selected_type_column, ""))
            if not normalized_type:
                continue
            for alias in id_aliases(raw_id):
                alias_lookup.setdefault(alias, set()).add(normalized_type)

        resolved_types = []
        resolved_sources = []
        for row in genome_type_df.to_dict("records"):
            if row["genome_type"]:
                resolved_types.append(row["genome_type"])
                resolved_sources.append(row["genome_type_source"])
                continue
            genome_aliases = id_aliases(row["genome_id"])
            matched_types = set()
            for alias in genome_aliases:
                matched_types.update(alias_lookup.get(alias, set()))
            if len(matched_types) == 1:
                resolved_types.append(sorted(matched_types)[0])
                resolved_sources.append(selected_type_column)
            else:
                resolved_types.append("")
                resolved_sources.append("")
        genome_type_df["genome_type"] = resolved_types
        genome_type_df["genome_type_source"] = resolved_sources

    unresolved = genome_type_df.loc[genome_type_df["genome_type"].eq(""), "genome_id"].astype(str).tolist()
    if unresolved:
        raise ValueError(
            "Experimental mobility screen could not resolve category labels for all genomes. "
            "Provide --experimental-mobility-genome-type-tsv and --experimental-mobility-genome-type-column. "
            f"Examples of unresolved genome IDs: {', '.join(unresolved[:5])}"
        )
    return genome_type_df


@lru_cache(maxsize=200000)
def classify_elemental_cycles(text):
    normalized = str(text).strip().lower()
    if not normalized:
        return tuple()

    matches = []
    for cycle in ELEMENTAL_CYCLE_ORDER:
        if any(keyword in normalized for keyword in ACTIVE_ELEMENTAL_CYCLE_TERMS[cycle]):
            matches.append(cycle)
    return tuple(matches)


@lru_cache(maxsize=200000)
def classify_elemental_modes(text):
    normalized = str(text).strip().lower()
    if not normalized:
        return tuple()

    matches = []
    for mode in ELEMENTAL_MODE_ORDER:
        if any(keyword in normalized for keyword in ACTIVE_ELEMENTAL_MODE_TERMS[mode]):
            matches.append(mode)
    return tuple(matches)


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


def elemental_families_from_modes(modes):
    families = {ELEMENTAL_MODE_FAMILY[mode] for mode in modes if mode in ELEMENTAL_MODE_FAMILY}
    return [family for family in ELEMENTAL_CYCLE_ORDER if family in families]


def extract_orf_elemental_cycles(row):
    cycles = set()
    modes = set()
    for column in FUNCTIONAL_SOURCE_COLUMNS:
        value = str(row.get(column, "")).strip()
        if not value:
            continue
        modes.update(classify_elemental_modes(value))
        cycles.update(classify_elemental_cycles(value))
    cycles.update(elemental_families_from_modes(modes))
    ordered = [cycle for cycle in ELEMENTAL_CYCLE_ORDER if cycle in cycles]
    return ";".join(ordered)


def extract_orf_elemental_modes(row):
    modes = set()
    for column in FUNCTIONAL_SOURCE_COLUMNS:
        value = str(row.get(column, "")).strip()
        if not value:
            continue
        modes.update(classify_elemental_modes(value))
    ordered = [mode for mode in ELEMENTAL_MODE_ORDER if mode in modes]
    return ";".join(ordered)


def build_annotation_audit_table(genome_id, orf_ids, orf_annotation_lookup):
    columns = [
        "genome_id",
        "orf_id",
        "source_db",
        "annotation_text",
        "annotation_category",
        "annotation_is_informative",
        "annotation_is_uncertain",
        "direct_family_ids",
        "direct_family_labels",
        "direct_family_keywords",
        "mode_ids",
        "mode_labels",
        "mode_keywords",
        "resolved_family_ids",
        "resolved_family_labels",
        "final_orf_family_ids",
        "final_orf_mode_ids",
    ]
    if orf_annotation_lookup is None or not orf_ids:
        return pd.DataFrame(columns=columns)

    rows = []
    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    for orf_id, row in lookup_hits.iterrows():
        annotation_category = row.get("annotation_category", "")
        is_informative = bool(row.get("annotation_is_informative", False))
        is_uncertain = bool(row.get("annotation_is_uncertain", False))
        final_family_ids = str(row.get("elemental_cycle_labels", "") or "")
        final_mode_ids = str(row.get("elemental_mode_labels", "") or "")
        for source_db in FUNCTIONAL_SOURCE_COLUMNS:
            annotation_text = str(row.get(source_db, "") or "").strip()
            if not annotation_text:
                continue
            family_matches = keyword_match_map(annotation_text, ELEMENTAL_CYCLE_ORDER, ACTIVE_ELEMENTAL_CYCLE_TERMS)
            mode_matches = keyword_match_map(annotation_text, ELEMENTAL_MODE_ORDER, ACTIVE_ELEMENTAL_MODE_TERMS)
            direct_family_ids = [family_id for family_id in ELEMENTAL_CYCLE_ORDER if family_id in family_matches]
            mode_ids = [mode_id for mode_id in ELEMENTAL_MODE_ORDER if mode_id in mode_matches]
            resolved_family_ids = []
            for family_id in direct_family_ids + elemental_families_from_modes(mode_ids):
                if family_id not in resolved_family_ids:
                    resolved_family_ids.append(family_id)
            rows.append(
                {
                    "genome_id": genome_id,
                    "orf_id": orf_id,
                    "source_db": source_db,
                    "annotation_text": annotation_text,
                    "annotation_category": annotation_category,
                    "annotation_is_informative": is_informative,
                    "annotation_is_uncertain": is_uncertain,
                    "direct_family_ids": ";".join(direct_family_ids),
                    "direct_family_labels": ";".join(
                        ELEMENTAL_CYCLE_LABELS[family_id] for family_id in direct_family_ids
                    ),
                    "direct_family_keywords": ";".join(
                        keyword
                        for family_id in direct_family_ids
                        for keyword in family_matches.get(family_id, [])
                    ),
                    "mode_ids": ";".join(mode_ids),
                    "mode_labels": ";".join(ELEMENTAL_MODE_LABELS[mode_id] for mode_id in mode_ids),
                    "mode_keywords": ";".join(
                        keyword
                        for mode_id in mode_ids
                        for keyword in mode_matches.get(mode_id, [])
                    ),
                    "resolved_family_ids": ";".join(resolved_family_ids),
                    "resolved_family_labels": ";".join(
                        ELEMENTAL_CYCLE_LABELS[family_id] for family_id in resolved_family_ids
                    ),
                    "final_orf_family_ids": final_family_ids,
                    "final_orf_mode_ids": final_mode_ids,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_pathway_audit_table(pwy_df):
    columns = [
        "genome_id",
        "pathway_id",
        "pathway_name",
        "pathway_common_name",
        "pathway_text",
        "direct_family_ids",
        "direct_family_labels",
        "direct_family_keywords",
        "mode_ids",
        "mode_labels",
        "mode_keywords",
        "resolved_family_ids",
        "resolved_family_labels",
        "final_pathway_family_ids",
        "final_pathway_mode_ids",
        "PWY_SCORE",
        "reaction_coverage_fraction",
    ]
    if pwy_df is None or pwy_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, row in pwy_df.iterrows():
        pathway_text = str(row.get("PWY_COMMON_NAME", "") or "").strip()
        if not pathway_text:
            pathway_text = str(row.get("PWY_NAME", "") or "").strip()
        family_matches = keyword_match_map(pathway_text, ELEMENTAL_CYCLE_ORDER, ACTIVE_ELEMENTAL_CYCLE_TERMS)
        mode_matches = keyword_match_map(pathway_text, ELEMENTAL_MODE_ORDER, ACTIVE_ELEMENTAL_MODE_TERMS)
        direct_family_ids = [family_id for family_id in ELEMENTAL_CYCLE_ORDER if family_id in family_matches]
        mode_ids = [mode_id for mode_id in ELEMENTAL_MODE_ORDER if mode_id in mode_matches]
        resolved_family_ids = []
        for family_id in direct_family_ids + elemental_families_from_modes(mode_ids):
            if family_id not in resolved_family_ids:
                resolved_family_ids.append(family_id)
        rows.append(
            {
                "genome_id": row.get("genome_id", ""),
                "pathway_id": row.get("PWY_NAME", ""),
                "pathway_name": row.get("PWY_NAME", ""),
                "pathway_common_name": row.get("PWY_COMMON_NAME", ""),
                "pathway_text": pathway_text,
                "direct_family_ids": ";".join(direct_family_ids),
                "direct_family_labels": ";".join(
                    ELEMENTAL_CYCLE_LABELS[family_id] for family_id in direct_family_ids
                ),
                "direct_family_keywords": ";".join(
                    keyword
                    for family_id in direct_family_ids
                    for keyword in family_matches.get(family_id, [])
                ),
                "mode_ids": ";".join(mode_ids),
                "mode_labels": ";".join(ELEMENTAL_MODE_LABELS[mode_id] for mode_id in mode_ids),
                "mode_keywords": ";".join(
                    keyword
                    for mode_id in mode_ids
                    for keyword in mode_matches.get(mode_id, [])
                ),
                "resolved_family_ids": ";".join(resolved_family_ids),
                "resolved_family_labels": ";".join(
                    ELEMENTAL_CYCLE_LABELS[family_id] for family_id in resolved_family_ids
                ),
                "final_pathway_family_ids": row.get("elemental_cycle_labels", ""),
                "final_pathway_mode_ids": row.get("elemental_mode_labels", ""),
                "PWY_SCORE": row.get("PWY_SCORE", np.nan),
                "reaction_coverage_fraction": row.get("reaction_coverage_fraction", np.nan),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_marker_audit_table(
    genome_id,
    orf_ids,
    orf_annotation_lookup,
    marker_manifest,
    marker_alias_to_rows=None,
    marker_alias_regex=None,
):
    columns = [
        "genome_id",
        "orf_id",
        "source_db",
        "annotation_text",
        "family_id",
        "family_label",
        "mode_id",
        "mode_label",
        "marker_id",
        "marker_label",
        "is_core",
        "matched_alias",
        "match_evidence_type",
        "match_resolution",
        "marker_specificity_class",
    ]
    if orf_annotation_lookup is None or not orf_ids or marker_manifest.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    alias_to_rows = marker_alias_to_rows or {}
    for orf_id, row in lookup_hits.iterrows():
        for source_db in FUNCTIONAL_SOURCE_COLUMNS:
            annotation_text = str(row.get(source_db, "") or "").strip()
            normalized = annotation_text.lower()
            if not normalized:
                continue
            aliases = set()
            if marker_alias_regex is not None:
                aliases.update(match.group(0) for match in marker_alias_regex.finditer(normalized))
            elif alias_to_rows:
                for alias in alias_to_rows:
                    if re.search(rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])", normalized):
                        aliases.add(alias)
            matched_rows = []
            for alias in aliases:
                for marker in alias_to_rows.get(alias, []):
                    evidence_type = marker_row_match_evidence(normalized, alias, marker)
                    if not evidence_type:
                        continue
                    matched_rows.append(
                        {
                            "genome_id": genome_id,
                            "orf_id": orf_id,
                            "source_db": source_db,
                            "annotation_text": annotation_text,
                            "family_id": marker["family_id"],
                            "family_label": ELEMENTAL_CYCLE_LABELS.get(marker["family_id"], marker["family_id"]),
                            "mode_id": marker["mode_id"],
                            "mode_label": ELEMENTAL_MODE_LABELS.get(marker["mode_id"], marker["mode_id"]),
                            "marker_id": marker["marker_id"],
                            "marker_label": marker["marker_label"],
                            "is_core": bool(marker["is_core"]),
                            "matched_alias": alias,
                            "match_evidence_type": evidence_type,
                            "match_resolution": "",
                            "marker_specificity_class": marker.get("_specificity_class", "specific"),
                        }
                    )
            rows.extend(resolve_marker_specificity(matched_rows))
    return pd.DataFrame(rows, columns=columns).drop_duplicates()


def summarize_marker_support(marker_audit_df):
    summary = {
        "marker_supported_orfs": 0,
        "marker_gene_count": 0,
        "core_marker_gene_count": 0,
        "specific_marker_gene_count": 0,
        "generic_marker_gene_count": 0,
    }
    for mode_id in ELEMENTAL_MODE_ORDER:
        summary[f"marker_{mode_id}_orf_count"] = 0
        summary[f"marker_{mode_id}_gene_count"] = 0
        summary[f"marker_{mode_id}_core_gene_count"] = 0
        summary[f"marker_{mode_id}_specific_gene_count"] = 0
        summary[f"marker_{mode_id}_generic_gene_count"] = 0

    if marker_audit_df is None or marker_audit_df.empty:
        return summary

    matched = marker_audit_df.loc[marker_audit_df["marker_id"].fillna("").ne("")].copy()
    summary["marker_supported_orfs"] = int(matched["orf_id"].nunique())
    summary["marker_gene_count"] = int(matched["marker_id"].nunique())
    summary["core_marker_gene_count"] = int(matched.loc[matched["is_core"], "marker_id"].nunique())
    summary["specific_marker_gene_count"] = int(
        matched.loc[matched["marker_specificity_class"].eq("specific"), "marker_id"].nunique()
    )
    summary["generic_marker_gene_count"] = int(
        matched.loc[matched["marker_specificity_class"].eq("generic"), "marker_id"].nunique()
    )
    for mode_id in ELEMENTAL_MODE_ORDER:
        mode_df = matched.loc[matched["mode_id"].eq(mode_id)]
        summary[f"marker_{mode_id}_orf_count"] = int(mode_df["orf_id"].nunique())
        summary[f"marker_{mode_id}_gene_count"] = int(mode_df["marker_id"].nunique())
        summary[f"marker_{mode_id}_core_gene_count"] = int(mode_df.loc[mode_df["is_core"], "marker_id"].nunique())
        summary[f"marker_{mode_id}_specific_gene_count"] = int(
            mode_df.loc[mode_df["marker_specificity_class"].eq("specific"), "marker_id"].nunique()
        )
        summary[f"marker_{mode_id}_generic_gene_count"] = int(
            mode_df.loc[mode_df["marker_specificity_class"].eq("generic"), "marker_id"].nunique()
        )
    return summary


def collapse_marker_audit(marker_audit_df):
    columns = [
        "genome_id",
        "family_id",
        "family_label",
        "mode_id",
        "mode_label",
        "marker_id",
        "marker_label",
        "is_core",
        "supporting_orf_count",
        "supporting_orf_ids",
        "source_dbs",
        "matched_aliases",
        "match_evidence_types",
        "marker_specificity_class",
        "supporting_annotations",
    ]
    if marker_audit_df is None or marker_audit_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = (
        marker_audit_df.groupby(
            [
                "genome_id",
                "family_id",
                "family_label",
                "mode_id",
                "mode_label",
                "marker_id",
                "marker_label",
                "is_core",
                "marker_specificity_class",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            supporting_orf_count=("orf_id", lambda s: int(pd.Series(s).astype(str).nunique())),
            supporting_orf_ids=("orf_id", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
            source_dbs=("source_db", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
            matched_aliases=("matched_alias", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
            match_evidence_types=("match_evidence_type", lambda s: ";".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
            supporting_annotations=("annotation_text", lambda s: " || ".join(sorted(pd.Series(s).astype(str).unique().tolist()))),
        )
        .reset_index()
    )
    return grouped[columns]


def build_reference_mode_audit_table(genome_id, orf_ids, orf_annotation_lookup, accession_mode_lookup):
    columns = [
        "genome_id",
        "orf_id",
        "source_db",
        "annotation_text",
        "matched_accession",
        "family_id",
        "family_label",
        "mode_id",
        "mode_label",
        "go_ids",
        "go_names",
    ]
    if orf_annotation_lookup is None or not orf_ids or not accession_mode_lookup:
        return pd.DataFrame(columns=columns)

    rows = []
    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    for orf_id, row in lookup_hits.iterrows():
        for source_db in FUNCTIONAL_SOURCE_COLUMNS:
            annotation_text = str(row.get(source_db, "") or "").strip()
            if not annotation_text:
                continue
            target_text = str(row.get(f"{source_db}_target", "") or "").strip()
            accession_text = target_text or annotation_text
            for accession in extract_uniprot_accessions(accession_text):
                evidence = accession_mode_lookup.get(accession)
                if not evidence:
                    continue
                mode_ids = evidence.get("mode_ids", [])
                family_ids = evidence.get("family_ids", [])
                family_map = {
                    mode_id: ELEMENTAL_MODE_FAMILY.get(mode_id, "")
                    for mode_id in mode_ids
                }
                for mode_id in mode_ids:
                    family_id = family_map.get(mode_id, "")
                    if not family_id and family_ids:
                        family_id = family_ids[0]
                    rows.append(
                        {
                            "genome_id": genome_id,
                            "orf_id": orf_id,
                            "source_db": source_db,
                            "annotation_text": annotation_text,
                            "matched_accession": accession,
                            "family_id": family_id,
                            "family_label": ELEMENTAL_CYCLE_LABELS.get(family_id, family_id),
                            "mode_id": mode_id,
                            "mode_label": ELEMENTAL_MODE_LABELS.get(mode_id, mode_id),
                            "go_ids": ";".join(evidence.get("go_ids", [])),
                            "go_names": ";".join(evidence.get("go_names", [])),
                        }
                    )
    return pd.DataFrame(rows, columns=columns).drop_duplicates()


def summarize_reference_mode_support(reference_mode_audit_df):
    summary = {
        "reference_mode_supported_orfs": 0,
        "reference_mode_supported_accessions": 0,
    }
    for mode_id in ELEMENTAL_MODE_ORDER:
        summary[f"reference_mode_{mode_id}_orf_count"] = 0
        summary[f"reference_mode_{mode_id}_accession_count"] = 0

    if reference_mode_audit_df is None or reference_mode_audit_df.empty:
        return summary

    summary["reference_mode_supported_orfs"] = int(reference_mode_audit_df["orf_id"].nunique())
    summary["reference_mode_supported_accessions"] = int(reference_mode_audit_df["matched_accession"].nunique())
    for mode_id in ELEMENTAL_MODE_ORDER:
        mode_df = reference_mode_audit_df.loc[reference_mode_audit_df["mode_id"].eq(mode_id)]
        summary[f"reference_mode_{mode_id}_orf_count"] = int(mode_df["orf_id"].nunique())
        summary[f"reference_mode_{mode_id}_accession_count"] = int(mode_df["matched_accession"].nunique())
    return summary


def marker_mode_possible_counts(marker_manifest):
    possible_counts = {mode_id: 0 for mode_id in ELEMENTAL_MODE_ORDER}
    core_possible_counts = {mode_id: 0 for mode_id in ELEMENTAL_MODE_ORDER}
    specific_possible_counts = {mode_id: 0 for mode_id in ELEMENTAL_MODE_ORDER}
    generic_possible_counts = {mode_id: 0 for mode_id in ELEMENTAL_MODE_ORDER}
    if marker_manifest is None or marker_manifest.empty:
        return possible_counts, core_possible_counts, specific_possible_counts, generic_possible_counts

    specificity_map = classify_marker_manifest_specificity(marker_manifest)
    deduped = marker_manifest[["mode_id", "marker_id", "is_core"]].drop_duplicates().copy()
    deduped["marker_specificity_class"] = deduped.apply(
        lambda row: specificity_map.get(
            (str(row["mode_id"]).strip().lower(), str(row["marker_id"]).strip()),
            "specific",
        ),
        axis=1,
    )
    for mode_id in ELEMENTAL_MODE_ORDER:
        mode_df = deduped.loc[deduped["mode_id"].eq(mode_id)]
        possible_counts[mode_id] = int(mode_df["marker_id"].nunique())
        core_possible_counts[mode_id] = int(mode_df.loc[mode_df["is_core"], "marker_id"].nunique())
        specific_possible_counts[mode_id] = int(
            mode_df.loc[mode_df["marker_specificity_class"].eq("specific"), "marker_id"].nunique()
        )
        generic_possible_counts[mode_id] = int(
            mode_df.loc[mode_df["marker_specificity_class"].eq("generic"), "marker_id"].nunique()
        )
    return possible_counts, core_possible_counts, specific_possible_counts, generic_possible_counts


def summarize_elemental_annotation_lookup(orf_ids, orf_annotation_lookup):
    summary = {
        "annotation_elemental_assigned_orfs": np.nan,
        "annotation_elemental_assigned_fraction": np.nan,
    }
    for cycle in ELEMENTAL_CYCLE_ORDER:
        summary[f"annotation_{cycle}_orfs"] = np.nan
        summary[f"annotation_{cycle}_fraction"] = np.nan

    if orf_annotation_lookup is None or not orf_ids:
        return summary

    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    total = len(orf_ids)
    assigned_mask = lookup_hits["elemental_cycle_labels"].fillna("").ne("")
    summary["annotation_elemental_assigned_orfs"] = int(assigned_mask.sum())
    summary["annotation_elemental_assigned_fraction"] = (
        float(assigned_mask.sum()) / total if total else np.nan
    )
    for cycle in ELEMENTAL_CYCLE_ORDER:
        cycle_hits = lookup_hits[f"elemental_{cycle}"].fillna(False)
        count = int(cycle_hits.sum())
        summary[f"annotation_{cycle}_orfs"] = count
        summary[f"annotation_{cycle}_fraction"] = count / total if total else np.nan
    return summary


def summarize_elemental_mode_lookup(orf_ids, orf_annotation_lookup, prefix, denominator_label):
    summary = {
        f"{prefix}_elemental_mode_assigned_{denominator_label}": np.nan,
        f"{prefix}_elemental_mode_assigned_fraction": np.nan,
        f"{prefix}_elemental_mode_uniquely_assigned_{denominator_label}": np.nan,
        f"{prefix}_elemental_mode_uniquely_assigned_fraction": np.nan,
        f"{prefix}_elemental_mode_ambiguously_assigned_{denominator_label}": np.nan,
        f"{prefix}_elemental_mode_ambiguously_assigned_fraction": np.nan,
    }
    for mode in ELEMENTAL_MODE_ORDER:
        summary[f"{prefix}_{mode}_{denominator_label}"] = np.nan
        summary[f"{prefix}_{mode}_fraction"] = np.nan
        summary[f"{prefix}_unique_{mode}_{denominator_label}"] = np.nan
        summary[f"{prefix}_unique_{mode}_fraction"] = np.nan
        summary[f"{prefix}_ambiguous_{mode}_{denominator_label}"] = np.nan
        summary[f"{prefix}_ambiguous_{mode}_fraction"] = np.nan

    if orf_annotation_lookup is None or not orf_ids:
        return summary

    lookup_hits = orf_annotation_lookup.reindex(orf_ids)
    total = len(orf_ids)
    assigned_mask = lookup_hits["elemental_mode_labels"].fillna("").ne("")
    mode_counts = (
        lookup_hits["elemental_mode_labels"]
        .fillna("")
        .astype(str)
        .map(lambda text: len([token for token in text.split(";") if token]))
    )
    unique_mask = assigned_mask & mode_counts.eq(1)
    ambiguous_mask = assigned_mask & mode_counts.gt(1)
    summary[f"{prefix}_elemental_mode_assigned_{denominator_label}"] = int(assigned_mask.sum())
    summary[f"{prefix}_elemental_mode_assigned_fraction"] = (
        float(assigned_mask.sum()) / total if total else np.nan
    )
    summary[f"{prefix}_elemental_mode_uniquely_assigned_{denominator_label}"] = int(unique_mask.sum())
    summary[f"{prefix}_elemental_mode_uniquely_assigned_fraction"] = (
        float(unique_mask.sum()) / total if total else np.nan
    )
    summary[f"{prefix}_elemental_mode_ambiguously_assigned_{denominator_label}"] = int(ambiguous_mask.sum())
    summary[f"{prefix}_elemental_mode_ambiguously_assigned_fraction"] = (
        float(ambiguous_mask.sum()) / total if total else np.nan
    )
    for mode in ELEMENTAL_MODE_ORDER:
        mode_hits = lookup_hits[f"elemental_mode_{mode}"].fillna(False)
        count = int(mode_hits.sum())
        summary[f"{prefix}_{mode}_{denominator_label}"] = count
        summary[f"{prefix}_{mode}_fraction"] = count / total if total else np.nan
        unique_count = int((mode_hits & unique_mask).sum())
        summary[f"{prefix}_unique_{mode}_{denominator_label}"] = unique_count
        summary[f"{prefix}_unique_{mode}_fraction"] = unique_count / total if total else np.nan
        ambiguous_count = int((mode_hits & ambiguous_mask).sum())
        summary[f"{prefix}_ambiguous_{mode}_{denominator_label}"] = ambiguous_count
        summary[f"{prefix}_ambiguous_{mode}_fraction"] = ambiguous_count / total if total else np.nan
    return summary


def summarize_elemental_pathway_support(pwy2orf_df, orf_annotation_lookup):
    summary = {
        "pathway_support_elemental_assigned_orfs": np.nan,
        "pathway_support_elemental_assigned_fraction": np.nan,
    }
    for cycle in ELEMENTAL_CYCLE_ORDER:
        summary[f"pathway_support_{cycle}_orfs"] = np.nan
        summary[f"pathway_support_{cycle}_fraction"] = np.nan

    if pwy2orf_df is None or pwy2orf_df.empty or orf_annotation_lookup is None:
        return summary

    support_orfs = sorted(pwy2orf_df["orf_id"].dropna().astype(str).unique().tolist())
    if not support_orfs:
        return summary

    lookup_hits = orf_annotation_lookup.reindex(support_orfs)
    total = len(support_orfs)
    assigned_mask = lookup_hits["elemental_cycle_labels"].fillna("").ne("")
    summary["pathway_support_elemental_assigned_orfs"] = int(assigned_mask.sum())
    summary["pathway_support_elemental_assigned_fraction"] = (
        float(assigned_mask.sum()) / total if total else np.nan
    )
    for cycle in ELEMENTAL_CYCLE_ORDER:
        cycle_hits = lookup_hits[f"elemental_{cycle}"].fillna(False)
        count = int(cycle_hits.sum())
        summary[f"pathway_support_{cycle}_orfs"] = count
        summary[f"pathway_support_{cycle}_fraction"] = count / total if total else np.nan
    return summary


def summarize_elemental_mode_pathway_support(pwy2orf_df, orf_annotation_lookup):
    support_orfs = []
    if pwy2orf_df is not None and not pwy2orf_df.empty:
        support_orfs = sorted(pwy2orf_df["orf_id"].dropna().astype(str).unique().tolist())
    return summarize_elemental_mode_lookup(support_orfs, orf_annotation_lookup, "pathway_support", "orfs")


def summarize_elemental_pathways(pwy_df):
    summary = {
        "pathway_elemental_assigned_pathways": np.nan,
        "pathway_elemental_assigned_fraction": np.nan,
    }
    for cycle in ELEMENTAL_CYCLE_ORDER:
        summary[f"pathway_{cycle}_count"] = np.nan
        summary[f"pathway_{cycle}_fraction"] = np.nan

    if pwy_df is None or pwy_df.empty:
        return summary

    total = len(pwy_df)
    assigned_mask = pwy_df["elemental_cycle_labels"].fillna("").ne("")
    summary["pathway_elemental_assigned_pathways"] = int(assigned_mask.sum())
    summary["pathway_elemental_assigned_fraction"] = (
        float(assigned_mask.sum()) / total if total else np.nan
    )
    for cycle in ELEMENTAL_CYCLE_ORDER:
        cycle_hits = pwy_df[f"elemental_{cycle}"].fillna(False)
        count = int(cycle_hits.sum())
        summary[f"pathway_{cycle}_count"] = count
        summary[f"pathway_{cycle}_fraction"] = count / total if total else np.nan
    return summary


def summarize_elemental_pathway_modes(pwy_df):
    summary = {
        "pathway_elemental_mode_assigned_pathways": np.nan,
        "pathway_elemental_mode_assigned_fraction": np.nan,
        "pathway_elemental_mode_uniquely_assigned_pathways": np.nan,
        "pathway_elemental_mode_uniquely_assigned_fraction": np.nan,
        "pathway_elemental_mode_ambiguously_assigned_pathways": np.nan,
        "pathway_elemental_mode_ambiguously_assigned_fraction": np.nan,
    }
    for mode in ELEMENTAL_MODE_ORDER:
        summary[f"pathway_{mode}_count"] = np.nan
        summary[f"pathway_{mode}_fraction"] = np.nan
        summary[f"pathway_unique_{mode}_count"] = np.nan
        summary[f"pathway_unique_{mode}_fraction"] = np.nan
        summary[f"pathway_ambiguous_{mode}_count"] = np.nan
        summary[f"pathway_ambiguous_{mode}_fraction"] = np.nan

    if pwy_df is None or pwy_df.empty:
        return summary

    total = len(pwy_df)
    assigned_mask = pwy_df["elemental_mode_labels"].fillna("").ne("")
    mode_counts = (
        pwy_df["elemental_mode_labels"]
        .fillna("")
        .astype(str)
        .map(lambda text: len([token for token in text.split(";") if token]))
    )
    unique_mask = assigned_mask & mode_counts.eq(1)
    ambiguous_mask = assigned_mask & mode_counts.gt(1)
    summary["pathway_elemental_mode_assigned_pathways"] = int(assigned_mask.sum())
    summary["pathway_elemental_mode_assigned_fraction"] = (
        float(assigned_mask.sum()) / total if total else np.nan
    )
    summary["pathway_elemental_mode_uniquely_assigned_pathways"] = int(unique_mask.sum())
    summary["pathway_elemental_mode_uniquely_assigned_fraction"] = (
        float(unique_mask.sum()) / total if total else np.nan
    )
    summary["pathway_elemental_mode_ambiguously_assigned_pathways"] = int(ambiguous_mask.sum())
    summary["pathway_elemental_mode_ambiguously_assigned_fraction"] = (
        float(ambiguous_mask.sum()) / total if total else np.nan
    )
    for mode in ELEMENTAL_MODE_ORDER:
        mode_hits = pwy_df[f"elemental_mode_{mode}"].fillna(False)
        count = int(mode_hits.sum())
        summary[f"pathway_{mode}_count"] = count
        summary[f"pathway_{mode}_fraction"] = count / total if total else np.nan
        unique_count = int((mode_hits & unique_mask).sum())
        summary[f"pathway_unique_{mode}_count"] = unique_count
        summary[f"pathway_unique_{mode}_fraction"] = unique_count / total if total else np.nan
        ambiguous_count = int((mode_hits & ambiguous_mask).sum())
        summary[f"pathway_ambiguous_{mode}_count"] = ambiguous_count
        summary[f"pathway_ambiguous_{mode}_fraction"] = ambiguous_count / total if total else np.nan
    return summary


def summarize_ptinput_layer(orf_ids, ptinput_lookup):
    if ptinput_lookup is None or not orf_ids:
        return {
            "pathway_input_orfs": np.nan,
            "pathway_input_fraction": np.nan,
            "pathway_input_annotated_orfs": np.nan,
            "pathway_input_ec_orfs": np.nan,
            "pathway_input_taxon_orfs": np.nan,
            "pathway_input_partial_orfs": np.nan,
            "pathway_input_metacyc_orfs": np.nan,
            "pathway_input_swissprot_orfs": np.nan,
            "pathway_input_uniref50_orfs": np.nan,
            "nonfunctional_ptinput_rows": np.nan,
        }

    lookup_hits = ptinput_lookup.reindex(orf_ids)
    cds_hits = lookup_hits.loc[lookup_hits["is_cds"].fillna(False)].copy()
    pathway_input_orfs = int(cds_hits.index.nunique())
    total_orfs = len(orf_ids)
    return {
        "pathway_input_orfs": pathway_input_orfs,
        "pathway_input_fraction": pathway_input_orfs / total_orfs if total_orfs else np.nan,
        "pathway_input_annotated_orfs": int(cds_hits["is_annotated"].fillna(False).sum()),
        "pathway_input_ec_orfs": int(cds_hits["has_ec"].fillna(False).sum()),
        "pathway_input_taxon_orfs": int(cds_hits["has_taxon"].fillna(False).sum()),
        "pathway_input_partial_orfs": int(cds_hits["is_partial"].fillna(False).sum()),
        "pathway_input_metacyc_orfs": int(cds_hits.loc[cds_hits["sourcedb"].eq("metacyc")].index.nunique()),
        "pathway_input_swissprot_orfs": int(cds_hits.loc[cds_hits["sourcedb"].eq("swissprot")].index.nunique()),
        "pathway_input_uniref50_orfs": int(cds_hits.loc[cds_hits["sourcedb"].eq("uniref50")].index.nunique()),
        "nonfunctional_ptinput_rows": int(lookup_hits["is_nonfunctional_feature"].fillna(False).sum()),
    }


def summarize_pathway_table(pwy_df, high_conf_threshold):
    if pwy_df.empty:
        return {
            "total_pathways": 0,
            "high_confidence_pathways": 0,
            "complete_pathways": 0,
            "well_covered_pathways": 0,
            "median_pathway_score": np.nan,
            "mean_pathway_score": np.nan,
            "median_reaction_coverage": np.nan,
            "mean_reaction_coverage": np.nan,
            "total_pathway_orf_memberships": 0,
            "median_pathway_orf_count": np.nan,
            "mean_pathway_orf_count": np.nan,
        }

    coverage = pwy_df["reaction_coverage_fraction"]
    return {
        "total_pathways": int(len(pwy_df)),
        "high_confidence_pathways": int((pwy_df["PWY_SCORE"] >= high_conf_threshold).sum()),
        "complete_pathways": int((coverage >= 1.0).sum()),
        "well_covered_pathways": int((coverage >= 0.75).sum()),
        "median_pathway_score": float(pwy_df["PWY_SCORE"].median()),
        "mean_pathway_score": float(pwy_df["PWY_SCORE"].mean()),
        "median_reaction_coverage": float(coverage.median()),
        "mean_reaction_coverage": float(coverage.mean()),
        "total_pathway_orf_memberships": int(pwy_df["ORF_COUNT"].fillna(0).sum()),
        "median_pathway_orf_count": float(pwy_df["ORF_COUNT"].median()),
        "mean_pathway_orf_count": float(pwy_df["ORF_COUNT"].mean()),
    }


def summarize_pathway_orf_table(pwy2orf_df):
    if pwy2orf_df is None or pwy2orf_df.empty:
        return {
            "pathway_support_rows": 0,
            "pathway_support_orfs": 0,
            "pathway_support_ec_orfs": 0,
            "pathway_support_reactions": 0,
            "pathway_support_sources": 0,
        }

    ec_present = normalize_text(pwy2orf_df.get("EC", pd.Series("", index=pwy2orf_df.index))).ne("")
    rxn_present = normalize_text(pwy2orf_df.get("RXN", pd.Series("", index=pwy2orf_df.index))).ne("")
    source_present = normalize_text(pwy2orf_df.get("ref dbname", pd.Series("", index=pwy2orf_df.index))).ne("")
    return {
        "pathway_support_rows": int(len(pwy2orf_df)),
        "pathway_support_orfs": int(pwy2orf_df["orf_id"].nunique()),
        "pathway_support_ec_orfs": int(pwy2orf_df.loc[ec_present, "orf_id"].nunique()),
        "pathway_support_reactions": int(normalize_text(pwy2orf_df.get("RXN", pd.Series("", index=pwy2orf_df.index))).loc[rxn_present].nunique()),
        "pathway_support_sources": int(normalize_text(pwy2orf_df.get("ref dbname", pd.Series("", index=pwy2orf_df.index))).loc[source_present].nunique()),
    }


def load_pathway_tables(record, high_conf_threshold):
    if record["pwy_path"] is not None and Path(record["pwy_path"]).exists():
        pwy_df = read_table(record["pwy_path"]).copy()
        pwy_df["genome_id"] = record["genome_id"]
        pwy_df["PWY_SCORE"] = pd.to_numeric(pwy_df["PWY_SCORE"], errors="coerce")
        pwy_df["NUM_REACTIONS"] = pd.to_numeric(pwy_df["NUM_REACTIONS"], errors="coerce")
        pwy_df["NUM_COVERED_REACTIONS"] = pd.to_numeric(pwy_df["NUM_COVERED_REACTIONS"], errors="coerce")
        pwy_df["ORF_COUNT"] = pd.to_numeric(pwy_df["ORF_COUNT"], errors="coerce")
        pwy_df["reaction_coverage_fraction"] = np.where(
            pwy_df["NUM_REACTIONS"].fillna(0) > 0,
            pwy_df["NUM_COVERED_REACTIONS"] / pwy_df["NUM_REACTIONS"],
            np.nan,
        )
        pwy_df["high_confidence_pathway"] = pwy_df["PWY_SCORE"] >= high_conf_threshold
        pwy_df["complete_pathway"] = pwy_df["reaction_coverage_fraction"] >= 1.0
        pathway_text = normalize_text(
            pwy_df.get("PWY_COMMON_NAME", pd.Series("", index=pwy_df.index))
        )
        if "PWY_NAME" in pwy_df.columns:
            pathway_text = pathway_text.where(
                pathway_text.ne(""),
                normalize_text(pwy_df["PWY_NAME"]),
            )
        pwy_df["elemental_mode_labels"] = pathway_text.map(
            lambda text: ";".join(classify_elemental_modes(text))
        )
        for mode in ELEMENTAL_MODE_ORDER:
            pwy_df[f"elemental_mode_{mode}"] = pwy_df["elemental_mode_labels"].str.contains(mode, regex=False)
        pwy_df["elemental_cycle_labels"] = pathway_text.map(
            lambda text: ";".join(
                elemental_families_from_modes(classify_elemental_modes(text)) or classify_elemental_cycles(text)
            )
        )
        for cycle in ELEMENTAL_CYCLE_ORDER:
            pwy_df[f"elemental_{cycle}"] = pwy_df["elemental_cycle_labels"].str.contains(cycle, regex=False)
    else:
        pwy_df = build_empty_pathway_table(record["genome_id"])

    pwy2orf_df = None
    if record["pwy2orf_path"] is not None and Path(record["pwy2orf_path"]).exists():
        pwy2orf_df = read_table(record["pwy2orf_path"]).copy()
        pwy2orf_df["genome_id"] = record["genome_id"]
        pwy2orf_df["orf_id"] = normalize_text(pwy2orf_df["orf_id"])
    return pwy_df, pwy2orf_df


def select_genome_orf_ids(record, all_records, orf_annotation_lookup, pwy2orf_df):
    # In single-genome/community runs, the annotation table already represents the full ORF set.
    if (
        orf_annotation_lookup is not None
        and len(all_records) == 1
    ):
        return sorted(orf_annotation_lookup.index.astype(str).tolist()), "orf_annotation_table"

    if record["genes_dat_path"] is not None and Path(record["genes_dat_path"]).exists():
        return parse_genes_dat(record["genes_dat_path"]), "genes.dat"

    if pwy2orf_df is not None:
        return sorted(pwy2orf_df["orf_id"].dropna().astype(str).unique().tolist()), "pwy2orf"

    return [], "none"


def summarize_genome_record(
    record,
    all_records,
    high_conf_threshold,
    orf_annotation_lookup,
    ptinput_lookup,
    marker_manifest,
    marker_alias_to_rows,
    marker_alias_regex,
    accession_mode_lookup,
):
    genome_id = record["genome_id"]
    pwy_df, pwy2orf_df = load_pathway_tables(record, high_conf_threshold)
    pathway_audit_df = build_pathway_audit_table(pwy_df)

    genome_orf_ids, broad_orf_source = select_genome_orf_ids(
        record,
        all_records,
        orf_annotation_lookup,
        pwy2orf_df,
    )
    annotation_audit_df = build_annotation_audit_table(genome_id, genome_orf_ids, orf_annotation_lookup)
    marker_audit_df = build_marker_audit_table(
        genome_id,
        genome_orf_ids,
        orf_annotation_lookup,
        marker_manifest,
        marker_alias_to_rows=marker_alias_to_rows,
        marker_alias_regex=marker_alias_regex,
    )
    reference_mode_audit_df = build_reference_mode_audit_table(
        genome_id,
        genome_orf_ids,
        orf_annotation_lookup,
        accession_mode_lookup,
    )

    annotation_summary = summarize_annotation_lookup(genome_orf_ids, orf_annotation_lookup)
    elemental_annotation_summary = summarize_elemental_annotation_lookup(genome_orf_ids, orf_annotation_lookup)
    elemental_annotation_mode_summary = summarize_elemental_mode_lookup(
        genome_orf_ids,
        orf_annotation_lookup,
        "annotation",
        "orfs",
    )
    ptinput_summary = summarize_ptinput_layer(genome_orf_ids, ptinput_lookup)
    pathway_summary = summarize_pathway_table(pwy_df, high_conf_threshold)
    pathway_orf_summary = summarize_pathway_orf_table(pwy2orf_df)
    elemental_pathway_support_summary = summarize_elemental_pathway_support(pwy2orf_df, orf_annotation_lookup)
    elemental_pathway_support_mode_summary = summarize_elemental_mode_pathway_support(
        pwy2orf_df,
        orf_annotation_lookup,
    )
    elemental_pathway_summary = summarize_elemental_pathways(pwy_df)
    elemental_pathway_mode_summary = summarize_elemental_pathway_modes(pwy_df)
    marker_support_summary = summarize_marker_support(marker_audit_df)
    reference_mode_support_summary = summarize_reference_mode_support(reference_mode_audit_df)
    summary_row = {
        "genome_id": genome_id,
        "results_mode": record["mode"],
        "genome_dir": str(record["genome_dir"]),
        "broad_orf_source": broad_orf_source,
        "pathway_status": record.get("pathway_status", "present"),
        "pathway_missing_reason": record.get("pathway_missing_reason", ""),
        "pathway_table_present": bool(record.get("pwy_path")),
        "pathway_orf_table_present": bool(record.get("pwy2orf_path")),
        **annotation_summary,
        **elemental_annotation_summary,
        **elemental_annotation_mode_summary,
        **ptinput_summary,
        **pathway_summary,
        **pathway_orf_summary,
        **elemental_pathway_support_summary,
        **elemental_pathway_support_mode_summary,
        **elemental_pathway_summary,
        **elemental_pathway_mode_summary,
        **marker_support_summary,
        **reference_mode_support_summary,
    }
    return (
        summary_row,
        pwy_df,
        pwy2orf_df,
        annotation_audit_df,
        pathway_audit_df,
        marker_audit_df,
        reference_mode_audit_df,
    )


def build_summary_tables(
    results_dir,
    high_conf_threshold,
    allowed_genomes=None,
    taxonomy_label_lookup=None,
    marker_manifest=None,
    reference_mappings_dir=None,
    workers=1,
    progress=True,
    progress_interval=10,
    reference_chunk_size=500000,
    reference_progress_rows=2000000,
    heartbeat_seconds=60,
    reference_force_full_index=False,
    prep_workers=0,
):
    prep_worker_count = max(1, int(prep_workers)) if int(prep_workers) > 0 else max(1, int(workers))
    run_with_heartbeat(
        "configuring reference term mappings",
        lambda: configure_reference_term_maps(reference_mappings_dir),
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    marker_manifest = marker_manifest if marker_manifest is not None else load_marker_manifest(MARKER_MANIFEST_PATH)
    marker_alias_to_rows, marker_alias_regex = prepare_marker_matcher(marker_manifest)
    progress_log(
        f"[done] loaded marker manifest: rows={len(marker_manifest):,} aliases={len(marker_alias_to_rows):,}",
        enabled=progress,
    )
    records = run_with_heartbeat(
        "scanning MetaPathways genome records",
        lambda: find_genome_records(results_dir),
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    progress_log(f"[done] discovered genome records: {len(records):,}", enabled=progress)
    if allowed_genomes is not None:
        before = len(records)
        records = records.loc[
            records["genome_id"].astype(str).map(
                lambda genome_id: bool(id_aliases(genome_id) & allowed_genomes)
            )
        ].copy()
        progress_log(
            f"[done] genome filter applied: kept {len(records):,} / {before:,}",
            enabled=progress,
        )
        if records.empty:
            raise ValueError("No MetaPathways genome records remain after applying --genome-filter-tsv.")
    orf_annotation_lookup = run_with_heartbeat(
        "building ORF annotation lookup",
        lambda: build_orf_annotation_lookup(
            results_dir,
            workers=prep_worker_count,
            progress=progress,
        ),
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    orf_lookup_rows = 0 if orf_annotation_lookup is None else len(orf_annotation_lookup)
    progress_log(f"[done] ORF annotation lookup rows: {orf_lookup_rows:,}", enabled=progress)
    target_accessions = set()
    if not reference_force_full_index:
        target_accessions = run_with_heartbeat(
            "extracting target accessions from ORF annotations",
            lambda: collect_target_accessions_from_orf_lookup(
                orf_annotation_lookup,
                progress=progress,
                progress_interval=max(50000, int(reference_progress_rows) // 20),
                workers=prep_worker_count,
            ),
            enabled=progress,
            heartbeat_seconds=heartbeat_seconds,
        )
        progress_log(
            f"[done] extracted target accessions: {len(target_accessions):,}",
            enabled=progress,
        )
    else:
        progress_log(
            "[start] reference full-index mode enabled; skipping target accession extraction",
            enabled=progress,
        )
    accession_mode_lookup = run_with_heartbeat(
        "loading accession-mode reference lookup",
        lambda: load_reference_accession_modes(
            reference_mappings_dir,
            progress=progress,
            progress_interval_rows=reference_progress_rows,
            chunk_size=reference_chunk_size,
            target_accessions=None if reference_force_full_index else target_accessions,
            workers=prep_worker_count,
        ),
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
        emit_start=False,
        emit_done=False,
    )
    progress_log(
        f"[done] accession-mode lookup entries: {len(accession_mode_lookup):,}",
        enabled=progress,
    )
    ptinput_lookup = run_with_heartbeat(
        "building ptinput lookup",
        lambda: build_ptinput_lookup(results_dir),
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
    )
    ptinput_rows = 0 if ptinput_lookup is None else len(ptinput_lookup)
    progress_log(f"[done] ptinput lookup rows: {ptinput_rows:,}", enabled=progress)

    summary_rows = []
    pathway_frames = []
    pathway_orf_frames = []
    annotation_audit_frames = []
    pathway_audit_frames = []
    marker_audit_frames = []
    reference_mode_audit_frames = []
    all_records = records.to_dict("records")
    worker_count = max(1, int(workers))
    stage = StageProgress(
        label="per-genome summarization",
        total=len(all_records),
        workers=worker_count,
        interval_seconds=progress_interval,
        enabled=progress,
    )
    stage.start_stage()
    if worker_count > 1 and len(all_records) > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    summarize_genome_record,
                    record,
                    all_records,
                    high_conf_threshold,
                    orf_annotation_lookup,
                    ptinput_lookup,
                    marker_manifest,
                    marker_alias_to_rows,
                    marker_alias_regex,
                    accession_mode_lookup,
                )
                for record in all_records
            ]
            done = 0
            for future in as_completed(futures):
                (
                    summary_row,
                    pwy_df,
                    pwy2orf_df,
                    annotation_audit_df,
                    pathway_audit_df,
                    marker_audit_df,
                    reference_mode_audit_df,
                ) = future.result()
                summary_rows.append(summary_row)
                pathway_frames.append(pwy_df)
                pathway_audit_frames.append(pathway_audit_df)
                if pwy2orf_df is not None:
                    pathway_orf_frames.append(pwy2orf_df)
                annotation_audit_frames.append(annotation_audit_df)
                marker_audit_frames.append(marker_audit_df)
                reference_mode_audit_frames.append(reference_mode_audit_df)
                done += 1
                stage.update(done)
    else:
        done = 0
        for record in all_records:
            (
                summary_row,
                pwy_df,
                pwy2orf_df,
                annotation_audit_df,
                pathway_audit_df,
                marker_audit_df,
                reference_mode_audit_df,
            ) = summarize_genome_record(
                record,
                all_records,
                high_conf_threshold,
                orf_annotation_lookup,
                ptinput_lookup,
                marker_manifest,
                marker_alias_to_rows,
                marker_alias_regex,
                accession_mode_lookup,
            )
            summary_rows.append(summary_row)
            pathway_frames.append(pwy_df)
            pathway_audit_frames.append(pathway_audit_df)
            if pwy2orf_df is not None:
                pathway_orf_frames.append(pwy2orf_df)
            annotation_audit_frames.append(annotation_audit_df)
            marker_audit_frames.append(marker_audit_df)
            reference_mode_audit_frames.append(reference_mode_audit_df)
            done += 1
            stage.update(done)
    stage.done()

    progress_log("[start] concatenating summary tables", enabled=progress)
    genome_summary = pd.DataFrame(summary_rows).sort_values("genome_id").reset_index(drop=True)
    if taxonomy_label_lookup:
        genome_summary = apply_taxonomy_labels(genome_summary, taxonomy_label_lookup)
    else:
        genome_summary["genome_display_label"] = genome_summary["genome_id"].astype(str)
    pathway_long = (
        pd.concat(pathway_frames, ignore_index=True)
        if pathway_frames
        else pd.DataFrame()
    )
    pathway_orf_long = (
        pd.concat(pathway_orf_frames, ignore_index=True)
        if pathway_orf_frames
        else pd.DataFrame()
    )
    annotation_audit_long = (
        pd.concat(annotation_audit_frames, ignore_index=True)
        if annotation_audit_frames
        else pd.DataFrame()
    )
    pathway_audit_long = (
        pd.concat(pathway_audit_frames, ignore_index=True)
        if pathway_audit_frames
        else pd.DataFrame()
    )
    marker_audit_long = (
        pd.concat(marker_audit_frames, ignore_index=True)
        if marker_audit_frames
        else pd.DataFrame()
    )
    reference_mode_audit_long = (
        pd.concat(reference_mode_audit_frames, ignore_index=True)
        if reference_mode_audit_frames
        else pd.DataFrame()
    )
    progress_log(
        "[done] concatenated tables: "
        f"genomes={len(genome_summary):,}, pathways={len(pathway_long):,}, "
        f"pathway_orfs={len(pathway_orf_long):,}",
        enabled=progress,
    )
    return (
        genome_summary,
        pathway_long,
        pathway_orf_long,
        annotation_audit_long,
        pathway_audit_long,
        marker_audit_long,
        reference_mode_audit_long,
    )


def build_annotation_source_table(genome_summary):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        "total_orfs",
        "annotated_orfs",
        "annotation_fraction",
        "informative_annotation_orfs",
        "informative_annotation_fraction",
        "uncertain_annotation_orfs",
        "uncertain_annotation_fraction",
        "unannotated_orfs",
        "hypothetical_annotation_orfs",
        "uncharacterized_annotation_orfs",
        "domain_family_annotation_orfs",
        "qualifier_annotation_orfs",
        "fragment_annotation_orfs",
        "other_uncertain_annotation_orfs",
        "metacyc_orfs",
        "swissprot_orfs",
        "uniref50_orfs",
        "multi_source_orfs",
        "pathway_input_orfs",
        "pathway_input_fraction",
        "pathway_input_annotated_orfs",
        "pathway_input_ec_orfs",
        "pathway_input_metacyc_orfs",
        "pathway_input_swissprot_orfs",
        "pathway_input_uniref50_orfs",
        "nonfunctional_ptinput_rows",
    ]
    return genome_summary[[column for column in columns if column in genome_summary.columns]].copy()


def build_annotation_quality_table(genome_summary):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        "total_orfs",
        "annotated_orfs",
        "annotation_fraction",
        "informative_annotation_orfs",
        "informative_annotation_fraction",
        "uncertain_annotation_orfs",
        "uncertain_annotation_fraction",
        "hypothetical_annotation_orfs",
        "uncharacterized_annotation_orfs",
        "domain_family_annotation_orfs",
        "qualifier_annotation_orfs",
        "fragment_annotation_orfs",
        "other_uncertain_annotation_orfs",
        "unannotated_orfs",
    ]
    return genome_summary[[column for column in columns if column in genome_summary.columns]].copy()


def build_elemental_summary_table(genome_summary, prefix, denominator_column, unit_label, assigned_unit_label=None):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        denominator_column,
    ]
    assigned_unit_label = assigned_unit_label or unit_label
    assigned_count = f"{prefix}_elemental_assigned_{assigned_unit_label}"
    assigned_fraction = f"{prefix}_elemental_assigned_fraction"
    if assigned_count in genome_summary.columns:
        columns.append(assigned_count)
    if assigned_fraction in genome_summary.columns:
        columns.append(assigned_fraction)
    for cycle in ELEMENTAL_CYCLE_ORDER:
        count_column = f"{prefix}_{cycle}_{unit_label}"
        fraction_column = f"{prefix}_{cycle}_fraction"
        if count_column in genome_summary.columns:
            columns.append(count_column)
        if fraction_column in genome_summary.columns:
            columns.append(fraction_column)
    return genome_summary[[column for column in columns if column in genome_summary.columns]].copy()


def build_elemental_mode_summary_table(
    genome_summary,
    prefix,
    denominator_column,
    unit_label,
    assigned_unit_label=None,
    scope="inclusive",
):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        denominator_column,
    ]
    assigned_unit_label = assigned_unit_label or unit_label
    if scope == "inclusive":
        assigned_count = f"{prefix}_elemental_mode_assigned_{assigned_unit_label}"
        assigned_fraction = f"{prefix}_elemental_mode_assigned_fraction"
        column_prefix = f"{prefix}_"
    elif scope == "unique":
        assigned_count = f"{prefix}_elemental_mode_uniquely_assigned_{assigned_unit_label}"
        assigned_fraction = f"{prefix}_elemental_mode_uniquely_assigned_fraction"
        column_prefix = f"{prefix}_unique_"
    elif scope == "ambiguous":
        assigned_count = f"{prefix}_elemental_mode_ambiguously_assigned_{assigned_unit_label}"
        assigned_fraction = f"{prefix}_elemental_mode_ambiguously_assigned_fraction"
        column_prefix = f"{prefix}_ambiguous_"
    else:
        raise ValueError(f"Unsupported elemental mode summary scope: {scope}")
    if assigned_count in genome_summary.columns:
        columns.append(assigned_count)
    if assigned_fraction in genome_summary.columns:
        columns.append(assigned_fraction)
    for mode in ELEMENTAL_MODE_ORDER:
        count_column = f"{column_prefix}{mode}_{unit_label}"
        fraction_column = f"{column_prefix}{mode}_fraction"
        if count_column in genome_summary.columns:
            columns.append(count_column)
        if fraction_column in genome_summary.columns:
            columns.append(fraction_column)
    return genome_summary[[column for column in columns if column in genome_summary.columns]].copy()


def build_marker_summary_table(genome_summary, marker_manifest=None):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        "marker_supported_orfs",
        "marker_gene_count",
        "core_marker_gene_count",
        "specific_marker_gene_count",
        "generic_marker_gene_count",
    ]
    for mode_id in ELEMENTAL_MODE_ORDER:
        columns.extend(
            [
                f"marker_{mode_id}_orf_count",
                f"marker_{mode_id}_gene_count",
                f"marker_{mode_id}_core_gene_count",
                f"marker_{mode_id}_specific_gene_count",
                f"marker_{mode_id}_generic_gene_count",
            ]
        )
    summary = genome_summary[[column for column in columns if column in genome_summary.columns]].copy()
    possible_counts, core_possible_counts, specific_possible_counts, generic_possible_counts = marker_mode_possible_counts(marker_manifest)
    for mode_id in ELEMENTAL_MODE_ORDER:
        summary[f"marker_{mode_id}_possible_gene_count"] = possible_counts.get(mode_id, 0)
        summary[f"marker_{mode_id}_possible_core_gene_count"] = core_possible_counts.get(mode_id, 0)
        summary[f"marker_{mode_id}_possible_specific_gene_count"] = specific_possible_counts.get(mode_id, 0)
        summary[f"marker_{mode_id}_possible_generic_gene_count"] = generic_possible_counts.get(mode_id, 0)
    return summary


def build_reference_mode_summary_table(genome_summary):
    columns = [
        "genome_id",
        "genome_display_label",
        "taxonomy_display_rank",
        "taxonomy_display_value",
        "taxonomy_display_status",
        "taxonomy_match_method",
        "reference_mode_supported_orfs",
        "reference_mode_supported_accessions",
    ]
    for mode_id in ELEMENTAL_MODE_ORDER:
        columns.extend(
            [
                f"reference_mode_{mode_id}_orf_count",
                f"reference_mode_{mode_id}_accession_count",
            ]
        )
    return genome_summary[[column for column in columns if column in genome_summary.columns]].copy()


def build_annotation_category_definitions_table():
    rows = []
    for category in ANNOTATION_CATEGORY_ORDER:
        if category == "informative":
            definition = "Annotation text classified as informative rather than uncertain or unannotated."
            triggers = ""
        elif category == "unannotated":
            definition = "No usable functional annotation text detected across supported source columns."
            triggers = ""
        elif category == "other_uncertain":
            definition = "Annotated ORF with non-informative text that did not match a more specific uncertain category."
            triggers = ""
        else:
            trigger_list = UNCERTAIN_ANNOTATION_PATTERNS.get(category, [])
            definition = "Annotated ORF assigned to this uncertain annotation grouping."
            triggers = "; ".join(trigger_list)
        rows.append(
            {
                "annotation_category": category,
                "display_label": ANNOTATION_CATEGORY_LABELS.get(category, category),
                "classification_group": "uncertain"
                if category not in {"informative", "unannotated"}
                else ("informative" if category == "informative" else "unannotated"),
                "definition": definition,
                "trigger_patterns": triggers,
            }
        )
    return pd.DataFrame(rows)


def build_experimental_mobility_hit_table(annotation_audit_long, keyword_map):
    columns = [
        "genome_id",
        "orf_id",
        "source_db",
        "annotation_text",
        "category",
        "matched_keywords",
    ]
    if annotation_audit_long is None or annotation_audit_long.empty:
        return pd.DataFrame(columns=columns)

    working = annotation_audit_long.copy()
    working["annotation_text"] = working["annotation_text"].astype(str).str.strip()
    working = working.loc[working["annotation_text"].ne("")].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for record in working.to_dict("records"):
        hits = detect_experimental_mobility_hits(record.get("annotation_text", ""), keyword_map)
        if not hits:
            continue
        for category, matched_keywords in hits.items():
            rows.append(
                {
                    "genome_id": str(record.get("genome_id", "")).strip(),
                    "orf_id": str(record.get("orf_id", "")).strip(),
                    "source_db": str(record.get("source_db", "")).strip(),
                    "annotation_text": str(record.get("annotation_text", "")).strip(),
                    "category": category,
                    "matched_keywords": ";".join(matched_keywords),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_experimental_mobility_per_genome_summary(genome_summary, genome_type_df, hit_df):
    columns = [
        "genome_id",
        "genome_display_label",
        "genome_type",
        "genome_type_source",
        "plasmid_associated",
        "phage_prophage_associated",
        "generic_mge_associated",
        "genomic_island_associated",
    ]
    if genome_summary.empty:
        return pd.DataFrame(columns=columns)

    base = genome_summary.copy()
    display_column = "genome_display_label" if "genome_display_label" in base.columns else "genome_id"
    base = base.loc[:, ["genome_id", display_column]].drop_duplicates("genome_id").copy()
    base = base.rename(columns={display_column: "genome_display_label"})
    base = base.merge(genome_type_df, on="genome_id", how="left")

    for category in EXPERIMENTAL_MOBILITY_CATEGORY_ORDER:
        base[EXPERIMENTAL_MOBILITY_CATEGORY_COLUMNS[category]] = False

    if hit_df is not None and not hit_df.empty:
        positive_pairs = hit_df.loc[:, ["genome_id", "category"]].drop_duplicates().copy()
        for category in EXPERIMENTAL_MOBILITY_CATEGORY_ORDER:
            positive_ids = set(
                positive_pairs.loc[positive_pairs["category"].eq(category), "genome_id"].astype(str).tolist()
            )
            base[EXPERIMENTAL_MOBILITY_CATEGORY_COLUMNS[category]] = base["genome_id"].astype(str).isin(positive_ids)

    return base.loc[:, columns].sort_values(by=["genome_type", "genome_id"], kind="mergesort").reset_index(drop=True)


def build_experimental_mobility_prevalence_summary(per_genome_df):
    rows = []
    if per_genome_df is None or per_genome_df.empty:
        return pd.DataFrame(
            columns=[
                "genome_type",
                "category",
                "n_genomes_total",
                "n_genomes_positive",
                "prevalence_fraction",
                "prevalence_percent",
            ]
        )

    for genome_type, group in per_genome_df.groupby("genome_type", dropna=False):
        total = int(group["genome_id"].astype(str).nunique())
        for category in EXPERIMENTAL_MOBILITY_CATEGORY_ORDER:
            column = EXPERIMENTAL_MOBILITY_CATEGORY_COLUMNS[category]
            positive = int(group[column].fillna(False).astype(bool).sum())
            fraction = (positive / total) if total else np.nan
            rows.append(
                {
                    "genome_type": str(genome_type),
                    "category": category,
                    "n_genomes_total": total,
                    "n_genomes_positive": positive,
                    "prevalence_fraction": fraction,
                    "prevalence_percent": (fraction * 100.0) if pd.notna(fraction) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_experimental_mobility_prevalence(prevalence_df, output_base):
    ensure_plotting()
    if prevalence_df is None or prevalence_df.empty:
        return False

    working = prevalence_df.copy()
    working["category"] = pd.Categorical(
        working["category"],
        categories=EXPERIMENTAL_MOBILITY_CATEGORY_ORDER,
        ordered=True,
    )
    working = working.sort_values(by=["category", "genome_type"], kind="mergesort")
    genome_types = sorted(
        [
            value
            for value in working["genome_type"].astype(str).unique().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
    )
    if not genome_types:
        return False

    plt_local = ensure_plotting()
    x_positions = np.arange(len(EXPERIMENTAL_MOBILITY_CATEGORY_ORDER), dtype=float)
    width = min(0.8 / max(1, len(genome_types)), 0.36) if len(genome_types) > 1 else 0.55
    color_map = {}
    if len(genome_types) <= 10:
        cmap = plt_local.get_cmap("tab10")
        for index, genome_type in enumerate(genome_types):
            color_map[genome_type] = cmap(index % 10)
    else:
        cmap = plt_local.get_cmap("viridis")
        denominator = max(1, len(genome_types) - 1)
        for index, genome_type in enumerate(genome_types):
            color_map[genome_type] = cmap(index / denominator)
    fig, ax = plt_local.subplots(figsize=(11, 6))

    for index, genome_type in enumerate(genome_types):
        subset = working.loc[working["genome_type"].astype(str).eq(genome_type)].copy()
        value_map = {
            row["category"]: float(row["prevalence_percent"])
            for _, row in subset.iterrows()
            if pd.notna(row["prevalence_percent"])
        }
        y_values = [value_map.get(category, 0.0) for category in EXPERIMENTAL_MOBILITY_CATEGORY_ORDER]
        offset = (index - ((len(genome_types) - 1) / 2.0)) * width
        ax.bar(
            x_positions + offset,
            y_values,
            width=width,
            color=color_map.get(genome_type, "#808080"),
            edgecolor="#333333",
            linewidth=0.6,
            label=genome_type,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(EXPERIMENTAL_MOBILITY_CATEGORY_ORDER, rotation=20, ha="right")
    ax.set_ylabel("Prevalence (%)")
    ax.set_xlabel("Category")
    ax.set_title("Experimental candidate mobility marker prevalence by genome type")
    ax.set_ylim(0, max(5.0, float(working["prevalence_percent"].fillna(0).max()) * 1.15))
    ax.legend(frameon=False, title="Genome type")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def build_experimental_mobility_outputs(
    genome_summary,
    annotation_audit_long,
    genome_type_tsv=None,
    genome_type_column=None,
    genome_type_id_column=None,
    include_optional_broad_screen=False,
):
    outputs = {}
    genome_type_df = resolve_experimental_genome_type_lookup(
        genome_summary,
        genome_type_tsv=genome_type_tsv,
        genome_type_column=genome_type_column,
        genome_type_id_column=genome_type_id_column,
    )
    main_hit_df = build_experimental_mobility_hit_table(
        annotation_audit_long,
        EXPERIMENTAL_MOBILITY_KEYWORDS,
    )
    per_genome_df = build_experimental_mobility_per_genome_summary(
        genome_summary,
        genome_type_df,
        main_hit_df,
    )
    prevalence_df = build_experimental_mobility_prevalence_summary(per_genome_df)
    outputs["genome_type_lookup"] = genome_type_df
    outputs["hit_table"] = main_hit_df
    outputs["per_genome_table"] = per_genome_df
    outputs["prevalence_table"] = prevalence_df

    if include_optional_broad_screen:
        broad_keyword_map = {"optional broad screen": EXPERIMENTAL_MOBILITY_OPTIONAL_BROAD_KEYWORDS}
        broad_hit_df = build_experimental_mobility_hit_table(annotation_audit_long, broad_keyword_map)
        broad_per_genome_df = genome_type_df.copy()
        broad_per_genome_df = genome_summary.loc[:, ["genome_id"]].drop_duplicates().merge(
            broad_per_genome_df,
            on="genome_id",
            how="left",
        )
        positive_ids = set(broad_hit_df["genome_id"].astype(str).tolist()) if not broad_hit_df.empty else set()
        broad_per_genome_df["optional_broad_screen"] = broad_per_genome_df["genome_id"].astype(str).isin(positive_ids)
        broad_rows = []
        for genome_type, group in broad_per_genome_df.groupby("genome_type", dropna=False):
            total = int(group["genome_id"].astype(str).nunique())
            positive = int(group["optional_broad_screen"].fillna(False).astype(bool).sum())
            fraction = (positive / total) if total else np.nan
            broad_rows.append(
                {
                    "genome_type": str(genome_type),
                    "category": "optional broad screen",
                    "n_genomes_total": total,
                    "n_genomes_positive": positive,
                    "prevalence_fraction": fraction,
                    "prevalence_percent": (fraction * 100.0) if pd.notna(fraction) else np.nan,
                }
            )
        outputs["optional_broad_hit_table"] = broad_hit_df
        outputs["optional_broad_per_genome_table"] = broad_per_genome_df
        outputs["optional_broad_prevalence_table"] = pd.DataFrame(broad_rows)

    return outputs


def build_pathway_matrix(pathway_long, value_column):
    if pathway_long.empty or value_column not in pathway_long.columns:
        return pd.DataFrame()
    matrix = (
        pathway_long.pivot_table(
            index="genome_id",
            columns="PWY_NAME",
            values=value_column,
            aggfunc="max",
            fill_value=0,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    matrix.columns.name = None
    matrix = matrix.reset_index()
    return matrix


def genome_order(genome_summary):
    if genome_summary.empty:
        return []
    if "total_pathways" in genome_summary.columns:
        return (
            genome_summary.sort_values(
                by=["total_pathways", "annotated_orfs", "total_orfs", "genome_id"],
                ascending=[False, False, False, True],
            )["genome_id"]
            .astype(str)
            .tolist()
        )
    return genome_summary["genome_id"].astype(str).tolist()


def ordered_display_labels(genome_summary, order):
    if not order:
        return []
    if "genome_display_label" not in genome_summary.columns:
        return [str(genome_id) for genome_id in order]
    ordered = genome_summary.set_index("genome_id").reindex(order)
    labels = ordered["genome_display_label"].fillna(ordered.index.to_series().astype(str)).astype(str)
    return labels.tolist()


def grayscale_palette(n_values, start=0.25, stop=0.8):
    if n_values <= 1:
        return ["#4d4d4d"]
    values = np.linspace(start, stop, n_values)
    return [str(value) for value in values]


def annotate_bar_values(ax, values):
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        ax.text(index, float(value), f"{int(round(float(value)))}", ha="center", va="bottom", fontsize=8)


def save_figure(fig, output_base):
    fig.savefig(str(output_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def heatmap_text_color(value, vmax):
    if vmax <= 0:
        return "black"
    return "white" if float(value) >= (0.55 * float(vmax)) else "black"


def family_tinted_rgb(family_id, value, vmax):
    ensure_plotting()
    from matplotlib.colors import to_rgb

    base_rgb = np.array(to_rgb(ELEMENTAL_CYCLE_COLORS[family_id]), dtype=float)
    if vmax <= 0 or value <= 0:
        return np.array([1.0, 1.0, 1.0], dtype=float)
    factor = max(0.18, min(1.0, float(value) / float(vmax)))
    return 1.0 - factor * (1.0 - base_rgb)


def plot_compact_summary(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    plot_df = genome_summary.set_index("genome_id").reindex(order).reset_index()
    x_labels = ordered_display_labels(genome_summary, order)
    x_positions = np.arange(len(plot_df))
    palette = grayscale_palette(len(plot_df))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, COMPACT_METRICS):
        ax.bar(
            x_positions,
            plot_df[metric].fillna(0).astype(float),
            color=palette,
            edgecolor="black",
            linewidth=0.6,
        )
        annotate_bar_values(ax, plot_df[metric].fillna(0).tolist())
        ax.set_xlabel("Genome")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90)

    fig.suptitle("MetaPathways compact genome summary", fontsize=16, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, output_base)
    return True


def plot_annotation_sources(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    plot_df = genome_summary.set_index("genome_id").reindex(order)
    x_labels = ordered_display_labels(genome_summary, order)
    x_positions = np.arange(len(plot_df))
    source_columns = [
        ("metacyc_orfs", "MetaCyc", "#1a1a1a"),
        ("swissprot_orfs", "SwissProt", "#5f5f5f"),
        ("uniref50_orfs", "UniRef50", "#9a9a9a"),
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 0.35), 6.5))
    bottom = np.zeros(len(plot_df), dtype=float)

    for column, label, color in source_columns:
        values = plot_df[column].fillna(0).astype(float).values if column in plot_df.columns else np.zeros(len(plot_df))
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += values

    ax.set_xlabel("Genome")
    ax.set_ylabel("ORF count")
    ax.set_title("Annotation sources by genome")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_figure(fig, output_base)
    return True


def plot_annotation_quality(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    plot_df = genome_summary.set_index("genome_id").reindex(order)
    x_labels = ordered_display_labels(genome_summary, order)
    x_positions = np.arange(len(plot_df))
    category_columns = [
        ("informative", "informative_annotation_orfs"),
        ("hypothetical", "hypothetical_annotation_orfs"),
        ("uncharacterized", "uncharacterized_annotation_orfs"),
        ("domain_family", "domain_family_annotation_orfs"),
        ("qualifier", "qualifier_annotation_orfs"),
        ("fragment", "fragment_annotation_orfs"),
        ("other_uncertain", "other_uncertain_annotation_orfs"),
        ("unannotated", "unannotated_orfs"),
    ]
    fig, ax = plt.subplots(figsize=(max(10, len(plot_df) * 0.4), 7))
    bottom = np.zeros(len(plot_df), dtype=float)

    for category, column in category_columns:
        values = plot_df[column].fillna(0).astype(float).values if column in plot_df.columns else np.zeros(len(plot_df))
        ax.bar(
            x_positions,
            values,
            bottom=bottom,
            label=ANNOTATION_CATEGORY_LABELS[category],
            color=ANNOTATION_CATEGORY_COLORS[category],
            edgecolor="black",
            linewidth=0.5,
        )
        bottom += values

    informative_fraction = plot_df["informative_annotation_fraction"] if "informative_annotation_fraction" in plot_df.columns else pd.Series(np.nan, index=plot_df.index)
    for index, genome_id in enumerate(plot_df.index.tolist()):
        total = bottom[index]
        if total <= 0:
            continue
        frac = informative_fraction.loc[genome_id]
        label = f"{frac:.2f}" if pd.notna(frac) else f"{int(total)}"
        ax.text(index, total, label, ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Genome")
    ax.set_ylabel("ORF count")
    ax.set_title("Annotation information content by genome")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    save_figure(fig, output_base)
    return True


def plot_pathway_metric_panels(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    plot_df = genome_summary.set_index("genome_id").reindex(order).reset_index()
    x_labels = ordered_display_labels(genome_summary, order)
    x_positions = np.arange(len(plot_df))
    metric_info = [
        ("pathway_input_orfs", "Pathway-input ORFs"),
        ("pathway_support_orfs", "Pathway-supporting ORFs"),
        ("median_pathway_score", "Median pathway score"),
        ("mean_reaction_coverage", "Mean reaction coverage"),
    ]
    palette = grayscale_palette(len(plot_df))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.ravel()

    for ax, (metric, label) in zip(axes, metric_info):
        values = plot_df[metric].fillna(0).astype(float)
        ax.bar(
            x_positions,
            values,
            color=palette,
            edgecolor="black",
            linewidth=0.6,
        )
        if metric in {"median_pathway_score", "mean_reaction_coverage", "annotation_fraction"}:
            ax.set_ylim(0, max(1.02, float(values.max()) + 0.05))
        ax.set_xlabel("Genome")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=90)

    fig.suptitle("MetaPathways pathway and annotation metrics", fontsize=16, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, output_base)
    return True


def plot_elemental_metabolism(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    y_labels = ordered_display_labels(genome_summary, order)
    panel_specs = [
        ("annotation", "Annotation-supported ORFs", "orfs"),
        ("pathway_support", "Pathway-supporting ORFs", "orfs"),
        ("pathway", "Inferred pathways", "count"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(order) * 0.22)), sharey=True)

    for ax, (prefix, title, unit_label) in zip(axes, panel_specs):
        value_columns = [f"{prefix}_{cycle}_{unit_label}" for cycle in ELEMENTAL_CYCLE_ORDER]
        available_columns = [column for column in value_columns if column in genome_summary.columns]
        if not available_columns:
            ax.axis("off")
            ax.text(0.5, 0.5, "No elemental summary available", ha="center", va="center")
            continue

        heat_df = (
            genome_summary.set_index("genome_id")
            .reindex(order)[available_columns]
            .rename(columns={f"{prefix}_{cycle}_{unit_label}": ELEMENTAL_CYCLE_LABELS[cycle] for cycle in ELEMENTAL_CYCLE_ORDER if f"{prefix}_{cycle}_{unit_label}" in genome_summary.columns})
            .fillna(0)
        )
        vmax = max(1.0, float(np.nanmax(heat_df.values)))
        image = ax.imshow(
            heat_df.values,
            aspect="auto",
            cmap="Greys",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(heat_df.columns)))
        ax.set_xticklabels(heat_df.columns, rotation=90)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        for row_index in range(len(order)):
            for col_index in range(len(heat_df.columns)):
                ax.text(
                    col_index,
                    row_index,
                    str(int(heat_df.iat[row_index, col_index])),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=heatmap_text_color(heat_df.iat[row_index, col_index], vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Count")

    axes[0].set_ylabel("Genome")
    fig.suptitle("Elemental-cycle metabolism summary", fontsize=16, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, output_base)
    return True


def plot_elemental_modes(genome_summary, output_base, scope="inclusive"):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    y_labels = ordered_display_labels(genome_summary, order)
    if scope == "inclusive":
        scope_prefix = ""
        scope_title = "All assignments"
    elif scope == "unique":
        scope_prefix = "unique_"
        scope_title = "Unique assignments only"
    elif scope == "ambiguous":
        scope_prefix = "ambiguous_"
        scope_title = "Ambiguous assignments only"
    else:
        raise ValueError(f"Unsupported elemental mode plot scope: {scope}")
    panel_specs = [
        ("annotation", "Annotation-supported modes", "orfs"),
        ("pathway_support", "Pathway-supporting modes", "orfs"),
        ("pathway", "Inferred pathway modes", "count"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, max(7, len(order) * 0.24)), sharey=True)

    for ax, (prefix, title, unit_label) in zip(axes, panel_specs):
        value_columns = [f"{prefix}_{scope_prefix}{mode}_{unit_label}" for mode in ELEMENTAL_MODE_ORDER]
        available_columns = [column for column in value_columns if column in genome_summary.columns]
        if not available_columns:
            ax.axis("off")
            ax.text(0.5, 0.5, "No metabolism-mode summary available", ha="center", va="center")
            continue

        heat_df = (
            genome_summary.set_index("genome_id")
            .reindex(order)[available_columns]
            .rename(
                columns={
                    f"{prefix}_{scope_prefix}{mode}_{unit_label}": ELEMENTAL_MODE_LABELS[mode]
                    for mode in ELEMENTAL_MODE_ORDER
                    if f"{prefix}_{scope_prefix}{mode}_{unit_label}" in genome_summary.columns
                }
            )
            .fillna(0)
        )
        vmax = max(1.0, float(np.nanmax(heat_df.values)))
        image = ax.imshow(
            heat_df.values,
            aspect="auto",
            cmap="Greys",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(heat_df.columns)))
        ax.set_xticklabels(heat_df.columns, rotation=90)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(y_labels)
        ax.set_title(title)
        for row_index in range(len(order)):
            for col_index in range(len(heat_df.columns)):
                value = heat_df.iat[row_index, col_index]
                ax.text(
                    col_index,
                    row_index,
                    str(int(value)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=heatmap_text_color(value, vmax),
                )
        cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
        cbar.set_label("Count")

    axes[0].set_ylabel("Genome")
    fig.suptitle(f"Specific metabolism-mode summary ({scope_title})", fontsize=16, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_figure(fig, output_base)
    return True


def plot_marker_heatmap(genome_summary, output_base, marker_manifest=None, specificity="all"):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    y_labels = ordered_display_labels(genome_summary, order)
    if specificity == "all":
        marker_columns = [f"marker_{mode_id}_gene_count" for mode_id in ELEMENTAL_MODE_ORDER]
        plot_title = "Marker-support heatmap"
        figure_title = "Mode support by curated marker genes"
    elif specificity == "specific":
        marker_columns = [f"marker_{mode_id}_specific_gene_count" for mode_id in ELEMENTAL_MODE_ORDER]
        plot_title = "Specific-marker heatmap"
        figure_title = "Mode support by specific curated marker genes"
    elif specificity == "generic":
        marker_columns = [f"marker_{mode_id}_generic_gene_count" for mode_id in ELEMENTAL_MODE_ORDER]
        plot_title = "Generic-marker heatmap"
        figure_title = "Mode support by generic curated marker genes"
    else:
        raise ValueError(f"Unsupported marker heatmap specificity: {specificity}")
    available_columns = [column for column in marker_columns if column in genome_summary.columns]
    if not available_columns:
        return False

    possible_counts, _, specific_possible_counts, generic_possible_counts = marker_mode_possible_counts(marker_manifest)
    mode_ids = [
        mode_id for mode_id in ELEMENTAL_MODE_ORDER
        if (
            (specificity == "all" and f"marker_{mode_id}_gene_count" in available_columns)
            or (specificity == "specific" and f"marker_{mode_id}_specific_gene_count" in available_columns)
            or (specificity == "generic" and f"marker_{mode_id}_generic_gene_count" in available_columns)
        )
    ]
    if specificity == "specific":
        label_counts = specific_possible_counts
    elif specificity == "generic":
        label_counts = generic_possible_counts
    else:
        label_counts = possible_counts
    heat_df = (
        genome_summary.set_index("genome_id")
        .reindex(order)[available_columns]
        .rename(
            columns={
                column: ELEMENTAL_MODE_LABELS[mode_id]
                for mode_id in ELEMENTAL_MODE_ORDER
                for column in [f"marker_{mode_id}_gene_count", f"marker_{mode_id}_specific_gene_count", f"marker_{mode_id}_generic_gene_count"]
                if column in genome_summary.columns
            }
        )
        .fillna(0)
        .astype(float)
    )
    if heat_df.empty:
        return False

    vmax = max(1.0, float(np.nanmax(heat_df.values)))
    rgb = np.ones((heat_df.shape[0], heat_df.shape[1], 3), dtype=float)
    family_ids = [ELEMENTAL_MODE_FAMILY[mode_id] for mode_id in mode_ids]
    for col_index, family_id in enumerate(family_ids):
        for row_index in range(heat_df.shape[0]):
            value = heat_df.iat[row_index, col_index]
            rgb[row_index, col_index, :] = family_tinted_rgb(family_id, value, vmax)

    plt_local = ensure_plotting()
    fig = plt_local.figure(figsize=(max(18, len(mode_ids) * 0.68), max(10.5, len(order) * 0.36)))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.38)
    header_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    header_rgb = np.ones((1, len(mode_ids), 3), dtype=float)
    from matplotlib.colors import to_rgb
    for col_index, family_id in enumerate(family_ids):
        header_rgb[0, col_index, :] = np.array(to_rgb(ELEMENTAL_CYCLE_COLORS[family_id]), dtype=float)
    header_ax.imshow(header_rgb, aspect="auto")
    header_ax.set_yticks([])
    family_centers = []
    family_labels = []
    start = 0
    while start < len(family_ids):
        family_id = family_ids[start]
        end = start
        while end + 1 < len(family_ids) and family_ids[end + 1] == family_id:
            end += 1
        family_centers.append((start + end) / 2.0)
        family_labels.append(ELEMENTAL_CYCLE_LABELS[family_id])
        header_ax.axvline(end + 0.5, color="white", linewidth=1.2)
        start = end + 1
    header_ax.set_xticks(family_centers)
    header_ax.set_xticklabels(family_labels, fontsize=9)
    header_ax.xaxis.set_ticks_position("top")
    header_ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=4)
    for spine in header_ax.spines.values():
        spine.set_visible(False)

    ax.imshow(rgb, aspect="auto")
    ax.set_xticks(np.arange(len(mode_ids)))
    ax.set_xticklabels(
        [f"{ELEMENTAL_MODE_LABELS[mode_id]}\n(n={label_counts.get(mode_id, 0)})" for mode_id in mode_ids],
        rotation=90,
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Metabolism mode")
    ax.set_ylabel("Genome")
    ax.set_title(plot_title, pad=2)

    last_family = None
    for col_index, family_id in enumerate(family_ids):
        if last_family is not None and family_id != last_family:
            ax.axvline(col_index - 0.5, color="black", linewidth=1.2)
        last_family = family_id

    for row_index in range(len(order)):
        for col_index in range(len(mode_ids)):
            value = int(round(float(heat_df.iat[row_index, col_index])))
            color = heatmap_text_color(value, vmax)
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=7, color=color)

    fig.suptitle(figure_title, fontsize=15, y=0.992)
    fig.text(
        0.5,
        0.965,
        "Axis labels show possible curated markers for the displayed marker class.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.84, hspace=0.36)
    save_figure(fig, output_base)
    return True


def plot_reference_mode_heatmap(genome_summary, output_base):
    ensure_plotting()
    if genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    y_labels = ordered_display_labels(genome_summary, order)
    value_columns = [f"reference_mode_{mode_id}_accession_count" for mode_id in ELEMENTAL_MODE_ORDER]
    available_columns = [column for column in value_columns if column in genome_summary.columns]
    if not available_columns:
        return False

    heat_df = (
        genome_summary.set_index("genome_id")
        .reindex(order)[available_columns]
        .rename(
            columns={
                f"reference_mode_{mode_id}_accession_count": ELEMENTAL_MODE_LABELS[mode_id]
                for mode_id in ELEMENTAL_MODE_ORDER
            }
        )
        .fillna(0)
        .astype(float)
    )
    if heat_df.empty:
        return False

    vmax = max(1.0, float(np.nanmax(heat_df.values)))
    mode_ids = [mode_id for mode_id in ELEMENTAL_MODE_ORDER if f"reference_mode_{mode_id}_accession_count" in available_columns]
    family_ids = [ELEMENTAL_MODE_FAMILY[mode_id] for mode_id in mode_ids]
    rgb = np.ones((heat_df.shape[0], heat_df.shape[1], 3), dtype=float)
    for col_index, family_id in enumerate(family_ids):
        for row_index in range(heat_df.shape[0]):
            value = heat_df.iat[row_index, col_index]
            rgb[row_index, col_index, :] = family_tinted_rgb(family_id, value, vmax)

    plt_local = ensure_plotting()
    fig = plt_local.figure(figsize=(max(18, len(mode_ids) * 0.65), max(10.5, len(order) * 0.36)))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 1.0], hspace=0.38)
    header_ax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0])

    header_rgb = np.ones((1, len(mode_ids), 3), dtype=float)
    from matplotlib.colors import to_rgb
    for col_index, family_id in enumerate(family_ids):
        header_rgb[0, col_index, :] = np.array(to_rgb(ELEMENTAL_CYCLE_COLORS[family_id]), dtype=float)
    header_ax.imshow(header_rgb, aspect="auto")
    header_ax.set_yticks([])
    family_centers = []
    family_labels = []
    start = 0
    while start < len(family_ids):
        family_id = family_ids[start]
        end = start
        while end + 1 < len(family_ids) and family_ids[end + 1] == family_id:
            end += 1
        family_centers.append((start + end) / 2.0)
        family_labels.append(ELEMENTAL_CYCLE_LABELS[family_id])
        header_ax.axvline(end + 0.5, color="white", linewidth=1.2)
        start = end + 1
    header_ax.set_xticks(family_centers)
    header_ax.set_xticklabels(family_labels, fontsize=9)
    header_ax.xaxis.set_ticks_position("top")
    header_ax.tick_params(axis="x", labeltop=True, labelbottom=False, length=0, pad=4)
    for spine in header_ax.spines.values():
        spine.set_visible(False)

    ax.imshow(rgb, aspect="auto")
    ax.set_xticks(np.arange(len(mode_ids)))
    ax.set_xticklabels([ELEMENTAL_MODE_LABELS[mode_id] for mode_id in mode_ids], rotation=90)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Metabolism mode")
    ax.set_ylabel("Genome")
    ax.set_title("Reference-supported mode evidence", pad=2)

    last_family = None
    for col_index, family_id in enumerate(family_ids):
        if last_family is not None and family_id != last_family:
            ax.axvline(col_index - 0.5, color="black", linewidth=1.2)
        last_family = family_id

    for row_index in range(len(order)):
        for col_index in range(len(mode_ids)):
            value = int(round(float(heat_df.iat[row_index, col_index])))
            color = heatmap_text_color(value, vmax)
            ax.text(col_index, row_index, str(value), ha="center", va="center", fontsize=7, color=color)

    fig.suptitle("Mode support from accession-level GO reference mappings", fontsize=15, y=0.992)
    fig.text(
        0.5,
        0.965,
        "Cell text = number of matched accessions linked to that mode via GOA reference mappings",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.84, hspace=0.36)
    save_figure(fig, output_base)
    return True


def plot_top_pathway_heatmap(pathway_long, genome_summary, output_base, top_n_pathways):
    ensure_plotting()
    if pathway_long.empty or genome_summary.empty:
        return False

    order = genome_order(genome_summary)
    y_labels = ordered_display_labels(genome_summary, order)
    top_pathways = (
        pathway_long.groupby("PWY_NAME")["genome_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n_pathways)
        .index
        .tolist()
    )
    if not top_pathways:
        return False

    heat_df = (
        pathway_long.loc[pathway_long["PWY_NAME"].isin(top_pathways)]
        .pivot_table(
            index="genome_id",
            columns="PWY_NAME",
            values="PWY_SCORE",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=order, columns=top_pathways, fill_value=0)
    )
    count_df = (
        pathway_long.loc[pathway_long["PWY_NAME"].isin(top_pathways)]
        .assign(pathway_present=1)
        .pivot_table(
            index="genome_id",
            columns="PWY_NAME",
            values="pathway_present",
            aggfunc="max",
            fill_value=0,
        )
        .reindex(index=order, columns=top_pathways, fill_value=0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(top_pathways) * 0.35), max(6, len(order) * 0.25)))
    image = ax.imshow(heat_df.values, aspect="auto", cmap="Greys", vmin=0, vmax=max(1.0, float(np.nanmax(heat_df.values))))
    ax.set_xticks(np.arange(len(top_pathways)))
    ax.set_xticklabels(top_pathways, rotation=90)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Pathway")
    ax.set_ylabel("Genome")
    ax.set_title("Top pathways by genome (cell text = presence)")
    for row_index in range(len(order)):
        for col_index in range(len(top_pathways)):
            value = count_df.iat[row_index, col_index]
            ax.text(col_index, row_index, str(int(value)), ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Pathway score")
    fig.tight_layout()
    save_figure(fig, output_base)
    return True


def write_outputs(
    output_dir,
    prefix,
    genome_summary,
    pathway_long,
    pathway_orf_long,
    annotation_audit_long,
    pathway_audit_long,
    marker_audit_long=None,
    reference_mode_audit_long=None,
    marker_manifest=None,
    top_n_pathways=20,
    progress=True,
    heartbeat_seconds=60,
    experimental_mobility_screen=False,
    experimental_mobility_genome_type_tsv=None,
    experimental_mobility_genome_type_column=None,
    experimental_mobility_genome_type_id_column=None,
    experimental_mobility_include_broad_screen=False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / prefix
    if marker_audit_long is None:
        marker_audit_long = pd.DataFrame()
    if reference_mode_audit_long is None:
        reference_mode_audit_long = pd.DataFrame()

    annotation_source = build_annotation_source_table(genome_summary)
    annotation_quality = build_annotation_quality_table(genome_summary)
    marker_summary = build_marker_summary_table(genome_summary, marker_manifest=marker_manifest)
    marker_audit_collapsed = collapse_marker_audit(marker_audit_long)
    reference_mode_summary = build_reference_mode_summary_table(genome_summary)
    elemental_annotation = build_elemental_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
    elemental_mode_annotation = build_elemental_mode_summary_table(genome_summary, "annotation", "total_orfs", "orfs")
    elemental_mode_annotation_unique = build_elemental_mode_summary_table(
        genome_summary, "annotation", "total_orfs", "orfs", scope="unique"
    )
    elemental_mode_annotation_ambiguous = build_elemental_mode_summary_table(
        genome_summary, "annotation", "total_orfs", "orfs", scope="ambiguous"
    )
    elemental_pathway_support = build_elemental_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
    elemental_mode_pathway_support = build_elemental_mode_summary_table(genome_summary, "pathway_support", "pathway_support_orfs", "orfs")
    elemental_mode_pathway_support_unique = build_elemental_mode_summary_table(
        genome_summary, "pathway_support", "pathway_support_orfs", "orfs", scope="unique"
    )
    elemental_mode_pathway_support_ambiguous = build_elemental_mode_summary_table(
        genome_summary, "pathway_support", "pathway_support_orfs", "orfs", scope="ambiguous"
    )
    elemental_pathway = build_elemental_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
    elemental_mode_pathway = build_elemental_mode_summary_table(genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways")
    elemental_mode_pathway_unique = build_elemental_mode_summary_table(
        genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways", scope="unique"
    )
    elemental_mode_pathway_ambiguous = build_elemental_mode_summary_table(
        genome_summary, "pathway", "total_pathways", "count", assigned_unit_label="pathways", scope="ambiguous"
    )
    pathway_presence = build_pathway_matrix(
        pathway_long.assign(pathway_present=1),
        "pathway_present",
    )
    pathway_score = build_pathway_matrix(pathway_long, "PWY_SCORE")
    pathway_coverage = build_pathway_matrix(pathway_long, "reaction_coverage_fraction")
    annotation_category_definitions = build_annotation_category_definitions_table()
    taxonomy_label_table = genome_summary[
        [
            column
            for column in [
                "genome_id",
                "genome_display_label",
                "taxonomy_display_rank",
                "taxonomy_display_value",
                "taxonomy_display_status",
                "taxonomy_match_method",
            ]
            if column in genome_summary.columns
        ]
    ].copy()
    experimental_mobility_outputs = {}
    if experimental_mobility_screen:
        experimental_mobility_outputs = build_experimental_mobility_outputs(
            genome_summary,
            annotation_audit_long,
            genome_type_tsv=experimental_mobility_genome_type_tsv,
            genome_type_column=experimental_mobility_genome_type_column,
            genome_type_id_column=experimental_mobility_genome_type_id_column,
            include_optional_broad_screen=experimental_mobility_include_broad_screen,
        )

    def write_tables():
        genome_summary.to_csv(f"{base}_genome_summary.tsv", sep="\t", index=False)
        annotation_source.to_csv(f"{base}_annotation_summary.tsv", sep="\t", index=False)
        annotation_quality.to_csv(f"{base}_annotation_quality_summary.tsv", sep="\t", index=False)
        elemental_annotation.to_csv(f"{base}_elemental_annotation_summary.tsv", sep="\t", index=False)
        elemental_mode_annotation.to_csv(f"{base}_elemental_mode_annotation_summary.tsv", sep="\t", index=False)
        elemental_mode_annotation_unique.to_csv(f"{base}_elemental_mode_annotation_unique_summary.tsv", sep="\t", index=False)
        elemental_mode_annotation_ambiguous.to_csv(f"{base}_elemental_mode_annotation_ambiguous_summary.tsv", sep="\t", index=False)
        elemental_pathway_support.to_csv(f"{base}_elemental_pathway_support_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway_support.to_csv(f"{base}_elemental_mode_pathway_support_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway_support_unique.to_csv(f"{base}_elemental_mode_pathway_support_unique_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway_support_ambiguous.to_csv(f"{base}_elemental_mode_pathway_support_ambiguous_summary.tsv", sep="\t", index=False)
        elemental_pathway.to_csv(f"{base}_elemental_pathway_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway.to_csv(f"{base}_elemental_mode_pathway_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway_unique.to_csv(f"{base}_elemental_mode_pathway_unique_summary.tsv", sep="\t", index=False)
        elemental_mode_pathway_ambiguous.to_csv(f"{base}_elemental_mode_pathway_ambiguous_summary.tsv", sep="\t", index=False)
        reference_mode_summary.to_csv(f"{base}_reference_mode_summary.tsv", sep="\t", index=False)
        pathway_long.to_csv(f"{base}_pathway_long.tsv", sep="\t", index=False)
        pathway_orf_long.to_csv(f"{base}_pathway_orf_long.tsv", sep="\t", index=False)
        annotation_audit_long.to_csv(f"{base}_elemental_annotation_audit.tsv", sep="\t", index=False)
        pathway_audit_long.to_csv(f"{base}_elemental_pathway_audit.tsv", sep="\t", index=False)
        marker_summary.to_csv(f"{base}_marker_summary.tsv", sep="\t", index=False)
        marker_audit_collapsed.to_csv(f"{base}_marker_audit.tsv", sep="\t", index=False)
        marker_audit_long.to_csv(f"{base}_marker_audit_raw.tsv", sep="\t", index=False)
        reference_mode_audit_long.to_csv(f"{base}_reference_mode_audit.tsv", sep="\t", index=False)
        pathway_presence.to_csv(f"{base}_pathway_presence_matrix.tsv", sep="\t", index=False)
        pathway_score.to_csv(f"{base}_pathway_score_matrix.tsv", sep="\t", index=False)
        pathway_coverage.to_csv(f"{base}_pathway_coverage_matrix.tsv", sep="\t", index=False)
        annotation_category_definitions.to_csv(f"{base}_annotation_category_definitions.tsv", sep="\t", index=False)
        taxonomy_label_table.to_csv(f"{base}_taxonomy_labels.tsv", sep="\t", index=False)
        if experimental_mobility_outputs:
            experimental_mobility_outputs["genome_type_lookup"].to_csv(
                f"{base}_experimental_candidate_mobility_genome_type_lookup.tsv",
                sep="\t",
                index=False,
            )
            experimental_mobility_outputs["hit_table"].to_csv(
                f"{base}_experimental_candidate_mobility_hits.tsv",
                sep="\t",
                index=False,
            )
            experimental_mobility_outputs["per_genome_table"].to_csv(
                f"{base}_experimental_candidate_mobility_per_genome.tsv",
                sep="\t",
                index=False,
            )
            experimental_mobility_outputs["prevalence_table"].to_csv(
                f"{base}_experimental_candidate_mobility_prevalence.tsv",
                sep="\t",
                index=False,
            )
            if "optional_broad_hit_table" in experimental_mobility_outputs:
                experimental_mobility_outputs["optional_broad_hit_table"].to_csv(
                    f"{base}_experimental_candidate_mobility_optional_broad_hits.tsv",
                    sep="\t",
                    index=False,
                )
                experimental_mobility_outputs["optional_broad_per_genome_table"].to_csv(
                    f"{base}_experimental_candidate_mobility_optional_broad_per_genome.tsv",
                    sep="\t",
                    index=False,
                )
                experimental_mobility_outputs["optional_broad_prevalence_table"].to_csv(
                    f"{base}_experimental_candidate_mobility_optional_broad_prevalence.tsv",
                    sep="\t",
                    index=False,
                )

    run_with_heartbeat(
        "writing summary tables",
        write_tables,
        enabled=progress,
        heartbeat_seconds=heartbeat_seconds,
        emit_start=False,
        emit_done=True,
    )

    wrote_files = [
        f"{base}_genome_summary.tsv",
        f"{base}_annotation_summary.tsv",
        f"{base}_annotation_quality_summary.tsv",
        f"{base}_marker_summary.tsv",
        f"{base}_reference_mode_summary.tsv",
        f"{base}_elemental_annotation_summary.tsv",
        f"{base}_elemental_mode_annotation_summary.tsv",
        f"{base}_elemental_mode_annotation_unique_summary.tsv",
        f"{base}_elemental_mode_annotation_ambiguous_summary.tsv",
        f"{base}_elemental_pathway_support_summary.tsv",
        f"{base}_elemental_mode_pathway_support_summary.tsv",
        f"{base}_elemental_mode_pathway_support_unique_summary.tsv",
        f"{base}_elemental_mode_pathway_support_ambiguous_summary.tsv",
        f"{base}_elemental_pathway_summary.tsv",
        f"{base}_elemental_mode_pathway_summary.tsv",
        f"{base}_elemental_mode_pathway_unique_summary.tsv",
        f"{base}_elemental_mode_pathway_ambiguous_summary.tsv",
        f"{base}_pathway_long.tsv",
        f"{base}_pathway_orf_long.tsv",
        f"{base}_elemental_annotation_audit.tsv",
        f"{base}_elemental_pathway_audit.tsv",
        f"{base}_marker_audit.tsv",
        f"{base}_marker_audit_raw.tsv",
        f"{base}_reference_mode_audit.tsv",
        f"{base}_pathway_presence_matrix.tsv",
        f"{base}_pathway_score_matrix.tsv",
        f"{base}_pathway_coverage_matrix.tsv",
        f"{base}_annotation_category_definitions.tsv",
        f"{base}_taxonomy_labels.tsv",
    ]
    if experimental_mobility_outputs:
        wrote_files.extend(
            [
                f"{base}_experimental_candidate_mobility_genome_type_lookup.tsv",
                f"{base}_experimental_candidate_mobility_hits.tsv",
                f"{base}_experimental_candidate_mobility_per_genome.tsv",
                f"{base}_experimental_candidate_mobility_prevalence.tsv",
            ]
        )
        if "optional_broad_hit_table" in experimental_mobility_outputs:
            wrote_files.extend(
                [
                    f"{base}_experimental_candidate_mobility_optional_broad_hits.tsv",
                    f"{base}_experimental_candidate_mobility_optional_broad_per_genome.tsv",
                    f"{base}_experimental_candidate_mobility_optional_broad_prevalence.tsv",
                ]
            )
    plot_specs = [
        ("compact_summary", lambda: plot_compact_summary(genome_summary, f"{base}_compact_summary")),
        ("annotation_sources", lambda: plot_annotation_sources(genome_summary, f"{base}_annotation_sources")),
        ("annotation_quality", lambda: plot_annotation_quality(genome_summary, f"{base}_annotation_quality")),
        ("elemental_metabolism", lambda: plot_elemental_metabolism(genome_summary, f"{base}_elemental_metabolism")),
        ("elemental_modes", lambda: plot_elemental_modes(genome_summary, f"{base}_elemental_modes")),
        ("elemental_modes_unique", lambda: plot_elemental_modes(genome_summary, f"{base}_elemental_modes_unique", scope="unique")),
        ("elemental_modes_ambiguous", lambda: plot_elemental_modes(genome_summary, f"{base}_elemental_modes_ambiguous", scope="ambiguous")),
        ("marker_heatmap", lambda: plot_marker_heatmap(genome_summary, f"{base}_marker_heatmap", marker_manifest=marker_manifest, specificity="all")),
        ("marker_heatmap_specific", lambda: plot_marker_heatmap(genome_summary, f"{base}_marker_heatmap_specific", marker_manifest=marker_manifest, specificity="specific")),
        ("marker_heatmap_generic", lambda: plot_marker_heatmap(genome_summary, f"{base}_marker_heatmap_generic", marker_manifest=marker_manifest, specificity="generic")),
        ("reference_mode_heatmap", lambda: plot_reference_mode_heatmap(genome_summary, f"{base}_reference_mode_heatmap")),
        ("pathway_metrics", lambda: plot_pathway_metric_panels(genome_summary, f"{base}_pathway_metrics")),
        ("top_pathways", lambda: plot_top_pathway_heatmap(pathway_long, genome_summary, f"{base}_top_pathways", top_n_pathways)),
    ]
    if experimental_mobility_outputs:
        plot_specs.append(
            (
                "experimental_candidate_mobility_prevalence",
                lambda: plot_experimental_mobility_prevalence(
                    experimental_mobility_outputs["prevalence_table"],
                    f"{base}_experimental_candidate_mobility_prevalence",
                ),
            )
        )
    for index, (name, plotter) in enumerate(plot_specs, start=1):
        progress_log(
            f"[progress] rendering plot {index}/{len(plot_specs)}: {name}",
            enabled=progress,
        )
        run_with_heartbeat(
            f"rendering plot {index}/{len(plot_specs)}: {name}",
            plotter,
            enabled=progress,
            heartbeat_seconds=heartbeat_seconds,
            emit_start=False,
            emit_done=False,
        )
    wrote_files.extend(
        [
            f"{base}_compact_summary.png",
            f"{base}_compact_summary.pdf",
            f"{base}_annotation_sources.png",
            f"{base}_annotation_sources.pdf",
            f"{base}_annotation_quality.png",
            f"{base}_annotation_quality.pdf",
            f"{base}_elemental_metabolism.png",
            f"{base}_elemental_metabolism.pdf",
            f"{base}_elemental_modes.png",
            f"{base}_elemental_modes.pdf",
            f"{base}_elemental_modes_unique.png",
            f"{base}_elemental_modes_unique.pdf",
            f"{base}_elemental_modes_ambiguous.png",
            f"{base}_elemental_modes_ambiguous.pdf",
            f"{base}_marker_heatmap.png",
            f"{base}_marker_heatmap.pdf",
            f"{base}_marker_heatmap_specific.png",
            f"{base}_marker_heatmap_specific.pdf",
            f"{base}_marker_heatmap_generic.png",
            f"{base}_marker_heatmap_generic.pdf",
            f"{base}_reference_mode_heatmap.png",
            f"{base}_reference_mode_heatmap.pdf",
            f"{base}_pathway_metrics.png",
            f"{base}_pathway_metrics.pdf",
            f"{base}_top_pathways.png",
            f"{base}_top_pathways.pdf",
        ]
    )
    if experimental_mobility_outputs:
        wrote_files.extend(
            [
                f"{base}_experimental_candidate_mobility_prevalence.png",
                f"{base}_experimental_candidate_mobility_prevalence.pdf",
            ]
        )
    return wrote_files


def main():
    args = build_parser().parse_args()
    progress_enabled = not bool(args.quiet)
    run_start = time.time()
    if int(args.threads) > 0:
        worker_count = max(1, int(args.threads))
        prep_worker_count = worker_count
    else:
        worker_count = max(1, int(args.workers))
        prep_worker_count = max(1, int(args.prep_workers)) if int(args.prep_workers) > 0 else worker_count
    input_results_dirs = expand_results_dirs(args.results_dirs or [])
    if not args.reference_index_only:
        if not input_results_dirs:
            raise ValueError("At least one results directory is required unless --reference-index-only is used.")
        for results_dir in input_results_dirs:
            if not results_dir.exists():
                raise FileNotFoundError(f"Results directory not found: {results_dir}")

    output_dir = None
    if not args.reference_index_only:
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else default_output_dir_for_results(input_results_dirs)
        )
    prefix = sanitize_label(args.prefix)
    allowed_genomes = None
    if (not args.reference_index_only) and args.genome_filter_tsv:
        allowed_genomes, _ = load_allowed_genomes(
            Path(args.genome_filter_tsv).expanduser().resolve(),
            filter_id_column=args.filter_id_column,
            filter_tier_column=args.filter_tier_column,
            include_tiers=split_csv_arg(args.include_tiers),
        )
    taxonomy_label_lookup = None
    taxonomy_source = None
    if not args.reference_index_only:
        taxonomy_source = args.taxonomy_label_tsv or args.genome_filter_tsv
    if taxonomy_source:
        taxonomy_source_path = Path(taxonomy_source).expanduser().resolve()
        (
            taxonomy_label_lookup,
            selected_tax_id_column,
            ambiguous_taxonomy_aliases,
        ) = load_taxonomy_label_lookup(
            taxonomy_source_path,
            taxonomy_id_column=args.taxonomy_id_column,
        )
        progress_log(
            f"[done] loaded taxonomy labels: aliases={len(taxonomy_label_lookup):,} "
            f"from={taxonomy_source_path} id_column={selected_tax_id_column} "
            f"ambiguous_aliases_dropped={ambiguous_taxonomy_aliases:,}",
            enabled=progress_enabled,
        )
    elif not args.reference_index_only:
        progress_log(
            "[info] taxonomy label source not provided; plot labels will use raw genome IDs.",
            enabled=progress_enabled,
        )

    progress_log(
        f"[start] summarize_metapathways_genomes: workers={worker_count} "
        f"prep_workers={prep_worker_count} "
        f"threads_override={max(0, int(args.threads))} "
        f"progress_interval={max(1, int(args.progress_interval))}s "
        f"reference_chunk_size={max(1, int(args.reference_chunk_size)):,} "
        f"reference_progress_rows={max(1, int(args.reference_progress_rows)):,} "
        f"heartbeat_seconds={max(1, int(args.heartbeat_seconds))} "
        f"reference_force_full_index={bool(args.reference_force_full_index)} "
        f"reference_index_only={bool(args.reference_index_only)}",
        enabled=progress_enabled,
    )
    if not args.reference_index_only:
        progress_log(
            f"[start] input results directories: {len(input_results_dirs)}",
            enabled=progress_enabled,
        )

    if args.reference_index_only:
        run_with_heartbeat(
            "building/loading accession reference index cache",
            lambda: load_reference_accession_modes(
                reference_mappings_dir,
                progress=progress_enabled,
                progress_interval_rows=args.reference_progress_rows,
                chunk_size=args.reference_chunk_size,
                target_accessions=None,
                workers=prep_worker_count,
            ),
            enabled=progress_enabled,
            heartbeat_seconds=args.heartbeat_seconds,
            emit_start=True,
            emit_done=True,
        )
        elapsed = time.time() - run_start
        progress_log(
            f"[done] reference index-only mode completed in {elapsed/60:.2f}m",
            enabled=progress_enabled,
        )
        return

    marker_manifest_path = resolve_optional_path_arg(args.marker_manifest)
    if marker_manifest_path is None:
        marker_manifest = pd.DataFrame(columns=["family_id", "mode_id", "marker_id", "marker_label", "is_core", "alias"])
    else:
        marker_manifest = load_marker_manifest(marker_manifest_path)
    reference_mappings_dir = resolve_optional_path_arg(args.reference_mappings_dir)
    if len(input_results_dirs) == 1:
        (
            genome_summary,
            pathway_long,
            pathway_orf_long,
            annotation_audit_long,
            pathway_audit_long,
            marker_audit_long,
            reference_mode_audit_long,
        ) = build_summary_tables(
            results_dir=input_results_dirs[0],
            high_conf_threshold=args.high_confidence_threshold,
            allowed_genomes=allowed_genomes,
            taxonomy_label_lookup=taxonomy_label_lookup,
            marker_manifest=marker_manifest,
            reference_mappings_dir=reference_mappings_dir,
            workers=worker_count,
            progress=progress_enabled,
            progress_interval=args.progress_interval,
            reference_chunk_size=args.reference_chunk_size,
            reference_progress_rows=args.reference_progress_rows,
            heartbeat_seconds=args.heartbeat_seconds,
            reference_force_full_index=args.reference_force_full_index,
            prep_workers=prep_worker_count,
        )
    else:
        genome_parts = []
        pathway_parts = []
        pathway_orf_parts = []
        annotation_audit_parts = []
        pathway_audit_parts = []
        marker_audit_parts = []
        reference_mode_audit_parts = []
        total_dirs = len(input_results_dirs)
        skipped_dirs = []
        for index, results_dir in enumerate(input_results_dirs, start=1):
            progress_log(
                f"[start] ({index}/{total_dirs}) summarizing results directory: {results_dir}",
                enabled=progress_enabled,
            )
            try:
                (
                    part_genome_summary,
                    part_pathway_long,
                    part_pathway_orf_long,
                    part_annotation_audit_long,
                    part_pathway_audit_long,
                    part_marker_audit_long,
                    part_reference_mode_audit_long,
                ) = build_summary_tables(
                    results_dir=results_dir,
                    high_conf_threshold=args.high_confidence_threshold,
                    allowed_genomes=allowed_genomes,
                    taxonomy_label_lookup=taxonomy_label_lookup,
                    marker_manifest=marker_manifest,
                    reference_mappings_dir=reference_mappings_dir,
                    workers=worker_count,
                    progress=progress_enabled,
                    progress_interval=args.progress_interval,
                    reference_chunk_size=args.reference_chunk_size,
                    reference_progress_rows=args.reference_progress_rows,
                    heartbeat_seconds=args.heartbeat_seconds,
                    reference_force_full_index=args.reference_force_full_index,
                    prep_workers=prep_worker_count,
                )
            except ValueError as exc:
                message = str(exc)
                if "No MetaPathways genome records remain after applying --genome-filter-tsv." in message:
                    skipped_dirs.append(str(results_dir))
                    progress_log(
                        f"[warn] ({index}/{total_dirs}) skipped after filter: {results_dir}",
                        enabled=progress_enabled,
                    )
                    continue
                raise
            if args.also_write_individual_outputs:
                individual_output_dir = results_dir / args.individual_output_subdir
                progress_log(
                    f"[start] ({index}/{total_dirs}) writing individual output set: {individual_output_dir}",
                    enabled=progress_enabled,
                )
                _ = write_outputs(
                    individual_output_dir,
                    prefix,
                    part_genome_summary,
                    part_pathway_long,
                    part_pathway_orf_long,
                    part_annotation_audit_long,
                    part_pathway_audit_long,
                    part_marker_audit_long,
                    reference_mode_audit_long=part_reference_mode_audit_long,
                    marker_manifest=marker_manifest,
                    top_n_pathways=args.top_n_pathways,
                    progress=progress_enabled,
                    heartbeat_seconds=args.heartbeat_seconds,
                    experimental_mobility_screen=bool(args.experimental_mobility_screen),
                    experimental_mobility_genome_type_tsv=args.experimental_mobility_genome_type_tsv,
                    experimental_mobility_genome_type_column=args.experimental_mobility_genome_type_column,
                    experimental_mobility_genome_type_id_column=args.experimental_mobility_genome_type_id_column,
                    experimental_mobility_include_broad_screen=bool(args.experimental_mobility_include_broad_screen),
                )
                progress_log(
                    f"[done] ({index}/{total_dirs}) individual outputs in: {individual_output_dir}",
                    enabled=progress_enabled,
                )
            source_value = str(results_dir)
            for frame in [
                part_genome_summary,
                part_pathway_long,
                part_pathway_orf_long,
                part_annotation_audit_long,
                part_pathway_audit_long,
                part_marker_audit_long,
                part_reference_mode_audit_long,
            ]:
                if not frame.empty:
                    frame["results_root_dir"] = source_value
            genome_parts.append(part_genome_summary)
            pathway_parts.append(part_pathway_long)
            pathway_orf_parts.append(part_pathway_orf_long)
            annotation_audit_parts.append(part_annotation_audit_long)
            pathway_audit_parts.append(part_pathway_audit_long)
            marker_audit_parts.append(part_marker_audit_long)
            reference_mode_audit_parts.append(part_reference_mode_audit_long)
            progress_log(
                f"[done] ({index}/{total_dirs}) genomes={len(part_genome_summary):,} from {results_dir}",
                enabled=progress_enabled,
            )

        genome_summary = pd.concat(genome_parts, ignore_index=True) if genome_parts else pd.DataFrame()
        pathway_long = pd.concat(pathway_parts, ignore_index=True) if pathway_parts else pd.DataFrame()
        pathway_orf_long = pd.concat(pathway_orf_parts, ignore_index=True) if pathway_orf_parts else pd.DataFrame()
        annotation_audit_long = pd.concat(annotation_audit_parts, ignore_index=True) if annotation_audit_parts else pd.DataFrame()
        pathway_audit_long = pd.concat(pathway_audit_parts, ignore_index=True) if pathway_audit_parts else pd.DataFrame()
        marker_audit_long = pd.concat(marker_audit_parts, ignore_index=True) if marker_audit_parts else pd.DataFrame()
        reference_mode_audit_long = (
            pd.concat(reference_mode_audit_parts, ignore_index=True)
            if reference_mode_audit_parts
            else pd.DataFrame()
        )
        if not genome_summary.empty:
            sort_columns = [column for column in ["total_pathways", "annotated_orfs", "total_orfs", "genome_id"] if column in genome_summary.columns]
            if sort_columns:
                ascending = [False if column in {"total_pathways", "annotated_orfs", "total_orfs"} else True for column in sort_columns]
                genome_summary = genome_summary.sort_values(by=sort_columns, ascending=ascending).reset_index(drop=True)
        if skipped_dirs:
            progress_log(
                f"[warn] skipped {len(skipped_dirs)} input directories after tier/filter selection",
                enabled=progress_enabled,
            )
        if genome_summary.empty:
            raise ValueError(
                "No genomes remained across all input directories after applying --genome-filter-tsv / --include-tiers."
            )
        progress_log(
            f"[done] combined multi-input summary: genomes={len(genome_summary):,}, "
            f"pathways={len(pathway_long):,}, pathway_orfs={len(pathway_orf_long):,}",
            enabled=progress_enabled,
        )
    progress_log("[start] writing output tables and figures", enabled=progress_enabled)
    wrote_files = write_outputs(
        output_dir,
        prefix,
        genome_summary,
        pathway_long,
        pathway_orf_long,
        annotation_audit_long,
        pathway_audit_long,
        marker_audit_long,
        reference_mode_audit_long=reference_mode_audit_long,
        marker_manifest=marker_manifest,
        top_n_pathways=args.top_n_pathways,
        progress=progress_enabled,
        heartbeat_seconds=args.heartbeat_seconds,
        experimental_mobility_screen=bool(args.experimental_mobility_screen),
        experimental_mobility_genome_type_tsv=args.experimental_mobility_genome_type_tsv,
        experimental_mobility_genome_type_column=args.experimental_mobility_genome_type_column,
        experimental_mobility_genome_type_id_column=args.experimental_mobility_genome_type_id_column,
        experimental_mobility_include_broad_screen=bool(args.experimental_mobility_include_broad_screen),
    )
    elapsed = time.time() - run_start
    progress_log(
            f"[done] summarize_metapathways_genomes completed in {elapsed/60:.2f}m; files={len(wrote_files)}",
            enabled=progress_enabled,
        )
    for path in wrote_files:
        print(path)


if __name__ == "__main__":
    main()
