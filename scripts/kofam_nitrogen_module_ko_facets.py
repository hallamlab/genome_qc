#!/usr/bin/env python3
"""Plot nitrogen-cycle module KO prevalence differences from KOfam enrichment tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import summarize_metapathways_genomes as mp_genomes  # noqa: E402
from summarize_metapathways_genomes import ensure_plotting, save_figure  # noqa: E402


DEFAULT_RESULTS_DIR = Path(
    "/home/ryan/SABer_dat/SI_data/SI_METAGs/SABer_260109/genome_qc_results"
)
ELEMENTAL_CYCLE_MODULE_IDS = {
    "nitrogen": [
        "M00175", "M00528", "M00804", "M00529", "M00530", "M00531",
        "M00615", "M00973", "M00029", "M00546", "M00978",
    ],
    "sulfur": [
        "M00017", "M00021", "M00034", "M00035", "M00058", "M00059",
        "M00067", "M00076", "M00077", "M00078", "M00079", "M00176",
        "M00338", "M00368", "M00595", "M00596", "M00609", "M00616",
        "M00953", "M00984", "M00985", "M00986", "M00987", "M00990",
        "M00991", "M00992", "M00993", "M01048", "M01049", "M01050",
        "M01051", "M01052",
    ],
    "carbon": [
        "M00002", "M00010", "M00011", "M00012", "M00013", "M00032",
        "M00036", "M00088", "M00165", "M00173", "M00174", "M00307",
        "M00344", "M00345", "M00346", "M00356", "M00357", "M00374",
        "M00375", "M00376", "M00377", "M00422", "M00545", "M00563",
        "M00567", "M00569", "M00579", "M00620", "M00777", "M00878",
        "M00957", "M00990", "M00991", "M00992",
    ],
    "other": [
        "M00004", "M00006", "M00007", "M00124", "M00130", "M00131",
        "M00132", "M00143", "M00146", "M00147", "M00148", "M00149",
        "M00580", "M00597", "M00598", "M00916", "M00968",
    ],
    "phosphorus": [],
    "iron": [],
    "trace_metals": [],
    "mobile_genetic_elements": [],
}
CYCLE_LABELS = {
    "nitrogen": "Nitrogen metabolism",
    "sulfur": "Sulfur metabolism",
    "carbon": "Carbon metabolism",
    "other": "Other elemental metabolism",
    "phosphorus": "Phosphorus metabolism",
    "iron": "Iron metabolism",
    "trace_metals": "Trace metal metabolism",
    "mobile_genetic_elements": "Mobile genetic elements",
}
COMPARISONS = {
    "sag_xpg": ("SAGs", "xPG_SAGs", "SAG vs SAG-xPG"),
    "mag_xpg": ("MAGs", "xPG_MAGs", "MAG vs MAG-xPG"),
    "sag_xpg_mag": ("SAGs", "xPG_MAGs", "SAG vs MAG-xPG"),
    "mag_xpg_sag": ("MAGs", "xPG_SAGs", "MAG vs SAG-xPG"),
    "mag_sag": ("SAGs", "MAGs", "SAG vs MAG"),
    "xpg_mag_xpg_sag": ("xPG_MAGs", "xPG_SAGs", "MAG-xPG vs SAG-xPG"),
}
MIN_PLOT_FONT_SIZE = 22
PANEL_LABEL_FONT_SIZE = 28
MAX_KOS_PER_FACET = 14
MAX_FACETS_PER_COMPARISON_FIG = 12
MGE_CLASS_PATTERNS = [
    (
        "MGE01",
        "Phage and viral elements",
        re.compile(
            r"\b(phage|prophage|bacteriophage|viral|virus|virion|capsid|terminase|portal protein|"
            r"baseplate|tail fiber|tail sheath|tail tube|head-tail|holin|endolysin|excisionase)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MGE02",
        "Transposases and insertion sequences",
        re.compile(r"\b(transposase|transposition|insertion sequence|IS[0-9A-Z_-]*|resolvase)\b", re.IGNORECASE),
    ),
    (
        "MGE03",
        "Integrases and recombinases",
        re.compile(
            r"\b(integrase|recombinase|site-specific recombination|site-specific recombinase|"
            r"tyrosine recombinase|serine recombinase|xerC|xerD)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MGE04",
        "Plasmid replication and partition",
        re.compile(
            r"\b(plasmid|partition protein|partitioning protein|parA|parB|replication initiation protein|"
            r"plasmid stabilization|addiction module)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MGE05",
        "Conjugation and transfer",
        re.compile(
            r"\b(conjugation|conjugative|conjugal|relaxase|mobilization protein|mob[A-Z]?|"
            r"type IV secretion|type 4 secretion|T4SS|tra[A-Z]|trb[A-Z])\b",
            re.IGNORECASE,
        ),
    ),
    (
        "MGE06",
        "Toxin-antitoxin systems",
        re.compile(r"\b(toxin-antitoxin|antitoxin|toxin component|pemK|pemI|relE|relB|mazF|mazE)\b", re.IGNORECASE),
    ),
]
MGE_DEFINITION_SEARCH_REGEX = (
    "phage|prophage|bacteriophage|viral|virus|virion|capsid|terminase|portal protein|"
    "baseplate|tail fiber|tail sheath|tail tube|head-tail|holin|endolysin|excisionase|"
    "transposase|transposition|insertion sequence|resolvase|integrase|recombinase|"
    "site-specific recombination|plasmid|partition protein|partitioning protein|"
    "replication initiation protein|conjugation|conjugative|conjugal|relaxase|"
    "mobilization protein|type IV secretion|type 4 secretion|T4SS|toxin-antitoxin|"
    "antitoxin|addiction module"
)
KEYWORD_CYCLE_PATTERNS = {
    "phosphorus": [
        (
            "P01",
            "Phosphate transport and regulation",
            re.compile(
                r"\b(phosphate[- ](?:specific )?(?:transport|transporter|ABC transporter|permease|binding protein)|"
                r"phosphate starvation|phosphate regul|Pho[BRU]|Pst[ABCS])\b",
                re.IGNORECASE,
            ),
        ),
        (
            "P02",
            "Phosphonate and phosphite utilization",
            re.compile(r"\b(phosphonate|phosphite|C-P lyase|phn[A-Z])\b", re.IGNORECASE),
        ),
        (
            "P03",
            "Polyphosphate metabolism",
            re.compile(r"\b(polyphosphate|exopolyphosphatase|polyphosphate kinase)\b", re.IGNORECASE),
        ),
        (
            "P04",
            "Organic phosphorus hydrolysis",
            re.compile(r"\b(alkaline phosphatase|acid phosphatase|phosphodiesterase|phosphotriesterase|phytase)\b", re.IGNORECASE),
        ),
    ],
    "iron": [
        (
            "FE01",
            "Iron and heme transport",
            re.compile(r"\b(ferric|ferrous|iron[- ](?:complex )?(?:transport|transporter|uptake)|heme transport|hemin transport)\b", re.IGNORECASE),
        ),
        (
            "FE02",
            "Siderophore uptake and biosynthesis",
            re.compile(r"\b(siderophore|enterobactin|aerobactin|ferrichrome|pyoverdine|yersiniabactin)\b", re.IGNORECASE),
        ),
        (
            "FE03",
            "Iron storage and stress",
            re.compile(r"\b(ferritin|bacterioferritin|rubrerythrin|rubredoxin|Dps protein|iron storage)\b", re.IGNORECASE),
        ),
        (
            "FE04",
            "Iron-sulfur cluster assembly",
            re.compile(r"\b(iron-sulfur|Fe-S cluster|suf[ABCDSE]|isc[USAH]|nif[US])\b", re.IGNORECASE),
        ),
        (
            "FE05",
            "Heme and siroheme metabolism",
            re.compile(r"\b(heme|siroheme|coproporphyrin|uroporphyrinogen|protoporphyrin)\b", re.IGNORECASE),
        ),
    ],
    "trace_metals": [
        (
            "TM01",
            "Copper transport and resistance",
            re.compile(r"\b(copper|cuprous|cupri|Cop[ABCD]|Cus[ABCFRS])\b", re.IGNORECASE),
        ),
        (
            "TM02",
            "Nickel and cobalt metabolism",
            re.compile(r"\b(nickel|cobalt|NiCoT|urease accessory|hydrogenase nickel|cobalamin)\b", re.IGNORECASE),
        ),
        (
            "TM03",
            "Molybdenum and tungsten cofactors",
            re.compile(r"\b(molybdenum|molybdopterin|tungsten|tungstate|molybdate)\b", re.IGNORECASE),
        ),
        (
            "TM04",
            "Manganese and zinc transport",
            re.compile(r"\b(manganese|zinc|Mn2|Zn2|Znu[ABC]|Mnt[ABC]|manganous)\b", re.IGNORECASE),
        ),
        (
            "TM05",
            "Toxic metal resistance",
            re.compile(r"\b(arsenic|arsenate|arsenite|mercury|mercuric|cadmium|chromate|tellurite|silver resistance|Czc[ABC])\b", re.IGNORECASE),
        ),
    ],
}
KEYWORD_CYCLE_SEARCH_REGEX = {
    cycle: "|".join(pattern.pattern for _class_id, _class_name, pattern in patterns)
    for cycle, patterns in KEYWORD_CYCLE_PATTERNS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--cycle",
        choices=sorted(ELEMENTAL_CYCLE_MODULE_IDS),
        default="nitrogen",
    )
    parser.add_argument(
        "--taxon-filter",
        choices=["all", "gammaproteobacteria", "sup05"],
        default="all",
        help="Restrict prevalence calculations to a taxonomic subset.",
    )
    parser.add_argument("--q-threshold", type=float, default=0.05)
    parser.add_argument(
        "--delta-limit",
        type=float,
        default=100.0,
        help="Symmetric x-axis limit in prevalence percentage points.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def wrap_label(value: object, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False)) or str(value)


def apply_minimum_font_size(fig, minimum: float = MIN_PLOT_FONT_SIZE) -> None:
    for text in fig.findobj(match=lambda artist: hasattr(artist, "get_fontsize")):
        try:
            if float(text.get_fontsize()) < minimum:
                text.set_fontsize(minimum)
        except Exception:
            continue


def save_nitrogen_figure(fig, output_base: Path | str) -> None:
    mp_genomes.PLOT_FONT_SIZES["panel_label"] = max(
        PANEL_LABEL_FONT_SIZE,
        mp_genomes.PLOT_FONT_SIZES.get("panel_label", 0),
    )
    apply_minimum_font_size(fig)
    save_figure(fig, str(output_base))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return q_values
    n_tests = valid.shape[0]
    adjusted = valid * n_tests / np.arange(1, n_tests + 1, dtype=float)
    adjusted = adjusted.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    q_values.loc[adjusted.index] = adjusted
    return q_values


def fisher_exact_p_value(a_present: int, a_absent: int, b_present: int, b_absent: int) -> float:
    try:
        from scipy.stats import fisher_exact
    except Exception:
        return np.nan
    _odds, p_value = fisher_exact([[a_present, a_absent], [b_present, b_absent]], alternative="two-sided")
    return float(p_value)


def prepare_cycle_modules(results_dir: Path, module_path: Path, cycle: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    extra_ko_names: dict[str, str] = {}
    if cycle == "mobile_genetic_elements":
        definition_hits, extra_ko_names = load_mge_definition_hits(results_dir / "metadata.tsv")
        cycle_modules = definition_hits
    elif cycle in KEYWORD_CYCLE_PATTERNS:
        definition_hits, extra_ko_names = load_keyword_cycle_definition_hits(cycle, results_dir / "metadata.tsv")
        cycle_modules = definition_hits
    else:
        cycle_modules = None
    if cycle == "mobile_genetic_elements" or cycle in KEYWORD_CYCLE_PATTERNS:
        enrichment_path = results_dir / "kofam_ko.enrichment_results.compact.tsv"
        if enrichment_path.is_file():
            enrichment = read_tsv(enrichment_path)
            if "item" in enrichment.columns:
                observed_kos = set(enrichment["item"].astype(str))
                cycle_modules = cycle_modules.loc[cycle_modules["ko"].astype(str).isin(observed_kos)].copy()
        if cycle_modules.empty:
            raise ValueError(f"No selected {cycle} KOs were found from KOfam definitions or BRITE annotations")
        cycle_modules["ko"] = cycle_modules["ko"].astype(str)
        cycle_modules["module_id"] = cycle_modules["module_id"].astype(str)
        cycle_modules["module_name"] = cycle_modules["module_name"].astype(str)
        class_patterns = MGE_CLASS_PATTERNS if cycle == "mobile_genetic_elements" else KEYWORD_CYCLE_PATTERNS[cycle]
        class_order = {class_id: index for index, (class_id, _name, _pattern) in enumerate(class_patterns)}
        cycle_modules["module_order"] = cycle_modules["module_id"].map(class_order)
        cycle_modules = cycle_modules.sort_values(["module_order", "module_id", "ko"]).drop_duplicates(
            ["module_id", "ko"], keep="first"
        )
        cycle_modules["ko_order"] = cycle_modules.groupby("module_id").cumcount()
        cycle_modules = split_large_facets(cycle_modules)
        module_summary = (
            cycle_modules.sort_values(["module_order", "module_name"])
            .groupby(
                ["module_order", "module_id", "module_name", "source_module_id", "source_module_name"],
                as_index=False,
            )
            .agg(n_kos=("ko", "nunique"))
            .sort_values("module_order")
        )
        return cycle_modules, module_summary, extra_ko_names

    ko_to_module = read_tsv(module_path)
    required = {"ko", "module_id", "module_name"}
    missing = required - set(ko_to_module.columns)
    if missing:
        raise ValueError(f"{module_path} is missing columns: {', '.join(sorted(missing))}")

    module_ids = ELEMENTAL_CYCLE_MODULE_IDS[cycle]
    cycle_modules = ko_to_module.loc[
        ko_to_module["module_id"].astype(str).isin(module_ids),
        ["ko", "module_id", "module_name"],
    ].copy()
    if cycle_modules.empty:
        raise ValueError(f"No selected {cycle} modules were found in {module_path}")

    cycle_modules["module_id"] = cycle_modules["module_id"].astype(str)
    cycle_modules["ko"] = cycle_modules["ko"].astype(str)
    cycle_modules["module_order"] = cycle_modules["module_id"].map(
        {module_id: index for index, module_id in enumerate(module_ids)}
    )
    cycle_modules["ko_order"] = np.arange(len(cycle_modules), dtype=int)
    cycle_modules = cycle_modules.sort_values(["module_order", "ko_order"]).drop_duplicates(
        ["module_id", "ko"], keep="first"
    )
    cycle_modules = split_large_facets(cycle_modules)

    module_summary = (
        cycle_modules.sort_values(["module_order", "module_name"])
        .groupby(
            ["module_order", "module_id", "module_name", "source_module_id", "source_module_name"],
            as_index=False,
        )
        .agg(n_kos=("ko", "nunique"))
        .sort_values("module_order")
    )
    return cycle_modules, module_summary, extra_ko_names


def clean_ko_definition(value: object) -> str:
    text = str(value).strip().strip('"')
    text = re.sub(r"\s*\[EC:[^\]]+\]\s*", "", text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_ko_definition_map(
    metadata_path: Path,
    selected_kos: set[str],
) -> dict[str, str]:
    if not metadata_path.is_file():
        return {}
    metadata = read_tsv(metadata_path)
    if "passed_path" not in metadata.columns:
        return {}
    ko_names: dict[str, str] = {}
    for path_text in metadata["passed_path"].dropna().astype(str):
        if selected_kos <= set(ko_names):
            break
        path = Path(path_text).expanduser()
        if not path.is_file():
            continue
        try:
            with path.open("rt", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                for row in reader:
                    if len(row) < 7:
                        continue
                    ko = row[2].strip()
                    if ko not in selected_kos or ko in ko_names:
                        continue
                    definition = clean_ko_definition(row[6])
                    if definition:
                        ko_names[ko] = definition
        except OSError:
            continue
    return ko_names


def load_ko_definition_map_rg(metadata_path: Path, selected_kos: set[str]) -> dict[str, str]:
    ko_names: dict[str, str] = {}
    if not selected_kos:
        return ko_names
    results_dir = metadata_path.parent
    selected = sorted(selected_kos)
    chunk_size = 80
    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        pattern = "|".join(re.escape(ko) for ko in chunk)
        command = [
            "rg",
            "--no-heading",
            "--with-filename",
            "--glob",
            "*.kofamscan.passed.tsv",
            pattern,
            str(results_dir),
        ]
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError:
            continue
        if completed.returncode not in {0, 1}:
            continue
        for output_line in completed.stdout.splitlines():
            if ":" not in output_line:
                continue
            _path, line = output_line.split(":", 1)
            row = line.split("\t")
            if len(row) < 7:
                continue
            ko = row[2].strip()
            if ko not in selected_kos or ko in ko_names:
                continue
            definition = clean_ko_definition(row[6])
            if definition:
                ko_names[ko] = definition
        if selected_kos <= set(ko_names):
            break
    return ko_names


def classify_mge_text(text: object) -> list[tuple[str, str]]:
    value = str(text)
    matches = []
    for class_id, class_name, pattern in MGE_CLASS_PATTERNS:
        if pattern.search(value):
            matches.append((class_id, class_name))
    return matches


def load_mge_definition_hits(metadata_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    ko_names: dict[str, str] = {}
    seen = set()
    results_dir = metadata_path.parent
    command = [
        "rg",
        "--ignore-case",
        "--no-heading",
        "--with-filename",
        "--glob",
        "*.kofamscan.passed.tsv",
        MGE_DEFINITION_SEARCH_REGEX,
        str(results_dir),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        completed = subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    if completed.returncode not in {0, 1}:
        completed = subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    for output_line in completed.stdout.splitlines():
        if ":" not in output_line:
            continue
        _path, line = output_line.split(":", 1)
        row = line.split("\t")
        if len(row) < 7:
            continue
        ko = row[2].strip()
        if not re.fullmatch(r"K[0-9]+", ko):
            continue
        definition = clean_ko_definition(row[6])
        if definition and ko not in ko_names:
            ko_names[ko] = definition
        for class_id, class_name in classify_mge_text(definition):
            key = (ko, class_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"ko": ko, "module_id": class_id, "module_name": class_name})
    return pd.DataFrame(rows), ko_names


def classify_keyword_cycle_text(cycle: str, text: object) -> list[tuple[str, str]]:
    value = str(text)
    matches = []
    for class_id, class_name, pattern in KEYWORD_CYCLE_PATTERNS[cycle]:
        if pattern.search(value):
            matches.append((class_id, class_name))
    return matches


def load_keyword_cycle_definition_hits(cycle: str, metadata_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    ko_names: dict[str, str] = {}
    seen = set()
    results_dir = metadata_path.parent
    command = [
        "rg",
        "--ignore-case",
        "--no-heading",
        "--with-filename",
        "--glob",
        "*.kofamscan.passed.tsv",
        KEYWORD_CYCLE_SEARCH_REGEX[cycle],
        str(results_dir),
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        completed = subprocess.CompletedProcess(command, 1, stdout="", stderr="")
    if completed.returncode not in {0, 1}:
        completed = subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    for output_line in completed.stdout.splitlines():
        if ":" not in output_line:
            continue
        _path, line = output_line.split(":", 1)
        row = line.split("\t")
        if len(row) < 7:
            continue
        ko = row[2].strip()
        if not re.fullmatch(r"K[0-9]+", ko):
            continue
        definition = clean_ko_definition(row[6])
        if definition and ko not in ko_names:
            ko_names[ko] = definition
        for class_id, class_name in classify_keyword_cycle_text(cycle, definition):
            key = (ko, class_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"ko": ko, "module_id": class_id, "module_name": class_name})
    return pd.DataFrame(rows), ko_names


def load_mge_brite_hits(brite_path: Path) -> pd.DataFrame:
    if not brite_path.is_file():
        return pd.DataFrame(columns=["ko", "module_id", "module_name"])
    brite = read_tsv(brite_path)
    if "ko" not in brite.columns:
        return pd.DataFrame(columns=["ko", "module_id", "module_name"])
    rows = []
    seen = set()
    for record in brite.to_dict("records"):
        ko = str(record.get("ko", "")).strip()
        if not re.fullmatch(r"K[0-9]+", ko):
            continue
        text = " ".join(str(record.get(column, "")) for column in brite.columns)
        for class_id, class_name in classify_mge_text(text):
            key = (ko, class_id)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"ko": ko, "module_id": class_id, "module_name": class_name})
    return pd.DataFrame(rows)


def split_large_facets(cycle_modules: pd.DataFrame, max_kos: int = MAX_KOS_PER_FACET) -> pd.DataFrame:
    modules = cycle_modules.copy()
    modules["source_module_id"] = modules["module_id"].astype(str)
    modules["source_module_name"] = modules["module_name"].astype(str)
    split_rows = []
    module_order = 0
    for (_source_id, _source_name), group in modules.groupby(
        ["source_module_id", "source_module_name"], sort=False
    ):
        group = group.sort_values("ko_order").copy()
        n_chunks = int(math.ceil(group.shape[0] / float(max_kos))) if group.shape[0] else 1
        base_size = group.shape[0] // n_chunks
        remainder = group.shape[0] % n_chunks
        start = 0
        for chunk_index in range(n_chunks):
            chunk_size = base_size + (1 if chunk_index < remainder else 0)
            chunk = group.iloc[start : start + chunk_size].copy()
            start += chunk_size
            if n_chunks > 1:
                chunk["module_id"] = chunk["source_module_id"] + f".{chunk_index + 1}"
                chunk["module_name"] = (
                    chunk["source_module_name"] + f" ({chunk_index + 1}/{n_chunks})"
                )
            chunk["module_order"] = module_order
            chunk["ko_order"] = np.arange(chunk.shape[0], dtype=int)
            split_rows.append(chunk)
            module_order += 1
    return pd.concat(split_rows, ignore_index=True) if split_rows else modules


def attach_ko_labels(cycle_modules: pd.DataFrame, ko_names: dict[str, str]) -> pd.DataFrame:
    modules = cycle_modules.copy()
    modules["ko_name"] = modules["ko"].map(ko_names).fillna("")
    modules["ko_label"] = np.where(
        modules["ko_name"].astype(str).str.len().gt(0),
        modules["ko_name"].astype(str) + " (" + modules["ko"].astype(str) + ")",
        modules["ko"].astype(str),
    )
    return modules


def build_plot_table(
    nitrogen_modules: pd.DataFrame,
    enrichment_path: Path,
    q_threshold: float,
) -> pd.DataFrame:
    enrichment = read_tsv(enrichment_path)
    required = {
        "item",
        "adjusted_q_value",
        "associated_groups",
        "p_SAGs",
        "N_SAGs",
        "p_xPG_SAGs",
        "N_xPG_SAGs",
        "p_MAGs",
        "N_MAGs",
        "p_xPG_MAGs",
        "N_xPG_MAGs",
    }
    missing = required - set(enrichment.columns)
    if missing:
        raise ValueError(f"{enrichment_path} is missing columns: {', '.join(sorted(missing))}")

    keep_columns = sorted(required)
    enrichment = enrichment.loc[:, keep_columns].rename(columns={"item": "ko"})
    enrichment["ko"] = enrichment["ko"].astype(str)
    for column in enrichment.columns:
        if column.startswith(("p_", "N_")) or column == "adjusted_q_value":
            enrichment[column] = pd.to_numeric(enrichment[column], errors="coerce")
    category_n = {
        column: float(enrichment[column].dropna().max())
        for column in enrichment.columns
        if column.startswith("N_") and enrichment[column].dropna().size
    }

    joined = nitrogen_modules.merge(enrichment, on="ko", how="left")
    joined["adjusted_q_value"] = pd.to_numeric(joined["adjusted_q_value"], errors="coerce")
    joined["significant_q_le_threshold"] = joined["adjusted_q_value"].le(q_threshold).fillna(False)
    joined["associated_groups"] = joined["associated_groups"].fillna("")

    for column in ["p_SAGs", "p_xPG_SAGs", "p_MAGs", "p_xPG_MAGs"]:
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(0.0)
        joined[f"{column}_percent"] = joined[column] * 100.0
    for column, value in category_n.items():
        joined[column] = pd.to_numeric(joined[column], errors="coerce").fillna(value)

    rows = []
    for comparison_id, (base_category, xpg_category, comparison_label) in COMPARISONS.items():
        base_column = f"p_{base_category}"
        xpg_column = f"p_{xpg_category}"
        records = joined.copy()
        records["comparison_id"] = comparison_id
        records["comparison_label"] = comparison_label
        records["base_category"] = base_category
        records["xpg_category"] = xpg_category
        records["base_prevalence_percent"] = records[base_column] * 100.0
        records["xpg_prevalence_percent"] = records[xpg_column] * 100.0
        records["delta_prevalence_percent_points"] = (
            records["xpg_prevalence_percent"] - records["base_prevalence_percent"]
        )
        records["base_n_genomes"] = pd.to_numeric(records[f"N_{base_category}"], errors="coerce")
        records["xpg_n_genomes"] = pd.to_numeric(records[f"N_{xpg_category}"], errors="coerce")
        records["delta_se_percent_points"] = (
            np.sqrt(
                (records[xpg_column] * (1.0 - records[xpg_column]) / records["xpg_n_genomes"])
                + (records[base_column] * (1.0 - records[base_column]) / records["base_n_genomes"])
            )
            * 100.0
        )
        records["delta_ci95_low"] = (
            records["delta_prevalence_percent_points"] - 1.96 * records["delta_se_percent_points"]
        ).clip(lower=-100.0, upper=100.0)
        records["delta_ci95_high"] = (
            records["delta_prevalence_percent_points"] + 1.96 * records["delta_se_percent_points"]
        ).clip(lower=-100.0, upper=100.0)
        records["present_in_one_category_only"] = records[base_column].eq(0.0) ^ records[xpg_column].eq(0.0)
        rows.append(records)

    plot_table = pd.concat(rows, ignore_index=True)
    output_columns = [
        "comparison_id",
        "comparison_label",
        "base_category",
        "xpg_category",
        "module_id",
        "module_name",
        "source_module_id",
        "source_module_name",
        "ko",
        "ko_name",
        "ko_label",
        "base_prevalence_percent",
        "xpg_prevalence_percent",
        "base_n_genomes",
        "xpg_n_genomes",
        "delta_prevalence_percent_points",
        "delta_se_percent_points",
        "delta_ci95_low",
        "delta_ci95_high",
        "adjusted_q_value",
        "significant_q_le_threshold",
        "present_in_one_category_only",
        "associated_groups",
        "module_order",
        "ko_order",
    ]
    return plot_table.loc[:, output_columns].sort_values(
        ["comparison_id", "module_order", "ko_order"]
    )


def strip_bin_size_suffix(value: object) -> str:
    text = str(value).strip()
    if "." not in text:
        return text
    prefix, suffix = text.rsplit(".", 1)
    return prefix if suffix.isdigit() else text


def load_taxonomy_table(metadata: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    rows = []
    taxonomy_columns = ["Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"]
    key_columns = ["Genome_Id", "Bin Id", "Class", "Genus", "Species"]
    quality_columns = [
        "qscore", "Completeness", "Contamination", "N50", "sum_len",
        "16S_rRNA", "23S_rRNA", "5S_rRNA", "trna_unique",
    ]
    for (category, sample_id), _group in metadata.groupby(["category", "sample_id"], dropna=False):
        atlas_path = results_dir / str(category) / str(sample_id) / "Master_genome_QC.atlas.tsv"
        if not atlas_path.is_file():
            atlas_path = results_dir / str(category) / str(sample_id) / "Master_genome_QC.tsv"
        if not atlas_path.is_file():
            continue
        table = read_tsv(atlas_path)
        missing = set(key_columns) - set(table.columns)
        if missing:
            continue
        optional_columns = (
            (["pre_ani_bin_key"] if "pre_ani_bin_key" in table.columns else [])
            + [column for column in taxonomy_columns if column in table.columns and column not in key_columns]
            + [column for column in quality_columns if column in table.columns]
        )
        read_columns = key_columns + optional_columns
        for record in table.loc[:, read_columns].to_dict("records"):
            candidate_ids = {
                str(record.get("Genome_Id", "")).strip(),
                str(record.get("Bin Id", "")).strip(),
                strip_bin_size_suffix(record.get("Bin Id", "")),
                str(record.get("pre_ani_bin_key", "")).strip(),
            }
            for original_bin_id in candidate_ids:
                if not original_bin_id:
                    continue
                rows.append(
                    {
                        "category": str(category),
                        "sample_id": str(sample_id),
                        "original_bin_id": original_bin_id,
                        **{
                            column: str(record.get(column, "")).strip()
                            for column in taxonomy_columns
                        },
                        "qscore": record.get("qscore", np.nan),
                        "Completeness": record.get("Completeness", np.nan),
                        "Contamination": record.get("Contamination", np.nan),
                        "N50": record.get("N50", np.nan),
                        "sum_len": record.get("sum_len", np.nan),
                        "16S_rRNA": record.get("16S_rRNA", np.nan),
                        "23S_rRNA": record.get("23S_rRNA", np.nan),
                        "5S_rRNA": record.get("5S_rRNA", np.nan),
                        "trna_unique": record.get("trna_unique", np.nan),
                    }
                )
    taxonomy = pd.DataFrame(rows).drop_duplicates(["category", "sample_id", "original_bin_id"])
    return taxonomy


def taxon_mask(metadata: pd.DataFrame, taxon_filter: str) -> pd.Series:
    if taxon_filter == "all":
        return pd.Series(True, index=metadata.index)
    if taxon_filter == "gammaproteobacteria":
        return metadata["Class"].fillna("").astype(str).str.casefold().eq("gammaproteobacteria")
    if taxon_filter == "sup05":
        return metadata["Genus"].fillna("").astype(str).str.casefold().isin({"sup05", "thioglobus"})
    raise ValueError(f"Unsupported taxon filter: {taxon_filter}")


def add_quality_tier(metadata: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata.copy()
    for column in ["Completeness", "Contamination", "16S_rRNA", "23S_rRNA", "5S_rRNA", "trna_unique"]:
        metadata[column] = pd.to_numeric(metadata.get(column, np.nan), errors="coerce")
    high = (
        metadata["Completeness"].gt(90)
        & metadata["Contamination"].lt(5)
        & metadata["16S_rRNA"].fillna(0).gt(0)
        & metadata["23S_rRNA"].fillna(0).gt(0)
        & metadata["5S_rRNA"].fillna(0).gt(0)
        & metadata["trna_unique"].fillna(0).ge(18)
    )
    medium = metadata["Completeness"].ge(50) & metadata["Contamination"].lt(10)
    low = metadata["Completeness"].lt(50) & metadata["Contamination"].lt(10)
    metadata["quality_tier"] = np.select(
        [high, medium, low],
        ["high", "medium", "low"],
        default="failed",
    )
    return metadata


def taxon_label(taxon_filter: str) -> str:
    return {
        "all": "all genomes",
        "gammaproteobacteria": "Gammaproteobacteria",
        "sup05": "SUP05 + Thioglobus",
    }[taxon_filter]


def parse_unique_mapper_bin_id(value: object) -> str:
    text = str(value).strip()
    if "||" in text:
        return text.split("||", 1)[0]
    return text.rsplit("-", 1)[0] if "-" in text else text


def load_selected_ko_presence(
    mapper_unique_path: Path,
    selected_bins: set[str],
    selected_kos: set[str],
) -> pd.DataFrame:
    rows = []
    seen = set()
    with mapper_unique_path.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            ko = parts[1].strip()
            if ko not in selected_kos:
                continue
            bin_id = parse_unique_mapper_bin_id(parts[0])
            if bin_id not in selected_bins:
                continue
            key = (bin_id, ko)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"bin_id": bin_id, "ko": ko, "present": 1})
    return pd.DataFrame(rows)


def build_quality_filtered_plot_table(
    nitrogen_modules: pd.DataFrame,
    metadata_path: Path,
    mapper_unique_path: Path,
    enrichment_path: Path,
    results_dir: Path,
    taxon_filter: str,
    q_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = read_tsv(metadata_path)
    for column in ["bin_id", "original_bin_id", "category", "sample_id"]:
        if column not in metadata.columns:
            raise ValueError(f"{metadata_path} is missing required column: {column}")
        metadata[column] = metadata[column].astype(str)

    taxonomy = load_taxonomy_table(metadata, results_dir)
    filtered_metadata = metadata.merge(
        taxonomy,
        on=["category", "sample_id", "original_bin_id"],
        how="left",
    )
    filtered_metadata = add_quality_tier(filtered_metadata)
    filtered_metadata = filtered_metadata.loc[taxon_mask(filtered_metadata, taxon_filter)].copy()
    filtered_metadata = filtered_metadata.loc[filtered_metadata["quality_tier"].isin(["high", "medium"])].copy()
    if filtered_metadata.empty:
        raise ValueError(f"No medium-or-better genomes matched taxon filter {taxon_filter}")
    filtered_counts = (
        filtered_metadata.groupby(["category", "quality_tier"], as_index=False)
        .agg(n_genomes=("bin_id", "nunique"))
    )
    filtered_counts = (
        filtered_counts.pivot_table(
            index="category",
            columns="quality_tier",
            values="n_genomes",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    for column in ["high", "medium"]:
        if column not in filtered_counts.columns:
            filtered_counts[column] = 0
    filtered_counts["n_genomes"] = filtered_counts["high"].astype(int) + filtered_counts["medium"].astype(int)
    filtered_counts = filtered_counts.loc[:, ["category", "n_genomes", "high", "medium"]]
    filtered_counts["taxon_filter"] = taxon_filter
    filtered_counts["taxon_label"] = taxon_label(taxon_filter)
    filtered_counts["quality_filter"] = "medium_or_better"

    selected_kos = set(nitrogen_modules["ko"].astype(str))
    selected_bins = set(filtered_metadata["bin_id"].astype(str))
    presence = load_selected_ko_presence(mapper_unique_path, selected_bins, selected_kos)
    if presence.empty:
        presence = pd.DataFrame(columns=["bin_id", "ko", "present"])

    category_counts = filtered_metadata.groupby("category")["bin_id"].nunique().to_dict()
    bin_category = filtered_metadata.loc[:, ["bin_id", "category"]].drop_duplicates()
    present_by_category = (
        presence.merge(bin_category, on="bin_id", how="inner")
        .groupby(["category", "ko"], as_index=False)["bin_id"]
        .nunique()
        .rename(columns={"bin_id": "n_present"})
    )

    all_rows = []
    for comparison_id, (base_category, xpg_category, comparison_label) in COMPARISONS.items():
        base_n = int(category_counts.get(base_category, 0))
        xpg_n = int(category_counts.get(xpg_category, 0))
        p_values = []
        row_indices = []
        records = nitrogen_modules.copy()
        records["comparison_id"] = comparison_id
        records["comparison_label"] = comparison_label
        records["base_category"] = base_category
        records["xpg_category"] = xpg_category
        records["base_n_genomes"] = float(base_n)
        records["xpg_n_genomes"] = float(xpg_n)

        base_present = present_by_category.loc[present_by_category["category"].eq(base_category)]
        xpg_present = present_by_category.loc[present_by_category["category"].eq(xpg_category)]
        base_counts = base_present.set_index("ko")["n_present"]
        xpg_counts = xpg_present.set_index("ko")["n_present"]
        records["base_n_present"] = records["ko"].map(base_counts).fillna(0).astype(int)
        records["xpg_n_present"] = records["ko"].map(xpg_counts).fillna(0).astype(int)

        if base_n > 0:
            records["base_prevalence_percent"] = records["base_n_present"] / base_n * 100.0
            base_p = records["base_n_present"] / base_n
        else:
            records["base_prevalence_percent"] = np.nan
            base_p = pd.Series(np.nan, index=records.index)
        if xpg_n > 0:
            records["xpg_prevalence_percent"] = records["xpg_n_present"] / xpg_n * 100.0
            xpg_p = records["xpg_n_present"] / xpg_n
        else:
            records["xpg_prevalence_percent"] = np.nan
            xpg_p = pd.Series(np.nan, index=records.index)

        records["delta_prevalence_percent_points"] = (
            records["xpg_prevalence_percent"] - records["base_prevalence_percent"]
        )
        records["delta_se_percent_points"] = (
            np.sqrt(
                (xpg_p * (1.0 - xpg_p) / xpg_n if xpg_n > 0 else np.nan)
                + (base_p * (1.0 - base_p) / base_n if base_n > 0 else np.nan)
            )
            * 100.0
        )
        records["delta_ci95_low"] = (
            records["delta_prevalence_percent_points"] - 1.96 * records["delta_se_percent_points"]
        ).clip(lower=-100.0, upper=100.0)
        records["delta_ci95_high"] = (
            records["delta_prevalence_percent_points"] + 1.96 * records["delta_se_percent_points"]
        ).clip(lower=-100.0, upper=100.0)
        records["present_in_one_category_only"] = (
            records["base_n_present"].eq(0) & records["xpg_n_present"].gt(0)
        ) | (records["xpg_n_present"].eq(0) & records["base_n_present"].gt(0))

        records["unadjusted_p_value"] = np.nan
        if base_n > 0 and xpg_n > 0:
            for row_index, row in records.iterrows():
                p_value = fisher_exact_p_value(
                    int(row["base_n_present"]),
                    int(base_n - row["base_n_present"]),
                    int(row["xpg_n_present"]),
                    int(xpg_n - row["xpg_n_present"]),
                )
                p_values.append(p_value)
                row_indices.append(row_index)
            records.loc[row_indices, "unadjusted_p_value"] = p_values

        records["adjusted_q_value"] = benjamini_hochberg(records["unadjusted_p_value"])
        records["significant_q_le_threshold"] = records["adjusted_q_value"].le(q_threshold).fillna(False)
        records["associated_groups"] = np.where(
            records["significant_q_le_threshold"],
            records["comparison_label"],
            "",
        )
        records["taxon_filter"] = taxon_filter
        records["taxon_label"] = taxon_label(taxon_filter)
        all_rows.append(records)

    plot_table = pd.concat(all_rows, ignore_index=True)
    output_columns = [
        "taxon_filter",
        "taxon_label",
        "comparison_id",
        "comparison_label",
        "base_category",
        "xpg_category",
        "module_id",
        "module_name",
        "source_module_id",
        "source_module_name",
        "ko",
        "ko_name",
        "ko_label",
        "base_n_present",
        "xpg_n_present",
        "base_prevalence_percent",
        "xpg_prevalence_percent",
        "base_n_genomes",
        "xpg_n_genomes",
        "delta_prevalence_percent_points",
        "delta_se_percent_points",
        "delta_ci95_low",
        "delta_ci95_high",
        "unadjusted_p_value",
        "adjusted_q_value",
        "significant_q_le_threshold",
        "present_in_one_category_only",
        "associated_groups",
        "module_order",
        "ko_order",
    ]
    return (
        plot_table.loc[:, output_columns].sort_values(["comparison_id", "module_order", "ko_order"]),
        filtered_counts,
    )


def short_module_title(module_id: str, module_name: str) -> str:
    replacements = {
        "Complete nitrification, comammox, ammonia => nitrite => nitrate": (
            "Complete nitrification / comammox"
        ),
        "Dissimilatory nitrate reduction, nitrate => ammonia": (
            "Dissimilatory nitrate reduction"
        ),
        "Assimilatory nitrate reduction, nitrate => ammonia": (
            "Assimilatory nitrate reduction"
        ),
        "Nitrogen fixation, nitrogen => ammonia": "Nitrogen fixation",
        "Nitrification, ammonia => nitrite": "Nitrification",
        "Denitrification, nitrate => nitrogen": "Denitrification",
        "Anammox, nitrite + ammonia => nitrogen": "Anammox",
        "Urea cycle": "Urea cycle",
        "Nitrate assimilation": "Nitrate assimilation",
        "Purine degradation, xanthine => urea": "Purine degradation to urea",
    }
    title = f"{module_id} {replacements.get(module_name, module_name)}"
    return wrap_label(title, width=28)


def draw_comparison_plot(
    plot_table: pd.DataFrame,
    module_summary: pd.DataFrame,
    comparison_id: str,
    output_base: Path,
    delta_limit: float,
    q_threshold: float,
    cycle_label: str,
    page_label: str = "",
) -> list[Path]:
    if module_summary.shape[0] > MAX_FACETS_PER_COMPARISON_FIG and not page_label:
        written = []
        n_pages = int(math.ceil(module_summary.shape[0] / float(MAX_FACETS_PER_COMPARISON_FIG)))
        for page_index in range(n_pages):
            page_summary = module_summary.iloc[
                page_index * MAX_FACETS_PER_COMPARISON_FIG : (page_index + 1) * MAX_FACETS_PER_COMPARISON_FIG
            ].copy()
            page_base = Path(f"{output_base}_part{page_index + 1:02d}")
            written.extend(
                draw_comparison_plot(
                    plot_table,
                    page_summary,
                    comparison_id,
                    page_base,
                    delta_limit,
                    q_threshold,
                    cycle_label,
                    page_label=f"part {page_index + 1}/{n_pages}",
                )
            )
        return written

    comparison = plot_table.loc[plot_table["comparison_id"].eq(comparison_id)].copy()
    if comparison.empty:
        raise ValueError(f"No rows available for comparison {comparison_id}")

    plt = ensure_plotting()
    n_modules = int(module_summary.shape[0])
    n_cols = 2
    n_rows = int(math.ceil(n_modules / float(n_cols)))
    max_kos = max(1, int(module_summary["n_kos"].max()))
    panel_height = max(8.0, max_kos * 1.15 + 2.8)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(38.0, max(24.0, n_rows * panel_height)),
        squeeze=False,
        sharex=True,
    )
    axes = axes.ravel()

    wrote_any = False
    for ax, (_, module_record) in zip(axes, module_summary.iterrows()):
        module_id = str(module_record["module_id"])
        module_name = str(module_record["module_name"])
        module_data = comparison.loc[comparison["module_id"].eq(module_id)].sort_values("ko_order")
        if module_data.empty:
            ax.axis("off")
            continue

        y_positions = np.arange(module_data.shape[0], dtype=float)
        deltas = module_data["delta_prevalence_percent_points"].to_numpy(dtype=float)
        ci_low = module_data["delta_ci95_low"].to_numpy(dtype=float)
        ci_high = module_data["delta_ci95_high"].to_numpy(dtype=float)
        significant = module_data["significant_q_le_threshold"].to_numpy(dtype=bool)
        one_category_only = module_data["present_in_one_category_only"].to_numpy(dtype=bool)

        ax.axvline(0, color="black", linewidth=0.9, zorder=1)
        for y_pos, delta, low, high, is_significant, is_one_category_only in zip(
            y_positions,
            deltas,
            ci_low,
            ci_high,
            significant,
            one_category_only,
        ):
            if is_one_category_only:
                color = "black"
                facecolor = "white"
                edge = "black"
                size = 52
            elif is_significant:
                color = "black"
                facecolor = "black"
                edge = "black"
                size = 46
            else:
                color = "#bdbdbd"
                facecolor = "#bdbdbd"
                edge = "#737373"
                size = 34
            ax.errorbar(
                delta,
                y_pos,
                xerr=np.array([[delta - low], [high - delta]], dtype=float),
                fmt="none",
                ecolor=color,
                elinewidth=3.0,
                capsize=6.0,
                capthick=2.2,
                alpha=0.85,
                zorder=2,
            )
            ax.scatter(
                delta,
                y_pos,
                s=size * 3.8,
                facecolors=facecolor,
                edgecolors=edge,
                linewidths=1.8,
                zorder=3,
            )

        ax.set_yticks(y_positions)
        label_column = "ko_label" if "ko_label" in module_data.columns else "ko"
        ax.set_yticklabels([wrap_label(label, width=26) for label in module_data[label_column].tolist()])
        ax.invert_yaxis()
        ax.set_title(
            short_module_title(module_id, module_name),
            fontsize=MIN_PLOT_FONT_SIZE + 4,
            fontweight="bold",
            pad=14,
        )
        ax.set_xlim(-abs(delta_limit), abs(delta_limit))
        ax.grid(axis="x", color="#d9d9d9", linestyle="-", linewidth=1.1)
        ax.grid(axis="y", color="#eeeeee", linestyle="-", linewidth=0.9)
        ax.tick_params(axis="x", labelsize=MIN_PLOT_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=MIN_PLOT_FONT_SIZE)
        wrote_any = True

    for index in range(n_modules, len(axes)):
        axes[index].axis("off")

    if not wrote_any:
        plt.close(fig)
        raise ValueError(f"No plottable modules for comparison {comparison_id}")

    first = comparison["base_category"].iat[0].replace("_", " ")
    second = comparison["xpg_category"].iat[0].replace("_", " ")
    fig.supxlabel(
        wrap_label(f"Prevalence difference, {second} - {first} (percentage points)", width=78),
        fontsize=MIN_PLOT_FONT_SIZE,
        y=0.045,
    )
    fig.tight_layout(rect=[0.035, 0.08, 1, 0.98], w_pad=4.8, h_pad=4.4)
    save_nitrogen_figure(fig, output_base)
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def category_prevalence_table(plot_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in plot_table.to_dict("records"):
        common = {
            "taxon_filter": record.get("taxon_filter", "all"),
            "taxon_label": record.get("taxon_label", "all genomes"),
            "ko": record.get("ko", ""),
            "ko_name": record.get("ko_name", ""),
            "ko_label": record.get("ko_label", record.get("ko", "")),
        }
        rows.append(
            {
                **common,
                "category": record.get("base_category", ""),
                "prevalence_percent": record.get("base_prevalence_percent", np.nan),
                "n_genomes": record.get("base_n_genomes", np.nan),
            }
        )
        rows.append(
            {
                **common,
                "category": record.get("xpg_category", ""),
                "prevalence_percent": record.get("xpg_prevalence_percent", np.nan),
                "n_genomes": record.get("xpg_n_genomes", np.nan),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.drop_duplicates(["taxon_filter", "ko", "category"])


def build_quadrant_table(plot_table: pd.DataFrame) -> pd.DataFrame:
    plot_table = plot_table.copy()
    if "taxon_filter" not in plot_table.columns:
        plot_table["taxon_filter"] = "all"
    if "taxon_label" not in plot_table.columns:
        plot_table["taxon_label"] = "all genomes"
    prevalence = category_prevalence_table(plot_table)
    if prevalence.empty:
        return prevalence
    wide = prevalence.pivot_table(
        index=["taxon_filter", "taxon_label", "ko", "ko_name", "ko_label"],
        columns="category",
        values="prevalence_percent",
        aggfunc="first",
    ).reset_index()
    for category in ["MAGs", "SAGs", "xPG_MAGs", "xPG_SAGs"]:
        if category not in wide.columns:
            wide[category] = np.nan
    wide["xpg_effect_percent_points"] = (
        wide[["xPG_MAGs", "xPG_SAGs"]].mean(axis=1)
        - wide[["MAGs", "SAGs"]].mean(axis=1)
    )
    wide["sag_effect_percent_points"] = (
        wide[["SAGs", "xPG_SAGs"]].mean(axis=1)
        - wide[["MAGs", "xPG_MAGs"]].mean(axis=1)
    )
    wide["max_prevalence_percent"] = wide[["MAGs", "SAGs", "xPG_MAGs", "xPG_SAGs"]].max(axis=1)
    sig_rows = plot_table.loc[
        plot_table["significant_q_le_threshold"].fillna(False),
        [
            "taxon_filter",
            "taxon_label",
            "ko",
            "comparison_label",
            "base_category",
            "xpg_category",
            "delta_prevalence_percent_points",
        ],
    ].copy()
    if not sig_rows.empty:
        sig_rows["significant_favored_category"] = np.where(
            pd.to_numeric(sig_rows["delta_prevalence_percent_points"], errors="coerce").ge(0),
            sig_rows["xpg_category"],
            sig_rows["base_category"],
        )
        sig_rows["significant_pair_categories"] = (
            sig_rows["base_category"].astype(str)
            + "|"
            + sig_rows["xpg_category"].astype(str)
        )
        sig_rows = sig_rows.drop_duplicates()
    if sig_rows.empty:
        wide["any_pairwise_significant_q_le_threshold"] = False
        wide["significant_comparisons"] = ""
        wide["significant_pair_categories"] = ""
        wide["significant_favored_categories"] = ""
    else:
        sig_summary = (
            sig_rows.groupby(["taxon_filter", "taxon_label", "ko"], as_index=False)
            .agg(
                significant_comparisons=("comparison_label", lambda values: "; ".join(sorted(set(values)))),
                significant_pair_categories=("significant_pair_categories", lambda values: "; ".join(sorted(set(values)))),
                significant_favored_categories=(
                    "significant_favored_category",
                    lambda values: "; ".join(sorted(set(values))),
                ),
            )
            .assign(any_pairwise_significant_q_le_threshold=True)
        )
        wide = wide.merge(sig_summary, on=["taxon_filter", "taxon_label", "ko"], how="left")
        wide["any_pairwise_significant_q_le_threshold"] = wide["any_pairwise_significant_q_le_threshold"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        wide["significant_comparisons"] = wide["significant_comparisons"].fillna("")
        wide["significant_pair_categories"] = wide["significant_pair_categories"].fillna("")
        wide["significant_favored_categories"] = wide["significant_favored_categories"].fillna("")

    def dominant_quadrant_category(record: pd.Series) -> str:
        x_value = float(record.get("xpg_effect_percent_points", 0.0))
        y_value = float(record.get("sag_effect_percent_points", 0.0))
        if x_value >= 0 and y_value >= 0:
            return "xPG_SAGs"
        if x_value < 0 and y_value >= 0:
            return "SAGs"
        if x_value >= 0 and y_value < 0:
            return "xPG_MAGs"
        return "MAGs"

    wide["quadrant_category"] = wide.apply(dominant_quadrant_category, axis=1)
    wide["quadrant_supported_significant_q_le_threshold"] = wide.apply(
        lambda record: record["quadrant_category"]
        in {
            category.strip()
            for category in str(record.get("significant_favored_categories", "")).split(";")
            if category.strip()
        },
        axis=1,
    )
    wide["any_significant_q_le_threshold"] = wide["quadrant_supported_significant_q_le_threshold"]
    wide = wide.sort_values("ko").reset_index(drop=True)
    wide["label_number"] = np.arange(1, wide.shape[0] + 1)
    return wide


def draw_quadrant_plot(quadrant_table: pd.DataFrame, output_base: Path, cycle_label: str) -> list[Path]:
    if quadrant_table.empty:
        return []
    plt = ensure_plotting()
    quadrant_table = quadrant_table.copy().reset_index(drop=True)
    if "label_number" not in quadrant_table.columns:
        quadrant_table["label_number"] = np.arange(1, quadrant_table.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(26.0, 24.0))
    x = quadrant_table["xpg_effect_percent_points"].to_numpy(dtype=float)
    y = quadrant_table["sag_effect_percent_points"].to_numpy(dtype=float)
    sizes = 680.0 + quadrant_table["max_prevalence_percent"].fillna(0).to_numpy(dtype=float) * 14.0
    significant = quadrant_table.get(
        "any_significant_q_le_threshold",
        pd.Series(False, index=quadrant_table.index),
    ).fillna(False).to_numpy(dtype=bool)
    ax.scatter(
        x[~significant],
        y[~significant],
        s=sizes[~significant],
        facecolors="#d9d9d9",
        edgecolors="#737373",
        linewidths=2.0,
        alpha=0.88,
        zorder=3,
    )
    ax.scatter(
        x[significant],
        y[significant],
        s=sizes[significant],
        facecolors="black",
        edgecolors="black",
        linewidths=2.0,
        alpha=0.92,
        zorder=4,
    )
    for _, record in quadrant_table.iterrows():
        ax.annotate(
            str(int(record["label_number"])),
            (
                float(record["xpg_effect_percent_points"]),
                float(record["sag_effect_percent_points"]),
            ),
            ha="center",
            va="center",
            fontsize=MIN_PLOT_FONT_SIZE,
            color="white" if bool(record.get("any_significant_q_le_threshold", False)) else "black",
            fontweight="bold" if bool(record.get("any_significant_q_le_threshold", False)) else "normal",
            zorder=7,
        )

    ax.axhline(0, color="#2b7bba", linestyle="--", linewidth=2.4, zorder=1)
    ax.axvline(0, color="#2b7bba", linestyle="--", linewidth=2.4, zorder=1)

    def padded_limits(values: np.ndarray) -> tuple[float, float]:
        finite_values = values[np.isfinite(values)]
        if not finite_values.size:
            return (-5.0, 5.0)
        low = min(0.0, float(np.nanmin(finite_values)))
        high = max(0.0, float(np.nanmax(finite_values)))
        span = high - low
        if span <= 0:
            span = max(2.0, abs(high) * 0.2)
            low -= span / 2.0
            high += span / 2.0
        pad = max(1.5, span * 0.12)
        low = max(-100.0, low - pad)
        high = min(100.0, high + pad)
        if high - low < 6.0:
            mid = (high + low) / 2.0
            low = max(-100.0, mid - 3.0)
            high = min(100.0, mid + 3.0)
        return low, high

    ax.set_xlim(*padded_limits(x))
    ax.set_ylim(*padded_limits(y))
    ax.grid(color="#e6e6e6", linestyle="-", linewidth=1.2)
    scope = quadrant_table["taxon_label"].iat[0] if "taxon_label" in quadrant_table.columns else "all genomes"
    ax.set_title(
        wrap_label(f"{cycle_label} KO quadrant plot: {scope}", width=52),
        fontsize=MIN_PLOT_FONT_SIZE + 10,
        fontweight="bold",
        pad=22,
    )
    ax.set_xlabel(
        wrap_label("xPG effect: mean(xPG groups) - mean(non-xPG groups), percentage points", width=58),
        fontsize=MIN_PLOT_FONT_SIZE,
    )
    ax.set_ylabel(
        wrap_label("SAG effect: mean(SAG groups) - mean(MAG groups), percentage points", width=58),
        fontsize=MIN_PLOT_FONT_SIZE,
    )
    ax.tick_params(axis="both", labelsize=MIN_PLOT_FONT_SIZE)
    ax.text(0.06, 0.94, "SAGs", transform=ax.transAxes, ha="left", va="top", fontsize=MIN_PLOT_FONT_SIZE + 4, fontweight="bold")
    ax.text(0.94, 0.94, "xPG_SAGs", transform=ax.transAxes, ha="right", va="top", fontsize=MIN_PLOT_FONT_SIZE + 4, fontweight="bold")
    ax.text(0.06, 0.06, "MAGs", transform=ax.transAxes, ha="left", va="bottom", fontsize=MIN_PLOT_FONT_SIZE + 4, fontweight="bold")
    ax.text(0.94, 0.06, "xPG_MAGs", transform=ax.transAxes, ha="right", va="bottom", fontsize=MIN_PLOT_FONT_SIZE + 4, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.13)
    save_nitrogen_figure(fig, output_base)
    return [Path(str(output_base) + ".png"), Path(str(output_base) + ".pdf")]


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "elemental_cycles" / args.cycle / args.taxon_filter
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cycle_label = CYCLE_LABELS[args.cycle]
    prefix = args.prefix or f"kofam_{args.cycle}_modules"
    module_path = results_dir / "SI_kofam.ko_to_module.tsv"
    enrichment_path = results_dir / "kofam_ko.enrichment_results.compact.tsv"
    metadata_path = results_dir / "metadata.tsv"
    mapper_unique_path = results_dir / "all_kofam.mapper.unique.tsv"

    cycle_modules, module_summary, extra_ko_names = prepare_cycle_modules(results_dir, module_path, args.cycle)
    if args.cycle == "mobile_genetic_elements" or args.cycle in KEYWORD_CYCLE_PATTERNS:
        ko_names = load_ko_definition_map_rg(metadata_path, set(cycle_modules["ko"].astype(str)))
        ko_names.update({ko: name for ko, name in extra_ko_names.items() if ko not in ko_names})
    else:
        ko_names = load_ko_definition_map(metadata_path, set(cycle_modules["ko"].astype(str)))
        ko_names.update({ko: name for ko, name in extra_ko_names.items() if ko not in ko_names})
    cycle_modules = attach_ko_labels(cycle_modules, ko_names)
    plot_table, filtered_counts = build_quality_filtered_plot_table(
        cycle_modules,
        metadata_path,
        mapper_unique_path,
        enrichment_path,
        results_dir,
        args.taxon_filter,
        args.q_threshold,
    )

    selected_modules_path = output_dir / f"{prefix}.selected_modules.tsv"
    plot_table_path = output_dir / f"{prefix}.ko_delta_plotdata.tsv"
    module_summary.to_csv(selected_modules_path, sep="\t", index=False)
    plot_table.to_csv(plot_table_path, sep="\t", index=False)

    written = [selected_modules_path, plot_table_path]
    if not filtered_counts.empty:
        filtered_counts_path = output_dir / f"{prefix}.taxon_counts.tsv"
        filtered_counts.to_csv(filtered_counts_path, sep="\t", index=False)
        written.append(filtered_counts_path)
    quadrant_table = build_quadrant_table(plot_table)
    quadrant_table_path = output_dir / f"{prefix}.quadrant_plotdata.tsv"
    quadrant_table.to_csv(quadrant_table_path, sep="\t", index=False)
    written.append(quadrant_table_path)
    written.extend(draw_quadrant_plot(quadrant_table, output_dir / f"{prefix}_quadrant", cycle_label))
    for comparison_id in COMPARISONS:
        output_base = output_dir / f"{prefix}_{comparison_id}_delta"
        written.extend(
            draw_comparison_plot(
                plot_table,
                module_summary,
                comparison_id,
                output_base,
                args.delta_limit,
                args.q_threshold,
                cycle_label,
            )
        )

    print("[done] wrote:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
