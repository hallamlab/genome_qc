#!/usr/bin/env python3

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd


DEFAULT_PALETTE = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#2F4B7C",
    "#A05195",
    "#D45087",
    "#F95D6A",
    "#FFA600",
]

TEXT_WIDTH_FACTOR = 0.56

YAML_HEADER = """# ETE phylogeny render configuration
# This file is machine-readable YAML and safe to edit by hand.
# Lines beginning with # are comments. Keep indentation with spaces.
#
# Common edits:
# - Hide a color strip: remove or comment one item under metadata_strips.
# - Hide a text column: remove or comment one item under text_columns.
# - Move Phylum from leaves to clade labels: enable clade_brackets and remove Phylum from text_columns.
# - Color branches by taxonomy: enable branch_colors with column: Phylum.
# - Fix clipping: increase canvas.margin_right or canvas.width_mm.
# - Change rooting: set rooting.method to midpoint, outgroup, or as-is.
# - Tighten Family/Species spacing: adjust spacing.between_text_columns.
# - Add a small gap after Family: increase spacing.between_text_columns by 1-4 units.
# - Mark best-of-best genomes: enable best_representatives and set mark_best_representative on a text column.
#
# Key sections:
# tree / metadata: input files and metadata tables.
# rooting: display rooting before render; midpoint is automated, outgroup requires outgroup_id.
# support: FastTree local support labels; these are SH-like supports, not classic bootstraps.
# metadata_strips: colored blocks beside leaves, with optional custom palettes.
# branch_colors: colors pure branches by a metadata column; mixed ancestral branches can stay black.
# best_representatives: optional table used to append a marker to selected leaf labels.
# text_columns: aligned leaf text columns; width: auto uses the longest value in that column.
# legend: legend font sizes, swatch size, and padding.
# canvas: figure size and outside margins.
#
"""


DEFAULT_CONFIG = {
    "_help": "Edit this config to control reproducible ETE tree rendering. Keys beginning with _help are ignored by the renderer.",
    "tree": "",
    "_help_tree": "Path to the Newick tree file to render.",
    "metadata": [],
    "_help_metadata": "TSV metadata files. If empty, files named <tree_stem>_*_metadata.tsv beside the tree are auto-discovered.",
    "id_column": "tree_id",
    "_help_id_column": "Metadata column matching tree tip names.",
    "output_prefix": "",
    "_help_output_prefix": "Output path without extension. If empty, the tree path without suffix is used.",
    "output_formats": ["svg", "pdf", "png"],
    "_help_output_formats": "Any ETE-supported render formats, usually svg, pdf, png.",
    "newick_format": 0,
    "_help_newick_format": "ETE Newick parser/format. Keep 0 unless the tree fails to parse.",
    "rooting": {
        "_help": "Controls display rooting before rendering.",
        "method": "midpoint",
        "_help_method": "Allowed: midpoint, outgroup, as-is. Midpoint is automated and reproducible; outgroup requires outgroup_id.",
        "outgroup_id": "",
        "_help_outgroup_id": "Tree tip name to use when method is outgroup.",
    },
    "layout": "rectangular",
    "_help_layout": "rectangular or circular.",
    "ladderize": True,
    "_help_ladderize": "Sort branches for a cleaner display.",
    "show_leaf_names": True,
    "_help_show_leaf_names": "If true, show tip names or leaf_name_column values before metadata columns.",
    "leaf_name_column": "tree_id",
    "_help_leaf_name_column": "Metadata column used for tip labels when show_leaf_names is true. Use tree_id for raw tree names.",
    "leaf_name_font_size": 7,
    "branch_width": 1,
    "spacing": {
        "_help": "Horizontal gaps between aligned leaf elements, in ETE render units.",
        "after_leaf_name": 8,
        "before_strips": 16,
        "between_strips": 4,
        "between_strips_and_text": 12,
        "between_text_columns": 4,
    },
    "support": {
        "_help": "Internal branch support label settings. FastTree values are SH-like supports, not classic bootstraps.",
        "show": True,
        "min_value": 0.9,
        "_help_min_value": "Only show support values at or above this threshold. Use 0 to show all.",
        "font_size": 6,
        "scale": "auto",
        "_help_scale": "auto converts 0-1 support values to 0-100 labels. Use raw or percent to force behavior.",
        "position": "branch-top",
    },
    "clade_brackets": [
        {
            "_help": "Annotate monophyletic clades using a metadata column. Non-monophyletic values are split into separate clade blocks when possible.",
            "column": "Phylum",
            "label": "Phylum",
            "show": False,
            "font_size": 7,
            "position": "branch-right",
            "min_tips": 2,
        }
    ],
    "_help_clade_brackets": "Use this to show ranks such as Phylum as clade labels instead of repeated leaf text.",
    "branch_colors": [
        {
            "_help": "Color branches when all descendant leaves share the same metadata value.",
            "column": "Phylum",
            "label": "Phylum",
            "show": False,
            "palette": "default",
            "mixed_color": "#808080",
        }
    ],
    "_help_branch_colors": "Use this for taxonomy-aware branch coloring, for example by Phylum.",
    "metadata_strips": [
        {
            "_help": "Colored square strip aligned to each leaf.",
            "column": "sample",
            "label": "Sample",
            "width": 12,
            "palette": "default",
        },
        {
            "_help": "Palette can be 'default' or a mapping from metadata values to hex colors.",
            "column": "mimag_tier",
            "label": "Quality",
            "width": 12,
            "palette": {"high": "#2CA25F", "medium": "#F1C40F", "low": "#D95F0E"},
        },
    ],
    "_help_metadata_strips": "Remove entries from metadata_strips to hide colored strips.",
    "heatmap_columns": [],
    "_help_heatmap_columns": "Optional grayscale heatmap cells between color strips and text columns.",
    "best_representatives": {
        "_help": "Optional best-of-best marker source. All rows in table are treated as best representatives.",
        "show": False,
        "table": "",
        "marker": "*",
        "match_columns": ["sample", "category", "Genome_Id"],
        "output_column": "best_representative",
    },
    "_help_best_representatives": "Use this to mark leaves present in the wrapper best_of_best/selected_set/master.tsv.",
    "text_columns": [
        {
            "_help": "Text metadata column aligned to each leaf. Width reserves horizontal space for cleaner spacing.",
            "column": "category",
            "label": "Category",
            "font_size": 7,
            "max_length": 12,
            "width": 72,
        },
        {"column": "Phylum", "label": "Phylum", "font_size": 6, "max_length": 24, "width": "auto"},
        {"column": "Family", "label": "Family", "font_size": 6, "max_length": 24, "width": "auto"},
        {
            "column": "Species",
            "label": "Species",
            "font_size": 6,
            "max_length": 30,
            "width": "auto",
            "mark_best_representative": True,
        },
    ],
    "_help_text_columns": "Remove entries to hide taxonomy/metadata text columns. Useful keys: column, label, font_size, max_length, width.",
    "numeric_bars": [],
    "_help_numeric_bars": "Optional aligned numeric bars. Empty list disables them.",
    "legend": {"show": True, "max_items_per_strip": 20},
    "_help_legend": "Legend applies to metadata_strips. If there are no strips, set show to false.",
    "canvas": {
        "_help": "Canvas and margin settings. Increase margin_right or width_mm if labels are clipped.",
        "width_mm": 320,
        "dpi": 300,
        "arc_start": 0,
        "arc_span": 359,
        "margin_top": 30,
        "margin_right": 180,
        "margin_bottom": 30,
        "margin_left": 30,
        "show_scale": True,
    },
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def load_ete():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from ete4 import Tree
        from ete4.treeview import NodeStyle, RectFace, StackedBarFace, TextFace, TreeStyle

        return {
            "version": "ete4",
            "Tree": Tree,
            "TreeStyle": TreeStyle,
            "TextFace": TextFace,
            "RectFace": RectFace,
            "StackedBarFace": StackedBarFace,
            "NodeStyle": NodeStyle,
        }
    except ImportError as ete4_exc:
        try:
            import ete4  # noqa: F401
        except ImportError:
            pass
        else:
            message = str(ete4_exc)
            if "PyQt6" in message:
                raise ImportError(
                    "ETE4 is installed, but its tree renderer requires PyQt6. "
                    "Install it in the render environment with: "
                    "mamba install -n genome_qc_ete_render -c conda-forge pyqt"
                ) from ete4_exc
            raise
    try:
        from ete3 import Tree, TreeStyle, TextFace, RectFace, NodeStyle
        try:
            from ete3 import StackedBarFace
        except ImportError:
            StackedBarFace = None

        return {
            "version": "ete3",
            "Tree": Tree,
            "TreeStyle": TreeStyle,
            "TextFace": TextFace,
            "RectFace": RectFace,
            "StackedBarFace": StackedBarFace,
            "NodeStyle": NodeStyle,
        }
    except ImportError as exc:
        raise ImportError(
            "ETE is required for rendering. Install ete4 if possible, or ete3 as a fallback."
        ) from exc


def deep_copy_config(config):
    return json.loads(json.dumps(config))


def read_config(path):
    path = Path(path)
    with open(path, "r") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs require PyYAML. Install pyyaml in this environment.") from exc
            user_config = yaml.safe_load(handle) or {}
        else:
            user_config = json.load(handle)
    config = deep_copy_config(DEFAULT_CONFIG)
    merge_dict(config, user_config)
    return config


def merge_dict(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge_dict(base[key], value)
        else:
            base[key] = value


def write_config(path, config):
    path = Path(path)
    with open(path, "w") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("YAML configs require PyYAML. Install pyyaml in this environment.") from exc
            handle.write(YAML_HEADER)
            yaml.safe_dump(strip_help_entries(config), handle, sort_keys=False, default_flow_style=False, allow_unicode=True)
        else:
            json.dump(config, handle, indent=2)
            handle.write("\n")


def strip_help_entries(value):
    if isinstance(value, dict):
        return {
            key: strip_help_entries(item)
            for key, item in value.items()
            if not str(key).startswith("_help")
        }
    if isinstance(value, list):
        return [strip_help_entries(item) for item in value]
    return value


def clean_value(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def truncate_text(text, max_length):
    text = clean_value(text)
    if max_length is None or clean_value(max_length).lower() in {"", "none", "null", "false"}:
        return text
    if len(text) <= int(max_length):
        return text
    if int(max_length) <= 3:
        return text[: int(max_length)]
    return text[: int(max_length) - 3] + "..."


def metadata_paths_from_config(config, tree_path=None):
    paths = [Path(path).expanduser() for path in config.get("metadata", []) if clean_value(path)]
    if paths or tree_path is None:
        return paths
    tree_path = Path(tree_path)
    stem = tree_path.stem
    return sorted(tree_path.parent.glob(f"{stem}_*_metadata.tsv"))


def fasta_suffixless_name(path_text):
    name = Path(str(path_text)).name
    for suffix in [".fasta.gz", ".fa.gz", ".fna.gz", ".fasta", ".fa", ".fna"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def load_metadata(paths, id_column):
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        if id_column not in frame.columns:
            raise ValueError(f"Metadata file is missing id column '{id_column}': {path}")
        frame = frame.drop_duplicates(id_column)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[id_column])
    merged = frames[0]
    for frame in frames[1:]:
        overlap = [column for column in frame.columns if column in merged.columns and column != id_column]
        if overlap:
            frame = frame.drop(columns=overlap)
        merged = merged.merge(frame, on=id_column, how="outer")
    return merged.fillna("")


def normalized_key(values):
    return tuple(clean_value(value).lower() for value in values)


def annotate_best_representatives(metadata, best_config, id_column):
    if metadata.empty or not best_config.get("show", False):
        return metadata
    table_text = clean_value(best_config.get("table", ""))
    output_column = clean_value(best_config.get("output_column", "best_representative")) or "best_representative"
    metadata = metadata.copy()
    metadata[output_column] = "false"
    if not table_text:
        return metadata
    table_path = Path(table_text).expanduser()
    if not table_path.exists() or not table_path.is_file():
        log(f"[warn] best representative table not found: {table_path}")
        return metadata

    best_df = pd.read_csv(table_path, sep="\t", dtype=str).fillna("")
    match_columns = [clean_value(column) for column in best_config.get("match_columns", []) if clean_value(column)]
    best_keys = set()
    if match_columns and all(column in metadata.columns for column in match_columns) and all(column in best_df.columns for column in match_columns):
        best_keys = {normalized_key(row) for row in best_df.loc[:, match_columns].itertuples(index=False, name=None)}
        metadata_keys = metadata.loc[:, match_columns].apply(lambda row: normalized_key(row.tolist()), axis=1)
        metadata[output_column] = metadata_keys.map(lambda key: "true" if key in best_keys else "false")
    elif id_column in best_df.columns:
        best_ids = {clean_value(value) for value in best_df[id_column].tolist()}
        metadata[output_column] = metadata[id_column].map(lambda value: "true" if clean_value(value) in best_ids else "false")
    elif "copied_fasta_path" in best_df.columns:
        best_ids = {fasta_suffixless_name(value) for value in best_df["copied_fasta_path"].tolist() if clean_value(value)}
        metadata[output_column] = metadata[id_column].map(lambda value: "true" if clean_value(value) in best_ids else "false")
    else:
        log("[warn] best representative table could not be matched; no usable match columns found")
        return metadata

    marked = int(metadata[output_column].eq("true").sum())
    log(f"[info] best representatives marked: {marked}")
    return metadata


def palette_lookup(values, palette_config, used_colors=None, palette_offset=0):
    used_colors = used_colors if used_colors is not None else set()
    cleaned_values = [clean_value(value) for value in values if clean_value(value)]
    unique_values = sorted(set(cleaned_values), key=lambda value: value.lower())
    if isinstance(palette_config, dict):
        colors = {str(key): value for key, value in palette_config.items()}
    else:
        colors = {}
    for index, value in enumerate(unique_values):
        if value in colors:
            used_colors.add(str(colors[value]).lower())
            continue
        for step in range(len(DEFAULT_PALETTE)):
            color = DEFAULT_PALETTE[(index + palette_offset + step) % len(DEFAULT_PALETTE)]
            if color.lower() not in used_colors:
                colors[value] = color
                used_colors.add(color.lower())
                break
        colors.setdefault(value, DEFAULT_PALETTE[(index + palette_offset) % len(DEFAULT_PALETTE)])
    colors.setdefault("", "#D9D9D9")
    return colors


def support_value(node):
    candidates = []
    if hasattr(node, "support"):
        candidates.append(getattr(node, "support"))
    if hasattr(node, "name"):
        candidates.append(getattr(node, "name"))
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def is_leaf_node(node):
    value = getattr(node, "is_leaf", False)
    if callable(value):
        return bool(value())
    return bool(value)


def load_tree_with_parser(ete, tree_path, parser_value):
    Tree = ete["Tree"]
    if ete["version"] == "ete4":
        with open(tree_path, "r") as handle:
            return Tree(handle, parser=int(parser_value))
    return Tree(str(tree_path), format=int(parser_value))


def apply_rooting(tree, config):
    rooting = config.get("rooting", {})
    method = clean_value(rooting.get("method", "")).lower()
    if method in {"", "none", "as-is", "asis", "original"}:
        return "as-is"
    if method == "midpoint":
        tree.set_midpoint_outgroup()
        return "midpoint"
    if method in {"outgroup", "outgroup_id", "tip"}:
        outgroup_id = clean_value(rooting.get("outgroup_id", ""))
        if not outgroup_id:
            raise ValueError("rooting.method is 'outgroup' but rooting.outgroup_id is empty")
        matches = tree.search_nodes(name=outgroup_id)
        if not matches:
            matches = tree.search_leaves_by_name(outgroup_id)
        if not matches:
            raise ValueError(f"Could not find outgroup tip in tree: {outgroup_id}")
        tree.set_outgroup(matches[0])
        return f"outgroup:{outgroup_id}"
    raise ValueError(f"Unsupported rooting.method: {method}")


def format_support(value, scale):
    if value is None:
        return ""
    if scale == "percent" or (scale == "auto" and value <= 1.0):
        return f"{value * 100:.0f}"
    return f"{value:.0f}" if value >= 10 else f"{value:.2f}".rstrip("0").rstrip(".")


def add_face(ete, face, node, column, position):
    node.add_face(face, column=column, position=position)


def make_text_face(ete, text, size, color="#222222"):
    face = ete["TextFace"](str(text), fsize=int(size), fgcolor=color)
    return face


def apply_face_width(face, width):
    if width is None:
        return face
    actual_width = getattr(face, "width", None)
    if actual_width is None:
        return face
    try:
        face.margin_right = max(0, int(width) - int(actual_width))
    except Exception:
        try:
            face.inner_background.color = "#FFFFFF"
            face.opacity = 0
        except Exception:
            pass
    return face


def make_blank_face(ete, width=1, height=1):
    return make_rect_face(ete, width, height, "#FFFFFF")


def make_rect_face(ete, width, height, color):
    return ete["RectFace"](int(width), int(height), fgcolor=color, bgcolor=color)


def make_heatmap_colorbar_face(ete, legend_config):
    StackedBarFace = ete.get("StackedBarFace")
    if StackedBarFace is None:
        return None
    width = int(legend_config.get("heatmap_colorbar_width", 90))
    height = int(legend_config.get("heatmap_colorbar_bar_height", 8))
    steps = max(8, int(legend_config.get("heatmap_colorbar_steps", 32)))
    percents = [100.0 / steps] * steps
    colors = [grayscale_hex(index / (steps - 1), 0, 1) for index in range(steps)]
    return StackedBarFace(percents, width=width, height=height, colors=colors, line_color="#333333")


def svg_size_mm(svg_path):
    text = Path(svg_path).read_text(errors="ignore")[:1000]
    match = re.search(r"<svg[^>]*\bwidth=\"([0-9.]+)mm\"[^>]*\bheight=\"([0-9.]+)mm\"", text)
    if not match:
        raise ValueError(f"Could not read SVG millimeter dimensions from {svg_path}")
    return float(match.group(1)), float(match.group(2))


def convert_svg_to_pdf(svg_path, pdf_path, dpi=300):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets
    except ImportError as exc:
        raise ImportError("PDF output requires PyQt6 SVG/PDF support when converting from SVG.") from exc

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    width_mm, height_mm = svg_size_mm(svg_path)
    writer = QtGui.QPdfWriter(str(pdf_path))
    writer.setResolution(int(dpi))
    page_size = QtGui.QPageSize(
        QtCore.QSizeF(width_mm, height_mm),
        QtGui.QPageSize.Unit.Millimeter,
    )
    layout = QtGui.QPageLayout(
        page_size,
        QtGui.QPageLayout.Orientation.Portrait,
        QtCore.QMarginsF(0, 0, 0, 0),
        QtGui.QPageLayout.Unit.Millimeter,
    )
    writer.setPageLayout(layout)
    renderer = QtSvg.QSvgRenderer(str(svg_path))
    painter = QtGui.QPainter(writer)
    renderer.render(painter)
    painter.end()


def convert_svg_to_png(svg_path, png_path, dpi=300):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6 import QtCore, QtGui, QtSvg, QtWidgets
    except ImportError as exc:
        raise ImportError("PNG output requires PyQt6 SVG support when converting from SVG.") from exc

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    renderer = QtSvg.QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise RuntimeError(f"Could not read SVG input for PNG conversion: {svg_path}")

    # ETE writes physically large SVG dimensions. Expanding those millimeters
    # at print DPI can produce enormous blank/unopenable PNGs, so use the SVG
    # renderer's intrinsic pixel size and keep dpi as metadata.
    default_size = renderer.defaultSize()
    if default_size.isValid() and default_size.width() > 0 and default_size.height() > 0:
        width_px = int(default_size.width())
        height_px = int(default_size.height())
    else:
        width_mm, height_mm = svg_size_mm(svg_path)
        width_px = max(1, int(round(width_mm / 25.4 * int(dpi))))
        height_px = max(1, int(round(height_mm / 25.4 * int(dpi))))
    image = QtGui.QImage(width_px, height_px, QtGui.QImage.Format.Format_ARGB32)
    image.setDotsPerMeterX(int(round(int(dpi) / 25.4 * 1000.0)))
    image.setDotsPerMeterY(int(round(int(dpi) / 25.4 * 1000.0)))
    image.fill(QtGui.QColor("#FFFFFF"))

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
    renderer.render(painter, QtCore.QRectF(0, 0, width_px, height_px))
    painter.end()
    if not image.save(str(png_path)):
        raise RuntimeError(f"Could not write PNG output: {png_path}")


def align_svg_heatmap_labels(svg_path, heatmap_configs, legend_config=None):
    """Center rotated aligned-foot heatmap labels over their rendered cells.

    ETE positions rotated TextFace labels from the text baseline, which leaves
    the labels visually offset from narrow heatmap columns. The SVG is the
    source for PDF conversion too, so correcting it once keeps both outputs in
    agreement.
    """
    visible_heatmaps = [column for column in heatmap_configs if column.get("show", True)]
    if not visible_heatmaps:
        return
    try:
        import xml.etree.ElementTree as ET

        ET.register_namespace("", "http://www.w3.org/2000/svg")
        svg_path = Path(svg_path)
        root = ET.fromstring(svg_path.read_text(errors="ignore"))
        namespace = "{http://www.w3.org/2000/svg}"

        def parse_matrix(value):
            match = re.search(r"matrix\(([^)]*)\)", value or "")
            if not match:
                return None
            parts = [float(part) for part in re.split(r"[ ,]+", match.group(1).strip()) if part]
            return parts if len(parts) == 6 else None

        def format_matrix(parts):
            return "matrix(" + ",".join(f"{part:.6g}" for part in parts) + ")"

        heatmap_sizes = {
            (float(column.get("width", 8)), float(column.get("height", 10)))
            for column in visible_heatmaps
        }
        expected_labels = {
            clean_value(column.get("label", column.get("column", ""))) or clean_value(column.get("column", ""))
            for column in visible_heatmaps
        }

        legend_config = legend_config or {}
        visual_center_fraction = float(legend_config.get("heatmap_label_visual_center_fraction", 0.58))
        manual_offset_px = float(legend_config.get("heatmap_label_center_offset_px", 0.0))

        centers = Counter()
        labels = []
        for group in root.iter(namespace + "g"):
            matrix = parse_matrix(group.get("transform"))
            if matrix is None:
                continue
            a, b, c, d, e, f = matrix
            for rect in group.findall(namespace + "rect"):
                width = float(rect.get("width", "0"))
                height = float(rect.get("height", "0"))
                if not any(abs(width - hw) < 0.01 and abs(height - hh) < 0.01 for hw, hh in heatmap_sizes):
                    continue
                x_value = float(rect.get("x", "0"))
                y_value = float(rect.get("y", "0"))
                center_x = e + a * (x_value + width / 2.0) + c * (y_value + height / 2.0)
                centers[round(center_x, 2)] += 1
            for text_element in group.findall(namespace + "text"):
                text = clean_value("".join(text_element.itertext()))
                if text not in expected_labels:
                    continue
                if not (abs(a) < 1e-6 and b < 0 and c > 0 and abs(d) < 1e-6):
                    continue
                text_x = float(text_element.get("x", "0"))
                text_y = float(text_element.get("y", "0"))
                current_x = e + a * text_x + c * text_y
                labels.append((current_x, group, matrix, text_x, text_y))

        if not centers or not labels:
            return
        repeated_centers = sorted(center for center, count in centers.items() if count > 1)
        if len(repeated_centers) < len(labels):
            return
        labels.sort(key=lambda item: item[0])
        if len(repeated_centers) != len(labels):
            repeated_centers = [
                min(repeated_centers, key=lambda center: abs(center - label[0]))
                for label in labels
            ]

        for target_center, (_current_x, group, matrix, text_x, text_y) in zip(repeated_centers, labels):
            a, b, c, d, _e, f = matrix
            visual_center_y = text_y * visual_center_fraction
            matrix[4] = target_center + manual_offset_px - a * text_x - c * visual_center_y
            group.set("transform", format_matrix(matrix))

        svg_path.write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            + ET.tostring(root, encoding="unicode")
        )
    except Exception:
        log("[warn] could not center SVG heatmap labels")


def add_svg_heatmap_dividers(svg_path, heatmap_configs):
    visible_heatmaps = [column for column in heatmap_configs if column.get("show", True)]
    divider_indices = [
        index for index, column in enumerate(visible_heatmaps[:-1])
        if column.get("divider_after", False)
    ]
    if not visible_heatmaps or not divider_indices:
        return
    try:
        import xml.etree.ElementTree as ET

        ET.register_namespace("", "http://www.w3.org/2000/svg")
        svg_path = Path(svg_path)
        root = ET.fromstring(svg_path.read_text(errors="ignore"))
        namespace = "{http://www.w3.org/2000/svg}"

        def parse_matrix(value):
            match = re.search(r"matrix\(([^)]*)\)", value or "")
            if not match:
                return None
            parts = [float(part) for part in re.split(r"[ ,]+", match.group(1).strip()) if part]
            return parts if len(parts) == 6 else None

        heatmap_sizes = {
            (float(column.get("width", 8)), float(column.get("height", 10)))
            for column in visible_heatmaps
        }
        centers = Counter()
        y_values = []
        for group in root.iter(namespace + "g"):
            matrix = parse_matrix(group.get("transform"))
            if matrix is None:
                continue
            a, b, c, d, e, f = matrix
            for rect in group.findall(namespace + "rect"):
                width = float(rect.get("width", "0"))
                height = float(rect.get("height", "0"))
                if not any(abs(width - hw) < 0.01 and abs(height - hh) < 0.01 for hw, hh in heatmap_sizes):
                    continue
                x_value = float(rect.get("x", "0"))
                y_value = float(rect.get("y", "0"))
                corners = [
                    (x_value, y_value),
                    (x_value + width, y_value),
                    (x_value, y_value + height),
                    (x_value + width, y_value + height),
                ]
                transformed = [(e + a * x + c * y, f + b * x + d * y) for x, y in corners]
                center_x = sum(point[0] for point in transformed) / 4.0
                centers[round(center_x, 2)] += 1
                y_values.extend(point[1] for point in transformed)

        repeated_centers = sorted(center for center, count in centers.items() if count > 1)
        if len(repeated_centers) < len(visible_heatmaps) or not y_values:
            return
        y_min = min(y_values)
        y_max = max(y_values)
        for divider_index in divider_indices:
            if divider_index + 1 >= len(repeated_centers):
                continue
            divider_config = visible_heatmaps[divider_index]
            x_value = (repeated_centers[divider_index] + repeated_centers[divider_index + 1]) / 2.0
            offset = float(divider_config.get("divider_offset_px", 0.0))
            ET.SubElement(
                root,
                namespace + "line",
                {
                    "x1": f"{x_value + offset:.3f}",
                    "y1": f"{y_min:.3f}",
                    "x2": f"{x_value + offset:.3f}",
                    "y2": f"{y_max:.3f}",
                    "stroke": clean_value(divider_config.get("divider_color", "#4A4A4A")) or "#4A4A4A",
                    "stroke-width": str(divider_config.get("divider_width", 1.5)),
                    "stroke-dasharray": clean_value(divider_config.get("divider_dash", "6,4")) or "6,4",
                    "stroke-linecap": "butt",
                },
            )

        svg_path.write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            + ET.tostring(root, encoding="unicode")
        )
    except Exception:
        log("[warn] could not add SVG heatmap divider")


def add_gap_face(ete, node, column, width, height=1, position="aligned"):
    if width and int(width) > 0:
        add_face(ete, make_blank_face(ete, int(width), int(height)), node, column, position)
        return column + 1
    return column


def estimate_text_width(text, font_size):
    return int(math.ceil(len(clean_value(text)) * float(font_size) * TEXT_WIDTH_FACTOR)) + 8


def grayscale_hex(value, min_value, max_value, reverse=False, missing_color="#F2F2F2"):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return missing_color
    if not math.isfinite(numeric):
        return missing_color
    min_value = float(min_value)
    max_value = float(max_value)
    if max_value <= min_value:
        fraction = 0.5
    else:
        fraction = max(0.0, min((numeric - min_value) / (max_value - min_value), 1.0))
    if reverse:
        fraction = 1.0 - fraction
    shade = int(round(250 - (fraction * 230)))
    shade = max(0, min(255, shade))
    return f"#{shade:02X}{shade:02X}{shade:02X}"


def heatmap_numeric_value(value, heatmap_config):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(numeric):
        return math.nan
    numeric = numeric * float(heatmap_config.get("value_scale", 1.0))
    numeric = numeric + float(heatmap_config.get("value_offset", 0.0))
    if heatmap_config.get("clamp", True):
        try:
            min_value = float(heatmap_config.get("min_value", 0.0))
            max_value = float(heatmap_config.get("max_value", 1.0))
        except (TypeError, ValueError):
            return numeric
        lower = min(min_value, max_value)
        upper = max(min_value, max_value)
        numeric = max(lower, min(numeric, upper))
    return numeric


def heatmap_start_column(strip_configs, heatmap_configs, text_configs, spacing_config):
    column_index = 0
    if int(spacing_config.get("before_strips", 0)) > 0:
        column_index += 1
    for strip in strip_configs:
        column_index += 1
        if int(spacing_config.get("between_strips", 0)) > 0:
            column_index += 1
    if strip_configs and (text_configs or heatmap_configs) and int(spacing_config.get("between_strips_and_text", 0)) > 0:
        column_index += 1
    return column_index


def heatmap_column_indices(strip_configs, heatmap_configs, text_configs, spacing_config):
    column_index = heatmap_start_column(strip_configs, heatmap_configs, text_configs, spacing_config)
    indices = []
    for heatmap_config in heatmap_configs:
        indices.append(column_index)
        column_index += 1
        if int(heatmap_config.get("gap_after", 0)) > 0:
            column_index += 1
    return indices


def resolve_heatmap_columns(heatmap_configs, metadata):
    resolved = []
    for heatmap_config in heatmap_configs:
        if not heatmap_config.get("show", True):
            continue
        item = dict(heatmap_config)
        column = item.get("column", "")
        if column in metadata.columns:
            values = pd.to_numeric(metadata[column], errors="coerce")
            if "value_scale" in item or "value_offset" in item:
                values = values.map(lambda value: heatmap_numeric_value(value, item))
        else:
            values = pd.Series(dtype=float)
        if item.get("min_value", None) is None:
            item["min_value"] = float(values.min()) if values.notna().any() else 0.0
        if item.get("max_value", None) is None:
            item["max_value"] = float(values.max()) if values.notna().any() else 1.0
        resolved.append(item)
    return resolved


def resolve_text_column_widths(text_configs, metadata):
    resolved = []
    for text_config in text_configs:
        item = dict(text_config)
        width = item.get("width", "auto")
        if clean_value(width).lower() == "auto":
            column = item.get("column", "")
            font_size = int(item.get("font_size", 6))
            max_length = item.get("max_length", None)
            values = []
            if column in metadata.columns:
                values = [truncate_text(value, max_length) for value in metadata[column].tolist()]
                if item.get("mark_best_representative", False):
                    best_column = clean_value(item.get("best_representative_column", "best_representative"))
                    if best_column in metadata.columns:
                        marker = clean_value(item.get("best_representative_marker", "*")) or "*"
                        flags = metadata[best_column].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
                        values = [
                            f"{value}{marker}" if is_best else value
                            for value, is_best in zip(values, flags.tolist())
                        ]
            label = item.get("label", column)
            longest = max([label] + values, key=lambda value: len(clean_value(value))) if values or label else ""
            item["width"] = max(24, estimate_text_width(longest, font_size))
            item["_resolved_width"] = item["width"]
        elif clean_value(width).lower() in {"natural", "none", "false"}:
            item["width"] = None
            item["_resolved_width"] = "natural"
        resolved.append(item)
    return resolved


def apply_text_alignment(face, text, text_config):
    alignment = clean_value(text_config.get("align", "left")).lower()
    if alignment not in {"right", "center"}:
        return face
    width = text_config.get("width")
    if width is None or clean_value(width).lower() in {"", "auto", "natural", "none", "false"}:
        width = text_config.get("_resolved_width")
    try:
        width = int(width)
    except (TypeError, ValueError):
        return face
    text_width = estimate_text_width(text, int(text_config.get("font_size", 6)))
    extra = max(0, width - text_width)
    if alignment == "center":
        extra = extra // 2
    try:
        face.margin_left = int(extra)
    except Exception:
        pass
    return face


def annotate_branch_colors(tree, metadata_by_id, branch_color_configs):
    if not branch_color_configs:
        return [], {}
    annotations = []
    legends = {}
    leaves = leaf_nodes(tree)
    used_colors = set()
    for config_index, color_config in enumerate(branch_color_configs):
        if not color_config.get("show", False):
            continue
        column = color_config.get("column", "")
        if not column:
            continue
        values = [
            clean_value(metadata_by_id.get(leaf.name, {}).get(column, ""))
            for leaf in leaves
            if clean_value(metadata_by_id.get(leaf.name, {}).get(column, ""))
        ]
        color_map = palette_lookup(values, color_config.get("palette", "default"), used_colors=used_colors, palette_offset=config_index * 5)
        prop_name = f"_branch_color_{column}"
        value_prop_name = f"_branch_value_{column}"
        for node in tree.traverse():
            desc_values = {
                clean_value(metadata_by_id.get(leaf.name, {}).get(column, ""))
                for leaf in leaf_nodes(node)
            }
            desc_values.discard("")
            if len(desc_values) == 1:
                value = next(iter(desc_values))
                node.add_prop(prop_name, color_map.get(value, "#000000"))
                node.add_prop(value_prop_name, value)
                annotations.append({"column": column, "value": value})
            elif color_config.get("color_mixed", False):
                node.add_prop(prop_name, color_config.get("mixed_color", "#808080"))
        legends[column] = {"label": color_config.get("label", column), "colors": color_map}
    return annotations, legends


def set_node_style(ete, node, branch_width):
    style = ete["NodeStyle"]()
    style["size"] = 0
    style["hz_line_width"] = int(branch_width)
    style["vt_line_width"] = int(branch_width)
    node.set_style(style)


def set_branch_style(ete, node, branch_width, color=None):
    style = ete["NodeStyle"]()
    style["size"] = 0
    style["hz_line_width"] = int(branch_width)
    style["vt_line_width"] = int(branch_width)
    if color:
        style["hz_line_color"] = color
        style["vt_line_color"] = color
    node.set_style(style)


def leaf_nodes(tree):
    return [node for node in tree.traverse() if is_leaf_node(node)]


def descendant_leaf_names(node):
    return {leaf.name for leaf in leaf_nodes(node)}


def annotate_clade_brackets(tree, metadata_by_id, bracket_configs):
    if not bracket_configs:
        return []
    leaves = leaf_nodes(tree)
    leaf_name_set = {leaf.name for leaf in leaves}
    annotated = []
    for bracket_config in bracket_configs:
        if not bracket_config.get("show", False):
            continue
        column = bracket_config.get("column", "")
        if not column:
            continue
        value_to_tips = {}
        for leaf in leaves:
            value = clean_value(metadata_by_id.get(leaf.name, {}).get(column, ""))
            if not value:
                continue
            value_to_tips.setdefault(value, set()).add(leaf.name)
        min_tips = int(bracket_config.get("min_tips", 2))
        prop_name = f"_bracket_{column}"
        for value, tips in sorted(value_to_tips.items(), key=lambda item: item[0].lower()):
            if len(tips) < min_tips:
                continue
            blocks = []
            for node in tree.traverse():
                node_tips = descendant_leaf_names(node)
                if len(node_tips) >= min_tips and node_tips.issubset(tips):
                    parent = node.up if hasattr(node, "up") else getattr(node, "parent", None)
                    parent_tips = descendant_leaf_names(parent) if parent is not None else leaf_name_set
                    if parent is None or not parent_tips.issubset(tips):
                        blocks.append((node, node_tips))
            if blocks:
                for block_index, (node, node_tips) in enumerate(blocks, start=1):
                    label = value if len(blocks) == 1 else f"{value} ({block_index})"
                    node.add_prop(prop_name, label)
                    annotated.append({"column": column, "value": value, "tips": len(node_tips), "blocks": len(blocks)})
    return annotated


def render_tree(config, dry_run=False):
    tree_path = Path(config["tree"]).expanduser()
    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")
    metadata_paths = metadata_paths_from_config(config, tree_path=tree_path)
    id_column = config.get("id_column", "tree_id")
    metadata = load_metadata(metadata_paths, id_column)
    metadata = annotate_best_representatives(metadata, config.get("best_representatives", {}), id_column)
    metadata_by_id = metadata.set_index(id_column).to_dict("index") if not metadata.empty else {}
    output_prefix = clean_value(config.get("output_prefix")) or str(tree_path.with_suffix(""))
    outputs = [Path(output_prefix).expanduser().with_suffix("." + str(output_format).lstrip(".")) for output_format in config.get("output_formats", ["svg"])]
    if dry_run:
        return outputs, metadata_paths, "dry-run"

    ete = load_ete()
    try:
        tree = load_tree_with_parser(ete, tree_path, config.get("newick_format", 0))
    except Exception:
        if int(config.get("newick_format", 0)) == 0:
            tree = load_tree_with_parser(ete, tree_path, 1)
        else:
            tree = load_tree_with_parser(ete, tree_path, 0)
    rooting_label = apply_rooting(tree, config)
    log(f"[info] rooting: {rooting_label}")

    if config.get("ladderize", True):
        tree.ladderize()

    branch_color_annotations, branch_color_legends = annotate_branch_colors(tree, metadata_by_id, config.get("branch_colors", []))
    if branch_color_annotations:
        log(f"[info] branch colors: {len(branch_color_annotations)} branch(es)")
    bracket_annotations = annotate_clade_brackets(tree, metadata_by_id, config.get("clade_brackets", []))
    if bracket_annotations:
        log(f"[info] clade brackets: {len(bracket_annotations)} block(s)")

    strip_configs = config.get("metadata_strips", [])
    strip_color_maps = {}
    used_strip_colors = set()
    for strip_index, strip in enumerate(strip_configs):
        column = strip.get("column", "")
        strip_color_maps[column] = palette_lookup(
            metadata[column].tolist() if column in metadata.columns else [],
            strip.get("palette", "default"),
            used_colors=used_strip_colors,
            palette_offset=strip_index * 5,
        )
    text_configs = resolve_text_column_widths(config.get("text_columns", []), metadata)
    heatmap_configs = resolve_heatmap_columns(config.get("heatmap_columns", []), metadata)
    spacing_config = config.get("spacing", {})

    support_config = config.get("support", {})
    support_min = support_config.get("min_value", None)
    if support_min is not None:
        support_min = float(support_min)

    def layout(node):
        branch_color = None
        for branch_color_config in config.get("branch_colors", []):
            if branch_color_config.get("show", False):
                branch_color = node.get_prop(f"_branch_color_{branch_color_config.get('column', '')}") if hasattr(node, "get_prop") else None
                if branch_color:
                    break
        set_branch_style(ete, node, config.get("branch_width", 1), color=branch_color)
        if is_leaf_node(node):
            row = metadata_by_id.get(node.name, {})
            column_index = 0
            if config.get("show_leaf_names", True):
                label_column = config.get("leaf_name_column", "tree_id")
                label = row.get(label_column, node.name) if label_column != "tree_id" else node.name
                add_face(ete, make_text_face(ete, label, config.get("leaf_name_font_size", 7)), node, column_index, "aligned")
                column_index += 1
                column_index = add_gap_face(ete, node, column_index, spacing_config.get("after_leaf_name", 8), position="aligned")
            column_index = add_gap_face(ete, node, column_index, spacing_config.get("before_strips", 16), position="aligned")
            for strip in strip_configs:
                column = strip.get("column", "")
                value = clean_value(row.get(column, ""))
                color = strip_color_maps.get(column, {}).get(value, "#D9D9D9")
                add_face(ete, make_rect_face(ete, strip.get("width", 12), strip.get("height", 10), color), node, column_index, "aligned")
                column_index += 1
                column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_strips", 4), position="aligned")
            if strip_configs and (text_configs or heatmap_configs):
                column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_strips_and_text", 12), position="aligned")
            if heatmap_configs:
                for heatmap_config in heatmap_configs:
                    column = heatmap_config.get("column", "")
                    value = row.get(column, "")
                    value = heatmap_numeric_value(value, heatmap_config)
                    color = grayscale_hex(
                        value,
                        heatmap_config.get("min_value", 0),
                        heatmap_config.get("max_value", 1),
                        reverse=bool(heatmap_config.get("reverse", False)),
                        missing_color=heatmap_config.get("missing_color", "#F2F2F2"),
                    )
                    add_face(
                        ete,
                        make_rect_face(
                            ete,
                            heatmap_config.get("width", 8),
                            heatmap_config.get("height", 10),
                            color,
                        ),
                        node,
                        column_index,
                        "aligned",
                    )
                    column_index += 1
                    column_index = add_gap_face(ete, node, column_index, heatmap_config.get("gap_after", 1), position="aligned")
                if text_configs:
                    column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_heatmap_and_text", 8), position="aligned")
            for bar in config.get("numeric_bars", []):
                value = row.get(bar.get("column", ""), "")
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = 0.0
                max_value = float(bar.get("max_value", max(numeric_value, 1.0)))
                width = max(1, int(float(bar.get("width", 40)) * max(0.0, min(numeric_value / max_value, 1.0))))
                add_face(ete, make_rect_face(ete, width, bar.get("height", 8), bar.get("color", "#777777")), node, column_index, "aligned")
                column_index += 1
            for text_index, text_config in enumerate(text_configs):
                value = truncate_text(row.get(text_config.get("column", ""), ""), text_config.get("max_length", 30))
                if text_config.get("mark_best_representative", False):
                    best_column = clean_value(text_config.get("best_representative_column", "best_representative"))
                    best_value = clean_value(row.get(best_column, "")).lower()
                    if best_value in {"true", "1", "yes", "y"}:
                        marker = clean_value(
                            text_config.get(
                                "best_representative_marker",
                                config.get("best_representatives", {}).get("marker", "*"),
                            )
                        ) or "*"
                        value = f"{value}{marker}"
                face = make_text_face(ete, value, text_config.get("font_size", 6), text_config.get("color", "#333333"))
                apply_text_alignment(face, value, text_config)
                apply_face_width(face, text_config.get("width"))
                add_face(ete, face, node, column_index, "aligned")
                column_index += 1
                if text_index + 1 < len(text_configs):
                    column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_text_columns", 8), position="aligned")
        elif support_config.get("show", True):
            value = support_value(node)
            if value is not None and (support_min is None or value >= support_min):
                label = format_support(value, support_config.get("scale", "auto"))
                if label:
                    add_face(
                        ete,
                        make_text_face(ete, label, support_config.get("font_size", 6), support_config.get("color", "#555555")),
                        node,
                        0,
                        support_config.get("position", "branch-top"),
                    )
        for bracket_config in config.get("clade_brackets", []):
            if not bracket_config.get("show", False):
                continue
            column = bracket_config.get("column", "")
            prop_name = f"_bracket_{column}"
            if hasattr(node, "get_prop"):
                label = clean_value(node.get_prop(prop_name))
            else:
                label = clean_value(getattr(node, prop_name, ""))
            if not label:
                continue
            face = make_text_face(
                ete,
                label,
                bracket_config.get("font_size", 7),
                bracket_config.get("color", "#111111"),
            )
            face.margin_left = int(bracket_config.get("margin_left", 6))
            add_face(
                ete,
                face,
                node,
                int(bracket_config.get("column_index", 0)),
                bracket_config.get("position", "branch-right"),
            )

    tree_style = ete["TreeStyle"]()
    tree_style.mode = "c" if config.get("layout", "rectangular").lower().startswith("circ") else "r"
    tree_style.show_leaf_name = False
    tree_style.show_branch_support = False
    tree_style.show_branch_length = False
    tree_style.layout_fn = layout
    canvas_config = config.get("canvas", {})
    tree_style.margin_top = int(canvas_config.get("margin_top", 30))
    tree_style.margin_right = int(canvas_config.get("margin_right", 180))
    tree_style.margin_bottom = int(canvas_config.get("margin_bottom", 30))
    tree_style.margin_left = int(canvas_config.get("margin_left", 30))
    tree_style.show_scale = bool(canvas_config.get("show_scale", True))
    if tree_style.mode == "c":
        tree_style.arc_start = int(canvas_config.get("arc_start", 0))
        tree_style.arc_span = int(canvas_config.get("arc_span", 359))
        add_circular_heatmap_title_labels(ete, tree_style, heatmap_configs, config.get("legend", {}))
    heatmap_label_columns = heatmap_column_indices(strip_configs, heatmap_configs, text_configs, spacing_config)
    add_heatmap_labels(ete, tree_style, heatmap_configs, config.get("legend", {}), heatmap_label_columns)

    if config.get("legend", {}).get("show", True):
        add_legend(
            ete,
            tree_style,
            strip_configs,
            strip_color_maps,
            config.get("legend", {}),
            branch_color_legends=branch_color_legends,
            heatmap_configs=heatmap_configs,
            heatmap_labels_in_legend=(tree_style.mode == "c"),
        )

    pdf_outputs = [path for path in outputs if path.suffix.lower() == ".pdf"]
    png_outputs = [path for path in outputs if path.suffix.lower() == ".png"]
    direct_outputs = [
        path for path in outputs
        if path.suffix.lower() not in {".pdf", ".png"}
    ]
    rendered_svg_path = None
    for out_path in direct_outputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.render(str(out_path), w=int(canvas_config.get("width_mm", 320)), units="mm", dpi=int(canvas_config.get("dpi", 300)), tree_style=tree_style)
        if out_path.suffix.lower() == ".svg":
            align_svg_heatmap_labels(out_path, heatmap_configs, config.get("legend", {}))
            add_svg_heatmap_dividers(out_path, heatmap_configs)
            if rendered_svg_path is None:
                rendered_svg_path = out_path
    temporary_svg_path = None
    if (pdf_outputs or png_outputs) and rendered_svg_path is None:
        handle = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        temporary_svg_path = Path(handle.name)
        handle.close()
        tree.render(
            str(temporary_svg_path),
            w=int(canvas_config.get("width_mm", 320)),
            units="mm",
            dpi=int(canvas_config.get("dpi", 300)),
            tree_style=tree_style,
        )
        align_svg_heatmap_labels(temporary_svg_path, heatmap_configs, config.get("legend", {}))
        add_svg_heatmap_dividers(temporary_svg_path, heatmap_configs)
        rendered_svg_path = temporary_svg_path
    for out_path in pdf_outputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_svg_to_pdf(rendered_svg_path, out_path, dpi=int(canvas_config.get("dpi", 300)))
    for out_path in png_outputs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        convert_svg_to_png(rendered_svg_path, out_path, dpi=int(canvas_config.get("dpi", 300)))
    if temporary_svg_path is not None:
        try:
            temporary_svg_path.unlink()
        except OSError:
            pass
    return outputs, metadata_paths, ete["version"]


def add_legend(
    ete,
    tree_style,
    strip_configs,
    strip_color_maps,
    legend_config,
    branch_color_legends=None,
    heatmap_configs=None,
    heatmap_labels_in_legend=False,
):
    branch_color_legends = branch_color_legends or {}
    heatmap_configs = heatmap_configs or []
    max_items = int(legend_config.get("max_items_per_strip", 20))
    title_size = int(legend_config.get("title_font_size", 8))
    item_size = int(legend_config.get("item_font_size", 6))
    swatch_size = int(legend_config.get("swatch_size", 10))
    swatch_text_gap = int(legend_config.get("swatch_text_gap", 2))
    section_gap = int(legend_config.get("section_gap", max(2, swatch_size // 2)))
    right_padding = int(legend_config.get("right_padding", 0))

    def add_title_row(title):
        title_face = make_text_face(ete, title, title_size, "#222222")
        title_face.margin_top = max(2, swatch_size // 3)
        title_face.margin_bottom = 1
        tree_style.legend.add_face(make_blank_face(ete, 1, swatch_size), column=0)
        tree_style.legend.add_face(title_face, column=1)

    def add_legend_row(swatch_face, label_face=None):
        swatch_face.margin_right = swatch_text_gap
        if label_face is not None:
            label_face.margin_left = 0
            label_face.margin_top = 0
            label_face.margin_bottom = 0
        tree_style.legend.add_face(swatch_face, column=0)
        tree_style.legend.add_face(label_face if label_face is not None else make_blank_face(ete, 1, swatch_size), column=1)
        if right_padding > 0:
            tree_style.legend.add_face(make_blank_face(ete, right_padding, 1), column=2)

    try:
        legend_sections = []
        for column, legend in branch_color_legends.items():
            legend_sections.append((legend.get("label", column), legend.get("colors", {}), "line"))
        for strip in strip_configs:
            column = strip.get("column", "")
            legend_sections.append((strip.get("label", column), strip_color_maps.get(column, {}), "box"))
        for label, color_map, legend_type in legend_sections:
            if not color_map:
                continue
            add_title_row(label)
            shown = 0
            entries = [(value, color) for value, color in sorted(color_map.items(), key=lambda item: item[0].lower()) if value != ""]
            for value, color in entries:
                if value == "":
                    continue
                if shown >= max_items:
                    add_legend_row(
                        make_blank_face(ete, swatch_size, swatch_size),
                        make_text_face(ete, f"... {len(entries) - shown} more", item_size, "#555555"),
                    )
                    break
                if legend_type == "line":
                    swatch = make_rect_face(ete, max(swatch_size * 2, 16), max(2, swatch_size // 3), color)
                else:
                    swatch = make_rect_face(ete, swatch_size, swatch_size, color)
                add_legend_row(swatch, make_text_face(ete, value, item_size, "#333333"))
                shown += 1
            add_legend_row(make_blank_face(ete, 1, section_gap), make_blank_face(ete, 1, section_gap))
        if heatmap_configs and heatmap_labels_in_legend and legend_config.get("show_heatmap_column_labels", True):
            add_title_row(legend_config.get("heatmap_column_label", "Heatmap columns"))
            for heatmap_config in heatmap_configs:
                label = clean_value(heatmap_config.get("label", heatmap_config.get("column", "")))
                if not label:
                    label = clean_value(heatmap_config.get("column", ""))
                add_legend_row(
                    make_blank_face(ete, swatch_size, swatch_size),
                    make_text_face(ete, label, item_size, "#333333"),
                )
            add_legend_row(make_blank_face(ete, 1, section_gap), make_blank_face(ete, 1, section_gap))
        if heatmap_configs and legend_config.get("show_heatmap_colorbar", True):
            add_title_row(legend_config.get("heatmap_colorbar_label", "Heatmap scale"))
            colorbar_face = make_heatmap_colorbar_face(ete, legend_config)
            if colorbar_face is not None:
                add_legend_row(colorbar_face, make_blank_face(ete, 1, int(legend_config.get("heatmap_colorbar_bar_height", 8))))
                scale_face = make_text_face(ete, "0.0        0.5        1.0", item_size, "#333333")
                add_legend_row(scale_face, make_blank_face(ete, 1, item_size + 2))
            else:
                for value, label in [(1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")]:
                    add_legend_row(
                        make_rect_face(ete, swatch_size, swatch_size, grayscale_hex(value, 0, 1)),
                        make_text_face(ete, label, item_size, "#333333"),
                    )
    except Exception:
        log("[warn] could not attach legend; rendering tree without a legend")


def add_heatmap_labels(ete, tree_style, heatmap_configs, legend_config, label_columns=None):
    if not heatmap_configs:
        return
    label_size = int(legend_config.get("heatmap_label_font_size", 5))
    label_columns = label_columns or list(range(len(heatmap_configs)))
    try:
        for _index, heatmap_config in enumerate(heatmap_configs):
            label = clean_value(heatmap_config.get("label", heatmap_config.get("column", "")))
            if not label:
                label = clean_value(heatmap_config.get("column", ""))
            face = make_text_face(ete, label, label_size, heatmap_config.get("label_color", "#333333"))
            face.rotation = int(heatmap_config.get("label_rotation", -90))
            face.rotable = True
            face.hz_align = int(heatmap_config.get("label_hz_align", 1))
            face.vt_align = int(heatmap_config.get("label_vt_align", 1))
            face.margin_top = int(heatmap_config.get("label_margin_top", 4))
            face.margin_left = int(heatmap_config.get("label_margin_left", 0))
            face.margin_right = int(heatmap_config.get("label_margin_right", 0))
            tree_style.aligned_foot.add_face(face, column=int(heatmap_config.get("label_column", label_columns[_index])))
    except Exception:
        log("[warn] could not attach heatmap labels")


def add_circular_heatmap_title_labels(ete, tree_style, heatmap_configs, legend_config):
    if not heatmap_configs or not legend_config.get("show_heatmap_column_labels", True):
        return
    labels = []
    for heatmap_config in heatmap_configs:
        label = clean_value(heatmap_config.get("label", heatmap_config.get("column", "")))
        if not label:
            label = clean_value(heatmap_config.get("column", ""))
        if label:
            labels.append(label)
    if not labels:
        return
    title = legend_config.get("heatmap_column_label", "Heatmap columns")
    text = f"{title}: " + " | ".join(labels)
    try:
        face = make_text_face(ete, text, int(legend_config.get("title_font_size", 8)), "#222222")
        face.margin_bottom = int(legend_config.get("heatmap_title_margin_bottom", 8))
        tree_style.title.add_face(face, column=0)
    except Exception:
        log("[warn] could not attach circular heatmap title labels")


def discover_trees(root):
    root = Path(root).expanduser()
    patterns = ["*_tree.nwk", "*.tree", "*.nwk", "*.newick"]
    paths = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    unique = []
    seen = set()
    for path in sorted(paths):
        if path.stat().st_size <= 0:
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def command_init_config(args):
    config = deep_copy_config(DEFAULT_CONFIG)
    if args.tree:
        tree_path = Path(args.tree).expanduser()
        config["tree"] = str(tree_path)
        config["metadata"] = [str(path) for path in metadata_paths_from_config(config, tree_path=tree_path)]
        config["output_prefix"] = str(tree_path.with_suffix("")) + "_ete"
    if args.metadata:
        config["metadata"] = args.metadata
    if args.output_prefix:
        config["output_prefix"] = args.output_prefix
    write_config(args.output, config)
    print(args.output)


def command_render(args):
    config = read_config(args.config)
    if args.tree:
        config["tree"] = args.tree
    if args.metadata:
        config["metadata"] = args.metadata
    if args.output_prefix:
        config["output_prefix"] = args.output_prefix
    if args.formats:
        config["output_formats"] = args.formats
    outputs, metadata_paths, ete_version = render_tree(config, dry_run=args.dry_run)
    log(f"[info] renderer: {ete_version}")
    log(f"[info] metadata files: {len(metadata_paths)}")
    for path in outputs:
        print(path)


def command_batch(args):
    base_config = read_config(args.config)
    trees = discover_trees(args.root)
    if args.limit:
        trees = trees[: int(args.limit)]
    if not trees:
        raise FileNotFoundError(f"No non-empty tree files found under {args.root}")
    wrote = []
    for tree_path in trees:
        config = deep_copy_config(base_config)
        config["tree"] = str(tree_path)
        config["metadata"] = [str(path) for path in metadata_paths_from_config(config, tree_path=tree_path)]
        out_dir = Path(args.output_dir).expanduser() if args.output_dir else tree_path.parent
        config["output_prefix"] = str(out_dir / f"{tree_path.stem}_ete")
        outputs, _metadata_paths, _ete_version = render_tree(config, dry_run=args.dry_run)
        wrote.extend(outputs)
    for path in wrote:
        print(path)


def command_convert_config(args):
    config = read_config(args.input)
    write_config(args.output, config)
    print(args.output)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render reproducible, metadata-annotated phylogeny figures with ETE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-config", help="Write a reusable JSON rendering config.")
    init_parser.add_argument("-o", "--output", default="phylogeny_ete_config.json", help="Output JSON config path.")
    init_parser.add_argument("--tree", default="", help="Optional tree path to seed autodiscovered metadata/output paths.")
    init_parser.add_argument("--metadata", nargs="*", default=None, help="Optional metadata TSV paths.")
    init_parser.add_argument("--output-prefix", default="", help="Optional output prefix for rendered figures.")
    init_parser.set_defaults(func=command_init_config)

    render_parser = subparsers.add_parser("render", help="Render one tree from a JSON config.")
    render_parser.add_argument("-c", "--config", required=True, help="JSON rendering config.")
    render_parser.add_argument("--tree", default="", help="Override tree path from config.")
    render_parser.add_argument("--metadata", nargs="*", default=None, help="Override metadata TSV paths from config.")
    render_parser.add_argument("--output-prefix", default="", help="Override output prefix from config.")
    render_parser.add_argument("--formats", nargs="+", default=None, help="Override output formats, e.g. svg pdf png.")
    render_parser.add_argument("--dry-run", action="store_true", help="Print intended output paths without rendering.")
    render_parser.set_defaults(func=command_render)

    batch_parser = subparsers.add_parser("batch", help="Render every non-empty Newick tree under a root directory.")
    batch_parser.add_argument("root", help="Root directory to search for trees.")
    batch_parser.add_argument("-c", "--config", required=True, help="Base JSON rendering config.")
    batch_parser.add_argument("-o", "--output-dir", default="", help="Optional directory for rendered outputs.")
    batch_parser.add_argument("--limit", type=int, default=0, help="Render only the first N discovered trees.")
    batch_parser.add_argument("--dry-run", action="store_true", help="Print intended output paths without rendering.")
    batch_parser.set_defaults(func=command_batch)

    convert_parser = subparsers.add_parser("convert-config", help="Convert JSON/YAML render configs while preserving renderer defaults.")
    convert_parser.add_argument("input", help="Input JSON or YAML config.")
    convert_parser.add_argument("-o", "--output", required=True, help="Output JSON/YAML config path.")
    convert_parser.set_defaults(func=command_convert_config)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
