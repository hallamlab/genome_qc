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
DEFAULT_FONT_FAMILY = "Times New Roman"

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
# - Add HQ/MQ/LQ x category matrix: enable quality_category_matrix with a component table.
#
# Key sections:
# tree / metadata: input files and metadata tables.
# rooting: display rooting before render; midpoint is automated, outgroup requires outgroup_id.
# support: FastTree local support labels; these are SH-like supports, not classic bootstraps.
# metadata_strips: colored blocks beside leaves, with optional custom palettes.
# branch_colors: colors pure branches by a metadata column; mixed ancestral branches can stay black.
# best_representatives: optional table used to append a marker to selected leaf labels.
# quality_category_matrix: optional grouped Qscore matrix by quality tier and category.
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
    "quality_category_matrix": {
        "_help": "Optional grouped matrix heatmap. Rows are tree leaves; columns are quality tiers x categories. Values are drawn from a table matched by component/species-cluster ID.",
        "show": False,
        "table": "",
        "leaf_component_column": "component_id",
        "leaf_match_columns": ["sample", "category", "Genome_Id"],
        "table_component_column": "component_id",
        "quality_column": "mimag_tier",
        "category_column": "category",
        "value_column": "qscore",
        "selection": "best",
        "auto_disambiguate_match_columns": True,
        "quality_order": ["high", "medium", "low"],
        "quality_labels": {"high": "HQ", "medium": "MQ", "low": "LQ"},
        "categories": ["SAGs", "xPG_SAGs", "MAGs", "xPG_MAGs"],
        "min_value": 0,
        "max_value": 100,
        "width": 8,
        "height": 10,
        "gap_after": 1,
        "tier_gap_after": 4,
        "missing_color": "#FFFFFF",
        "missing_border": True,
        "missing_border_color": "#BDBDBD",
        "missing_border_width": 0.45,
        "missing_border_dash": "0.35,1.4",
        "label_font_size": 5,
        "label_rotation": -90,
        "tiered_labels": True,
        "svg_header_labels": True,
        "tier_label_font_size": 7,
        "category_label_font_size": 5,
        "category_label_rotation": -90,
        "category_label_margin_bottom": 28,
    },
    "_help_quality_category_matrix": "Use to draw HQ/MQ/LQ x category Qscore presence matrices beside each tree leaf.",
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
    "numeric_dots": [],
    "_help_numeric_dots": "Optional aligned numeric dots. Empty list disables them.",
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
        "branch_vertical_margin": 0,
    },
}


def log(message):
    print(message, file=sys.stderr, flush=True)


def load_ete():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from ete4 import Tree
        from ete4.treeview import NodeStyle, RectFace, StackedBarFace, StaticItemFace, TextFace, TreeStyle

        return {
            "version": "ete4",
            "Tree": Tree,
            "TreeStyle": TreeStyle,
            "TextFace": TextFace,
            "RectFace": RectFace,
            "StackedBarFace": StackedBarFace,
            "StaticItemFace": StaticItemFace,
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
        try:
            from ete3 import StaticItemFace
        except ImportError:
            StaticItemFace = None

        return {
            "version": "ete3",
            "Tree": Tree,
            "TreeStyle": TreeStyle,
            "TextFace": TextFace,
            "RectFace": RectFace,
            "StackedBarFace": StackedBarFace,
            "StaticItemFace": StaticItemFace,
            "NodeStyle": NodeStyle,
        }
    except ImportError as exc:
        raise ImportError(
            "ETE is required for rendering. Install ete4 if possible, or ete3 as a fallback."
        ) from exc


def set_qt_default_font():
    try:
        from PyQt6 import QtGui, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtGui, QtWidgets
        except ImportError:
            return
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    app.setFont(QtGui.QFont(DEFAULT_FONT_FAMILY))


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


def sanitize_token(value):
    text = clean_value(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


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
    face = ete["TextFace"](str(text), ftype=DEFAULT_FONT_FAMILY, fsize=int(size), fgcolor=color)
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


def make_static_rect_face(
    ete,
    width,
    height,
    color,
    border_color=None,
    border_width=0,
    border_style="solid",
    dash_pattern=None,
    fill=True,
):
    StaticItemFace = ete.get("StaticItemFace")
    if StaticItemFace is None:
        return make_rect_face(ete, width, height, color)
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError:
            return make_rect_face(ete, width, height, color)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    width = int(width)
    height = int(height)
    container = QtWidgets.QGraphicsRectItem(0, 0, width, height)
    container.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    container.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    if border_color and border_width:
        inset = max(0.5, float(border_width) / 2.0)
        rect = QtWidgets.QGraphicsRectItem(
            inset,
            inset,
            max(0.0, float(width) - (2.0 * inset)),
            max(0.0, float(height) - (2.0 * inset)),
            container,
        )
        pen = QtGui.QPen(QtGui.QColor(border_color))
        pen.setWidthF(float(border_width))
        if str(border_style).lower() == "dash":
            if dash_pattern:
                try:
                    pen.setDashPattern([float(value) for value in str(dash_pattern).split(",") if str(value).strip()])
                except ValueError:
                    pen.setStyle(QtCore.Qt.PenStyle.DotLine)
            else:
                pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        rect.setPen(pen)
    else:
        rect = QtWidgets.QGraphicsRectItem(0, 0, width, height, container)
        rect.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    if fill:
        rect.setBrush(QtGui.QBrush(QtGui.QColor(color)))
    else:
        rect.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    return StaticItemFace(container)


def make_static_circle_face(
    ete,
    diameter,
    color,
    border_color="#111111",
    border_width=0,
    fill=True,
    canvas_size=None,
):
    StaticItemFace = ete.get("StaticItemFace")
    if StaticItemFace is None:
        return make_rect_face(ete, diameter, diameter, color)
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError:
            return make_rect_face(ete, diameter, diameter, color)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    diameter = max(0.0, float(diameter))
    canvas_size = max(diameter, float(canvas_size if canvas_size is not None else diameter))
    container = QtWidgets.QGraphicsRectItem(0, 0, canvas_size, canvas_size)
    container.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    container.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.transparent))
    offset = (canvas_size - diameter) / 2.0
    item = QtWidgets.QGraphicsEllipseItem(offset, offset, diameter, diameter, container)
    pen = QtGui.QPen(QtGui.QColor(border_color))
    pen.setWidthF(float(border_width))
    if border_width <= 0:
        pen.setStyle(QtCore.Qt.PenStyle.NoPen)
    item.setPen(pen)
    item.setBrush(QtGui.QBrush(QtGui.QColor(color)) if fill else QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    return StaticItemFace(container)


def make_heatmap_colorbar_face(ete, legend_config):
    StackedBarFace = ete.get("StackedBarFace")
    if StackedBarFace is None:
        return None
    width = int(legend_config.get("heatmap_colorbar_width", 90))
    height = int(legend_config.get("heatmap_colorbar_bar_height", 8))
    steps = max(8, int(legend_config.get("heatmap_colorbar_steps", 32)))
    percents = [100.0 / steps] * steps
    colors = [
        grayscale_hex(
            index / (steps - 1),
            0,
            1,
            lightest_shade=legend_config.get("heatmap_colorbar_lightest_shade", 225),
            darkest_shade=legend_config.get("heatmap_colorbar_darkest_shade", 20),
        )
        for index in range(steps)
    ]
    return StackedBarFace(percents, width=width, height=height, colors=colors, line_color="#333333")


def make_heatmap_colorbar_scale_face(ete, legend_config, item_size):
    StaticItemFace = ete.get("StaticItemFace")
    if StaticItemFace is None:
        return None
    width = int(legend_config.get("heatmap_colorbar_width", 90))
    font_size = int(legend_config.get("heatmap_colorbar_tick_font_size", item_size))
    height = int(legend_config.get("heatmap_colorbar_tick_height", font_size + 6))
    tick_height = int(legend_config.get("heatmap_colorbar_tick_mark_height", 3))
    label_color = str(legend_config.get("heatmap_colorbar_tick_color", "#333333"))
    labels = legend_config.get("heatmap_colorbar_ticks", ["0.0", "0.5", "1.0"])
    positions = legend_config.get("heatmap_colorbar_tick_positions", [0.0, 0.5, 1.0])
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError:
            return None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    container = QtWidgets.QGraphicsRectItem(0, 0, width, height)
    container.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    container.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    font = QtGui.QFont(DEFAULT_FONT_FAMILY)
    font.setPointSize(font_size)
    pen = QtGui.QPen(QtGui.QColor(label_color))
    for label, position in zip(labels, positions):
        try:
            fraction = min(1.0, max(0.0, float(position)))
        except Exception:
            continue
        x = fraction * float(width)
        tick = QtWidgets.QGraphicsLineItem(x, 0, x, tick_height, container)
        tick.setPen(pen)
        text_item = QtWidgets.QGraphicsSimpleTextItem(str(label), container)
        text_item.setFont(font)
        text_item.setBrush(QtGui.QBrush(QtGui.QColor(label_color)))
        label_width = float(text_item.boundingRect().width())
        if fraction <= 0.0:
            label_x = 0.0
        elif fraction >= 1.0:
            label_x = float(width) - label_width
        else:
            label_x = x - (label_width / 2.0)
        text_item.setPos(label_x, tick_height)
    return StaticItemFace(container)


def make_vertical_heatmap_colorbar_face(ete, legend_config, item_size):
    StaticItemFace = ete.get("StaticItemFace")
    if StaticItemFace is None:
        return None
    bar_width = int(legend_config.get("heatmap_colorbar_vertical_bar_width", legend_config.get("heatmap_colorbar_bar_height", 8)))
    bar_height = int(legend_config.get("heatmap_colorbar_vertical_height", legend_config.get("heatmap_colorbar_width", 90)))
    steps = max(8, int(legend_config.get("heatmap_colorbar_steps", 32)))
    font_size = int(legend_config.get("heatmap_colorbar_tick_font_size", item_size))
    tick_length = int(legend_config.get("heatmap_colorbar_tick_mark_height", 3))
    label_gap = int(legend_config.get("heatmap_colorbar_tick_label_gap", 3))
    label_color = str(legend_config.get("heatmap_colorbar_tick_color", "#333333"))
    labels = legend_config.get("heatmap_colorbar_ticks", ["0.0", "0.5", "1.0"])
    positions = legend_config.get("heatmap_colorbar_tick_positions", [0.0, 0.5, 1.0])
    reverse_vertical = bool(legend_config.get("heatmap_colorbar_vertical_reverse", False))
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError:
        try:
            from PyQt5 import QtCore, QtGui, QtWidgets
        except ImportError:
            return None

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    font = QtGui.QFont(DEFAULT_FONT_FAMILY)
    font.setPointSize(font_size)
    label_widths = []
    for label in labels:
        item = QtWidgets.QGraphicsSimpleTextItem(str(label))
        item.setFont(font)
        label_widths.append(float(item.boundingRect().width()))
    max_label_width = max(label_widths or [0.0])
    width = bar_width + tick_length + label_gap + int(math.ceil(max_label_width))
    height = bar_height
    container = QtWidgets.QGraphicsRectItem(0, 0, width, height)
    container.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    container.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))

    gradient = QtGui.QLinearGradient(0, 0, 0, bar_height)
    if reverse_vertical:
        gradient.setColorAt(0.0, QtGui.QColor(grayscale_hex(0, 0, 1)))
        gradient.setColorAt(1.0, QtGui.QColor(grayscale_hex(1, 0, 1)))
    else:
        gradient.setColorAt(0.0, QtGui.QColor(grayscale_hex(1, 0, 1)))
        gradient.setColorAt(1.0, QtGui.QColor(grayscale_hex(0, 0, 1)))
    bar = QtWidgets.QGraphicsRectItem(0, 0, bar_width, bar_height, container)
    bar.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
    bar.setBrush(QtGui.QBrush(gradient))

    outline = QtWidgets.QGraphicsRectItem(0, 0, bar_width, bar_height, container)
    outline.setPen(QtGui.QPen(QtGui.QColor(label_color)))
    outline.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))
    pen = QtGui.QPen(QtGui.QColor(label_color))
    for label, position in zip(labels, positions):
        try:
            fraction = min(1.0, max(0.0, float(position)))
        except Exception:
            continue
        y = (1.0 - fraction if not reverse_vertical else fraction) * float(bar_height)
        tick = QtWidgets.QGraphicsLineItem(bar_width, y, bar_width + tick_length, y, container)
        tick.setPen(pen)
        text_item = QtWidgets.QGraphicsSimpleTextItem(str(label), container)
        text_item.setFont(font)
        text_item.setBrush(QtGui.QBrush(QtGui.QColor(label_color)))
        label_height = float(text_item.boundingRect().height())
        label_y = min(max(0.0, y - (label_height / 2.0)), max(0.0, float(bar_height) - label_height))
        text_item.setPos(bar_width + tick_length + label_gap, label_y)
    return StaticItemFace(container)


def svg_size_mm(svg_path):
    text = Path(svg_path).read_text(errors="ignore")[:1000]
    match = re.search(r"<svg[^>]*\bwidth=\"([0-9.]+)mm\"[^>]*\bheight=\"([0-9.]+)mm\"", text)
    if not match:
        raise ValueError(f"Could not read SVG millimeter dimensions from {svg_path}")
    return float(match.group(1)), float(match.group(2))


def normalize_svg_font_family(svg_path, font_family=DEFAULT_FONT_FAMILY):
    path = Path(svg_path)
    text = path.read_text(errors="ignore")
    text = re.sub(r'font-family="[^"]*"', f'font-family="{font_family}"', text)
    text = re.sub(r"font-family='[^']*'", f"font-family='{font_family}'", text)
    text = re.sub(r"font-family:[^;\"']+", f"font-family:{font_family}", text)
    path.write_text(text)


def style_svg_guiding_lines(svg_path, guiding_config):
    if not guiding_config.get("show", False):
        return
    path = Path(svg_path)
    text = path.read_text(errors="ignore")
    color = str(guiding_config.get("color", "#000000"))
    width = str(guiding_config.get("width", 0.35))
    dasharray = str(guiding_config.get("dasharray", "0.8,1.1"))

    def replace_guiding_group(match):
        group = match.group(0)
        if 'stroke-dasharray="' not in group:
            return group
        group = re.sub(r'stroke="#[0-9A-Fa-f]{6}"', f'stroke="{color}"', group, count=1)
        group = re.sub(r'stroke-width="[^"]+"', f'stroke-width="{width}"', group, count=1)
        group = re.sub(r'stroke-dasharray="[^"]+"', f'stroke-dasharray="{dasharray}"', group, count=1)
        return group

    text = re.sub(r'<g\b[^>]*stroke-dasharray="[^"]+"[^>]*>.*?</g>', replace_guiding_group, text)
    path.write_text(text)


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
        expected_labels = set()
        category_labels = set()
        for column in visible_heatmaps:
            label = clean_value(column.get("label", column.get("column", ""))) or clean_value(column.get("column", ""))
            if label:
                expected_labels.add(label)
            category_label = clean_value(column.get("category_label", ""))
            if category_label:
                expected_labels.add(category_label)
                category_labels.add(category_label)

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
                labels.append((current_x, group, matrix, text_x, text_y, text in category_labels))

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

        for target_center, (_current_x, group, matrix, text_x, text_y, is_category_label) in zip(repeated_centers, labels):
            a, b, c, d, _e, f = matrix
            label_center_fraction = 1.0 if is_category_label else visual_center_fraction
            visual_center_y = text_y * label_center_fraction
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


def add_svg_quality_matrix_labels(svg_path, matrix_columns):
    visible_columns = [column for column in matrix_columns if column.get("show", True)]
    if not visible_columns or not any(column.get("tiered_labels", True) for column in visible_columns):
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
            for column in visible_columns
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

        x_centers = sorted(center for center, count in centers.items() if count > 1)
        if len(x_centers) < len(visible_columns) or not y_values:
            return
        x_centers = x_centers[: len(visible_columns)]
        top_y = min(y_values)

        first = visible_columns[0]
        category_font_size = int(first.get("category_label_font_size", first.get("label_font_size", 5))) * 4
        tier_font_size = int(first.get("tier_label_font_size", 7)) * 4
        label_color = clean_value(first.get("label_color", "#333333")) or "#333333"
        category_gap = float(first.get("category_label_svg_gap", 12))
        tier_gap = float(first.get("tier_label_svg_gap", 18))
        category_y = top_y - category_gap
        max_category_label_height = max(
            (
                len(clean_value(column.get("category_label", column.get("category", ""))))
                * category_font_size
                * TEXT_WIDTH_FACTOR
                for column in visible_columns
            ),
            default=0,
        )

        label_root = ET.SubElement(root, namespace + "g", {"id": "quality-category-matrix-svg-labels"})
        for center_x, column in zip(x_centers, visible_columns):
            label = clean_value(column.get("category_label", column.get("category", "")))
            if not label:
                continue
            text = ET.SubElement(
                label_root,
                namespace + "text",
                {
                    "x": f"{center_x:.3f}",
                    "y": f"{category_y:.3f}",
                    "transform": f"rotate(-90 {center_x:.3f} {category_y:.3f})",
                    "font-size": str(category_font_size),
                    "font-family": DEFAULT_FONT_FAMILY,
                    "fill": label_color,
                    "text-anchor": "start",
                    "dominant-baseline": "middle",
                },
            )
            text.text = label

        quality_to_indices = {}
        for index, column in enumerate(visible_columns):
            quality = clean_value(column.get("quality", ""))
            quality_to_indices.setdefault(quality, []).append(index)
        # Keep the quality tier labels in a separate header row above the
        # rotated category labels. The tier row accounts for the estimated
        # height of the longest rotated category label so the two header rows
        # stay paired without colliding.
        tier_y = category_y - max_category_label_height - tier_gap
        for quality, indices in quality_to_indices.items():
            if not quality or not indices:
                continue
            center_x = sum(x_centers[index] for index in indices) / float(len(indices))
            label = clean_value(visible_columns[indices[0]].get("quality_label", quality.upper()))
            text = ET.SubElement(
                label_root,
                namespace + "text",
                {
                    "x": f"{center_x:.3f}",
                    "y": f"{tier_y:.3f}",
                    "font-size": str(tier_font_size),
                    "font-family": DEFAULT_FONT_FAMILY,
                    "fill": label_color,
                    "text-anchor": "middle",
                    "dominant-baseline": "middle",
                },
            )
            text.text = label

        svg_path.write_text(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
            + ET.tostring(root, encoding="unicode")
        )
    except Exception:
        log("[warn] could not add SVG quality-category matrix labels")


def add_gap_face(ete, node, column, width, height=1, position="aligned"):
    if width and int(width) > 0:
        add_face(ete, make_blank_face(ete, int(width), int(height)), node, column, position)
        return column + 1
    return column


def estimate_text_width(text, font_size):
    return int(math.ceil(len(clean_value(text)) * float(font_size) * TEXT_WIDTH_FACTOR)) + 8


def grayscale_hex(
    value,
    min_value,
    max_value,
    reverse=False,
    missing_color="#F2F2F2",
    lightest_shade=250,
    darkest_shade=20,
):
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
    lightest_shade = int(lightest_shade)
    darkest_shade = int(darkest_shade)
    shade = int(round(lightest_shade - (fraction * (lightest_shade - darkest_shade))))
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


def text_before_heatmap(config):
    value = clean_value(config.get("aligned_order", "")).lower().replace("-", "_")
    return value in {"text_before_heatmap", "taxonomy_before_heatmap", "taxa_before_heatmap"}


def text_column_count(text_configs, spacing_config):
    if not text_configs:
        return 0
    count = len(text_configs)
    if int(spacing_config.get("between_text_columns", 0)) > 0:
        count += max(0, len(text_configs) - 1)
    return count


def numeric_dot_column_count(numeric_dot_configs):
    count = 0
    for dot_config in numeric_dot_configs or []:
        if not dot_config.get("show", True):
            continue
        count += 1
        if int(dot_config.get("gap_after", 0)) > 0:
            count += 1
    return count


def heatmap_start_column(
    strip_configs,
    heatmap_configs,
    text_configs,
    spacing_config,
    matrix_configs=None,
    text_first=False,
    numeric_dot_configs=None,
):
    column_index = 0
    if int(spacing_config.get("before_strips", 0)) > 0:
        column_index += 1
    for strip in strip_configs:
        column_index += 1
        if int(spacing_config.get("between_strips", 0)) > 0:
            column_index += 1
    matrix_configs = matrix_configs or []
    if strip_configs and (text_configs or heatmap_configs or matrix_configs) and int(spacing_config.get("between_strips_and_text", 0)) > 0:
        column_index += 1
    column_index += numeric_dot_column_count(numeric_dot_configs)
    if text_first and text_configs:
        column_index += text_column_count(text_configs, spacing_config)
        if (heatmap_configs or matrix_configs) and int(spacing_config.get("between_heatmap_and_text", 0)) > 0:
            column_index += 1
    return column_index


def heatmap_column_indices(
    strip_configs,
    heatmap_configs,
    text_configs,
    spacing_config,
    matrix_configs=None,
    text_first=False,
    numeric_dot_configs=None,
):
    column_index = heatmap_start_column(
        strip_configs,
        heatmap_configs,
        text_configs,
        spacing_config,
        matrix_configs=matrix_configs,
        text_first=text_first,
        numeric_dot_configs=numeric_dot_configs,
    )
    indices = []
    for heatmap_config in heatmap_configs:
        indices.append(column_index)
        column_index += 1
        if int(heatmap_config.get("gap_after", 0)) > 0:
            column_index += 1
    return indices


def quality_matrix_column_indices(
    strip_configs,
    heatmap_configs,
    matrix_configs,
    text_configs,
    spacing_config,
    text_first=False,
    numeric_dot_configs=None,
):
    column_index = heatmap_start_column(
        strip_configs,
        heatmap_configs,
        text_configs,
        spacing_config,
        matrix_configs=matrix_configs,
        text_first=text_first,
        numeric_dot_configs=numeric_dot_configs,
    )
    for heatmap_config in heatmap_configs:
        column_index += 1
        if int(heatmap_config.get("gap_after", 0)) > 0:
            column_index += 1
    indices = []
    for matrix_config in matrix_configs:
        indices.append(column_index)
        column_index += 1
        if int(matrix_config.get("gap_after", 0)) > 0:
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


def normalize_quality_tier(value):
    text = clean_value(value).lower()
    aliases = {
        "hq": "high",
        "high": "high",
        "high_quality": "high",
        "mq": "medium",
        "medium": "medium",
        "medium_quality": "medium",
        "lq": "low",
        "low": "low",
        "low_quality": "low",
    }
    return aliases.get(text, text)


def infer_tree_id_from_row(row):
    for column in ["tree_id", "copied_fasta_path", "fasta_path", "mp_fasta_path", "ani_fasta_path"]:
        value = clean_value(row.get(column, ""))
        if not value:
            continue
        if column == "tree_id":
            return value
        return fasta_suffixless_name(value)
    genome_id = clean_value(row.get("Genome_Id", row.get("genome_id", row.get("mp_genome_id", ""))))
    sample = clean_value(row.get("sample", ""))
    category = clean_value(row.get("category", ""))
    if genome_id:
        return sanitize_token("__".join(part for part in [sample, category, genome_id] if part))
    return ""


def prepare_quality_category_matrix(matrix_config, metadata, id_column):
    if not matrix_config or not matrix_config.get("show", False):
        return metadata, [], {}
    table_text = clean_value(matrix_config.get("table", ""))
    if not table_text:
        log("[warn] quality_category_matrix.show is true, but no table was provided")
        return metadata, [], {}
    table_path = Path(table_text).expanduser()
    if not table_path.exists():
        log(f"[warn] quality category matrix table not found: {table_path}")
        return metadata, [], {}

    table = pd.read_csv(table_path, sep="\t", dtype=str).fillna("")
    component_column = clean_value(matrix_config.get("table_component_column", "component_id")) or "component_id"
    quality_column = clean_value(matrix_config.get("quality_column", "mimag_tier")) or "mimag_tier"
    category_column = clean_value(matrix_config.get("category_column", "category")) or "category"
    value_column = clean_value(matrix_config.get("value_column", "qscore")) or "qscore"
    required = {component_column, quality_column, category_column, value_column}
    missing = [column for column in required if column not in table.columns]
    if missing:
        log(f"[warn] quality category matrix table missing columns: {', '.join(missing)}")
        return metadata, [], {}

    working = table.copy()
    if "tree_id" not in working.columns:
        working["tree_id"] = [infer_tree_id_from_row(row) for row in working.to_dict("records")]
    working["_component_id"] = working[component_column].astype(str).str.strip()
    working["_quality"] = working[quality_column].map(normalize_quality_tier)
    working["_category"] = working[category_column].astype(str).str.strip()
    working["_value"] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.loc[
        working["_component_id"].ne("")
        & working["_quality"].ne("")
        & working["_category"].ne("")
        & working["_value"].notna()
    ].copy()
    if working.empty:
        log("[warn] quality category matrix table had no usable rows")
        return metadata, [], {}

    configured_match_columns = [
        clean_value(column)
        for column in matrix_config.get("match_columns", [component_column])
        if clean_value(column)
    ]
    if not configured_match_columns:
        configured_match_columns = [component_column]
    missing_match_columns = [
        column
        for column in configured_match_columns
        if column != component_column and column not in working.columns
    ]
    if missing_match_columns:
        log(
            "[warn] quality category matrix match columns missing from table: "
            f"{', '.join(missing_match_columns)}; using component_id only"
        )
        configured_match_columns = [component_column]

    auto_disambiguate = bool(matrix_config.get("auto_disambiguate_match_columns", True))
    if auto_disambiguate:
        # Component IDs can be reused across sample/global scopes. If a configured
        # key merges multiple taxa, add the most specific shared taxonomy columns
        # available in both the matrix table and tree metadata.
        for tax_column in ["Species", "Genus", "Family"]:
            if tax_column in configured_match_columns:
                continue
            if tax_column not in working.columns or tax_column not in metadata.columns:
                continue
            if not configured_match_columns:
                continue
            nonempty_tax = working.loc[working[tax_column].astype(str).str.strip().ne("")].copy()
            if nonempty_tax.empty:
                continue
            tax_counts = (
                nonempty_tax.groupby(configured_match_columns, dropna=False)[tax_column]
                .nunique()
            )
            if int(tax_counts.gt(1).sum()) > 0:
                configured_match_columns.append(tax_column)
                log(
                    "[info] quality-category matrix match key auto-disambiguated with "
                    f"{tax_column}; using {','.join(configured_match_columns)}"
                )

        remaining_tax_collisions = []
        for tax_column in ["Species", "Genus", "Family"]:
            if tax_column not in working.columns:
                continue
            nonempty_tax = working.loc[working[tax_column].astype(str).str.strip().ne("")].copy()
            if nonempty_tax.empty:
                continue
            tax_counts = (
                nonempty_tax.groupby(configured_match_columns, dropna=False)[tax_column]
                .nunique()
            )
            collision_count = int(tax_counts.gt(1).sum())
            if collision_count:
                remaining_tax_collisions.append(f"{tax_column}:{collision_count}")
        if remaining_tax_collisions:
            log(
                "[warn] quality-category matrix match key still groups multiple taxa "
                f"({'; '.join(remaining_tax_collisions)}); consider adding more match_columns"
            )

    def matrix_group_key_from_table(row):
        values = []
        for column in configured_match_columns:
            if column == component_column:
                values.append(row.get("_component_id", ""))
            else:
                values.append(row.get(column, ""))
        return normalized_key(values)

    working["_matrix_group_key"] = working.apply(matrix_group_key_from_table, axis=1)

    quality_order = [
        normalize_quality_tier(value)
        for value in matrix_config.get("quality_order", ["high", "medium", "low"])
        if clean_value(value)
    ]
    category_order_source = matrix_config.get("categories", matrix_config.get("category_order", []))
    category_order_config = [
        clean_value(value)
        for value in category_order_source
        if clean_value(value)
    ]
    if not category_order_config:
        category_order_config = sorted(working["_category"].dropna().unique().tolist(), key=lambda value: value.lower())
    quality_labels = matrix_config.get("quality_labels", {}) or {}
    category_labels = matrix_config.get("category_labels", {}) or {}
    width = int(matrix_config.get("width", 8))
    height = int(matrix_config.get("height", 10))
    gap_after = int(matrix_config.get("gap_after", matrix_config.get("gap_between_categories", 1)))
    tier_gap_after = int(
        matrix_config.get(
            "tier_gap_after",
            matrix_config.get("gap_between_quality_groups", gap_after),
        )
    )

    columns = []
    for quality in quality_order:
        quality_label = clean_value(quality_labels.get(quality, quality.upper()))
        for category_index, category in enumerate(category_order_config):
            is_last_in_tier = category_index == len(category_order_config) - 1
            category_label = clean_value(category_labels.get(category, category))
            columns.append(
                {
                    "quality": quality,
                    "quality_label": quality_label,
                    "category": category,
                    "category_label": category_label,
                    "label": f"{quality_label} {category}",
                    "width": width,
                    "height": height,
                    "gap_after": tier_gap_after if is_last_in_tier else gap_after,
                    "divider_after": is_last_in_tier and quality != quality_order[-1],
                    "min_value": matrix_config.get("min_value", 0),
                    "max_value": matrix_config.get("max_value", 100),
                    "missing_color": matrix_config.get("missing_color", "#FFFFFF"),
                    "missing_border": bool(matrix_config.get("missing_border", True)),
                    "missing_border_color": matrix_config.get("missing_border_color", matrix_config.get("border_color", "#BDBDBD")),
                    "missing_border_width": matrix_config.get("missing_border_width", matrix_config.get("border_width", 1)),
                    "missing_border_dash": matrix_config.get("missing_border_dash", "0.35,1.4"),
                    "tiered_labels": bool(matrix_config.get("tiered_labels", True)),
                    "svg_header_labels": bool(matrix_config.get("svg_header_labels", True)),
                    "tier_label_font_size": matrix_config.get("tier_label_font_size", 7),
                    "tier_label_margin_top": matrix_config.get("tier_label_margin_top", 1),
                    "tier_label_margin_bottom": matrix_config.get("tier_label_margin_bottom", 1),
                    "tier_label_svg_gap": matrix_config.get("tier_label_svg_gap", 18),
                    "category_label_font_size": matrix_config.get("category_label_font_size", matrix_config.get("label_font_size", 5)),
                    "category_label_rotation": matrix_config.get("category_label_rotation", matrix_config.get("label_rotation", -90)),
                    "category_label_margin_top": matrix_config.get("category_label_margin_top", 1),
                    "category_label_margin_bottom": matrix_config.get("category_label_margin_bottom", 28),
                    "category_label_svg_gap": matrix_config.get("category_label_svg_gap", 12),
                    "label_font_size": matrix_config.get("label_font_size", 5),
                    "label_rotation": matrix_config.get("label_rotation", -90),
                    "label_color": matrix_config.get("label_color", "#333333"),
                }
            )

    # If tree metadata does not already carry the component ID, borrow it from
    # the same table using the tree_id generated for the representative genome.
    leaf_component_column = clean_value(matrix_config.get("leaf_component_column", "component_id")) or "component_id"
    if leaf_component_column not in metadata.columns or metadata[leaf_component_column].astype(str).str.strip().eq("").all():
        metadata = metadata.copy()
        leaf_match_columns = [
            clean_value(column)
            for column in matrix_config.get("leaf_match_columns", ["sample", "category", "Genome_Id"])
            if clean_value(column)
        ]
        if leaf_match_columns and all(column in metadata.columns for column in leaf_match_columns) and all(column in working.columns for column in leaf_match_columns):
            table_keys = working.loc[:, leaf_match_columns].apply(lambda row: normalized_key(row.tolist()), axis=1)
            key_to_component = {}
            for key, component_id in zip(table_keys, working["_component_id"]):
                key_to_component.setdefault(key, component_id)
            metadata_keys = metadata.loc[:, leaf_match_columns].apply(lambda row: normalized_key(row.tolist()), axis=1)
            metadata[leaf_component_column] = metadata_keys.map(lambda key: key_to_component.get(key, ""))
            matched_components = int(metadata[leaf_component_column].astype(str).str.strip().ne("").sum())
            log(f"[info] quality-category matrix leaf components matched by {','.join(leaf_match_columns)}: {matched_components}")
        else:
            id_to_component = (
                working.loc[working["tree_id"].astype(str).str.strip().ne("")]
                .drop_duplicates("tree_id")
                .set_index("tree_id")["_component_id"]
                .to_dict()
            )
            metadata = metadata.copy()
            metadata[leaf_component_column] = metadata[id_column].map(lambda value: id_to_component.get(clean_value(value), ""))
            matched_components = int(metadata[leaf_component_column].astype(str).str.strip().ne("").sum())
            log(f"[info] quality-category matrix leaf components matched by tree_id: {matched_components}")

    leaf_group_column = "_quality_category_matrix_group_key"
    metadata = metadata.copy()

    def matrix_group_key_from_leaf(row):
        values = []
        for column in configured_match_columns:
            if column == component_column:
                values.append(row.get(leaf_component_column, ""))
            elif column in metadata.columns:
                values.append(row.get(column, ""))
            else:
                values.append("")
        return normalized_key(values)

    metadata[leaf_group_column] = metadata.apply(matrix_group_key_from_leaf, axis=1)

    selection = clean_value(matrix_config.get("selection", "best")).lower() or "best"
    if selection not in {"best", "max", "highest"}:
        log(f"[warn] unsupported quality_category_matrix.selection={selection!r}; using 'best'")
        selection = "best"
    value_lookup = {}
    sorted_working = working.sort_values(by="_value", ascending=False, kind="mergesort")
    duplicate_cells = 0
    for row in sorted_working.to_dict("records"):
        key = (row["_matrix_group_key"], row["_quality"], row["_category"])
        if key not in value_lookup:
            value_lookup[key] = float(row["_value"])
        else:
            duplicate_cells += 1
    log(
        "[info] quality-category matrix cells available: "
        f"{len(value_lookup)}; collapsed duplicate linked genomes by best {value_column}: {duplicate_cells}"
    )
    return metadata, columns, {
        "values": value_lookup,
        "leaf_component_column": leaf_component_column,
        "leaf_group_column": leaf_group_column,
        "match_columns": configured_match_columns,
    }


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
        present_colors = {value: color_map[value] for value in sorted(set(values), key=lambda item: item.lower()) if value in color_map}
        legends[column] = {"label": color_config.get("label", column), "colors": present_colors}
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
    metadata, quality_matrix_columns, quality_matrix_lookup = prepare_quality_category_matrix(
        config.get("quality_category_matrix", {}),
        metadata,
        id_column,
    )
    metadata_by_id = metadata.set_index(id_column).to_dict("index") if not metadata.empty else {}
    output_prefix = clean_value(config.get("output_prefix")) or str(tree_path.with_suffix(""))
    outputs = [Path(output_prefix).expanduser().with_suffix("." + str(output_format).lstrip(".")) for output_format in config.get("output_formats", ["svg"])]
    if dry_run:
        return outputs, metadata_paths, "dry-run"

    ete = load_ete()
    set_qt_default_font()
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
    numeric_dot_configs = [dict(item) for item in config.get("numeric_dots", []) if item.get("show", True)]
    spacing_config = config.get("spacing", {})
    text_first = text_before_heatmap(config)

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

            def add_heatmap_faces(start_column):
                current_column = start_column
                if heatmap_configs:
                    for heatmap_config in heatmap_configs:
                        column = heatmap_config.get("column", "")
                        value = row.get(column, "")
                        value = heatmap_numeric_value(value, heatmap_config)
                        if clean_value(heatmap_config.get("style", "")).lower() in {"dot", "circle", "point"}:
                            min_value = float(heatmap_config.get("min_value", 0))
                            max_value = float(heatmap_config.get("max_value", 100))
                            if math.isfinite(value) and max_value > min_value:
                                fraction = max(0.0, min((float(value) - min_value) / (max_value - min_value), 1.0))
                            else:
                                fraction = math.nan
                            canvas_size = max(float(heatmap_config.get("width", 8)), float(heatmap_config.get("height", 10)))
                            min_diameter = float(heatmap_config.get("min_diameter", 0))
                            max_diameter = float(heatmap_config.get("max_diameter", canvas_size))
                            if not math.isfinite(fraction) or fraction <= 0:
                                face = make_blank_face(ete, canvas_size, canvas_size)
                            else:
                                diameter = min_diameter + (fraction * (max_diameter - min_diameter))
                                highlight_value = heatmap_config.get(
                                    "highlight_value",
                                    config.get("legend", {}).get("heatmap_dot_legend_highlight_value", None),
                                )
                                try:
                                    highlight_value = float(highlight_value) if highlight_value is not None else math.nan
                                except (TypeError, ValueError):
                                    highlight_value = math.nan
                                highlighted = math.isfinite(highlight_value) and abs(float(value) - highlight_value) < 1e-9
                                face = make_static_circle_face(
                                    ete,
                                    diameter,
                                    heatmap_config.get(
                                        "highlight_color",
                                        config.get("legend", {}).get("heatmap_dot_legend_highlight_color", "#BDBDBD"),
                                    )
                                    if highlighted
                                    else heatmap_config.get("color", "#111111"),
                                    border_color=heatmap_config.get("border_color", heatmap_config.get("color", "#111111")),
                                    border_width=float(heatmap_config.get("border_width", 0)),
                                    fill=True if highlighted else bool(heatmap_config.get("fill", True)),
                                    canvas_size=canvas_size,
                                )
                            add_face(ete, face, node, current_column, "aligned")
                        else:
                            color = grayscale_hex(
                                value,
                                heatmap_config.get("min_value", 0),
                                heatmap_config.get("max_value", 1),
                                reverse=bool(heatmap_config.get("reverse", False)),
                                missing_color=heatmap_config.get("missing_color", "#F2F2F2"),
                                lightest_shade=heatmap_config.get("lightest_shade", 250),
                                darkest_shade=heatmap_config.get("darkest_shade", 20),
                            )
                            add_face(
                                ete,
                                make_static_rect_face(
                                    ete,
                                    heatmap_config.get("width", 8),
                                    heatmap_config.get("height", 10),
                                    color,
                                ),
                                node,
                                current_column,
                                "aligned",
                            )
                        current_column += 1
                        current_column = add_gap_face(ete, node, current_column, heatmap_config.get("gap_after", 1), position="aligned")
                return current_column

            def add_numeric_dot_faces(start_column):
                current_column = start_column
                for dot_config in numeric_dot_configs:
                    value = row.get(dot_config.get("column", ""), "")
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        numeric_value = math.nan
                    min_value = float(dot_config.get("min_value", 0))
                    max_value = float(dot_config.get("max_value", max(numeric_value if math.isfinite(numeric_value) else 1.0, 1.0)))
                    if math.isfinite(numeric_value) and max_value > min_value:
                        fraction = max(0.0, min((numeric_value - min_value) / (max_value - min_value), 1.0))
                    else:
                        fraction = 0.0
                    canvas_size = max(float(dot_config.get("width", 12)), float(dot_config.get("height", 12)))
                    min_diameter = float(dot_config.get("min_diameter", 3))
                    max_diameter = float(dot_config.get("max_diameter", canvas_size))
                    diameter = min_diameter + (fraction * (max_diameter - min_diameter))
                    face = make_static_circle_face(
                        ete,
                        diameter,
                        dot_config.get("color", "#111111"),
                        border_color=dot_config.get("border_color", dot_config.get("color", "#111111")),
                        border_width=float(dot_config.get("border_width", 0)),
                        fill=bool(dot_config.get("fill", True)),
                        canvas_size=canvas_size,
                    )
                    add_face(ete, face, node, current_column, "aligned")
                    current_column += 1
                    current_column = add_gap_face(ete, node, current_column, dot_config.get("gap_after", 1), position="aligned")
                return current_column

            def add_text_faces(start_column):
                current_column = start_column
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
                    add_face(ete, face, node, current_column, "aligned")
                    current_column += 1
                    if text_index + 1 < len(text_configs):
                        current_column = add_gap_face(ete, node, current_column, spacing_config.get("between_text_columns", 8), position="aligned")
                return current_column

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
            if strip_configs and (text_configs or heatmap_configs or quality_matrix_columns):
                column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_strips_and_text", 12), position="aligned")
            if numeric_dot_configs:
                column_index = add_numeric_dot_faces(column_index)
            if text_first and text_configs:
                column_index = add_text_faces(column_index)
                if heatmap_configs or quality_matrix_columns:
                    column_index = add_gap_face(ete, node, column_index, spacing_config.get("between_heatmap_and_text", 8), position="aligned")
            if heatmap_configs:
                column_index = add_heatmap_faces(column_index)
            if quality_matrix_columns:
                matrix_group_key = row.get(
                    quality_matrix_lookup.get("leaf_group_column", quality_matrix_lookup.get("leaf_component_column", "component_id")),
                    "",
                )
                matrix_values = quality_matrix_lookup.get("values", {})
                for matrix_column in quality_matrix_columns:
                    matrix_value = matrix_values.get((matrix_group_key, matrix_column["quality"], matrix_column["category"]), math.nan)
                    try:
                        matrix_value_float = float(matrix_value)
                    except (TypeError, ValueError):
                        matrix_value_float = math.nan
                    is_missing_matrix_value = not math.isfinite(matrix_value_float)
                    color = grayscale_hex(
                        matrix_value_float,
                        matrix_column.get("min_value", 0),
                        matrix_column.get("max_value", 100),
                        missing_color=matrix_column.get("missing_color", "#FFFFFF"),
                    )
                    border_color = None
                    border_width = 0
                    fill_cell = True
                    if is_missing_matrix_value and matrix_column.get("missing_border", True):
                        border_color = matrix_column.get("missing_border_color", "#BDBDBD")
                        border_width = matrix_column.get("missing_border_width", 1)
                        fill_cell = False
                    add_face(
                        ete,
                        make_static_rect_face(
                            ete,
                            matrix_column.get("width", 8),
                            matrix_column.get("height", 10),
                            color,
                            border_color=border_color,
                            border_width=border_width,
                            border_style="dash",
                            dash_pattern=matrix_column.get("missing_border_dash", "0.35,1.4"),
                            fill=fill_cell,
                        ),
                        node,
                        column_index,
                        "aligned",
                    )
                    column_index += 1
                    column_index = add_gap_face(
                        ete,
                        node,
                        column_index,
                        matrix_column.get("gap_after", 1),
                        height=matrix_column.get("height", 10),
                        position="aligned",
                    )
            if not text_first and (heatmap_configs or quality_matrix_columns) and text_configs:
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
                if bar.get("show_value", False):
                    value_format = str(bar.get("value_format", "g"))
                    try:
                        value_label = format(numeric_value, value_format)
                    except (ValueError, TypeError):
                        value_label = f"{numeric_value:g}"
                    value_face = make_text_face(
                        ete,
                        value_label,
                        bar.get("font_size", 7),
                        bar.get("text_color", "#111111"),
                    )
                    value_face.margin_left = int(bar.get("value_margin_left", 3))
                    add_face(ete, value_face, node, column_index, "aligned")
                    column_index += 1
            if not text_first:
                column_index = add_text_faces(column_index)
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
    if hasattr(tree_style, "branch_vertical_margin"):
        tree_style.branch_vertical_margin = int(canvas_config.get("branch_vertical_margin", 0))
    guiding_config = config.get("guiding_lines", {})
    if hasattr(tree_style, "draw_guiding_lines"):
        tree_style.draw_guiding_lines = bool(guiding_config.get("show", False))
    if hasattr(tree_style, "guiding_lines_type"):
        tree_style.guiding_lines_type = int(guiding_config.get("type", 1))
    if hasattr(tree_style, "guiding_lines_color"):
        tree_style.guiding_lines_color = guiding_config.get("color", "#9A9A9A")
    if tree_style.mode == "c":
        tree_style.arc_start = int(canvas_config.get("arc_start", 0))
        tree_style.arc_span = int(canvas_config.get("arc_span", 359))
        add_circular_heatmap_title_labels(ete, tree_style, heatmap_configs, config.get("legend", {}))
    heatmap_label_columns = heatmap_column_indices(
        strip_configs,
        heatmap_configs,
        text_configs,
        spacing_config,
        matrix_configs=quality_matrix_columns,
        text_first=text_first,
        numeric_dot_configs=numeric_dot_configs,
    )
    add_heatmap_labels(ete, tree_style, heatmap_configs, config.get("legend", {}), heatmap_label_columns)
    matrix_label_columns = quality_matrix_column_indices(
        strip_configs,
        heatmap_configs,
        quality_matrix_columns,
        text_configs,
        spacing_config,
        text_first=text_first,
        numeric_dot_configs=numeric_dot_configs,
    )
    add_quality_category_matrix_labels(ete, tree_style, quality_matrix_columns, matrix_label_columns)
    heatmap_postprocess_configs = list(heatmap_configs) + list(quality_matrix_columns)

    if config.get("legend", {}).get("show", True):
        add_legend(
            ete,
            tree_style,
            strip_configs,
            strip_color_maps,
            config.get("legend", {}),
            branch_color_legends=branch_color_legends,
            heatmap_configs=heatmap_postprocess_configs,
            heatmap_labels_in_legend=(tree_style.mode == "c"),
            numeric_dot_configs=numeric_dot_configs,
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
            align_svg_heatmap_labels(out_path, heatmap_postprocess_configs, config.get("legend", {}))
            add_svg_quality_matrix_labels(out_path, quality_matrix_columns)
            add_svg_heatmap_dividers(out_path, heatmap_postprocess_configs)
            style_svg_guiding_lines(out_path, config.get("guiding_lines", {}))
            normalize_svg_font_family(out_path)
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
        align_svg_heatmap_labels(temporary_svg_path, heatmap_postprocess_configs, config.get("legend", {}))
        add_svg_quality_matrix_labels(temporary_svg_path, quality_matrix_columns)
        add_svg_heatmap_dividers(temporary_svg_path, heatmap_postprocess_configs)
        style_svg_guiding_lines(temporary_svg_path, config.get("guiding_lines", {}))
        normalize_svg_font_family(temporary_svg_path)
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
    numeric_dot_configs=None,
):
    branch_color_legends = branch_color_legends or {}
    heatmap_configs = heatmap_configs or []
    numeric_dot_configs = numeric_dot_configs or []
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
        dot_heatmaps = [
            heatmap_config
            for heatmap_config in heatmap_configs
            if clean_value(heatmap_config.get("style", "")).lower() in {"dot", "circle", "point"}
        ]
        if dot_heatmaps and legend_config.get("show_heatmap_dot_legend", False):
            example = dot_heatmaps[0]
            add_title_row(legend_config.get("heatmap_dot_legend_label", "KO prevalence"))
            canvas_size = max(float(example.get("width", 8)), float(example.get("height", 10)))
            min_diameter = float(example.get("min_diameter", 0))
            max_diameter = float(example.get("max_diameter", canvas_size))
            highlight_value = legend_config.get("heatmap_dot_legend_highlight_value", None)
            try:
                highlight_value = float(highlight_value) if highlight_value is not None else math.nan
            except (TypeError, ValueError):
                highlight_value = math.nan
            for value in legend_config.get("heatmap_dot_legend_values", [25, 50, 100]):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                min_value = float(example.get("min_value", 0))
                max_value = float(example.get("max_value", 100))
                fraction = 0.0 if max_value <= min_value else max(0.0, min((numeric_value - min_value) / (max_value - min_value), 1.0))
                diameter = min_diameter + (fraction * (max_diameter - min_diameter))
                highlighted = math.isfinite(highlight_value) and abs(numeric_value - highlight_value) < 1e-9
                add_legend_row(
                    make_static_circle_face(
                        ete,
                        diameter,
                        legend_config.get("heatmap_dot_legend_highlight_color", "#BDBDBD") if highlighted else example.get("color", "#111111"),
                        border_color=example.get("border_color", example.get("color", "#111111")),
                        border_width=float(example.get("border_width", 0)),
                        fill=True if highlighted else bool(example.get("fill", True)),
                        canvas_size=canvas_size,
                    ),
                    make_text_face(ete, f"{numeric_value:g}%", item_size, "#333333"),
                )
            add_legend_row(make_blank_face(ete, 1, section_gap), make_blank_face(ete, 1, section_gap))
        if numeric_dot_configs and legend_config.get("show_numeric_dot_legend", True):
            for dot_config in numeric_dot_configs:
                label = clean_value(dot_config.get("label", dot_config.get("column", "Dot size")))
                add_title_row(label)
                canvas_size = max(float(dot_config.get("width", 12)), float(dot_config.get("height", 12)))
                min_diameter = float(dot_config.get("min_diameter", 3))
                max_diameter = float(dot_config.get("max_diameter", canvas_size))
                min_value = float(dot_config.get("min_value", 0))
                max_value = float(dot_config.get("max_value", 1))
                values = dot_config.get("legend_values") or legend_config.get("numeric_dot_legend_values")
                if not values:
                    mid_value = (min_value + max_value) / 2.0
                    values = [min_value if min_value > 0 else 1, mid_value, max_value]
                for value in values:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError):
                        continue
                    fraction = 0.0 if max_value <= min_value else max(0.0, min((numeric_value - min_value) / (max_value - min_value), 1.0))
                    diameter = min_diameter + (fraction * (max_diameter - min_diameter))
                    add_legend_row(
                        make_static_circle_face(
                            ete,
                            diameter,
                            dot_config.get("color", "#111111"),
                            border_color=dot_config.get("border_color", dot_config.get("color", "#111111")),
                            border_width=float(dot_config.get("border_width", 0)),
                            fill=bool(dot_config.get("fill", True)),
                            canvas_size=canvas_size,
                        ),
                        make_text_face(ete, f"{numeric_value:g}", item_size, "#333333"),
                    )
                add_legend_row(make_blank_face(ete, 1, section_gap), make_blank_face(ete, 1, section_gap))
        if heatmap_configs and legend_config.get("show_heatmap_colorbar", True):
            add_title_row(legend_config.get("heatmap_colorbar_label", "Heatmap scale"))
            colorbar_orientation = clean_value(legend_config.get("heatmap_colorbar_orientation", "horizontal")).lower()
            added_vertical_colorbar = False
            if colorbar_orientation in {"vertical", "v"}:
                colorbar_face = make_vertical_heatmap_colorbar_face(ete, legend_config, item_size)
                if colorbar_face is not None:
                    add_legend_row(colorbar_face, make_blank_face(ete, 1, int(legend_config.get("heatmap_colorbar_vertical_height", legend_config.get("heatmap_colorbar_width", 90)))))
                    added_vertical_colorbar = True
            else:
                colorbar_face = make_heatmap_colorbar_face(ete, legend_config)
            if added_vertical_colorbar:
                pass
            elif colorbar_face is not None:
                add_legend_row(colorbar_face, make_blank_face(ete, 1, int(legend_config.get("heatmap_colorbar_bar_height", 8))))
                scale_face = make_heatmap_colorbar_scale_face(ete, legend_config, item_size)
                if scale_face is None:
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


def add_quality_category_matrix_labels(ete, tree_style, matrix_columns, label_columns=None):
    if not matrix_columns:
        return
    if any(matrix_column.get("tiered_labels", True) and matrix_column.get("svg_header_labels", True) for matrix_column in matrix_columns):
        return
    label_columns = label_columns or list(range(len(matrix_columns)))
    try:
        if any(matrix_column.get("tiered_labels", True) for matrix_column in matrix_columns):
            quality_to_indices = {}
            for index, matrix_column in enumerate(matrix_columns):
                quality = clean_value(matrix_column.get("quality", ""))
                quality_to_indices.setdefault(quality, []).append(index)

            for quality, indices in quality_to_indices.items():
                if not quality or not indices:
                    continue
                center_index = indices[len(indices) // 2]
                quality_label = clean_value(matrix_columns[center_index].get("quality_label", quality.upper()))
                for index in indices:
                    column = int(matrix_columns[index].get("label_column", label_columns[index]))
                    if index == center_index:
                        face = make_text_face(
                            ete,
                            quality_label,
                            int(matrix_columns[index].get("tier_label_font_size", 7)),
                            matrix_columns[index].get("label_color", "#333333"),
                        )
                        face.hz_align = 1
                        face.vt_align = 1
                        face.margin_top = int(matrix_columns[index].get("tier_label_margin_top", 2))
                        face.margin_bottom = int(matrix_columns[index].get("tier_label_margin_bottom", 2))
                    else:
                        face = make_blank_face(
                            ete,
                            int(matrix_columns[index].get("width", 8)),
                            int(matrix_columns[index].get("tier_label_font_size", 7)) + 4,
                        )
                    tree_style.aligned_header.add_face(face, column=column)

            for index, matrix_column in enumerate(matrix_columns):
                label = clean_value(matrix_column.get("category_label", matrix_column.get("category", "")))
                if not label:
                    continue
                face = make_text_face(
                    ete,
                    label,
                    int(matrix_column.get("category_label_font_size", matrix_column.get("label_font_size", 5))),
                    matrix_column.get("label_color", "#333333"),
                )
                face.rotation = int(matrix_column.get("category_label_rotation", -90))
                face.rotable = True
                face.hz_align = int(matrix_column.get("label_hz_align", 1))
                face.vt_align = int(matrix_column.get("label_vt_align", 1))
                face.margin_top = int(matrix_column.get("category_label_margin_top", 1))
                face.margin_bottom = int(matrix_column.get("category_label_margin_bottom", 28))
                face.margin_left = int(matrix_column.get("label_margin_left", 0))
                face.margin_right = int(matrix_column.get("label_margin_right", 0))
                tree_style.aligned_header.add_face(face, column=int(matrix_column.get("label_column", label_columns[index])))
            return

        for index, matrix_column in enumerate(matrix_columns):
            label = clean_value(matrix_column.get("label", ""))
            if not label:
                continue
            face = make_text_face(
                ete,
                label,
                int(matrix_column.get("label_font_size", 5)),
                matrix_column.get("label_color", "#333333"),
            )
            face.rotation = int(matrix_column.get("label_rotation", -90))
            face.hz_align = int(matrix_column.get("label_hz_align", 1))
            face.vt_align = int(matrix_column.get("label_vt_align", 1))
            face.margin_top = int(matrix_column.get("label_margin_top", 4))
            face.margin_left = int(matrix_column.get("label_margin_left", 0))
            face.margin_right = int(matrix_column.get("label_margin_right", 0))
            tree_style.aligned_foot.add_face(face, column=int(matrix_column.get("label_column", label_columns[index])))
    except Exception:
        log("[warn] could not attach quality-category matrix labels")


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
