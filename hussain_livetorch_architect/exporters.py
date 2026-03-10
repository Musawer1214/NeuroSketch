from __future__ import annotations

import ast
from dataclasses import dataclass
import html
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

from .ir import GraphIR, GraphNode

try:
    from graphviz import Source as GraphvizSource

    GRAPHVIZ_PY_AVAILABLE = True
except Exception:  # pragma: no cover
    GraphvizSource = None
    GRAPHVIZ_PY_AVAILABLE = False


@dataclass(frozen=True)
class ThemePreset:
    d2_theme: int
    node_fill_draft: str
    node_fill_verified: str
    node_fill_io: str
    canvas: str
    stage_fill: str
    stage_stroke: str
    legend_fill: str
    legend_stroke: str
    stroke: str
    edge: str
    edge_width_dot: float
    edge_width_d2: int
    subtext: str
    font: str


THEMES: dict[str, ThemePreset] = {
    "journal-light": ThemePreset(
        d2_theme=8,  # Colorblind Clear
        node_fill_draft="#F7FAFF",
        node_fill_verified="#EAF5FF",
        node_fill_io="#FFF4E5",
        canvas="#FCFDFF",
        stage_fill="#F8FAFC",
        stage_stroke="#CBD5E1",
        legend_fill="#FFFFFF",
        legend_stroke="#CBD5E1",
        stroke="#22324A",
        edge="#334155",
        edge_width_dot=1.4,
        edge_width_d2=2,
        subtext="#5B6678",
        font="Source Sans Pro",
    ),
    "journal-gray": ThemePreset(
        d2_theme=301,  # Terminal Grayscale
        node_fill_draft="#F5F5F5",
        node_fill_verified="#EAEAEA",
        node_fill_io="#F0F0F0",
        canvas="#FBFBFB",
        stage_fill="#F4F4F5",
        stage_stroke="#D4D4D8",
        legend_fill="#FFFFFF",
        legend_stroke="#D4D4D8",
        stroke="#2E2E2E",
        edge="#3F3F46",
        edge_width_dot=1.35,
        edge_width_d2=2,
        subtext="#666666",
        font="Source Sans Pro",
    ),
    "journal-minimal": ThemePreset(
        d2_theme=0,  # Neutral Default
        node_fill_draft="#FFFFFF",
        node_fill_verified="#FFFFFF",
        node_fill_io="#FFFFFF",
        canvas="#FFFFFF",
        stage_fill="#FAFAFA",
        stage_stroke="#D1D5DB",
        legend_fill="#FFFFFF",
        legend_stroke="#D1D5DB",
        stroke="#1F2937",
        edge="#1F2937",
        edge_width_dot=1.3,
        edge_width_d2=2,
        subtext="#6B7280",
        font="Source Sans Pro",
    ),
}


def _select_theme(theme: str) -> ThemePreset:
    return THEMES.get(theme, THEMES["journal-light"])


def _ensure_tool_available(executable: str, windows_candidates: list[str]) -> str | None:
    existing = shutil.which(executable)
    if existing:
        return existing

    for candidate in windows_candidates:
        path = Path(candidate)
        if path.exists():
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            found = shutil.which(executable)
            if found:
                return found
    return None


def _ensure_dot_available() -> str | None:
    return _ensure_tool_available(
        executable="dot",
        windows_candidates=[r"C:\Program Files\Graphviz\bin"],
    )


def _ensure_d2_available() -> str | None:
    return _ensure_tool_available(
        executable="d2",
        windows_candidates=[r"C:\Program Files\D2"],
    )


def _sanitize_dot(text: str) -> str:
    return text.replace('"', '\\"')


def _escape_d2_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n")
    return escaped


def _safe_d2_identifier(value: str, idx: int) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not candidate or candidate[0].isdigit():
        candidate = f"node_{idx}"
    return candidate


def _node_shape(kind: str) -> str:
    cat = _node_category(kind)
    if cat == "input":
        return "oval"
    if cat == "output":
        return "oval"
    if cat == "activation":
        return "circle"
    if cat == "pool":
        return "hexagon"
    return "rectangle"


def _node_category(kind: str) -> str:
    k = kind.lower()
    if k in {"input", "output"}:
        return k
    if k.startswith("conv"):
        return "conv"
    if "batchnorm" in k or "layernorm" in k:
        return "norm"
    if k in {"relu", "gelu", "silu", "sigmoid", "tanh", "softmax", "leakyrelu"}:
        return "activation"
    if "pool" in k:
        return "pool"
    if k == "linear":
        return "linear"
    if "dropout" in k:
        return "dropout"
    if k == "flatten":
        return "reshape"
    return "other"


def _node_colors(node: GraphNode, preset: ThemePreset) -> tuple[str, str]:
    if node.kind.lower() in {"input", "output"}:
        return preset.node_fill_io, preset.stroke

    palette = {
        "conv": ("#DBEAFE", "#1D4ED8"),
        "norm": ("#EDE9FE", "#6D28D9"),
        "activation": ("#FEF3C7", "#B45309"),
        "pool": ("#CCFBF1", "#0F766E"),
        "linear": ("#DCFCE7", "#166534"),
        "dropout": ("#FEE2E2", "#B91C1C"),
        "reshape": ("#E2E8F0", "#475569"),
        "other": (preset.node_fill_verified if node.status == "verified" else preset.node_fill_draft, preset.stroke),
    }
    fill, stroke = palette.get(_node_category(node.kind), palette["other"])
    if node.status == "draft":
        # Keep draft mode visually distinct while preserving layer-type colors.
        return fill, "#64748B"
    return fill, stroke


def _node_label_lines(node: GraphNode) -> list[str]:
    def _short_label(label: str) -> str:
        if "." not in label:
            return label
        _prefix, tail = label.split(".", 1)
        if tail.isdigit():
            return ""
        return tail

    lines: list[str] = []
    if node.kind.lower() == "input":
        lines = ["Input"]
    elif node.kind.lower() == "output":
        lines = ["Output"]
    else:
        lines = [node.kind]
        short = _short_label(node.label) if node.label else ""
        if short and short.lower() != node.kind.lower():
            lines.append(short)
    param_text = _param_summary(node.kind, node.params)
    if param_text:
        lines.append(param_text)
    param_count = _estimate_param_count(node.kind, node.params)
    if param_count > 0:
        lines.append(f"params~{_format_compact_count(param_count)}")
    out_shapes = node.shapes.get("output", [])
    if out_shapes:
        preview = ", ".join(_shorten_shape_text(s) for s in out_shapes[:1])
        lines.append(f"shape: {preview}")
    return lines


def _group_name_from_label(label: str) -> str | None:
    if "." not in label:
        return None
    prefix = label.split(".", 1)[0].strip()
    if not prefix:
        return None
    if prefix.lower() in {"input", "output"}:
        return None
    return prefix


def _first_param(params: dict, *keys: str):
    for key in keys:
        if key in params:
            return params[key]
    return None


def _param_summary(kind: str, params: dict) -> str:
    if not params:
        return ""
    kind_lower = kind.lower()

    if kind_lower.startswith("conv"):
        in_ch = _first_param(params, "in_channels", "arg0")
        out_ch = _first_param(params, "out_channels", "arg1")
        kernel = _first_param(params, "kernel_size", "arg2")
        stride = _first_param(params, "stride")
        padding = _first_param(params, "padding")
        parts = []
        if in_ch is not None and out_ch is not None:
            parts.append(f"{in_ch}->{out_ch}")
        if kernel is not None:
            parts.append(f"k={kernel}")
        if stride is not None:
            parts.append(f"s={stride}")
        if padding is not None:
            parts.append(f"p={padding}")
        return " ".join(parts)

    if kind_lower == "linear":
        in_f = _first_param(params, "in_features", "arg0")
        out_f = _first_param(params, "out_features", "arg1")
        if in_f is not None and out_f is not None:
            return f"{in_f}->{out_f}"
        return ""

    if "batchnorm" in kind_lower:
        num_f = _first_param(params, "num_features", "arg0")
        return f"n={num_f}" if num_f is not None else ""

    if "dropout" in kind_lower:
        p = _first_param(params, "p", "arg0")
        return f"p={p}" if p is not None else ""

    if "pool" in kind_lower:
        kernel = _first_param(params, "kernel_size", "arg0")
        stride = _first_param(params, "stride")
        parts = []
        if kernel is not None:
            parts.append(f"k={kernel}")
        if stride is not None:
            parts.append(f"s={stride}")
        return " ".join(parts)

    if kind_lower == "flatten":
        start_dim = _first_param(params, "start_dim", "arg0")
        end_dim = _first_param(params, "end_dim", "arg1")
        parts = []
        if start_dim is not None:
            parts.append(f"start={start_dim}")
        if end_dim is not None:
            parts.append(f"end={end_dim}")
        return " ".join(parts)

    # Generic fallback: keep only first two entries to avoid clutter.
    items = [f"{k}={v}" for k, v in params.items() if not str(k).startswith("__")]
    return ", ".join(items[:2])


def _coerce_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if re.fullmatch(r"-?\d+", raw):
            try:
                return int(raw)
            except Exception:
                return None
        try:
            parsed = ast.literal_eval(raw)
            return _coerce_int(parsed)
        except Exception:
            return None
    return None


def _coerce_product(value) -> int | None:
    direct = _coerce_int(value)
    if direct is not None:
        return direct
    if isinstance(value, (list, tuple)):
        prod = 1
        for item in value:
            iv = _coerce_int(item)
            if iv is None:
                return None
            prod *= iv
        return prod
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = ast.literal_eval(raw)
            if parsed != value:
                return _coerce_product(parsed)
        except Exception:
            pass
        parts = [int(p) for p in re.findall(r"\d+", raw)]
        if not parts:
            return None
        prod = 1
        for part in parts:
            prod *= part
        return prod
    return None


def _format_compact_count(value: int) -> str:
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if value >= 1_000:
        text = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}K"
    return str(value)


def _bool_from_param(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no"}:
            return False
        if lowered in {"true", "1", "yes"}:
            return True
    return default


def _estimate_param_count(kind: str, params: dict) -> int:
    kind_lower = kind.lower()
    if not params:
        return 0

    if kind_lower.startswith("conv"):
        in_ch = _coerce_int(_first_param(params, "in_channels", "arg0"))
        out_ch = _coerce_int(_first_param(params, "out_channels", "arg1"))
        kernel_prod = _coerce_product(_first_param(params, "kernel_size", "arg2"))
        groups = _coerce_int(_first_param(params, "groups")) or 1
        bias = _bool_from_param(_first_param(params, "bias"), default=True)
        if in_ch is None or out_ch is None or kernel_prod is None or groups <= 0:
            return 0
        in_per_group = in_ch // groups
        weight = out_ch * in_per_group * kernel_prod
        return weight + (out_ch if bias else 0)

    if kind_lower == "linear":
        in_f = _coerce_product(_first_param(params, "in_features", "arg0"))
        out_f = _coerce_product(_first_param(params, "out_features", "arg1"))
        bias = _bool_from_param(_first_param(params, "bias"), default=True)
        if in_f is None or out_f is None:
            return 0
        weight = in_f * out_f
        return weight + (out_f if bias else 0)

    if "batchnorm" in kind_lower:
        num_f = _coerce_int(_first_param(params, "num_features", "arg0"))
        if num_f is None:
            return 0
        return num_f * 2

    if "layernorm" in kind_lower:
        norm_size = _coerce_product(_first_param(params, "normalized_shape", "arg0"))
        if norm_size is None:
            return 0
        return norm_size * 2

    if kind_lower == "embedding":
        n = _coerce_int(_first_param(params, "num_embeddings", "arg0"))
        d = _coerce_int(_first_param(params, "embedding_dim", "arg1"))
        if n is None or d is None:
            return 0
        return n * d

    return 0


def _shorten_shape_text(shape: str) -> str:
    text = str(shape)
    return text if len(text) <= 24 else f"{text[:21]}..."


def to_dot(graph: GraphIR, theme: str = "journal-light") -> str:
    preset = _select_theme(theme)
    lines = [
        "digraph HLA {",
        '  rankdir="LR";',
        f'  graph [pad="0.45", nodesep="0.56", ranksep="0.78", splines="spline", bgcolor="{preset.canvas}"];',
        f'  node [shape="box", style="rounded,filled", fontname="{preset.font}", fontsize="11", penwidth="1.45", color="{preset.stroke}"];',
        f'  edge [color="{preset.edge}", penwidth="{preset.edge_width_dot}", arrowsize="0.86", fontname="{preset.font}", fontsize="10"];',
    ]

    for node in graph.nodes:
        fill, stroke = _node_colors(node, preset)
        lines_for_label = _node_label_lines(node)
        rows: list[str] = []
        for i, line in enumerate(lines_for_label):
            escaped = html.escape(line)
            if i == 0:
                rows.append(f'<TR><TD ALIGN="LEFT"><B>{escaped}</B></TD></TR>')
            else:
                rows.append(
                    f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9.5" COLOR="{preset.subtext}">{escaped}</FONT></TD></TR>'
                )
        html_label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="2">{"".join(rows)}</TABLE>>'

        style_bits = ['rounded', 'filled']
        if node.status == "draft":
            style_bits.append("dashed")
        style_text = ",".join(style_bits)
        lines.append(
            f'  "{_sanitize_dot(node.id)}" [label={html_label}, fillcolor="{fill}", color="{stroke}", shape="{_node_shape(node.kind)}", style="{style_text}"];'
        )

    # Add stage grouping for a cleaner paper-style visual.
    group_nodes: dict[str, list[str]] = {}
    for node in graph.nodes:
        gname = _group_name_from_label(node.label)
        if not gname:
            continue
        group_nodes.setdefault(gname, []).append(node.id)

    for gname, node_ids in group_nodes.items():
        if len(node_ids) < 2:
            continue
        lines.append(f'  subgraph "cluster_{_sanitize_dot(gname)}" {{')
        lines.append(f'    label="{_sanitize_dot(gname)}";')
        lines.append('    style="rounded,filled";')
        lines.append(f'    color="{preset.stage_stroke}";')
        lines.append(f'    fillcolor="{preset.stage_fill}";')
        lines.append(f'    fontcolor="{preset.stroke}";')
        lines.append('    fontsize=11;')
        lines.append('    penwidth=1.1;')
        for nid in node_ids:
            lines.append(f'    "{_sanitize_dot(nid)}";')
        lines.append("  }")

    for edge in graph.edges:
        if edge.label:
            lines.append(
                f'  "{_sanitize_dot(edge.source)}" -> "{_sanitize_dot(edge.target)}" [label="{_sanitize_dot(edge.label)}", fontcolor="{preset.subtext}"];'
            )
        else:
            lines.append(f'  "{_sanitize_dot(edge.source)}" -> "{_sanitize_dot(edge.target)}";')

    lines.append("}")
    return "\n".join(lines)


def to_d2(graph: GraphIR, theme: str = "journal-light") -> str:
    preset = _select_theme(theme)
    lines = [
        "direction: right",
        "style: {",
        f'  stroke: "{preset.edge}"',
        f"  stroke-width: {preset.edge_width_d2}",
        f'  font-color: "{preset.stroke}"',
        "  font-size: 14",
        "}",
    ]

    def emit_node_block(
        out_lines: list[str],
        ident: str,
        node: GraphNode,
        indent: str = "",
    ) -> None:
        label = _escape_d2_string("\n".join(_node_label_lines(node)))
        fill, stroke = _node_colors(node, preset)
        shape = _node_shape(node.kind)
        allow_3d = shape in {"rectangle", "square", "hexagon"}
        out_lines.extend(
            [
                f'{indent}{ident}: "{label}" {{',
                f"{indent}  shape: {shape}",
                f"{indent}  style: {{",
                f'{indent}    fill: "{fill}"',
                f'{indent}    stroke: "{stroke}"',
                f"{indent}    stroke-width: 1",
                f'{indent}    font-color: "{preset.stroke}"',
                f"{indent}    font-size: 12",
                f"{indent}    shadow: true",
                f"{indent}  }}",
                f"{indent}}}",
            ]
        )
        if allow_3d:
            out_lines.insert(len(out_lines) - 2, f"{indent}    3d: true")

    # Stable node ids for rendering.
    id_map: dict[str, str] = {}
    used_ids: set[str] = set()
    for idx, node in enumerate(graph.nodes, start=1):
        sid = _safe_d2_identifier(node.id, idx)
        if sid in used_ids:
            sid = f"{sid}_{idx}"
        used_ids.add(sid)
        id_map[node.id] = sid

    # Group by stage prefix (e.g., features.0, classifier.1) for cleaner paper-style blocks.
    stage_nodes: dict[str, list[GraphNode]] = {}
    stage_order: list[str] = []
    grouped_node_ids: set[str] = set()
    for node in graph.nodes:
        stage = _group_name_from_label(node.label)
        if not stage:
            continue
        if stage not in stage_nodes:
            stage_nodes[stage] = []
            stage_order.append(stage)
        stage_nodes[stage].append(node)
    stage_ids: dict[str, str] = {}
    for idx, stage in enumerate(stage_order, start=1):
        if len(stage_nodes.get(stage, [])) < 2:
            continue
        stage_id = _safe_d2_identifier(f"stage_{stage}", idx)
        stage_ids[stage] = stage_id

    # Emit grouped stage containers.
    for stage in stage_order:
        nodes = stage_nodes.get(stage, [])
        if len(nodes) < 2:
            continue
        stage_id = stage_ids[stage]
        lines.extend(
            [
                f'{stage_id}: {{',
                f'  label: "{_escape_d2_string(stage)}"',
                "  direction: down",
                "  style: {",
                f'    fill: "{preset.stage_fill}"',
                f'    stroke: "{preset.stage_stroke}"',
                "    stroke-width: 1",
                f'    font-color: "{preset.stroke}"',
                "    font-size: 12",
                "    shadow: true",
                "  }",
            ]
        )
        for node in nodes:
            local_id = id_map[node.id]
            emit_node_block(lines, local_id, node, indent="  ")
            grouped_node_ids.add(node.id)
        lines.append("}")

    # Emit ungrouped nodes.
    for node in graph.nodes:
        if node.id in grouped_node_ids:
            continue
        emit_node_block(lines, id_map[node.id], node)

    # Build path references for edges.
    node_ref: dict[str, str] = {}
    for node in graph.nodes:
        stage = _group_name_from_label(node.label)
        if stage and stage in stage_ids and len(stage_nodes.get(stage, [])) >= 2:
            node_ref[node.id] = f"{stage_ids[stage]}.{id_map[node.id]}"
        else:
            node_ref[node.id] = id_map[node.id]

    for edge in graph.edges:
        src = node_ref.get(edge.source)
        dst = node_ref.get(edge.target)
        if not src or not dst:
            continue
        if edge.label:
            lines.extend(
                [
                    f'{src} -> {dst}: "{_escape_d2_string(edge.label)}" {{',
                    "  style: {",
                    f'    stroke: "{preset.edge}"',
                    f"    stroke-width: {preset.edge_width_d2}",
                    f'    font-color: "{preset.subtext}"',
                    "  }",
                    "}",
                ]
            )
        else:
            lines.extend(
                [
                    f"{src} -> {dst}: {{",
                    "  style: {",
                    f'    stroke: "{preset.edge}"',
                    f"    stroke-width: {preset.edge_width_d2}",
                    "  }",
                    "}",
                ]
            )

    # Legend block for color/shape semantics.
    lines.extend(
        [
            'legend: {',
            '  label: "Legend"',
            "  direction: down",
            "  style: {",
            f'    fill: "{preset.legend_fill}"',
            f'    stroke: "{preset.legend_stroke}"',
            "    stroke-width: 1",
            f'    font-color: "{preset.stroke}"',
            "    font-size: 12",
            "  }",
            '  conv: "Conv / Feature Layer" {',
            "    shape: rectangle",
            "    style: {",
            '      fill: "#DBEAFE"',
            '      stroke: "#1D4ED8"',
            "    }",
            "  }",
            '  norm: "Normalization" {',
            "    shape: rectangle",
            "    style: {",
            '      fill: "#EDE9FE"',
            '      stroke: "#6D28D9"',
            "    }",
            "  }",
            '  act: "Activation" {',
            "    shape: circle",
            "    style: {",
            '      fill: "#FEF3C7"',
            '      stroke: "#B45309"',
            "    }",
            "  }",
            '  linear: "Linear / Dense" {',
            "    shape: rectangle",
            "    style: {",
            '      fill: "#DCFCE7"',
            '      stroke: "#166534"',
            "    }",
            "  }",
            '  pool: "Pooling" {',
            "    shape: hexagon",
            "    style: {",
            '      fill: "#CCFBF1"',
            '      stroke: "#0F766E"',
            "    }",
            "  }",
            "}",
        ]
    )

    return "\n".join(lines)


def resolve_renderer(renderer: str, formats: Iterable[str]) -> str:
    desired = renderer.strip().lower()
    if desired not in {"auto", "d2", "graphviz"}:
        desired = "auto"

    render_targets = {fmt.strip().lower() for fmt in formats}.intersection({"svg", "png", "pdf"})
    if desired == "d2":
        return "d2"
    if desired == "graphviz":
        return "graphviz"

    # auto
    if render_targets and _ensure_d2_available():
        return "d2"
    return "graphviz"


def _render_with_graphviz(dot_text: str, target_dir: Path, render_targets: set[str]) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    dot_bin = _ensure_dot_available()
    if not dot_bin or not GRAPHVIZ_PY_AVAILABLE:
        return outputs
    for fmt in sorted(render_targets):
        try:
            src = GraphvizSource(dot_text)
            rendered = src.render(
                filename="graph",
                directory=str(target_dir),
                format=fmt,
                cleanup=True,
            )
            outputs[fmt] = Path(rendered)
        except Exception:
            pass
    return outputs


def _render_with_d2(
    d2_text: str,
    target_dir: Path,
    render_targets: set[str],
    theme: str,
    layout: str,
    pad: int,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    d2_bin = _ensure_d2_available()
    if not d2_bin:
        return outputs

    theme_id = str(_select_theme(theme).d2_theme)
    source_path = target_dir / "graph.d2"
    for fmt in sorted(render_targets):
        output_path = target_dir / f"graph.{fmt}"
        cmd = [
            d2_bin,
            "--layout",
            layout,
            "--theme",
            theme_id,
            "--pad",
            str(pad),
            str(source_path),
            str(output_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and output_path.exists():
                outputs[fmt] = output_path
        except Exception:
            pass
    return outputs


def export_graph(
    graph: GraphIR,
    out_dir: str | Path,
    formats: Iterable[str],
    renderer: str = "auto",
    theme: str = "journal-light",
    layout: str = "elk",
    pad: int = 40,
) -> tuple[dict[str, Path], str]:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    requested = {fmt.strip().lower() for fmt in formats}
    outputs: dict[str, Path] = {}

    if "json" in requested:
        outputs["json"] = graph.write_json(target_dir / "graph.json")

    effective_renderer = resolve_renderer(renderer, requested)
    render_targets = requested.intersection({"svg", "png", "pdf"})

    if effective_renderer == "d2":
        d2_text = to_d2(graph, theme=theme)
        d2_path = target_dir / "graph.d2"
        d2_path.write_text(d2_text, encoding="utf-8")
        outputs["d2"] = d2_path
        if "dot" in requested:
            dot_text = to_dot(graph, theme=theme)
            dot_path = target_dir / "graph.dot"
            dot_path.write_text(dot_text, encoding="utf-8")
            outputs["dot"] = dot_path
        outputs.update(
            _render_with_d2(
                d2_text=d2_text,
                target_dir=target_dir,
                render_targets=render_targets,
                theme=theme,
                layout=layout,
                pad=pad,
            )
        )
        return outputs, effective_renderer

    dot_text = to_dot(graph, theme=theme)
    dot_path = target_dir / "graph.dot"
    dot_path.write_text(dot_text, encoding="utf-8")
    outputs["dot"] = dot_path
    outputs.update(_render_with_graphviz(dot_text, target_dir, render_targets))
    return outputs, effective_renderer
