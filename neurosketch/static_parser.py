from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ir import GraphEdge, GraphIR, GraphNode, SourceRef


KNOWN_LAYER_NAMES = {
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "ReLU",
    "GELU",
    "SiLU",
    "LayerNorm",
    "Dropout",
    "Embedding",
    "MultiheadAttention",
    "TransformerEncoder",
    "TransformerDecoder",
    "Sequential",
}


class ParseError(RuntimeError):
    pass


@dataclass
class LayerDecl:
    name: str
    kind: str
    params: dict[str, Any]
    line_start: int
    line_end: int
    sequential_children: list["LayerDecl"] = field(default_factory=list)


class _IdFactory:
    def __init__(self) -> None:
        self._n = 1
        self._e = 1

    def node(self) -> str:
        out = f"n{self._n}"
        self._n += 1
        return out

    def edge(self) -> str:
        out = f"e{self._e}"
        self._e += 1
        return out


def _call_name(expr: ast.AST) -> str:
    if isinstance(expr, ast.Attribute):
        prefix = _call_name(expr.value)
        return f"{prefix}.{expr.attr}" if prefix else expr.attr
    if isinstance(expr, ast.Name):
        return expr.id
    return ""


def _literal_or_code(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.List):
        return [_literal_or_code(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return [_literal_or_code(elt) for elt in node.elts]
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _extract_call_params(call: ast.Call) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for idx, arg in enumerate(call.args):
        params[f"arg{idx}"] = _literal_or_code(arg)
    for kw in call.keywords:
        if kw.arg is not None:
            params[kw.arg] = _literal_or_code(kw.value)
    return params


def _extract_sequential_children(parent_name: str, call: ast.Call) -> list[LayerDecl]:
    children: list[LayerDecl] = []

    def _push_child(suffix: str, child_expr: ast.AST) -> None:
        if not isinstance(child_expr, ast.Call):
            return
        child_call_name = _call_name(child_expr.func)
        if not child_call_name:
            return
        child_kind = child_call_name.split(".")[-1]
        child_name = f"{parent_name}.{suffix}"
        children.append(
            LayerDecl(
                name=child_name,
                kind=child_kind,
                params=_extract_call_params(child_expr),
                line_start=getattr(child_expr, "lineno", getattr(call, "lineno", 1)),
                line_end=getattr(
                    child_expr,
                    "end_lineno",
                    getattr(child_expr, "lineno", getattr(call, "lineno", 1)),
                ),
                sequential_children=[],
            )
        )

    # nn.Sequential({"name": layer, ...}) style.
    if len(call.args) == 1 and isinstance(call.args[0], ast.Dict):
        mapping = call.args[0]
        for idx, (k, v) in enumerate(zip(mapping.keys, mapping.values)):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                suffix = k.value
            else:
                suffix = str(idx)
            _push_child(suffix, v)
        return children

    # nn.Sequential(OrderedDict([("name", layer), ...])) style.
    if len(call.args) == 1 and isinstance(call.args[0], ast.Call):
        maybe_ordered = call.args[0]
        ordered_name = _call_name(maybe_ordered.func).split(".")[-1].lower()
        if ordered_name == "ordereddict" and maybe_ordered.args:
            tuples_expr = maybe_ordered.args[0]
            if isinstance(tuples_expr, (ast.List, ast.Tuple)):
                for idx, elt in enumerate(tuples_expr.elts):
                    if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
                        continue
                    key, value = elt.elts
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        suffix = key.value
                    else:
                        suffix = str(idx)
                    _push_child(suffix, value)
                return children

    # nn.Sequential(layer0, layer1, ...)
    for idx, arg in enumerate(call.args):
        _push_child(str(idx), arg)
    return children


def _is_nn_module_subclass(cls: ast.ClassDef) -> bool:
    for base in cls.bases:
        name = _call_name(base)
        if name.endswith("nn.Module") or name.endswith("Module") or name == "Module":
            return True
    return False


class _ForwardGraphBuilder:
    def __init__(self, source_path: Path, model_name: str, layers: dict[str, LayerDecl]):
        self.source_path = source_path
        self.model_name = model_name
        self.layers = layers
        self.ids = _IdFactory()
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.var_to_node: dict[str, list[str]] = {}
        self.input_node_id: str | None = None
        self.output_node_id: str | None = None
        self._edge_seen: set[tuple[str, str, str | None]] = set()

    def add_node(
        self,
        kind: str,
        label: str,
        params: dict[str, Any] | None = None,
        status: str = "draft",
        source_line: tuple[int, int] | None = None,
    ) -> str:
        source = None
        if source_line is not None:
            source = SourceRef(
                file=str(self.source_path),
                line_start=source_line[0],
                line_end=source_line[1],
            )
        node_id = self.ids.node()
        self.nodes.append(
            GraphNode(
                id=node_id,
                kind=kind,
                label=label,
                params=params or {},
                status=status,
                source=source,
            )
        )
        return node_id

    def add_edge(self, src: str, dst: str, label: str | None = None) -> None:
        key = (src, dst, label)
        if key in self._edge_seen:
            return
        self._edge_seen.add(key)
        self.edges.append(GraphEdge(id=self.ids.edge(), source=src, target=dst, label=label))

    def _collect_arg_nodes(self, call: ast.Call) -> list[str]:
        out: list[str] = []
        # Method-style calls (e.g., x.flatten(1)) carry the tensor on the receiver.
        if isinstance(call.func, ast.Attribute):
            if not (isinstance(call.func.value, ast.Name) and call.func.value.id == "self"):
                out.extend(self.parse_expr(call.func.value))
        for arg in call.args:
            out.extend(self.parse_expr(arg))
        for kw in call.keywords:
            out.extend(self.parse_expr(kw.value))
        if not out and self.input_node_id:
            out.append(self.input_node_id)
        return list(dict.fromkeys(out))

    def parse_expr(self, expr: ast.AST) -> list[str]:
        if isinstance(expr, ast.Name):
            return self.var_to_node.get(expr.id, [])

        if isinstance(expr, ast.Constant):
            return []

        if isinstance(expr, ast.Attribute):
            return []

        if isinstance(expr, ast.Subscript):
            return self.parse_expr(expr.value)

        if isinstance(expr, ast.Tuple) or isinstance(expr, ast.List):
            out: list[str] = []
            for elt in expr.elts:
                out.extend(self.parse_expr(elt))
            return list(dict.fromkeys(out))

        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            left_nodes = self.parse_expr(expr.left)
            right_nodes = self.parse_expr(expr.right)
            node_id = self.add_node(
                kind="ResidualAdd",
                label="add",
                source_line=(getattr(expr, "lineno", 1), getattr(expr, "end_lineno", getattr(expr, "lineno", 1))),
            )
            for src in set(left_nodes + right_nodes):
                self.add_edge(src, node_id)
            return [node_id]

        if isinstance(expr, ast.Call):
            name = _call_name(expr.func)
            line_start = getattr(expr, "lineno", 1)
            line_end = getattr(expr, "end_lineno", line_start)

            # self.layer(...)
            if (
                isinstance(expr.func, ast.Attribute)
                and isinstance(expr.func.value, ast.Name)
                and expr.func.value.id == "self"
                and expr.func.attr in self.layers
            ):
                layer = self.layers[expr.func.attr]
                call_sources = self._collect_arg_nodes(expr)

                # Expand nn.Sequential into explicit internal layer chain.
                if layer.kind == "Sequential" and layer.sequential_children:
                    prev_ids = call_sources
                    created_ids: list[str] = []
                    for child in layer.sequential_children:
                        child_id = self.add_node(
                            kind=child.kind,
                            label=child.name,
                            params=child.params,
                            source_line=(child.line_start, child.line_end),
                        )
                        for src in prev_ids:
                            self.add_edge(src, child_id)
                        created_ids.append(child_id)
                        prev_ids = [child_id]
                    if created_ids:
                        return [created_ids[-1]]
                    return call_sources

                node_id = self.add_node(
                    kind=layer.kind,
                    label=layer.name,
                    params=layer.params,
                    source_line=(layer.line_start, layer.line_end),
                )
                for src in call_sources:
                    self.add_edge(src, node_id)
                return [node_id]

            lower = name.lower()
            if lower.endswith("torch.cat") or lower.endswith(".cat") or lower == "cat":
                node_id = self.add_node(
                    kind="Concat",
                    label="concat",
                    params=_extract_call_params(expr),
                    source_line=(line_start, line_end),
                )
                for src in self._collect_arg_nodes(expr):
                    self.add_edge(src, node_id)
                return [node_id]

            if lower.endswith("relu") or lower.endswith("gelu") or lower.endswith("silu"):
                kind = name.split(".")[-1].capitalize()
                node_id = self.add_node(
                    kind=kind,
                    label=name.split(".")[-1],
                    params=_extract_call_params(expr),
                    source_line=(line_start, line_end),
                )
                for src in self._collect_arg_nodes(expr):
                    self.add_edge(src, node_id)
                return [node_id]

            # Generic function op.
            op_name = name.split(".")[-1] if name else "call"
            node_id = self.add_node(
                kind=op_name,
                label=op_name,
                params=_extract_call_params(expr),
                source_line=(line_start, line_end),
            )
            for src in self._collect_arg_nodes(expr):
                self.add_edge(src, node_id)
            return [node_id]

        return []

    def parse_stmt(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            producers = self.parse_expr(stmt.value)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    self.var_to_node[target.id] = producers
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.var_to_node[elt.id] = producers
            return

        if isinstance(stmt, ast.AnnAssign):
            producers = self.parse_expr(stmt.value) if stmt.value else []
            if isinstance(stmt.target, ast.Name):
                self.var_to_node[stmt.target.id] = producers
            return

        if isinstance(stmt, ast.Expr):
            self.parse_expr(stmt.value)
            return

        if isinstance(stmt, ast.Return):
            producers = self.parse_expr(stmt.value) if stmt.value else []
            self.output_node_id = self.add_node(
                kind="Output",
                label="output",
                source_line=(getattr(stmt, "lineno", 1), getattr(stmt, "end_lineno", getattr(stmt, "lineno", 1))),
            )
            for src in producers:
                self.add_edge(src, self.output_node_id)
            return

        if isinstance(stmt, ast.If):
            for child in stmt.body:
                self.parse_stmt(child)
            for child in stmt.orelse:
                self.parse_stmt(child)
            return

        if isinstance(stmt, (ast.For, ast.While)):
            for child in stmt.body:
                self.parse_stmt(child)
            for child in stmt.orelse:
                self.parse_stmt(child)
            return

    def build(self, forward_fn: ast.FunctionDef) -> tuple[list[GraphNode], list[GraphEdge]]:
        self.input_node_id = self.add_node(kind="Input", label="input")

        forward_args = [a.arg for a in forward_fn.args.args if a.arg != "self"]
        for arg in forward_args:
            self.var_to_node[arg] = [self.input_node_id]

        for stmt in forward_fn.body:
            self.parse_stmt(stmt)

        if self.output_node_id is None:
            self.output_node_id = self.add_node(kind="Output", label="output")
            # If forward was not parsed, connect last known tensors.
            for producers in self.var_to_node.values():
                for src in producers:
                    self.add_edge(src, self.output_node_id)

        return self.nodes, self.edges


def parse_pytorch_source(source_path: str | Path, class_name: str | None = None) -> GraphIR:
    path = Path(source_path)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)

    module_classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and _is_nn_module_subclass(n)]
    if not module_classes:
        raise ParseError("No nn.Module class found in source file.")

    selected: ast.ClassDef | None = None
    if class_name:
        for cls in module_classes:
            if cls.name == class_name:
                selected = cls
                break
        if selected is None:
            raise ParseError(f"Class '{class_name}' not found among nn.Module classes.")
    else:
        selected = module_classes[0]

    assert selected is not None
    chosen_class_name = selected.name
    layers: dict[str, LayerDecl] = {}
    init_fn: ast.FunctionDef | None = None
    forward_fn: ast.FunctionDef | None = None

    for item in selected.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            init_fn = item
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            forward_fn = item

    if init_fn:
        for stmt in ast.walk(init_fn):
            assign_value: ast.AST | None = None
            assign_targets: list[ast.AST] = []
            if isinstance(stmt, ast.Assign):
                assign_value = stmt.value
                assign_targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                assign_value = stmt.value
                assign_targets = [stmt.target]
            if assign_value is None or not isinstance(assign_value, ast.Call):
                continue

            call_name = _call_name(assign_value.func)
            kind = call_name.split(".")[-1] if call_name else "Layer"
            if kind not in KNOWN_LAYER_NAMES and not call_name.startswith(("nn.", "torch.nn.")):
                continue

            for target in assign_targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    layers[target.attr] = LayerDecl(
                        name=target.attr,
                        kind=kind,
                        params=_extract_call_params(assign_value),
                        line_start=getattr(stmt, "lineno", 1),
                        line_end=getattr(stmt, "end_lineno", getattr(stmt, "lineno", 1)),
                        sequential_children=(
                            _extract_sequential_children(target.attr, assign_value)
                            if kind == "Sequential"
                            else []
                        ),
                    )

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    if forward_fn:
        builder = _ForwardGraphBuilder(path, chosen_class_name, layers)
        nodes, edges = builder.build(forward_fn)
    else:
        # No forward: create layer chain in declaration order.
        idf = _IdFactory()
        in_id = idf.node()
        nodes.append(GraphNode(id=in_id, kind="Input", label="input"))
        last = in_id
        for layer in layers.values():
            if layer.kind == "Sequential" and layer.sequential_children:
                for child in layer.sequential_children:
                    nid = idf.node()
                    nodes.append(
                        GraphNode(
                            id=nid,
                            kind=child.kind,
                            label=child.name,
                            params=child.params,
                            source=SourceRef(
                                file=str(path),
                                line_start=child.line_start,
                                line_end=child.line_end,
                            ),
                        )
                    )
                    edges.append(GraphEdge(id=idf.edge(), source=last, target=nid))
                    last = nid
            else:
                nid = idf.node()
                nodes.append(
                    GraphNode(
                        id=nid,
                        kind=layer.kind,
                        label=layer.name,
                        params=layer.params,
                        source=SourceRef(
                            file=str(path),
                            line_start=layer.line_start,
                            line_end=layer.line_end,
                        ),
                    )
                )
                edges.append(GraphEdge(id=idf.edge(), source=last, target=nid))
                last = nid
        out_id = idf.node()
        nodes.append(GraphNode(id=out_id, kind="Output", label="output"))
        edges.append(GraphEdge(id=idf.edge(), source=last, target=out_id))

    return GraphIR(
        model_name=chosen_class_name,
        nodes=nodes,
        edges=edges,
        framework="pytorch",
        meta={"mode": "draft", "parser": "ast"},
    )
