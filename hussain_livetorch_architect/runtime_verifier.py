from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .ir import GraphEdge, GraphIR, GraphNode

try:
    import torch
    import torch.fx as fx

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - tested via TORCH_AVAILABLE
    torch = None
    fx = None
    TORCH_AVAILABLE = False


class RuntimeVerificationError(RuntimeError):
    pass


def _shape_str(tensor: Any) -> str:
    if hasattr(tensor, "shape"):
        shape = tuple(int(x) for x in tensor.shape)
        return "x".join(str(v) for v in shape)
    return str(type(tensor).__name__)


def _flatten_tensors(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten_tensors(item))
        return out
    return [value]


def _collect_node_args(arg: Any) -> list[Any]:
    if fx is None:
        return []
    if isinstance(arg, fx.Node):
        return [arg]
    if isinstance(arg, (list, tuple)):
        out: list[Any] = []
        for item in arg:
            out.extend(_collect_node_args(item))
        return out
    if isinstance(arg, dict):
        out = []
        for item in arg.values():
            out.extend(_collect_node_args(item))
        return out
    return []


def _load_model_class(source_path: Path, class_name: str) -> type:
    module_name = f"hla_user_model_{source_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeVerificationError(f"Could not load module from: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, class_name):
        raise RuntimeVerificationError(f"Class '{class_name}' not found in {source_path}")
    cls = getattr(module, class_name)
    if not isinstance(cls, type):
        raise RuntimeVerificationError(f"'{class_name}' is not a class.")
    return cls


def _build_runtime_graph_from_fx(
    traced: Any,
    module_index: dict[str, Any],
    shape_info: dict[str, dict[str, list[str]]],
    model_name: str,
) -> GraphIR:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    fx_to_graph_id: dict[str, str] = {}

    node_idx = 1
    edge_idx = 1

    for fx_node in traced.graph.nodes:
        kind = fx_node.op
        label = str(fx_node.target) if fx_node.target is not None else fx_node.name
        params: dict[str, Any] = {}
        status = "verified"
        shapes: dict[str, list[str]] = {}

        if fx_node.op == "placeholder":
            kind = "Input"
            label = str(fx_node.name)
        elif fx_node.op == "output":
            kind = "Output"
            label = "output"
        elif fx_node.op == "call_module":
            module = module_index.get(str(fx_node.target))
            if module is not None:
                kind = module.__class__.__name__
            key = str(fx_node.target)
            if key in shape_info:
                shapes = shape_info[key]
        elif fx_node.op == "call_function":
            kind = getattr(fx_node.target, "__name__", "call_function")
            label = kind
        elif fx_node.op == "call_method":
            kind = str(fx_node.target)
            label = str(fx_node.target)

        graph_id = f"n{node_idx}"
        node_idx += 1
        fx_to_graph_id[fx_node.name] = graph_id
        nodes.append(
            GraphNode(
                id=graph_id,
                kind=kind,
                label=label,
                params=params,
                shapes=shapes,
                status=status,
            )
        )

    for fx_node in traced.graph.nodes:
        target_id = fx_to_graph_id[fx_node.name]
        for arg in _collect_node_args(fx_node.args):
            source_id = fx_to_graph_id.get(arg.name)
            if source_id:
                edges.append(GraphEdge(id=f"e{edge_idx}", source=source_id, target=target_id))
                edge_idx += 1
        for kwarg in _collect_node_args(fx_node.kwargs):
            source_id = fx_to_graph_id.get(kwarg.name)
            if source_id:
                edges.append(GraphEdge(id=f"e{edge_idx}", source=source_id, target=target_id))
                edge_idx += 1

    return GraphIR(
        model_name=model_name,
        nodes=nodes,
        edges=edges,
        framework="pytorch",
        meta={"mode": "verified", "runtime": "fx"},
    )


def _build_runtime_graph_from_hooks(
    hook_order: list[str],
    module_index: dict[str, Any],
    shape_info: dict[str, dict[str, list[str]]],
    model_name: str,
) -> GraphIR:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_idx = 1
    edge_idx = 1

    input_id = f"n{node_idx}"
    node_idx += 1
    nodes.append(GraphNode(id=input_id, kind="Input", label="input", status="verified"))
    prev = input_id

    for name in hook_order:
        module = module_index.get(name)
        kind = module.__class__.__name__ if module is not None else "Module"
        nid = f"n{node_idx}"
        node_idx += 1
        nodes.append(
            GraphNode(
                id=nid,
                kind=kind,
                label=name,
                shapes=shape_info.get(name, {}),
                status="verified",
            )
        )
        edges.append(GraphEdge(id=f"e{edge_idx}", source=prev, target=nid))
        edge_idx += 1
        prev = nid

    out_id = f"n{node_idx}"
    nodes.append(GraphNode(id=out_id, kind="Output", label="output", status="verified"))
    edges.append(GraphEdge(id=f"e{edge_idx}", source=prev, target=out_id))

    return GraphIR(
        model_name=model_name,
        nodes=nodes,
        edges=edges,
        framework="pytorch",
        meta={"mode": "verified", "runtime": "hooks"},
    )


def verify_runtime(
    source_path: str | Path,
    class_name: str,
    input_shape: list[int],
    init_args: list[Any] | None = None,
    init_kwargs: dict[str, Any] | None = None,
) -> GraphIR:
    if not TORCH_AVAILABLE:
        raise RuntimeVerificationError(
            "PyTorch is not installed. Install with: python -m pip install -e .[runtime]"
        )

    source = Path(source_path)
    cls = _load_model_class(source, class_name)
    init_args = init_args or []
    init_kwargs = init_kwargs or {}

    model = cls(*init_args, **init_kwargs)
    model.eval()

    shape_info: dict[str, dict[str, list[str]]] = {}
    hook_order: list[str] = []
    module_index = dict(model.named_modules())
    hooks = []

    def _make_hook(name: str):
        def _hook(_module, inputs, output):
            if name not in hook_order:
                hook_order.append(name)
            in_shapes = [_shape_str(t) for t in _flatten_tensors(inputs)]
            out_shapes = [_shape_str(t) for t in _flatten_tensors(output)]
            shape_info[name] = {"input": in_shapes, "output": out_shapes}

        return _hook

    for name, module in model.named_modules():
        if not name:
            continue
        if any(True for _ in module.children()):
            continue
        hooks.append(module.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            sample = torch.randn(*input_shape)
            _ = model(sample)
    except Exception as exc:
        raise RuntimeVerificationError(f"Runtime execution failed: {exc}") from exc
    finally:
        for h in hooks:
            h.remove()

    # Try FX graph first; if not traceable, fall back to hook sequence.
    try:
        traced = fx.symbolic_trace(model)
        return _build_runtime_graph_from_fx(
            traced=traced,
            module_index=module_index,
            shape_info=shape_info,
            model_name=class_name,
        )
    except Exception:
        return _build_runtime_graph_from_hooks(
            hook_order=hook_order,
            module_index=module_index,
            shape_info=shape_info,
            model_name=class_name,
        )
