from __future__ import annotations

from .ir import GraphEdge, GraphIR, GraphNode


def merge_graphs(draft: GraphIR, verified: GraphIR) -> GraphIR:
    merged_nodes: list[GraphNode] = []
    merged_edges: list[GraphEdge] = []

    id_to_node: dict[str, GraphNode] = {}
    label_index: dict[str, str] = {}

    node_id_counter = 1
    edge_id_counter = 1

    def next_node_id() -> str:
        nonlocal node_id_counter
        out = f"n{node_id_counter}"
        node_id_counter += 1
        return out

    def next_edge_id() -> str:
        nonlocal edge_id_counter
        out = f"e{edge_id_counter}"
        edge_id_counter += 1
        return out

    for node in draft.nodes:
        new_id = next_node_id()
        clone = GraphNode(
            id=new_id,
            kind=node.kind,
            label=node.label,
            params=dict(node.params),
            shapes=dict(node.shapes),
            status=node.status,
            source=node.source,
        )
        merged_nodes.append(clone)
        id_to_node[node.id] = clone
        if node.label not in label_index:
            label_index[node.label] = clone.id

    draft_id_map = {old.id: new.id for old, new in zip(draft.nodes, merged_nodes)}
    for edge in draft.edges:
        src = draft_id_map.get(edge.source)
        dst = draft_id_map.get(edge.target)
        if src and dst:
            merged_edges.append(GraphEdge(id=next_edge_id(), source=src, target=dst, label=edge.label))

    verified_id_map: dict[str, str] = {}
    for node in verified.nodes:
        merged_id = None
        if node.label in label_index:
            merged_id = label_index[node.label]
        elif node.kind == "Input" and "input" in label_index:
            merged_id = label_index["input"]
        elif node.kind == "Output" and "output" in label_index:
            merged_id = label_index["output"]

        if merged_id is not None:
            merged_node = next(n for n in merged_nodes if n.id == merged_id)
            merged_node.status = "verified"
            if node.shapes:
                merged_node.shapes = dict(node.shapes)
            if merged_node.kind in {"Module", "call_module"} and node.kind:
                merged_node.kind = node.kind
            verified_id_map[node.id] = merged_id
        else:
            new_id = next_node_id()
            clone = GraphNode(
                id=new_id,
                kind=node.kind,
                label=node.label,
                params=dict(node.params),
                shapes=dict(node.shapes),
                status=node.status,
                source=node.source,
            )
            merged_nodes.append(clone)
            label_index[node.label] = clone.id
            verified_id_map[node.id] = clone.id

    edge_seen = {(e.source, e.target, e.label) for e in merged_edges}
    for edge in verified.edges:
        src = verified_id_map.get(edge.source)
        dst = verified_id_map.get(edge.target)
        if not src or not dst:
            continue
        key = (src, dst, edge.label)
        if key in edge_seen:
            continue
        edge_seen.add(key)
        merged_edges.append(GraphEdge(id=next_edge_id(), source=src, target=dst, label=edge.label))

    meta = dict(draft.meta)
    meta.update(
        {
            "mode": "merged",
            "draft_nodes": len(draft.nodes),
            "verified_nodes": len(verified.nodes),
        }
    )
    return GraphIR(
        model_name=draft.model_name,
        nodes=merged_nodes,
        edges=merged_edges,
        framework=draft.framework,
        meta=meta,
    )
