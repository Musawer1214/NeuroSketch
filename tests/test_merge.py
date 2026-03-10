from hussain_livetorch_architect.ir import GraphEdge, GraphIR, GraphNode
from hussain_livetorch_architect.merge import merge_graphs


def test_merge_marks_verified() -> None:
    draft = GraphIR(
        model_name="Toy",
        nodes=[
            GraphNode(id="n1", kind="Input", label="input"),
            GraphNode(id="n2", kind="Linear", label="fc", status="draft"),
            GraphNode(id="n3", kind="Output", label="output"),
        ],
        edges=[
            GraphEdge(id="e1", source="n1", target="n2"),
            GraphEdge(id="e2", source="n2", target="n3"),
        ],
        meta={"mode": "draft"},
    )
    verified = GraphIR(
        model_name="Toy",
        nodes=[
            GraphNode(id="n1", kind="Input", label="input", status="verified"),
            GraphNode(
                id="n2",
                kind="Linear",
                label="fc",
                status="verified",
                shapes={"input": ["1x4"], "output": ["1x2"]},
            ),
        ],
        edges=[GraphEdge(id="e1", source="n1", target="n2")],
        meta={"mode": "verified"},
    )

    merged = merge_graphs(draft, verified)
    fc = next(node for node in merged.nodes if node.label == "fc")
    assert fc.status == "verified"
    assert fc.shapes["output"] == ["1x2"]
