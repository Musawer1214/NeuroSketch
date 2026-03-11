from pathlib import Path

from neurosketch.exporters import export_graph, to_d2
from neurosketch.ir import GraphEdge, GraphIR, GraphNode


def test_export_json_and_dot(tmp_path: Path) -> None:
    graph = GraphIR(
        model_name="Toy",
        nodes=[
            GraphNode(id="n1", kind="Input", label="input"),
            GraphNode(id="n2", kind="Linear", label="fc"),
            GraphNode(id="n3", kind="Output", label="output"),
        ],
        edges=[
            GraphEdge(id="e1", source="n1", target="n2"),
            GraphEdge(id="e2", source="n2", target="n3"),
        ],
    )
    outputs, renderer = export_graph(graph, tmp_path, formats=["json", "dot"])
    assert (tmp_path / "graph.json").exists()
    assert (tmp_path / "graph.dot").exists()
    assert renderer in {"graphviz", "d2"}
    assert "json" in outputs and "dot" in outputs


def test_generate_d2_source() -> None:
    graph = GraphIR(
        model_name="Toy",
        nodes=[
            GraphNode(id="n1", kind="Input", label="input"),
            GraphNode(id="n2", kind="Linear", label="fc", status="verified"),
            GraphNode(
                id="n3",
                kind="Output",
                label="output",
                status="verified",
                shapes={"output": ["1x2"]},
            ),
        ],
        edges=[
            GraphEdge(id="e1", source="n1", target="n2"),
            GraphEdge(id="e2", source="n2", target="n3"),
        ],
    )
    d2 = to_d2(graph, theme="journal-light")
    assert "direction: right" in d2
    assert "fc" in d2
    assert "n1 -> n2" in d2


def test_d2_includes_param_summary() -> None:
    graph = GraphIR(
        model_name="ToyParams",
        nodes=[
            GraphNode(id="n1", kind="Input", label="input"),
            GraphNode(
                id="n2",
                kind="Linear",
                label="fc1",
                params={"arg0": 128, "arg1": 64},
            ),
            GraphNode(id="n3", kind="Output", label="output"),
        ],
        edges=[
            GraphEdge(id="e1", source="n1", target="n2"),
            GraphEdge(id="e2", source="n2", target="n3"),
        ],
    )
    d2 = to_d2(graph, theme="journal-light")
    assert "128->64" in d2
