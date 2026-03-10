from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any


@dataclass
class SourceRef:
    file: str
    line_start: int
    line_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    shapes: dict[str, list[str]] = field(default_factory=dict)
    status: str = "draft"
    source: SourceRef | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "params": self.params,
            "shapes": self.shapes,
            "status": self.status,
        }
        if self.source:
            out["source"] = self.source.to_dict()
        return out


@dataclass
class GraphEdge:
    id: str
    source: str
    target: str
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {"id": self.id, "source": self.source, "target": self.target}
        if self.label:
            out["label"] = self.label
        return out


@dataclass
class GraphIR:
    model_name: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    framework: str = "pytorch"
    version: str = "1.0"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "model": {"name": self.model_name, "framework": self.framework},
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "meta": self.meta,
        }

    def write_json(self, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out
