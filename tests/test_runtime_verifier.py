import importlib.util
from pathlib import Path

import pytest

from hussain_livetorch_architect.runtime_verifier import TORCH_AVAILABLE, verify_runtime


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_runtime_verify_simple_model(tmp_path: Path) -> None:
    code = """
import torch
import torch.nn as nn

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
"""
    src = tmp_path / "toy_runtime.py"
    src.write_text(code, encoding="utf-8")
    graph = verify_runtime(src, "Toy", [1, 4])
    assert graph.meta["mode"] == "verified"
    assert len(graph.nodes) >= 3


def test_runtime_module_importable() -> None:
    assert importlib.util.find_spec("hussain_livetorch_architect.runtime_verifier") is not None
