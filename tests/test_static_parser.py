from pathlib import Path

from hussain_livetorch_architect.static_parser import parse_pytorch_source


def test_parse_minimal_module(tmp_path: Path) -> None:
    code = """
import torch.nn as nn
import torch.nn.functional as F

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(72, 4)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
"""
    src = tmp_path / "toy.py"
    src.write_text(code, encoding="utf-8")

    graph = parse_pytorch_source(src, class_name="Toy")

    labels = {n.label for n in graph.nodes}
    assert graph.model_name == "Toy"
    assert "input" in labels
    assert "conv" in labels
    assert "fc" in labels
    assert "output" in labels
    assert len(graph.edges) > 0


def test_parse_sequential_expansion(tmp_path: Path) -> None:
    code = """
import torch.nn as nn

class ToySeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
"""
    src = tmp_path / "toy_seq.py"
    src.write_text(code, encoding="utf-8")

    graph = parse_pytorch_source(src, class_name="ToySeq")
    labels = [n.label for n in graph.nodes]
    kinds = [n.kind for n in graph.nodes]

    assert graph.model_name == "ToySeq"
    assert any(lbl.startswith("features.") for lbl in labels)
    assert any(lbl.startswith("classifier.") for lbl in labels)
    assert "Sequential" not in kinds
    assert kinds.count("ReLU") >= 2
