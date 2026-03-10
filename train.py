import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 10) -> TrainNet:
    return TrainNet(num_classes=num_classes)


def train_one_step() -> float:
    model = build_model(num_classes=10)
    model.train()

    batch = torch.randn(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))

    logits = model(batch)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    return float(loss.item())


if __name__ == "__main__":
    print("single-step loss:", train_one_step())
