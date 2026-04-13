# src/models/cnn.py

import torch.nn as nn

class SketchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 28x28 → 14x14
            nn.Dropout(p=0.25),

            # block 2 — TODO: add a second conv block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14x14 → 7x7
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            # TODO: add a linear layer
            # hint: what is the input size after two maxpools on 28x28?
            # 28 → 14 → 7, so you have 64 * 7 * 7 features
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # TODO: add final output layer
            # hint: output size = num_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)    # flatten
        x = self.classifier(x)
        return x                       # raw logits, no softmax