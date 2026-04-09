"""Entry point for running experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from config import BATCH_SIZE, CATEGORIES, EPOCHS, LEARNING_RATE, SAMPLES_PER_CLASS
from src.data.dataset import QuickDrawBitmapDataset
from src.models.cnn import SketchCNN
from src.training.evaluate import evaluate
from src.training.train import train


def main() -> None:
    dataset = QuickDrawBitmapDataset(categories=CATEGORIES, samples_per_class=SAMPLES_PER_CLASS)
    if len(dataset) == 0:
        print("Dataset is empty. Check internet connection and category names.")
        return

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SketchCNN(num_classes=len(CATEGORIES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {loss:.4f}")

    accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
