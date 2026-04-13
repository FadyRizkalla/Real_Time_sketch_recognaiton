"""Plot model accuracy versus drawing stroke count."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import CATEGORIES, SAMPLES_PER_CLASS
from src.data.download import load_strokes
from src.data.preprocess import strokes_to_image
from src.models.cnn import SketchCNN


def build_samples(samples_per_class: int, image_size: int = 28) -> list[tuple[torch.Tensor, int, int]]:
    samples: list[tuple[torch.Tensor, int, int]] = []

    for label, category in enumerate(CATEGORIES):
        drawings = load_strokes(category=category, max_samples=samples_per_class)
        for drawing in drawings:
            image = strokes_to_image(drawing, image_size=image_size)
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
            stroke_count = len(drawing)
            samples.append((image_tensor, label, stroke_count))

    return samples


def evaluate_by_stroke_count(
    model: torch.nn.Module,
    samples: list[tuple[torch.Tensor, int, int]],
    device: torch.device,
) -> dict[int, tuple[int, int]]:
    stats: dict[int, tuple[int, int]] = defaultdict(lambda: (0, 0))
    model.eval()

    with torch.no_grad():
        for image_tensor, label, stroke_count in samples:
            logits = model(image_tensor.unsqueeze(0).to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            correct, total = stats[stroke_count]
            stats[stroke_count] = (correct + int(pred == label), total + 1)

    return dict(stats)


def plot_strokes_vs_accuracy(stats: dict[int, tuple[int, int]], output_path: Path) -> None:
    stroke_counts = sorted(stats.keys())
    accuracies = [stats[count][0] / stats[count][1] for count in stroke_counts]
    totals = [stats[count][1] for count in stroke_counts]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(stroke_counts, accuracies, marker="o", linewidth=2)
    ax1.set_xlabel("Number of strokes")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("CNN accuracy vs. drawing stroke count")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.bar(stroke_counts, totals, alpha=0.2, width=0.8)
    ax2.set_ylabel("Samples")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> None:
    weights_path = Path("best_cnn.pth")
    if not weights_path.exists():
        raise FileNotFoundError("best_cnn.pth not found. Train first to save best weights.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SketchCNN(num_classes=len(CATEGORIES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    samples = build_samples(samples_per_class=SAMPLES_PER_CLASS)
    if not samples:
        raise RuntimeError("No samples were loaded. Check categories and dataset files.")

    stats = evaluate_by_stroke_count(model, samples, device)
    output_path = Path("strokes_vs_accuracy.png")
    plot_strokes_vs_accuracy(stats, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
