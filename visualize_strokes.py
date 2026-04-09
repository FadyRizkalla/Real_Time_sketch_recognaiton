import sys

sys.path.append(".")

import matplotlib.pyplot as plt
from src.data.download import load_strokes
from src.data.preprocess import strokes_to_image


def render_partial(drawing, n_strokes):
    partial = drawing[:n_strokes]
    return strokes_to_image(partial)


if __name__ == "__main__":
    print("loading cat drawings...")
    drawings = load_strokes("airplane", max_samples=5)
    drawing = drawings[3]
    total = len(drawing)
    print(f"drawing has {total} strokes")

    fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
    if total == 1:
        axes = [axes]

    for n in range(1, total + 1):
        img = render_partial(drawing, n)
        axes[n - 1].imshow(img, cmap="gray")
        axes[n - 1].set_title(f"{n} stroke{'s' if n > 1 else ''}")
        axes[n - 1].axis("off")

    plt.suptitle("cat - stroke by stroke", fontsize=14)
    plt.tight_layout()
    plt.savefig("evolution.png", dpi=100)
    plt.show()
    print("saved evolution2.png")
