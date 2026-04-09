import sys

sys.path.append(".")

import matplotlib.pyplot as plt
from config import CATEGORIES
from src.data.download import load_strokes
from src.data.preprocess import strokes_to_image


def render_partial(drawing, n_strokes):
    partial = drawing[:n_strokes]
    return strokes_to_image(partial)


if __name__ == "__main__":
    for category in CATEGORIES:
        print(f"loading {category} drawings...")
        drawings = load_strokes(category, max_samples=100)
        if not drawings:
            print(f"skipping {category}: no recognized drawings found")
            continue

        drawing = max(drawings, key=len)
        total = len(drawing)
        print(f"{category}: selected drawing has {total} strokes")

        fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
        if total == 1:
            axes = [axes]

        for n in range(1, total + 1):
            img = render_partial(drawing, n)
            axes[n - 1].imshow(img, cmap="gray")
            axes[n - 1].set_title(f"{n} stroke{'s' if n > 1 else ''}")
            axes[n - 1].axis("off")

        plt.suptitle(f"{category} - stroke by stroke", fontsize=14)
        plt.tight_layout()
        output_name = f"evolution_{category.lower().replace(' ', '_')}.png"
        plt.savefig(output_name, dpi=100)
        plt.close(fig)
        print(f"saved {output_name}")
