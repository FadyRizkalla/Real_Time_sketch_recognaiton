from quickdraw import QuickDrawDataGroup

# load just 3 drawings to inspect
group = QuickDrawDataGroup("cat", max_drawings=3)

for i, drawing in enumerate(group.drawings):
    print(f"\n--- Drawing {i} ---")
    print(f"Number of strokes: {len(drawing.strokes)}")

    for s_idx, stroke in enumerate(drawing.strokes):
        print(f"  Stroke {s_idx}: {stroke}")
