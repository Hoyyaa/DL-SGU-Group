import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"
tracks = defaultdict(list)

with (raw_dir / "Train.csv").open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        if int(row["ClassId"]) == 20:
            path = Path(row["Path"])
            _, track_id, frame_id = path.stem.split("_")
            tracks[int(track_id)].append((int(frame_id), path))

cell_width, cell_height = 200, 180
grid = Image.new("RGB", (3 * cell_width, 3 * cell_height), "white")
draw = ImageDraw.Draw(grid)

for row_index, track_id in enumerate(sorted(tracks)[:3]):
    frames = sorted(tracks[track_id])
    selected = [frames[0], frames[len(frames) // 2], frames[-1]]

    for col_index, (frame_id, path) in enumerate(selected):
        with Image.open(raw_dir / path) as image:
            preview = ImageOps.contain(image.convert("RGB"), (180, 140))

        x = col_index * cell_width
        y = row_index * cell_height
        grid.paste(preview, (x + (cell_width - preview.width) // 2, y + 5))
        draw.text((x + 10, y + 150), f"Track {track_id} | Frame {frame_id}", fill="black")

output = project_dir / "reports" / "figures" / "class20_tracks.png"
output.parent.mkdir(parents=True, exist_ok=True)
grid.save(output)
print(f"Đã lưu: {output}")