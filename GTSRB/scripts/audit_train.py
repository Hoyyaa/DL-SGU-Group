import csv
from collections import Counter, defaultdict
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
raw_dir = project_dir / "data" / "raw"

images_per_class = Counter()
tracks_per_class = defaultdict(set)
missing_files = 0

with (raw_dir / "Train.csv").open(encoding="utf-8", newline="") as csv_file:
    for row in csv.DictReader(csv_file):
        class_id = int(row["ClassId"])
        image_path = raw_dir / row["Path"]

        # Ví dụ: 00020_00000_00001.png → lớp 20, track 0, frame 1
        class_part, track_part, _ = image_path.stem.split("_")
        if int(class_part) != class_id:
            raise ValueError(f"ClassId không khớp tên ảnh: {row['Path']}")

        images_per_class[class_id] += 1
        tracks_per_class[class_id].add(int(track_part))

        if not image_path.is_file():
            missing_files += 1

print(f"{'ClassId':>7} {'Số ảnh':>8} {'Số track':>9}")
for class_id in sorted(images_per_class):
    print(f"{class_id:>7} {images_per_class[class_id]:>8} "
          f"{len(tracks_per_class[class_id]):>9}")

print(f"\nTổng số ảnh: {sum(images_per_class.values())}")
print(f"Tổng số lớp: {len(images_per_class)}")
print(f"Tổng số track: {sum(len(tracks) for tracks in tracks_per_class.values())}")
print(f"Số đường dẫn thiếu file: {missing_files}")