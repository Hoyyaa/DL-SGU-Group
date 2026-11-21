# Phân loại biển báo giao thông GTSRB (môn Học sâu)

So sánh CNN nhỏ với ResNet-18 trên GTSRB (43 lớp), kiểm tra vai trò skip connection,
và xem augmentation độ sáng / làm mờ có giúp mô hình bền hơn với ảnh tối và ảnh mờ không.

- Chỉ số chính: **Macro-F1** (trung bình 3 seed: 42, 43, 44).
- Kế hoạch đầy đủ: mục 1–9 trong tài liệu kế hoạch của nhóm. Hợp đồng code: [docs/CONTRACTS.md](docs/CONTRACTS.md).
- Việc cần làm và ai làm: [docs/TASKS.md](docs/TASKS.md). Quy tắc cho AI trợ lý: [AGENTS.md](AGENTS.md).

## Cài đặt (Linux/WSL)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q
```

## Chuẩn bị dữ liệu (làm 1 lần, đứng trong thư mục GTSRB/)

Đặt dữ liệu GTSRB (bản có `Train/`, `Test/`, `Meta/`, `Train.csv`, `Test.csv`) vào `data/raw/`.

```bash
python -m gtsrb.data.make_split --raw data/raw/Train      # -> data/splits/split_manifest.csv
python -m gtsrb.data.cache --raw data/raw/Train           # -> data/processed/gtsrb64.npz + data/splits/norm_stats.json
python -m gtsrb.data.cache --test_dir data/raw/Test --test_csv data/raw/Test.csv   # cache test (chỉ dùng lúc chấm cuối)
```

`split_manifest.csv` và `norm_stats.json` nằm trong Git. `data/raw/` và `data/processed/` thì không (quá nặng).
Chia train/val theo (lớp, track) với `split_seed=0`, nên mọi máy ra cùng một kết quả: 31.379 train / 7.830 val.

## Cấu trúc

```
GTSRB/
├── configs/            base.yaml + một file YAML cho mỗi thí nghiệm (chỉ ghi phần khác base)
├── data/               raw/ processed/ (không vào Git); splits/ (vào Git)
├── docs/               CONTRACTS.md, TASKS.md
├── src/gtsrb/
│   ├── data/           make_split.py, cache.py, dataset.py, corruptions.py (C)
│   ├── models/         small_cnn.py, resnet18.py, plain18.py, __init__.py: build_model (B)
│   ├── utils/          config.py (load_config, ROOT), seed.py (set_seed)
│   ├── train.py        huấn luyện
│   └── evaluate.py     chấm test theo điều kiện orig / dark / blur
├── tests/              pytest; test_contracts.py là "đặc tả chạy được"
└── notebooks/          00 đến 06, mỗi notebook một người phụ trách
```

## Trạng thái

| Phần | Người phụ trách | Trạng thái |
| --- | --- | --- |
| Khung repo, config, seed | A | xong |
| Chia tập, cache, norm_stats, dataset | A | xong (dataset.py cần chạy thử với PyTorch thật) |
| `train.py`, `small_cnn.py` | A | chưa làm |
| `resnet18.py`, `plain18.py`, `build_model` | B | chưa làm |
| `corruptions.py`, `evaluate.py` | C | chưa làm |
| Notebook, demo, báo cáo | cả nhóm | chưa làm |

Cập nhật bảng này mỗi khi merge một PR.

## Quy ước làm việc

- Nhánh: `feat/<việc>`, `fix/<việc>`. Không push thẳng vào `main`. Mỗi PR cần một người khác xem.
- Mỗi PR có test chạy xanh (`pytest -q`) trước khi xin review.
- Không commit dữ liệu, `.npz`, `.pt`, `runs/`.
- Không sửa hợp đồng trong CONTRACTS.md một mình: sửa thì mở Issue và báo cả nhóm.
