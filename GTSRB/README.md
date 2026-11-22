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
│   ├── models/         __init__.py (Classifier, build_model, NETS), small_cnn.py (A); resnet18.py, plain18.py (B)
│   ├── utils/          config.py (load_config, ROOT), seed.py (set_seed)
│   ├── train.py        huấn luyện (A)
│   └── evaluate.py     chấm test theo điều kiện orig / dark / blur (C)
├── tests/              pytest; test_contracts.py là "đặc tả chạy được"
└── notebooks/          00 đến 06, mỗi notebook một người phụ trách
```

## Chạy huấn luyện

```bash
# thử pipeline: ~10% dữ liệu, tối đa 2 epoch, ghi vào thư mục tạm
python -m gtsrb.train --config configs/e1_small_cnn.yaml --seed 42 --out_dir runs_smoke --fast

# chạy thật (Colab: out_dir nằm trên Drive để không mất khi ngắt phiên)
python -m gtsrb.train --config configs/e1_small_cnn.yaml --seed 42 --out_dir /content/drive/MyDrive/gtsrb/runs

# Colab ngắt giữa chừng: chạy lại đúng lệnh trên, thêm --resume
```

Mỗi run ra `<out_dir>/<name>_s<seed>/` gồm `best.pt`, `last.pt`, `config.yaml`, `log.csv`, `env.json`.
Thư mục đã có kết quả sẽ không bị ghi đè (cần `--resume`, hoặc xóa thư mục nếu muốn chạy lại từ đầu).
Sáu cấu hình thí nghiệm E1 đến E6 nằm trong `configs/`.

## Quản lý thí nghiệm

- **Kết quả:** thư mục Google Drive chung (`runs/`, `results/`, `figures/`). Checkpoint `.pt` chỉ để trên Drive, không vào Git.
- **Nhật ký:** một Google Sheet, mỗi lần chạy một dòng (cột ở mục 6 của `docs/CONTRACTS.md`). Người chạy tự ghi dòng của mình ngay sau khi run xong.
- Cột Macro-F1 test chỉ được điền sau khi đã gắn tag `exp-v1`.

## Trạng thái

| Phần | Người phụ trách | Trạng thái |
| --- | --- | --- |
| Khung repo, config, seed | A | xong |
| Chia tập, cache train và test, norm_stats, dataset | A | xong, đã chạy thử với PyTorch |
| `models/__init__.py` (Classifier, build_model), `small_cnn.py`, `train.py`, configs E1–E6 | A | PR `feat/train` |
| `resnet18.py`, `plain18.py` và đăng ký vào `NETS` | B | chưa làm |
| `corruptions.py`, `evaluate.py` (kèm params và latency) | C | chưa làm |
| Thư mục Drive, Google Sheet nhật ký | C | chưa làm |
| Notebook, demo, slide, báo cáo | cả nhóm | chưa làm |

Cập nhật bảng này mỗi khi merge một PR.

## Quy ước làm việc

- Nhánh: `feat/<việc>`, `fix/<việc>`. Không push thẳng vào `main`. Mỗi PR cần một người khác xem.
- Mỗi PR có test chạy xanh (`pytest -q`) trước khi xin review.
- Không commit dữ liệu, `.npz`, `.pt`, `runs/`.
- Không sửa hợp đồng trong CONTRACTS.md một mình: sửa thì mở Issue và báo cả nhóm.
