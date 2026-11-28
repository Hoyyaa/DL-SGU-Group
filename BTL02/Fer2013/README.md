# BTL02 – CNN nhận diện cảm xúc khuôn mặt (FER2013, PyTorch)

Đề bài: `../DB/Emotion_Recognition.pdf`. Báo cáo mẫu (Keras): `../DB/demo_report.pdf`.
Pipeline này dịch luồng của báo cáo mẫu sang PyTorch và bổ sung phần đề bài bắt buộc
(nhiều thí nghiệm, trực quan hóa conv cuối, phân tích đúng/sai). Thiết kế chi tiết: [docs/PIPELINE.md](docs/PIPELINE.md).

```
data/raw/{train,test}/ ──prepare──► data/processed/fer48.npz ──train──► runs/<exp>_s42/best.pt, log.csv
                                                                          │
                     results/<exp>_s42/  ◄──evaluate── (val + test)       ┤
                     results/<exp>_s42/viz/ ◄──visualize── (filters, actmax, feature map, Grad-CAM)
                     results/summary.csv, summary.md ◄──summarize── (bảng so sánh mọi thí nghiệm)
```

## Cài đặt

```bash
cd BTL02/Fer2013
pip install -r requirements.txt
pip install -e .
python -m pytest -q          # smoke test trên dữ liệu giả, ~30s CPU
```

## Dữ liệu

Tải bản Kaggle [`msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013), giải nén vào `data/raw/`:

```text
data/raw/train/<emotion>/Training_*.jpg                       28.709 ảnh
data/raw/test/<emotion>/PublicTest_*.jpg, PrivateTest_*.jpg   3.589 + 3.589 ảnh
```

```bash
python -m fer.data.prepare                  # hoặc --src <thư mục chứa train/ và test/>
```

Kaggle chỉ chia train / test nhưng pipeline cần thêm tập **val** để chọn checkpoint và early stopping:

| Tập | Lấy từ | Số ảnh | Dùng để |
| --- | --- | --- | --- |
| train | toàn bộ `train/` | 28.709 | học |
| val | ảnh `PublicTest_*` trong `test/` | 3.589 | chọn `best.pt`, early stopping, giảm lr |
| test | ảnh `PrivateTest_*` trong `test/` | 3.589 | chỉ để báo cáo |

Cách chia này trùng với cách chia gốc của cuộc thi, nên so sánh được với mốc 69% / 71% của đề.
Nếu tên file trong `test/` không có tiền tố trên, `prepare` tự tách 10% `train/` (đều theo lớp, seed 42) làm val
và dùng toàn bộ `test/` làm test. Kết quả in ra phải khớp Bảng 1 của đề.

## Chạy một thí nghiệm

```bash
python -m fer.train     --config configs/e5_large_bn.yaml --seed 42
python -m fer.evaluate  --run runs/e5_large_bn_s42
python -m fer.visualize --run runs/e5_large_bn_s42

# thử nhanh pipeline (10% dữ liệu, 2 epoch, không dùng làm số liệu)
python -m fer.train --config configs/e1_small.yaml --out_dir runs_smoke --fast
# đổi tham số không cần tạo file mới
python -m fer.train --config configs/e5_large_bn.yaml --set train.lr=0.0003 name=e5_lr3e-4
# Colab bị ngắt: chạy lại đúng lệnh cũ, thêm --resume
```

## Ghi log lên Weights & Biases (tùy chọn)

```bash
pip install wandb
wandb login                       # dán API key ở https://wandb.ai/authorize (một lần). KHÔNG commit key
python -m fer.train --config configs/e5_large_bn.yaml --wandb
python -m fer.evaluate  --run runs/e5_large_bn_s42      # tự nối vào cùng run W&B
python -m fer.visualize --run runs/e5_large_bn_s42      # (thêm --no_wandb nếu không muốn)
```

- Một thí nghiệm = một run trên W&B, tên `<exp>_s<seed>`. Id lưu ở `runs/<exp>_s<seed>/wandb_id.txt`,
  nên `evaluate`, `visualize` và `train --resume` (Colab bị ngắt) đều ghi tiếp vào đúng run đó.
- **train:** đường loss / acc / macro-F1 / lr / `train_val_gap` theo epoch; summary `best_val_acc`, `best_epoch`, `params`.
- **evaluate:** summary `val/acc`, `test/acc`, `val/macro_f1`, `test/macro_f1`; bảng F1 từng lớp; confusion matrix,
  true/wrong cases, curves (mục `eval/`).
- **visualize:** filter conv, activation maximization, feature map, Grad-CAM (mục `viz/`).
- Mỗi config có `wandb.group` (A_architecture, B_reference, C_training, D_activation, E_transfer) để lọc,
  so sánh trên dashboard. Tag tự gắn: tên mô hình, activation, optimizer, aug / no_aug.
- Cả nhóm dùng chung một project: tạo team trên W&B rồi `--set wandb.entity=<tên team>` (hoặc sửa `configs/base.yaml`).
- Không có mạng: `--set wandb.mode=offline`, sau đó `wandb sync runs/wandb/wandb/offline-run-*`.
- `--set wandb.log_model=true` đẩy `best.pt` lên Artifact; `wandb.watch=true` ghi histogram gradient.
- Lỗi mạng hoặc chưa login thì chỉ in cảnh báo, việc train vẫn tiếp tục. Số liệu gốc vẫn nằm ở `runs/` và `results/`.

Chạy toàn bộ trên Windows: `powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1 -Wandb` (log ở `logs/`, chạy lại sẽ tiếp tục từ chỗ dừng). Linux/Colab: `bash scripts/run_all.sh` (thêm `WANDB=1` ở đầu để ghi log lên W&B) (hoặc truyền danh sách config: `bash scripts/run_all.sh configs/e1_small.yaml configs/e5_large_bn.yaml`).

## Cấu trúc

| Đường dẫn | Vai trò |
| --- | --- |
| `configs/base.yaml` | Mặc định chung; `e*.yaml` chỉ ghi phần khác (`_base_: base.yaml`) |
| `src/fer/data/prepare.py` | `data/raw/{train,test}` → `fer48.npz` (train / val / test + mean/std tập train) |
| `src/fer/data/dataset.py` | `FERDataset`, augmentation (chỉ train), DataLoader |
| `src/fer/models/` | `small`, `medium`, `medium01`, `large` (theo báo cáo mẫu), `vgg16` transfer |
| `src/fer/train.py` | Vòng huấn luyện: checkpoint, CSV log, EarlyStopping, ReduceLROnPlateau, resume |
| `src/fer/evaluate.py` | Acc / Macro-F1 Public & Private, confusion matrix, true/wrong cases, learning curves |
| `src/fer/visualize.py` | Filter conv đầu / cuối, activation maximization, feature map, Grad-CAM |
| `src/fer/summarize.py` | Gom `metrics.json` → `results/summary.csv` / `summary.md` |
| `tests/` | Smoke test end-to-end |

`data/raw`, `data/processed`, `runs/`, `*.pt` không đưa lên Git. `results/` (ảnh + json) được commit để viết báo cáo.
