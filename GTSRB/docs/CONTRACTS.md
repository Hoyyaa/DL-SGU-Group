# Hợp đồng giữa các module

Mọi module phải khớp các điều dưới đây. Nếu cần đổi, mở Issue và báo cả nhóm trước khi sửa code.
Điều nào cũng có test tương ứng trong `tests/test_contracts.py` (test tự bỏ qua nếu module chưa tồn tại).

## 1. Dữ liệu (đã làm xong, A)

| Thành phần | Quy ước |
| --- | --- |
| `data/splits/split_manifest.csv` | Cột `path,label,track,x1,y1,x2,y2,split`. `split` là `train` hoặc `val`. **Thứ tự dòng = thứ tự ảnh trong cache. Không sắp xếp lại.** |
| `data/processed/gtsrb64.npz` | `x` (N,64,64,3) uint8 đã cắt ROI và resize; `y` (N,) int64; `paths`; `size`; `use_roi`. |
| `data/splits/norm_stats.json` | `{"mean": [3 số], "std": [3 số]}`, tính trên ảnh [0,1] của tập train. |
| `build_dataloaders(cfg, fast=False)` | Trả về `(train_loader, val_loader)`. Mỗi batch `x`: (B,3,64,64) float32 **trong [0,1], chưa chuẩn hóa**; `y`: (B,) int64 trong 0..42. |
| Tập test | `gtsrb64_test.npz`, cùng định dạng. Chỉ `evaluate.py` được dùng, và chỉ sau khi gắn tag `exp-v1`. |

Ảnh luôn là [0,1] khi đi qua các bước làm tối / làm mờ. **Chuẩn hóa mean/std nằm trong mô hình** (xem mục 2).

## 2. Mô hình (B)

```python
from gtsrb.models import build_model
model = build_model("resnet18")   # "small_cnn" | "resnet18" | "plain18"
logits = model(x)                 # x: (B,3,64,64) trong [0,1]  ->  logits: (B,43)
```

- `build_model(name, num_classes=43, mean=None, std=None)`. Nếu `mean` là None thì đọc `data/splits/norm_stats.json`.
- Mô hình tự chuẩn hóa đầu vào (lớp `Classifier(net, mean, std)` bọc mạng thật). Mạng thật lấy qua `model.net`.
- ResNet-18: stem kiểu CIFAR (conv 3×3 stride 1 rồi MaxPool(3,2,1)), ra 64×32×32 rồi qua 4 stage, kết thúc 512×4×4.
- Plain-18 = ResNet-18 bỏ phép cộng và các shortcut 1×1, mọi thứ khác giữ nguyên.
- Số tham số: small_cnn khoảng 99.019; resnet18 khoảng 11,19 triệu; plain18 khoảng 11,02 triệu.
- Grad-CAM lấy lớp cuối ở `model.net.layers[-1]`.
- Chi tiết kiến trúc và code mẫu: mục 6.3 của tài liệu kế hoạch.

## 3. Làm tối và làm mờ (C), `src/gtsrb/data/corruptions.py`

```python
CONDITIONS = {"orig": ["-"], "dark": [0.70, 0.55, 0.35], "blur": [1.0, 1.5, 2.5]}
darken(x, factor)   # x: (B,3,H,W) trong [0,1]; trả về x * factor, vẫn trong [0,1]
blur(x, sigma)      # Gaussian blur, kernel 2*ceil(3*sigma)+1, giữ nguyên shape, vẫn trong [0,1]
apply_condition(x, condition, level)   # condition trong CONDITIONS; level='-' cho "orig" (trả lại x nguyên)
```

Các mức trong `CONDITIONS` đã chốt trước khi chạy, không đổi sau khi thấy kết quả.

## 4. Huấn luyện (A), `python -m gtsrb.train`

```bash
python -m gtsrb.train --config configs/e2_resnet18.yaml --seed 42 --out_dir runs [--fast] [--resume]
```

Thư mục run: `<out_dir>/<name>_s<seed>/` (ví dụ `runs/e2_s42/`, `name` lấy từ config). Gồm:

| File | Nội dung |
| --- | --- |
| `best.pt` | dict `{"model": state_dict, "cfg": ..., "epoch": ..., "val_macro_f1": ...}`; chọn theo Macro-F1 val trên ảnh gốc |
| `last.pt` | trạng thái cuối mỗi epoch, để `--resume` |
| `config.yaml` | config đã gộp, kèm seed |
| `log.csv` | `epoch,train_loss,val_loss,val_acc,val_macro_f1,lr,sec` |
| `env.json` | commit hash, phiên bản torch / CUDA, tên GPU, Python |

Seed luôn lấy từ `--seed`, không ghi cứng trong YAML thí nghiệm. `best.pt` và `last.pt` không vào Git.

## 5. Chấm điểm (C), `python -m gtsrb.evaluate`

```bash
python -m gtsrb.evaluate --run runs/e2_s42 --conditions orig,dark,blur
```

Ghi `results/eval/<name>_s<seed>.csv` (mỗi run một file, để nhiều người commit không xung đột) với các cột:

`exp, seed, condition, level, macro_f1, acc`

- `level` là chuỗi: `-`, `0.70`, `0.55`, `0.35`, `1.0`, `1.5`, `2.5`.
- `macro_f1` và `acc` tính theo thang 0–1.
- Đọc CSV bằng `pd.read_csv(p, dtype={"level": str})`.
