# Việc cần làm (dán từng mục vào một GitHub Issue)

Quy tắc chung: đọc `docs/CONTRACTS.md` trước. Mỗi việc là một nhánh, một PR, có test xanh.
Phần nào chờ phần khác thì dùng mô hình giả (stub) để không bị chặn.

## A. Nhóm trưởng: dữ liệu và huấn luyện

**A1. Chạy thử `dataset.py` với PyTorch thật** (30 phút)
- Lấy một batch từ `build_dataloaders(cfg, fast=True)`. Kỳ vọng `x.shape == (64,3,64,64)`, float32, min ≥ 0, max ≤ 1.
- Xong khi: dán kết quả vào PR; nếu min/max ngoài [0,1] thì sửa `build_train_transform`.

**A2. `src/gtsrb/models/small_cnn.py`** (nhánh `feat/small-cnn`)
- Lớp `SmallCNN(num_classes=43)`, nhận ảnh đã chuẩn hóa, ra logits (B,43). Khoảng 99.019 tham số (mục 6.3 kế hoạch).
- Xong khi: test số tham số trong khoảng 9,5e4 đến 1,05e5; forward (2,3,64,64) ra (2,43).

**A3. `src/gtsrb/train.py`** (nhánh `feat/train`)
- CLI theo mục 4 của CONTRACTS: `--config --seed --out_dir --fast --resume`.
- AdamW, scheduler cosine, AMP, chọn `best.pt` theo Macro-F1 val (dùng `sklearn.metrics.f1_score(average="macro")`).
- Ghi `log.csv`, `config.yaml`, `env.json`, `best.pt`, `last.pt`.
- Xong khi: `--fast` chạy hết 1–2 epoch trên ResNet-18 hoặc SmallCNN; `--resume` tiếp tục đúng epoch; chạy lại cùng seed ra loss epoch đầu gần giống nhau.
- Việc này chặn B và C ở bước chạy thật, nên làm sớm.

**A4. Notebook 00 (Colab setup) và 01 (EDA)**; **A5. README tái lập** (làm dần, xong trước ngày nộp).

## B. Mô hình

**B1. `src/gtsrb/models/resnet18.py`, `plain18.py`, `__init__.py`** (nhánh `feat/models`)
- `BasicBlock(cin, cout, stride=1, skip=True)`, `ResNet18(num_classes=43, skip=True)`, `Plain18` = ResNet18 với `skip=False`.
- `build_model(name, num_classes=43, mean=None, std=None)` và `Classifier(net, mean, std)` theo mục 2 của CONTRACTS.
- Xong khi: test số tham số (resnet18: 1,11e7–1,13e7; plain18: 1,09e7–1,11e7), forward (2,3,64,64) ra (2,43), và đổi `mean` thì output đổi (chuẩn hóa nằm trong mô hình).
- `build_model("small_cnn")` import `SmallCNN` từ A2; trong lúc chờ có thể dùng lớp giả.

**B2. Tuning lr và weight decay cho E2** sau khi A3 merge: lưới lr ∈ {1e-3, 3e-4} × wd ∈ {1e-4, 1e-2}, 10 epoch, seed 42. Chọn theo Macro-F1 val rồi sửa `configs/base.yaml`. Gắn tag `exp-v1`.

**B3. Notebook 03 (mô hình) và chạy E2, E6** (3 seed). Phần phương pháp trong báo cáo.

## C. Đánh giá, demo, báo cáo (khối lượng nhẹ hơn nhưng độc lập)

**C1. `src/gtsrb/data/corruptions.py`** (nhánh `feat/corruptions`)
- Theo mục 3 của CONTRACTS. Chỉ cần PyTorch và torchvision, không phụ thuộc mô hình hay dữ liệu thật.
- Xong khi: `tests/test_contracts.py` phần corruptions xanh; thêm test riêng: tối làm giảm trung bình ảnh, mờ làm giảm phương sai ảnh, kết quả luôn trong [0,1].

**C2. `src/gtsrb/evaluate.py`**
- CLI theo mục 5 của CONTRACTS; dùng `torch.load(best.pt)` rồi `build_model` từ `cfg` trong checkpoint.
- Dùng mô hình giả (nhận (B,3,64,64), ra (B,43)) để viết trước khi A3 và B1 xong.
- Xong khi: chạy trên checkpoint của `train.py --fast`, ra đúng file CSV với đúng cột.

**C3. Notebook 04 (tổng hợp và kết luận RQ) và 05 (Grad-CAM, phân tích lỗi), demo Gradio, poster, slide.** Chạy E4 và E5 (3 seed).

## Thứ tự

1. A merge PR dữ liệu hiện tại. B và C bắt đầu B1, C1, C2 song song.
2. A làm A1, A2, A3. Khi A3 merge thì B và C chạy được `--fast` thật.
3. Cả nhóm họp ngắn ngày chạy smoke test đầu tiên để chốt kết quả mốc 1 trong kế hoạch.
