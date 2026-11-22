# Việc cần làm (dán từng mục vào một GitHub Issue)

Quy tắc chung: đọc `docs/CONTRACTS.md` trước. Mỗi việc là một nhánh, một PR, có test xanh.
Phần nào chờ phần khác thì dùng mô hình giả (stub) để không bị chặn.

## A. Nhóm trưởng: dữ liệu và huấn luyện

**A0. Dọn commit gần đây** (15 phút, làm đầu tiên)
- `dataset.py` đang có `from src.gtsrb...`: đổi lại thành `from gtsrb...`.
- `cache_check.png` đã bị commit nhầm: `git rm --cached cache_check.png` rồi thêm dòng `/*.png` vào `GTSRB/.gitignore` (chỉ chặn ảnh nằm ở thư mục gốc; ảnh minh họa chính thức để trong `docs/figures/`).

**A1. Chạy thử `dataset.py` với PyTorch thật**: xong (x trong [0,1], float32, 3.137 train / 783 val với `--fast`).

**A2. `models/__init__.py` (Classifier, build_model, NETS) và `models/small_cnn.py`** (nhánh `feat/train`, cùng PR với A3)
- `SmallCNN(num_classes=43)` có 99.019 tham số. `build_model` có registry `NETS`; B chỉ cần thêm dòng cho resnet18 và plain18.
- Xong khi: `tests/test_contracts.py` phần small_cnn xanh.

**A3. `src/gtsrb/train.py` và `configs/e1` đến `e6`** (nhánh `feat/train`)
- CLI theo mục 4 của CONTRACTS: `--config --seed --out_dir --fast --resume`.
- AdamW, cosine, AMP, chọn `best.pt` theo Macro-F1 val.
- Xong khi: `pytest -q` xanh (gồm `test_train_smoke.py`), `--fast` chạy hết trên E1 với dữ liệu thật, `--resume` tiếp tục đúng epoch.
- Việc này chặn B và C ở bước chạy thật, nên làm và merge trước.

**A4. Tuning E1 rồi chạy E1 và E3** (3 seed). Ghi từng run vào Google Sheet.
**A5. Notebook 00 và 01, README tái lập, báo cáo chương 1, 3, 5.**

## B. Mô hình

**B1. `src/gtsrb/models/resnet18.py`, `plain18.py` và đăng ký vào `NETS`** (nhánh `feat/models`, bắt đầu sau khi PR `feat/train` của A được merge)
- `BasicBlock(cin, cout, stride=1, skip=True)`, `ResNet18(num_classes=43, skip=True)`; Plain-18 là `ResNet18(skip=False)`. Đăng ký hai dòng vào `NETS` (xem chú thích trong `models/__init__.py`); `Classifier` và `build_model` đã do A viết.
- Xong khi: `tests/test_contracts.py` xanh cho resnet18 và plain18 (số tham số 1,11e7–1,13e7 và 1,09e7–1,11e7; forward (2,3,64,64) ra (2,43)).
- Trong lúc chờ PR của A, B viết và thử lớp mô hình ngay trong nhánh của mình.

**B2. Tuning lr và weight decay cho E2** sau khi A3 merge: lưới lr ∈ {1e-3, 3e-4} × wd ∈ {1e-4, 1e-2}, 10 epoch, seed 42. Chọn theo Macro-F1 val rồi sửa `configs/base.yaml`. Gắn tag `exp-v1`.

**B3. Notebook 03 (mô hình) và chạy E2, E6** (3 seed), ghi từng run vào Google Sheet. Báo cáo chương 2 (CNN, ResNet, skip connection).

## C. Đánh giá, nhật ký, demo, báo cáo (khối lượng nhẹ hơn nhưng độc lập)

**C0. Thư mục Drive chung và Google Sheet nhật ký** (30 phút, làm ngay)
- Cấu trúc và cột theo mục 6 của CONTRACTS. Chia quyền sửa cho cả nhóm, quyền xem cho giảng viên khi nộp.
- Điền sẵn sáu dòng E1–E6 (trạng thái "chưa chạy") theo ma trận thí nghiệm.

**C1. `src/gtsrb/data/corruptions.py`** (nhánh `feat/corruptions`)
- Theo mục 3 của CONTRACTS. Chỉ cần PyTorch và torchvision, không phụ thuộc mô hình hay dữ liệu thật.
- Xong khi: `tests/test_contracts.py` phần corruptions xanh; thêm test riêng: tối làm giảm trung bình ảnh, mờ làm giảm phương sai ảnh, kết quả luôn trong [0,1].

**C2. `src/gtsrb/evaluate.py`**
- CLI theo mục 5 của CONTRACTS; dùng `torch.load(best.pt)` rồi `build_model` từ `cfg` trong checkpoint.
- Ghi thêm `results/efficiency.csv` (params, latency) theo mục 5.
- Viết trước bằng mô hình giả; khi PR của A merge thì thử trên checkpoint `--fast` thật.
- Xong khi: ra đúng file CSV với đúng cột.

**C3. Chạy E4 và E5** (3 seed), ghi từng run vào Google Sheet, chép log lên Drive.

**C4. Notebook 04 (tổng hợp và kết luận RQ) và 05 (Grad-CAM, ma trận nhầm lẫn, ví dụ sai), demo Gradio.**

**C5. Slide, báo cáo chương 4, tài liệu tham khảo, bảng phân công và tỷ lệ đóng góp.** Slide theo khung của file mẫu: Introduction, Related Works, Proposed Method, Experiments and Discussion, Conclusion.

## Thứ tự

1. A làm A0, rồi A2 và A3 trong một PR (`feat/train`). Cùng lúc, C làm C0, C1, C2; B viết mô hình trong nhánh riêng.
2. Khi PR `feat/train` được merge, B đăng ký resnet18 và plain18, C thử `evaluate.py` trên checkpoint thật, mọi người chạy được `--fast`.
3. Cả nhóm họp ngắn ngày chạy smoke test đầu tiên để chốt kết quả mốc 1 trong kế hoạch.
