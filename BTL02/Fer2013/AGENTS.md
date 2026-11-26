# AGENTS.md – Quy tắc làm việc cho BTL02 FER2013

Áp dụng cho mọi thành viên và AI coding agent làm việc trong `BTL02/FER2013/`.

## Bất biến (không tự ý đổi)
1. **Chỉ PyTorch.** Không dùng Keras/TensorFlow (code mẫu trong `demo_report.pdf` là Keras – chỉ tham khảo ý tưởng).
2. **Chia tập cố định theo `Usage`** (cột CSV hoặc tiền tố tên file ảnh): Training / PublicTest (validation) / PrivateTest (test).
   Không dùng PrivateTest để chọn checkpoint, early stopping hay tinh chỉnh siêu tham số.
3. **Augmentation chỉ cho tập train**; transform của val/test phải xác định.
4. **Seed 42** cho mọi thí nghiệm báo cáo chính.
5. **Không commit** `data/raw/*`, `outputs/`, `mlruns/`, `*.pt`. Kết quả nặng đưa lên Google Drive.

## Cách thêm thành phần
- **Thí nghiệm mới:** tạo YAML trong `experiments/` (chỉ ghi phần khác `configs/base.yaml`), `exp_id` duy nhất.
  Ablation nhiều biến thể dùng dạng `sweep: true / base / variants`.
- **Mô hình mới:** kế thừa `BaseEmotionModel`, cài `forward()` trả logits `[B, 7]` và `get_target_layer()`
  (tầng conv cuối cho Grad-CAM), đăng ký trong `build_model()`; thêm test shape vào `tests/test_models.py`.
- **Loss / optimizer / scheduler mới:** thêm nhánh trong `build_loss` / `build_optimizer` / `build_scheduler`, kèm test.

## Quy trình
- Nhánh: `feat/BTL02_<TaskID>_<ten>` (code) hoặc `exp/BTL02_<TaskID>_<ten>` (thí nghiệm); PR vào nhánh `BTL02`.
- Trước khi mở PR: `pytest -q` phải pass.
- Bằng chứng trong `docs/`:
  - `daily_logs/<YYMMDD>_<HHMM>.md` – nhật ký ngày
  - `experiences/<EXP_ID>_<YYMMDD>_<HHMM>.md` – mỗi thí nghiệm quan trọng (có phân tích, không chỉ con số)
  - `fix_logs/<YYMMDD>_<HHMM>_<van_de>.md` – lỗi kỹ thuật đáng ghi lại
  - `reports/<TaskID>_<YYMMDD>_<HHMM>.md` – nghiệm thu task
- Số liệu trong báo cáo phải truy được về `outputs/runs/<exp_id>/metrics.json` hoặc `experiments_summary.csv`.
  Không bịa hoặc làm tròn "đẹp" kết quả.

## Docstring & comment
Viết tiếng Việt, ngắn gọn; giải thích *vì sao* hơn là *làm gì*.
