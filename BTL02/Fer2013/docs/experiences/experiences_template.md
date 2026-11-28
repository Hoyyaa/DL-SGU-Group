# [EXPERIMENT] <EXP_ID>: <Tiêu đề ngắn>

- **Tên file:** `docs/experiences/<EXP_ID>_<YYMMDD>_<HHMM>.md`
- **Người thực hiện:** <Tên thành viên>
- **Task ID / Nhánh Git:** `T0xxx` / `exp/BTL02_T0xxx_<ten>`
- **Config:** `experiments/<...>.yaml` (+ override CLI nếu có)
- **Run dir:** `outputs/runs/<EXP_ID>/` (đã upload Google Drive: <link>)
- **Trạng thái:** Success | Failed | Inconclusive

## 1. Mục tiêu & giả thuyết
- **RQ liên quan:** RQ1 | RQ2 | RQ3 | RQ4
- **Giả thuyết:** …
- **Tiêu chí chấp nhận:** ví dụ private acc tăng ≥ 1 điểm so với <EXP_ID đối chứng>.

## 2. Thiết lập (copy từ `outputs/runs/<EXP_ID>/config.yaml`)
| Mục | Giá trị |
|---|---|
| Model / input | `medium_cnn`, 48×48×1 |
| Activation / BN / Dropout | |
| Optimizer / LR / WD | |
| Scheduler | |
| Loss | |
| Augmentation | |
| Batch / Epochs / Patience | |
| Seed / GPU | 42 / |

## 3. Kết quả (từ `metrics.json`)
| best_epoch | train_acc | public_acc (val) | private_acc (test) | macro-F1 | params (M) | thời gian (phút) |
|---|---|---|---|---|---|---|
| | | | | | | |

Hình: `figures/training_curves.png`, `figures/confusion_matrix.png`, `figures/last_conv_filters.png`, `figures/gradcam_per_class.png`, `figures/wrong_cases.png`.

## 4. Phân tích (bằng lời của nhóm)
- **Hội tụ / overfitting:** khoảng cách train–val, epoch tốt nhất, LR giảm khi nào.
- **Theo lớp:** lớp tốt nhất / tệ nhất (classification_report.csv); cặp hay nhầm (top_confusions.csv).
- **Grad-CAM & filter:** mô hình nhìn vào vùng nào (mắt, mày, miệng)? Filter tầng cuối có cấu trúc hay nhiễu?
- **Fail cases:** nguyên nhân (che khuất, góc nghiêng, nhãn nhiễu, ảnh mờ…).

## 5. Kết luận & bước tiếp theo
- Giả thuyết: chấp nhận / bác bỏ.
- Thí nghiệm tiếp theo: …
