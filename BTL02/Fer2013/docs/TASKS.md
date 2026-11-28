# Kế hoạch & phân công BTL02 – FER2013

Phân công theo yêu cầu đề: **trục dọc** = nhóm công việc có người chịu trách nhiệm,
**trục ngang** = chia nhỏ thành task cho từng thành viên. Điền tên vào cột "Phụ trách" / "Thực hiện".

## Trục dọc – nhóm công việc

| WP | Nội dung | Phụ trách | Đầu ra chính |
|---|---|---|---|
| WP1 | Dữ liệu & EDA | | `src/data/`, `outputs/figures/eda/`, mục 1 báo cáo |
| WP2 | Custom CNN (RQ1) | | `src/models/cnn.py`, EXP01–EXP05 |
| WP3 | Transfer learning (RQ2) | | `src/models/pretrained.py`, EXP06–EXP10 |
| WP4 | Ablation: activation, optimizer, aug, loss, scheduler (RQ3) | | `experiments/ablation/` |
| WP5 | Trực quan, debug, giải thích (RQ4) + demo | | `src/evaluate/`, `demo/app.py` |
| WP6 | Báo cáo, slide, nhật ký, đóng gói nộp bài | | `docs/`, file zip `BTL02_Emotion_HoTen_MaSV.zip` |

## Trục ngang – danh sách task

| Task | WP | Mô tả | Thực hiện | Trạng thái |
|---|---|---|---|---|
| T0101 | WP1 | Tải & kiểm tra FER2013 (`scripts/download_data.py`) | | ☐ |
| T0102 | WP1 | EDA: histogram lớp, ảnh mẫu (`python -m src.data.explore`) | | ☐ |
| T0103 | WP1 | Chạy `pytest` trên máy/Colab của mỗi thành viên | | ☐ |
| T0201 | WP2 | EXP01 Small CNN (baseline) | | ☐ |
| T0202 | WP2 | EXP02–EXP03 Medium CNN | | ☐ |
| T0203 | WP2 | EXP04 Large CNN, EXP05 Large + BN + GAP | | ☐ |
| T0301 | WP3 | EXP06 VGG16 48×48 (tái hiện fail case) | | ☐ |
| T0302 | WP3 | EXP07 VGG16 128, EXP08 ResNet50 128 | | ☐ |
| T0303 | WP3 | EXP09 EfficientNet-B0, EXP10 ResNet50 scratch | | ☐ |
| T0401 | WP4 | Ablation activation + dropout/BN | | ☐ |
| T0402 | WP4 | Ablation optimizer + scheduler | | ☐ |
| T0403 | WP4 | Ablation augmentation + loss (Disgust, macro-F1) | | ☐ |
| T0501 | WP5 | Confusion matrix, classification report cho model tốt nhất | | ☐ |
| T0502 | WP5 | Filter tầng conv cuối + feature maps + Grad-CAM | | ☐ |
| T0503 | WP5 | Phân tích true/wrong cases, giải thích thành công/thất bại | | ☐ |
| T0504 | WP5 | Demo Streamlit | | ☐ |
| T0601 | WP6 | Tổng hợp `outputs/logs/experiments_summary.csv` → Google Sheets | | ☐ |
| T0602 | WP6 | Viết báo cáo (cấu trúc theo `demo_report.pdf`) + slide | | ☐ |
| T0603 | WP6 | Đóng gói nộp: code + report + slide + nhật ký + bảng phân công | | ☐ |

## Công cụ cộng tác
- Code: GitHub (nhánh `feat/BTL02_<Task>_<ten>` hoặc `exp/BTL02_<Task>_<ten>`, PR vào `BTL02`)
- Kết quả thí nghiệm (run dir, checkpoint): Google Drive
- Nhật ký thí nghiệm: Google Sheets (dán từ `experiments_summary.csv`) + `docs/experiences/`
- Hình & slide: Google Slides / Canva — Kế hoạch & viết: Google Docs
