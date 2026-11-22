# Hướng dẫn cho AI trợ lý (Codex, Claude, Cursor, ...)

Bạn đang giúp một thành viên của nhóm 3 người làm đồ án Học sâu: phân loại biển báo GTSRB
(CNN nhỏ so với ResNet-18, có ablation skip connection và augmentation tối/mờ). Chỉ số chính là Macro-F1.

## Đọc trước khi viết code

1. `docs/CONTRACTS.md`: hình dạng dữ liệu, chữ ký hàm, định dạng file. Phải khớp từng điểm.
2. `docs/TASKS.md`: việc nào của ai. Chỉ làm việc người dùng giao, không sang file của người khác.
3. `tests/test_contracts.py`: là đặc tả chạy được. Làm cho nó xanh, đừng sửa test để nó xanh.

## Quy tắc cứng

- Ảnh trong DataLoader và trong `darken`, `blur` là float32 trong [0,1]. Chuẩn hóa mean/std chỉ nằm trong mô hình.
- Không lật ngang ảnh (làm sai nghĩa biển báo).
- Không chia train/val lại. Dùng `data/splits/split_manifest.csv`; không sắp xếp lại dòng. Cache lệch manifest thì `load_checked_cache` báo lỗi, đừng tắt lỗi.
- Tập test chỉ dùng trong `evaluate.py`, sau khi đã gắn tag `exp-v1`. Không dùng test để chọn lr, epoch hay mô hình.
- Seed lấy từ `--seed`, không ghi cứng trong YAML. Mọi run ghi `config.yaml` và `env.json`.
- Không commit dữ liệu, `.npz`, `.pt`, `runs/`, ảnh kiểm tra tạm (`cache_check.png`...). Không dùng `git add .`.
- Import luôn là `from gtsrb...` (package cài bằng `pip install -e .`). Không bao giờ `from src.gtsrb...`: IDE hay tự gợi ý như vậy và làm hỏng import khi chạy ở nơi khác.
- Không ghi đè thư mục run đã có. Chạy lại cùng seed thì dùng `--resume` hoặc chọn thư mục khác.
- Sau mỗi run thật, ghi một dòng vào Google Sheet nhật ký và đẩy log lên Drive (xem README).
- Không đổi hợp đồng một mình. Nếu cần đổi, nói rõ với người dùng để họ mở Issue.

## Cách làm

- Code ngắn, dễ đọc, dùng thư viện chuẩn (PyTorch, torchvision, sklearn, pandas). Ưu tiên cơ bản hơn tinh vi.
- Chạy trong thư mục `GTSRB/` với `.venv` đã kích hoạt: `pytest -q` trước khi báo xong.
- Mỗi tính năng một nhánh `feat/<tên>`, commit nhỏ, mô tả rõ. Người dùng tự push và mở PR.
- Nếu chưa chạy được code (thiếu dữ liệu, thiếu GPU), nói rõ phần nào chưa kiểm tra.
- Số liệu, kết luận trong báo cáo phải đến từ kết quả chạy thật. Không bịa số, không điều chỉnh ngưỡng sau khi thấy kết quả.
