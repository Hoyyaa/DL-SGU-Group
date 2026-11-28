# Dữ liệu FER2013

Không commit dữ liệu lên Git. Pipeline đọc được **2 định dạng** (đặt `data_path` trong `configs/base.yaml`):

### 1. Thư mục ảnh – mặc định (`data_path: data/raw`)

Bản Kaggle [`msambare/fer2013`](https://www.kaggle.com/datasets/msambare/fer2013), ảnh xám 48×48:

```text
data/raw/
├── train/<emotion>/Training_*.jpg                    # 28.709 ảnh
└── test/<emotion>/PublicTest_*.jpg, PrivateTest_*.jpg # 3.589 + 3.589 ảnh
```

- `<emotion>` ∈ `angry, disgust, fear, happy, sad, surprise, neutral` → nhãn 0–6 theo `EMOTION_LABELS`.
- `Usage` lấy từ **tiền tố tên file**, nên dù `test/` gộp cả hai tập vẫn tách lại đúng PublicTest / PrivateTest gốc.
- Lần đầu đọc (~vài phút trên Windows) sẽ tạo cache `data/processed/fer2013_raw.npz` (~80 MB);
  các lần sau chỉ vài giây. Cache tự làm mới khi số ảnh thay đổi; muốn ép đọc lại thì xóa file này.

### 2. CSV gốc của cuộc thi (`data_path: data/raw/fer2013.csv`)

| Cột | Ý nghĩa |
|---|---|
| `emotion` | 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral |
| `pixels` | 2304 giá trị 0–255 cách nhau bởi dấu cách (ảnh xám 48×48) |
| `Usage` | `Training` (28.709) · `PublicTest` (3.589) · `PrivateTest` (3.589) |

## Cách lấy

Cần `pip install kagglehub` và API token Kaggle ở `~/.kaggle/kaggle.json`.

```bash
python scripts/download_data.py            # bản thư mục ảnh → data/raw/train, data/raw/test
python scripts/download_data.py --csv      # bản CSV (phải bấm Join/Accept rules của cuộc thi trước)
python scripts/download_data.py --verify   # chỉ kiểm tra dữ liệu đã có (thêm --path <...> cho CSV)
```

Tải thủ công thì giải nén sao cho đúng cấu trúc trên rồi chạy `--verify`; phải thấy cả 3 dòng `[OK]`.

## Quy ước chia tập (không thay đổi)
- `Training` → huấn luyện
- `PublicTest` → validation: chọn checkpoint, early stopping, giảm LR
- `PrivateTest` → test cuối, chỉ dùng để báo cáo (mốc world-best: 69% public / 71% private)
