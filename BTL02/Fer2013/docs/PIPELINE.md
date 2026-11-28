# Thiết kế pipeline BTL02 – FER2013

## 1. Đề bài yêu cầu gì → pipeline đáp ứng ở đâu

| Yêu cầu (Emotion_Recognition.pdf) | Thành phần |
| --- | --- |
| FER2013, 7 lớp, giữ chia Training / PublicTest / PrivateTest | `data/prepare.py` đọc Kaggle `train/`, `test/`; tách `test/` theo tiền tố tên file → val / test (28709 / 3589 / 3589) |
| Input 48×48, output vector 7 chiều | Mọi mô hình: `(B,1,48,48)` trong [0,1] → logits `(B,7)` |
| Cải tiến bằng thiết kế CNN (Conv, MaxPool, Dropout, FC) | 4 preset của báo cáo mẫu + BatchNorm (`models/`) |
| Kỹ thuật huấn luyện: optimizer, data augmentation | `train.optimizer`, `train.scheduler`, `aug.*` trong config |
| Activation: ReLU, sigmoid, … | `model.activation`: relu, leaky_relu, elu, gelu, tanh, sigmoid |
| Kết quả nhiều thí nghiệm với tham số khác nhau | 15 config `e1`–`e15` + `--set`, bảng `results/summary.md` |
| **Visualize the final convolution filters** | `visualize.py`: trọng số conv cuối, activation maximization, feature map, Grad-CAM |
| Phân tích, giải thích kết quả; giải thích thành công / thất bại | confusion matrix, true / wrong cases, Grad-CAM trên ảnh đúng và sai |
| Code PyTorch | Toàn bộ, không Keras |

## 2. Các bước

1. **prepare** – đọc `data/raw/{train,test}`, lưu `x (N,48,48) uint8, y, split, files, mean, std`. Chạy một lần.
   `train/` → train; trong `test/`: `PublicTest_*` → val, `PrivateTest_*` → test.
   Không có tiền tố thì val = 10% train (theo lớp), test = toàn bộ `test/`.
2. **train** – train trên `train`, theo dõi `val`.
   - Chọn `best.pt` theo `val_acc`.
   - EarlyStopping theo `early_stop_patience` (mặc định 10, như báo cáo mẫu).
   - ReduceLROnPlateau theo `val_loss` (factor 0.1, patience 3). Báo cáo mẫu tắt callback này nên e1–e4 dùng `scheduler: none` để tái lập đúng.
   - `log.csv`: `epoch, train_loss, train_acc, val_loss, val_acc, val_macro_f1, lr, sec`.
3. **evaluate** – chấm `best.pt` trên val và **test**. Test là con số chính để báo cáo vì val đã
   dùng để chọn checkpoint (hơi lạc quan). So với mốc world-best 69% (PublicTest) / 71% (PrivateTest).
4. **visualize** – các hình giải thích mô hình (mục 5).
5. **summarize** – bảng so sánh mọi run cho báo cáo.
6. **W&B (tùy chọn)** – `--wandb` khi train; evaluate / visualize tự nối vào cùng run qua
   `wandb_id.txt`. Code ở `utils/wandb_logger.py`; mọi lỗi W&B chỉ cảnh báo, không làm dừng việc train.

## 3. Quy ước (không đổi khi chưa báo cả nhóm)

- Thứ tự nhãn: `0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral`.
  Thư mục Kaggle xếp theo ABC (angry, disgust, fear, happy, **neutral, sad, surprise**) được ánh xạ lại về thứ tự này.
- Batch đưa vào mô hình luôn ở [0,1]; mô hình tự chuẩn hóa (`FerCNN`: mean/std tập train, `vgg16`: ImageNet).
- **Augmentation chỉ cho train.** val / test không bao giờ bị biến đổi (`test_no_aug_on_eval`).
- Tập test chỉ được đụng trong `evaluate.py` / `visualize.py`.
- Seed mặc định 42; thư mục run là `<name>_s<seed>`; `train.py` không ghi đè run đã có.
- Mô hình mới: viết lớp có `forward`, `first_conv()`, `last_conv()`, `cam_layer()` rồi thêm một dòng vào `PRESETS`.

## 4. Ma trận thí nghiệm

Nhóm A tái lập báo cáo mẫu. Từ nhóm C trở đi **chỉ đổi một yếu tố so với e5** để so sánh công bằng.

| Exp | Thay đổi | Câu hỏi trả lời trong báo cáo |
| --- | --- | --- |
| e1_small | Small, không BN | Tái lập mẫu (mẫu: val_acc 0.64) |
| e2_medium | Medium, không BN | (mẫu: 0.67) |
| e3_medium01 | Medium01, không BN | (mẫu: 0.65) |
| e4_large | Large, không BN | (mẫu: 0.67). Sâu hơn có tốt hơn? |
| **e5_large_bn** | Large + BatchNorm | **Mô hình tham chiếu.** BN giúp mạng sâu hội tụ? |
| e6_no_aug | tắt augmentation | Augmentation giảm overfit bao nhiêu? (xem khoảng cách train/val trong `curves.png`) |
| e7_aug_erasing | + RandomErasing 0.5 | Che một phần mặt có giúp tổng quát? |
| e8_sgd | SGD Nesterov lr 0.01 | Adam vs SGD |
| e9_adamw_cosine | AdamW + cosine 60 epoch | Lịch lr khác |
| e10_class_weight | CE có trọng số lớp | F1 của lớp Disgust (547 ảnh) có tăng? |
| e11_sigmoid | sigmoid | Bão hòa gradient ở mạng sâu |
| e12_elu / e13_leaky_relu | ELU / LeakyReLU | So với ReLU |
| e14_vgg16_48 | VGG16 ImageNet, ảnh 48 | Fail case của mẫu: 5 MaxPool → đặc trưng 1×1 |
| e15_vgg16_96 | VGG16 ImageNet, phóng lên 96 | Phóng ảnh có cứu được transfer learning? |

Lưu ý khi phân tích: `small` có ~38,9 triệu tham số, nhiều nhất trong 4 mô hình, vì chỉ có 2 lần
MaxPool nên lớp Flatten ra 12×12×256 = 36.864 chiều trước Dense(1024). `medium` ~11,3 triệu, `large` ~20,4 triệu.

## 5. Hình trực quan hóa và cách dùng trong báo cáo

| File | Dùng để nói gì |
| --- | --- |
| `curves.png` | overfit hay underfit, epoch tốt nhất, lúc lr giảm |
| `confusion_test.png` | cặp lớp hay nhầm (Fear↔Sad, Angry↔Disgust, Neutral↔Sad thường gặp) |
| `true_cases.png`, `wrong_cases.png` | ví dụ cụ thể; nhiều ảnh sai do nhãn mơ hồ, che mặt, nghiêng, ảnh không phải mặt |
| `viz/filters_first_conv.png` | conv đầu học bộ lọc cạnh / gradient sáng tối |
| `viz/filters_last_conv.png` | trọng số 3×3 conv cuối (đề bắt buộc); khó diễn giải trực tiếp → dẫn sang actmax |
| `viz/actmax_last_conv.png` | mẫu hình mà mỗi bộ lọc conv cuối "tìm kiếm" |
| `viz/featuremaps_last_conv.png` | bộ lọc nào bật với vùng mắt / miệng ở từng cảm xúc |
| `viz/gradcam.png` | mô hình nhìn vào đâu khi đúng và khi sai |

## 6. Gợi ý chia việc (5 người)

| Người | Việc | File |
| --- | --- | --- |
| 1 (trưởng nhóm) | Tải dữ liệu Kaggle, `prepare`, EDA (histogram 3 tập, ảnh mẫu); chạy e1–e5 | `data/`, `notebooks/` |
| 2 | Nhóm C: e6–e10 | `configs/` |
| 3 | Nhóm D + E: e11–e15 | `configs/`, `models/transfer.py` |
| 4 | Chạy `evaluate`/`visualize` cho mọi run, viết phần phân tích đúng / sai và filter | `results/` |
| 5 | Báo cáo + slide + nhật ký thí nghiệm (từ `results/summary.md`) | `docs/` |

## 7. Khung báo cáo (theo mục "Submission" của đề)

1. Tóm tắt mã nguồn: sơ đồ ở README + bảng cấu trúc + kiến trúc 4 mô hình.
2. Dữ liệu: phân bố lớp (lệch nặng ở Disgust), ảnh mẫu.
3. Kết quả các thí nghiệm: `results/summary.md` + `curves.png` của từng nhóm.
4. Trực quan hóa conv cuối: các hình ở mục 5.
5. Phân tích: confusion matrix, wrong cases, Grad-CAM; vì sao mô hình đúng / sai; so sánh với 69% / 71%.
