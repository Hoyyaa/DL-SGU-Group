"""BTL02 – Nhận diện cảm xúc khuôn mặt FER2013 (PyTorch)."""

import sys

# Console Windows mặc định cp1252 → print tiếng Việt gây UnicodeEncodeError làm chết run giữa chừng.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and (_stream.encoding or "").lower().replace("-", "") != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass  # stream bị thay thế (pytest capture, notebook) – để nguyên

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTION_LABELS)
