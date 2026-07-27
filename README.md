# AI Avatar - Hệ Thống Animation Khuôn Mặt

> **Stage 1:** Face morphing (Delaunay) — `avatar_all_in_one_v1.ipynb`
> **Stage 2:** Neural AI Avatar (FOMM + MediaPipe) — `ai_avatar_liveportrait.ipynb` ⭐

---

## Module Chính

### Module 2 (Mới): AI Avatar Neural — LivePortrait-Style

**File:** `notebooks/ai_avatar_liveportrait.ipynb`

Hệ thống AI Avatar dùng pretrained neural network (FOMM) để tạo khuôn mặt ảo hoạt động tự nhiên, điều khiển bởi webcam theo thời gian thực.

**Kiến trúc:**

```
[Ảnh input] → Identity Builder (InsightFace + MediaPipe) → Canonical Face
                                                              ↓
[Webcam]    → Motion Tracker (MediaPipe 468 landmarks)    → Yaw/Pitch/Roll/Blink/Mouth
                                                              ↓
              Avatar Renderer (FOMM Neural Network)       → Animated Face
                                                              ↓
              Controller FSM (Idle / Look / Smile / Blink / Talk)
```

**Không dùng:** Delaunay triangulation, convex hull morphing, triangle blending, frame interpolation
**Dùng:** Dense motion fields từ neural network, pretrained trên VoxCeleb

### Module 1 (Cũ): Face Morphing

### Module 1: AI Avatar Điều Khiển Thời Gian Thực

Từ asset khuôn mặt (ảnh hoặc video), tạo ra khuôn mặt ảo có thể điều khiển quay trái/phải/thẳng theo lệnh giọng nói hoặc bàn phím, với 2 chế độ:

- **Chế độ Fixed:** Nội suy mượt giữa 3 góc cố định (trái/thẳng/phải)
- **Chế độ Motion:** Học chuyển động tự nhiên từ video tham chiếu, playback lại đúng đường cong chuyển động của người thật

### Module 2: Auto-Train Từ Video

Đưa vào 1 video ngắn bất kỳ (người đang quay đầu trái-phải), hệ thống sẽ:

- Tự động phát hiện khuôn mặt trong từng frame
- Tự động phân loại góc: trái, thẳng, phải
- Tự động capture frame tốt nhất, gán nhãn, lưu vào dataset
- Tự động trích xuất motion curve để Module 1 dùng
- Hỗ trợ cả ảnh và video, đa dạng góc gần/xa

---

## Cấu Trúc Thư Mục

```
train ai/
├── data/
│   ├── angles/                  # Dataset khuôn mặt đã gán nhãn
│   │   ├── left/                #   Ảnh xoay trái
│   │   ├── center/              #   Ảnh nhìn thẳng
│   │   └── right/               #   Ảnh xoay phải
│   └── driving_videos/          # Video motion reference
├── models/                      # Model + motion curve JSON
├── notebooks/
│   └── avatar_all_in_one_v1.ipynb  # ⭐ Google Colab All-in-One (v1)
├── src/
│   ├── collect_angles.py        # Chụp ảnh 3 góc HOẶC quay video
│   ├── auto_train.py            # MODULE 2: Auto-train từ video/ảnh
│   ├── live_avatar.py           # MODULE 1: AI Avatar thời gian thực
│   └── utils/
│       ├── face_mesh_utils.py   # MediaPipe 3D face mesh
│       ├── face_warp_utils.py   # FaceRotator + MotionSequence
│       ├── video_utils.py       # Xử lý video, auto-capture
│       └── voice_utils.py       # Nhận diện giọng nói
├── requirements.txt
└── README.md
```

---

## Cài Đặt

```bash
# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

> Nếu lỗi pyaudio trên Windows: `pip install pipwin && pipwin install pyaudio`

---

## Hướng Dẫn Sử Dụng

### ⭐ Cách Chính: Google Colab All-in-One (Khuyên Dùng)

**Chỉ cần 1 file notebook, Colab làm hết.** Không cần GPU máy local, không cần cài gì ngoài VS Code + Colab extension.

```bash
# Bước 1: Upload video/ảnh vào Google Drive
# Tạo folder: AI_Face_Data/input/
# Bỏ video (.mp4) hoặc ảnh (.jpg) vào đó

# Bước 2: Mở notebook trong VS Code
# File: notebooks/avatar_all_in_one_v1.ipynb
# Ctrl+Shift+P -> "Colab: Connect to Colab"
# Runtime -> Change runtime type -> T4 GPU

# Bước 3: Chạy Run All
# Colab tự động: phát hiện mặt -> phân loại góc -> face warp -> tạo video -> live demo
```

**Kết quả đầu ra trong Drive:** `AI_Face_Data/output/`

- `avatar_animation.mp4` — video animation hoàn chỉnh
- `best_left/center/right.jpg` — ảnh tốt nhất mỗi góc
- `face_data.pkl` — model cho máy local

---

### Cách Phụ: Chạy Trên Máy Local (Cần CPU, Không Cần GPU)

Dành cho test nhanh hoặc demo voice control.

```bash
# Cài thư viện (1 lần)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Chụp 3 ảnh góc
python src/collect_angles.py

# Chạy avatar (điều khiển = giọng nói)
python src/live_avatar.py
```

> ⚠️ Local chỉ chạy chế độ Fixed (nội suy 3 góc). Muốn Motion tự nhiên + upscale ảnh: dùng Colab.

---

## Điều Khiển

### Giọng Nói (tiếng Việt)

| Lệnh                                   | Hành động          |
| -------------------------------------- | ------------------ |
| "Quay trái" / "Nhìn trái" / "Bên trái" | Mặt xoay sang trái |
| "Quay phải" / "Nhìn phải" / "Bên phải" | Mặt xoay sang phải |
| "Nhìn thẳng" / "Chính giữa"            | Mặt về thẳng       |
| "Thoát" / "Dừng lại"                   | Thoát chương trình |

### Bàn Phím

| Phím | Hành động  |
| ---- | ---------- |
| A    | Quay trái  |
| S    | Nhìn thẳng |
| D    | Quay phải  |
| Q    | Thoát      |

---

## Google Colab All-in-One

### Luồng Tự Động Trong Notebook

```
Video/Ảnh Input → Auto phát hiện mặt → Phân loại góc (trái/thẳng/phải)
→ Trích 468 landmarks → Delaunay Face Warp → Tạo video animation
→ Live demo webcam → Xuất model + video về Drive
```

### Cách Dùng Với VS Code + Colab Extension

1. Mở `notebooks/avatar_all_in_one_v1.ipynb` trong VS Code
2. Ctrl+Shift+P → "Colab: Connect to Colab"
3. Runtime → Change runtime type → **T4 GPU**
4. Upload video/ảnh vào Google Drive: `AI_Face_Data/input/`
5. Run All (Ctrl+F9)

### Cấu Trúc Google Drive

```
Google Drive/
└── AI_Face_Data/
    ├── input/              ← BỎ VIDEO/ẢNH VÀO ĐÂY
    │   └── video_cua_ban.mp4
    └── output/             ← KẾT QUẢ TỰ ĐỘNG RA ĐÂY
        ├── avatar_animation.mp4
        ├── best_left.jpg
        ├── best_center.jpg
        ├── best_right.jpg
        └── face_data.pkl
```

---

## Luồng Hoạt Động Chi Tiết

### Module 1: AI Avatar

```
Ảnh/Videos -> MediaPipe 468 landmarks -> Delaunay Triangulation
-> Affine Warp từng tam giác -> Alpha Blend giữa các reference
-> Smooth animation (ease-out + moving average) -> Hiển thị OpenCV
```

### Module 2: Auto-Train

```
Video Input -> Đọc frame -> MediaPipe phát hiện mặt -> Tính yaw
-> Phân loại: left (yaw<-15), center (-15~15), right (yaw>+15)
-> Chọn frame tốt nhất (độ sắc nét cao) -> Lưu vào data/angles/
-> (Optional) Trích motion curve -> Lưu JSON
```

### Chế Độ Motion Reference

```
Video tham chiếu -> Trích yaw curve -> MotionSequence
-> Khi nhận lệnh "quay trái": playback curve đảo ngược
-> Khi nhận lệnh "quay phải": playback curve xuôi
-> Kết quả: chuyển động GIỐNG HỆT người trong video tham chiếu
```

---

## Các Tham Số Dòng Lệnh

### collect_angles.py

| Tham số    | Mặc định | Mô tả                              |
| ---------- | -------- | ---------------------------------- |
| --video    | False    | Chế độ quay video motion reference |
| --duration | 10       | Thời lượng video (giây)            |

### auto_train.py

| Tham số         | Mặc định                 | Mô tả                            |
| --------------- | ------------------------ | -------------------------------- |
| --input         | (bắt buộc)               | Đường dẫn video hoặc thư mục ảnh |
| --type          | video                    | video hoặc images                |
| --output        | data/angles/             | Thư mục lưu ảnh                  |
| --sample        | 3                        | Lấy 1 frame mỗi N frame          |
| --fps           | 30                       | FPS của video nguồn              |
| --motion        | False                    | Trích xuất motion curve          |
| --output-motion | models/motion_curve.json | File JSON output                 |
| --quality       | 0.3                      | Ngưỡng chất lượng ảnh            |

### live_avatar.py

| Tham số                 | Mặc định        | Mô tả                        |
| ----------------------- | --------------- | ---------------------------- |
| --left/--center/--right | data/angles/... | Ảnh 3 góc                    |
| --motion                | None            | File motion curve JSON       |
| --speed                 | 2.5             | Tốc độ xoay (deg/s)          |
| --no-voice              | False           | Tắt giọng nói, dùng bàn phím |

---

## Xử Lý Lỗi Thường Gặp

| Lỗi                             | Cách Sửa                                        |
| ------------------------------- | ----------------------------------------------- |
| Không tìm thấy khuôn mặt        | Đảm bảo ánh sáng tốt, mặt trong khung hình      |
| pyaudio không cài được          | pipwin install pyaudio                          |
| Giọng nói không nhận            | Kiểm tra microphone, thử --no-voice             |
| Ảnh không có trong data/angles/ | Chạy collect_angles.py hoặc auto_train.py trước |
| Video không đọc được            | Chuyển sang định dạng .mp4 (H.264)              |
| Chuyển động bị giật             | Giảm --speed, dùng --motion mode                |

---

## Công Nghệ Sử Dụng

- **MediaPipe Face Mesh**: 468 điểm 3D landmarks
- **Delaunay Triangulation + Affine Warp**: Xoay mặt mượt
- **SpeechRecognition + Google STT**: Nhận diện giọng nói tiếng Việt (miễn phí)
- **pyttsx3**: Phản hồi giọng nói (offline)
- **GFPGAN** (Colab): Super resolution ảnh
- **OpenCV**: Xử lý ảnh, hiển thị real-time
