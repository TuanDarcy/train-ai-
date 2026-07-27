"""
video_utils.py - Xử lý video: trích xuất frame, phân loại góc khuôn mặt.

Module này phục vụ 2 mục đích:
1. AUTO-TRAIN (Module 2): Tự động quét video, phát hiện khuôn mặt,
   phân loại góc (trái/thẳng/phải), capture frame tốt nhất và gán nhãn.
2. MOTION REFERENCE (Module 1): Trích xuất đường cong chuyển động (yaw curve)
   từ video tham chiếu để tạo animation tự nhiên.

Hỗ trợ input: file video (.mp4, .avi, .mov...) và thư mục ảnh.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.face_mesh_utils import get_face_landmarks, estimate_head_pose


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FaceFrame:
    """Một frame chứa khuôn mặt đã phân tích."""
    frame_idx: int
    timestamp: float        # giây
    yaw: float              # độ
    pitch: float
    roll: float
    landmarks_2d: np.ndarray
    image: np.ndarray       # ảnh RGB
    angle_label: str        # "left", "center", "right", "transition"


@dataclass
class MotionCurve:
    """Đường cong chuyển động trích từ video tham chiếu."""
    timestamps: List[float]
    yaws: List[float]
    pitches: List[float]
    rolls: List[float]
    total_duration: float
    source_video: str


# ============================================================
# 1. ĐỌC VIDEO / ẢNH
# ============================================================

def read_frames_from_video(
    video_path: str,
    sample_every: int = 3,
    max_frames: int = 500,
) -> List[np.ndarray]:
    """
    Đọc frame từ video, lấy mẫu cách nhau sample_every frame.

    Args:
        video_path: Đường dẫn video.
        sample_every: Lấy 1 frame mỗi N frame (giảm tải xử lý).
        max_frames: Số frame tối đa cần đọc.

    Returns:
        List[np.ndarray]: Danh sách frame RGB.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    frames = []
    frame_idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        frame_idx += 1

    cap.release()
    return frames


def read_images_from_folder(folder_path: str) -> List[Tuple[str, np.ndarray]]:
    """
    Đọc tất cả ảnh từ thư mục.

    Returns:
        List[Tuple[str, np.ndarray]]: [(tên_file, ảnh_RGB), ...]
    """
    folder = Path(folder_path)
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    results = []

    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in exts:
            img = cv2.imread(str(f))
            if img is not None:
                results.append((f.name, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    return results


# ============================================================
# 2. PHÂN TÍCH KHUÔN MẶT TRONG FRAME
# ============================================================

def analyze_frame(
    image: np.ndarray,
    frame_idx: int,
    fps: float = 30.0,
    sample_every: int = 3,
) -> Optional[FaceFrame]:
    """
    Phân tích 1 frame: tìm khuôn mặt, tính góc đầu, phân loại.

    Returns:
        FaceFrame hoặc None nếu không tìm thấy khuôn mặt.
    """
    result = get_face_landmarks(image)
    if result is None:
        return None

    lm_2d, lm_3d = result
    yaw, pitch, roll = estimate_head_pose(lm_2d, image.shape[:2])

    # Tính timestamp
    timestamp = (frame_idx * sample_every) / fps

    return FaceFrame(
        frame_idx=frame_idx,
        timestamp=timestamp,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        landmarks_2d=lm_2d,
        image=image,
        angle_label=_classify_angle(yaw),
    )


def _classify_angle(yaw: float) -> str:
    """
    Phân loại góc dựa trên yaw.

    Ngưỡng:
        yaw < -15   -> "left"
        -15 <= yaw <= +15 -> "center"
        yaw > +15   -> "right"
    """
    if yaw < -15:
        return "left"
    elif yaw > 15:
        return "right"
    else:
        return "center"


# ============================================================
# 3. TRÍCH XUẤT MOTION CURVE (MODULE 1)
# ============================================================

def extract_motion_curve(
    video_path: str,
    sample_every: int = 2,
    fps: float = 30.0,
) -> Optional[MotionCurve]:
    """
    Trích xuất đường cong chuyển động từ video tham chiếu.

    Video tham chiếu: 1 người nhìn thẳng -> quay trái -> quay phải -> thẳng.

    Args:
        video_path: Đường dẫn video tham chiếu.
        sample_every: Lấy frame mỗi N frame.
        fps: FPS của video gốc.

    Returns:
        MotionCurve hoặc None nếu không tìm thấy khuôn mặt.
    """
    print(f"[Motion] Dang trich xuat motion curve tu: {Path(video_path).name}")

    frames = read_frames_from_video(video_path, sample_every=sample_every)
    if len(frames) == 0:
        print("[ERROR] Khong doc duoc frame nao!")
        return None

    timestamps, yaws, pitches, rolls = [], [], [], []
    face_count = 0

    for i, frame in enumerate(frames):
        ff = analyze_frame(frame, i, fps=fps, sample_every=sample_every)
        if ff is not None:
            timestamps.append(ff.timestamp)
            yaws.append(ff.yaw)
            pitches.append(ff.pitch)
            rolls.append(ff.roll)
            face_count += 1

    if face_count < 5:
        print(f"[ERROR] Chi tim thay {face_count} frame co khuon mat!")
        return None

    total_duration = timestamps[-1] if timestamps else 0

    print(f"[OK] Trich xuat duoc {face_count} diem motion")
    print(f"     Yaw range: {min(yaws):.1f} -> {max(yaws):.1f} deg")
    print(f"     Duration: {total_duration:.1f}s")

    return MotionCurve(
        timestamps=timestamps,
        yaws=yaws,
        pitches=pitches,
        rolls=rolls,
        total_duration=total_duration,
        source_video=str(video_path),
    )


# ============================================================
# 4. AUTO-CAPTURE TỪ VIDEO (MODULE 2)
# ============================================================

def auto_capture_from_video(
    video_path: str,
    output_dir: str,
    sample_every: int = 3,
    fps: float = 30.0,
    quality_threshold: float = 0.3,
) -> Dict[str, int]:
    """
    Tự động quét video, phát hiện khuôn mặt, phân loại góc,
    lưu frame tốt nhất vào data/angles/.

    TIÊU CHÍ CHỌN FRAME TỐT:
    - Khuôn mặt được phát hiện rõ
    - Góc yaw nằm trong ngưỡng của nhãn
    - Không bị motion blur (kiểm tra độ sắc nét)
    - Đa dạng: không lấy quá nhiều frame giống nhau

    Args:
        video_path: Đường dẫn video.
        output_dir: Thư mục output (data/angles/).
        sample_every: Lấy frame mỗi N frame.
        fps: FPS video.
        quality_threshold: Ngưỡng chất lượng Laplacian.

    Returns:
        Dict[str, int]: {"left": N, "center": N, "right": N}
    """
    video_name = Path(video_path).stem
    output = Path(output_dir)

    # Tạo thư mục
    for angle in ['left', 'center', 'right']:
        (output / angle).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AUTO-TRAIN: {Path(video_path).name}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Tong so frame: {total_frames}")
    print(f"Sample rate: 1/{sample_every}")

    # Lưu frame tốt nhất cho mỗi góc
    # Key: angle, Value: (frame, yaw, quality_score)
    best_frames: Dict[str, Tuple[np.ndarray, float, float]] = {}
    captured: Dict[str, int] = {"left": 0, "center": 0, "right": 0}
    face_found = 0
    processed = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        processed += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phân tích frame
        ff = analyze_frame(frame_rgb, frame_idx, fps=fps, sample_every=sample_every)
        if ff is None:
            frame_idx += 1
            continue

        face_found += 1

        # Đánh giá chất lượng ảnh
        quality = _estimate_quality(frame_rgb)
        if quality < quality_threshold:
            frame_idx += 1
            continue

        # Xác định góc
        angle = ff.angle_label
        if angle not in ['left', 'center', 'right']:
            angle = 'center'

        # Lưu frame tốt nhất
        if angle not in best_frames or quality > best_frames[angle][2]:
            best_frames[angle] = (frame_rgb, ff.yaw, quality)

        # Lưu frame nếu đạt yêu cầu (cách nhau ít nhất ~0.5s)
        if _should_capture(ff, angle, captured[angle]):
            save_path = output / angle / f"{video_name}_auto_{captured[angle]:04d}.jpg"
            cv2.imwrite(str(save_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            captured[angle] += 1

        frame_idx += 1

    cap.release()

    # In kết quả
    print(f"\nDa xu ly: {processed} frame, tim thay mat: {face_found}")
    print("-" * 40)
    total_captured = 0
    for angle in ['left', 'center', 'right']:
        count = captured[angle]
        total_captured += count
        best_info = ""
        if angle in best_frames:
            _, yaw_val, qual = best_frames[angle]
            best_info = f"  (best yaw={yaw_val:.1f}, quality={qual:.3f})"
        print(f"  {angle}: {count} ảnh{best_info}")
    print("-" * 40)
    print(f"Tong: {total_captured} ảnh")
    print(f"Luu tai: {output}")
    print(f"{'='*60}\n")

    return captured


def _estimate_quality(image: np.ndarray) -> float:
    """
    Đánh giá chất lượng ảnh bằng Laplacian variance (độ sắc nét).
    Giá trị càng cao = ảnh càng sắc nét.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _should_capture(ff: FaceFrame, angle: str, already_captured: int) -> bool:
    """
    Quyết định có nên capture frame này không.

    Tiêu chí:
    - Góc đúng ngưỡng
    - Cách frame trước ít nhất 0.3s (tránh trùng lặp)
    - Giới hạn 100 ảnh/góc
    """
    if already_captured >= 100:
        return False

    # Kiểm tra góc thực sự đúng
    if angle == "left" and ff.yaw > -10:
        return False
    if angle == "right" and ff.yaw < 10:
        return False
    if angle == "center" and abs(ff.yaw) > 12:
        return False

    return True


# ============================================================
# 5. AUTO-CAPTURE TỪ THƯ MỤC ẢNH
# ============================================================

def auto_capture_from_images(
    folder_path: str,
    output_dir: str,
) -> Dict[str, int]:
    """
    Tự động quét thư mục ảnh, phân loại và lưu vào data/angles/.

    Args:
        folder_path: Thư mục chứa ảnh.
        output_dir: data/angles/.

    Returns:
        Dict[str, int]: Số ảnh đã lưu mỗi góc.
    """
    images = read_images_from_folder(folder_path)
    if len(images) == 0:
        print("[ERROR] Khong tim thay anh nao!")
        return {"left": 0, "center": 0, "right": 0}

    output = Path(output_dir)
    for angle in ['left', 'center', 'right']:
        (output / angle).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AUTO-TRAIN tu anh: {Path(folder_path).name}")
    print(f"So anh: {len(images)}")
    print(f"{'='*60}")

    captured = {"left": 0, "center": 0, "right": 0}

    for name, img in images:
        ff = analyze_frame(img, 0)
        if ff is None:
            continue

        angle = ff.angle_label
        if angle not in captured:
            continue

        quality = _estimate_quality(img)
        if quality < 100:  # Ngưỡng tối thiểu
            continue

        save_path = output / angle / f"{Path(name).stem}_auto.jpg"
        cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        captured[angle] += 1

    # In kết quả
    total = sum(captured.values())
    print(f"Da xu ly: {len(images)} anh")
    for angle in ['left', 'center', 'right']:
        print(f"  {angle}: {captured[angle]} anh")
    print(f"Tong: {total} anh da luu")
    print(f"{'='*60}\n")

    return captured


# ============================================================
# 6. LƯU / TẢI MOTION CURVE
# ============================================================

def save_motion_curve(curve: MotionCurve, filepath: str):
    """Lưu MotionCurve ra JSON."""
    data = {
        "timestamps": curve.timestamps,
        "yaws": curve.yaws,
        "pitches": curve.pitches,
        "rolls": curve.rolls,
        "total_duration": curve.total_duration,
        "source_video": curve.source_video,
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Motion curve da luu: {filepath}")


def load_motion_curve(filepath: str) -> Optional[MotionCurve]:
    """Tải MotionCurve từ JSON."""
    path = Path(filepath)
    if not path.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return MotionCurve(
        timestamps=data["timestamps"],
        yaws=data["yaws"],
        pitches=data["pitches"],
        rolls=data["rolls"],
        total_duration=data["total_duration"],
        source_video=data.get("source_video", ""),
    )
