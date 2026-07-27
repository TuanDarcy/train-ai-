"""
auto_train.py - Module 2: TỰ ĐỘNG HỌC TỪ VIDEO/ẢNH.

CÁCH DÙNG:
    # Từ video:
    python src/auto_train.py --input video_nguoi_quay.mp4

    # Từ thư mục ảnh:
    python src/auto_train.py --input thu_muc_anh/ --type images

    # Kết hợp: vừa capture ảnh, vừa trích motion curve:
    python src/auto_train.py --input video.mp4 --motion --output-motion models/motion.json

LUỒNG HOẠT ĐỘNG:
    1. Đọc video/thư mục ảnh
    2. Với mỗi frame/ảnh: phát hiện khuôn mặt, tính yaw/pitch/roll
    3. Phân loại: left (yaw<-15), center (-15~+15), right (yaw>+15)
    4. Auto-capture frame tốt nhất, lưu vào data/angles/{left,center,right}/
    5. (Tùy chọn) Trích xuất motion curve để Module 1 dùng
    6. In thống kê dataset

KẾT QUẢ: Dataset trong data/angles/ đã sẵn sàng để dùng với live_avatar.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.video_utils import (
    auto_capture_from_video,
    auto_capture_from_images,
    extract_motion_curve,
    save_motion_curve,
)

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "angles"
MODELS_DIR = ROOT_DIR / "models"


def main():
    parser = argparse.ArgumentParser(
        description="AUTO-TRAIN: Tu dong hoc khuon mat tu video/anh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VI DU:
  python src/auto_train.py --input video_cua_toi.mp4
  python src/auto_train.py --input video.mp4 --motion
  python src/auto_train.py --input folder_anh/ --type images
  python src/auto_train.py --input video.mp4 --sample 5 --fps 30
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Duong dan video hoac thu muc anh",
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="video",
        choices=["video", "images"],
        help="Loai input: 'video' hoac 'images' (mac dinh: video)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DATA_DIR),
        help=f"Thu muc output (mac dinh: {DATA_DIR})",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=3,
        help="Lay 1 frame moi N frame (mac dinh: 3)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS cua video (mac dinh: 30)",
    )
    parser.add_argument(
        "--motion",
        action="store_true",
        help="Trich xuat motion curve cho Module 1",
    )
    parser.add_argument(
        "--output-motion",
        type=str,
        default=str(MODELS_DIR / "motion_curve.json"),
        help="File output motion curve JSON",
    )
    parser.add_argument(
        "--quality",
        type=float,
        default=0.3,
        help="Nguong chat luong anh (0-1, mac dinh: 0.3)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Khong tim thay: {args.input}")
        return

    # ============================================================
    # 1. AUTO-CAPTURE
    # ============================================================
    if args.type == "video":
        captured = auto_capture_from_video(
            video_path=str(input_path),
            output_dir=args.output,
            sample_every=args.sample,
            fps=args.fps,
            quality_threshold=args.quality,
        )
    else:
        captured = auto_capture_from_images(
            folder_path=str(input_path),
            output_dir=args.output,
        )

    total = sum(captured.values())
    if total == 0:
        print("\n[WARNING] Khong capture duoc anh nao!")
        print("[?] Thu dieu chinh: giam --sample, giam --quality, hoac dung video co khuon mat ro hon")
        return

    # ============================================================
    # 2. EXTRACT MOTION CURVE (tùy chọn)
    # ============================================================
    if args.motion and args.type == "video":
        print("\n[?] Trich xuat motion curve...")
        curve = extract_motion_curve(
            video_path=str(input_path),
            sample_every=args.sample,
            fps=args.fps,
        )
        if curve:
            output_motion = Path(args.output_motion)
            output_motion.parent.mkdir(parents=True, exist_ok=True)
            save_motion_curve(curve, str(output_motion))
            print(f"\n[?] Dung motion curve voi live_avatar.py:")
            print(f"    python src/live_avatar.py --motion {output_motion}")

    # ============================================================
    # 3. TỔNG KẾT
    # ============================================================
    print(f"\n[?] BUOC TIEP THEO:")
    print(f"    python src/live_avatar.py")
    print(f"    (Hoac upload data/angles/ len Google Colab de xu ly nang cao)")


if __name__ == "__main__":
    main()
