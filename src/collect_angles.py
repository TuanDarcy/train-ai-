"""
collect_angles.py - Thu thập dữ liệu khuôn mặt: ảnh 3 góc HOẶC quay video.

2 CHẾ ĐỘ:
  1. Chụp ảnh: 3 ảnh tĩnh (trái/thẳng/phải) -> data/angles/
  2. Quay video: video ngắn làm motion reference -> data/driving_videos/

CÁCH DÙNG:
    # Chế độ chụp ảnh (mặc định)
    python src/collect_angles.py

    # Chế độ quay video motion reference
    python src/collect_angles.py --video
"""

import cv2
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.face_mesh_utils import get_face_landmarks, estimate_head_pose, draw_landmarks


DATA_DIR = Path(__file__).parent.parent / "data" / "angles"
VIDEO_DIR = Path(__file__).parent.parent / "data" / "driving_videos"
WINDOW_NAME = "Thu Thap Du Lieu Khuon Mat"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():
    parser = argparse.ArgumentParser(description="Thu thap du lieu khuon mat")
    parser.add_argument("--video", action="store_true",
                        help="Che do quay video motion reference (thay vi chup anh)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Thoi luong video (giay, mac dinh: 10)")
    args = parser.parse_args()

    if args.video:
        record_motion_video(args.duration)
    else:
        capture_three_angles()


def record_motion_video(duration: int = 10):
    """
    Quay 1 video ngan nguoi dung quay dau trai-phai de lam motion reference.
    """
    print("\n" + "=" * 60)
    print("   QUAY VIDEO MOTION REFERENCE")
    print("=" * 60)
    print()
    print("HUONG DAN:")
    print("  - Nhin thang -> tu tu quay trai -> ve thang -> quay phai -> ve thang")
    print("  - Chuyen dong CHAM, TU NHIEN nhu nguoi that")
    print(f"  - Video dai {duration}s, FPS: 30")
    print("  - Nhan SPACE de bat dau, Q de thoat")
    print("=" * 60)

    input("\nNhan ENTER khi san sang...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong the mo camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Tạo video output
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = str(VIDEO_DIR / f"motion_ref_{timestamp}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (FRAME_WIDTH, FRAME_HEIGHT))

    recording = False
    start_time = 0
    running = True

    print(f"\n[?] Video se luu tai: {video_path}")
    print("[?] Nhan SPACE de bat dau quay...")

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        if recording:
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)

            # Lưu frame
            out.write(frame)

            # UI
            cv2.putText(display, f"DANG QUAY... {remaining:.1f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Progress bar
            bar_w = 400
            bar_h = 20
            bar_x = (FRAME_WIDTH - bar_w) // 2
            bar_y = FRAME_HEIGHT - 80
            progress = elapsed / duration
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 2)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + bar_h), (0, 255, 0), -1)

            if elapsed >= duration:
                recording = False
                print("\n[OK] Da quay xong!")
                break
        else:
            # Phát hiện khuôn mặt
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = get_face_landmarks(rgb)
            if result:
                lm_2d, _ = result
                display = draw_landmarks(display, lm_2d)
                cv2.putText(display, "San sang! Nhan SPACE de quay",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "KHONG TIM THAY MAT!",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, "Quay dau tu nhien: trai -> thang -> phai -> thang",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(WINDOW_NAME + " - Video", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord(' ') and not recording:
            recording = True
            start_time = time.time()
            print("[REC] Dang quay...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if recording or Path(video_path).exists():
        print(f"\n{'='*60}")
        print(f"Video da luu: {video_path}")
        print(f"[?] BUOC TIEP THEO:")
        print(f"    python src/auto_train.py --input \"{video_path}\" --motion")
        print(f"{'='*60}\n")


def capture_three_angles():
    print("\n" + "=" * 60)
    print("   THU THAP ANH 3 GOC KHUON MAT")
    print("   (Trai - Thang - Phai)")
    print("=" * 60)
    print()
    print("HUONG DAN:")
    print("  1. Ngoi thang, nhin THANG vao camera -> nhan SPACE de chup")
    print("  2. Xoay mat sang TRAI ~45°         -> nhan SPACE de chup")
    print("  3. Xoay mat sang PHAI ~45°         -> nhan SPACE de chup")
    print("  Nhan R de chup lai, Q de thoat")
    print("=" * 60)

    input("\nNhan ENTER de bat dau...")

    # Mở camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Khong the mo camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    angles = ["center", "left", "right"]
    angle_labels = {
        "center": "NHIN THANG (Straight)",
        "left": "Xoay TRAI (Left ~45)",
        "right": "Xoay PHAI (Right ~45)",
    }
    captured = {a: None for a in angles}
    current_idx = 0
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện face mesh
        result = get_face_landmarks(rgb_frame)
        current_angle = angles[current_idx]

        display = frame.copy()

        # Overlay hướng dẫn
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 120), (30, 30, 30), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

        cv2.putText(display, f"BUOC {current_idx + 1}/3: {angle_labels[current_angle]}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if result:
            lm_2d, lm_3d = result
            yaw, pitch, roll = estimate_head_pose(lm_2d, (FRAME_HEIGHT, FRAME_WIDTH))

            # Vẽ landmarks
            display = draw_landmarks(display, lm_2d, color=(0, 255, 0), radius=1)

            # Hiển thị góc hiện tại
            cv2.putText(display, f"Yaw: {yaw:+.1f}deg  Pitch: {pitch:+.1f}deg",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Kiểm tra góc phù hợp
            angle_ok = False
            if current_angle == "center" and abs(yaw) < 15:
                angle_ok = True
                hint = "[OK] Goc tot! Nhan SPACE de chup"
                hint_color = (0, 255, 0)
            elif current_angle == "left" and yaw < -20:
                angle_ok = True
                hint = "[OK] Goc tot! Nhan SPACE de chup"
                hint_color = (0, 255, 0)
            elif current_angle == "right" and yaw > 20:
                angle_ok = True
                hint = "[OK] Goc tot! Nhan SPACE de chup"
                hint_color = (0, 255, 0)
            else:
                hint = "[...] Hay xoay mat den goc phu hop"
                hint_color = (0, 165, 255)

            cv2.putText(display, hint,
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hint_color, 2)

        else:
            cv2.putText(display, "KHONG TIM THAY KHUON MAT!",
                        (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Hiện trạng thái đã chụp
        status_y = FRAME_HEIGHT - 60
        for i, a in enumerate(angles):
            status = "[V] DA CHUP" if captured[a] is not None else "[ ] chua chup"
            color = (0, 255, 0) if captured[a] is not None else (150, 150, 150)
            cv2.putText(display, f"{status} {angle_labels[a]}",
                        (20, status_y + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            running = False
        elif key == ord(' '):  # SPACE = chụp
            if result and current_idx < 3:
                a = angles[current_idx]
                save_path = DATA_DIR / a / f"{a}_face.jpg"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_path), frame)
                captured[a] = str(save_path)
                print(f"[OK] Da luu: {save_path}")
                current_idx += 1
                if current_idx >= 3:
                    print("\n[HOAN THANH] Da chup du 3 goc!")
                    time.sleep(1)
                    running = False
        elif key == ord('r'):  # R = chụp lại góc hiện tại
            if current_idx < 3:
                current_idx = max(0, current_idx)

    cap.release()
    cv2.destroyAllWindows()

    # Tổng kết
    print("\n" + "=" * 60)
    print("KET QUA:")
    for a in angles:
        if captured[a]:
            print(f"  [{a}] {captured[a]}")
        else:
            print(f"  [{a}] CHUA CHUP!")
    print(f"\n[?] BUOC TIEP THEO:")
    print(f"    1. Chay Google Colab notebook de xu ly anh")
    print(f"    2. Hoac chay truc tiep: python src/live_avatar.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
