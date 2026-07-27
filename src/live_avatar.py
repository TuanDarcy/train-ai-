"""
live_avatar.py - AI Avatar điều khiển khuôn mặt thời gian thực bằng GIỌNG NÓI.

2 CHẾ ĐỘ:
  MODE 1 - "fixed": Load 3 ảnh, nội suy góc cố định.
  MODE 2 - "motion": Học chuyển động tự nhiên từ video tham chiếu.

CÁCH DÙNG:
    # Mode 1: 3 ảnh cố định
    python src/live_avatar.py

    # Mode 2: Học motion từ video tham chiếu
    python src/live_avatar.py --motion models/motion_curve.json

    # Mode 1 + ảnh tùy chỉnh
    python src/live_avatar.py --left imgL.jpg --center imgC.jpg --right imgR.jpg
"""

import cv2
import sys
import time
import argparse
from pathlib import Path
from collections import deque
import numpy as np
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from utils.face_mesh_utils import get_face_landmarks, estimate_head_pose
from utils.face_warp_utils import FaceRotator, MotionSequence
from utils.voice_utils import VoiceRecognizer, Command, speak


# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "angles"

DEFAULT_IMAGES = {
    'left': DATA_DIR / "left" / "left_face.jpg",
    'center': DATA_DIR / "center" / "center_face.jpg",
    'right': DATA_DIR / "right" / "right_face.jpg",
}

WINDOW_NAME = "AI AVATAR - Dieu khien khuon mat bang giong noi"
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Animation config
ANIMATION_SPEED = 2.5       # Độ/giây - tốc độ xoay (càng cao càng nhanh)
SMOOTHING_WINDOW = 5        # Số frame làm mượt chuyển động


class LiveAvatar:
    """
    AI Avatar điều khiển khuôn mặt real-time.

    Hỗ trợ 2 mode:
    - "fixed": Nội suy giữa 3 góc cố định (trái/thẳng/phải)
    - "motion": Playback chuyển động tự nhiên từ video tham chiếu
    """

    def __init__(
        self,
        img_left: np.ndarray,
        img_center: np.ndarray,
        img_right: np.ndarray,
        use_voice: bool = True,
        animation_speed: float = ANIMATION_SPEED,
        motion_curve: Optional[str] = None,
    ):
        self.use_voice = use_voice
        self.animation_speed = animation_speed
        self.running = True

        # ---- MODE: motion reference ----
        self.motion_mode = motion_curve is not None
        self.motion_seq: Optional[MotionSequence] = None
        self.motion_elapsed = 0.0       # Thời gian playback
        self.motion_paused = True        # Tạm dừng motion khi idle

        if self.motion_mode:
            self.motion_seq = MotionSequence.from_json(motion_curve)
            if self.motion_seq is None:
                print(f"[WARNING] Khong load duoc motion curve: {motion_curve}")
                print("          Chuyen sang mode 'fixed'")
                self.motion_mode = False

        # ---- 1. Trích xuất landmarks ----
        print("[1/4] Trich xuat 3D face mesh...")
        result_left = get_face_landmarks(img_left)
        result_center = get_face_landmarks(img_center)
        result_right = get_face_landmarks(img_right)

        if not all([result_left, result_center, result_right]):
            print("[ERROR] Khong tim thay khuon mat trong 1 trong 3 anh!")
            print("Hay kiem tra lai anh trong data/angles/")
            raise ValueError("Face not detected in reference images")

        lm_left_2d, _ = result_left
        lm_center_2d, _ = result_center
        lm_right_2d, _ = result_right

        # ---- 2. Tính yaw thực tế ----
        yaw_left, _, _ = estimate_head_pose(lm_left_2d, img_left.shape[:2])
        yaw_center, _, _ = estimate_head_pose(lm_center_2d, img_center.shape[:2])
        yaw_right, _, _ = estimate_head_pose(lm_right_2d, img_right.shape[:2])

        self.yaw_left = yaw_left
        self.yaw_center = yaw_center
        self.yaw_right = yaw_right

        print(f"   Goc trai:  {yaw_left:.1f} deg")
        print(f"   Goc thang: {yaw_center:.1f} deg")
        print(f"   Goc phai:  {yaw_right:.1f} deg")

        if self.motion_mode:
            print(f"   Motion duration: {self.motion_seq.duration:.1f}s")
            print(f"   Motion yaw range: {min(self.motion_seq.yaws):.1f} -> {max(self.motion_seq.yaws):.1f}")

        # ---- 3. Khởi tạo FaceRotator ----
        print("[2/4] Khoi tao FaceRotator...")
        self.rotator = FaceRotator(
            img_left, img_center, img_right,
            lm_left_2d, lm_center_2d, lm_right_2d,
            left_yaw=yaw_left,
            center_yaw=yaw_center,
            right_yaw=yaw_right,
        )
        self.target_yaw = 0.0
        self.current_yaw = 0.0

        # Smoothing
        self.yaw_history = deque([0.0] * SMOOTHING_WINDOW, maxlen=SMOOTHING_WINDOW)

        # ---- 4. Voice ----
        if self.use_voice:
            print("[3/4] Khoi tao nhan dien giong noi...")
            try:
                self.voice = VoiceRecognizer(language="vi-VN")
                self.voice.start()
                self.voice_ok = True
                print("   [OK] San sang lang nghe!")
            except Exception as e:
                print(f"   [!] Loi voice: {e}")
                self.voice_ok = False
        else:
            self.voice_ok = False

        print("[4/4] San sang!")
        print("=" * 60)
        if self.motion_mode:
            print("  MODE: Motion Reference (chuyen dong tu nhien)")
        else:
            print("  MODE: Fixed Angles (noi suy goc co dinh)")
        if self.voice_ok:
            print("  LENH GIONG NOI:")
        else:
            print("  LENH BAN PHIM:")
        print("  Quay trai  -> 'quay trai'  / phim A")
        print("  Quay phai  -> 'quay phai'  / phim D")
        print("  Nhin thang -> 'nhin thang' / phim S")
        print("  Thoat      -> 'thoat'      / phim Q")
        print("=" * 60)

        self.last_command_time = 0
        self.last_command_text = "San sang"
        self.motion_direction = 0   # -1=trái, 0=dừng, 1=phải

    # ============================================================
    # PROCESS COMMAND
    # ============================================================

    def _handle_command(self, command: Command):
        """Xử lý lệnh."""
        now = time.time()

        if command == Command.LOOK_LEFT:
            if self.motion_mode:
                self.motion_direction = -1
                self.motion_paused = False
                self.last_command_text = "<<< QUAY TRAI (motion) <<<"
            else:
                self.target_yaw = self.yaw_left
                self.last_command_text = "<<< QUAY TRAI <<<"
            self.last_command_time = now
            speak("Quay trái")

        elif command == Command.LOOK_RIGHT:
            if self.motion_mode:
                self.motion_direction = 1
                self.motion_paused = False
                self.last_command_text = ">>> QUAY PHAI (motion) >>>"
            else:
                self.target_yaw = self.yaw_right
                self.last_command_text = ">>> QUAY PHAI >>>"
            self.last_command_time = now
            speak("Quay phải")

        elif command == Command.LOOK_CENTER:
            if self.motion_mode:
                self.motion_direction = 0
                self.motion_paused = True
                self.last_command_text = "^^^ NHIN THANG ^^^"
            else:
                self.target_yaw = 0.0
                self.last_command_text = "^^^ NHIN THANG ^^^"
            self.last_command_time = now
            speak("Nhìn thẳng")

        elif command == Command.QUIT:
            self.running = False
            speak("Tạm biệt")

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def run(self):
        """Vòng lặp chính."""
        last_frame_time = time.time()

        while self.running:
            now = time.time()
            dt = min(now - last_frame_time, 0.1)  # Cap dt
            last_frame_time = now

            # ---- Input ----
            if self.voice_ok:
                cmd = self.voice.get_last_command()
                if cmd != Command.UNKNOWN:
                    self._handle_command(cmd)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key in [ord('a'), ord('A')]:
                self._handle_command(Command.LOOK_LEFT)
            elif key in [ord('d'), ord('D')]:
                self._handle_command(Command.LOOK_RIGHT)
            elif key in [ord('s'), ord('S')]:
                self._handle_command(Command.LOOK_CENTER)

            # ---- Animation ----
            if self.motion_mode:
                # MODE: Motion reference playback
                if not self.motion_paused:
                    self.motion_elapsed += dt * abs(self.motion_direction)
                    if self.motion_direction < 0:
                        # Đảo ngược motion để quay trái
                        inv_t = max(0, self.motion_seq.duration - self.motion_elapsed % self.motion_seq.duration)
                        target = self.motion_seq.get_yaw_at(inv_t)
                    else:
                        target = self.motion_seq.get_yaw_at(self.motion_elapsed)
                    self.target_yaw = target
                else:
                    # Idle: từ từ về center
                    self.target_yaw = 0.0

                # Smooth approach
                yaw_diff = self.target_yaw - self.current_yaw
                step = self.animation_speed * dt
                if abs(yaw_diff) < step:
                    self.current_yaw = self.target_yaw
                else:
                    self.current_yaw += np.sign(yaw_diff) * step * min(1.0, abs(yaw_diff) / 15.0)

                self.yaw_history.append(self.current_yaw)
                smoothed_yaw = np.mean(self.yaw_history)
            else:
                # MODE: Fixed angles
                yaw_diff = self.target_yaw - self.current_yaw
                if abs(yaw_diff) > 0.1:
                    step = self.animation_speed * dt
                    if abs(yaw_diff) < step:
                        self.current_yaw = self.target_yaw
                    else:
                        ease = min(1.0, abs(yaw_diff) / 20.0)
                        self.current_yaw += np.sign(yaw_diff) * step * ease
                    self.yaw_history.append(self.current_yaw)
                    smoothed_yaw = np.mean(self.yaw_history)
                else:
                    self.current_yaw = self.target_yaw
                    smoothed_yaw = self.current_yaw

            # ---- Render ----
            rendered = self.rotator.rotate_to(smoothed_yaw)
            display = cv2.resize(rendered, (SCREEN_WIDTH, SCREEN_HEIGHT))

            # ---- UI Overlay ----
            # Nền dưới cho text
            overlay = display.copy()
            cv2.rectangle(overlay, (0, SCREEN_HEIGHT - 80), (SCREEN_WIDTH, SCREEN_HEIGHT), (20, 20, 20), -1)
            display = cv2.addWeighted(overlay, 0.5, display, 0.5, 0)

            # Hiển thị góc
            cv2.putText(display, f"Yaw: {smoothed_yaw:+.1f} deg  |  Target: {self.target_yaw:+.1f} deg",
                        (20, SCREEN_HEIGHT - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Hiển thị lệnh cuối
            if now - self.last_command_time < 2.0:
                alpha = 1.0 - (now - self.last_command_time) / 2.0
                color = tuple(int(255 * alpha) for _ in range(3))
                cv2.putText(display, self.last_command_text,
                            (SCREEN_WIDTH // 2 - 150, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # FPS
            fps = 1.0 / max(dt, 0.001)
            cv2.putText(display, f"FPS: {fps:.0f}",
                        (SCREEN_WIDTH - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            # Hướng dẫn
            if self.voice_ok:
                hint = "Giong noi: 'quay trai' | 'quay phai' | 'nhin thang' | 'thoat'"
            else:
                hint = "Ban phim: A=Trai | S=Thang | D=Phai | Q=Thoat"
            cv2.putText(display, hint,
                        (20, SCREEN_HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            cv2.imshow(WINDOW_NAME, display)

        # ---- Cleanup ----
        if self.voice_ok:
            self.voice.stop()
        cv2.destroyAllWindows()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Avatar - Dieu khien khuon mat bang giong noi"
    )
    parser.add_argument("--left", type=str, default=str(DEFAULT_IMAGES['left']),
                        help="Anh khuon mat goc trai")
    parser.add_argument("--center", type=str, default=str(DEFAULT_IMAGES['center']),
                        help="Anh khuon mat nhin thang")
    parser.add_argument("--right", type=str, default=str(DEFAULT_IMAGES['right']),
                        help="Anh khuon mat goc phai")
    parser.add_argument("--no-voice", action="store_true",
                        help="Tat giong noi, dung ban phim")
    parser.add_argument("--speed", type=float, default=ANIMATION_SPEED,
                        help=f"Toc do xoay (deg/s, mac dinh: {ANIMATION_SPEED})")
    parser.add_argument("--motion", type=str, default=None,
                        help="File motion curve JSON (tu auto_train.py --motion)")
    args = parser.parse_args()

    # ---- Load ảnh ----
    print("\n" + "=" * 60)
    print("   AI AVATAR - KHUON MAT DIEU KHIEN BANG GIONG NOI")
    print("=" * 60)

    for side, path in [("Trai", args.left), ("Thang", args.center), ("Phai", args.right)]:
        p = Path(path)
        if not p.exists():
            print(f"\n[ERROR] Khong tim thay anh goc {side}: {path}")
            print(f"[?] Hay chay: python src/collect_angles.py")
            return
        print(f"   Anh {side}: {p}")

    img_left = cv2.cvtColor(cv2.imread(args.left), cv2.COLOR_BGR2RGB)
    img_center = cv2.cvtColor(cv2.imread(args.center), cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(args.right), cv2.COLOR_BGR2RGB)

    print()

    # ---- Chạy ----
    try:
        avatar = LiveAvatar(
            img_left, img_center, img_right,
            use_voice=not args.no_voice,
            animation_speed=args.speed,
            motion_curve=args.motion,
        )
        avatar.run()
    except ValueError as e:
        print(f"\n[ERROR] {e}")
    except KeyboardInterrupt:
        print("\n[INFO] Da thoat.")
    except Exception as e:
        print(f"\n[ERROR] Loi: {e}")

    print("\nTam biet!\n")


if __name__ == "__main__":
    main()
