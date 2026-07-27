"""
face_warp_utils.py - Warp & xoay khuôn mặt 3D tự nhiên.

Nguyên lý:
1. Từ 3 ảnh reference (Trái - Thẳng - Phải), trích xuất landmarks
2. Khi cần xoay đến 1 góc bất kỳ:
   a. Nội suy vị trí landmarks đích từ 2 reference gần nhất
   b. Tính Delaunay triangulation trên landmarks trung bình
   c. Morph từng tam giác từ ảnh nguồn sang đích = affine transform
3. Blend alpha giữa 2 ảnh reference gần nhất để chuyển mượt

KỸ THUẬT DÙNG:
- Delaunay Triangulation: chia mặt thành lưới tam giác
- Affine Transform: biến đổi từng tam giác
- Alpha Blending: hòa trộn mượt giữa các góc
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple, Dict, Optional
from pathlib import Path


# ============================================================
# 1. DELAUNAY TRIANGULATION
# ============================================================

def get_triangulation(points: np.ndarray) -> np.ndarray:
    """
    Tính Delaunay triangulation cho tập điểm.

    Args:
        points: (N, 2) - landmarks 2D.

    Returns:
        np.ndarray: (M, 3) - indices của các tam giác.
    """
    tri = Delaunay(points)
    return tri.simplices


def get_triangle_indices(
    hull: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Tính Delaunay triangulation trên convex hull của landmarks + 8 góc ảnh.
    Thêm 8 góc ảnh để toàn bộ khung hình được phủ.

    Returns:
        np.ndarray: (M, 3) - indices.
    """
    h, w = image_shape

    # Thêm 8 điểm biên (góc + cạnh)
    border_points = np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 2], [w - 1, h // 2],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
    ], dtype=np.float32)

    all_points = np.vstack([hull.astype(np.float32), border_points])

    tri = Delaunay(all_points)
    return tri.simplices, all_points


# ============================================================
# 2. MORPH ẢNH (WARP TỪNG TAM GIÁC)
# ============================================================

def warp_triangle(
    src_img: np.ndarray,
    dst_img: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> np.ndarray:
    """
    Warp 1 tam giác từ ảnh nguồn sang ảnh đích.

    Args:
        src_img: Ảnh nguồn.
        dst_img: Ảnh đích (sẽ bị ghi đè vùng tam giác).
        src_tri: 3 điểm tam giác nguồn (3, 2).
        dst_tri: 3 điểm tam giác đích (3, 2).

    Returns:
        np.ndarray: Ảnh đích sau khi warp.
    """
    # Bounding box cho tam giác
    src_rect = cv2.boundingRect(src_tri.astype(np.int32))
    dst_rect = cv2.boundingRect(dst_tri.astype(np.int32))

    # Offset tọa độ về bounding box
    src_tri_offset = src_tri - src_rect[:2]
    dst_tri_offset = dst_tri - dst_rect[:2]

    # Crop vùng chứa tam giác
    src_crop = src_img[
        src_rect[1]:src_rect[1] + src_rect[3],
        src_rect[0]:src_rect[0] + src_rect[2],
    ]

    if src_crop.size == 0:
        return dst_img

    # Tính affine transform
    warp_mat = cv2.getAffineTransform(
        src_tri_offset.astype(np.float32),
        dst_tri_offset.astype(np.float32),
    )

    # Warp
    dst_crop = cv2.warpAffine(
        src_crop, warp_mat,
        (dst_rect[2], dst_rect[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    # Tạo mask và copy vào ảnh đích
    mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.float32)

    dst_tri_offset_int = dst_tri_offset.astype(np.int32)
    cv2.fillConvexPoly(mask, dst_tri_offset_int, 1.0)

    # Apply mask
    dst_roi = dst_img[
        dst_rect[1]:dst_rect[1] + dst_rect[3],
        dst_rect[0]:dst_rect[0] + dst_rect[2],
    ]

    if dst_crop.shape != dst_roi.shape:
        return dst_img

    mask_3ch = np.stack([mask] * 3, axis=-1)
    blended = (dst_crop * mask_3ch + dst_roi * (1 - mask_3ch)).astype(np.uint8)

    dst_img[
        dst_rect[1]:dst_rect[1] + dst_rect[3],
        dst_rect[0]:dst_rect[0] + dst_rect[2],
    ] = blended

    return dst_img


def morph_faces(
    src_img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    triangles: np.ndarray,
    all_src_points: np.ndarray,
    all_dst_points: np.ndarray,
) -> np.ndarray:
    """
    Morph toàn bộ khuôn mặt từ src -> dst bằng Delaunay triangulation.

    Args:
        src_img: Ảnh nguồn (khuôn mặt thẳng).
        src_points: Landmarks nguồn (trên ảnh thẳng).
        dst_points: Landmarks đích (vị trí sau khi xoay).
        triangles: Delaunay triangles indices.
        all_src_points: Tất cả điểm nguồn (landmarks + biên).
        all_dst_points: Tất cả điểm đích (landmarks + biên).

    Returns:
        np.ndarray: Ảnh đã morph.
    """
    h, w = src_img.shape[:2]
    dst_img = np.zeros_like(src_img)

    for tri_idx in triangles:
        src_tri = all_src_points[tri_idx].astype(np.float32)
        dst_tri = all_dst_points[tri_idx].astype(np.float32)
        dst_img = warp_triangle(src_img, dst_img, src_tri, dst_tri)

    return dst_img


# ============================================================
# 3. NỘI SUY LANDMARKS THEO GÓC YAW
# ============================================================

def interpolate_landmarks(
    landmarks_left: np.ndarray,
    landmarks_center: np.ndarray,
    landmarks_right: np.ndarray,
    target_yaw: float,
    left_yaw: float = -45.0,
    center_yaw: float = 0.0,
    right_yaw: float = 45.0,
) -> np.ndarray:
    """
    Nội suy vị trí landmarks cho 1 góc yaw bất kỳ.

    Args:
        landmarks_left: Landmarks ở góc trái (yaw = -45).
        landmarks_center: Landmarks nhìn thẳng (yaw = 0).
        landmarks_right: Landmarks ở góc phải (yaw = 45).
        target_yaw: Góc yaw mong muốn.
        left_yaw, center_yaw, right_yaw: Góc tương ứng với từng reference.

    Returns:
        np.ndarray: Landmarks nội suy.
    """
    target_yaw = np.clip(target_yaw, left_yaw, right_yaw)

    if target_yaw <= center_yaw:
        # Nội suy giữa left và center
        t = (target_yaw - left_yaw) / (center_yaw - left_yaw + 1e-8)
        return (1 - t) * landmarks_left + t * landmarks_center
    else:
        # Nội suy giữa center và right
        t = (target_yaw - center_yaw) / (right_yaw - center_yaw + 1e-8)
        return (1 - t) * landmarks_center + t * landmarks_right


# ============================================================
# 4. BLEND 2 ẢNH THEO GÓC
# ============================================================

def blend_images(
    img_a: np.ndarray,
    img_b: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Alpha blend 2 ảnh.

    Args:
        img_a, img_b: Ảnh cùng kích thước.
        alpha: Trọng số (0 = img_a, 1 = img_b).

    Returns:
        np.ndarray: Ảnh blended.
    """
    return (img_a * (1 - alpha) + img_b * alpha).astype(np.uint8)


# ============================================================
# 5. HÀM CHÍNH: TẠO ẢNH MẶT XOAY THEO GÓC
# ============================================================

class FaceRotator:
    """
    Class quản lý việc xoay khuôn mặt từ 3 ảnh reference.

    Cách dùng:
        rotator = FaceRotator(img_left, img_center, img_right, lm_left, lm_center, lm_right)
        rotated = rotator.rotate_to(yaw=30)  # quay phải 30 độ
    """

    def __init__(
        self,
        img_left: np.ndarray,
        img_center: np.ndarray,
        img_right: np.ndarray,
        lm_left: np.ndarray,
        lm_center: np.ndarray,
        lm_right: np.ndarray,
        left_yaw: float = -45.0,
        center_yaw: float = 0.0,
        right_yaw: float = 45.0,
    ):
        h, w = img_center.shape[:2]
        self.img_size = (w, h)

        self.images = {
            'left': img_left,
            'center': img_center,
            'right': img_right,
        }
        self.landmarks = {
            'left': lm_left,
            'center': lm_center,
            'right': lm_right,
        }
        self.yaws = {
            'left': left_yaw,
            'center': center_yaw,
            'right': right_yaw,
        }

        # Precompute triangulation (trên landmarks center + border)
        hull_center = cv2.convexHull(lm_center.astype(np.int32)).squeeze()
        self.triangles, self.all_center_pts = get_triangle_indices(hull_center, (h, w))

        # Precompute all_points cho left và right
        self.all_left_pts = self._build_all_points(lm_left, hull_center, 'left')
        self.all_right_pts = self._build_all_points(lm_right, hull_center, 'right')

        self.current_yaw = 0.0
        self.current_image = img_center.copy()

    def _build_all_points(
        self,
        landmarks: np.ndarray,
        hull_center: np.ndarray,
        side: str,
    ) -> np.ndarray:
        """Ghép landmarks + border points."""
        h, w = self.img_size[1], self.img_size[0]

        # Sử dụng hull từ center (các điểm biên giữ nguyên)
        border_points = np.array([
            [0, 0], [w // 2, 0], [w - 1, 0],
            [0, h // 2], [w - 1, h // 2],
            [0, h - 1], [w // 2, h - 1], [w - 1, h - 1],
        ], dtype=np.float32)

        hull_side = cv2.convexHull(landmarks.astype(np.int32)).squeeze()
        return np.vstack([hull_side.astype(np.float32), border_points])

    def get_target_landmarks(self, target_yaw: float) -> np.ndarray:
        """Nội suy landmarks cho góc yaw."""
        return interpolate_landmarks(
            self.landmarks['left'],
            self.landmarks['center'],
            self.landmarks['right'],
            target_yaw,
            self.yaws['left'],
            self.yaws['center'],
            self.yaws['right'],
        )

    def get_target_all_points(self, target_yaw: float) -> np.ndarray:
        """Nội suy all_points (landmarks + border)."""
        lm_target = self.get_target_landmarks(target_yaw)

        if target_yaw <= 0:
            t = (target_yaw - self.yaws['left']) / (self.yaws['center'] - self.yaws['left'] + 1e-8)
            return (1 - t) * self.all_left_pts + t * self.all_center_pts
        else:
            t = (target_yaw - self.yaws['center']) / (self.yaws['right'] - self.yaws['center'] + 1e-8)
            return (1 - t) * self.all_center_pts + t * self.all_right_pts

    def rotate_to(self, target_yaw: float) -> np.ndarray:
        """
        Xoay khuôn mặt đến góc yaw mong muốn.

        Args:
            target_yaw: Góc xoay (độ). Âm = trái, 0 = thẳng, Dương = phải.

        Returns:
            np.ndarray: Ảnh khuôn mặt đã xoay.
        """
        target_yaw = np.clip(
            target_yaw,
            self.yaws['left'],
            self.yaws['right'],
        )

        # Chọn 1 hoặc 2 ảnh reference
        if abs(target_yaw - self.yaws['center']) < 2:
            self.current_yaw = target_yaw
            self.current_image = self.images['center'].copy()
            return self.current_image

        if target_yaw <= self.yaws['center']:
            src_img = self.images['left']
            target_all_pts = self.get_target_all_points(target_yaw)
            morphed = morph_faces(
                src_img,
                self.landmarks['left'],
                self.get_target_landmarks(target_yaw),
                self.triangles,
                self.all_left_pts,
                target_all_pts,
            )

            # Blend với center nếu gần center
            t = (target_yaw - self.yaws['left']) / (self.yaws['center'] - self.yaws['left'] + 1e-8)
            center_morphed = morph_faces(
                self.images['center'],
                self.landmarks['center'],
                self.get_target_landmarks(target_yaw),
                self.triangles,
                self.all_center_pts,
                target_all_pts,
            )
            result = blend_images(morphed, center_morphed, t)
        else:
            src_img = self.images['center']
            target_all_pts = self.get_target_all_points(target_yaw)
            morphed = morph_faces(
                src_img,
                self.landmarks['center'],
                self.get_target_landmarks(target_yaw),
                self.triangles,
                self.all_center_pts,
                target_all_pts,
            )

            t = (target_yaw - self.yaws['center']) / (self.yaws['right'] - self.yaws['center'] + 1e-8)
            right_morphed = morph_faces(
                self.images['right'],
                self.landmarks['right'],
                self.get_target_landmarks(target_yaw),
                self.triangles,
                self.all_right_pts,
                target_all_pts,
            )
            result = blend_images(morphed, right_morphed, t)

        self.current_yaw = target_yaw
        self.current_image = result
        return result


# ============================================================
# 6. MOTION SEQUENCE: HỌC CHUYỂN ĐỘNG TỪ VIDEO THAM CHIẾU
# ============================================================

class MotionSequence:
    """
    Lớp playback đường cong chuyển động từ video tham chiếu.

    Thay vì nội suy góc cố định, class này chơi lại (playback)
    đường cong yaw tự nhiên được trích từ video người thật.

    Cách dùng:
        seq = MotionSequence.from_json("models/motion_curve.json")
        yaw = seq.get_yaw_at(elapsed_time)  # Lấy góc tại thời điểm t
    """

    def __init__(self, timestamps, yaws):
        self.timestamps = list(timestamps)
        self.yaws = list(yaws)
        self.duration = self.timestamps[-1] if self.timestamps else 0

    @classmethod
    def from_json(cls, filepath: str) -> Optional["MotionSequence"]:
        """Tạo MotionSequence từ file JSON (tạo bởi auto_train.py --motion)."""
        import json
        path = Path(filepath)
        if not path.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data["timestamps"], data["yaws"])

    def get_yaw_at(self, elapsed: float) -> float:
        """
        Lấy góc yaw tại thời điểm elapsed (giây) theo motion curve.

        Args:
            elapsed: Thời gian đã trôi qua (giây).

        Returns:
            float: Góc yaw nội suy.
        """
        if self.duration == 0 or len(self.timestamps) < 2:
            return 0.0

        # Wrap around nếu elapsed > duration (lặp lại)
        t = elapsed % self.duration

        # Tìm 2 điểm gần nhất
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] <= t <= self.timestamps[i + 1]:
                alpha = (t - self.timestamps[i]) / (self.timestamps[i + 1] - self.timestamps[i])
                return self.yaws[i] + alpha * (self.yaws[i + 1] - self.yaws[i])

        return self.yaws[-1]

    def get_segment_yaws(self, start_t: float, end_t: float, steps: int = 10) -> List[float]:
        """
        Lấy danh sách yaw trong 1 đoạn thời gian.

        Args:
            start_t: Thời gian bắt đầu.
            end_t: Thời gian kết thúc.
            steps: Số bước.

        Returns:
            List[float]: Danh sách góc yaw.
        """
        result = []
        for i in range(steps):
            t = start_t + (end_t - start_t) * i / max(steps - 1, 1)
            result.append(self.get_yaw_at(t))
        return result
