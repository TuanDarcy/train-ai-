"""
face_mesh_utils.py - Trích xuất 3D Face Mesh bằng MediaPipe.

MediaPipe Face Mesh cho ra 468 điểm landmark 3D (x, y, z).
Từ đó ta tính được:
- Góc yaw (quay trái/phải), pitch (lên/xuống), roll (nghiêng)
- Vùng tam giác Delaunay để warp ảnh
- Mặt nạ khuôn mặt
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional


# ============================================================
# CONFIG
# ============================================================

mp_face_mesh = mp.solutions.face_mesh
FaceMesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,    # Lấy thêm iris (mắt)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Indices cho contour khuôn mặt (dùng để vẽ mask)
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


# ============================================================
# 1. LẤY LANDMARKS 3D
# ============================================================

def get_face_landmarks(
    image: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Trích xuất 468 landmarks 3D từ ảnh.

    Args:
        image: Ảnh RGB (H, W, 3)

    Returns:
        (landmarks_2d, landmarks_3d) hoặc None nếu không tìm thấy mặt.
        - landmarks_2d: (468, 2) - tọa độ pixel
        - landmarks_3d: (468, 3) - tọa độ 3D metric
    """
    h, w = image.shape[:2]
    results = FaceMesh.process(image)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    lm_2d = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in face_landmarks.landmark
    ], dtype=np.float32)

    lm_3d = np.array([
        (lm.x, lm.y, lm.z)
        for lm in face_landmarks.landmark
    ], dtype=np.float32)

    return lm_2d, lm_3d


# ============================================================
# 2. TÍNH GÓC ĐẦU (HEAD POSE)
# ============================================================

def estimate_head_pose(
    landmarks_2d: np.ndarray,
    image_shape: Tuple[int, int],
) -> Tuple[float, float, float]:
    """
    Ước lượng góc đầu (yaw, pitch, roll) từ landmarks.

    Sử dụng SolvePnP với model 3D generic.

    Returns:
        (yaw, pitch, roll) - đơn vị độ.
        yaw   > 0: quay phải, < 0: quay trái
        pitch > 0: nhìn lên,  < 0: nhìn xuống
        roll  > 0: nghiêng phải, < 0: nghiêng trái
    """
    h, w = image_shape

    # Chọn 6 điểm chính (mũi, cằm, mắt trái, mắt phải, miệng trái, miệng phải)
    # MediaPipe indices
    key_indices = [1, 152, 33, 263, 61, 291]  # nose tip, chin, left eye, right eye, left mouth, right mouth

    # Tọa độ 3D model (generic face model, chuẩn hóa)
    model_3d = np.array([
        [0.0, 0.0, 0.0],          # Nose tip
        [0.0, -0.33, -0.07],      # Chin
        [-0.165, 0.17, -0.17],    # Left eye outer
        [0.165, 0.17, -0.17],     # Right eye outer
        [-0.15, -0.15, -0.17],    # Left mouth
        [0.15, -0.15, -0.17],     # Right mouth
    ], dtype=np.float32)

    # Lấy 2D points tương ứng
    img_points = np.array([landmarks_2d[i] for i in key_indices], dtype=np.float32)

    # Camera matrix (giả định)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        model_3d, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0, 0.0, 0.0

    # Chuyển rotation vector -> Euler angles
    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # Tính yaw, pitch, roll từ rotation matrix
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
        roll = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
    else:
        pitch = np.arctan2(-rot_mat[2, 0], sy)
        yaw = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        roll = 0

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


# ============================================================
# 3. LẤY CONVEX HULL KHUÔN MẶT
# ============================================================

def get_face_hull(landmarks_2d: np.ndarray) -> np.ndarray:
    """
    Lấy convex hull của khuôn mặt từ landmarks.

    Returns:
        np.ndarray: (N, 2) - các điểm contour.
    """
    face_points = landmarks_2d[FACE_OVAL_IDX].astype(np.int32)
    hull = cv2.convexHull(face_points)
    return hull.squeeze()


# ============================================================
# 4. TẠO MASK KHUÔN MẶT
# ============================================================

def create_face_mask(
    image_shape: Tuple[int, int],
    landmarks_2d: np.ndarray,
    dilate: int = 10,
) -> np.ndarray:
    """
    Tạo binary mask cho vùng khuôn mặt.

    Returns:
        np.ndarray: mask uint8 (H, W).
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = get_face_hull(landmarks_2d)
    cv2.fillConvexPoly(mask, hull, 255)

    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


# ============================================================
# 5. VẼ LANDMARKS (debug)
# ============================================================

def draw_landmarks(
    image: np.ndarray,
    landmarks_2d: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 2,
) -> np.ndarray:
    """Vẽ landmarks lên ảnh (BGR)."""
    img = image.copy()
    for pt in landmarks_2d.astype(np.int32):
        cv2.circle(img, tuple(pt), radius, color, -1)

    # Vẽ contour mặt
    hull = get_face_hull(landmarks_2d)
    cv2.polylines(img, [hull], True, (255, 0, 0), 2)

    return img
