"""
utils/__init__.py - Module tiện ích cho AI Avatar khuôn mặt.

Cấu trúc:
    utils/
    ├── __init__.py            # File này
    ├── face_mesh_utils.py     # MediaPipe 3D face mesh + head pose
    ├── face_warp_utils.py     # Face warping + FaceRotator + MotionSequence
    ├── voice_utils.py         # Voice command recognition
    └── video_utils.py         # Video processing, auto-capture, motion curve
"""

# 3D Face Mesh
from .face_mesh_utils import (
    get_face_landmarks,
    estimate_head_pose,
    get_face_hull,
    create_face_mask,
    draw_landmarks,
)

# Face Warping & Motion
from .face_warp_utils import (
    FaceRotator,
    MotionSequence,
    get_triangulation,
    morph_faces,
    interpolate_landmarks,
    blend_images,
)

# Voice Commands
from .voice_utils import (
    VoiceRecognizer,
    Command,
    parse_command,
    speak,
)

# Video Processing
from .video_utils import (
    read_frames_from_video,
    read_images_from_folder,
    analyze_frame,
    FaceFrame,
    MotionCurve,
    extract_motion_curve,
    auto_capture_from_video,
    auto_capture_from_images,
    save_motion_curve,
    load_motion_curve,
)
