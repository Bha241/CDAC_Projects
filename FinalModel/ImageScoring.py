"""
Pose Scorer — scores 800 human pose images
Outputs: poses_scored.csv with filename + score (0-100)

Scoring criteria:
  - Visibility: all key joints detected?
  - Balance: body symmetry / center of gravity
  - Clarity: joints not occluded
  - Spread: how expressive/open the pose is
  - Alignment: spine/shoulder/hip alignment
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from pathlib import Path
import math
from types import SimpleNamespace
from tqdm import tqdm

pose_model_tasks = None
mp_pose = None
use_tasks = False
PoseLandmark = None

try:
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision as mp_tasks_vision
except Exception:
    mp_tasks_python = None
    mp_tasks_vision = None


def load_pose_model():
    global mp_pose, pose_model_tasks, use_tasks, PoseLandmark
    
    # First, try to use mediapipe.solutions (most compatible)
    if hasattr(mp, 'solutions') and getattr(mp, 'solutions') is not None:
        try:
            mp_pose = mp.solutions.pose
            PoseLandmark = mp_pose.PoseLandmark
            use_tasks = False
            return
        except Exception as e:
            print(f"Warning: Could not load mp.solutions.pose: {e}")
            mp_pose = None

    # Try task-based model as fallback
    if mp_tasks_python is not None and mp_tasks_vision is not None:
        model_path = Path('pose_landmarker_heavy.task')
        if model_path.exists():
            try:
                opts = mp_tasks_vision.PoseLandmarkerOptions(
                    base_options=mp_tasks_python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_tasks_vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.5,
                    min_pose_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_segmentation_masks=False,
                )
                pose_model_tasks = mp_tasks_vision.PoseLandmarker.create_from_options(opts)
                PoseLandmark = mp_tasks_vision.PoseLandmark
                use_tasks = True
                return
            except Exception as e:
                print(f"Warning: Could not load task-based model: {e}")
                pose_model_tasks = None

    raise RuntimeError(
        'No compatible MediaPipe pose model available. '
        'Install a supported mediapipe or place pose_landmarker_heavy.task in the workspace.'
    )


class PoseTaskResult:
    def __init__(self, landmarks):
        self.pose_landmarks = SimpleNamespace(landmark=landmarks) if landmarks else None


# ── helpers ──────────────────────────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle at point b formed by a-b-c (degrees)."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


def landmark_vis(lm, indices):
    """Mean visibility of given landmark indices."""
    return np.mean([lm[i].visibility for i in indices])

# ── scoring components ───────────────────────────────────────────────────────

def score_visibility(lm):
    """All key joints visible? Max 25 pts."""
    key = [
        PoseLandmark.NOSE,
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,    PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.LEFT_WRIST,    PoseLandmark.RIGHT_WRIST,
        PoseLandmark.LEFT_HIP,      PoseLandmark.RIGHT_HIP,
        PoseLandmark.LEFT_KNEE,     PoseLandmark.RIGHT_KNEE,
        PoseLandmark.LEFT_ANKLE,    PoseLandmark.RIGHT_ANKLE,
    ]
    vis = landmark_vis(lm, [k.value for k in key])
    return vis * 25.0


def score_balance(lm):
    """Horizontal symmetry around center. Max 20 pts."""
    l_sh = lm[PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[PoseLandmark.RIGHT_SHOULDER.value]
    l_hp = lm[PoseLandmark.LEFT_HIP.value]
    r_hp = lm[PoseLandmark.RIGHT_HIP.value]

    shoulder_sym = 1 - abs(l_sh.y - r_sh.y)
    hip_sym      = 1 - abs(l_hp.y - r_hp.y)
    center_x     = (l_sh.x + r_sh.x + l_hp.x + r_hp.x) / 4
    center_bias  = 1 - 2 * abs(center_x - 0.5)

    score = (shoulder_sym * 0.4 + hip_sym * 0.4 + center_bias * 0.2) * 20
    return max(0, score)


def score_spine_alignment(lm):
    """Spine roughly vertical or naturally curved. Max 20 pts."""
    nose   = lm[PoseLandmark.NOSE.value]
    mid_sh = type('P', (), {
        'x': (lm[PoseLandmark.LEFT_SHOULDER.value].x + lm[PoseLandmark.RIGHT_SHOULDER.value].x) / 2,
        'y': (lm[PoseLandmark.LEFT_SHOULDER.value].y + lm[PoseLandmark.RIGHT_SHOULDER.value].y) / 2,
    })()
    mid_hp = type('P', (), {
        'x': (lm[PoseLandmark.LEFT_HIP.value].x + lm[PoseLandmark.RIGHT_HIP.value].x) / 2,
        'y': (lm[PoseLandmark.LEFT_HIP.value].y + lm[PoseLandmark.RIGHT_HIP.value].y) / 2,
    })()

    horizontal_lean = abs(mid_sh.x - mid_hp.x)
    lean_score = max(0, 1 - horizontal_lean * 3)

    spine_angle = angle_between(nose, mid_sh, mid_hp)
    angle_score = min(1.0, spine_angle / 170.0)

    return (lean_score * 0.5 + angle_score * 0.5) * 20


def score_expressiveness(lm):
    """How open/expressive the pose. Arm spread, leg spread. Max 20 pts."""
    l_wr = lm[PoseLandmark.LEFT_WRIST.value]
    r_wr = lm[PoseLandmark.RIGHT_WRIST.value]
    l_an = lm[PoseLandmark.LEFT_ANKLE.value]
    r_an = lm[PoseLandmark.RIGHT_ANKLE.value]
    l_sh = lm[PoseLandmark.LEFT_SHOULDER.value]
    r_sh = lm[PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_width = abs(l_sh.x - r_sh.x) + 1e-6
    arm_spread  = abs(l_wr.x - r_wr.x) / shoulder_width
    leg_spread  = abs(l_an.x - r_an.x) / shoulder_width

    arm_score = min(1.0, arm_spread / 2.5)
    leg_score = min(1.0, leg_spread / 1.5)

    return (arm_score * 0.6 + leg_score * 0.4) * 20


def score_joint_angles(lm):
    """Elbow/knee bend adds dynamism. Max 15 pts."""
    l_elbow = angle_between(
        lm[PoseLandmark.LEFT_SHOULDER.value],
        lm[PoseLandmark.LEFT_ELBOW.value],
        lm[PoseLandmark.LEFT_WRIST.value],
    )
    r_elbow = angle_between(
        lm[PoseLandmark.RIGHT_SHOULDER.value],
        lm[PoseLandmark.RIGHT_ELBOW.value],
        lm[PoseLandmark.RIGHT_WRIST.value],
    )
    l_knee = angle_between(
        lm[PoseLandmark.LEFT_HIP.value],
        lm[PoseLandmark.LEFT_KNEE.value],
        lm[PoseLandmark.LEFT_ANKLE.value],
    )
    r_knee = angle_between(
        lm[PoseLandmark.RIGHT_HIP.value],
        lm[PoseLandmark.RIGHT_KNEE.value],
        lm[PoseLandmark.RIGHT_ANKLE.value],
    )

    def bend_score(angle):
        if 60 <= angle <= 150:
            return 1.0
        elif angle < 60:
            return angle / 60
        else:
            return max(0, (180 - angle) / 30)

    avg = np.mean([bend_score(l_elbow), bend_score(r_elbow),
                   bend_score(l_knee),  bend_score(r_knee)])
    return avg * 15


def quality_label(score, status):
    if status != 'ok' or score is None:
        return 'poor'
    if score >= 85:
        return 'excellent'
    if score >= 70:
        return 'good'
    if score >= 55:
        return 'average'
    return 'poor'


# ── main scorer ───────────────────────────────────────────────────────────────

def score_image(img_path, pose_model):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, "cannot_read"

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_tasks:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = pose_model.detect(mp_img)
        if not result.pose_landmarks:
            return 0.0, "no_pose_detected"
        result = PoseTaskResult(result.pose_landmarks[0])
    else:
        result = pose_model.process(img_rgb)
        if not result.pose_landmarks:
            return 0.0, "no_pose_detected"

    lm = result.pose_landmarks.landmark

    v  = score_visibility(lm)
    b  = score_balance(lm)
    sp = score_spine_alignment(lm)
    ex = score_expressiveness(lm)
    ja = score_joint_angles(lm)

    total = v + b + sp + ex + ja
    return round(total, 2), "ok"


def score_folder(image_dir, output_csv="poses_scored.csv"):
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    print(f"Found {len(images)} images in {image_dir}")

    records = []
    if use_tasks:
        pose = pose_model_tasks
        for img_path in tqdm(images, desc="Scoring poses"):
            score, status = score_image(img_path, pose)
            records.append({
                "filename":      img_path.name,
                "filepath":      str(img_path),
                "score":         score,
                "quality_label": quality_label(score, status),
                "status":        status,
            })
    else:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        ) as pose:
            for img_path in tqdm(images, desc="Scoring poses"):
                score, status = score_image(img_path, pose)
                records.append({
                    "filename":      img_path.name,
                    "filepath":      str(img_path),
                    "score":         score,
                    "quality_label": quality_label(score, status),
                    "status":        status,
                })

    df = pd.DataFrame(records).sort_values("score", ascending=False)
    # Ensure the output has the notebook-compatible label columns first.
    df = df[["filename", "score", "quality_label", "filepath", "status"]]
    df.to_csv(output_csv, index=False)
    print(f"\nDone! Saved → {output_csv}")
    print(df["score"].describe())
    print(f"\nTop 5 poses:\n{df.head()}")
    print(f"\nFailed (no pose detected): {(df['status'] != 'ok').sum()}")
    return df


if __name__ == "__main__":
    import argparse
    load_pose_model()

    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", help="Folder with pose images")
    parser.add_argument("--output", default="poses_scored.csv")
    args = parser.parse_args()

    score_folder(args.image_dir, args.output)
