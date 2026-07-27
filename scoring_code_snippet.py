import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json, numpy as np, os
from types import SimpleNamespace

# ── Init models ───────────────────────────────────────────────────────────────
def init_pose(model_path='pose_landmarker_heavy.task'):
    if os.path.exists(model_path):
        opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(opts)
    return None

def init_face(model_path='blaze_face_short_range.tflite'):
    if os.path.exists(model_path):
        opts = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.4,
        )
        return vision.FaceDetector.create_from_options(opts)
    return None

pose_model = init_pose()  # task-based model if available
face_model = init_face()  # task-based model if available

# Fallback: use mediapipe.solutions if task models are not present
sol_pose = None
sol_face = None
if pose_model is None:
    try:
        sol_pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.4)
    except Exception:
        sol_pose = None
if face_model is None:
    try:
        sol_face = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.4)
    except Exception:
        sol_face = None


# ── Scoring helpers ───────────────────────────────────────────────────────────
def score_alignment(p1, p2, img_h):
    """
    FIX: Use pixel-space Y difference relative to image height.
    A 5% height difference = already quite tilted.
    """
    dy = abs(p1.y - p2.y)          # normalized 0–1
    dy_pct = dy * 100               # as % of image height
    score = max(0, 100 - dy_pct * 8)  # 5% tilt → score 60; 12% → score 4
    return int(score)


def score_body_symmetry(lm):
    """
    FIX: Compare left vs right landmark X positions relative to body center.
    Indices: LShoulder=11, RShoulder=12, LHip=23, RHip=24
    """
    ls, rs = lm[11], lm[12]
    lh, rh = lm[23], lm[24]

    sh_center  = (ls.x + rs.x) / 2
    hip_center = (lh.x + rh.x) / 2
    sh_width   = abs(ls.x - rs.x) + 1e-6

    # Spine tilt = how much hip center deviates from shoulder center
    spine_tilt = abs(hip_center - sh_center) / sh_width
    sym = max(0, 100 - spine_tilt * 150)

    # Left/right reach symmetry
    l_reach = sh_center - ls.x
    r_reach = rs.x - sh_center
    reach_diff = abs(l_reach - r_reach) / (max(l_reach, r_reach) + 1e-6)
    sym -= reach_diff * 30

    return int(max(0, min(100, sym)))


def score_visibility_from_landmarks(lm):
    """
    FIX: Use world_landmarks for visibility, or compute from
    how many key landmarks have high confidence.
    Key upper-body landmarks: nose(0), shoulders(11,12), elbows(13,14),
    wrists(15,16), hips(23,24)
    """
    key_ids = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    vis_vals = []
    for i in key_ids:
        if i < len(lm):
            # presence is more reliable than visibility in Tasks API
            v = getattr(lm[i], 'presence', None) or getattr(lm[i], 'visibility', 0)
            vis_vals.append(float(v))
    if not vis_vals:
        return 50
    return int(min(100, np.mean(vis_vals) * 100))


def score_sharpness(image_bgr):
    """Laplacian variance — normalized properly."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    # typical sharp photo: var > 200; blurry: < 50
    return int(min(100, var / 4.0))


def score_face_orientation(face_res, lm=None):
    """
    FIX: Use nose + ear landmark positions to estimate yaw.
    nose=0, left_ear=7, right_ear=8 in pose landmarks.
    """
    # If face detector fired → at least semi-frontal
    if face_res and getattr(face_res, 'detections', None):
        score = 75
        # Bonus if ears are roughly symmetric (more frontal)
        if lm:
            le = lm[7]; re = lm[8]; nose = lm[0]
            le_dist = abs(nose.x - le.x)
            re_dist = abs(nose.x - re.x)
            ear_sym = 1 - abs(le_dist - re_dist) / (max(le_dist, re_dist) + 1e-6)
            score = int(60 + ear_sym * 30)   # 60–90
        return score

    # No face detected — estimate from pose
    if lm:
        nose_vis = getattr(lm[0], 'presence', getattr(lm[0], 'visibility', 0))
        if nose_vis > 0.7:
            return 45   # probably profile
        elif nose_vis > 0.3:
            return 30   # partial back
        return 15       # back of head

    return 20


def weighted_total(scores):
    return round(
        scores['body_symmetry']      * 0.20 +
        scores['shoulder_alignment'] * 0.20 +
        scores['hip_alignment']      * 0.15 +
        scores['face_orientation']   * 0.15 +
        scores['visibility']         * 0.30,
        1
    )


# ── Main analysis ─────────────────────────────────────────────────────────────
def analyze_images(image_folder):
    results = []
    files   = [f for f in os.listdir(image_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Processing {len(files)} images...")

    for filename in sorted(files):
        path  = os.path.join(image_folder, filename)
        image = cv2.imread(path)
        if image is None:
            continue

        h, w  = image.shape[:2]
        rgb   = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Prepare detection results; support both Tasks API and solutions fallback
        mp_img = None
        pose_res = None
        face_res = None
        if pose_model:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose_res = pose_model.detect(mp_img)
        elif sol_pose:
            sol_res = sol_pose.process(rgb)
            pose_res = SimpleNamespace()
            pose_res.pose_landmarks = [sol_res.pose_landmarks.landmark] if sol_res and sol_res.pose_landmarks else None

        if face_model:
            mp_img = mp_img or mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_res = face_model.detect(mp_img)
        elif sol_face:
            sol_fres = sol_face.process(rgb)
            face_res = SimpleNamespace()
            face_res.detections = sol_fres.detections if sol_fres and getattr(sol_fres, 'detections', None) else None

        sharp = score_sharpness(image)

        # ── Default (no person) ──────────────────────────────────────────
        scores = {
            "body_symmetry":      10,
            "shoulder_alignment": 10,
            "hip_alignment":      10,
            "face_orientation":   10,
            "visibility":         min(sharp, 20),
        }
        annotation = {
            "has_face":           False,
            "face_type":          "none",
            "has_both_shoulders": False,
            "has_hips":           False,
            "is_sharp":           sharp > 25,
            "is_occluded":        True,
            "person_count":       0,
            "pose_notes":         "no person detected",
        }

        # ── Pose scoring ─────────────────────────────────────────────────
        if pose_res and pose_res.pose_landmarks:            # ← FIX: .pose_landmarks not .landmarks
            lm = pose_res.pose_landmarks[0]
            annotation["person_count"] = 1
            annotation["is_occluded"]  = False

            l_sh  = lm[11]; r_sh  = lm[12]
            l_hip = lm[23]; r_hip = lm[24]

            sh_vis  = (getattr(l_sh,  'visibility', 0) + getattr(r_sh,  'visibility', 0)) / 2
            hip_vis = (getattr(l_hip, 'visibility', 0) + getattr(r_hip, 'visibility', 0)) / 2

            annotation["has_both_shoulders"] = sh_vis  > 0.5
            annotation["has_hips"]           = hip_vis > 0.4

            scores["shoulder_alignment"] = score_alignment(l_sh, r_sh, h)
            scores["hip_alignment"]      = score_alignment(l_hip, r_hip, h) if hip_vis > 0.4 else 50
            scores["body_symmetry"]      = score_body_symmetry(lm)
            scores["visibility"]         = max(score_visibility_from_landmarks(lm),
                                               int(sharp * 0.3))

            notes = []
            if scores["shoulder_alignment"] < 70: notes.append("shoulders tilted")
            if scores["body_symmetry"]      < 60: notes.append("body asymmetric")
            if not annotation["has_hips"]:        notes.append("hips not visible")
            if sharp < 30:                        notes.append("image blurry")
            annotation["pose_notes"] = "; ".join(notes) if notes else "good pose alignment"

        # ── Face scoring ─────────────────────────────────────────────────
        lm_for_face = pose_res.pose_landmarks[0] if (pose_res and pose_res.pose_landmarks) else None
        scores["face_orientation"] = score_face_orientation(face_res, lm_for_face)

        if face_res and getattr(face_res, 'detections', None):
            annotation["has_face"]  = True
            annotation["face_type"] = "frontal"

        # ── Final ─────────────────────────────────────────────────────────
        total = weighted_total(scores)
        label = ("excellent" if total >= 85 else "good" if total >= 70
                 else "average" if total >= 50 else "poor")

        results.append({
            "file":          filename,
            "scores":        scores,
            "total":         total,
            "quality_label": label,
            "annotation":    annotation,
        })
        print(f"  {filename}: {total} ({label})")

    return results


# ── Run ───────────────────────────────────────────────────────────────────────
IMAGE_PATH = r"D:\CDAC PGCP AI\Project Idea\Pose Suggestor\unsplash_pose_dataset"

output = analyze_images(IMAGE_PATH)

with open("unsplash_annotations.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nDone! {len(output)} images → unsplash_annotations.json")
