#Alexander Tu

import sys
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker.task"
FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def percent(good: int, valid: int):
    if valid == 0:
        return None
    return round((good / valid) * 100.0, 1)


def analyze(video_path: str, sample_fps: int = 10):
    video_file = Path(video_path)

    if not video_file.exists():
        return {"error": f"video_not_found: {video_file}"}

    if not POSE_MODEL_PATH.exists():
        return {"error": f"missing_pose_model: {POSE_MODEL_PATH}"}

    if not FACE_MODEL_PATH.exists():
        return {"error": f"missing_face_model: {FACE_MODEL_PATH}"}

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        return {"error": f"could_not_open_video: {video_file}"}

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    frame_step = max(1, int(round(fps / sample_fps)))

    total_sampled = 0
    posture_valid = 0
    posture_good = 0
    eye_valid = 0
    eye_good = 0
    expression_valid = 0
    expression_good = 0

    SHOULDER_LEVEL_MAX = 0.06
    LEAN_MAX = 0.18
    FACE_YAW_MAX = 12.0
    FACE_PITCH_MAX = 12.0
    SMILE_MIN = 0.25
    FROWN_MAX = 0.35
    MOUTH_PUCKER_MAX = 0.35

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    with vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.FaceLandmarker.create_from_options(face_options) as face_landmarker:

        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            frame_idx += 1
            total_sampled += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                landmarks = pose_result.pose_landmarks[0]

                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12
                LEFT_HIP = 23
                RIGHT_HIP = 24
                NOSE = 0

                left_shoulder = landmarks[LEFT_SHOULDER]
                right_shoulder = landmarks[RIGHT_SHOULDER]
                left_hip = landmarks[LEFT_HIP]
                right_hip = landmarks[RIGHT_HIP]
                nose = landmarks[NOSE]

                posture_valid += 1

                shoulder_level_diff = abs(left_shoulder.y - right_shoulder.y)
                hip_center_x = (left_hip.x + right_hip.x) / 2.0
                lean = abs(nose.x - hip_center_x)

                if shoulder_level_diff <= SHOULDER_LEVEL_MAX and lean <= LEAN_MAX:
                    posture_good += 1

            if face_result.face_landmarks and len(face_result.face_landmarks) > 0:
                face_landmarks = face_result.face_landmarks[0]
                eye_valid += 1
                expression_valid += 1

                left_eye_outer = np.array([face_landmarks[33].x, face_landmarks[33].y])
                right_eye_outer = np.array([face_landmarks[263].x, face_landmarks[263].y])
                nose_tip = np.array([face_landmarks[1].x, face_landmarks[1].y])

                eye_width = float(np.linalg.norm(left_eye_outer - right_eye_outer)) + 1e-6
                left_dist = float(np.linalg.norm(nose_tip - left_eye_outer)) / eye_width
                right_dist = float(np.linalg.norm(nose_tip - right_eye_outer)) / eye_width

                yaw_ratio = left_dist - right_dist
                yaw_deg = yaw_ratio * 60.0

                eye_line_y = (left_eye_outer[1] + right_eye_outer[1]) / 2.0
                pitch_ratio = (nose_tip[1] - eye_line_y) / eye_width
                pitch_deg = pitch_ratio * 60.0

                if abs(yaw_deg) <= FACE_YAW_MAX and abs(pitch_deg) <= FACE_PITCH_MAX:
                    eye_good += 1

                blendshape_map = {}
                if face_result.face_blendshapes and len(face_result.face_blendshapes) > 0:
                    categories = face_result.face_blendshapes[0]
                    for cat in categories:
                        blendshape_map[cat.category_name] = cat.score

                smile_left = blendshape_map.get("mouthSmileLeft", 0.0)
                smile_right = blendshape_map.get("mouthSmileRight", 0.0)
                brow_inner_up = blendshape_map.get("browInnerUp", 0.0)
                frown_left = blendshape_map.get("mouthFrownLeft", 0.0)
                frown_right = blendshape_map.get("mouthFrownRight", 0.0)
                pucker = blendshape_map.get("mouthPucker", 0.0)

                smile_score = max(smile_left, smile_right)
                frown_score = max(frown_left, frown_right)

                if (
                    (smile_score >= SMILE_MIN or brow_inner_up >= 0.15)
                    and frown_score <= FROWN_MAX
                    and pucker <= MOUTH_PUCKER_MAX
                ):
                    expression_good += 1

    cap.release()

    return {
        "video": {
            "sample_fps": sample_fps,
            "sampled_frames": total_sampled,
        },
        "posture": {
            "valid_frames": posture_valid,
            "good_frames": posture_good,
            "good_percent": percent(posture_good, posture_valid),
        },
        "eye_contact": {
            "valid_frames": eye_valid,
            "good_frames": eye_good,
            "good_percent": percent(eye_good, eye_valid),
        },
        "facial_expression": {
            "valid_frames": expression_valid,
            "good_frames": expression_good,
            "good_percent": percent(expression_good, expression_valid),
        },
        "summary": {
            "notes": [
                "Posture is estimated using shoulder balance and lean.",
                "Eye contact is estimated from face orientation, not true gaze tracking.",
                "Facial expression uses Face Landmarker blendshape scores."
            ]
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing video_path argument"}))
        sys.exit(1)

    result = analyze(sys.argv[1])
    print(json.dumps(result))
