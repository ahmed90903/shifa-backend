"""
backend/services/pose_analyzer.py
Core AI: MediaPipe → angle calculation → rep counter → feedback.
Rewritten to use the advanced AI modules from `backend.services.ai`.
"""
import base64
import time
from typing import Optional

import cv2
import numpy as np

from backend.services.ai.pose_detector import PoseDetector
from backend.services.ai.knee_exercise import KneeExercise
from backend.services.ai.elbow_exercise import ElbowExercise
from backend.services.ai.feedback_system import FeedbackSystem, HUDData, FeedbackStatus


class PoseAnalyzer:
    """
    One instance per WebSocket connection.
    Decodes a base64 JPEG, runs MediaPipe, returns a result dict.
    """

    def __init__(self, exercise_key: str):
        self.exercise_key = exercise_key

        # لا حاجة لفتح الكاميرا، نستعمل process_image فقط
        self.detector = PoseDetector(
            camera_index=None,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.frame_w = 640
        self.frame_h = 480
        self.hud = FeedbackSystem(frame_w=self.frame_w, frame_h=self.frame_h)

        if exercise_key == "knee_extension":
            self.exercise = KneeExercise(target_reps=10, voice_enabled=False)
        elif exercise_key == "bicep_curl":
            self.exercise = ElbowExercise(target_reps=10, voice_enabled=False)
        else:
            self.exercise = None

    def analyze(self, frame_b64: str) -> dict:
        """Decode frame, run pose, return result dict."""
        if not self.exercise:
            return self._error_response("Unsupported exercise")

        # فك تشفير الصورة
        try:
            if "," in frame_b64:
                frame_b64 = frame_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return self._error_response("Failed to decode frame")
        except Exception as e:
            return self._error_response(f"Decode error: {e}")

        # مزامنة أبعاد HUD
        if frame.shape[1] != self.hud.fw or frame.shape[0] != self.hud.fh:
            self.hud.fw = frame.shape[1]
            self.hud.fh = frame.shape[0]
            self.frame_w = frame.shape[1]
            self.frame_h = frame.shape[0]

        # تشغيل MediaPipe
        frame, landmarks = self.detector.process_image(frame, mirror=True)

        if not landmarks:
            frame = self.hud.render_no_pose(frame)
            return {
                **self._base_response(),
                "landmarks_visible": False,
                "feedback_message": "Position yourself so your full body is visible",
                "annotated_frame": self._encode_frame(frame),
            }

        # تحديث التمرين
        self.exercise.update(landmarks, frame_h=self.frame_h)

        # بناء بيانات HUD
        a_min, a_max = (self.exercise.safe_angle_range()
                        if hasattr(self.exercise, "safe_angle_range")
                        else (0, 180))

        hud_data = HUDData(
            exercise_name  = self.exercise.exercise_name,
            current_angle  = self.exercise.current_angle,
            rep_count      = self.exercise.rep_count,
            target_reps    = self.exercise.target_reps,
            accuracy       = self.exercise.accuracy,
            status         = self.exercise.status,
            phase_label    = self.exercise.phase_label,
            feedback_msg   = self.exercise.feedback_msg,
            progress       = self.exercise.progress,
            session_time   = time.monotonic() - self.exercise.start_time,
            angle_min      = a_min,
            angle_max      = a_max,
        )

        # رسم الهيكل
        highlight = list(getattr(self.exercise, "active_joints", []))
        frame = self.detector.draw_skeleton(
            frame, landmarks,
            highlight_joints=highlight,
            joint_color=(180, 180, 180),
            line_color=(100, 100, 100),
        )

        # عرض HUD
        frame = self.hud.render(
            frame, hud_data,
            landmarks       = landmarks,
            exercise_joints = getattr(self.exercise, "active_joints", []),
        )

        max_angle = getattr(self.exercise, "_peak_flex_angle", 0.0)
        # ⬇️ إضافة الدقة اللحظية
        instant_accuracy = getattr(self.exercise, "current_accuracy", 0.0)

        return {
            "joint_angle": round(self.exercise.current_angle, 2),
            "is_correct_form": self.exercise.status == FeedbackStatus.CORRECT,
            "feedback_message": self.exercise.feedback_msg,
            "rep_count": self.exercise.rep_count,
            "stage": self.exercise.phase_label,
            "progress": round(self.exercise.progress, 2),
            "max_angle_achieved": round(max_angle, 2),
            "session_accuracy": round(self.exercise.accuracy, 2),
            "current_accuracy": round(instant_accuracy, 2),   # ← المضاف
            "landmarks_visible": True,
            "annotated_frame": self._encode_frame(frame),
        }

    def close(self) -> None:
        if self.detector:
            self.detector.release()

    def get_session_summary(self) -> dict:
        if not self.exercise:
            return {"total_reps": 0, "accuracy_score": 0.0, "max_angle_achieved": 0.0}

        summary = self.exercise.get_session_summary()
        max_angle = getattr(self.exercise, "_peak_flex_angle", 0.0)
        return {
            "total_reps": summary.get("reps", 0),
            "accuracy_score": summary.get("accuracy", 0.0),
            "max_angle_achieved": round(max_angle, 2),
        }

    # ── helper methods ─────────────────────────────────────────────────────
    def _encode_frame(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _base_response(self) -> dict:
        return {
            "joint_angle": 0.0,
            "is_correct_form": False,
            "rep_count": 0,
            "stage": "up",
            "max_angle_achieved": 0.0,
            "session_accuracy": 0.0,
            "current_accuracy": 0.0,
            "landmarks_visible": False,
            "annotated_frame": "",
        }

    def _error_response(self, msg: str) -> dict:
        return {**self._base_response(), "feedback_message": msg}