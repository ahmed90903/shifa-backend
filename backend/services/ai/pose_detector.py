"""
pose_detector.py
----------------
Real-time human pose detection using MediaPipe Pose.
Captures webcam frames, detects body landmarks, and exposes
normalized + pixel-coordinate landmark data to downstream modules.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class LandmarkPoint:
    """Holds a single landmark's pixel coords and visibility confidence."""
    x: float          # pixel x
    y: float          # pixel y
    z: float          # relative depth (MediaPipe)
    visibility: float # [0, 1]


# MediaPipe landmark indices we care about
LANDMARK_IDS = {
    # Upper body
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
    # Lower body
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
    # Torso helper
    "LEFT_EAR":       7,
    "RIGHT_EAR":      8,
}

# Connections to draw skeleton lines (pairs of landmark names)
SKELETON_CONNECTIONS = [
    ("LEFT_SHOULDER",  "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER",  "LEFT_ELBOW"),
    ("LEFT_ELBOW",     "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW",    "RIGHT_WRIST"),
    ("LEFT_SHOULDER",  "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP",       "RIGHT_HIP"),
    ("LEFT_HIP",       "LEFT_KNEE"),
    ("LEFT_KNEE",      "LEFT_ANKLE"),
    ("RIGHT_HIP",      "RIGHT_KNEE"),
    ("RIGHT_KNEE",     "RIGHT_ANKLE"),
]


class PoseDetector:
    """
    Wraps MediaPipe Pose for real-time landmark detection.

    Usage
    -----
    detector = PoseDetector()
    while True:
        frame, landmarks = detector.process_frame()
        if frame is None:
            break
        ...
    detector.release()
    """

    def __init__(
        self,
        camera_index: int = 0,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 0,
    ):
        self.cap = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self._frame_w = 640  # Default width for websocket frames
        self._frame_h = 480  # Default height for websocket frames

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Read one frame from the webcam, run pose inference.

        Returns
        -------
        frame : BGR numpy array or None if camera read failed.
        landmarks : dict[str -> LandmarkPoint] or None if no pose detected.
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        frame = cv2.flip(frame, 1)                     # mirror for UX
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if results.pose_landmarks is None:
            return frame, None

        landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
        return frame, landmarks

    def process_image(
        self,
        frame: np.ndarray,
        mirror: bool = True,             # ← NEW: optional mirror for static images
    ) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Process a single image frame (BGR numpy array).

        Parameters
        ----------
        frame : BGR numpy array.
        mirror : if True, mirror the image horizontally (default True
                 for consistency with webcam feed; set False for non‑UX use).

        Returns
        -------
        frame : BGR numpy array (optionally mirrored)
        landmarks : dict[str -> LandmarkPoint] or None if no pose detected.
        """
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if results.pose_landmarks is None:
            return frame, None

        landmarks = self._extract_landmarks(results.pose_landmarks, frame.shape)
        return frame, landmarks

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: dict,
        joint_color: Tuple[int, int, int] = (0, 255, 0),
        line_color: Tuple[int, int, int] = (255, 255, 255),
        highlight_joints: Optional[List[str]] = None,
        highlight_color: Tuple[int, int, int] = (0, 200, 255),
    ) -> np.ndarray:
        """
        Draw skeleton onto frame.

        Parameters
        ----------
        highlight_joints : list of landmark names to draw with a distinct colour
                           (used to emphasise the joints being tracked by the
                           active exercise).
        """
        if landmarks is None:
            return frame

        # Draw connections
        for a_name, b_name in SKELETON_CONNECTIONS:
            if a_name in landmarks and b_name in landmarks:
                a = landmarks[a_name]
                b = landmarks[b_name]
                if a.visibility > 0.4 and b.visibility > 0.4:
                    cv2.line(
                        frame,
                        (int(a.x), int(a.y)),
                        (int(b.x), int(b.y)),
                        line_color, 2, cv2.LINE_AA,
                    )

        # Draw joints
        for name, lm in landmarks.items():
            if lm.visibility < 0.4:
                continue
            color = highlight_color if (highlight_joints and name in highlight_joints) else joint_color
            radius = 8 if (highlight_joints and name in highlight_joints) else 5
            cv2.circle(frame, (int(lm.x), int(lm.y)), radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(lm.x), int(lm.y)), radius + 2, (0, 0, 0), 1, cv2.LINE_AA)

        return frame

    def get_frame_size(self) -> Tuple[int, int]:
        """Return (width, height) of camera frames."""
        return self._frame_w, self._frame_h

    def release(self):
        if self.cap is not None:
            self.cap.release()
        self.pose.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_landmarks(self, pose_landmarks, shape) -> dict:
        h, w = shape[:2]
        result = {}
        for name, idx in LANDMARK_IDS.items():
            lm = pose_landmarks.landmark[idx]
            result[name] = LandmarkPoint(
                x=lm.x * w,
                y=lm.y * h,
                z=lm.z,
                visibility=lm.visibility,
            )
        return result