"""
elbow_exercise.py
-----------------
Elbow Flexion / Extension rehabilitation exercise module.

... (الوصف كما في الأصل) ...
"""

import time
import math
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import numpy as np

from .pose_detector import LandmarkPoint
from .angle_calculator import (
    AngleSmoother, AngleVelocityTracker,
    elbow_angle, shoulder_elevation_delta, wrist_deviation,
)
from .feedback_system import FeedbackStatus
from .voice_feedback import VoiceMessages

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
class ElbowConfig:
    FLEX_TARGET_ANGLE   = 65.0
    FLEX_IDEAL_ANGLE    = 55.0
    EXTEND_THRESHOLD    = 155.0
    UNSAFE_MIN_ANGLE    = 20.0
    UNSAFE_MAX_ANGLE    = 190.0
    MAX_SHOULDER_RISE   = 0.08
    MAX_WRIST_DEVIATION = 45.0
    MAX_VELOCITY        = 28.0
    HOLD_FRAMES_REQUIRED = 3
    MAX_UPPER_ARM_TILT  = 25.0

# ---------------------------------------------------------------------------
class ElbowState(Enum):
    IDLE      = auto()
    EXTENDING = auto()
    BENDING   = auto()
    HOLD      = auto()

# ---------------------------------------------------------------------------
@dataclass
class ElbowRepRecord:
    rep_number:      int
    peak_flex_angle: float
    is_correct:      bool
    issues:          List[str] = field(default_factory=list)
    score:           float = 0.0

# ---------------------------------------------------------------------------
@dataclass
class ElbowSessionMetrics:
    exercise_name:  str = "Elbow Flexion / Extension"
    total_reps:     int = 0
    correct_reps:   int = 0
    accuracy:       float = 0.0
    avg_angle:      float = 0.0
    session_start:  float = field(default_factory=time.monotonic)
    session_end:    Optional[float] = None
    rep_records:    List[ElbowRepRecord] = field(default_factory=list)
    angle_samples:  List[float] = field(default_factory=list)
    current_accuracy: float = 0.0

    @property
    def duration(self) -> float:
        end = self.session_end or time.monotonic()
        return end - self.session_start

    def update_accuracy(self):
        if self.rep_records:
            self.accuracy = sum(r.score for r in self.rep_records) / len(self.rep_records)
        else:
            self.accuracy = self.current_accuracy
        if self.angle_samples:
            self.avg_angle = float(np.mean(self.angle_samples))

# ---------------------------------------------------------------------------
class ElbowExercise:
    SIDE_PREFERENCE = ["RIGHT", "LEFT"]

    def __init__(self, target_reps: int = 10, voice_enabled: bool = True):
        self.target_reps    = target_reps
        self.cfg            = ElbowConfig()
        self.metrics        = ElbowSessionMetrics()

        self._state             = ElbowState.IDLE
        self._smoother          = AngleSmoother(alpha=0.38)
        self._velocity_tracker  = AngleVelocityTracker(history_len=6)

        self._current_angle     = 0.0
        self._peak_flex_angle   = 180.0
        self._hold_counter      = 0
        self._rep_is_valid      = True
        self._form_penalty      = 0.0
        self._max_progress      = 0.0

        self._shoulder_baseline_y: Optional[float] = None
        self._forearm_ref_angle: Optional[float]   = None   # store in degrees now
        self._upper_arm_baseline_angle: Optional[float] = None
        self._calibration_frames = 0

        self._active_side: Optional[str] = None
        self.active_joints: Tuple[str, str, str] = (
            "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"
        )

        self.feedback_msg:   str = "Hold your arm straight by your side"
        self.status:         str = FeedbackStatus.NEUTRAL
        self.voice_msg:      Optional[str] = None
        self.voice_priority: bool = False

        logger.info("ElbowExercise initialised — target reps: %d", target_reps)

    # ------------------------------------------------------------------
    # Properties needed by main.py
    # ------------------------------------------------------------------
    @property
    def exercise_name(self) -> str:
        return self.metrics.exercise_name

    @property
    def start_time(self) -> float:
        return self.metrics.session_start

    @property
    def rep_count(self) -> int:
        return self.metrics.total_reps

    @property
    def accuracy(self) -> float:
        return self.metrics.accuracy

    @property
    def current_angle(self) -> float:
        return self._current_angle

    @property
    def session_complete(self) -> bool:
        return self.metrics.total_reps >= self.target_reps

    @property
    def phase_label(self) -> str:
        return {
            ElbowState.IDLE:      "READY",
            ElbowState.EXTENDING: "EXTENDING",
            ElbowState.BENDING:   "CURLING",
            ElbowState.HOLD:      "HOLD",
        }.get(self._state, "")

    @property
    def progress(self) -> float:
        if self._state in (ElbowState.IDLE, ElbowState.EXTENDING):
            return 0.0
        total = self.cfg.EXTEND_THRESHOLD - self.cfg.FLEX_TARGET_ANGLE
        done  = self.cfg.EXTEND_THRESHOLD - self._current_angle
        return float(np.clip(done / max(total, 1), 0, 1))

    # ------------------------------------------------------------------
    def update(self, landmarks: Dict[str, LandmarkPoint], frame_h: int = 720) -> None:
        self.voice_msg      = None
        self.voice_priority = False

        side = self._pick_side(landmarks)
        if side is None:
            self.feedback_msg = "Face the camera and hold arm straight"
            self.status       = FeedbackStatus.NEUTRAL
            return

        sh_name  = f"{side}_SHOULDER"
        el_name  = f"{side}_ELBOW"
        wr_name  = f"{side}_WRIST"
        self.active_joints = (sh_name, el_name, wr_name)

        shoulder = landmarks[sh_name]
        elbow_lm = landmarks[el_name]
        wrist    = landmarks[wr_name]

        raw_angle = elbow_angle(shoulder, elbow_lm, wrist)
        smooth    = self._smoother.update(raw_angle)
        velocity  = self._velocity_tracker.update(smooth)
        self._current_angle = smooth
        self.metrics.angle_samples.append(smooth)

        self._calibrate_baseline(shoulder, elbow_lm, wrist, frame_h)

        if self._check_unsafe(smooth):
            self.metrics.update_accuracy()
            return

        issues = self._check_form(shoulder, elbow_lm, wrist, velocity, frame_h)
        self._run_state_machine(smooth, issues)
        self.metrics.update_accuracy()

    # ------------------------------------------------------------------
    def _run_state_machine(self, angle: float, issues: List[str]):
        prev = self._state

        if self._state == ElbowState.IDLE:
            if angle >= self.cfg.EXTEND_THRESHOLD:
                self._state = ElbowState.EXTENDING
                self._reset_rep()
                self.feedback_msg = "Start curling your arm"
                self.status       = FeedbackStatus.NEUTRAL

        elif self._state == ElbowState.EXTENDING:
            if angle < self.cfg.EXTEND_THRESHOLD:
                self._state = ElbowState.BENDING
            self.feedback_msg = "Curl your arm upward"
            self.status       = FeedbackStatus.NEUTRAL

        elif self._state == ElbowState.BENDING:
            if angle < self._peak_flex_angle:
                self._peak_flex_angle = angle

            if issues:
                self._rep_is_valid = False
                self.status        = FeedbackStatus.WARNING
                self.feedback_msg  = issues[0]
                if len(self.metrics.angle_samples) % 30 == 1:
                    self.voice_msg = issues[0]
            else:
                self.status       = FeedbackStatus.CORRECT
                self.feedback_msg = "Good form"

            if angle > self.cfg.FLEX_TARGET_ANGLE + 25:
                self.feedback_msg = "Bend your elbow more"
                self.status       = FeedbackStatus.WARNING

            if angle <= self.cfg.FLEX_TARGET_ANGLE:
                self._hold_counter += 1
                if self._hold_counter >= self.cfg.HOLD_FRAMES_REQUIRED:
                    self._state = ElbowState.HOLD
            else:
                self._hold_counter = 0

        elif self._state == ElbowState.HOLD:
            self.feedback_msg = "Now lower your arm fully"
            self.status       = FeedbackStatus.CORRECT

            if angle >= self.cfg.EXTEND_THRESHOLD:
                self._complete_rep(issues)
                self._state = ElbowState.EXTENDING
                self._reset_rep()

        if self._state in (ElbowState.BENDING, ElbowState.HOLD):
            if issues:
                self._form_penalty += 0.5
            self._max_progress = max(self._max_progress, self.progress)
            self.metrics.current_accuracy = float(np.clip(self._max_progress * 100.0 - self._form_penalty, 0.0, 100.0))
        else:
            self.metrics.current_accuracy = 0.0

        if self._state != prev:
            logger.debug("Elbow state: %s → %s  angle=%.1f",
                         prev.name, self._state.name, angle)

    # ------------------------------------------------------------------
    def _reset_rep(self):
        self._peak_flex_angle = 180.0
        self._hold_counter    = 0
        self._rep_is_valid    = True
        self._form_penalty    = 0.0
        self._max_progress    = 0.0

    def _complete_rep(self, issues: List[str]):
        is_correct = self._rep_is_valid and len(issues) == 0
        self.metrics.total_reps += 1
        if is_correct:
            self.metrics.correct_reps += 1

        self.metrics.rep_records.append(ElbowRepRecord(
            rep_number      = self.metrics.total_reps,
            peak_flex_angle = self._peak_flex_angle,
            is_correct      = is_correct,
            issues          = list(issues),
            score           = self.metrics.current_accuracy,
        ))

        if is_correct:
            self.feedback_msg = f"Rep {self.metrics.total_reps} complete! Nice curl"
            self.voice_msg    = VoiceMessages.ELBOW_REP_COMPLETE
        else:
            self.feedback_msg = f"Rep {self.metrics.total_reps} done — improve your form"
            self.voice_msg    = VoiceMessages.ELBOW_GOOD_FORM

        logger.info("Elbow rep %d completed — correct=%s peak_flex=%.1f°",
                    self.metrics.total_reps, is_correct, self._peak_flex_angle)

    # ------------------------------------------------------------------
    def _check_unsafe(self, angle: float) -> bool:
        if angle < self.cfg.UNSAFE_MIN_ANGLE or angle > self.cfg.UNSAFE_MAX_ANGLE:
            self.feedback_msg   = "⛔ UNSAFE MOVEMENT — STOP"
            self.status         = FeedbackStatus.ERROR
            self.voice_msg      = VoiceMessages.ELBOW_ANGLE_UNSAFE
            self.voice_priority = True
            self._rep_is_valid  = False
            return True
        return False

    # ------------------------------------------------------------------
    def _check_form(self, shoulder: LandmarkPoint, elbow_lm: LandmarkPoint,
                    wrist: LandmarkPoint, velocity: float, frame_h: int) -> List[str]:
        issues: List[str] = []

        if self._shoulder_baseline_y is not None:
            rise = shoulder_elevation_delta(
                self._shoulder_baseline_y, shoulder.y, frame_h
            )
            if rise > self.cfg.MAX_SHOULDER_RISE:
                issues.append("Keep your elbow stable")

        if self._forearm_ref_angle is not None:
            # forearm_ref_angle is now in degrees
            dev = wrist_deviation(elbow_lm, wrist, self._forearm_ref_angle)
            if dev > self.cfg.MAX_WRIST_DEVIATION:
                issues.append("Perform the movement correctly")

        current_ua_angle = math.degrees(
            math.atan2(elbow_lm.y - shoulder.y, elbow_lm.x - shoulder.x)
        )
        if self._upper_arm_baseline_angle is not None:
            tilt = abs(current_ua_angle - self._upper_arm_baseline_angle)
            if tilt > 180:
                tilt = 360 - tilt
            if tilt > self.cfg.MAX_UPPER_ARM_TILT:
                issues.append("Keep your elbow fixed")

        if velocity > self.cfg.MAX_VELOCITY:
            issues.append("Move your arm slowly")

        return issues

    # ------------------------------------------------------------------
    def _calibrate_baseline(self, shoulder: LandmarkPoint, elbow_lm: LandmarkPoint,
                            wrist: LandmarkPoint, frame_h: int):
        if self._calibration_frames >= 15:
            return

        alpha = 1.0 / (self._calibration_frames + 1)

        if self._shoulder_baseline_y is None:
            self._shoulder_baseline_y = shoulder.y
        else:
            self._shoulder_baseline_y = (
                (1 - alpha) * self._shoulder_baseline_y + alpha * shoulder.y
            )

        # تخزين الزاوية المرجعية للساعد بالدرجات
        forearm_angle_rad = math.atan2(wrist.y - elbow_lm.y, wrist.x - elbow_lm.x)
        forearm_angle_deg = math.degrees(forearm_angle_rad)
        if self._forearm_ref_angle is None:
            self._forearm_ref_angle = forearm_angle_deg
        else:
            self._forearm_ref_angle = (
                (1 - alpha) * self._forearm_ref_angle + alpha * forearm_angle_deg
            )

        ua_angle = math.degrees(
            math.atan2(elbow_lm.y - shoulder.y, elbow_lm.x - shoulder.x)
        )
        if self._upper_arm_baseline_angle is None:
            self._upper_arm_baseline_angle = ua_angle
        else:
            self._upper_arm_baseline_angle = (
                (1 - alpha) * self._upper_arm_baseline_angle + alpha * ua_angle
            )

        self._calibration_frames += 1

    # ------------------------------------------------------------------
    def _pick_side(self, landmarks: Dict[str, LandmarkPoint]) -> Optional[str]:
        if self._active_side:
            needed = [f"{self._active_side}_{j}"
                      for j in ("SHOULDER", "ELBOW", "WRIST")]
            if all(landmarks.get(n) and landmarks[n].visibility > 0.5 for n in needed):
                return self._active_side

        for side in self.SIDE_PREFERENCE:
            needed = [f"{side}_{j}" for j in ("SHOULDER", "ELBOW", "WRIST")]
            if all(landmarks.get(n) and landmarks[n].visibility > 0.5 for n in needed):
                self._active_side = side
                return side
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def safe_angle_range() -> Tuple[float, float]:
        return ElbowConfig.FLEX_TARGET_ANGLE, ElbowConfig.EXTEND_THRESHOLD

    def get_session_summary(self) -> Dict:
        self.metrics.session_end = time.monotonic()
        self.metrics.update_accuracy()
        return {
            "exercise":     self.metrics.exercise_name,
            "reps":         self.metrics.total_reps,
            "correct_reps": self.metrics.correct_reps,
            "accuracy":     round(self.metrics.accuracy, 1),
            "avg_angle":    round(self.metrics.avg_angle, 1),
            "duration":     round(self.metrics.duration, 1),
        }