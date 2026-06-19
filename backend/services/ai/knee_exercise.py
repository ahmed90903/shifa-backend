"""
knee_exercise.py
----------------
Knee Flexion / Extension rehabilitation exercise module.

State machine
-------------
IDLE           → waiting for patient to enter start position
EXTENDING      → leg approaching full extension (angle near 170°+)
BENDING        → knee flexing toward target angle
HOLD           → patient held at flexion target (optional hold requirement)

A repetition is counted when:
  1. Knee bends to ≤ FLEX_TARGET_ANGLE (patient reaches flexion goal).
  2. Knee returns to ≥ EXTEND_THRESHOLD (full extension).

Safety rule: if angle ever drops below UNSAFE_MIN or exceeds UNSAFE_MAX
             the rep is invalidated and a priority voice alert fires.
"""

import time
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import numpy as np

from .pose_detector import LandmarkPoint
from .angle_calculator import (
    AngleSmoother, AngleVelocityTracker,
    knee_angle, lateral_lean_angle,
)
from .feedback_system import FeedbackStatus
from .voice_feedback import VoiceMessages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------

class KneeConfig:
    # Target range for flexion (full bend)
    FLEX_TARGET_ANGLE   = 100.0     # must reach ≤ this to count
    FLEX_IDEAL_ANGLE    = 90.0      # ideal 90° bend

    # Full extension threshold
    EXTEND_THRESHOLD    = 160.0     # must return to ≥ this to complete rep

    # Safety absolute limits
    UNSAFE_MIN_ANGLE    = 40.0      # hyper-flexion danger
    UNSAFE_MAX_ANGLE    = 185.0     # hyper-extension danger

    # Lean / posture limits
    MAX_LATERAL_LEAN    = 35.0      # degrees from vertical (now absolute)
    HIP_DRIFT_THRESHOLD = 60        # pixels; hip should stay relatively fixed

    # Speed limits (degrees / frame at ~30 FPS)
    MAX_VELOCITY        = 30.0
    SLOW_VELOCITY       = 0.3       # threshold below which motion is "frozen"

    # How long angle must stay in target zone to register flexion (frames)
    HOLD_FRAMES_REQUIRED = 3


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class KneeState(Enum):
    IDLE      = auto()
    EXTENDING = auto()
    BENDING   = auto()
    HOLD      = auto()


# ---------------------------------------------------------------------------
# Per-rep record
# ---------------------------------------------------------------------------

@dataclass
class RepRecord:
    rep_number:      int
    peak_flex_angle: float
    is_correct:      bool
    issues:          List[str] = field(default_factory=list)
    duration:        float = 0.0          # seconds


# ---------------------------------------------------------------------------
# Session metrics
# ---------------------------------------------------------------------------

@dataclass
class KneeSessionMetrics:
    exercise_name:  str = "Knee Flexion / Extension"
    total_reps:     int = 0
    correct_reps:   int = 0
    accuracy:       float = 0.0
    avg_angle:      float = 0.0
    session_start:  float = field(default_factory=time.monotonic)
    session_end:    Optional[float] = None
    rep_records:    List[RepRecord] = field(default_factory=list)
    angle_samples:  List[float] = field(default_factory=list)
    current_accuracy: float = 0.0

    @property
    def duration(self) -> float:
        end = self.session_end or time.monotonic()
        return end - self.session_start

    def update_accuracy(self):
        if self.total_reps > 0:
            self.accuracy = (self.correct_reps / self.total_reps) * 100.0
        else:
            self.accuracy = 0.0
            
        if self.angle_samples:
            self.avg_angle = float(np.mean(self.angle_samples))


# ---------------------------------------------------------------------------
# KneeExercise
# ---------------------------------------------------------------------------

class KneeExercise:
    """
    Tracks a knee flexion/extension session.

    Call ``update(landmarks, frame_h)`` every frame.
    Retrieve feedback via ``feedback_msg``, ``status``, ``voice_msg``.
    """

    SIDE_PREFERENCE = ["LEFT", "RIGHT"]   # try left first; fall back to right

    def __init__(self, target_reps: int = 10, voice_enabled: bool = True):
        self.target_reps    = target_reps
        self.cfg            = KneeConfig()
        self.metrics        = KneeSessionMetrics()

        self._state             = KneeState.IDLE
        self._smoother          = AngleSmoother(alpha=0.40)
        self._velocity_tracker  = AngleVelocityTracker(history_len=6)

        self._current_angle     = 0.0
        self._peak_flex_angle   = 180.0   # smallest angle seen this rep
        self._hold_counter      = 0
        self._rep_is_valid      = True    # flag for current rep

        # Posture baseline (captured in first few frames of IDLE)
        self._hip_baseline_y: Optional[float] = None
        self._shoulder_baseline_y: Optional[float] = None
        self._calibration_frames = 0

        # Output state (read by main.py each frame)
        self.feedback_msg: str = "Stand sideways to camera"
        self.status:       str = FeedbackStatus.NEUTRAL
        self.voice_msg:    Optional[str] = None
        self.voice_priority: bool = False

        # Which side's joints to use
        self._active_side: Optional[str] = None

        # Joint names (set once side is detected)
        self.active_joints: Tuple[str, str, str] = ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE")

        logger.info("KneeExercise initialised — target reps: %d", target_reps)

    # ------------------------------------------------------------------
    # Properties for HUD (compatibility with main.py)
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
        labels = {
            KneeState.IDLE:      "READY",
            KneeState.EXTENDING: "EXTENDING",
            KneeState.BENDING:   "BENDING",
            KneeState.HOLD:      "HOLD",
        }
        return labels.get(self._state, "")

    @property
    def progress(self) -> float:
        """0→1 showing how far into the flex phase the patient is."""
        if self._state in (KneeState.IDLE, KneeState.EXTENDING):
            return 0.0
        total_range = self.cfg.EXTEND_THRESHOLD - self.cfg.FLEX_TARGET_ANGLE
        done = self.cfg.EXTEND_THRESHOLD - self._current_angle
        return float(np.clip(done / max(total_range, 1), 0, 1))

    # ------------------------------------------------------------------
    # Main update — call every frame
    # ------------------------------------------------------------------

    def update(
        self,
        landmarks: Dict[str, LandmarkPoint],
        frame_h: int = 720,
    ) -> None:
        """
        Process one frame of landmarks.
        After calling, read ``feedback_msg``, ``status``, ``voice_msg``.
        """
        self.voice_msg      = None
        self.voice_priority = False

        side = self._pick_side(landmarks)
        if side is None:
            self.feedback_msg = "Stand sideways to the camera"
            self.status       = FeedbackStatus.NEUTRAL
            return

        hip_name    = f"{side}_HIP"
        knee_name   = f"{side}_KNEE"
        ankle_name  = f"{side}_ANKLE"
        self.active_joints = (hip_name, knee_name, ankle_name)

        hip    = landmarks[hip_name]
        knee   = landmarks[knee_name]
        ankle  = landmarks[ankle_name]

        raw_angle = knee_angle(hip, knee, ankle)
        smooth    = self._smoother.update(raw_angle)
        velocity  = self._velocity_tracker.update(smooth)
        self._current_angle = smooth
        self.metrics.angle_samples.append(smooth)

        self._calibrate_baseline(landmarks, side, frame_h)

        if self._check_unsafe(smooth):
            self.metrics.update_accuracy()
            return

        issues = self._check_posture(landmarks, side, velocity)
        self._run_state_machine(smooth, issues)
        self.metrics.update_accuracy()

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _run_state_machine(self, angle: float, issues: List[str]):
        prev_state = self._state

        if self._state == KneeState.IDLE:
            if angle >= self.cfg.EXTEND_THRESHOLD:
                self._state = KneeState.EXTENDING
                self._reset_rep()
                self.feedback_msg = "Start bending your knee"
                self.status       = FeedbackStatus.NEUTRAL

        elif self._state == KneeState.EXTENDING:
            if angle < self.cfg.EXTEND_THRESHOLD:
                self._state = KneeState.BENDING
            self.feedback_msg = "Bend your knee slowly"
            self.status       = FeedbackStatus.NEUTRAL

        elif self._state == KneeState.BENDING:
            if angle < self._peak_flex_angle:
                self._peak_flex_angle = angle

            if issues:
                self._rep_is_valid = False
                self.status       = FeedbackStatus.WARNING
                self.feedback_msg = issues[0]
                self.metrics.current_accuracy = max(0.0, self.metrics.current_accuracy - 1.5)
                if len(self.metrics.angle_samples) % 30 == 1:
                    self.voice_msg = issues[0]
            else:
                self.status       = FeedbackStatus.CORRECT
                self.feedback_msg = "Good movement"
                self.metrics.current_accuracy = min(100.0, self.metrics.current_accuracy + 0.5)

            if angle <= self.cfg.FLEX_TARGET_ANGLE:
                self._hold_counter += 1
                if self._hold_counter >= self.cfg.HOLD_FRAMES_REQUIRED:
                    self._state = KneeState.HOLD
            else:
                self._hold_counter = 0

            if angle > self.cfg.FLEX_TARGET_ANGLE + 30:
                self.feedback_msg = "Bend your knee more"
                self.status       = FeedbackStatus.WARNING

        elif self._state == KneeState.HOLD:
            self.feedback_msg = "Now extend your leg"
            self.status       = FeedbackStatus.CORRECT

            if angle >= self.cfg.EXTEND_THRESHOLD:
                self._complete_rep(issues)
                self._state = KneeState.EXTENDING
                self._reset_rep()

        if self._state != prev_state:
            logger.debug("Knee state: %s → %s  angle=%.1f",
                         prev_state.name, self._state.name, angle)

    # ------------------------------------------------------------------
    # Rep management
    # ------------------------------------------------------------------

    def _reset_rep(self):
        self._peak_flex_angle = 180.0
        self._hold_counter    = 0
        self._rep_is_valid    = True

    def _complete_rep(self, issues: List[str]):
        is_correct = self._rep_is_valid and len(issues) == 0
        self.metrics.total_reps += 1
        if is_correct:
            self.metrics.correct_reps += 1

        rec = RepRecord(
            rep_number      = self.metrics.total_reps,
            peak_flex_angle = self._peak_flex_angle,
            is_correct      = is_correct,
            issues          = list(issues),
        )
        self.metrics.rep_records.append(rec)

        if is_correct:
            self.feedback_msg = f"Rep {self.metrics.total_reps} complete! Great job"
            self.voice_msg    = VoiceMessages.KNEE_REP_COMPLETE
        else:
            self.feedback_msg = f"Rep {self.metrics.total_reps} done — fix your form"
            self.voice_msg    = VoiceMessages.KNEE_GOOD_MOVEMENT

        logger.info("Knee rep %d completed — correct=%s peak_flex=%.1f°",
                    self.metrics.total_reps, is_correct, self._peak_flex_angle)

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------

    def _check_unsafe(self, angle: float) -> bool:
        if angle < self.cfg.UNSAFE_MIN_ANGLE or angle > self.cfg.UNSAFE_MAX_ANGLE:
            self.feedback_msg   = "⛔ UNSAFE MOVEMENT — STOP"
            self.status         = FeedbackStatus.ERROR
            self.voice_msg      = VoiceMessages.KNEE_ANGLE_UNSAFE
            self.voice_priority = True
            self._rep_is_valid  = False
            return True
        return False

    # ------------------------------------------------------------------
    # Posture checks (IMPROVED: uses abs for lateral lean)
    # ------------------------------------------------------------------

    def _check_posture(
        self,
        landmarks: Dict[str, LandmarkPoint],
        side: str,
        velocity: float,
    ) -> List[str]:
        issues: List[str] = []

        hip_name       = f"{side}_HIP"
        shoulder_name  = f"{side}_SHOULDER"
        knee_name      = f"{side}_KNEE"

        hip      = landmarks.get(hip_name)
        shoulder = landmarks.get(shoulder_name)

        # 1. Lateral lean (check absolute deviation to capture both left/right)
        if hip and shoulder:
            lean = lateral_lean_angle(shoulder, hip)
            if abs(lean) > self.cfg.MAX_LATERAL_LEAN:
                issues.append("Keep your back straight")

        # 2. Hip drift
        if hip and self._hip_baseline_y is not None:
            drift = abs(hip.y - self._hip_baseline_y)
            if drift > self.cfg.HIP_DRIFT_THRESHOLD:
                issues.append("Do not move your hip")

        # 3. Speed check
        if velocity > self.cfg.MAX_VELOCITY:
            issues.append("Slow down")
            if not issues:   # voice only if no other issues
                self.voice_msg = VoiceMessages.TOO_FAST

        return issues

    # ------------------------------------------------------------------
    # Side detection
    # ------------------------------------------------------------------

    def _pick_side(self, landmarks: Dict[str, LandmarkPoint]) -> Optional[str]:
        """Return 'LEFT' or 'RIGHT' — whichever side has all 3 joints visible."""
        if self._active_side:
            needed = [f"{self._active_side}_{j}" for j in ("HIP", "KNEE", "ANKLE")]
            if all(landmarks.get(n) and landmarks[n].visibility > 0.5 for n in needed):
                return self._active_side

        for side in self.SIDE_PREFERENCE:
            needed = [f"{side}_{j}" for j in ("HIP", "KNEE", "ANKLE")]
            if all(landmarks.get(n) and landmarks[n].visibility > 0.5 for n in needed):
                self._active_side = side
                return side
        return None

    # ------------------------------------------------------------------
    # Baseline calibration
    # ------------------------------------------------------------------

    def _calibrate_baseline(
        self,
        landmarks: Dict[str, LandmarkPoint],
        side: str,
        frame_h: int,
    ):
        if self._calibration_frames >= 15:
            return
        hip_name       = f"{side}_HIP"
        shoulder_name  = f"{side}_SHOULDER"
        hip      = landmarks.get(hip_name)
        shoulder = landmarks.get(shoulder_name)
        if hip and shoulder:
            alpha = 1.0 / (self._calibration_frames + 1)
            if self._hip_baseline_y is None:
                self._hip_baseline_y      = hip.y
                self._shoulder_baseline_y = shoulder.y
            else:
                self._hip_baseline_y      = (1 - alpha) * self._hip_baseline_y + alpha * hip.y
                self._shoulder_baseline_y = (1 - alpha) * self._shoulder_baseline_y + alpha * shoulder.y
            self._calibration_frames += 1

    # ------------------------------------------------------------------
    # HUD helpers
    # ------------------------------------------------------------------

    @staticmethod
    def safe_angle_range() -> Tuple[float, float]:
        return KneeConfig.FLEX_TARGET_ANGLE, KneeConfig.EXTEND_THRESHOLD

    def get_session_summary(self) -> Dict:
        self.metrics.session_end = time.monotonic()
        self.metrics.update_accuracy()
        return {
            "exercise":    self.metrics.exercise_name,
            "reps":        self.metrics.total_reps,
            "correct_reps": self.metrics.correct_reps,
            "accuracy":    round(self.metrics.accuracy, 1),
            "avg_angle":   round(self.metrics.avg_angle, 1),
            "duration":    round(self.metrics.duration, 1),
        }