"""
angle_calculator.py
-------------------
Pure-math utilities for computing joint angles from 2-D / 3-D landmark
coordinates.  No OpenCV or MediaPipe imports — keeps the math testable
in isolation.
"""

import numpy as np
from typing import Tuple, Optional
from .pose_detector import LandmarkPoint


# ---------------------------------------------------------------------------
# Core angle function
# ---------------------------------------------------------------------------

def calculate_angle(
    a: LandmarkPoint,
    b: LandmarkPoint,
    c: LandmarkPoint,
    use_3d: bool = False,
) -> float:
    """
    Calculate the angle ∠ABC (vertex at B) in degrees.

    Parameters
    ----------
    a, b, c : LandmarkPoint
        Three joint landmarks; B is the vertex (e.g. knee, elbow).
    use_3d   : bool
        If True, include the Z component for a more accurate 3-D angle.
        Default False — 2-D is faster and sufficient for side-view rehab.

    Returns
    -------
    angle : float  [0, 180]
    """
    if use_3d:
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float64)
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float64)
    else:
        ba = np.array([a.x - b.x, a.y - b.y], dtype=np.float64)
        bc = np.array([c.x - b.x, c.y - b.y], dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0                          # degenerate — landmarks collapsed

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)   # guard floating-point drift
    return float(np.degrees(np.arccos(cos_angle)))


# ---------------------------------------------------------------------------
# Velocity / smoothing helpers
# ---------------------------------------------------------------------------

class AngleSmoother:
    """
    Exponential moving average over consecutive angle readings.
    Reduces jitter without adding latency beyond 1-2 frames.
    """

    def __init__(self, alpha: float = 0.35):
        """
        Parameters
        ----------
        alpha : float
            Smoothing factor ∈ (0, 1].  Higher → more responsive, less smooth.
        """
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, raw: float) -> float:
        if self._value is None:
            self._value = raw
        else:
            self._value = self.alpha * raw + (1 - self.alpha) * self._value
        return self._value

    def reset(self):
        self._value = None

    @property
    def value(self) -> Optional[float]:
        return self._value


class AngleVelocityTracker:
    """
    Tracks angular velocity (degrees / frame) and detects movement speed
    issues (too fast → risk of injury; too slow → stuck / loss of pose).
    """

    def __init__(self, history_len: int = 8):
        self._history = []
        self._max_len = history_len

    def update(self, angle: float) -> float:
        """Return smoothed velocity magnitude (deg/frame)."""
        self._history.append(angle)
        if len(self._history) > self._max_len:
            self._history.pop(0)
        if len(self._history) < 2:
            return 0.0
        deltas = [abs(self._history[i] - self._history[i - 1])
                  for i in range(1, len(self._history))]
        return float(np.mean(deltas))

    def reset(self):
        self._history.clear()

    @property
    def current_velocity(self) -> float:
        if len(self._history) < 2:
            return 0.0
        return abs(self._history[-1] - self._history[-2])


# ---------------------------------------------------------------------------
# Postural / stability helpers
# ---------------------------------------------------------------------------

def lateral_lean_angle(
    shoulder: LandmarkPoint,
    hip: LandmarkPoint,
) -> float:
    """
    Estimate trunk lateral lean in degrees from vertical.
    Used by knee exercise to detect excessive hip/trunk sway.

    Returns positive value for right-lean, negative for left-lean.
    0° = perfectly upright.
    """
    dx = shoulder.x - hip.x
    dy = hip.y - shoulder.y        # y increases downward in image space
    if abs(dy) < 1e-6:
        # Avoid division by zero; if dy ~0, person is horizontal → lean ~±90°
        return 90.0 if dx > 0 else -90.0 if dx < 0 else 0.0
    # Use dx without abs to preserve sign (right positive, left negative)
    return float(np.degrees(np.arctan2(dx, dy)))


def shoulder_elevation_delta(
    shoulder_start_y: float,
    shoulder_current_y: float,
    frame_height: int,
) -> float:
    """
    Compute normalised shoulder rise (0–1) relative to frame height.
    Used by elbow exercise to detect shoulder compensation.
    """
    return abs(shoulder_current_y - shoulder_start_y) / max(frame_height, 1)


def wrist_deviation(
    elbow: LandmarkPoint,
    wrist: LandmarkPoint,
    forearm_ref_angle: float,      # now in degrees
) -> float:
    """
    Measure how much the wrist deviates from the expected straight-arm path.

    Parameters
    ----------
    forearm_ref_angle : float
        The canonical forearm angle in **degrees** captured at the start of rep.

    Returns
    -------
    deviation : float   degrees off the expected path (0–180).
    """
    # Current forearm angle in degrees
    current_angle = float(np.degrees(
        np.arctan2(wrist.y - elbow.y, wrist.x - elbow.x)
    ))
    # Absolute smallest difference on the circle
    delta = abs(current_angle - forearm_ref_angle) % 360
    if delta > 180:
        delta = 360 - delta
    return delta


# ---------------------------------------------------------------------------
# Convenience wrappers used by exercise modules
# ---------------------------------------------------------------------------

def knee_angle(hip: LandmarkPoint, knee: LandmarkPoint, ankle: LandmarkPoint) -> float:
    """Angle at the knee joint (hip–knee–ankle)."""
    return calculate_angle(hip, knee, ankle)


def elbow_angle(
    shoulder: LandmarkPoint,
    elbow: LandmarkPoint,
    wrist: LandmarkPoint,
) -> float:
    """Angle at the elbow joint (shoulder–elbow–wrist)."""
    return calculate_angle(shoulder, elbow, wrist)