"""
feedback_system.py
------------------
All OpenCV drawing primitives for the physiotherapy HUD.

Responsibilities
----------------
* Render a semi-transparent info panel (reps, accuracy, angle, state).
* Draw per-joint angle arcs with colour-coded status.
* Show feedback messages with colour-coded banners.
* Draw a progress bar for range-of-motion completion.
* Render a session summary overlay at the end.

Colour convention
-----------------
GREEN  (#00C853)  →  movement is correct / within target range
YELLOW (#FFD600)  →  warning / approaching limit
RED    (#D50000)  →  incorrect / unsafe movement
CYAN   (#00BCD4)  →  neutral info / angle readout
WHITE             →  general text
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from .pose_detector import LandmarkPoint


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
class Color:
    GREEN       = (0,   200, 83)
    YELLOW      = (0,   214, 255)
    RED         = (0,   0,   213)
    CYAN        = (211, 188, 0)
    WHITE       = (255, 255, 255)
    BLACK       = (0,   0,   0)
    DARK_PANEL  = (20,  20,  20)
    LIGHT_GRAY  = (180, 180, 180)
    ORANGE      = (0,   140, 255)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------
class FeedbackStatus:
    CORRECT  = "correct"
    WARNING  = "warning"
    ERROR    = "error"
    NEUTRAL  = "neutral"


STATUS_COLOR = {
    FeedbackStatus.CORRECT:  Color.GREEN,
    FeedbackStatus.WARNING:  Color.YELLOW,
    FeedbackStatus.ERROR:    Color.RED,
    FeedbackStatus.NEUTRAL:  Color.CYAN,
}

STATUS_ICON = {
    FeedbackStatus.CORRECT:  "✓",
    FeedbackStatus.WARNING:  "!",
    FeedbackStatus.ERROR:    "✗",
    FeedbackStatus.NEUTRAL:  "i",
}


# ---------------------------------------------------------------------------
# FeedbackSystem
# ---------------------------------------------------------------------------

@dataclass
class HUDData:
    """Everything the HUD needs to render one frame."""
    exercise_name:   str
    current_angle:   float
    rep_count:       int
    target_reps:     int
    accuracy:        float              # 0-100
    status:          str                # FeedbackStatus constant
    phase_label:     str                # "BENDING", "EXTENDING", "HOLD", …
    feedback_msg:    str                # short visual message
    progress:        float              # 0.0 – 1.0  (ROM completion this rep)
    session_time:    float              # elapsed seconds
    angle_min:       float              # safe range lower bound
    angle_max:       float              # safe range upper bound


class FeedbackSystem:
    """
    Stateless renderer — every method takes a frame + data and returns the
    annotated frame. No internal state except font metrics.
    """

    def __init__(self, frame_w: int = 1280, frame_h: int = 720):
        self.fw = frame_w
        self.fh = frame_h

        # Font constants
        self.FONT         = cv2.FONT_HERSHEY_DUPLEX
        self.FONT_BOLD    = cv2.FONT_HERSHEY_TRIPLEX
        self.FONT_SMALL   = cv2.FONT_HERSHEY_SIMPLEX

        self._panel_alpha = 0.72       # transparency of info panels

    # ------------------------------------------------------------------
    # Main entry point — call once per frame
    # ------------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        hud: HUDData,
        landmarks: Optional[Dict[str, LandmarkPoint]] = None,
        exercise_joints: Optional[Tuple[str, str, str]] = None,
    ) -> np.ndarray:
        """
        Only draw the angle arc. The textual feedback is handled by the frontend UI.
        """
        overlay = frame.copy()

        if landmarks and exercise_joints:
            self._draw_angle_arc(overlay, landmarks, exercise_joints, hud)

        # Blend overlay → frame
        cv2.addWeighted(overlay, self._panel_alpha, frame, 1 - self._panel_alpha, 0, frame)

        return frame

    # ------------------------------------------------------------------
    # Session summary overlay (shown after exercise ends)
    # ------------------------------------------------------------------

    def render_summary(
        self,
        frame: np.ndarray,
        exercise_name: str,
        reps: int,
        accuracy: float,
        avg_angle: float,
        duration: float,
    ) -> np.ndarray:
        """Full-screen semi-transparent summary card."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Dark overlay
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Card
        card_x, card_y = w // 2 - 320, h // 2 - 230
        card_w, card_h = 640, 460
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      (35, 35, 35), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h),
                      Color.CYAN, 2)

        # Header
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 70),
                      Color.CYAN, -1)
        self._put_text(frame, "SESSION COMPLETE", (card_x + 140, card_y + 48),
                       self.FONT_BOLD, 1.0, Color.BLACK, 2)

        # Metrics
        metrics = [
            ("Exercise",    exercise_name),
            ("Repetitions", f"{reps}"),
            ("Accuracy",    f"{accuracy:.1f}%"),
            ("Avg Angle",   f"{avg_angle:.1f}°"),
            ("Duration",    f"{int(duration // 60):02d}:{int(duration % 60):02d}"),
        ]
        row_h = 68
        for i, (label, value) in enumerate(metrics):
            y = card_y + 100 + i * row_h
            # alternating row tint
            if i % 2 == 0:
                cv2.rectangle(frame, (card_x + 4, y - 22),
                              (card_x + card_w - 4, y + row_h - 24),
                              (45, 45, 45), -1)
            self._put_text(frame, label + ":", (card_x + 24, y + 12),
                           self.FONT_SMALL, 0.75, Color.LIGHT_GRAY, 1)
            val_color = Color.GREEN if label == "Accuracy" and accuracy >= 80 else \
                        Color.YELLOW if label == "Accuracy" and accuracy >= 60 else \
                        Color.RED if label == "Accuracy" else Color.WHITE
            self._put_text(frame, value, (card_x + 280, y + 12),
                           self.FONT_BOLD, 0.85, val_color, 2)

        # Footer hint
        self._put_text(frame, "Press  Q  to quit  |  R  to restart",
                       (card_x + 90, card_y + card_h - 24),
                       self.FONT_SMALL, 0.6, Color.LIGHT_GRAY, 1)

        return frame

    # ------------------------------------------------------------------
    # "No pose" placeholder
    # ------------------------------------------------------------------

    def render_no_pose(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        self._put_text(frame, "⚠  No pose detected — please step in front of the camera",
                       (20, 52), self.FONT_SMALL, 0.72, Color.YELLOW, 2)
        return frame

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_top_bar(self, overlay: np.ndarray, hud: HUDData):
        cv2.rectangle(overlay, (0, 0), (self.fw, 72), Color.DARK_PANEL, -1)

    def _draw_top_bar_text(self, frame: np.ndarray, hud: HUDData):
        # Exercise name (left)
        self._put_text(frame, hud.exercise_name.upper(),
                       (20, 48), self.FONT_BOLD, 1.0, Color.CYAN, 2)

        # Timer (centre)
        elapsed = int(hud.session_time)
        timer_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        self._put_text(frame, timer_str,
                       (self.fw // 2 - 45, 48), self.FONT_BOLD, 1.0, Color.WHITE, 2)

        # Phase label (right)
        phase_color = STATUS_COLOR.get(hud.status, Color.WHITE)
        self._put_text(frame, hud.phase_label,
                       (self.fw - 280, 48), self.FONT, 0.85, phase_color, 2)

    # ---- Side panel ----

    def _draw_side_panel(self, overlay: np.ndarray, hud: HUDData):
        panel_x = self.fw - 220
        cv2.rectangle(overlay, (panel_x, 72), (self.fw, self.fh - 120),
                      Color.DARK_PANEL, -1)

    def _draw_side_panel_text(self, frame: np.ndarray, hud: HUDData):
        px = self.fw - 210
        base_y = 130

        # ---- Reps ----
        self._section_label(frame, "REPS", px, base_y)
        rep_color = Color.GREEN if hud.rep_count >= hud.target_reps else Color.WHITE
        self._put_text(frame, f"{hud.rep_count}/{hud.target_reps}",
                       (px, base_y + 50), self.FONT_BOLD, 1.3, rep_color, 2)

        # ---- Accuracy ----
        self._section_label(frame, "ACCURACY", px, base_y + 120)
        acc_color = (Color.GREEN   if hud.accuracy >= 80 else
                     Color.YELLOW  if hud.accuracy >= 60 else Color.RED)
        self._put_text(frame, f"{hud.accuracy:.0f}%",
                       (px, base_y + 170), self.FONT_BOLD, 1.3, acc_color, 2)
        # accuracy bar
        bar_w = 180
        filled = int(bar_w * hud.accuracy / 100)
        cv2.rectangle(frame, (px, base_y + 182), (px + bar_w, base_y + 198),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (px, base_y + 182), (px + filled, base_y + 198),
                      acc_color, -1)

        # ---- Current angle ----
        self._section_label(frame, "ANGLE", px, base_y + 240)
        angle_color = STATUS_COLOR.get(hud.status, Color.CYAN)
        self._put_text(frame, f"{hud.current_angle:.1f}°",
                       (px, base_y + 290), self.FONT_BOLD, 1.2, angle_color, 2)

        # Safe range indicator
        self._put_text(frame, f"Safe: {hud.angle_min:.0f}°–{hud.angle_max:.0f}°",
                       (px, base_y + 320), self.FONT_SMALL, 0.55, Color.LIGHT_GRAY, 1)

        # ---- Status icon ----
        self._section_label(frame, "STATUS", px, base_y + 370)
        s_color = STATUS_COLOR.get(hud.status, Color.WHITE)
        icon = STATUS_ICON.get(hud.status, "?")
        self._put_text(frame, icon,
                       (px, base_y + 420), self.FONT_BOLD, 1.6, s_color, 3)

    # ---- Feedback banner ----

    def _draw_feedback_banner(self, overlay: np.ndarray, hud: HUDData):
        banner_color = STATUS_COLOR.get(hud.status, Color.CYAN)
        dark = tuple(max(0, c - 170) for c in banner_color)
        cv2.rectangle(overlay, (0, self.fh - 118), (self.fw - 220, self.fh),
                      dark, -1)
        # left accent bar
        cv2.rectangle(overlay, (0, self.fh - 118), (8, self.fh),
                      banner_color, -1)

    def _draw_feedback_banner_text(self, frame: np.ndarray, hud: HUDData):
        banner_color = STATUS_COLOR.get(hud.status, Color.CYAN)
        icon = STATUS_ICON.get(hud.status, "")
        msg = f"{icon}  {hud.feedback_msg}" if icon else hud.feedback_msg

        # Scale font size to message length
        font_scale = 1.1 if len(hud.feedback_msg) < 28 else 0.85
        self._put_text(frame, msg,
                       (24, self.fh - 58), self.FONT_BOLD, font_scale, banner_color, 2)

        # Sub-hint
        hint = "Keep going!" if hud.status == FeedbackStatus.CORRECT else \
               "Adjust your position" if hud.status == FeedbackStatus.WARNING else \
               "Incorrect movement — stop and reset" if hud.status == FeedbackStatus.ERROR else ""
        if hint:
            self._put_text(frame, hint,
                           (24, self.fh - 22), self.FONT_SMALL, 0.62, Color.LIGHT_GRAY, 1)

    # ---- ROM progress bar ----

    def _draw_progress_bar(self, overlay: np.ndarray, hud: HUDData):
        """Horizontal bar below the top-bar showing range-of-motion completion."""
        bar_x, bar_y = 0, 72
        bar_h = 10
        bar_w = self.fw - 220
        cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (50, 50, 50), -1)
        filled = int(bar_w * np.clip(hud.progress, 0, 1))
        bar_color = STATUS_COLOR.get(hud.status, Color.CYAN)
        if filled > 0:
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                          bar_color, -1)

    # ---- Angle arc at joint vertex ----

    def _draw_angle_arc(
        self,
        frame: np.ndarray,
        landmarks: Dict[str, LandmarkPoint],
        joints: Tuple[str, str, str],
        hud: HUDData,
    ):
        """Draw an arc + angle label at the vertex joint."""
        prox_name, vtx_name, dist_name = joints
        if not all(n in landmarks for n in (prox_name, vtx_name, dist_name)):
            return

        prox = landmarks[prox_name]
        vtx  = landmarks[vtx_name]
        dist = landmarks[dist_name]

        if any(lm.visibility < 0.4 for lm in (prox, vtx, dist)):
            return

        cx, cy = int(vtx.x), int(vtx.y)
        arc_r = 38

        # Angles to proximal and distal
        angle_to_prox = np.degrees(np.arctan2(prox.y - vtx.y, prox.x - vtx.x))
        angle_to_dist = np.degrees(np.arctan2(dist.y - vtx.y, dist.x - vtx.x))

        start_a = min(angle_to_prox, angle_to_dist)
        end_a   = max(angle_to_prox, angle_to_dist)
        if end_a - start_a > 180:
            start_a, end_a = end_a, start_a + 360

        arc_color = STATUS_COLOR.get(hud.status, Color.CYAN)

        cv2.ellipse(frame, (cx, cy), (arc_r, arc_r), 0,
                    start_a, end_a, arc_color, 2, cv2.LINE_AA)

        # Radial lines to proximal / distal
        for lm in (prox, dist):
            dx = lm.x - vtx.x
            dy = lm.y - vtx.y
            norm = max(np.hypot(dx, dy), 1)
            ex = int(cx + dx / norm * arc_r)
            ey = int(cy + dy / norm * arc_r)
            cv2.line(frame, (cx, cy), (ex, ey), arc_color, 1, cv2.LINE_AA)

        # Angle text beside joint
        lbl_x = cx + 44
        lbl_y = cy + 10
        self._put_text(frame, f"{hud.current_angle:.0f}°",
                       (lbl_x, lbl_y), self.FONT_SMALL, 0.72, arc_color, 2)

    # ---- Utilities ----

    @staticmethod
    def _put_text(
        frame: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font,
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ):
        # Black shadow for readability
        cv2.putText(frame, text, (org[0] + 1, org[1] + 1),
                    font, scale, Color.BLACK, thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, text, org, font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _section_label(frame: np.ndarray, label: str, x: int, y: int):
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, Color.LIGHT_GRAY, 1, cv2.LINE_AA)
