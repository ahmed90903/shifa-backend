"""
main.py
-------
Smart Physical Therapy System — AI Module Entry Point
... (الوصف كما في الأصل) ...
"""

import sys
import time
import argparse
import logging
import signal
import cv2
import numpy as np

# ---- Local modules (يمكن استخدام try/except للاستيراد النسبي) ----
try:
    from .pose_detector import PoseDetector
    from .knee_exercise import KneeExercise
    from .elbow_exercise import ElbowExercise
    from .feedback_system import FeedbackSystem, HUDData, FeedbackStatus
    from .voice_feedback import VoiceFeedback, VoiceMessages
except ImportError:
    from pose_detector import PoseDetector
    from knee_exercise import KneeExercise
    from elbow_exercise import ElbowExercise
    from feedback_system import FeedbackSystem, HUDData, FeedbackStatus
    from voice_feedback import VoiceFeedback, VoiceMessages

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("main")

TARGET_FPS     = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS
WINDOW_NAME    = "Smart Physical Therapy — AI Module"

KEY_QUIT    = {ord("q"), ord("Q"), 27}
KEY_KNEE    = {ord("k"), ord("K")}
KEY_ELBOW   = {ord("e"), ord("E")}
KEY_RESTART = {ord("r"), ord("R")}
KEY_PAUSE   = {ord(" ")}

# ---------------------------------------------------------------------------
def _draw_splash(frame: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.putText(frame, "SMART PHYSICAL THERAPY", (w//2-310, 100),
                cv2.FONT_HERSHEY_TRIPLEX, 1.4, (0,200,83), 2, cv2.LINE_AA)
    cv2.putText(frame, "AI Rehabilitation Module", (w//2-210, 148),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (200,200,200), 1, cv2.LINE_AA)
    cv2.line(frame, (80, 175), (w-80, 175), (60,60,60), 2)
    options = [
        ("[K]", "Knee Flexion / Extension", (0,200,83)),
        ("[E]", "Elbow Flexion / Extension", (0,200,83)),
        ("[Q]", "Quit", (0,80,200)),
    ]
    for i, (key, label, color) in enumerate(options):
        y = 240 + i*90
        cv2.putText(frame, key,   (120, y), cv2.FONT_HERSHEY_TRIPLEX, 1.3, color, 2, cv2.LINE_AA)
        cv2.putText(frame, label, (230, y), cv2.FONT_HERSHEY_DUPLEX, 0.95, (230,230,230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Camera ready  |  FPS {fps:.0f}", (80, h-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 1, cv2.LINE_AA)
    return frame

def _draw_pause(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w,h), (10,10,10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "⏸  PAUSED", (w//2-160, h//2),
                cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0,214,255), 3, cv2.LINE_AA)
    cv2.putText(frame, "Press SPACE to resume", (w//2-190, h//2+70),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (200,200,200), 1, cv2.LINE_AA)
    return frame

# ---------------------------------------------------------------------------
class PhysioApp:
    def __init__(self, exercise_choice="menu", target_reps=10, camera_index=0, tts_enabled=True):
        self.exercise_choice = exercise_choice.lower()
        self.target_reps = target_reps
        self.tts_enabled = tts_enabled

        logger.info("Initialising camera (index %d) …", camera_index)
        self.detector = PoseDetector(camera_index=camera_index)
        fw, fh = self.detector.get_frame_size()

        self.hud   = FeedbackSystem(frame_w=fw, frame_h=fh)
        self.voice = VoiceFeedback(enabled=tts_enabled)

        self.exercise = None
        self._paused  = False
        self._running = True

        self._fps_history = [TARGET_FPS] * 10
        self._last_frame_time = time.monotonic()
        signal.signal(signal.SIGINT, self._signal_handler)

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *self.detector.get_frame_size())

        if self.exercise_choice in ("knee", "elbow"):
            self._start_exercise(self.exercise_choice)

        while self._running:
            t_start = time.monotonic()
            frame, landmarks = self.detector.process_frame()
            if frame is None:
                logger.error("Camera read failed — exiting.")
                break

            key = cv2.waitKey(1) & 0xFF

            # --- Global keys ---
            if key in KEY_QUIT:
                break
            if key in KEY_KNEE:
                self._start_exercise("knee")
            if key in KEY_ELBOW:
                self._start_exercise("elbow")
            if key in KEY_RESTART and self.exercise:
                self._start_exercise(
                    "knee" if isinstance(self.exercise, KneeExercise) else "elbow"
                )
            if key in KEY_PAUSE:
                self._paused = not self._paused

            if self.exercise is None:
                fps = self._update_fps(t_start)
                frame = _draw_splash(frame, fps)
                cv2.imshow(WINDOW_NAME, frame)
                continue

            if self._paused:
                frame = _draw_pause(frame)
                cv2.imshow(WINDOW_NAME, frame)
                continue

            # --- Core processing ---
            if landmarks:
                frame = self._process_frame(frame, landmarks)
            else:
                frame = self.hud.render_no_pose(frame)

            # --- Session complete ---
            if self.exercise.session_complete:
                summary = self.exercise.get_session_summary()
                frame = self.hud.render_summary(
                    frame,
                    exercise_name=summary.get("exercise", "تمرين"),
                    reps=summary.get("reps", 0),
                    accuracy=summary.get("accuracy", 0.0),
                    avg_angle=summary.get("avg_angle", 0.0),
                    duration=summary.get("duration", 0.0),
                )
                cv2.imshow(WINDOW_NAME, frame)
                logger.info("Session complete: %s", summary)
                # انتظار إعادة التشغيل أو الخروج مع بقاء الشاشة مرئية
                while True:
                    k = cv2.waitKey(33) & 0xFF
                    if k in KEY_QUIT:
                        self._running = False
                        break
                    if k in KEY_RESTART:
                        self._start_exercise(
                            "knee" if isinstance(self.exercise, KneeExercise) else "elbow"
                        )
                        break
                    if k in KEY_KNEE:
                        self._start_exercise("knee")
                        break
                    if k in KEY_ELBOW:
                        self._start_exercise("elbow")
                        break
                continue

            cv2.imshow(WINDOW_NAME, frame)

            # --- حد السرعة ---
            elapsed = time.monotonic() - t_start
            sleep_t = FRAME_INTERVAL - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._shutdown()

    def _process_frame(self, frame, landmarks):
        # تحديث منطق التمرين
        self.exercise.update(landmarks, frame_h=frame.shape[0])

        # صوت
        if getattr(self.exercise, "voice_msg", None):
            self.voice.speak(
                self.exercise.voice_msg,
                priority=getattr(self.exercise, "voice_priority", False),
            )

        # النطاق الآمن للزاوية
        a_min, a_max = (0, 180)
        if hasattr(self.exercise, "safe_angle_range"):
            a_min, a_max = self.exercise.safe_angle_range()

        # الحصول على الخصائص بطريقة آمنة
        ex_name = getattr(self.exercise, "exercise_name", "تمرين")
        cur_angle = getattr(self.exercise, "current_angle", 0.0)
        rep_cnt   = getattr(self.exercise, "rep_count", 0)
        acc       = getattr(self.exercise, "accuracy", 0.0)
        status    = getattr(self.exercise, "status", FeedbackStatus.OK)
        phase_lbl = getattr(self.exercise, "phase_label", "")
        fb_msg    = getattr(self.exercise, "feedback_msg", "")
        progress  = getattr(self.exercise, "progress", 0.0)
        start_t   = getattr(self.exercise, "start_time", time.monotonic())

        hud_data = HUDData(
            exercise_name = ex_name,
            current_angle = cur_angle,
            rep_count     = rep_cnt,
            target_reps   = self.target_reps,
            accuracy      = acc,
            status        = status,
            phase_label   = phase_lbl,
            feedback_msg  = fb_msg,
            progress      = progress,
            session_time  = time.monotonic() - start_t,
            angle_min     = a_min,
            angle_max     = a_max,
        )

        # رسم الهيكل
        highlight = list(getattr(self.exercise, "active_joints", []))
        frame = self.detector.draw_skeleton(
            frame, landmarks,
            highlight_joints=highlight,
            joint_color=(180,180,180),
            line_color=(100,100,100),
        )

        # عرض HUD
        frame = self.hud.render(
            frame, hud_data,
            landmarks=landmarks,
            exercise_joints=getattr(self.exercise, "active_joints", []),
        )
        return frame

    def _start_exercise(self, choice):
        self.voice.clear_queue()
        if choice == "knee":
            self.exercise = KneeExercise(target_reps=self.target_reps)
            self.voice.speak(VoiceMessages.EXERCISE_START)
            logger.info("Started: Knee Flexion / Extension")
        elif choice == "elbow":
            self.exercise = ElbowExercise(target_reps=self.target_reps)
            self.voice.speak(VoiceMessages.EXERCISE_START)
            logger.info("Started: Elbow Flexion / Extension")
        else:
            logger.warning("Unknown exercise choice: %s", choice)

    def _update_fps(self, t_start):
        now = time.monotonic()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_history.append(1.0/dt)
            self._fps_history = self._fps_history[-10:]
        return float(np.mean(self._fps_history))

    def _shutdown(self):
        logger.info("Shutting down …")
        self.voice.shutdown()
        self.detector.release()
        cv2.destroyAllWindows()
        logger.info("Goodbye.")

    def _signal_handler(self, sig, _frame):
        logger.info("Interrupted — shutting down gracefully.")
        self._running = False

# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Physical Therapy — AI Rehabilitation Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                       # interactive menu
  python main.py --exercise knee       # start knee exercise directly
  python main.py --exercise elbow --reps 12
  python main.py --camera 1            # use secondary camera
  python main.py --no-voice            # disable TTS
        """
    )
    parser.add_argument("--exercise", "-x", choices=["knee","elbow","menu"], default="menu")
    parser.add_argument("--reps", "-r", type=int, default=10)
    parser.add_argument("--camera", "-c", type=int, default=0)
    parser.add_argument("--no-voice", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    app = PhysioApp(
        exercise_choice=args.exercise,
        target_reps=args.reps,
        camera_index=args.camera,
        tts_enabled=not args.no_voice,
    )
    app.run()

if __name__ == "__main__":
    main()