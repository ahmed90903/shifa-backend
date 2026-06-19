"""
voice_feedback.py
-----------------
Non-blocking text-to-speech feedback using pyttsx3.
Runs TTS in a dedicated daemon thread so the main loop never stalls.
A cooldown timer prevents the same message from being repeated too rapidly.
"""

import time
import threading
import queue
import logging 

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pyttsx3; degrade gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    logger.warning(
        "pyttsx3 not found.  Voice feedback disabled.  "
        "Install with:  pip install pyttsx3"
    )


# ---------------------------------------------------------------------------
# Pre-defined voice messages (keep them short for real-time rehab)
# ---------------------------------------------------------------------------

class VoiceMessages:
    """Catalogue of all TTS strings used by the system."""

    # ---- English defaults ----
    EXERCISE_START          = "Exercise started. Begin your movement."
    EXERCISE_COMPLETE       = "Great session! Exercise complete."
    SESSION_SUMMARY         = "Session complete. Well done!"
    POSE_NOT_DETECTED       = "Please step in front of the camera."

    UNSAFE_MOVEMENT         = "Stop. Unsafe movement detected."
    SLOW_DOWN               = "Slow down. Control your movement."
    TOO_FAST                = "You are moving too fast. Please slow down."

    KNEE_BEND_MORE          = "Please bend your knee more."
    KNEE_EXTEND_FULLY       = "Extend your leg fully."
    KNEE_HIP_STABLE         = "Keep your hip stable. Do not sway."
    KNEE_BACK_STRAIGHT      = "Keep your back straight."
    KNEE_GOOD_MOVEMENT      = "Good movement. Keep it up."
    KNEE_REP_COMPLETE       = "Repetition complete. Good job."
    KNEE_ANGLE_UNSAFE       = "Stop. Your knee angle is unsafe."

    ELBOW_BEND_MORE         = "Bend your elbow fully."
    ELBOW_EXTEND_FULLY      = "Extend your arm fully."
    ELBOW_KEEP_STABLE       = "Keep your elbow fixed."
    ELBOW_SHOULDER_STABLE   = "Do not raise your shoulder."
    ELBOW_WRIST_PATH        = "Perform the movement correctly."
    ELBOW_GOOD_FORM         = "Good form. Keep going."
    ELBOW_REP_COMPLETE      = "Repetition complete. Well done."
    ELBOW_ANGLE_UNSAFE      = "Stop. Your elbow angle is unsafe."

    # Default fallback (when key not found)
    DEFAULT                 = "Keep going."

    # Arabic translations
    _TRANSLATIONS = {
        "ar": {
            "EXERCISE_START": "بدء التمرين. ابدأ حركتك.",
            "EXERCISE_COMPLETE": "جلسة رائعة! اكتمل التمرين.",
            "SESSION_SUMMARY": "اكتملت الجلسة. أحسنت!",
            "POSE_NOT_DETECTED": "يرجى الوقوف أمام الكاميرا.",
            "UNSAFE_MOVEMENT": "توقف. تم الكشف عن حركة غير آمنة.",
            "SLOW_DOWN": "أبطئ. تحكم في حركتك.",
            "TOO_FAST": "أنت تتحرك بسرعة كبيرة. يرجى الإبطاء.",
            "KNEE_BEND_MORE": "يرجى ثني ركبتك أكثر.",
            "KNEE_EXTEND_FULLY": "افرد ساقك بالكامل.",
            "KNEE_HIP_STABLE": "حافظ على استقرار الورك. لا تتمايل.",
            "KNEE_BACK_STRAIGHT": "حافظ على استقامة ظهرك.",
            "KNEE_GOOD_MOVEMENT": "حركة جيدة. استمر.",
            "KNEE_REP_COMPLETE": "اكتمل التكرار. عمل جيد.",
            "KNEE_ANGLE_UNSAFE": "توقف. زاوية ركبتك غير آمنة.",
            "ELBOW_BEND_MORE": "اثنِ كوعك بالكامل.",
            "ELBOW_EXTEND_FULLY": "افرد ذراعك بالكامل.",
            "ELBOW_KEEP_STABLE": "حافظ على ثبات كوعك.",
            "ELBOW_SHOULDER_STABLE": "لا ترفع كتفك.",
            "ELBOW_WRIST_PATH": "قم بأداء الحركة بشكل صحيح.",
            "ELBOW_GOOD_FORM": "أداء جيد. استمر.",
            "ELBOW_REP_COMPLETE": "اكتمل التكرار. أحسنت.",
            "ELBOW_ANGLE_UNSAFE": "توقف. زاوية كوعك غير آمنة.",
            "DEFAULT": "استمر."
        }
    }

    @classmethod
    def get_messages(cls, lang: str = "en") -> dict:
        """Return the full message catalogue for the requested language."""
        if lang == "ar":
            return cls._TRANSLATIONS["ar"]
        # Build English dictionary from class attributes (uppercase strings)
        return {
            k: getattr(cls, k)
            for k in dir(cls)
            if k.isupper() and isinstance(getattr(cls, k), str)
        }


# ---------------------------------------------------------------------------
# VoiceFeedback engine
# ---------------------------------------------------------------------------

class VoiceFeedback:
    """
    Thread-safe, non-blocking TTS engine.

    * A daemon thread owns the pyttsx3 engine.
    * `speak()` enqueues messages; returns immediately.
    * Cooldown prevents rapid repeats of the same message.
    * Priority messages bypass cooldown (safety alerts).
    """

    def __init__(self, enabled: bool = False, rate: int = 150, volume: float = 1.0):
        self.enabled = enabled and _TTS_AVAILABLE
        self.cooldown_sec = 4.0
        self._queue: queue.Queue = queue.Queue(maxsize=3)
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()
        self._running = False
        self._shutdown_event = threading.Event()   # improved shutdown signal

        if self.enabled:
            self._thread = threading.Thread(
                target=self._worker,
                args=(rate, volume),
                daemon=True,
                name="VoiceFeedbackThread",
            )
            self._thread.start()
            self._running = True
            logger.info("VoiceFeedback engine started (pyttsx3).")
        else:
            logger.info("VoiceFeedback engine disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, message: str, priority: bool = False) -> bool:
        """Enqueue a message. Returns True if enqueued, False if suppressed."""
        if not self.enabled or not message:
            return False

        now = time.monotonic()
        with self._lock:
            last = self._last_spoken.get(message, 0.0)
            if not priority and (now - last) < self.cooldown_sec:
                return False
            self._last_spoken[message] = now

        # Drop oldest if queue full to stay real-time
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(message)
            except queue.Empty:
                pass
        return True

    def get_message(self, key: str, lang: str = "en") -> str:
        """Retrieve a localized message by key."""
        messages = VoiceMessages.get_messages(lang)
        return messages.get(key, messages.get("DEFAULT", ""))

    def speak_priority(self, message: str) -> bool:
        """Shorthand for priority speech (safety alerts)."""
        return self.speak(message, priority=True)

    def clear_queue(self):
        """Drain all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Stop the background thread gracefully."""
        self._running = False
        self._shutdown_event.set()       # Signal the worker to stop waiting
        if self.enabled:
            # Give the thread a moment to exit, then ignore
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
            # Final cleanup
            self.clear_queue()

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _worker(self, rate: int, volume: float):
        """Daemon thread that processes the TTS queue."""
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)

            # Prefer a clear English voice
            voices = engine.getProperty("voices")
            for v in voices:
                if "english" in v.name.lower() or "en" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception as exc:
            logger.error("pyttsx3 init failed: %s", exc)
            return

        try:
            while self._running:
                try:
                    # Wait for a message or shutdown event
                    msg = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._shutdown_event.is_set():
                        break
                    continue

                if msg is None:          # sentinel (alternative)
                    break

                try:
                    engine.say(msg)
                    engine.runAndWait()
                except Exception as exc:
                    logger.warning("TTS error: %s", exc)

            # Process any remaining items before shutdown? Not needed.
        finally:
            if engine:
                try:
                    engine.stop()
                except Exception:
                    pass
            logger.info("VoiceFeedback thread stopped.")