import importlib

from backend.services.ai.voice_feedback import VoiceMessages, VoiceFeedback


def test_arabic_messages_exist_and_are_arabic():
    arabic = VoiceMessages.get_messages("ar")
    assert arabic["EXERCISE_START"].startswith("بدء") or any(
        ch in arabic["EXERCISE_START"] for ch in "أبجددهوزحطيكلمنسعفصقرشتثخذضظغ"
    )
    assert arabic["KNEE_REP_COMPLETE"].startswith("ا") or any(
        ch in arabic["KNEE_REP_COMPLETE"] for ch in "أبجددهوزحطيكلمنسعفصقرشتثخذضظغ"
    )
    assert arabic["ELBOW_REP_COMPLETE"].startswith("ا") or any(
        ch in arabic["ELBOW_REP_COMPLETE"] for ch in "أبجددهوزحطيكلمنسعفصقرشتثخذضظغ"
    )


def test_voice_feedback_supports_language_selection():
    feedback = VoiceFeedback(enabled=False)
    message = feedback.get_message("EXERCISE_START", "ar")
    assert message is not None
    assert "بدء" in message or any(ch in message for ch in "أبجددهوزحطيكلمنسعفصقرشتثخذضظغ")
