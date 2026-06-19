"""
backend/models/database.py
SQLAlchemy 2.0 async ORM models.
"""
from datetime import datetime, date
from sqlalchemy import (
    Boolean, Column, DateTime, Date, Float, ForeignKey,
    Integer, String, Text, Table,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Association table ──────────────────────────────────────────────────────────
session_exercise = Table(
    "session_exercise",
    Base.metadata,
    Column("session_id", ForeignKey("exercise_sessions.id"), primary_key=True),
    Column("exercise_id", ForeignKey("exercises.id"), primary_key=True),
)


# ── Users ──────────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    username: str = Column(String(64), unique=True, index=True, nullable=False)
    email: str = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password: str = Column(String(256), nullable=False)
    is_active: bool = Column(Boolean, default=True)
    is_verified: bool = Column(Boolean, default=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="user", uselist=False)


# ── Email Verifications ────────────────────────────────────────────────────────
class EmailVerification(Base):
    __tablename__ = "email_verifications"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    email: str = Column(String(128), index=True, nullable=False)
    otp_code: str = Column(String(256), nullable=False) # Hashed OTP
    expiration_time: datetime = Column(DateTime, nullable=False)
    verified_status: bool = Column(Boolean, default=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")

# ── Password Resets ────────────────────────────────────────────────────────────
class PasswordReset(Base):
    __tablename__ = "password_resets"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    email: str = Column(String(128), index=True, nullable=False)
    reset_code: str = Column(String(256), nullable=False) # Hashed OTP
    reset_token: str = Column(String(256), nullable=True, unique=True) # UUID for final reset step
    expires_at: datetime = Column(DateTime, nullable=False)
    is_used: bool = Column(Boolean, default=False)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")

# ── Patients ───────────────────────────────────────────────────────────────────
class Patient(Base):
    __tablename__ = "patients"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    full_name: str = Column(String(128), nullable=False)
    date_of_birth: date = Column(Date, nullable=True)
    medical_condition: str = Column(Text, nullable=True)
    therapist_notes: str = Column(Text, nullable=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="patient")
    sessions = relationship("ExerciseSession", back_populates="patient")


# ── Exercises ──────────────────────────────────────────────────────────────────
class Exercise(Base):
    __tablename__ = "exercises"

    id: int = Column(Integer, primary_key=True, index=True)
    name: str = Column(String(128), nullable=False)
    description: str = Column(Text, nullable=True)
    target_body_part: str = Column(String(64), nullable=True)
    video_url: str = Column(String(256), nullable=True)
    landmark_config: str = Column(String(64), nullable=True)   # e.g. "knee_extension"
    min_angle: float = Column(Float, nullable=True)
    max_angle: float = Column(Float, nullable=True)
    target_reps: int = Column(Integer, default=10)
    tutorial_video: str = Column(String(256), nullable=True)
    instructions: str = Column(Text, nullable=True)
    safety_notes: str = Column(Text, nullable=True)
    difficulty_level: str = Column(String(32), nullable=True)
    target_angle: str = Column(String(64), nullable=True)


# ── Exercise Sessions ──────────────────────────────────────────────────────────
class ExerciseSession(Base):
    __tablename__ = "exercise_sessions"

    id: int = Column(Integer, primary_key=True, index=True)
    patient_id: int = Column(Integer, ForeignKey("patients.id"), nullable=False)
    start_time: datetime = Column(DateTime, default=datetime.utcnow)
    end_time: datetime = Column(DateTime, nullable=True)
    duration_seconds: int = Column(Integer, nullable=True)
    total_reps: int = Column(Integer, default=0)
    accuracy_score: float = Column(Float, nullable=True)
    max_angle_achieved: float = Column(Float, nullable=True)
    status: str = Column(String(32), default="active")   # active | completed | cancelled

    patient = relationship("Patient", back_populates="sessions")
    exercises = relationship("Exercise", secondary=session_exercise)
    ai_logs = relationship("AIAnalysisLog", back_populates="session")


# ── AI Analysis Logs ───────────────────────────────────────────────────────────
class AIAnalysisLog(Base):
    __tablename__ = "ai_analysis_logs"

    id: int = Column(Integer, primary_key=True, index=True)
    session_id: int = Column(Integer, ForeignKey("exercise_sessions.id"), nullable=False)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow)
    joint_angle: float = Column(Float, nullable=True)
    is_correct_form: bool = Column(Boolean, default=False)
    feedback_message: str = Column(Text, nullable=True)
    rep_count: int = Column(Integer, default=0)

    session = relationship("ExerciseSession", back_populates="ai_logs")
