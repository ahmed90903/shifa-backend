"""
backend/models/schemas.py
Pydantic v2 request / response schemas.
"""
from __future__ import annotations
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


# ── Auth ───────────────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str

class VerifyEmailRequest(BaseModel):
    email: EmailStr
    otp: str

class ResendOTPRequest(BaseModel):
    email: EmailStr

class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class VerifyResetCodeRequest(BaseModel):
    email: EmailStr
    code: str

class ResetPasswordRequest(BaseModel):
    reset_token: str
    new_password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str
    has_profile: bool = False


# ── Patient / Profile ──────────────────────────────────────────────────────────
class PatientProfileCreate(BaseModel):
    full_name: str
    date_of_birth: Optional[date] = None
    medical_condition: Optional[str] = None
    therapist_notes: Optional[str] = None


class PatientProfileResponse(BaseModel):
    id: int
    user_id: int
    full_name: str
    date_of_birth: Optional[date] = None
    medical_condition: Optional[str] = None
    therapist_notes: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Dashboard ──────────────────────────────────────────────────────────────────
class SessionSummary(BaseModel):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    total_reps: int
    accuracy_score: Optional[float] = None
    status: str
    exercise_names: List[str] = Field(default_factory=list)   # أفضل من [] المباشر

    model_config = {"from_attributes": True}


class DashboardStats(BaseModel):
    total_sessions: int
    total_reps: int
    avg_accuracy: float
    streak_days: int
    recent_sessions: List[SessionSummary]      # الإشارة آمنة بسبب `from __future__`


# ── Exercises ──────────────────────────────────────────────────────────────────
class ExerciseResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    target_body_part: Optional[str] = None
    video_url: Optional[str] = None
    landmark_config: Optional[str] = None
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    target_reps: int
    tutorial_video: Optional[str] = None
    instructions: Optional[str] = None
    safety_notes: Optional[str] = None
    difficulty_level: Optional[str] = None
    target_angle: Optional[str] = None

    model_config = {"from_attributes": True}


class TutorialResponse(BaseModel):
    exercise_name: str
    video_url: Optional[str] = None
    instructions: List[str] = Field(default_factory=list)    # أفضل
    safety_notes: Optional[str] = None
    target_angle: Optional[str] = None


# ── Sessions ───────────────────────────────────────────────────────────────────
class SessionStartRequest(BaseModel):
    exercise_id: int


class SessionStartResponse(BaseModel):
    session_id: int
    exercise_key: str
    exercise_name: str
    target_reps: int


class SessionCompleteRequest(BaseModel):
    total_reps: int
    accuracy_score: float
    max_angle_achieved: Optional[float] = None
    duration_seconds: Optional[int] = None


class SessionCompleteResponse(BaseModel):
    id: int
    status: str
    total_reps: int
    accuracy_score: Optional[float]
    duration_seconds: Optional[int]

    model_config = {"from_attributes": True}


class SessionHistoryItem(BaseModel):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    total_reps: int
    accuracy_score: Optional[float] = None
    max_angle_achieved: Optional[float] = None
    status: str
    exercise_names: List[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}