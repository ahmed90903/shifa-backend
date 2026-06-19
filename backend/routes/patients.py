"""
backend/routes/patients.py
POST /api/patients/profile   – create profile
GET  /api/patients/profile   – get own profile
GET  /api/patients/dashboard – stats + recent sessions
"""
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.models.db_engine import get_db
from backend.models.database import User, Patient, ExerciseSession, Exercise, session_exercise
from backend.models.schemas import (
    PatientProfileCreate,
    PatientProfileResponse,
    DashboardStats,
    SessionSummary,
)
from backend.services.auth_service import get_current_user

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("/profile", response_model=PatientProfileResponse, status_code=201)
async def create_profile(
    body: PatientProfileCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Upsert pattern — allow re-submission
    existing = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    patient = existing.scalar_one_or_none()
    if patient:
        patient.full_name = body.full_name
        patient.date_of_birth = body.date_of_birth
        patient.medical_condition = body.medical_condition
        patient.therapist_notes = body.therapist_notes
    else:
        patient = Patient(user_id=current_user.id, **body.model_dump())
        db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return patient


@router.get("/profile", response_model=PatientProfileResponse)
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Profile not found")
    return patient


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Get patient
    p_result = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    patient = p_result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")

    # Aggregate stats
    stats = await db.execute(
        select(
            func.count(ExerciseSession.id).label("total_sessions"),
            func.coalesce(func.sum(ExerciseSession.total_reps), 0).label("total_reps"),
            func.coalesce(func.avg(ExerciseSession.accuracy_score), 0.0).label("avg_accuracy"),
        ).where(
            ExerciseSession.patient_id == patient.id,
            ExerciseSession.status == "completed",
        )
    )
    row = stats.one()

    # Streak: count consecutive days with at least one completed session
    today = datetime.utcnow().date()
    streak = 0
    check_date = today
    while True:
        day_result = await db.execute(
            select(func.count(ExerciseSession.id)).where(
                ExerciseSession.patient_id == patient.id,
                ExerciseSession.status == "completed",
                func.date(ExerciseSession.start_time) == check_date,
            )
        )
        count = day_result.scalar()
        if count and count > 0:
            streak += 1
            check_date -= timedelta(days=1)
        else:
            break
        if streak > 365:
            break

    # Recent 5 sessions
    recent_result = await db.execute(
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == patient.id)
        .order_by(ExerciseSession.start_time.desc())
        .limit(5)
    )
    recent_sessions: List[SessionSummary] = []
    for session in recent_result.scalars():
        ex_result = await db.execute(
            select(Exercise)
            .join(session_exercise, session_exercise.c.exercise_id == Exercise.id)
            .where(session_exercise.c.session_id == session.id)
        )
        ex_names = [e.name for e in ex_result.scalars()]
        recent_sessions.append(
            SessionSummary(
                id=session.id,
                start_time=session.start_time,
                end_time=session.end_time,
                duration_seconds=session.duration_seconds,
                total_reps=session.total_reps,
                accuracy_score=session.accuracy_score,
                status=session.status,
                exercise_names=ex_names,
            )
        )

    return DashboardStats(
        total_sessions=row.total_sessions or 0,
        total_reps=row.total_reps or 0,
        avg_accuracy=round(float(row.avg_accuracy or 0.0), 1),
        streak_days=streak,
        recent_sessions=recent_sessions,
    )
