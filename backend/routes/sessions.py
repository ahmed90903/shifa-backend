"""
backend/routes/sessions.py
POST  /api/sessions/start           – create session record
PATCH /api/sessions/{id}/complete   – save final results
GET   /api/sessions/history         – last 20 sessions
WS    /api/sessions/ws/{key}        – live AI analysis (JWT via ?token=)
"""
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.db_engine import get_db, AsyncSessionLocal
from backend.models.database import ExerciseSession, Exercise, Patient, User, session_exercise, AIAnalysisLog
from backend.models.schemas import (
    SessionStartRequest,
    SessionStartResponse,
    SessionCompleteRequest,
    SessionCompleteResponse,
    SessionHistoryItem,
)
from backend.services.auth_service import get_current_user, get_current_patient, decode_token
from backend.services.pose_analyzer import PoseAnalyzer

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/start", response_model=SessionStartResponse, status_code=201)
async def start_session(
    body: SessionStartRequest,
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    ex_result = await db.execute(select(Exercise).where(Exercise.id == body.exercise_id))
    exercise = ex_result.scalar_one_or_none()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")

    session = ExerciseSession(patient_id=patient.id, status="active")
    db.add(session)
    await db.flush()  # get session.id
    await db.execute(
        session_exercise.insert().values(session_id=session.id, exercise_id=exercise.id)
    )
    await db.commit()
    await db.refresh(session)

    return SessionStartResponse(
        session_id=session.id,
        exercise_key=exercise.landmark_config or "",
        exercise_name=exercise.name,
        target_reps=exercise.target_reps,
    )


@router.patch("/{session_id}/complete", response_model=SessionCompleteResponse)
async def complete_session(
    session_id: int,
    body: SessionCompleteRequest,
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ExerciseSession).where(
            ExerciseSession.id == session_id,
            ExerciseSession.patient_id == patient.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    now = datetime.utcnow()
    session.end_time = now
    session.status = "completed"
    session.total_reps = body.total_reps
    session.accuracy_score = body.accuracy_score
    session.max_angle_achieved = body.max_angle_achieved
    if body.duration_seconds:
        session.duration_seconds = body.duration_seconds
    elif session.start_time:
        session.duration_seconds = int((now - session.start_time).total_seconds())

    await db.commit()
    await db.refresh(session)
    return session


@router.get("/history", response_model=List[SessionHistoryItem])
async def session_history(
    patient: Patient = Depends(get_current_patient),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ExerciseSession)
        .where(ExerciseSession.patient_id == patient.id)
        .order_by(ExerciseSession.start_time.desc())
        .limit(20)
    )
    sessions = result.scalars().all()
    items: List[SessionHistoryItem] = []
    for s in sessions:
        ex_result = await db.execute(
            select(Exercise)
            .join(session_exercise, session_exercise.c.exercise_id == Exercise.id)
            .where(session_exercise.c.session_id == s.id)
        )
        ex_names = [e.name for e in ex_result.scalars()]
        items.append(
            SessionHistoryItem(
                id=s.id,
                start_time=s.start_time,
                end_time=s.end_time,
                duration_seconds=s.duration_seconds,
                total_reps=s.total_reps,
                accuracy_score=s.accuracy_score,
                max_angle_achieved=s.max_angle_achieved,
                status=s.status,
                exercise_names=ex_names,
            )
        )
    return items


# ── WebSocket ──────────────────────────────────────────────────────────────────
@router.websocket("/ws/{exercise_key}")
async def ws_pose(
    websocket: WebSocket,
    exercise_key: str,
    token: str = Query(..., description="JWT token"),
    session_id: int = Query(..., description="Active session ID"),
):
    """
    WebSocket endpoint for live pose analysis.
    Client must pass ?token=<jwt>&session_id=<id>
    Client sends: {"frame": "<base64-JPEG>"}
    Server sends: {joint_angle, is_correct_form, feedback_message, rep_count,
                   stage, max_angle_achieved, session_accuracy,
                   landmarks_visible, annotated_frame}
    """
    # Validate JWT
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    analyzer = PoseAnalyzer(exercise_key)

    try:
        async with AsyncSessionLocal() as db:
            # Verify session belongs to this user's patient
            p_result = await db.execute(
                select(Patient).where(Patient.user_id == user_id)
            )
            patient = p_result.scalar_one_or_none()
            if not patient:
                await websocket.close(code=4003)
                return

            s_result = await db.execute(
                select(ExerciseSession).where(
                    ExerciseSession.id == session_id,
                    ExerciseSession.patient_id == patient.id,
                )
            )
            if not s_result.scalar_one_or_none():
                await websocket.close(code=4004)
                return

        # Main analysis loop
        async with AsyncSessionLocal() as db:
            last_rep_count = -1
            last_feedback = ""
            frame_count = 0
            
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    break

                frame_b64 = data.get("frame", "")
                result = analyzer.analyze(frame_b64)
                await websocket.send_json(result)

                rep_count = result.get("rep_count", 0)
                feedback_message = result.get("feedback_message", "")
                frame_count += 1

                # Persist AI log on significant changes or every 20 frames to avoid high latency
                if rep_count != last_rep_count or feedback_message != last_feedback or frame_count % 20 == 0:
                    last_rep_count = rep_count
                    last_feedback = feedback_message
                    try:
                        log = AIAnalysisLog(
                            session_id=session_id,
                            joint_angle=result.get("joint_angle"),
                            is_correct_form=result.get("is_correct_form", False),
                            feedback_message=feedback_message,
                            rep_count=rep_count,
                        )
                        db.add(log)
                        await db.commit()
                    except Exception:
                        await db.rollback()

    except WebSocketDisconnect:
        pass
    finally:
        analyzer.close()
