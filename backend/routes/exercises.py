"""
backend/routes/exercises.py
GET  /api/exercises/   – list all
POST /api/exercises/seed – seed default library
"""
from typing import List

import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.models.db_engine import get_db
from backend.models.database import Exercise
from backend.models.schemas import ExerciseResponse, TutorialResponse
from backend.services.auth_service import get_current_user

router = APIRouter(prefix="/api/exercises", tags=["exercises"])

_SEED_DATA = [
    {
        "name": "Knee Extension",
        "description": "Sit on a chair. Slowly lift your lower leg until it is straight, hold 2 seconds, then lower slowly.",
        "target_body_part": "Knee",
        "video_url": "/static/assets/knee_extension.mp4",
        "landmark_config": "knee_extension",
        "min_angle": 90.0,
        "max_angle": 160.0,
        "target_reps": 10,
        "tutorial_video": "/static/videos/knee_extension_tutorial.mp4",
        "instructions": json.dumps([
            "Stand or sit in the correct position.",
            "Keep your back straight.",
            "Slowly extend or bend your knee.",
            "Return to the starting position."
        ]),
        "safety_notes": "Do not exceed the recommended knee angle. Stop if you feel sharp pain.",
        "difficulty_level": "Beginner",
        "target_angle": "160 degrees"
    },
    {
        "name": "Bicep Curl",
        "description": "Stand upright. Curl your forearm toward your shoulder, keeping the upper arm still.",
        "target_body_part": "Elbow",
        "video_url": "/static/assets/bicep_curl.mp4",
        "landmark_config": "bicep_curl",
        "min_angle": 40.0,
        "max_angle": 160.0,
        "target_reps": 12,
        "tutorial_video": "/static/videos/elbow_flexion_tutorial.mp4",
        "instructions": json.dumps([
            "Keep shoulder stable.",
            "Slowly bend the elbow.",
            "Move until target angle is reached.",
            "Slowly return."
        ]),
        "safety_notes": "Avoid swinging your back. Keep movements controlled.",
        "difficulty_level": "Beginner",
        "target_angle": "40 degrees"
    },
]


@router.get("/", response_model=List[ExerciseResponse])
async def list_exercises(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Exercise))
    exercises = result.scalars().all()
    # Auto-seed on first request if empty
    if not exercises:
        await _do_seed(db)
        result = await db.execute(select(Exercise))
        exercises = result.scalars().all()
    return exercises


@router.post("/seed", status_code=201)
async def seed_exercises(db: AsyncSession = Depends(get_db)):
    count = await _do_seed(db)
    return {"message": f"Seeded {count} exercises"}


async def _do_seed(db: AsyncSession) -> int:
    existing = await db.execute(select(Exercise.landmark_config))
    existing_keys = {r for r in existing.scalars()}
    count = 0
    for data in _SEED_DATA:
        if data["landmark_config"] not in existing_keys:
            db.add(Exercise(**data))
            count += 1
    if count:
        await db.commit()
    return count

@router.get("/{exercise_id}/tutorial", response_model=TutorialResponse)
async def get_exercise_tutorial(exercise_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Exercise).where(Exercise.id == exercise_id))
    exercise = result.scalar_one_or_none()
    if not exercise:
        raise HTTPException(status_code=404, detail="Exercise not found")
        
    instructions_list = []
    if exercise.instructions:
        try:
            instructions_list = json.loads(exercise.instructions)
        except:
            instructions_list = [exercise.instructions]
            
    return TutorialResponse(
        exercise_name=exercise.name,
        video_url=exercise.tutorial_video or exercise.video_url,
        instructions=instructions_list,
        safety_notes=exercise.safety_notes,
        target_angle=exercise.target_angle
    )
 