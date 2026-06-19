# Smart Physical Therapy System

AI-powered web application for home-based physical rehabilitation.  
Uses **MediaPipe Pose** for sensor-less body tracking and provides real-time corrective feedback via WebSockets.

---

## Project Structure

```
smart_pt_system/
├── main.py                          # FastAPI application entry point
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (edit before running)
│
├── config/
│   └── settings.py                  # Pydantic settings (loads .env)
│
├── backend/
│   ├── models/
│   │   ├── database.py              # SQLAlchemy ORM models (User, Patient, Session, Exercise)
│   │   ├── db_engine.py             # Async engine + session factory + init_db()
│   │   └── schemas.py               # Pydantic v2 request/response schemas
│   │
│   ├── routes/
│   │   ├── auth.py                  # POST /api/auth/register  POST /api/auth/login
│   │   ├── patients.py              # GET/POST /api/patients/profile  GET /api/patients/dashboard
│   │   ├── exercises.py             # GET /api/exercises/  POST /api/exercises/seed
│   │   └── sessions.py              # POST /api/sessions/start  PATCH /api/sessions/{id}/complete
│   │                                # WS  /api/sessions/ws/{exercise_key}
│   │
│   └── services/
│       ├── pose_analyzer.py         # Core AI: MediaPipe → angle calc → rep counter → feedback
│       └── auth_service.py          # JWT creation, password hashing (bcrypt)
│
└── frontend/
    ├── templates/
    │   └── index.html               # Single-page app shell
    └── static/
        ├── css/
        │   └── main.css             # Dark-theme responsive stylesheet
        └── js/
            ├── api.js               # Fetch wrapper + WebSocket factory (JWT-aware)
            ├── session.js           # WebRTC capture → WS send → metrics display
            └── app.js               # SPA router, auth forms, dashboard, exercise library
```

---

## Quick Start

### 1. Clone / unzip the project

```bash
cd smart_pt_system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `mediapipe` requires Python 3.8–3.11.  
> On ARM Macs use `pip install mediapipe-silicon` instead.

### 4. Configure environment

Edit `.env` — at minimum change `SECRET_KEY`:

```env
SECRET_KEY=your-actual-secret-key
DATABASE_URL=sqlite+aiosqlite:///./smart_pt.db
```

### 5. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in Chrome or Edge (WebRTC + WebGL required).

### 6. Seed exercise library

On first run, visit the Exercises page — it will auto-seed via `POST /api/exercises/seed`.  
Or call it directly:

```bash
curl -X POST http://localhost:8000/api/exercises/seed
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Get JWT token |
| POST | `/api/patients/profile` | Create patient profile |
| GET  | `/api/patients/profile` | Get own profile |
| GET  | `/api/patients/dashboard` | Stats + recent sessions |
| GET  | `/api/exercises/` | List all exercises |
| POST | `/api/exercises/seed` | Seed default library |
| POST | `/api/sessions/start` | Start session record |
| PATCH| `/api/sessions/{id}/complete` | Save final results |
| GET  | `/api/sessions/history` | Last 20 sessions |
| WS   | `/api/sessions/ws/{exercise_key}` | Live AI analysis |

### Supported Exercise Keys (WebSocket)

| Key | Exercise |
|-----|----------|
| `knee_extension` | Seated knee extension |
| `bicep_curl` | Standing bicep curl |
| `shoulder_abduction` | Lateral shoulder raise |
| `squat` | Bodyweight squat |

### WebSocket Protocol

**Client → Server** (JSON):
```json
{ "frame": "<base64-JPEG-string>" }
```

**Server → Client** (JSON):
```json
{
  "joint_angle": 142.3,
  "is_correct_form": true,
  "feedback_message": "Good — straighten fully. Reps: 3",
  "rep_count": 3,
  "stage": "up",
  "max_angle_achieved": 155.2,
  "session_accuracy": 87.4,
  "landmarks_visible": true,
  "annotated_frame": "<base64-JPEG-with-skeleton-overlay>"
}
```

---

## How the AI Works

1. **Frame Capture** — Browser captures webcam frames at ~10 FPS via WebRTC and sends them over WebSocket as base64 JPEG.
2. **MediaPipe Pose** — Server decodes the frame and runs `mp_pose.Pose` to extract **33 3-D body landmarks**.
3. **Angle Calculation** — Uses the **dot-product / atan2** method on 3 landmarks (e.g. Hip→Knee→Ankle) to compute joint angle in degrees.
4. **State Machine** — Tracks `up`/`down` stage transitions; increments rep counter only on a completed cycle.
5. **Feedback** — Compares angle to safe ROM thresholds; returns colour-coded skeleton overlay + text/audio guidance.
6. **Session Storage** — Per-frame accuracy is averaged; final metrics are saved to SQLite via SQLAlchemy.

---

## Extending

### Add a new exercise

In `backend/services/pose_analyzer.py`:

```python
EXERCISE_CONFIGS["hip_flexion"] = ExerciseConfig(
    name="Hip Flexion",
    point_a=LANDMARK.LEFT_KNEE.value,
    point_b=LANDMARK.LEFT_HIP.value,
    point_c=LANDMARK.LEFT_SHOULDER.value,
    min_angle=170.0,
    max_angle=70.0,
    target_reps=10,
)
```

### Switch to PostgreSQL

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/smart_pt
```

```bash
pip install asyncpg
```

---

## Requirements Summary

- Python 3.9–3.11
- Modern browser with WebRTC (Chrome 80+, Edge 80+, Firefox 75+)
- Webcam ≥ 720p
- 4 GB RAM (8 GB recommended for smooth AI inference)
- Adequate lighting; clear background for best tracking accuracy
