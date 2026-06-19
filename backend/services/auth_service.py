"""
backend/services/auth_service.py
JWT creation and password hashing.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from config.settings import settings
from backend.models.db_engine import get_db
from backend.models.database import User, Patient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import bcrypt


# ── Password helpers ───────────────────────────────────────────────────────────
def hash_password(plain: str) -> str:
    """تشفير كلمة المرور باستخدام bcrypt مع عدد الجولات المحدد في الإعدادات."""
    rounds = getattr(settings, "BCRYPT_ROUNDS", 12)  # قيمة افتراضية 12 إذا لم تُحدد
    salt = bcrypt.gensalt(rounds=rounds)
    return bcrypt.hashpw(plain.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """التحقق من تطابق كلمة المرور مع التشفير المخزن."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        # قد يحدث خطأ إذا كان الهاش غير صالح الصيغة
        return False


# ── JWT helpers ────────────────────────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """إنشاء رمز JWT يحتوي على تاريخ انتهاء الصلاحية."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> dict:
    """فك تشفير رمز JWT وإرجاع المحتوى، أو رفع استثناء إذا كان غير صالح."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── OAuth2 scheme ──────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


# ── FastAPI dependencies ───────────────────────────────────────────────────────
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """اعتماد FastAPI لاستخراج المستخدم الحالي من رمز JWT."""
    payload = decode_token(token)
    user_id: int = payload.get("sub")
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    if getattr(user, 'is_verified', False) == False:
        raise HTTPException(status_code=403, detail="Please verify your email first")
    return user


async def get_current_patient(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Patient:
    """اعتماد FastAPI لاستخراج ملف المريض المرتبط بالمستخدم الحالي."""
    result = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(
            status_code=404,
            detail="Patient profile not found. Please complete your profile first.",
        )
    return patient