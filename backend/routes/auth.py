"""
backend/routes/auth.py
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid

from backend.models.db_engine import get_db
from backend.models.database import User, Patient, EmailVerification, PasswordReset
from backend.models.schemas import (
    RegisterRequest, LoginRequest, TokenResponse, 
    VerifyEmailRequest, ResendOTPRequest,
    ForgotPasswordRequest, VerifyResetCodeRequest, ResetPasswordRequest
)
from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)
from backend.services.email_service import generate_otp, send_verification_email
from datetime import datetime, timedelta, timezone

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # التحقق من عدم وجود اسم مستخدم أو بريد إلكتروني مكرر
    existing = await db.execute(
        select(User).where((User.username == body.username) | (User.email == body.email))
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username or email already registered")

    # إنشاء المستخدم
    user = User(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    await db.flush()  # للحصول على user.id

    # ربط ملف المريض
    patient = Patient(
        user_id=user.id,
        full_name=body.full_name or body.username,
    )
    db.add(patient)
    await db.commit()
    await db.refresh(user)

    # إنشاء رمز JWT
    token = create_access_token({"sub": str(user.id)})
    
    # Generate OTP
    otp = generate_otp()
    otp_hash = hash_password(otp)
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    email_verification = EmailVerification(
        user_id=user.id,
        email=user.email,
        otp_code=otp_hash,
        expiration_time=expire_time.replace(tzinfo=None)
    )
    db.add(email_verification)
    await db.commit()
    
    # Send Email
    await send_verification_email(user.email, otp)
    
    return TokenResponse(
        access_token=token,
        username=user.username,
        email=user.email,
        has_profile=True,          # دائمًا True لأننا أنشأنا الملف
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    # البحث عن المستخدم بالبريد الإلكتروني أو اسم المستخدم
    result = await db.execute(
        select(User).where((User.username == body.username) | (User.email == body.username))
    )
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")
    if getattr(user, 'is_verified', False) == False:
        raise HTTPException(status_code=403, detail="Please verify your email first")

    # وجود ملف مريض
    p_result = await db.execute(select(Patient).where(Patient.user_id == user.id))
    has_profile = p_result.scalar_one_or_none() is not None

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(
        access_token=token,
        username=user.username,
        email=user.email,
        has_profile=has_profile,
    )


@router.get("/me")
async def get_me(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    p_result = await db.execute(select(Patient).where(Patient.user_id == current_user.id))
    has_profile = p_result.scalar_one_or_none() is not None
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "has_profile": has_profile,
        "created_at": current_user.created_at,
    }


@router.post("/verify-email")
async def verify_email(body: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    if not body.otp:
        raise HTTPException(status_code=400, detail="Verification code is required")
        
    result = await db.execute(
        select(EmailVerification)
        .where((EmailVerification.email == body.email) & (EmailVerification.verified_status == False))
        .order_by(EmailVerification.created_at.desc())
    )
    verification = result.scalars().first()
    
    if not verification:
        raise HTTPException(status_code=400, detail="Invalid verification code")
        
    if verification.expiration_time < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Verification code expired")
        
    if not verify_password(body.otp, verification.otp_code):
        raise HTTPException(status_code=400, detail="Invalid verification code")
        
    # Mark verified
    verification.verified_status = True
    
    # Update user
    user_result = await db.execute(select(User).where(User.email == body.email))
    user = user_result.scalar_one_or_none()
    if user:
        user.is_verified = True
        
    await db.commit()
    return {"message": "Email verified successfully"}


@router.post("/resend-otp")
async def resend_otp(body: ResendOTPRequest, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.email == body.email))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if getattr(user, 'is_verified', False):
        raise HTTPException(status_code=400, detail="User is already verified")
        
    # Rate limit check (optional but recommended)
    recent_result = await db.execute(
        select(EmailVerification)
        .where((EmailVerification.email == body.email) & (EmailVerification.created_at >= datetime.utcnow() - timedelta(minutes=1)))
    )
    if recent_result.scalars().first():
        raise HTTPException(status_code=429, detail="Please wait before requesting a new OTP")
        
    otp = generate_otp()
    otp_hash = hash_password(otp)
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    email_verification = EmailVerification(
        user_id=user.id,
        email=user.email,
        otp_code=otp_hash,
        expiration_time=expire_time.replace(tzinfo=None)
    )
    db.add(email_verification)
    await db.commit()
    
    await send_verification_email(user.email, otp)
    return {"message": "OTP sent successfully"}


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    user_result = await db.execute(select(User).where(User.email == body.email))
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if getattr(user, 'is_verified', False) == False:
        raise HTTPException(status_code=403, detail="Please verify your email before resetting your password")

    otp = generate_otp()
    otp_hash = hash_password(otp)
    expire_time = datetime.now(timezone.utc) + timedelta(minutes=10)
    
    # Invalidate previous unused resets
    await db.execute(
        PasswordReset.__table__.update()
        .where((PasswordReset.email == body.email) & (PasswordReset.is_used == False))
        .values(is_used=True)
    )
    
    reset_record = PasswordReset(
        user_id=user.id,
        email=user.email,
        reset_code=otp_hash,
        expires_at=expire_time.replace(tzinfo=None)
    )
    db.add(reset_record)
    await db.commit()
    
    # Send email
    subject = "Password Reset Verification"
    body_text = f"Your password reset verification code is: {otp}"
    from backend.services.email_service import send_email_sync
    import asyncio
    await asyncio.to_thread(send_email_sync, user.email, subject, body_text)
        
    return {"message": "Verification code has been sent."}


@router.post("/verify-reset-code")
async def verify_reset_code(body: VerifyResetCodeRequest, db: AsyncSession = Depends(get_db)):
    if not body.code:
        raise HTTPException(status_code=400, detail="Code required")
        
    result = await db.execute(
        select(PasswordReset)
        .where((PasswordReset.email == body.email) & (PasswordReset.is_used == False))
        .order_by(PasswordReset.created_at.desc())
    )
    reset_record = result.scalars().first()
    
    if not reset_record:
        raise HTTPException(status_code=400, detail="Invalid verification code")
        
    if reset_record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Code expired")
        
    if not verify_password(body.code, reset_record.reset_code):
        raise HTTPException(status_code=400, detail="Invalid verification code")
        
    # Generate reset token
    reset_token = str(uuid.uuid4())
    reset_record.reset_token = reset_token
    await db.commit()
    
    return {"reset_token": reset_token}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(PasswordReset)
        .where((PasswordReset.reset_token == body.reset_token) & (PasswordReset.is_used == False))
    )
    reset_record = result.scalars().first()
    
    if not reset_record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
        
    if reset_record.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Code expired")
        
    # Get user
    user_result = await db.execute(select(User).where(User.id == reset_record.user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    # Update password
    user.hashed_password = hash_password(body.new_password)
    reset_record.is_used = True
    await db.commit()
    
    return {"message": "Password updated successfully"}

@router.get("/debug-smtp")
async def debug_smtp():
    from config.settings import settings
    import smtplib
    if not settings.SMTP_EMAIL or not settings.SMTP_PASSWORD:
        return {"status": "error", "message": "SMTP_EMAIL or SMTP_PASSWORD is not set"}
    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=5) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
        return {"status": "success", "message": f"Successfully connected to SMTP as {settings.SMTP_EMAIL}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}