"""
backend/services/email_service.py
Service for sending emails.
"""
import smtplib
import random
import asyncio
from email.message import EmailMessage
from config.settings import settings

def generate_otp() -> str:
    """Generate a 6-digit random OTP."""
    return str(random.randint(100000, 999999))

def send_email_sync(to_email: str, subject: str, body: str):
    """Sync function to send email via SMTP."""
    if not settings.SMTP_EMAIL or not settings.SMTP_PASSWORD:
        print(f"SMTP config missing. Mock sending to {to_email}: {body}")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

async def send_verification_email(to_email: str, otp_code: str):
    """Send verification OTP email."""
    subject = "Verify your account"
    body = f"Your verification code is: {otp_code}"
    await asyncio.to_thread(send_email_sync, to_email, subject, body)
