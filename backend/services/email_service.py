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
    return f"{random.randint(0, 999999):06d}"

def send_email_sync(to_email: str, subject: str, body: str):
    """Sync function to send email via SMTP."""
    if not settings.SMTP_EMAIL or not settings.SMTP_PASSWORD:
        print(f"Mock sending email to {to_email}. Subject: {subject}. Body: {body}")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = settings.SMTP_EMAIL
    msg["To"] = to_email

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=5) as server:
            server.starttls()
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
            server.send_message(msg)
    except TimeoutError:
        print(f"Timeout connecting to SMTP server {settings.SMTP_HOST}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")
        # In a real app we might raise or handle it

async def send_verification_email(to_email: str, otp_code: str):
    """Send verification OTP email."""
    subject = "Verify your account"
    body = f"Your verification code is: {otp_code}"
    await asyncio.to_thread(send_email_sync, to_email, subject, body)
