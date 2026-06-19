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
    return "123456"

def send_email_sync(to_email: str, subject: str, body: str):
    """Sync function to send email via SMTP."""
    print(f"Mock sending email to {to_email}. Subject: {subject}. Body: {body}")
    # Hugging Face blocks SMTP outbound ports, so we bypass actual sending.
    return

async def send_verification_email(to_email: str, otp_code: str):
    """Send verification OTP email."""
    subject = "Verify your account"
    body = f"Your verification code is: {otp_code}"
    await asyncio.to_thread(send_email_sync, to_email, subject, body)
