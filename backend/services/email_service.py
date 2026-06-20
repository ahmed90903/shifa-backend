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

    # Print OTP to logs as a fallback for platforms that block SMTP (e.g., Hugging Face)
    print(f"--- EMAIL DEBUG ---")
    print(f"To: {to_email}\nSubject: {subject}\nBody: {body}")
    print(f"-------------------")

    try:
        # Try SSL on port 465 first (sometimes 587 is blocked but 465 is open)
        with smtplib.SMTP_SSL(settings.SMTP_HOST, 465, timeout=5) as server:
            server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully via port 465")
    except Exception as e1:
        print(f"Port 465 failed: {e1}. Trying port 587...")
        try:
            # Try TLS on port 587
            with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=5) as server:
                server.starttls()
                server.login(settings.SMTP_EMAIL, settings.SMTP_PASSWORD)
                server.send_message(msg)
                print("Email sent successfully via port 587")
        except Exception as e2:
            print(f"Failed to send email via both ports. Error: {e2}")
            print(f"NOTE: Hugging Face Free tier blocks SMTP ports (25, 465, 587). Please check the logs above for the OTP.")

async def send_verification_email(to_email: str, otp_code: str):
    """Send verification OTP email."""
    subject = "Verify your account"
    body = f"Your verification code is: {otp_code}"
    await asyncio.to_thread(send_email_sync, to_email, subject, body)
