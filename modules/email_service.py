"""
email_service.py
-----------------
Handles email delivery using the Resend API.
Used primarily for sending OTP (One-Time Passwords) during candidate verification.
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("hiremind.email")

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_API_URL = "https://api.resend.com/emails"

def send_otp_email(to_email: str, otp_code: str) -> dict:
    """
    Sends a 6-digit OTP to the specified email address using Resend API.
    
    Returns:
        dict with 'success' (bool) and 'message' (str).
    """
    if not RESEND_API_KEY:
        error_msg = "Resend API key missing in environment variables (.env)."
        logger.error(error_msg)
        return {"success": False, "message": error_msg}

    if not to_email or "@" not in to_email:
        return {"success": False, "message": "Invalid email address."}

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    html_content = f"""
    <div style="font-family: sans-serif; max-width: 600px; margin: auto; padding: 20px; border: 1px solid #eee; border-radius: 10px;">
        <h2 style="color: #2563eb; text-align: center;">HireMind AI Verification</h2>
        <p>Hello,</p>
        <p>Your one-time password (OTP) for identity verification is:</p>
        <div style="background: #f8fafc; padding: 20px; text-align: center; border-radius: 8px; margin: 20px 0;">
            <span style="font-size: 32px; font-weight: 800; letter-spacing: 5px; color: #1e293b;">{otp_code}</span>
        </div>
        <p>This code will expire in 5 minutes. If you did not request this code, please ignore this email.</p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
        <p style="font-size: 12px; color: #64748b; text-align: center;">
            &copy; 2026 HireMind AI OS. Secure Autonomous Recruitment.
        </p>
    </div>
    """

    payload = {
        "from": "HireMind <onboarding@resend.dev>",
        "to": [to_email],
        "subject": f"{otp_code} is your HireMind verification code",
        "html": html_content,
    }

    try:
        response = requests.post(RESEND_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200 or response.status_code == 201:
            logger.info("OTP sent successfully to %s via Resend.", to_email)
            return {"success": True, "message": "OTP sent successfully."}
        else:
            resp_data = response.json()
            error_msg = resp_data.get("message", f"Resend API error: {response.status_code}")
            logger.error("Resend delivery failed: %s", error_msg)
            return {"success": False, "message": error_msg}

    except Exception as e:
        error_msg = f"Network error during email delivery: {str(e)}"
        logger.exception(error_msg)
        return {"success": False, "message": error_msg}


def send_hiring_email(to_email: str, candidate_name: str, job_role: str) -> dict:
    """
    Sends the official 'Selected' email to a candidate using Resend API.
    """
    if not RESEND_API_KEY:
        return {"success": False, "message": "Resend API key missing."}

    headers = {
        "Authorization": f"Bearer {RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    html_content = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: auto; padding: 40px; border: 1px solid #e2e8f0; border-radius: 24px; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
        <div style="text-align: center; margin-bottom: 30px;">
            <div style="font-size: 48px;">🎉</div>
            <h2 style="color: #0f172a; margin-top: 10px; font-size: 28px; font-weight: 800;">Congratulations!</h2>
        </div>
        
        <p style="font-size: 1.1rem; color: #1e293b;">Dear <strong>{candidate_name}</strong>,</p>
        
        <p style="font-size: 1rem; color: #475569; line-height: 1.7;">
            We are absolutely thrilled to inform you that you have been selected for the position of 
            <strong style="color: #4f46e5;">{job_role}</strong> at <strong>HireMind AI</strong>.
        </p>
        
        <p style="font-size: 1rem; color: #475569; line-height: 1.7;">
            Your performance throughout the selection process was exceptional. Out of many highly qualified candidates, 
            your skills, experience, and cultural fit stood out as a perfect match for our team.
        </p>
        
        <div style="background: #f8fafc; padding: 20px; border-radius: 16px; margin: 30px 0; border: 1px solid #f1f5f9;">
            <h4 style="margin: 0 0 10px 0; color: #1e293b; font-size: 1.1rem;">What's Next?</h4>
            <p style="margin: 0; color: #64748b; font-size: 0.95rem;">
                Our HR representative will contact you shortly with your official offer letter and further details 
                regarding the onboarding process.
            </p>
        </div>
        
        <p style="font-size: 1rem; color: #475569; line-height: 1.7;">
            Welcome to the future of hiring. We can't wait to see the incredible impact you'll make here!
        </p>
        
        <div style="margin-top: 40px; padding-top: 25px; border-top: 1px solid #f1f5f9; text-align: center;">
            <p style="font-size: 1.1rem; font-weight: 700; color: #1e293b; margin-bottom: 5px;">Best Regards,</p>
            <p style="font-size: 1rem; color: #6366f1; font-weight: 600; margin: 0;">The Hiring Team</p>
            <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px;">HireMind AI OS</p>
        </div>
    </div>
    """

    payload = {
        "from": "HireMind AI <onboarding@resend.dev>",
        "to": [to_email],
        "subject": "🎉 Congratulations! You're Selected",
        "html": html_content,
    }

    try:
        response = requests.post(RESEND_API_URL, headers=headers, json=payload, timeout=10)
        return {"success": response.status_code in [200, 201], "message": "Email sent"}
    except Exception as e:
        return {"success": False, "message": str(e)}

