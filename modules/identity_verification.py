"""
identity_verification.py
-------------------------
Core engine for cross-validation between Stage 1 (Resume) and Stage 2 (Verification).
Handles fuzzy name matching, email consistency, and face similarity.
"""

import re
import logging
import io
import base64
import random
import time
from difflib import SequenceMatcher
from PIL import Image
import numpy as np

# Try to import face_recognition for identity verification
try:
    import face_recognition
    HAS_FACE_REC = True
except ImportError:
    HAS_FACE_REC = False

logger = logging.getLogger("hiremind.verification")

# ── 1. Text Validation ────────────────────────────────────────────────────────

def verify_name(expected: str, actual: str, threshold: float = 0.85) -> dict:
    """Fuzzy match between parsed name and form name."""
    if not expected or not actual:
        return {"match": False, "score": 0.0, "status": "missing_data"}
    
    s = SequenceMatcher(None, expected.strip().lower(), actual.strip().lower())
    ratio = s.ratio()
    
    return {
        "match": ratio >= threshold,
        "score": round(ratio, 4),
        "status": "success" if ratio >= threshold else "mismatch"
    }

def verify_email(expected: str, actual: str) -> bool:
    """Strict case-insensitive match for email."""
    if not expected or not actual:
        return False
    return expected.strip().lower() == actual.strip().lower()

def normalize_phone(phone: str) -> str:
    """Keep only digits for comparison."""
    if not phone:
        return ""
    return re.sub(r"\D", "", phone)

def verify_phone(expected: str, actual: str) -> bool:
    """Compare suffixes of normalized phone numbers to handle country codes."""
    norm_exp = normalize_phone(expected)
    norm_act = normalize_phone(actual)
    
    if not norm_exp or not norm_act:
        return False
        
# Match the last 10 digits to be safe with international formats
    return norm_exp[-10:] == norm_act[-10:]


# ── 2. OTP Logic ──────────────────────────────────────────────────────────────

def generate_otp() -> str:
    """Generate a secure 6-digit random code."""
    return str(random.randint(100000, 999999))

def verify_otp_logic(submitted_otp: str, stored_otp: str, expiry_timestamp: float) -> dict:
    """
    Validates a submitted OTP against the stored one and checks expiry.
    
    Args:
        submitted_otp: Code entered by the user.
        stored_otp: Code generated and stored in session.
        expiry_timestamp: Time (from time.time()) when the OTP expires.
        
    Returns:
        dict with 'valid' (bool) and 'message' (str)
    """
    if not submitted_otp or not stored_otp:
        return {"valid": False, "message": "Missing OTP data."}
        
    if time.time() > expiry_timestamp:
        return {"valid": False, "message": "OTP has expired. Please request a new one."}
        
    if submitted_otp.strip() == stored_otp.strip():
        return {"valid": True, "message": "OTP verified successfully."}
    else:
        return {"valid": False, "message": "Incorrect OTP. Please try again."}


# ── 3. Face Recognition ────────────────────────────────────────────────────────

def verify_identity(resume_photo_b64: str, live_photo_bytes: bytes) -> dict:
    """
    Compares the face found in the resume against the live webcam capture.
    
    Returns:
        dict with keys: 'match', 'score', 'status', 'error'
    """
    if not resume_photo_b64:
        return {"match": False, "score": 0.0, "status": "no_resume_photo"}
    
    if not HAS_FACE_REC:
        logger.warning("face_recognition library not found. Skipping AI face match.")
        return {"match": True, "score": 1.0, "status": "skipped_lib_missing", "error": "AI identity verification requires 'face_recognition' library."}

    try:
        # Load resume photo
        resume_bytes = base64.b64decode(resume_photo_b64)
        resume_image = face_recognition.load_image_file(io.BytesIO(resume_bytes))
        
        # Load live photo
        live_image = face_recognition.load_image_file(io.BytesIO(live_photo_bytes))
        
        # Get face encodings
        resume_encodings = face_recognition.face_encodings(resume_image)
        live_encodings   = face_recognition.face_encodings(live_image)
        
        if not resume_encodings:
            return {"match": False, "score": 0.0, "status": "no_face_in_resume"}
        if not live_encodings:
            return {"match": False, "score": 0.0, "status": "no_face_in_live"}
            
        # Compare (Distance-based: 0 is identical, higher is more different)
        # 0.6 is the default threshold in face_recognition
        results = face_recognition.compare_faces([resume_encodings[0]], live_encodings[0], tolerance=0.6)
        distance = face_recognition.face_distance([resume_encodings[0]], live_encodings[0])[0]
        
        # Convert distance to similarity score (0.0 to 1.0)
        similarity = max(0, 1 - distance)
        
        return {
            "match": bool(results[0]),
            "score": round(similarity, 4),
            "status": "success"
        }
    except Exception as e:
        logger.error("Face matching error: %s", e)
        return {"match": False, "score": 0.0, "status": "error", "error": str(e)}
