import os
import re
import io
import json
import streamlit as st
from PIL import Image, ImageFilter
from dotenv import load_dotenv
from modules.data_ingestion import process_resume
from modules.text_processing import build_structured_json
from modules.embeddings import process_embeddings, INDEX_PATH
from modules.similarity import match_resume_to_jd
from modules.llm_analysis import generate_analysis_report, PROVIDER_GEMINI
from modules.ats_scorer import run_ats_analysis
from modules.fraud_detector import generate_fraud_report
from modules.recommendation_engine import build_recommendation_output
from modules.candidate_db import store_candidate_data, init_db
from modules.identity_verification import (
    verify_name, verify_email, verify_phone, verify_identity,
    generate_otp, verify_otp_logic
)
from modules.email_service import send_otp_email
from modules.interview_ui import render_mode_selection, render_interview_ui
from modules.voice_ui import render_voice_interview, render_combined_report
from modules.auth_db import init_auth_db
from modules.auth_ui import render_role_selection, render_login_page, render_candidate_dashboard, render_admin_dashboard
import time

# Load environment variables from .env file
load_dotenv()
_GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
_OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
_ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
_LLM_PROVIDER    = os.getenv("LLM_PROVIDER", PROVIDER_GEMINI)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="HireMind AI - Autonomous HR OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- CUSTOM CSS ---
def local_css():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    /* Header Styling */
    .header-container {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 2rem;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        color: white;
    }

    .main-title {
        color: #f8fafc;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Metric Card Styling */
    .metric-card-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563eb;
        margin: 10px 0;
    }

    .metric-label {
        color: #64748b;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Skill Tags */
    .tag {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.3rem;
    }

    .tag-match { background-color: #ecfdf5; color: #059669; border: 1px solid #10b981; }
    .tag-missing { background-color: #fff1f2; color: #e11d48; border: 1px solid #f43f5e; }

    /* Action Button Customization */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        border-radius: 12px;
        padding: 1rem 3rem;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3);
        transition: all 0.3s ease;
        display: block;
        margin: 2rem auto;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        transform: scale(1.02);
    }

    /* Input Sections */
    .stExpander {
        border-radius: 16px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: white !important;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05) !important;
    }

    /* Ensure text within input areas and labels is visible (black/dark) */
    .stExpander p, .stExpander label, .stExpander div {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- COMPONENTS ---

def display_header():
    st.markdown("""
        <div class="header-container">
            <div class="main-title">🎯 HireMind AI</div>
            <div class="subtitle">Elevating Recruitment with Autonomous HR Intelligence, <br>Advanced Screening, and Fraud Detection.</div>
        </div>
    """, unsafe_allow_html=True)

def display_input_workspace():
    st.markdown("### 📥 Evaluation Workspace")
    
    col_input, _ = st.columns([2, 1])
    
    with col_input:
        # Resume Section
        with st.expander("👤 Upload Candidate Resume", expanded=True):
            st.info("AI Analysis works best with clean text-based PDF/DOCX files.")
            resume_file = st.file_uploader(
                "Drop resume here",
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed",
                key="resume_uploader",
            )

            if resume_file:
                # Only re-process if the file changes
                last_name = st.session_state.get("last_resume_name")
                if last_name != resume_file.name:
                    with st.spinner("⚙️ Extracting and processing resume..."):
                        result = process_resume(resume_file)
                    st.session_state["resume_result"]    = result
                    st.session_state["last_resume_name"] = resume_file.name

                    # --- Text Processing: Parse into structured JSON ---
                    if result.get("status") == "success":
                        with st.spinner("🧠 Parsing resume with NLP..."):
                            parsed = build_structured_json(
                                result["cleaned_text"], 
                                profile_photo=result.get("profile_photo")
                            )
                        st.session_state["parsed_resume"] = parsed

                        # --- Embeddings: Generate & store vectors ---
                        with st.spinner("🔢 Generating vector embeddings..."):
                            emb_result = process_embeddings(parsed, source_label="resume")
                        st.session_state["embedding_result"] = emb_result

                result = st.session_state.get("resume_result", {})

                if result.get("status") == "success":
                    st.success(result["message"])
                    col_a, col_b = st.columns(2)
                    col_a.metric("📝 Word Count",  f"{result['word_count']:,}")
                    col_b.metric("🔢 Char Count",  f"{result['char_count']:,}")

                    with st.expander("📄 Preview Extracted Text (first 1000 chars)"):
                        st.code(result["cleaned_text"][:1000], language=None)

                    st.caption(f"📁 Saved to: `{result['raw_path'].name}`")
                else:
                    st.warning(result.get("message", "Processing failed."))

        # --- Parsed Resume Display (STRUCTURED) ---
        parsed = st.session_state.get("parsed_resume")
        if parsed:
            with st.expander("🧩 Candidate Professional Profile", expanded=True):
                # ── Header: Identity & Contact ──
                st.markdown("#### 👤 Primary Identification")
                c1, c2, c3 = st.columns([2, 2, 2])
                with c1:
                    st.markdown(f"**Name**  \n{parsed.get('name') or '—'}")
                with c2:
                    st.markdown(f"**Email**  \n{parsed.get('email') or '—'}")
                with c3:
                    st.markdown(f"**Phone**  \n{parsed.get('phone') or '—'}")
                
                st.markdown(f"**📍 Location:** {parsed.get('location') or 'Not specified'}")
                st.markdown("<br>", unsafe_allow_html=True)

                # ── Summary / Objective ──
                summary = parsed.get("summary")
                if summary:
                    with st.container(border=True):
                        st.markdown("**📝 Professional Summary**")
                        st.write(summary)
                    st.markdown("<br>", unsafe_allow_html=True)

                # ── Core Skills ──
                skills = parsed.get("skills", [])
                if skills:
                    st.markdown(f"#### 🛠 Skills & Expertise ({len(skills)})")
                    tags_html = "".join(
                        f'<span class="tag tag-match">{s}</span>' for s in skills
                    )
                    st.markdown(
                        f'<div style="margin-bottom: 1.5rem;">{tags_html}</div>', 
                        unsafe_allow_html=True
                    )

                # ── Professional Experience & Background ──
                st.markdown("#### 💼 Background Detail")
                sec_tabs = st.tabs(["🚀 Projects", "💼 Experience", "🎓 Education", "📋 Raw JSON"])
                
                with sec_tabs[0]:
                    content = parsed.get("projects")
                    if content:
                        st.markdown(content)
                    else:
                        st.info("No projects section detected.")

                with sec_tabs[1]:
                    content = parsed.get("experience")
                    if content:
                        st.markdown(content)
                    else:
                        st.info("No work experience section detected.")

                with sec_tabs[2]:
                    content = parsed.get("education")
                    if content:
                        st.markdown(content)
                    else:
                        st.info("No education section detected.")

                with sec_tabs[3]:
                    st.json(parsed)

        # --- Embedding Stats ---
        emb_result = st.session_state.get("embedding_result")
        if emb_result and emb_result.get("status") == "success":
            with st.expander("🔢 Vector Embedding Status", expanded=False):
                v1, v2, v3 = st.columns(3)
                v1.metric("📦 Text Chunks",  emb_result["total_chunks"])
                v2.metric("🧮 Vectors Stored", emb_result["total_vectors"])
                v3.metric("📐 Dimension",     emb_result["dimension"])
                st.success(emb_result["message"])
                if emb_result.get("storage"):
                    st.caption(f"💾 Index: `{emb_result['storage']['index_path'].name}`")

        # Job Description Section
        with st.expander("📝 Job Description Parameters", expanded=True):
            user_role = st.session_state.get("user_role")
            
            if user_role == "candidate":
                st.info("📌 You are applying for the following role. The Job Description is pre-loaded and read-only.")
                st.text_area(
                    "JD Content",
                    value=st.session_state.get("jd_text", ""),
                    height=250,
                    disabled=True,
                    key="jd_text_display"
                )
            else:
                input_mode = st.radio("Input Method", ["Paste Text", "Upload Document"], horizontal=True)

                if input_mode == "Paste Text":
                    st.text_area(
                        "JD Content",
                        placeholder="Enter the role requirements and context here...",
                        height=250,
                        key="jd_text",
                    )
                else:
                    jd_file = st.file_uploader("Load JD Source", type=["pdf", "docx"], key="jd_file_upload")
                    if jd_file:
                        # Quick extract for JD files (reuse ingestion helpers)
                        from modules.data_ingestion import extract_text, clean_text
                        from pathlib import Path
                        import tempfile, os
                        tmp_path = Path("data") / "jd_temp" / jd_file.name
                        tmp_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path.write_bytes(jd_file.getbuffer())
                        raw = extract_text(tmp_path)
                        jd_text_extracted = clean_text(raw)
                        st.session_state["jd_text"] = jd_text_extracted
                        st.success(f"✅ JD loaded: {jd_file.name}")
                        with st.expander("📄 JD Preview"):
                            st.text(jd_text_extracted[:800])

# ── ELIGIBILITY & INTERVIEW LOGIC ──────────────────────────────────────────────
def check_interview_eligibility():
    """
    Evaluates if candidate has completed Stage 1 (Screening).
    Now completion-based rather than score-based.
    """
    return st.session_state.get("stage_1_complete", False)


def render_interview_button():
    """Renders the Stage 1 → Stage 2 transition button."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🎉 **Stage 1 Complete! Candidate is ready for the Interview Stage.**")
    st.markdown("""
        <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
                    padding:1rem 1.5rem;margin-bottom:1rem;">
            <strong>📋 Next Step:</strong> Click below to complete identity verification before the interview.
        </div>
    """, unsafe_allow_html=True)
    if st.button("🚀 Proceed to Interview Rounds", use_container_width=True, type="primary"):
        st.session_state["current_page"] = "verification"
        st.rerun()


def render_recommendations():
    """Renders skill gap analysis and suggestions — available to all candidates who completed Stage 1."""
    rec_result = st.session_state.get("rec_result")
    if not rec_result or rec_result.get("status") != "success":
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Skill Gap Analysis & Recommendations")

    # LLM Career Plan
    llm_plan = rec_result.get("llm_career_plan")
    if llm_plan:
        with st.container(border=True):
            st.markdown("#### 🧠 AI Career Coach")
            st.write(llm_plan)

    # Learning Paths
    learning = rec_result.get("recommended_learning", [])
    if learning:
        with st.expander(f"📚 Suggested Learning Paths ({len(learning)})", expanded=True):
            for rec in learning:
                st.markdown(f"**{rec['priority']}. {rec['skill'].title()}** ({rec['domain']})")
                for r in rec.get("resources", []):
                    st.markdown(f"  {r}")
                st.markdown("---")

    # Alternative Roles
    alt_roles = rec_result.get("alternative_roles", [])
    if alt_roles:
        with st.expander(f"🔀 Alternative Roles ({len(alt_roles)})", expanded=False):
            for role in alt_roles:
                pct   = role["match_pct"]
                color = "#10b981" if pct >= 70 else ("#f59e0b" if pct >= 50 else "#64748b")
                st.markdown(
                    f"**{role['role']}** — "
                    f"<span style='color:{color};font-weight:700'>{pct}% match</span> "
                    f"({role['missing_count']} skills to go)",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — CANDIDATE VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_user_inputs(name: str, email: str, phone: str):
    """
    Validate personal-detail fields.

    Returns
    -------
    (is_valid: bool, errors: list[str])
    """
    errors = []

    if not name or len(name.strip()) < 2:
        errors.append("Full name must be at least 2 characters.")

    email_pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    if not email or not re.match(email_pattern, email.strip()):
        errors.append("Please enter a valid email address (e.g. name@domain.com).")

    phone_clean = re.sub(r"[\s\-\(\)\+]", "", phone or "")
    if not phone_clean.isdigit() or not (7 <= len(phone_clean) <= 15):
        errors.append("Please enter a valid phone number (7–15 digits; spaces/dashes allowed).")

    return len(errors) == 0, errors


def validate_webcam_image(image_file):
    """
    Basic image-quality heuristics using PIL.

    Checks
    ------
    1. Average brightness  — flags very dark images.
    2. Edge-variance       — flags blurry images.
    3. Centre-region variance — heuristic for face presence.

    Returns
    -------
    (is_valid: bool, warnings: list[str])
    """
    warnings_out = []
    try:
        img      = Image.open(image_file).convert("RGB")
        img_gray = img.convert("L")
        pixels   = list(img_gray.getdata())

        # 1. Brightness check
        avg_brightness = sum(pixels) / len(pixels)
        if avg_brightness < 55:
            warnings_out.append(
                "⚠️ Low lighting detected — please sit in a brighter environment."
            )
            return False, warnings_out

        # 2. Sharpness via edge-detection variance
        edge_pixels = list(img_gray.filter(ImageFilter.FIND_EDGES).getdata())
        mean_e      = sum(edge_pixels) / len(edge_pixels)
        variance    = sum((p - mean_e) ** 2 for p in edge_pixels) / len(edge_pixels)
        if variance < 80:
            warnings_out.append(
                "⚠️ Image appears blurry — please hold still and ensure good focus."
            )
            return False, warnings_out

        # 3. Centre-region activity heuristic (face presence)
        w, h         = img.size
        center_crop  = img_gray.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
        c_pixels     = list(center_crop.getdata())
        c_mean       = sum(c_pixels) / len(c_pixels)
        c_variance   = sum((p - c_mean) ** 2 for p in c_pixels) / len(c_pixels)
        if c_variance < 60:
            warnings_out.append(
                "⚠️ No face detected in the centre frame — "
                "please position your face in the middle of the camera view."
            )
            return False, warnings_out

        return True, []

    except Exception as exc:
        warnings_out.append(f"❌ Could not analyse image: {exc}")
        return False, warnings_out


def proceed_to_interview():
    """Shows verification success banner and the button to enter the Interview stage."""
    candidate_name = st.session_state.get("candidate_name", "Candidate")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
            border-radius: 20px; padding: 2.5rem; text-align: center;
            box-shadow: 0 15px 35px rgba(16,185,129,0.25); margin-bottom: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 0.5rem;">✅</div>
            <div style="color: #ecfdf5; font-size: 2rem; font-weight: 800;"
            >Verification Completed Successfully!</div>
            <div style="color: #6ee7b7; font-size: 1.1rem; margin-top: 0.6rem;">
                Welcome, <strong>{candidate_name}</strong>.
                Your identity has been confirmed. You may now enter the Interview Stage.
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🎙️ Enter Interview Interface",
        use_container_width=True,
        type="primary",
        key="enter_interview_btn",
    ):
        # Create DB session now that we have a candidate_id
        from modules.candidate_db import create_interview_session
        mode = st.session_state.get("interview_mode", "normal")
        candidate_id = st.session_state.get("candidate_id")
        
        if candidate_id:
            job_id = st.session_state.get("current_job_id", 0)
            
            # Extract Stage 1 score from analysis result
            ats_res = st.session_state.get("ats_result", {})
            ats_val = ats_res.get("ats_score", 0.0)
            
            session_id = create_interview_session(candidate_id, mode, job_id, ats_score=ats_val)
            st.session_state["interview_session_id"] = session_id
            st.session_state["current_page"] = "interview_mode"
            st.rerun()


        else:
            st.error("Candidate record not found. Please re-run verification.")


def render_verification_page():
    """
    Stage 2 Gateway — Candidate Identity Verification.

    Collects:
      • Full name, email, phone  (validated text inputs)
      • Live webcam photo        (via st.camera_input / getUserMedia — no uploads)

    On success:
      • Stores record in local SQLite via candidate_db.store_candidate_data()
      • Sets session_state["verification_complete"] = True
      • Routes the user to the Interview stage
    """
    # ── Stage guard ───────────────────────────────────────────────────────────
    if not st.session_state.get("stage_1_complete", False):
        st.error("❌ Access Denied: Please complete Stage 1 Screening first.")
        if st.button("⬅️ Return to Dashboard", key="verif_guard_back"):
            role = st.session_state.get("user_role", "candidate")
            st.session_state["current_page"] = f"{role}_dashboard"
            st.rerun()
        st.stop()

    local_css()

    # ── Already verified? Skip straight to confirmation ───────────────────────
    if st.session_state.get("verification_complete"):
        st.markdown("""
            <div class="header-container">
                <div class="main-title">🔐 Candidate Verification</div>
                <div class="subtitle">Stage 2 — Identity Confirmed</div>
            </div>
        """, unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            proceed_to_interview()
        return

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
        <div class="header-container">
            <div style="font-size:0.82rem;color:#94a3b8;text-transform:uppercase;
                        letter-spacing:0.12em;margin-bottom:0.6rem;">Stage 2 Pre-Check</div>
            <div class="main-title">🔐 Candidate Verification</div>
            <div class="subtitle">
                Complete identity verification before entering the interview.<br>
                All fields and a live photo are mandatory.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_back, _ = st.columns([1, 5])
    with col_back:
        if st.button("⬅️ Back to Results", key="verif_back_btn"):
            st.session_state["current_page"] = "screening"
            st.rerun()

    _, mid, _ = st.columns([1, 3, 1])
    with mid:

        # ── Step indicator ────────────────────────────────────────────────────
        st.markdown("""
            <div style="display:flex;align-items:flex-start;justify-content:center;
                        gap:0;margin:1.8rem 0 2.2rem;">
                <div style="text-align:center;min-width:90px;">
                    <div style="width:44px;height:44px;border-radius:50%;
                                background:#2563eb;color:white;font-weight:700;
                                font-size:1.1rem;line-height:44px;margin:0 auto;">1</div>
                    <div style="font-size:0.75rem;color:#2563eb;font-weight:600;
                                margin-top:0.35rem;">Personal<br>Details</div>
                </div>
                <div style="flex:1;border-top:2px dashed #cbd5e1;margin-top:22px;"></div>
                <div style="text-align:center;min-width:90px;">
                    <div style="width:44px;height:44px;border-radius:50%;
                                background:#2563eb;color:white;font-weight:700;
                                font-size:1.1rem;line-height:44px;margin:0 auto;">2</div>
                    <div style="font-size:0.75rem;color:#2563eb;font-weight:600;
                                margin-top:0.35rem;">Live<br>Photo</div>
                </div>
                <div style="flex:1;border-top:2px dashed #cbd5e1;margin-top:22px;"></div>
                <div style="text-align:center;min-width:90px;">
                    <div style="width:44px;height:44px;border-radius:50%;
                                background:#94a3b8;color:white;font-weight:700;
                                font-size:1.1rem;line-height:44px;margin:0 auto;">3</div>
                    <div style="font-size:0.75rem;color:#94a3b8;font-weight:600;
                                margin-top:0.35rem;">Confirm &amp;<br>Proceed</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── STEP 1 — Personal Details ─────────────────────────────────────────
        parsed = st.session_state.get("parsed_resume", {})
        name_prefilled = bool(parsed.get("name"))
        email_prefilled = bool(parsed.get("email"))
        
        st.markdown("#### 👤 Step 1 — Personal Details")
        if name_prefilled and email_prefilled:
            st.info("ℹ️ Identity details auto-filled from resume (read-only for security)")
        
        with st.container(border=True):
            full_name = st.text_input(
                "Full Name \u002a",
                value=parsed.get("name") or "",
                placeholder="Enter your full legal name",
                key="verif_name",
                disabled=name_prefilled,
                help="Locked to match resume data" if name_prefilled else None
            )
            col_e, col_p = st.columns(2)
            with col_e:
                email = st.text_input(
                    "Email Address \u002a",
                    value=parsed.get("email") or "",
                    placeholder="you@example.com",
                    key="verif_email",
                    disabled=email_prefilled,
                    help="Locked to match resume data" if email_prefilled else None
                )
            with col_p:
                phone = st.text_input(
                    "Phone Number \u002a",
                    value=parsed.get("phone") or "",
                    placeholder="+91 98765 43210",
                    key="verif_phone",
                )

        # ── Email Verification (Only if NOT prefilled) ────────────────────────
        if not email_prefilled:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown("📧 **Email Verification Required**")
                st.caption("Please verify your manual email entry to proceed.")
                
                c_otp1, c_otp2 = st.columns([2, 1])
                with c_otp2:
                    if st.button("📩 Send OTP", use_container_width=True):
                        if not email or "@" not in email:
                            st.error("Please enter a valid email first.")
                        else:
                            otp = generate_otp()
                            res = send_otp_email(email, otp)
                            if res["success"]:
                                st.session_state["stored_otp"] = otp
                                st.session_state["otp_expiry"] = time.time() + 300 # 5 mins
                                st.success("OTP sent!")
                            else:
                                st.error(res["message"])
                
                with c_otp1:
                    user_otp = st.text_input("Enter 6-digit Code", placeholder="XXXXXX", label_visibility="collapsed")
                    if user_otp:
                        verif = verify_otp_logic(
                            user_otp, 
                            st.session_state.get("stored_otp"), 
                            st.session_state.get("otp_expiry", 0)
                        )
                        if verif["valid"]:
                            st.session_state["email_verified"] = True
                            st.success("✅ Email Verified")
                        else:
                            st.error(verif["message"])
                            st.session_state["email_verified"] = False

        st.markdown("<br>", unsafe_allow_html=True)

        # ── STEP 2 — Live Webcam Photo (getUserMedia via st.camera_input) ─────
        st.markdown("#### 📸 Step 2 — Live Photo Capture (Mandatory)")
        with st.container(border=True):
            st.markdown("""
                <div style="background:#eff6ff;border-left:4px solid #2563eb;
                            border-radius:0 10px 10px 0;padding:1rem 1.2rem;
                            margin-bottom:1rem;">
                    <strong>📋 Photo Guidelines</strong>
                    <ul style="margin:0.6rem 0 0;padding-left:1.2rem;color:#334155;
                               line-height:1.8;">
                        <li>Sit in a <strong>well-lit room</strong> — avoid backlighting</li>
                        <li>Keep your <strong>face clearly visible</strong> and centred in the frame</li>
                        <li>Avoid <strong>blurry or dark</strong> captures — hold still</li>
                        <li>Only <strong>live webcam photos</strong> are accepted — file uploads are not allowed</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            photo = st.camera_input(
                "Click the button below to activate your webcam and capture a photo",
                key="verif_camera",
                help="Your device webcam will open via getUserMedia. Frame your face and click \u2018Take Photo\u2019.",
            )

            if photo:
                col_preview, col_status = st.columns([1, 1])
                with col_preview:
                    st.image(photo, caption="Captured photo preview", use_container_width=True)
                with col_status:
                    valid_img, img_warnings = validate_webcam_image(photo)
                    if valid_img:
                        st.success("✅ Photo quality check passed!\nImage is clear and well-lit.")
                    else:
                        for w in img_warnings:
                            st.warning(w)
                        st.info(
                            "💡 Please retake the photo by clicking **'Clear photo'** above, "
                            "then try again."
                        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SUBMIT & VERIFY ───────────────────────────────────────────────────
        if st.button(
            "🔐 Run Identity Check & Verify",
            use_container_width=True,
            type="primary",
            key="verif_submit_btn",
        ):
            name_val   = st.session_state.get("verif_name",  "").strip()
            email_val  = st.session_state.get("verif_email", "").strip()
            phone_val  = st.session_state.get("verif_phone", "").strip()
            photo_file = st.session_state.get("verif_camera")

            # 1. Basic field mandatory check
            if not name_val or not email_val or not phone_val or not photo_file:
                st.error("❌ All fields and a live photo are mandatory for verification.")
                st.stop()

            # 1b. OTP Check (Only if manual email)
            parsed = st.session_state.get("parsed_resume", {})
            email_prefilled = bool(parsed.get("email"))
            if not email_prefilled and not st.session_state.get("email_verified"):
                st.error("❌ Email verification required. Please verify the OTP sent to your email.")
                st.stop()

            # 2. Run Cross-Validation against Stage 1 Data
            
            with st.spinner("🕵️ Verification in progress..."):
                name_result  = verify_name(parsed.get("name"), name_val)
                email_match  = verify_email(parsed.get("email"), email_val)
                phone_match  = verify_phone(parsed.get("phone"), phone_val)
                face_result  = verify_identity(parsed.get("profile_photo"), photo_file.getvalue())
            
            # Store results for dashboard
            st.session_state["verif_results"] = {
                "name": name_result,
                "email": email_match,
                "phone": phone_match,
                "face": face_result
            }
            
            # MAJOR MISMATCH CHECK
            if not email_match:
                st.error("🛑 IDENTITY ALERT: Email does not match the resume. Access Blocked.")
                st.stop()
            
            if face_result["status"] == "success" and not face_result["match"]:
                st.error("🛑 IDENTITY ALERT: Live face does not match resume photo. Access Blocked.")
                st.stop()

            # If we reach here, minor mismatches are warnings but allowed
            if not name_result["match"] or not phone_match:
                st.warning("⚠️ Some details vary slightly from the resume, but identity is confirmed.")

            # — Persist to database
            try:
                candidate_id = store_candidate_data(
                    full_name  = name_val,
                    email      = email_val,
                    phone      = phone_val,
                    photo_bytes= photo_file.getvalue(),
                    session_id = str(id(st.session_state)),
                )
                st.session_state["candidate_id"]           = candidate_id
                st.session_state["candidate_name"]         = name_val
                st.session_state["candidate_email"]        = email_val
                st.session_state["verification_complete"]  = True
                st.balloons()
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Database error — could not save record: {exc}")

    # ── VERIFICATION DASHBOARD (Persistent after submission attempt) ─────────
    res = st.session_state.get("verif_results")
    if res:
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 3, 1])
        with mid:
            st.markdown("### 📋 Verification Summary")
            c1, c2, c3, c4 = st.columns(4)
            
            def get_status_ui(is_match):
                return ("✅ Match", "#10b981") if is_match else ("❌ Mismatch", "#ef4444")
                
            with c1:
                label, color = get_status_ui(res["name"]["match"])
                st.metric("Name (Fuzzy)", label, delta=f"{res['name']['score']*100:.0f}%", delta_color="normal")
            with c2:
                label, color = get_status_ui(res["email"])
                st.metric("Email (Exact)", label)
            with c3:
                label, color = get_status_ui(res["phone"])
                st.metric("Phone (Suffix)", label)
            with c4:
                face = res["face"]
                if face["status"] == "no_resume_photo":
                    st.metric("Face Match", "N/A", help="No photo found in resume.")
                elif face["status"] == "skipped_lib_missing":
                    st.metric("Face Match", "Skipped", help=face["error"])
                else:
                    label, color = get_status_ui(face["match"])
                    st.metric("Face ID", label, delta=f"{face['score']*100:.0f}%" if face["status"]=="success" else "0%", delta_color="normal")


# Unified Interview UI is now handled via modules/interview_ui.py

# Legacy interview function removed. System now uses modules/interview_ui.py


def display_dashboard():
    """Renders the analysis dashboard with LIVE data if available, placeholders otherwise."""
    match_result = st.session_state.get("match_result")

    st.markdown("<br><hr style='border-top: 2px solid #e2e8f0;'><br>", unsafe_allow_html=True)
    st.markdown("### 📊 Interviewer Dashboard")

    if match_result and match_result.get("status") == "success" and match_result["ranked"]:
        # ── LIVE DATA ──────────────────────────────────────────────────
        candidate = match_result["ranked"][0]   # Top candidate

        # Score tier label
        score = candidate["final_score"]
        if score >= 75:
            tier, tier_color = "Strong Match", "#10b981"
        elif score >= 50:
            tier, tier_color = "Moderate Match", "#f59e0b"
        else:
            tier, tier_color = "Weak Match", "#ef4444"

        # KPIs Row
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Semantic Alignment</div>
                    <div class="metric-value">{candidate['semantic_score']:.1f}%</div>
                    <div style="color: {tier_color}; font-weight: 600;">{tier}</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Skill Match</div>
                    <div class="metric-value">{candidate['skill_score']:.1f}%</div>
                    <div style="color: #3b82f6; font-weight: 600;">{len(candidate['matched_skills'])} of {len(candidate['matched_skills']) + len(candidate['missing_skills'])} skills</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Overall Score</div>
                    <div class="metric-value" style="color: {tier_color};">{score:.1f}%</div>
                    <div style="color: #64748b; font-weight: 600;">Weighted Final</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Detail Row
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            with st.container(border=True):
                st.markdown("#### 🔍 Skill Gap Analysis")
                matched = candidate.get("matched_skills", [])
                missing = candidate.get("missing_skills", [])

                if matched:
                    st.write("**Matched Skills:**")
                    tags = " ".join(f'<span class="tag tag-match">{s}</span>' for s in matched)
                    st.markdown(tags, unsafe_allow_html=True)

                if missing:
                    st.write("**Missing Skills:**")
                    tags = " ".join(f'<span class="tag tag-missing">{s}</span>' for s in missing)
                    st.markdown(tags, unsafe_allow_html=True)

                if not matched and not missing:
                    st.info("No JD skills extracted for comparison.")

        with detail_col2:
            with st.container(border=True):
                st.markdown("#### 🧠 Top Matching Segments")
                top_chunks = candidate.get("top_chunks", [])
                if top_chunks:
                    # Map section keys to readable labels
                    section_map = {
                        "experience": "💼 Work Experience",
                        "projects":   "🚀 Project / Portfolio",
                        "skills":     "🛠 Skills & Expertise",
                        "summary":     "📋 Summary / Objective",
                        "education":  "🎓 Education",
                        "overview":   "🔍 General Overview"
                    }
                    
                    for i, chunk in enumerate(top_chunks, 1):
                        score_pct = chunk['score'] * 100
                        section_display = section_map.get(chunk['section'], chunk['section'].title())
                        
                        st.markdown(f"""
                            <div style="margin-bottom: 0.5rem;">
                                <strong>#{i}</strong> 
                                <span style="background:#f1f5f9; color:#475569; padding:2px 8px; 
                                             border-radius:6px; font-size:0.75rem; font-weight:700; 
                                             margin-left:8px; border:1px solid #e2e8f0;">
                                    {section_display.upper()}
                                </span>
                                <span style="color:#2563eb; font-weight:700; margin-left:12px;">
                                    {score_pct:.1f}% Match
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Clean display of chunk text
                        st.write(chunk["chunk_text"])
                        
                        if i < len(top_chunks):
                            st.markdown("---")
                else:
                    st.info("No high-relevance segments found for this JD.")

        # JD Skills extracted
        jd_skills = match_result.get("jd_skills", [])
        if jd_skills:
            with st.expander(f"🎯 JD Skills Detected ({len(jd_skills)})", expanded=False):
                tags = " ".join(f'<span class="tag tag-match">{s}</span>' for s in jd_skills)
                st.markdown(tags, unsafe_allow_html=True)

        # ── AI INSIGHTS (LLM Analysis) ──────────────────────────────────
        llm_result = st.session_state.get("llm_result")
        if llm_result and llm_result.get("status") == "success":
            report = llm_result["report"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🧠 AI Deep Analysis")

            # Match percentage KPI
            ai_score = report.get("match_percentage", 0)
            if ai_score >= 75:
                ai_color = "#10b981"
            elif ai_score >= 50:
                ai_color = "#f59e0b"
            else:
                ai_color = "#ef4444"

            st.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">AI Match Score</div>
                    <div class="metric-value" style="color: {ai_color};">{ai_score}%</div>
                    <div style="color: #64748b; font-weight: 600;">LLM Assessed</div>
                </div>
            """, unsafe_allow_html=True)

            # Match Summary
            st.info(report.get("match_summary", ""))

            # Strengths & Weaknesses
            sw_col1, sw_col2 = st.columns(2)
            with sw_col1:
                with st.container(border=True):
                    st.markdown("#### ✅ Strengths")
                    for s in report.get("strengths", []):
                        st.markdown(f"- {s}")
            with sw_col2:
                with st.container(border=True):
                    st.markdown("#### ⚠️ Gaps & Weaknesses")
                    for w in report.get("weaknesses", []):
                        st.markdown(f"- {w}")

            # Missing Skills section (Explicitly requested)
            missing = report.get("missing_skills", [])
            if missing:
                with st.container(border=True):
                    st.markdown("#### 🎯 Missing Skills to Learn")
                    m_tags = " ".join(f'<span class="tag tag-missing">{s}</span>' for s in missing)
                    st.markdown(m_tags, unsafe_allow_html=True)
            
            # Soft Skills & Cultural Fit
            sc_col1, sc_col2 = st.columns(2)
            with sc_col1:
                with st.container(border=True):
                    st.markdown("#### 🧱 Soft Skills Analysis")
                    st.write(report.get("soft_skills_analysis", "Not available."))
            with sc_col2:
                with st.container(border=True):
                    st.markdown("#### 🏭 Cultural Fit")
                    st.write(report.get("cultural_fit", "Not available."))

            # Recommendations (Renamed as requested)
            with st.expander("💡 Suggested Improvements & Learning Path", expanded=True):
                for i, rec in enumerate(report.get("recommendations", []), 1):
                    st.markdown(f"**{i}.** {rec}")

            # Interview Questions
            with st.expander("🎯 Suggested Interview Questions", expanded=False):
                for i, q in enumerate(report.get("interview_questions", []), 1):
                    st.markdown(f"**Q{i}.** {q}")

        # ── ATS SCORE REPORT (always shows when analysis is done) ─────────────
        ats_result = st.session_state.get("ats_result")
        if ats_result and ats_result.get("status") == "success":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 ATS Score Report")

            ats_score = ats_result["ats_score"]
            grade     = ats_result.get("grade", "")
            if ats_score >= 80:
                ats_color, grade_icon = "#10b981", "✅"
            elif ats_score >= 60:
                ats_color, grade_icon = "#f59e0b", "🟡"
            elif ats_score >= 40:
                ats_color, grade_icon = "#f97316", "🟠"
            else:
                ats_color, grade_icon = "#ef4444", "🔴"

            # ATS score KPIs
            ats_k1, ats_k2, ats_k4 = st.columns(3)
            ats_k1.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">ATS Score</div>
                    <div class="metric-value" style="color:{ats_color}">{ats_score}</div>
                    <div style="color:#64748b;font-weight:600">{grade_icon} {grade}</div>
                </div>
            """, unsafe_allow_html=True)
            ats_k2.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Section Score</div>
                    <div class="metric-value" style="font-size:1.8rem">{ats_result['section_score']}</div>
                    <div style="color:#64748b;font-weight:600">Weight: 60%</div>
                </div>
            """, unsafe_allow_html=True)
            ats_k4.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Structure Score</div>
                    <div class="metric-value" style="font-size:1.8rem">{ats_result['structure_score']:.1f}</div>
                    <div style="color:#64748b;font-weight:600">Weight: 40%</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Formatting checklist
            cl_col, kw_col = st.columns(2)
            with cl_col:
                with st.container(border=True):
                    st.markdown("#### ☑️ Formatting Checklist")
                    checklist = ats_result.get("formatting_checklist", {})
                    for section, label in {
                        "Skills": "skills", "Experience": "experience",
                        "Education": "education", "Projects": "projects"
                    }.items():
                        is_present = checklist.get("sections_present", {}).get(label, False)
                        icon = "✅" if is_present else "❌"
                        st.markdown(f"{icon} **{section}** section")
                    st.markdown("---")
                    struct_ok = checklist.get("structure_valid", False)
                    struct_icon = "✅" if struct_ok else "⚠️"
                    st.markdown(f"{struct_icon} **Chronological** order")

            with kw_col:
                with st.container(border=True):
                    st.markdown("#### 🎯 Match Integrity")
                    st.info(
                        "Candidate's professional profile is cross-verified against "
                        "Job Description semantics. Direct keyword lists are hidden "
                        "to emphasize depth over buzzword density."
                    )

            # ATS Suggestions
            with st.expander("💡 ATS Improvement Suggestions", expanded=True):
                for s in ats_result.get("suggestions", []):
                    st.markdown(s)

        # ── FRAUD INTELLIGENCE REPORT ──────────────────────────────────
        fraud_result = st.session_state.get("fraud_result")
        if fraud_result and fraud_result.get("status") == "success":
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🕵️ Fraud & Integrity Intelligence")

            risk = fraud_result.get("fraud_risk", "Unknown")
            score = fraud_result.get("risk_score", 0)
            risk_colors = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
            risk_icons  = {"Low": "✅", "Medium": "🟠", "High": "🔴"}
            risk_color = risk_colors.get(risk, "#64748b")
            risk_icon  = risk_icons.get(risk, "❓")

            fraud_k1, fraud_k2, fraud_k3 = st.columns(3)
            fraud_k1.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Fraud Risk</div>
                    <div class="metric-value" style="color:{risk_color}">{risk_icon} {risk}</div>
                    <div style="color:#64748b;font-weight:600">Integrity Level</div>
                </div>
            """, unsafe_allow_html=True)
            fraud_k2.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Risk Score</div>
                    <div class="metric-value" style="color:{risk_color};font-size:2rem">{score}</div>
                    <div style="color:#64748b;font-weight:600">Out of 100</div>
                </div>
            """, unsafe_allow_html=True)
            fraud_k3.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Flags Detected</div>
                    <div class="metric-value" style="font-size:2rem">{len(fraud_result.get('flags', []))}</div>
                    <div style="color:#64748b;font-weight:600">Indicators</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Summary banner
            st.info(fraud_result.get("summary", ""))

            # Flagged issues by category
            flag_details = fraud_result.get("flag_details", {})
            has_flags = any(v for v in flag_details.values())

            if has_flags:
                with st.expander("🚩 Detected Fraud Indicators", expanded=True):
                    for category, flags_list in flag_details.items():
                        if flags_list:
                            st.markdown(f"**{category}**")
                            for f in flags_list:
                                st.markdown(f"> {f}")
            else:
                st.success("✅ All fraud checks passed. No suspicious patterns detected.")

            # LLM Verdict (optional, only in AI mode)
            llm_verdict = fraud_result.get("llm_verdict")
            if llm_verdict:
                with st.expander("🧠 AI Fraud Analysis Verdict", expanded=True):
                    st.write(llm_verdict)


        # ── STAGE 2 TRANSITION ──────────────────────────────
        if st.session_state.get("stage_1_complete"):
            render_interview_button()
            
        # Also show recommendations for improvement regardless of stage
        render_recommendations()


    else:
        # ── PLACEHOLDER (no analysis yet) ─────────────────────────────
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("""
                <div class="metric-card-container">
                    <div class="metric-label">Semantic Alignment</div>
                    <div class="metric-value">—</div>
                    <div style="color: #94a3b8; font-weight: 600;">Awaiting Analysis</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown("""
                <div class="metric-card-container">
                    <div class="metric-label">Skill Match</div>
                    <div class="metric-value">—</div>
                    <div style="color: #94a3b8; font-weight: 600;">Awaiting Analysis</div>
                </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown("""
                <div class="metric-card-container">
                    <div class="metric-label">Overall Score</div>
                    <div class="metric-value">—</div>
                    <div style="color: #94a3b8; font-weight: 600;">Awaiting Analysis</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("👆 Upload a resume, enter a Job Description, then click **Analyze Candidate** to see results.")

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_vector_store(parsed: dict) -> bool:
    """Rebuilds FAISS index if missing. Returns True on success."""
    if not INDEX_PATH.exists():
        with st.spinner("🔢 Rebuilding vector store..."):
            emb_result = process_embeddings(parsed, source_label="resume")
            st.session_state["embedding_result"] = emb_result
            if emb_result["status"] != "success":
                st.error(f"❌ Could not build vector store: {emb_result['message']}")
                return False
    return True


def analyze_candidate(parsed: dict, jd_text: str):
    """
    Option 1 — Fast local analysis.
    Runs similarity matching only. No external API calls.
    """
    if not _ensure_vector_store(parsed):
        return

    candidate_skills = parsed.get("skills", [])
    with st.spinner("🔍 Matching resume against Job Description..."):
        match_result = match_resume_to_jd(
            jd_text=jd_text,
            candidate_skills=candidate_skills,
            k=20,
        )
    st.session_state["match_result"] = match_result
    st.session_state["llm_result"]   = None  # Ensure no stale AI data shown

    # ATS Scoring (always local, no API)
    with st.spinner("📊 Running ATS analysis..."):
        ats_result = run_ats_analysis(parsed, jd_text)
    st.session_state["ats_result"] = ats_result

    # Fraud Detection (always local, no API)
    raw_text = st.session_state.get("resume_result", {}).get("cleaned_text", "")
    with st.spinner("🕵️ Running fraud detection..."):
        fraud_result = generate_fraud_report(parsed, raw_text)
    st.session_state["fraud_result"] = fraud_result

    # Recommendation Engine (always local, no API)
    jd_skills = match_result.get("jd_skills", [])
    with st.spinner("🎯 Building skill recommendations..."):
        rec_result = build_recommendation_output(candidate_skills, jd_skills)
    st.session_state["rec_result"] = rec_result

    if match_result["status"] == "success":
        st.session_state["stage_1_complete"] = True
        st.toast(match_result["message"], icon="✅")
        st.rerun()
    else:
        st.warning(match_result["message"])


def analyze_with_recommendation(parsed: dict, jd_text: str):
    """
    Option 2 — Full AI analysis with recommendations.
    Runs similarity matching THEN calls Gemini/OpenAI for deep insights.
    """
    if not _ensure_vector_store(parsed):
        return

    # Determine API key and provider
    api_key = _GEMINI_API_KEY if _LLM_PROVIDER == PROVIDER_GEMINI else _OPENAI_API_KEY

    if not api_key:
        st.error(
            "❌ API key not found. Please add `GEMINI_API_KEY` to your `.env` file."
        )
        return

    # Step 1: Similarity Matching (same as Option 1)
    candidate_skills = parsed.get("skills", [])
    with st.spinner("🔍 Matching resume against Job Description..."):
        match_result = match_resume_to_jd(
            jd_text=jd_text,
            candidate_skills=candidate_skills,
            k=20,
        )
    st.session_state["match_result"] = match_result

    # Step 2: LLM Deep Analysis
    resume_text = st.session_state.get("resume_result", {}).get("cleaned_text", "")
    with st.spinner("🧠 Analyzing with AI... (this may take 10-20 seconds)"):
        llm_result = generate_analysis_report(
            resume_text=resume_text,
            jd_text=jd_text,
            provider=_LLM_PROVIDER,
            api_key=api_key,
            structured_data=parsed,
        )
    st.session_state["llm_result"] = llm_result

    # Step 3: ATS Scoring (always local, regardless of LLM outcome)
    with st.spinner("📊 Running ATS analysis..."):
        ats_result = run_ats_analysis(parsed, jd_text)
    st.session_state["ats_result"] = ats_result

    # Step 4: Fraud Detection (with optional LLM validation in AI mode)
    raw_text = st.session_state.get("resume_result", {}).get("cleaned_text", "")
    with st.spinner("🕵️ Running fraud detection..."):
        fraud_result = generate_fraud_report(
            parsed, raw_text,
            api_key=api_key,
            provider=_LLM_PROVIDER,
            use_llm=True,
        )
    st.session_state["fraud_result"] = fraud_result

    # Step 5: Recommendation Engine (with optional LLM career plan)
    jd_skills = match_result.get("jd_skills", [])
    with st.spinner("🎯 Building skill recommendations..."):
        rec_result = build_recommendation_output(
            candidate_skills, jd_skills,
            api_key=api_key,
            provider=_LLM_PROVIDER,
            use_llm=True,
        )
    st.session_state["rec_result"] = rec_result

    if llm_result["status"] == "success":
        st.toast(llm_result["message"], icon="🧠")
        # Only rerun if BOTH matching and LLM succeeded
        if match_result["status"] == "success":
            st.session_state["stage_1_complete"] = True
            st.toast(match_result["message"], icon="✅")
            st.rerun()

    if match_result["status"] != "success":
        st.warning(match_result["message"])


# --- INTERVIEW VIEW WRAPPER ---

def render_screening_page():
    """Renders the Stage 1: Resume Screening & Matching interface."""
    display_header()
    
    # ── Main Content ─────────────────────────────────────────────────────
    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        display_input_workspace()

        # ── Analysis Mode Selector ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Select Analysis Type:",
            [
                "⚡ Analyze Candidate",
                "🧠 Analyze with AI Recommendations",
            ],
            horizontal=True,
            help=(
                "**Analyze Candidate**: Fast local matching — no API needed.\n\n"
                "**Analyze with AI Recommendations**: Deeper LLM insights using Gemini. Requires .env API key."
            ),
        )

        # ── Action Button ──────────────────────────────────────────────────
        btn_label = (
            "⚡ Analyze Candidate"
            if analysis_mode.startswith("⚡")
            else "🧠 Analyze with AI Recommendations"
        )
        if st.button(btn_label, use_container_width=True):
            parsed  = st.session_state.get("parsed_resume")
            jd_text = st.session_state.get("jd_text", "").strip()

            if not parsed:
                st.error("❌ Please upload a resume first.")
            elif not jd_text:
                st.error("❌ Please enter or upload a Job Description.")
            else:
                try:
                    if analysis_mode.startswith("⚡"):
                        analyze_candidate(parsed, jd_text)
                    else:
                        analyze_with_recommendation(parsed, jd_text)
                except Exception as e:
                    st.error(f"❌ Analysis error: {str(e)}")

        render_main_dashboard()


def render_main_dashboard():
    display_dashboard()


# --- APP ROUTER ---
def main():
    # ── Session state initialisation ─────────────────────────────────────────
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "role_selection"
    if "jd_text" not in st.session_state:
        st.session_state["jd_text"] = ""
    if "stage_1_complete" not in st.session_state:
        st.session_state["stage_1_complete"] = False
    if "verification_complete" not in st.session_state:
        st.session_state["verification_complete"] = False
    if "GEMINI_API_KEY" not in st.session_state:
        st.session_state["GEMINI_API_KEY"] = _GEMINI_API_KEY
    if "ASSEMBLYAI_API_KEY" not in st.session_state:
        st.session_state["ASSEMBLYAI_API_KEY"] = _ASSEMBLYAI_API_KEY

    # Ensure DB schema exists on every startup
    init_db()
    init_auth_db()

    current_page = st.session_state["current_page"]
    role = st.session_state.get("user_role")

    # ── Sidebar: Shared Content ──────────────────────────────────────────
    # Only show sidebar if logged in (except on role_selection/login)
    if current_page not in ["role_selection", "login"]:
        with st.sidebar:
            st.markdown("### 🧠 HireMind AI")
            st.caption("Autonomous HR Operating System")
            st.markdown("---")
            
            if role == "admin":
                if st.button("📊 Admin Dashboard", use_container_width=True):
                    st.session_state["current_page"] = "admin_dashboard"
                    st.rerun()
                if st.button("🔍 Screening Tool", use_container_width=True):
                    st.session_state["current_page"] = "screening"
                    st.rerun()
            elif role == "candidate":
                if st.button("🏠 Candidate Home", use_container_width=True):
                    st.session_state["current_page"] = "candidate_dashboard"
                    st.rerun()
            
            st.markdown("---")
            st.markdown("#### ⚙️ System Status")
            st.markdown("✅ Modules Operational")
            if _GEMINI_API_KEY:
                st.markdown("✅ Gemini AI — Ready")
            else:
                st.markdown("⚠️ LLM — No API key")

            st.markdown("---")
            if st.button("System Logout"):
                from modules.auth_ui import logout
                logout()

    # ── Central Router ───────────────────────────────────────────────────
    if current_page == "role_selection":
        render_role_selection()

    elif current_page == "login":
        render_login_page()

    elif current_page == "candidate_dashboard":
        render_candidate_dashboard()

    elif current_page == "admin_dashboard":
        render_admin_dashboard()

    elif current_page == "screening":
        local_css()
        render_screening_page()

    elif current_page == "verification":
        local_css()
        render_verification_page()

    elif current_page == "interview_mode":
        local_css()
        render_mode_selection()

    elif current_page == "interview":
        local_css()
        render_interview_ui()

    elif current_page == "voice_interview":
        local_css()
        render_voice_interview()

    elif current_page == "final_report":
        local_css()
        render_combined_report()

    else:
        # Fallback
        st.session_state["current_page"] = "role_selection"
        st.rerun()

# Logic extracted to render_screening_page()



if __name__ == "__main__":
    main()
