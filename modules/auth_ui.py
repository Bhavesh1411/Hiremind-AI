import streamlit as st
from modules.auth_db import (
    authenticate_user, create_user, get_all_jobs, add_job, apply_for_job
)
from modules.candidate_db import (
    get_all_candidates, get_job_applicants, update_hiring_status
)
from modules.email_service import send_hiring_email


def render_role_selection():
    # --- ADVANCED FUTURISTIC UI CSS ---
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Reset & Base */
    .stApp {
        background: #020617 !important;
        color: #f8fafc !important;
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Immersive Background with Neural Effects */
    .bg-wrapper {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: 
            radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 1) 0%, #020617 100%);
        z-index: -1;
        overflow: hidden;
    }
    
    .neural-grid {
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            radial-gradient(rgba(59, 130, 246, 0.1) 1px, transparent 1px),
            radial-gradient(rgba(139, 92, 246, 0.1) 1px, transparent 1px);
        background-size: 40px 40px;
        background-position: 0 0, 20px 20px;
        opacity: 0.3;
    }

    .glow-overlay {
        position: absolute;
        top: -10%; left: -10%; width: 120%; height: 120%;
        background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.15) 0%, transparent 50%);
        pointer-events: none;
    }

    /* Top Navigation Area */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 4rem;
        width: 100%;
    }
    .brand-area {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.4);
    }
    .brand-text {
        display: flex;
        flex-direction: column;
    }
    .brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        line-height: 1;
    }
    .brand-tagline {
        font-size: 0.75rem;
        color: #94a3b8;
        letter-spacing: 0.05em;
    }
    
    .status-badge-premium {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 8px 20px;
        border-radius: 100px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #4ade80;
        backdrop-filter: blur(10px);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 12px #22c55e;
        animation: pulse-glow 2s infinite;
    }
    @keyframes pulse-glow {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.1); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }

    /* Hero Section - Reference Match */
    .hero-main {
        text-align: center;
        padding: 5rem 1rem 3rem;
        max-width: 900px;
        margin: 0 auto;
    }
    .hero-welcome {
        color: #818cf8;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-shadow: 0 0 15px rgba(129, 140, 248, 0.3);
    }
    .hero-title-ref {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        background: linear-gradient(to right, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }
    .hero-subtitle-ref {
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        width: 100%;
    }
    .hero-desc-ref {
        color: #94a3b8;
        font-size: 1.15rem;
        line-height: 1.6;
        max-width: 700px;
        margin: 0 auto 2.5rem;
        text-align: center;
    }

    /* Feature Strip */
    .feature-strip {
        display: inline-flex;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 12px 30px;
        gap: 30px;
        backdrop-filter: blur(10px);
    }
    .strip-item {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 0.9rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    .strip-divider {
        width: 1px;
        height: 20px;
        background: rgba(255, 255, 255, 0.1);
    }

    /* Portal Cards - Reference Match */
    .portals-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2.5rem;
        max-width: 1100px;
        margin: 4rem auto;
        padding: 0 2rem;
    }
    .ref-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 3rem;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: all 0.4s ease;
        backdrop-filter: blur(20px);
    }
    .ref-card:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(59, 130, 246, 0.4);
    }
    .ref-card.admin:hover {
        border-color: rgba(139, 92, 246, 0.4);
    }
    
    .card-icon-round {
        width: 60px;
        height: 60px;
        background: rgba(59, 130, 246, 0.15);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 2rem;
        color: #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }
    .admin .card-icon-round {
        background: rgba(139, 92, 246, 0.15);
        color: #8b5cf6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
    }

    .ref-card-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
    }
    .ref-card-desc {
        color: #94a3b8;
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 2.5rem;
    }

    /* Card CTA Buttons */
    .cta-btn-ref {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background: linear-gradient(90deg, #1d4ed8, #3b82f6);
        color: white;
        padding: 16px 32px;
        border-radius: 14px;
        font-weight: 700;
        font-size: 1.1rem;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 0 10px 20px rgba(29, 78, 216, 0.3);
    }
    .admin .cta-btn-ref {
        background: linear-gradient(90deg, #6d28d9, #8b5cf6);
        box-shadow: 0 10px 20px rgba(109, 40, 217, 0.3);
    }
    
    /* Feature Section Below Cards */
    .choose-us-section {
        max-width: 1200px;
        margin: 6rem auto;
        padding: 4rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 32px;
        text-align: center;
    }
    .section-title-line {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 4rem;
    }
    .title-line {
        height: 1px;
        width: 100px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    .section-title-text {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: white;
    }
    
    .features-row-ref {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
    }
    .feat-card-ref {
        flex: 1;
        text-align: center;
    }
    .feat-icon-ref {
        width: 50px;
        height: 50px;
        margin: 0 auto 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .feat-name-ref {
        font-size: 1rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        display: block;
    }
    .feat-desc-ref {
        font-size: 0.8rem;
        color: #64748b;
        line-height: 1.4;
    }

    /* Footer - Reference Match */
    .footer-ref {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 4rem;
    }
    .footer-left {
        display: flex;
        align-items: center;
        gap: 15px;
        color: #f8fafc;
        font-size: 0.9rem;
    }
    .footer-center {
        color: #f8fafc;
        font-size: 0.85rem;
    }
    .footer-right {
        color: #f8fafc;
        font-size: 0.9rem;
    }
    .heart-icon { color: #ec4899; }

    /* Hide Streamlit Elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Button Overrides */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
</style>

<div class="bg-wrapper">
    <div class="neural-grid"></div>
    <div class="glow-overlay"></div>
</div>
""", unsafe_allow_html=True)

    # --- TOP NAVIGATION ---
    st.markdown("""
        <div class="top-nav">
            <div class="brand-area">
                <div class="brand-logo">🧠</div>
                <div class="brand-text">
                    <span class="brand-name">HireMind AI</span>
                    <span class="brand-tagline">Autonomous HR Operating System</span>
                </div>
            </div>
            <div class="status-badge-premium">
                <div class="pulse-dot"></div>
                AI Engine Active
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- HERO SECTION ---
    st.markdown("""
        <div class="hero-main">
            <div class="hero-welcome">Welcome to</div>
            <h1 class="hero-title-ref">HireMind AI</h1>
            <div class="hero-subtitle-ref">AI-Powered Autonomous Hiring Platform</div>
            <p class="hero-desc-ref">
                Streamline your recruitment with AI-driven resume screening, live interviews, 
                and intelligent fraud detection in one unified platform.
            </p>
            <div class="feature-strip">
                <div class="strip-item">✨ AI-Powered</div>
                <div class="strip-divider"></div>
                <div class="strip-item">🛡️ Secure</div>
                <div class="strip-divider"></div>
                <div class="strip-item">🧠 Smart</div>
                <div class="strip-divider"></div>
                <div class="strip-item">📈 Scalable</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- PORTAL CARDS ---
    col_cards = st.columns([1, 1])

    with col_cards[0]:
        st.markdown("""
            <div class="ref-card candidate">
                <div class="card-icon-round">👤</div>
                <div class="ref-card-title">Candidate</div>
                <p class="ref-card-desc">Find jobs, apply with your resume, and complete AI interviews in minutes.</p>
        """, unsafe_allow_html=True)
        if st.button("Access Candidate Portal →", key="btn_cand_ref"):
            st.session_state["auth_role"] = "candidate"
            st.session_state["current_page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_cards[1]:
        st.markdown("""
            <div class="ref-card admin">
                <div class="card-icon-round">💼</div>
                <div class="ref-card-title">Admin / Recruiter</div>
                <p class="ref-card-desc">Manage job postings, review candidates, and monitor system performance.</p>
        """, unsafe_allow_html=True)
        if st.button("Access Admin Dashboard →", key="btn_admin_ref"):
            st.session_state["auth_role"] = "admin"
            st.session_state["current_page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- WHY CHOOSE US SECTION ---
    st.markdown("""
        <div class="choose-us-section">
            <div class="section-title-line">
                <div class="title-line"></div>
                <span class="section-title-text">Why Choose HireMind AI?</span>
                <div class="title-line"></div>
            </div>
            <div class="features-row-ref">
                <div class="feat-card-ref">
                    <div class="feat-icon-ref">🎯</div>
                    <span class="feat-name-ref">AI Resume Screening</span>
                    <span class="feat-desc-ref">Smart matching with context understanding</span>
                </div>
                <div class="feat-card-ref">
                    <div class="feat-icon-ref">🛡️</div>
                    <span class="feat-name-ref">Fraud Detection</span>
                    <span class="feat-desc-ref">Advanced verification and anti-cheating</span>
                </div>
                <div class="feat-card-ref">
                    <div class="feat-icon-ref">🎙️</div>
                    <span class="feat-name-ref">Voice AI Interview</span>
                    <span class="feat-desc-ref">Natural conversations with AI interviewer</span>
                </div>
                <div class="feat-card-ref">
                    <div class="feat-icon-ref">📸</div>
                    <span class="feat-name-ref">Live Proctoring</span>
                    <span class="feat-desc-ref">Real-time monitoring and face verification</span>
                </div>
                <div class="feat-card-ref">
                    <div class="feat-icon-ref">📊</div>
                    <span class="feat-name-ref">Analytics Dashboard</span>
                    <span class="feat-desc-ref">Data-driven insights for better hiring</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
        <div class="footer-ref">
            <div class="footer-left">
                <span>🔒 Secure. Intelligent. Autonomous.</span>
            </div>
            <div class="footer-center">
                © 2024 HireMind AI. All rights reserved.
            </div>
            <div class="footer-right">
                Made with <span class="heart-icon">❤️</span> for a better future
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_login_page():
    role = st.session_state.get("auth_role", "candidate")
    
    st.markdown("""
<style>
    /* Global fixes for the login page */
    .stApp { background: #020617 !important; }
    
    /* Hide the default display_header if it's there */
    .header-container { display: none !important; }

    /* The target columns wrapper */
    div[data-testid="stHorizontalBlock"]:has(.right-heading) {
        gap: 0 !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 32px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 40px rgba(139, 92, 246, 0.1) !important;
        overflow: hidden !important;
        max-width: 1000px !important;
        margin: 4rem auto !important;
    }
    
    /* Left Column (Form) */
    div[data-testid="stHorizontalBlock"]:has(.right-heading) > div[data-testid="column"]:nth-of-type(1) {
        padding: 3rem 4rem !important;
        background: transparent !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }

    /* Right Column (Illustration) */
    div[data-testid="stHorizontalBlock"]:has(.right-heading) > div[data-testid="column"]:nth-of-type(2) {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(2, 6, 23, 0.8) 100%) !important;
        padding: 3rem 3rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* Typography */
    .login-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    .login-heading .highlight {
        background: linear-gradient(to right, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .login-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Form Overrides */
    div[data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    .stTextInput label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    .stTextInput input {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        padding: 0.7rem 1rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stTextInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.2) !important;
    }

    /* Sign In Buttons */
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        padding: 0.8rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFormSubmitButton"]:first-of-type > button {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(109, 40, 217, 0.3) !important;
    }
    div[data-testid="stFormSubmitButton"]:first-of-type > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 25px rgba(109, 40, 217, 0.5) !important;
    }
    
    div[data-testid="stFormSubmitButton"]:last-of-type > button {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #94a3b8 !important;
    }
    div[data-testid="stFormSubmitButton"]:last-of-type > button:hover {
        border-color: white !important;
        color: white !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    /* Right Panel Elements */
    .right-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .right-desc {
        color: #94a3b8;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 2rem;
    }
    .right-features {
        text-align: left;
        color: #e2e8f0;
        font-size: 0.95rem;
        font-weight: 500;
        background: rgba(0,0,0,0.2);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.05);
        width: 100%;
    }
    .right-features div {
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .right-features div:last-child {
        margin-bottom: 0;
    }
    
    /* Right Panel Create Account Button */
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid rgba(139, 92, 246, 0.5) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        margin-top: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    button[kind="secondary"]:hover {
        background: rgba(139, 92, 246, 0.1) !important;
        border-color: #8b5cf6 !important;
    }
    
    /* Hide the Streamlit Expander styling for signup */
    div[data-testid="stExpander"] {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown(f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom: 2rem;">
                <div style="width:40px;height:40px;background:linear-gradient(135deg, #ec4899, #8b5cf6);border-radius:10px;display:flex;align-items:center;justify-content:center;box-shadow:0 0 15px rgba(236,72,153,0.4);">
                    <span style="font-size:1.3rem;">🧠</span>
                </div>
                <div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:1.2rem;font-weight:700;color:white;line-height:1;">HireMind AI</div>
                    <div style="font-size:0.75rem;color:#94a3b8;letter-spacing:0.05em;">Autonomous HR OS</div>
                </div>
            </div>
            <h1 class="login-heading">Welcome <span class="highlight">back</span></h1>
            <p class="login-subtitle">Sign in to your {role} account</p>
        """, unsafe_allow_html=True)
        
        with st.form("login_form_premium"):
            email = st.text_input("Email Address", placeholder="alex@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.2rem; margin-bottom:1.5rem;">
                    <label style="color:#94a3b8; font-size:0.85rem; display:flex; align-items:center; gap:6px; cursor:pointer;">
                        <input type="checkbox" checked style="accent-color:#8b5cf6; width:14px; height:14px;"> Remember me
                    </label>
                    <a href="#" style="color:#8b5cf6; text-decoration:none; font-size:0.85rem; font-weight:500;">Forgot password?</a>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                submit = st.form_submit_button("Sign In →", use_container_width=True)
            with c2:
                cancel = st.form_submit_button("Cancel", use_container_width=True)
            
            if submit:
                user = authenticate_user(email, password)
                if user and user["role"] == role:
                    st.session_state["logged_in"] = True
                    st.session_state["user_data"] = user
                    st.session_state["user_role"] = role
                    st.session_state["current_page"] = f"{role}_dashboard"
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            
            if cancel:
                st.session_state["current_page"] = "role_selection"
                st.rerun()
                
        if role == "candidate":
            st.markdown("""
                <div style="display:flex; align-items:center; gap:10px; margin: 2rem 0 1rem;">
                    <div style="flex:1; height:1px; background:rgba(255,255,255,0.1);"></div>
                    <div style="color:#64748b; font-size:0.85rem; font-weight:500;">OR</div>
                    <div style="flex:1; height:1px; background:rgba(255,255,255,0.1);"></div>
                </div>
            """, unsafe_allow_html=True)
            render_signup("candidate")

    with right_col:
        st.markdown(f"""
            <img src="file:///C:/Users/LENOVO/.gemini/antigravity/brain/29d7a03e-26a0-4d00-bd20-6b40eca55824/ai_login_illustration_1778144662001.png" style="width:100%; max-width:280px; border-radius:16px; box-shadow:0 0 40px rgba(139,92,246,0.25); margin-bottom:1rem; border:1px solid rgba(255,255,255,0.1);">
            <h2 class="right-heading">New here?</h2>
            <p class="right-desc">Create an account to explore AI-powered hiring, apply for jobs, and more.</p>
            
            <div class="right-features">
                <div><span style="font-size:1.2rem; color:#a855f7;">🧠</span> <span>AI-Powered Screening</span></div>
                <div><span style="font-size:1.2rem; color:#3b82f6;">⚡</span> <span>Smart Job Matching</span></div>
                <div><span style="font-size:1.2rem; color:#22c55e;">🛡️</span> <span>Secure & Private</span></div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Create Account", key="create_account_btn", use_container_width=True):
            pass


def render_login(role):
    with st.form(f"login_form_{role}"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = authenticate_user(email, password)
            if user:
                if user["role"] == role:
                    st.session_state["logged_in"] = True
                    st.session_state["user_data"] = user
                    st.session_state["user_role"] = role
                    st.success(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error(f"Account exists but role is not {role}.")
            else:
                st.error("Invalid email or password.")

def render_signup(role):
    with st.expander("✨ Create your Account", expanded=False):
        with st.form(f"signup_form_{role}"):
            st.markdown("""
                <style>
                    .stForm { background: transparent !important; border: none !important; }
                </style>
            """, unsafe_allow_html=True)
            
            name = st.text_input("Full Name", placeholder="e.g. Alex Rivera")
            email = st.text_input("Email Address", placeholder="alex@example.com")
            
            c1, c2 = st.columns(2)
            with c1:
                password = st.text_input("Password", type="password", placeholder="••••••••")
            with c2:
                confirm = st.text_input("Confirm", type="password", placeholder="••••••••")
                
            if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                if password != confirm:
                    st.error("Passwords do not match.")
                elif not name or not email or not password:
                    st.error("Please fill all fields.")
                else:
                    user_id = create_user(name, email, password, role)
                    if user_id:
                        st.success("Account created! You can now sign in above.")
                    else:
                        st.error("Account already exists with this email.")

def render_candidate_dashboard():
    st.title("🎓 Candidate Dashboard")
    st.write(f"Welcome, **{st.session_state['user_data']['name']}**!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Available Jobs")
        jobs = get_all_jobs()
        if not jobs:
            st.info("No jobs posted yet. Check back soon!")
        else:
            for job in jobs:
                with st.container(border=True):
                    col_j1, col_j2 = st.columns([3, 1])
                    with col_j1:
                        st.markdown(f"**{job['title']}**")
                        st.caption(f"Posted on: {job['created_at'][:10]}")
                        st.write(job['description'][:150] + "...")
                        with st.expander("View Full JD"):
                            st.write(job['description'])
                    with col_j2:
                        if st.button("Apply Now", key=f"apply_{job['id']}", use_container_width=True):
                            # Store selected job details
                            st.session_state["selected_job"] = job
                            # Load JD into existing system state (single source of truth)
                            st.session_state["jd_text"] = job['description']
                            st.session_state["current_job_id"] = job['id']
                            # Set navigation state
                            st.session_state["current_page"] = "screening"
                            st.session_state["view"] = "dashboard" # Backward compatibility
                            st.rerun()
    
    with col2:
        st.subheader("📌 Your Status")
        st.info("Feature coming soon: Track your application status.")
        
    if st.sidebar.button("Logout"):
        logout()

def render_admin_dashboard():
    # --- AUTH CHECK ---
    if st.session_state.get("user_role") != "admin":
        st.error("Access Denied. Admins only.")
        if st.button("← Support Portal"):
            st.session_state["current_page"] = "role_selection"
            st.rerun()
        return

    st.markdown("""
        <style>
            .admin-header { background: #0f172a; color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; }
            .candidate-card { background: white; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; transition: all 0.3s ease; }
            .candidate-card:hover { border-color: #3b82f6; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
            .score-badge { background: #eff6ff; color: #1e40af; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; }
            .hired-badge { background: #f0fdf4; color: #166534; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 0.85rem; border: 1px solid #86efac; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="admin-header">
            <h1 style="margin:0; font-size:1.8rem;">💼 Hiring Control Center</h1>
            <p style="margin:0.2rem 0 0; color:#94a3b8;">Administrator: {st.session_state['user_data']['name']}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- TOP NAVIGATION & JOB SELECTION ---
    nav_col1, nav_col2, nav_col3 = st.columns([2, 1, 1])
    
    with nav_col1:
        st.subheader("🚀 Active Roles")
        jobs = get_all_jobs()
        if not jobs:
            st.info("No jobs posted yet.")
            selected_job = None
        else:
            job_titles = [j['title'] for j in jobs]
            selected_job_title = st.selectbox("Select Job to Review:", ["-- Select Job --"] + job_titles, label_visibility="collapsed")
            selected_job = next((j for j in jobs if j['title'] == selected_job_title), None)
            if selected_job:
                st.session_state["admin_view"] = "applicants"

    with nav_col2:
        st.write("") # Padding
        if st.button("➕ Create New Job", use_container_width=True):
            st.session_state["admin_view"] = "add_job"
    
    with nav_col3:
        st.write("") # Padding
        if st.button("📋 System Overview", use_container_width=True):
            st.session_state["admin_view"] = "overview"

    st.markdown("---")


    # --- MAIN CONTENT ---
    view = st.session_state.get("admin_view", "overview")

    if view == "add_job":
        st.subheader("🆕 Create New Job Posting")
        with st.form("add_job_form_main"):
            title = st.text_input("Job Title")
            description = st.text_area("Detailed Job Description", height=300)
            if st.form_submit_button("Publish Job Posting", type="primary"):
                if title and description:
                    add_job(title, description, st.session_state['user_data']['email'])
                    st.success(f"Job '{title}' published!")
                    st.session_state["admin_view"] = "overview"
                    st.rerun()
                else:
                    st.error("Missing fields.")
    
    elif selected_job:
        st.subheader(f"👥 Applicants for {selected_job['title']}")
        applicants = get_job_applicants(selected_job['id'])
        
        if not applicants:
            st.info("No applicants have completed the interview for this role yet.")
        else:
            # Ranking & Score Processing
            processed_applicants = []
            for app in applicants:
                # Normalization: ATS (0-100) + Interview (Avg 0-10 normalized to 100)
                ats_score = app.get("ats_score", 0.0)
                interview_score = app.get("final_score", 0.0) # This was normalized to 100 in voice_ui.py
                
                # [DEBUG LOGGING]
                print(f"[FETCH - ADMIN] Candidate: {app['full_name']} | S1 (ATS): {ats_score}% | S2 (INT): {interview_score}%")
                
                final_score = ats_score + interview_score

                processed_applicants.append({**app, "total_score": final_score})

            # Sort by total score DESC
            ranked_applicants = sorted(processed_applicants, key=lambda x: x["total_score"], reverse=True)

            # Table Header
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1.5])
            with col1: st.caption("CANDIDATE")
            with col2: st.caption("STAGE 1 (ATS)")
            with col3: st.caption("STAGE 2 (INT)")
            with col4: st.caption("FINAL / RANK")

            for i, app in enumerate(ranked_applicants):
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1.5])
                    with c1:
                        st.markdown(f"**{app['full_name']}**")
                        if app.get("hiring_status") == "hired":
                            st.markdown('<span class="hired-badge">SELECTED</span>', unsafe_allow_html=True)
                        else:
                            st.caption(app['email'])
                    with c2:
                        st.write(f"{app['ats_score']:.1f}%")
                    with c3:
                        st.write(f"{app['final_score']:.1f}%")
                    with c4:
                        st.markdown(f'<span class="score-badge">{app["total_score"]:.1f} / 200</span>', unsafe_allow_html=True)
                        st.caption(f"Rank #{i+1}")
                    
                    # Actions Row
                    a1, a2, _ = st.columns([1, 1, 3])
                    with a1:
                        if st.button("View Details", key=f"det_{app['id']}"):
                            show_candidate_details(app, selected_job['title'])
                    with a2:
                        if app.get("hiring_status") != "hired":
                            if st.button("🚀 HIRE", key=f"hire_{app['id']}", type="primary"):
                                update_hiring_status(app['id'], "hired")
                                result = send_hiring_email(app['email'], app['full_name'], selected_job['title'])
                                if result["success"]:
                                    st.success(f"Hiring email sent to {app['full_name']}!")
                                else:
                                    st.warning(f"Status updated, but email failed: {result['message']}")
                                st.rerun()
                    st.markdown("---")

    else:
        # DEFAULT OVERVIEW
        st.subheader("📋 System Overview")
        st.write("Select a job from the sidebar to view ranked applicants and manage hiring.")
        
        counts_col1, counts_col2 = st.columns(2)
        with counts_col1:
            st.metric("Total Jobs", len(jobs))
        with counts_col2:
            all_cands = get_all_candidates()
            st.metric("Identity Verified Candidates", len(all_cands))

@st.dialog("Candidate Detailed Report")
def show_candidate_details(app, job_title):
    st.markdown(f"### Report: {app['full_name']}")
    st.caption(f"Role: {job_title} | ID: #{app['id']}")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Contact Information**")
        st.write(f"📧 {app['email']}")
        st.write(f"📞 {app['phone']}")
    with col2:
        st.markdown("**Assessment Timeline**")
        st.write(f"🗓 Completed: {app.get('timestamp', 'N/A')[:10]}")
        st.write(f"📍 Status: {app.get('hiring_status', 'Pending').title()}")

    st.markdown("---")
    st.markdown("#### 📊 Performance Breakdown")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("ATS Score", f"{app['ats_score']:.1f}%")
    m2.metric("Interview Score", f"{app['final_score']:.1f}%")
    m3.metric("Final Weighted", f"{app['total_score']:.1f}", delta=f"Rank #{app.get('rank', 'N/A')}")

    # Add interview feedback if retrieval is needed
    from modules.candidate_db import get_interview_results
    results = get_interview_results(app['id'])
    
    with st.expander("📝 View Raw Interview Answers", expanded=False):
        for ans in results.get("answers", []):
            st.markdown(f"**Q: {ans['question_text']}**")
            st.info(f"Answer: {ans['answer_text']}")
            st.caption(f"Score: {ans['score']} | Eval: {ans['evaluation']}")
            st.markdown("---")

    if st.button("Close Report"):
        st.rerun()

def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_data"] = None
    st.session_state["user_role"] = None
    st.session_state["auth_role"] = None
    st.session_state["current_page"] = "role_selection"
    st.session_state["view"] = "dashboard"
    st.rerun()
