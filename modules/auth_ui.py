import streamlit as st
from modules.auth_db import (
    authenticate_user, create_user, get_all_jobs, add_job, apply_for_job
)
from modules.candidate_db import (
    get_all_candidates, get_job_applicants, update_hiring_status
)
from modules.email_service import send_hiring_email


def render_role_selection():
    # --- PREMIUM LIGHT THEME UI CSS (Matching Reference Image) ---
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap');

    /* Reset & Base */
    .stApp {
        background: #fdfdff !important;
        color: #1e293b !important;
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Subtle Pattern Backgrounds */
    .bg-elements {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        z-index: -1;
        pointer-events: none;
        overflow: hidden;
    }
    .neural-pattern-left {
        position: absolute;
        top: -100px; left: -100px;
        width: 600px; height: 600px;
        background: radial-gradient(circle at 30% 30%, rgba(59, 130, 246, 0.05), transparent 70%);
        filter: blur(40px);
    }
    .neural-pattern-right {
        position: absolute;
        top: 50px; right: -150px;
        width: 700px; height: 700px;
        background: radial-gradient(circle at 70% 30%, rgba(139, 92, 246, 0.05), transparent 70%);
        filter: blur(50px);
    }

    /* Top Navigation Area */
    .top-nav-ref {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 5rem;
        width: 100%;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(226, 232, 240, 0.5);
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-ref {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #a855f7 0%, #7c3aed 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }
    .brand-content {
        display: flex;
        flex-direction: column;
    }
    .brand-title-ref {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    .brand-subtitle-ref {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .status-badge-active {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 8px 18px;
        border-radius: 100px;
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #166534;
    }
    .dot-pulse {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 0 rgba(34, 197, 94, 0.4);
        animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
        0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    /* Hero Section */
    .hero-container-ref {
        text-align: center;
        padding: 4rem 1rem 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    .hero-welcome-text {
        color: #6366f1;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: none;
    }
    .hero-main-title {
        font-family: 'Outfit', sans-serif;
        font-size: 5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
    }
    .hero-tagline-ref {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1.2rem;
    }
    .hero-para-ref {
        color: #64748b;
        font-size: 1.1rem;
        line-height: 1.6;
        max-width: 750px;
        margin: 0 auto 2.5rem;
    }

    /* Badge Strip */
    .badge-strip-ref {
        display: inline-flex;
        background: white;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 12px;
        padding: 10px 24px;
        gap: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .badge-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9rem;
        color: #475569;
        font-weight: 600;
    }
    .badge-div {
        width: 1px;
        height: 18px;
        background: #e2e8f0;
    }

    /* Portal Grid */
    .portal-grid-ref {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2.5rem;
        max-width: 1050px;
        margin: 4rem auto;
        padding: 0 2rem;
    }
    .card-ref {
        background: white;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 24px;
        padding: 3rem;
        text-align: left;
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.03);
    }
    .card-ref:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.06);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .card-ref.admin-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    /* Background Illustrations for Cards */
    .card-illustration {
        position: absolute;
        right: 0; bottom: 0;
        width: 180px; height: 180px;
        opacity: 0.05;
        pointer-events: none;
    }

    .card-icon-box {
        width: 64px;
        height: 64px;
        background: #f1f5f9;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    .candidate-card .card-icon-box { color: #3b82f6; background: rgba(59, 130, 246, 0.05); }
    .admin-card .card-icon-box { color: #8b5cf6; background: rgba(139, 92, 246, 0.05); }

    .card-title-ref {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }
    .card-desc-ref {
        color: #64748b;
        font-size: 1.1rem;
        line-height: 1.5;
        margin-bottom: 2.5rem;
        max-width: 320px;
    }

    /* Buttons */
    .portal-btn-ref {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        padding: 16px 32px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
        width: 100%;
        border: none;
        cursor: pointer;
    }
    .admin-card .portal-btn-ref {
        background: linear-gradient(90deg, #7c3aed, #a855f7);
        box-shadow: 0 4px 14px rgba(124, 58, 237, 0.3);
    }
    .portal-btn-ref:hover {
        opacity: 0.95;
        transform: scale(1.02);
    }

    /* Features Section */
    .features-section-ref {
        max-width: 1100px;
        margin: 5rem auto;
        padding: 3rem;
        background: white;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 32px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
    }
    .feat-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 3.5rem;
    }
    .feat-line {
        height: 1px;
        flex: 1;
        max-width: 150px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }
    .feat-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    .feat-row-ref {
        display: flex;
        justify-content: space-between;
        gap: 2rem;
    }
    .feat-item-ref {
        flex: 1;
        text-align: center;
    }
    .feat-icon-round {
        width: 56px;
        height: 56px;
        margin: 0 auto 1.2rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        border-radius: 50%;
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    .feat-item-ref:hover .feat-icon-round {
        transform: scale(1.1);
        background: white;
        box-shadow: 0 8px 15px rgba(0,0,0,0.05);
    }
    .feat-name-ref {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.4rem;
        display: block;
    }
    .feat-desc-ref {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.4;
    }

    /* Footer */
    .footer-container-ref {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3rem 5rem;
        border-top: 1px solid #f1f5f9;
        margin-top: 4rem;
        color: #64748b;
        font-size: 0.85rem;
    }
    .footer-info { display: flex; align-items: center; gap: 8px; }
    .heart-red { color: #ef4444; }

    /* Hide Default Elements */
    #MainMenu, footer, header { visibility: hidden; }
    
    /* Streamlit Button Reset */
    .stButton > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        width: 100% !important;
    }
</style>

<div class="bg-elements">
    <div class="neural-pattern-left"></div>
    <div class="neural-pattern-right"></div>
</div>
""", unsafe_allow_html=True)

    # --- TOP NAVIGATION ---
    st.markdown("""
        <div class="top-nav-ref">
            <div class="brand-group">
                <div class="brand-icon-ref">🧠</div>
                <div class="brand-content">
                    <span class="brand-title-ref">HireMind AI</span>
                    <span class="brand-subtitle-ref">Autonomous HR Operating System</span>
                </div>
            </div>
            <div class="status-badge-active">
                <div class="dot-pulse"></div>
                AI Engine Active
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- HERO SECTION ---
    st.markdown("""
        <div class="hero-container-ref">
            <div class="hero-welcome-text">Welcome to</div>
            <h1 class="hero-main-title">HireMind AI</h1>
            <div class="hero-tagline-ref">AI-Powered Autonomous Hiring Platform</div>
            <p class="hero-para-ref">
                Streamline your recruitment with AI-driven resume screening, live interviews, 
                and intelligent fraud detection.
            </p>
            <div class="badge-strip-ref">
                <div class="badge-item"><span style="color:#3b82f6;">✦</span> AI-Powered</div>
                <div class="badge-div"></div>
                <div class="badge-item"><span style="color:#10b981;">🛡️</span> Secure</div>
                <div class="badge-div"></div>
                <div class="badge-item"><span style="color:#8b5cf6;">🧠</span> Smart</div>
                <div class="badge-div"></div>
                <div class="badge-item"><span style="color:#ec4899;">📈</span> Scalable</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- PORTAL CARDS ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div class="card-ref candidate-card">
                <div class="card-icon-box">👤</div>
                <div class="card-title-ref">Candidate</div>
                <p class="card-desc-ref">Find jobs, apply with your resume, and complete AI interviews.</p>
        """, unsafe_allow_html=True)
        if st.button("Access Candidate Portal →", key="portal_cand_ref"):
            st.session_state["auth_role"] = "candidate"
            st.session_state["current_page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="card-ref admin-card">
                <div class="card-icon-box">💼</div>
                <div class="card-title-ref">Admin / Recruiter</div>
                <p class="card-desc-ref">Manage job postings, review candidates, and monitor system performance.</p>
        """, unsafe_allow_html=True)
        if st.button("Access Admin Dashboard →", key="portal_admin_ref"):
            st.session_state["auth_role"] = "admin"
            st.session_state["current_page"] = "login"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- FEATURES SECTION ---
    st.markdown("""
        <div class="features-section-ref">
            <div class="feat-header">
                <div class="feat-line"></div>
                <span class="feat-title">Why Choose HireMind AI?</span>
                <div class="feat-line"></div>
            </div>
            <div class="feat-row-ref">
                <div class="feat-item-ref">
                    <div class="feat-icon-round" style="color:#3b82f6; background:rgba(59,130,246,0.05);">🎯</div>
                    <span class="feat-name-ref">AI Resume Screening</span>
                    <span class="feat-desc-ref">Smart matching with context understanding</span>
                </div>
                <div class="feat-item-ref">
                    <div class="feat-icon-round" style="color:#8b5cf6; background:rgba(139,92,246,0.05);">🛡️</div>
                    <span class="feat-name-ref">Fraud Detection</span>
                    <span class="feat-desc-ref">Advanced verification and anti-cheating</span>
                </div>
                <div class="feat-item-ref">
                    <div class="feat-icon-round" style="color:#10b981; background:rgba(16,185,129,0.05);">🎙️</div>
                    <span class="feat-name-ref">Voice AI Interview</span>
                    <span class="feat-desc-ref">Natural conversations with AI interviewer</span>
                </div>
                <div class="feat-item-ref">
                    <div class="feat-icon-round" style="color:#f59e0b; background:rgba(245,158,11,0.05);">📸</div>
                    <span class="feat-name-ref">Live Proctoring</span>
                    <span class="feat-desc-ref">Real-time monitoring and face verification</span>
                </div>
                <div class="feat-item-ref">
                    <div class="feat-icon-round" style="color:#ec4899; background:rgba(236,72,153,0.05);">📊</div>
                    <span class="feat-name-ref">Analytics Dashboard</span>
                    <span class="feat-desc-ref">Data-driven insights for better hiring</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- FOOTER ---
    st.markdown("""
        <div class="footer-container-ref">
            <div class="footer-info">
                <span style="font-size:1.1rem; filter:grayscale(100%);">🔒</span> 
                Secure. Intelligent. Autonomous. Building the future of hiring.
            </div>
            <div class="footer-copyright">
                © 2024 HireMind AI. All rights reserved.
            </div>
            <div class="footer-info">
                Made with <span class="heart-red">❤️</span> for a better future
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_login_page():
    role = st.session_state.get("auth_role", "candidate")
    
    # --- PREMIUM LOGIN UI CSS ---
    st.markdown("""
<style>
    /* Reset & Background */
    .stApp { background: #fdfdff !important; }
    
    /* Center the Login Card */
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 2rem;
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.03), transparent 40%),
                    radial-gradient(circle at 90% 90%, rgba(168, 85, 247, 0.03), transparent 40%);
    }

    /* Main Container */
    div[data-testid="stHorizontalBlock"]:has(.login-card-left) {
        background: white !important;
        border: 1px solid rgba(226, 232, 240, 0.8) !important;
        border-radius: 32px !important;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.04) !important;
        overflow: hidden !important;
        max-width: 1050px !important;
        margin: 4rem auto !important;
        gap: 0 !important;
    }
    
    /* Left Panel (Form) */
    div[data-testid="stHorizontalBlock"]:has(.login-card-left) > div[data-testid="column"]:nth-of-type(1) {
        padding: 4rem 5rem !important;
        background: white !important;
        border-right: 1px solid #f1f5f9 !important;
    }

    /* Right Panel (Illustration/Info) */
    div[data-testid="stHorizontalBlock"]:has(.login-card-left) > div[data-testid="column"]:nth-of-type(2) {
        background: #f8fafc !important;
        padding: 4rem !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* Typography */
    .login-brand-logo {
        width: 44px; height: 44px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem; color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
    }
    .login-title-text {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .login-subtitle-text {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }

    /* Form Styling */
    div[data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    .stTextInput label {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    .stTextInput input {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        color: #0f172a !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
    }
    .stTextInput input:focus {
        border-color: #6366f1 !important;
        background: white !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.05) !important;
    }

    /* Buttons */
    div[data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        padding: 0.9rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    /* Primary Action */
    div[data-testid="stFormSubmitButton"]:first-of-type > button {
        background: linear-gradient(90deg, #4f46e5, #6366f1) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.2) !important;
    }
    div[data-testid="stFormSubmitButton"]:first-of-type > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.3) !important;
        opacity: 0.95 !important;
    }
    /* Secondary/Cancel */
    div[data-testid="stFormSubmitButton"]:last-of-type > button {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        color: #64748b !important;
    }
    div[data-testid="stFormSubmitButton"]:last-of-type > button:hover {
        border-color: #cbd5e1 !important;
        color: #1e293b !important;
        background: #f8fafc !important;
    }

    /* Right Panel Content */
    .feature-img {
        width: 100%;
        max-width: 320px;
        border-radius: 24px;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.06);
    }
    .right-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }
    .right-text {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 2.5rem;
    }
    .feat-pill-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        width: 100%;
    }
    .feat-pill {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.5rem;
        border-radius: 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        text-align: left;
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.01);
    }

    /* Sign Up Expander Overrides */
    div[data-testid="stExpander"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        margin-top: 1.5rem !important;
    }
    div[data-testid="stExpanderSummary"] {
        font-weight: 700 !important;
        color: #4f46e5 !important;
    }

    /* Helper Classes */
    .login-card-left { display: block; }
</style>
""", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.2, 1])
    
    with left_col:
        st.markdown(f"""
            <div class="login-card-left">
                <div class="login-brand-logo">🧠</div>
                <h1 class="login-title-text">Welcome Back</h1>
                <p class="login-subtitle-text">Enter your details to access your {role} portal</p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form_premium"):
            email = st.text_input("Email Address", placeholder="name@company.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            st.markdown("""
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem; margin-top:0.5rem;">
                    <label style="display:flex; align-items:center; gap:8px; color:#64748b; font-size:0.9rem; cursor:pointer;">
                        <input type="checkbox" checked style="width:16px; height:16px; accent-color:#4f46e5;"> Remember me
                    </label>
                    <a href="#" style="color:#4f46e5; text-decoration:none; font-size:0.9rem; font-weight:600;">Forgot password?</a>
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
                <div style="display:flex; align-items:center; gap:16px; margin: 3rem 0 1rem;">
                    <div style="flex:1; height:1px; background:#e2e8f0;"></div>
                    <div style="color:#94a3b8; font-size:0.85rem; font-weight:700;">NEW TO HIREMIND?</div>
                    <div style="flex:1; height:1px; background:#e2e8f0;"></div>
                </div>
            """, unsafe_allow_html=True)
            render_signup("candidate")

    with right_col:
        # User requested image path: assets/WhatsApp Image 2026-05-07 at 4.29.58 PM.jpeg
        # We use a path-safe way or base64 if needed, but since it's local we can use the file path relative to the app or absolute
        img_path = r"assets/WhatsApp Image 2026-05-07 at 4.29.58 PM.jpeg"
        
        st.markdown(f"""
            <div style="position:relative; width:100%; display:flex; justify-content:center; align-items:center;">
                <!-- Glowing Aura Background -->
                <div style="position:absolute; width:280px; height:280px; background:radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%); filter:blur(40px); z-index:0;"></div>
                
                <img src="app/static/{img_path}" class="feature-img" alt="HireMind AI Illustration" 
                     style="position:relative; z-index:1; border:1px solid rgba(99, 102, 241, 0.2); box-shadow: 0 0 30px rgba(99, 102, 241, 0.15); mix-blend-mode: multiply; opacity: 0.95;">
            </div>
            
            <h2 class="right-title" style="margin-top: 1.5rem;">Smart Hiring, Powered by AI</h2>
            <p class="right-text">Experience the future of recruitment with our autonomous AI ecosystem.</p>
            
            <div class="feat-pill-list">
                <div class="feat-pill">
                    <span style="font-size:1.2rem;">✨</span>
                    <span>AI Resume Analysis</span>
                </div>
                <div class="feat-pill">
                    <span style="font-size:1.2rem;">🎙️</span>
                    <span>Real-time Voice Interviews</span>
                </div>
                <div class="feat-pill">
                    <span style="font-size:1.2rem;">🛡️</span>
                    <span>Proctored Assessments</span>
                </div>
            </div>
        """, unsafe_allow_html=True)



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
    with st.expander("✨ New here? Create your Account", expanded=False):
        with st.form(f"signup_form_{role}"):
            st.markdown("""
                <style>
                    .stForm { background: transparent !important; border: none !important; }
                </style>
            """, unsafe_allow_html=True)
            
            name = st.text_input("Full Name", placeholder="e.g. Bhavesh Chaudhary")
            email = st.text_input("Email Address", placeholder="name@example.com")
            
            c1, c2 = st.columns(2)
            with c1:
                password = st.text_input("Password", type="password", placeholder="••••••••")
            with c2:
                confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                
            if st.form_submit_button("Create Account →", use_container_width=True, type="primary"):
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
            col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1.2, 1.8])
            with col1: st.caption("CANDIDATE")
            with col2: st.caption("STAGE 1 (ATS)")
            with col3: st.caption("STAGE 2 (INT)")
            with col4: st.caption("FINAL SCORE")
            with col5: st.caption("ACTIONS")

            for i, app in enumerate(ranked_applicants):
                with st.container():
                    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1.2, 1.8])
                    
                    with c1:
                        st.markdown(f"**{app['full_name']}**")
                        h_status = app.get("hiring_status", "pending")
                        if h_status == "hired":
                            st.markdown('<span class="hired-badge" style="background:#f0fdf4; color:#166534; border:1px solid #86efac; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem;">✅ HIRED</span>', unsafe_allow_html=True)
                        elif h_status == "rejected":
                            st.markdown('<span style="background:#fef2f2; color:#991b1b; border:1px solid #fecaca; padding:2px 8px; border-radius:6px; font-weight:700; font-size:0.75rem;">❌ REJECTED</span>', unsafe_allow_html=True)
                        else:
                            st.caption(app['email'])
                            
                    with c2:
                        st.write(f"{app['ats_score']:.1f}%")
                    with c3:
                        st.write(f"{app['final_score']:.1f}%")
                    with c4:
                        st.markdown(f'<span class="score-badge">{app["total_score"]:.1f}</span>', unsafe_allow_html=True)
                        st.caption(f"Rank #{i+1}")
                    
                    with c5:
                        # Actions Row
                        a1, a2 = st.columns(2)
                        with a1:
                            if st.button("📄 View Report", key=f"det_{app['id']}", use_container_width=True):
                                show_candidate_details(app, selected_job['title'])
                        
                        with a2:
                            if h_status == "pending":
                                if st.button("🚀 Hire", key=f"hire_{app['id']}", type="primary", use_container_width=True):
                                    update_hiring_status(app['id'], "hired")
                                    result = send_hiring_email(app['email'], app['full_name'], selected_job['title'])
                                    if result["success"]:
                                        st.success(f"Hired {app['full_name']}!")
                                    else:
                                        st.warning(f"Status updated, but email failed: {result['message']}")
                                    st.rerun()
                                    
                                if st.button("❌ Reject", key=f"rej_{app['id']}", use_container_width=True):
                                    update_hiring_status(app['id'], "rejected")
                                    st.rerun()
                            elif h_status == "hired":
                                st.info("Hiring Confirmed")
                            elif h_status == "rejected":
                                st.error("Rejected")

                    st.markdown("<div style='margin: 0.5rem 0; border-bottom: 1px solid #f1f5f9;'></div>", unsafe_allow_html=True)

    else:
        # DEFAULT OVERVIEW
        st.subheader("📋 System Overview")
        st.write("Select a job from the sidebar to view ranked applicants and manage hiring.")
        
        counts_col1, counts_col2 = st.columns(2)
        with counts_col1:
            st.metric("Total Jobs", len(jobs))
        with counts_col2:
            from modules.candidate_db import get_all_candidates
            all_cands = get_all_candidates()
            st.metric("Identity Verified Candidates", len(all_cands))

@st.dialog("Candidate Detailed Report", width="large")
def show_candidate_details(app, job_title):
    st.markdown(f"### 📋 Detailed Assessment: {app['full_name']}")
    st.caption(f"Role: {job_title} | Session ID: #{app['id']}")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👤 Contact Information")
        st.write(f"📧 **Email:** {app['email']}")
        st.write(f"📞 **Phone:** {app['phone']}")
    with col2:
        st.markdown("#### 🗓 Assessment Timeline")
        st.write(f"✅ **Completed:** {app.get('timestamp', 'N/A')}")
        status = app.get('hiring_status', 'Pending').title()
        color = "#10b981" if status == "Hired" else "#ef4444" if status == "Rejected" else "#64748b"
        st.markdown(f"📍 **Status:** <span style='color:{color}; font-weight:700;'>{status}</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Performance Matrix")
    
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Stage 1 (ATS)", f"{app['ats_score']:.1f}%")
    with m2:
        st.metric("Stage 2 (Interview)", f"{app['final_score']:.1f}%")
    with m3:
        st.metric("Final Weighted Score", f"{app['total_score']:.1f}")

    st.markdown("---")
    
    # ── AI ANALYSIS REPORT ──────────────────────────────────────────────────
    st.markdown("#### 🧠 Final AI Evaluation Report")
    ai_report_raw = app.get("ai_report", "")
    if ai_report_raw:
        try:
            import json
            ai = json.loads(ai_report_raw)
            if isinstance(ai, dict) and ai.get("overall_summary"):
                st.markdown(f"""
                    <div style="background:#f8fafc; padding:20px; border-radius:16px; border:1px solid #e2e8f0; margin-bottom:20px;">
                        <p style="color:#475569; font-size:0.9rem; font-weight:700; text-transform:uppercase; margin-bottom:8px;">Executive Summary</p>
                        <p style="color:#1e293b; font-size:1.05rem; line-height:1.6; margin:0;">{ai.get('overall_summary')}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                sum_col1, sum_col2 = st.columns(2)
                with sum_col1:
                    st.markdown("**✅ Key Strengths**")
                    for s in ai.get("key_strengths", []):
                        st.markdown(f"- {s}")
                with sum_col2:
                    st.markdown("**⚠️ Key Weaknesses**")
                    for w in ai.get("key_weaknesses", []):
                        st.markdown(f"- {w}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**💡 Hiring Recommendation**")
                st.success(ai.get("hiring_recommendation", "N/A"))
                
                if ai.get("skill_gaps"):
                    st.markdown("**🛠 Skill Gaps & Development**")
                    st.warning(", ".join(ai.get("skill_gaps")))
            else:
                st.markdown(f"<div style='background:#f8fafc; padding:15px; border-radius:10px;'>{ai_report_raw}</div>", unsafe_allow_html=True)
        except:
            st.markdown(f"<div style='background:#f8fafc; padding:15px; border-radius:10px;'>{ai_report_raw}</div>", unsafe_allow_html=True)
    else:
        st.info("Detailed AI Analysis report is being processed or was not generated.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── RAW INTERVIEW ANSWERS ──────────────────────────────────────────────
    from modules.candidate_db import get_interview_results
    results = get_interview_results(app['id'])
    
    with st.expander("📝 View Detailed Interview Transcript", expanded=False):
        for ans in results.get("answers", []):
            st.markdown(f"**Q: {ans['question_text']}**")
            st.markdown(f"<div style='background:#f1f5f9; padding:10px; border-radius:8px; margin-bottom:5px;'><em>{ans['answer_text']}</em></div>", unsafe_allow_html=True)
            st.caption(f"Score: {ans['score']}/10 | Evaluation: {ans['evaluation']}")
            st.markdown("---")

    if st.button("Close Report", use_container_width=True):
        st.rerun()

def logout():
    st.session_state["logged_in"] = False
    st.session_state["user_data"] = None
    st.session_state["user_role"] = None
    st.session_state["auth_role"] = None
    st.session_state["current_page"] = "role_selection"
    st.session_state["view"] = "dashboard"
    st.rerun()
