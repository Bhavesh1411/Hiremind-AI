import os
import json
import streamlit as st
from dotenv import load_dotenv
from modules.data_ingestion import process_resume
from modules.text_processing import build_structured_json
from modules.embeddings import process_embeddings, INDEX_PATH
from modules.similarity import match_resume_to_jd
from modules.llm_analysis import generate_analysis_report, PROVIDER_GEMINI
from modules.ats_scorer import run_ats_analysis
from modules.fraud_detector import generate_fraud_report
from modules.recommendation_engine import build_recommendation_output

# Load environment variables from .env file
load_dotenv()
_GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
_OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
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
                            parsed = build_structured_json(result["cleaned_text"])
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

        # --- Parsed Resume Display ---
        parsed = st.session_state.get("parsed_resume")
        if parsed:
            with st.expander("🧩 Parsed Resume Data (NLP Output)", expanded=True):
                # Entity cards row
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("👤 Name",     parsed.get("name") or "—")
                e2.metric("📧 Email",    parsed.get("email") or "—")
                e3.metric("📞 Phone",    parsed.get("phone") or "—")
                e4.metric("📍 Location", parsed.get("location") or "—")

                st.markdown("---")

                # Skills display
                skills = parsed.get("skills", [])
                if skills:
                    st.markdown(f"**🛠 Extracted Skills ({len(skills)}):**")
                    tags_html = " ".join(
                        f'<span class="tag tag-match">{s}</span>' for s in skills
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)
                else:
                    st.info("No skills detected.")

                st.markdown("---")

                # Section previews
                sec_tab1, sec_tab2, sec_tab3, sec_tab4 = st.tabs(
                    ["🎓 Education", "💼 Experience", "🚀 Projects", "📋 Full JSON"]
                )
                with sec_tab1:
                    st.text(parsed.get("education") or "Not detected.")
                with sec_tab2:
                    st.text(parsed.get("experience") or "Not detected.")
                with sec_tab3:
                    st.text(parsed.get("projects") or "Not detected.")
                with sec_tab4:
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
            input_mode = st.radio("Input Method", ["Paste Text", "Upload Document"], horizontal=True)

            if input_mode == "Paste Text":
                jd_text = st.text_area(
                    "JD Content",
                    placeholder="Enter the role requirements and context here...",
                    height=250,
                    key="jd_text_input",
                )
            else:
                jd_file = st.file_uploader("Load JD Source", type=["pdf", "docx"], key="jd_file_upload")
                jd_text = ""
                if jd_file:
                    # Quick extract for JD files (reuse ingestion helpers)
                    from modules.data_ingestion import extract_text, clean_text
                    from pathlib import Path
                    import tempfile, os
                    tmp_path = Path("data") / "jd_temp" / jd_file.name
                    tmp_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path.write_bytes(jd_file.getbuffer())
                    raw = extract_text(tmp_path)
                    jd_text = clean_text(raw)
                    st.success(f"✅ JD loaded: {jd_file.name}")
                    with st.expander("📄 JD Preview"):
                        st.text(jd_text[:800])

# ── ELIGIBILITY & INTERVIEW LOGIC ──────────────────────────────────────────────
def check_interview_eligibility():
    """
    Evaluates candidate scores to decide if they proceed to interview.
    Criteria: Overall Score >= 80, ATS >= 75, Fraud = 'Low'.
    """
    match_result = st.session_state.get("match_result")
    ats_result   = st.session_state.get("ats_result")
    fraud_result = st.session_state.get("fraud_result")
    llm_result   = st.session_state.get("llm_result")

    if not match_result or not ats_result or not fraud_result:
        return False

    # Get Overall Score (favor LLM if available, fallback to similarity)
    overall_score = 0
    if llm_result and llm_result.get("status") == "success":
        report = llm_result.get("report", {})
        overall_score = report.get("match_percentage", 0)
    else:
        # Fallback to similarity score (match_result['score'] is 0.0-1.0)
        # Note: ranked results use 'final_score', we'll look at the top one
        if match_result.get("ranked"):
            overall_score = match_result["ranked"][0].get("final_score", 0)
        else:
            overall_score = match_result.get("score", 0) * 100

    ats_score  = ats_result.get("ats_score", 0)
    fraud_risk = fraud_result.get("fraud_risk", "Unknown")

    return overall_score >= 80 and ats_score >= 75 and fraud_risk == "Low"


def render_interview_button():
    """Renders the 'Proceed' button for eligible candidates."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.success("🎉 **Candidate has passed the initial screening!**")
    if st.button("🚀 Proceed to Interview Rounds", use_container_width=True, type="primary"):
        st.session_state["view"] = "interview"
        st.rerun()


def render_recommendations():
    """Renders gap analysis and suggestions for non-eligible candidates."""
    rec_result = st.session_state.get("rec_result")
    if rec_result and rec_result.get("status") == "success":
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning("📋 **Candidate doesn't meet the current interview threshold.**")
        st.markdown("### 🎯 Skill Gap Analysis & Recommendations")
        
        # Summary
        if rec_result.get("improvement_summary"):
            st.markdown(rec_result["improvement_summary"])

        # LLM Plan if available
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
                    pct = role['match_pct']
                    color = "#10b981" if pct >= 70 else ("#f59e0b" if pct >= 50 else "#64748b")
                    st.markdown(
                        f"**{role['role']}** — "
                        f"<span style='color:{color};font-weight:700'>{pct}% match</span> "
                        f"({role['missing_count']} skills to go)",
                        unsafe_allow_html=True,
                    )

def render_interview_ui():
    """The new AI Interview Engine interface."""
    st.markdown("""
        <div class="header-container">
            <h1 class="main-title">AI Interview Engine</h1>
            <p class="subtitle">Autonomous Candidate Assessment Phase</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize progress if missing
    if "interview_step" not in st.session_state:
        st.session_state["interview_step"] = 1
    if "answers" not in st.session_state:
        st.session_state["answers"] = {}

    current_step = st.session_state["interview_step"]
    total_steps = 5

    _, mid, _ = st.columns([1, 4, 1])
    with mid:
        # A. Interview Instructions
        with st.container(border=True):
            st.markdown("### 📝 Interview Instructions")
            st.write("""
                Welcome to the AI Interview round. You will be asked 5 questions based on your specialized skills and the job requirements.
                Please provide concise, detailed answers for the best assessment.
            """)

        # B. Start Interview / Status
        st.progress(current_step / total_steps, text=f"Question {current_step} of {total_steps}")

        # C. Question Display Area
        st.markdown(f"#### ❓ Question {current_step}")
        with st.container(border=True):
            st.write(f"This is a placeholder for Interview Question #{current_step}. [System under development]")

        # D. Answer Input
        st.markdown("#### 🖊️ Your Answer")
        answer = st.text_area("Type your response here:", key=f"ans_{current_step}", height=200)

        # Voice input placeholder
        st.caption("🎙️ Voice Input — Coming Soon (v0.8)")

        # E. Navigation Buttons
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Dashboard"):
                st.session_state["view"] = "dashboard"
                st.rerun()

        with col2:
            if current_step < total_steps:
                if st.button("Next Question ➡️", use_container_width=True):
                    st.session_state["interview_step"] += 1
                    st.rerun()
            else:
                if st.button("✔️ Submit Interview", use_container_width=True, type="primary"):
                    st.balloons()
                    st.success("✅ Interview submitted successfully! Our recruiters will review your responses shortly.")
                    if st.button("Return to Dashboard"):
                        st.session_state["view"] = "dashboard"
                        st.rerun()


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
                    for i, chunk in enumerate(top_chunks[:3], 1):
                        score_pct = chunk['score'] * 100
                        st.markdown(
                            f"**#{i}** ({chunk['section']}) — `{score_pct:.1f}%` match"
                        )
                        st.caption(chunk["chunk_text"][:200])
                        if i < len(top_chunks[:3]):
                            st.markdown("---")
                else:
                    st.info("No chunk-level data available.")

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
            ats_k1, ats_k2, ats_k3, ats_k4 = st.columns(4)
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
                    <div style="color:#64748b;font-weight:600">Weight: 40%</div>
                </div>
            """, unsafe_allow_html=True)
            ats_k3.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Keyword Score</div>
                    <div class="metric-value" style="font-size:1.8rem">{ats_result['keyword_score']:.1f}</div>
                    <div style="color:#64748b;font-weight:600">Weight: 40%</div>
                </div>
            """, unsafe_allow_html=True)
            ats_k4.markdown(f"""
                <div class="metric-card-container">
                    <div class="metric-label">Structure Score</div>
                    <div class="metric-value" style="font-size:1.8rem">{ats_result['structure_score']:.1f}</div>
                    <div style="color:#64748b;font-weight:600">Weight: 20%</div>
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
                    density_pct = checklist.get("keyword_density", 0) * 100
                    density_icon = "✅" if density_pct >= 50 else ("🟡" if density_pct >= 30 else "🔴")
                    st.markdown(f"{density_icon} **Keyword Density**: {density_pct:.1f}%")

            with kw_col:
                with st.container(border=True):
                    st.markdown("#### 🎯 Keyword Analysis")
                    matched = ats_result.get("matched_keywords", [])[:12]
                    unmatched = ats_result.get("unmatched_keywords", [])[:12]
                    if matched:
                        st.caption(f"✅ Matched ({len(ats_result.get('matched_keywords',[]))})")
                        tags_m = " ".join(f'<span class="tag tag-match">{k}</span>' for k in matched)
                        st.markdown(tags_m, unsafe_allow_html=True)
                    if unmatched:
                        st.markdown("")
                        st.caption(f"🔴 Missing ({len(ats_result.get('unmatched_keywords',[]))})")
                        tags_u = " ".join(f'<span class="tag tag-missing">{k}</span>' for k in unmatched)
                        st.markdown(tags_u, unsafe_allow_html=True)

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


        # ── INTERVIEW ELIGIBILITY OVERRIDE ──────────────────────────────
        eligible = check_interview_eligibility()
        if eligible:
            render_interview_button()
        else:
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
            st.toast(match_result["message"], icon="✅")
            st.rerun()

    if match_result["status"] != "success":
        st.warning(match_result["message"])


# --- INTERVIEW VIEW WRAPPER ---

def render_main_dashboard():
    display_dashboard()


# --- APP ROUTER ---
def main():
    if "view" not in st.session_state:
        st.session_state["view"] = "dashboard"

    local_css()

    if st.session_state["view"] == "interview":
        render_interview_ui()
    else:
        # ── Sidebar: System Status ───────────────────────────────────────────
        with st.sidebar:
            st.markdown("### 🧠 HireMind AI")
            st.caption("Autonomous HR Operating System")
            st.markdown("---")

            st.markdown("#### ⚙️ System Status")
            # Check which modules are ready
            st.markdown("✅ Data Ingestion")
            st.markdown("✅ NLP Parsing")
            st.markdown("✅ Vector Embeddings")
            st.markdown("✅ Semantic Matching")
            st.markdown("✅ ATS Integrity")
            st.markdown("✅ Fraud Detection")
            st.markdown("✅ Recommendation Engine")

            # Show API key status from .env
            if _GEMINI_API_KEY:
                st.markdown("✅ Gemini AI — Ready")
            elif _OPENAI_API_KEY:
                st.markdown("✅ OpenAI — Ready")
            else:
                st.markdown("⚠️ LLM — No API key in .env")

            st.markdown("---")
            st.caption("v0.7 | Modules: 7/8 complete")

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
                jd_text = st.session_state.get("jd_text_input", "").strip()

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



if __name__ == "__main__":
    main()
