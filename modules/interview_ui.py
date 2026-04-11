"""
HireMind AI - Interview UI Components (Refactored)
====================================================
Streamlit-based UI for the full 7-question interview session.

Flow:
  render_mode_selection()   -> Mode selection screen (Normal / AI)
  render_interview_ui()     -> Per-question LeetCode-style interface
  finalize_interview_flow() -> Completion/transition screen
  render_final_report()     -> Full performance report

Rules:
  - NO scores shown during the interview
  - All evaluation is silent on submission
  - Results only shown after all 7 questions are answered
"""

import streamlit as st
from typing import List
import time

from modules.interview_engine import (
    evaluate_answer,
    generate_normal_report,
    generate_ai_questions,
    evaluate_with_ai,
    generate_deep_ai_report,
)
from modules.question_bank import select_interview_questions
from modules.candidate_db import (
    add_interview_answer,
    finalize_interview,
    get_interview_results,
)
from modules.webcam_monitor import render_webcam_monitor, reset_monitor_state


# ── SHARED STYLES ─────────────────────────────────────────────────────────────

def apply_interview_styles():
    st.markdown("""
<style>
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    .stTextArea textarea {
        font-family: 'Source Code Pro', 'Consolas', monospace !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        background-color: #1e1e2e !important;
        color: #cdd6f4 !important;
        border-radius: 8px !important;
    }

    .q-header {
        background: linear-gradient(90deg, #1e293b, #334155);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }

    .badge-coding     { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-technical  { background:#eff6ff; color:#1e40af; border:1px solid #93c5fd; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-behavioral { background:#f0fdf4; color:#166534; border:1px solid #86efac; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── MODE SELECTION ────────────────────────────────────────────────────────────

def render_mode_selection():
    """Structured mode selection screen."""
    apply_interview_styles()

    st.markdown("""
    <div class="q-header">
        <h2 style="margin:0; font-size:1.6rem;">🎙️ Interview Configuration</h2>
        <p style="margin:0.3rem 0 0; color:#94a3b8;">Select your interview mode. This selection is final.</p>
    </div>
    """, unsafe_allow_html=True)

    col_norm, col_ai = st.columns(2)

    with col_norm:
        st.markdown("""
        <div style="background:white;padding:1.5rem;border-radius:12px;border:1px solid #e2e8f0;height:200px;">
            <h4 style="color:#1e293b;margin-top:0;">⚙️ Normal Mode</h4>
            <ul style="color:#64748b;font-size:0.9rem;line-height:1.8;">
                <li>Predefined 10-question bank</li>
                <li>7 questions per session</li>
                <li>Rule-based + RapidFuzz evaluation</li>
                <li>No external API dependencies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ai:
        st.markdown("""
        <div style="background:white;padding:1.5rem;border-radius:12px;border:1px solid #3b82f6;height:200px;">
            <h4 style="color:#2563eb;margin-top:0;">🤖 AI Mode</h4>
            <ul style="color:#64748b;font-size:0.9rem;line-height:1.8;">
                <li>Gemini-generated personalized questions</li>
                <li>AI-powered deep evaluation</li>
                <li>Resume-tailored skill assessment</li>
                <li>Requires valid Gemini API key</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    selected_mode = st.radio(
        "Select Interview Mode:",
        ["⚙️ Normal Mode", "🤖 AI Mode"],
        index=0,
        horizontal=True,
        key="mode_radio_selector",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Start Interview Sequence", use_container_width=True, type="primary"):
        mode = "normal" if "Normal" in selected_mode else "ai"

        if mode == "normal":
            selected_questions = select_interview_questions()
            if "interview_session_id" not in st.session_state:
                candidate_id = st.session_state.get("candidate_id", 0)
                mode = st.session_state.get("interview_mode", "normal")
                job_id = st.session_state.get("current_job_id", 0)
                st.session_state["interview_session_id"] = create_interview_session(candidate_id, mode, job_id)
            st.session_state["selected_questions"] = selected_questions
            st.session_state["interview_mode"]  = "normal"
            st.session_state["interview_answers"] = {}
            st.session_state["current_q_idx"]   = 0
            st.session_state["current_page"]    = "interview"
            st.rerun()

        else:
            api_key = st.session_state.get("GEMINI_API_KEY", "")
            if not api_key:
                st.error("❌ Gemini API key is required for AI Mode. Check your `.env` file.")
                return
            with st.spinner("🤖 Gemini is crafting your personalised interview..."):
                selected_questions = generate_ai_questions(
                    st.session_state.get("parsed_resume", {}), api_key
                )
            st.session_state["selected_questions"] = selected_questions
            st.session_state["interview_mode"]  = "ai"
            st.session_state["interview_answers"] = {}
            st.session_state["current_q_idx"]   = 0
            st.session_state["current_page"]    = "interview"
            st.rerun()


# ── INTERVIEW INTERFACE ───────────────────────────────────────────────────────

def render_interview_ui():
    """Main 7-question interview UI, one question at a time."""
    apply_interview_styles()

    questions   = st.session_state.get("selected_questions", [])
    current_idx = st.session_state.get("current_q_idx", 0)
    total       = len(questions)

    if not questions:
        st.info("⌛ Loading questions... Please wait.")
        # Fallback trigger if something went wrong during mode selection
        return

    if current_idx >= total:
        _finalize_interview_flow()
        return

    q = questions[current_idx]

    # ── Progress header ───────────────────────────────────────────────────────
    badge_class = f"badge-{q['type']}"
    pct         = (current_idx + 1) / total
    mode_label  = st.session_state.get("interview_mode", "normal").upper()

    st.markdown(f"""
    <div class="q-header" style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <span style="font-size:1.1rem;font-weight:700;">Question {current_idx + 1} of {total}</span>
            &nbsp;<span class="{badge_class}">{q['type'].title()}</span>
        </div>
        <div style="font-size:0.85rem;color:#94a3b8;">Mode: {mode_label}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(pct)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Layout: Webcam Monitor + Question Area ──────────────────────────────────
    col_monitor, col_main = st.columns([1, 2.5])

    with col_monitor:
        monitor_ok, webcam_active = render_webcam_monitor()

        if not monitor_ok:
            # 2nd violation — reset the interview
            reset_monitor_state()
            st.session_state["current_q_idx"] = 0
            st.session_state["interview_answers"] = {}
            if "interview_mode" in st.session_state:
                del st.session_state["interview_mode"]
            st.session_state["current_page"] = "interview_mode"
            time.sleep(2)  # Let user read the termination message
            st.rerun()
            return

    with col_main:
        if not webcam_active:
            st.markdown("""
            <div style="background:#f8fafc; padding:2rem; border-radius:12px; border:1px solid #e2e8f0; text-align:center; margin-top:1rem;">
                <div style="font-size:3rem; margin-bottom:1rem;">📸</div>
                <h3 style="margin:0; color:#1e293b;">Webcam Authorization Required</h3>
                <p style="color:#64748b; margin-top:0.5rem;">
                    Please click the <b>Start Interview</b> button in the monitor panel (left) 
                    and allow browser permissions to begin your session.
                </p>
            </div>
            """, unsafe_allow_html=True)
            return

        col_q, col_a = st.columns([1, 1.2])

        with col_q:
            with st.container(border=True):
                st.markdown(f"#### {q['title']}")
                st.markdown(q["description"])

                if q["type"] == "coding" and q.get("test_cases"):
                    st.markdown("**Example Test Cases (first 2):**")
                    for tc in q["test_cases"][:2]:
                        st.code(f"Input  : {tc['input']}\nExpected: {tc['expected']}", language=None)
                elif q["type"] != "coding":
                    st.caption("Tip: Be specific and use technical terminology where applicable.")

        with col_a:
            if q["type"] == "coding":
                default = st.session_state.get(f"draft_{current_idx}", q.get("starter_code", "# Write your code here\n"))
                user_input = st.text_area(
                    "Code Editor",
                    value=default,
                    height=380,
                    key=f"editor_{current_idx}",
                    label_visibility="visible",
                )

                # Run Code button (Normal mode only — shows test results immediately)
                run_col, _, submit_col = st.columns([1, 0.3, 1])
                with run_col:
                    if st.button("▶️ Run Code", use_container_width=True, key=f"run_{current_idx}"):
                        if st.session_state.get("interview_mode") == "normal":
                            with st.spinner("Running tests..."):
                                result = evaluate_answer(q, user_input)
                            for d in result.get("details", []):
                                if d.get("passed"):
                                    st.success(f"✔ Input: {d['input']} → {d.get('actual', '')}")
                                else:
                                    st.error(f"✖ Input: {d['input']} → Got: {d.get('actual', d.get('error', 'error'))}")
                        else:
                            st.info("Test runs are not available in AI Mode.")
            else:
                default = st.session_state.get(f"draft_{current_idx}", "")
                user_input = st.text_area(
                    "Your Answer",
                    value=default,
                    height=380,
                    key=f"text_{current_idx}",
                    placeholder="Type your detailed answer here...",
                )
                _, _, submit_col = st.columns([1, 0.3, 1])

            with submit_col:
                btn_label = "Submit & Next ⮕" if current_idx < total - 1 else "Submit & Finish ✓"
                if st.button(btn_label, key=f"submit_{current_idx}", use_container_width=True, type="primary"):
                    # Save draft so user can go back (not implemented, but state preserved)
                    st.session_state[f"draft_{current_idx}"] = user_input

                    # ── SILENT EVALUATION ─────────────────────────────────────────
                    with st.spinner("Saving answer..."):
                        mode = st.session_state.get("interview_mode", "normal")

                        if mode == "normal":
                            eval_result = evaluate_answer(q, user_input)
                            marks      = eval_result["marks"]
                            similarity = eval_result["similarity"]
                            evaluation = eval_result["evaluation"]
                        else:
                            api_key    = st.session_state.get("GEMINI_API_KEY", "")
                            eval_result = evaluate_with_ai(q, user_input, api_key)
                            marks      = eval_result.get("marks", 0)
                            similarity = 0.0
                            evaluation = eval_result.get("evaluation", "")

                        # ── PERSIST TO DB ─────────────────────────────────────────
                        add_interview_answer(
                            session_id      = st.session_state["interview_session_id"],
                            idx             = current_idx,
                            question        = q["description"],
                            answer          = user_input,
                            score           = marks,
                            evaluation      = evaluation,
                            q_type          = q["type"],
                            similarity      = similarity,
                            expected_answer = q.get("expected_answer", ""),
                        )

                    # ── ADVANCE ───────────────────────────────────────────────────
                    st.session_state["current_q_idx"] += 1
                    st.rerun()



# ── COMPLETION SCREEN ─────────────────────────────────────────────────────────

def _finalize_interview_flow():
    """Shown when all questions are answered."""
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;">🎊</div>
        <h2 style="margin:1rem 0 0.5rem;">Interview Complete!</h2>
        <p style="color:#64748b;max-width:400px;margin:0 auto;">
            All your answers have been securely stored and silently evaluated.
            Click below to view your full performance report.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("🎙️ Proceed to Voice Interview (Stage 2)", use_container_width=True, type="primary"):
            results = get_interview_results(st.session_state["interview_session_id"])
            answers = results["answers"]
            if answers:
                final_score = sum(a["score"] for a in answers) / len(answers)
                
                # Retrieve ATS score to ensure it is not lost
                ats_val = st.session_state.get("ats_result", {}).get("ats_score", 0.0)
                finalize_interview(st.session_state["interview_session_id"], final_score, ats_score=ats_val)

            st.session_state["stage1_answers"] = answers
            st.session_state["current_page"] = "voice_interview"
            st.rerun()


# ── FINAL REPORT ──────────────────────────────────────────────────────────────

def render_final_report():
    """Full post-interview performance dashboard."""
    apply_interview_styles()

    session_id = st.session_state.get("interview_session_id")
    if not session_id:
        st.error("No active interview session found.")
        return

    data    = get_interview_results(session_id)
    session = data["session"]
    answers = data["answers"]

    candidate = st.session_state.get("candidate_name", "Candidate")
    mode_used = session["mode"].upper()

    # ── Top summary banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="q-header">
        <h2 style="margin:0;">📊 Interview Performance Report</h2>
        <p style="margin:0.3rem 0 0;color:#94a3b8;">
            Candidate: <strong style="color:white;">{candidate}</strong> &nbsp;|&nbsp;
            Mode: <strong style="color:white;">{mode_used}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Compute normal report metrics
    from modules.interview_engine import generate_normal_report
    report = generate_normal_report(answers)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Score",    f"{report['total_marks']}/{report['max_marks']}")
    m2.metric("Percentage",     f"{report['percentage']}%")
    m3.metric("Questions Done", len(answers))
    m4.metric("Mode",           mode_used)

    st.markdown("---")

    # ── Tabs: Breakdown | AI Deep Dive ───────────────────────────────────────
    t1, t2 = st.tabs(["📝 Question-Wise Breakdown", "🧠 AI Deep Analysis"])

    with t1:
        st.markdown("#### Strengths vs Weaknesses")
        s_col, w_col = st.columns(2)
        with s_col:
            st.success(f"**Strengths** ({len(report['strengths'])})")
            for s in report["strengths"]:
                st.markdown(f"✅ {s}")
        with w_col:
            st.warning(f"**Needs Improvement** ({len(report['weaknesses'])})")
            for w in report["weaknesses"]:
                st.markdown(f"⚠️ {w}")

        st.markdown("#### Detailed Breakdown")
        for i, ans in enumerate(answers):
            score  = ans.get("score", 0)
            q_type = ans.get("type", "unknown")
            badge  = f'<span class="badge-{q_type}">{q_type.title()}</span>'
            icon   = "🟢" if score >= 7 else ("🟡" if score >= 5 else "🔴")

            with st.expander(f"Q{i+1}: {ans['question_text'][:70]}...  {icon} {score}/10", expanded=(i == 0)):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown("**Your Answer:**")
                lang = "python" if q_type == "coding" else None
                st.code(ans["answer_text"] or "(No answer provided)", language=lang)
                if ans.get("expected_answer"):
                    with st.expander("See Ideal Answer"):
                        st.code(ans["expected_answer"], language=lang)
                st.caption(f"**Evaluation:** {ans['evaluation']}")

    with t2:
        api_key = st.session_state.get("GEMINI_API_KEY", "")
        if api_key:
            if "deep_ai_report" not in st.session_state:
                with st.spinner("🤖 Generating Gemini deep analysis..."):
                    deep_report = generate_deep_ai_report(answers, api_key)
                    st.session_state["deep_ai_report"] = deep_report
            else:
                deep_report = st.session_state["deep_ai_report"]

            verdict_color = {
                "Strong Hire": "#10b981",
                "Hire":        "#f59e0b",
                "No Hire":     "#ef4444",
            }.get(deep_report.get("hireability_verdict", "N/A"), "#64748b")

            st.markdown(
                f"### Hireability Verdict: "
                f"<span style='color:{verdict_color};font-weight:800;'>"
                f"{deep_report.get('hireability_verdict','N/A')}</span>",
                unsafe_allow_html=True,
            )

            c_s, c_w = st.columns(2)
            with c_s:
                st.success("**Strengths**")
                for s in deep_report.get("strengths", []):
                    st.markdown(f"- {s}")
            with c_w:
                st.warning("**Areas to Improve**")
                for w in deep_report.get("weaknesses", []):
                    st.markdown(f"- {w}")

            with st.container(border=True):
                st.markdown("**Deep Analysis**")
                st.write(deep_report.get("deep_analysis", ""))
                st.caption(deep_report.get("summary", ""))
        else:
            st.info("Add a valid Gemini API Key to unlock AI deep dive analysis.")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⬅️ Return to Dashboard", use_container_width=True):
        role = st.session_state.get("user_role", "candidate")
        st.session_state["current_page"] = f"{role}_dashboard"
        st.rerun()
