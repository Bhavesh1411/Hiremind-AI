"""
HireMind AI - Stage 2 Voice Interview UI
=========================================
Clean, production-ready Streamlit UI for the voice-based interview stage.

UX Design:
  - Record audio → "Transcribing..." → text auto-fills
  - Submit → next question loads immediately
  - Gemini evaluation runs silently in background (no UI blocking)
  - No manual text fields shown while audio is being processed
"""

import os
import time
import threading
import streamlit as st

from modules.voice_interview import (
    generate_voice_questions,
    text_to_speech,
    transcribe_audio,
    evaluate_voice_answer_async,
    evaluate_voice_answer,
    generate_combined_report,
)
from modules.candidate_db import get_interview_results, finalize_interview
from modules.webcam_monitor import render_webcam_monitor, reset_monitor_state


# ════════════════════════════════════════════════════════════════════════════════
#  STYLES
# ════════════════════════════════════════════════════════════════════════════════

def _apply_voice_styles():
    st.markdown("""
<style>
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }

    :root {
        --ok:     #10b981;
        --warn:   #f59e0b;
        --danger: #ef4444;
        --hdr:    linear-gradient(135deg, #1e293b 0%, #334155 100%);
    }

    .voice-header {
        background: var(--hdr);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }

    .question-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
    }

    /* Transcript display box */
    .transcript-box {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 1rem;
        line-height: 1.6;
        color: #166534;
    }

    .badge-project_based  { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-skill_based    { background:#eff6ff; color:#1e40af; border:1px solid #93c5fd; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
    .badge-scenario_based { background:#fef3c7; color:#92400e; border:1px solid #fcd34d; padding:3px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }

    .verdict-banner {
        text-align: center;
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#  SESSION STATE HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _init_voice_state():
    defaults = {
        "voice_q_idx":        0,
        "voice_answers":      [],      # raw transcripts, one per question
        "voice_results":      [],      # {score, reasoning} — filled by bg threads
        "voice_eval_threads": [],      # background Thread objects
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _reset_voice_state():
    for key in [
        "voice_questions", "voice_q_idx", "voice_answers",
        "voice_results",   "voice_eval_threads",
    ]:
        st.session_state.pop(key, None)


# ════════════════════════════════════════════════════════════════════════════════
#  STAGE 2: VOICE INTERVIEW
# ════════════════════════════════════════════════════════════════════════════════

def render_voice_interview():
    """
    Main Stage 2 UI.

    Flow per question:
      1. Show question text + play TTS audio
      2. User records via st.audio_input()
      3. On recording received → transcribe (shows spinner "Transcribing...")
      4. Transcript stored in session; shown in a clean display box
      5. Submit → store transcript, fire background eval thread, advance index
      6. Next question renders immediately
    """
    _apply_voice_styles()
    _init_voice_state()

    gemini_key  = st.session_state.get("GEMINI_API_KEY", "")
    assembly_key = (
        st.session_state.get("ASSEMBLYAI_API_KEY")
        or os.getenv("ASSEMBLYAI_API_KEY", "")
    )

    # ── Generate questions once ────────────────────────────────────────────────
    if "voice_questions" not in st.session_state:
        resume_data = st.session_state.get("parsed_resume", {})
        jd_text     = st.session_state.get("jd_text", "")
        if not resume_data and not jd_text:
            st.error("⚠️ Resume or Job Description data not found. Please return to screening.")
            if st.button("← Back to Screening"):
                st.session_state["current_page"] = "screening"
                st.rerun()
            return
        with st.spinner("🤖 Generating personalised voice interview questions..."):
            qs = generate_voice_questions(resume_data, jd_text, gemini_key)
        st.session_state["voice_questions"] = qs
        st.rerun()

    questions = st.session_state["voice_questions"]
    q_idx     = st.session_state["voice_q_idx"]
    total_q   = len(questions)

    # ── All done → wait for bg threads then go to report ──────────────────────
    if q_idx >= total_q:
        _wait_for_evaluations_and_advance()
        return

    # ── Progress header ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="voice-header">
        <h2 style="margin:0; font-size:1.5rem;">🎙️ Voice Interview — Stage 2</h2>
        <p style="margin:0.3rem 0 0; color:#94a3b8;">
            Question {q_idx + 1} of {total_q}
            &nbsp;|&nbsp; Speak clearly into your microphone
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.progress((q_idx + 1) / total_q)

    # ── 3-Column Layout ────────────────────────────────────────────────────────
    col_cam, col_q, col_a = st.columns([1, 2, 2])

    # Webcam proctoring ────────────────────────────────────────────────────────
    with col_cam:
        monitor_ok, _ = render_webcam_monitor()
        if not monitor_ok:
            _handle_proctoring_violation()
            return

    current_q = questions[q_idx]
    q_type    = current_q.get("type", "general")
    q_text    = current_q.get("question", "")

    # Question + TTS ───────────────────────────────────────────────────────────
    with col_q:
        badge = f"badge-{q_type}"
        st.markdown(
            f'<span class="{badge}">{q_type.replace("_", " ").title()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
        <div class="question-card">
            <p style="font-size:1.05rem;font-weight:600;margin:0;color:#1e293b;">
                {q_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

        tts_key = f"voice_tts_{q_idx}"
        if tts_key not in st.session_state:
            with st.spinner("🔊 Preparing audio..."):
                st.session_state[tts_key] = text_to_speech(q_text)

        tts_bytes = st.session_state.get(tts_key)
        if tts_bytes:
            st.markdown("**🔊 Listen to the question:**")
            st.audio(tts_bytes, format="audio/mp3", autoplay=True)
        else:
            st.info("🔇 Audio unavailable — read the question above.")

    # Answer recording ─────────────────────────────────────────────────────────
    with col_a:
        st.markdown("**🎤 Record your answer:**")

        # Key for already-transcribed answers this session
        transcript_cache_key = f"voice_transcript_{q_idx}"

        # ── If already transcribed, show result box + Submit button
        if transcript_cache_key in st.session_state:
            saved_transcript = st.session_state[transcript_cache_key]
            st.success("✅ Transcription complete!")
            st.markdown(
                f'<div class="transcript-box">💬 {saved_transcript}</div>',
                unsafe_allow_html=True,
            )

            if st.button(
                "✅ Submit & Continue →",
                type="primary",
                use_container_width=True,
                key=f"submit_{q_idx}",
            ):
                _submit_answer(q_idx, current_q, saved_transcript, gemini_key)
            return  # Don't re-show the recorder widget

        # ── Show recorder widget
        recorded_audio = st.audio_input(
            "Click the mic to record your answer",
            key=f"audio_input_{q_idx}",
        )

        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()

            if not audio_bytes or len(audio_bytes) < 500:
                st.warning("⚠️ Recording too short or empty. Please try again.")
                return

            # ── Transcribe (only spinner, nothing else)
            with st.spinner("⏳ Transcribing..."):
                transcript = transcribe_audio(audio_bytes, assembly_api_key=assembly_key)

            if transcript:
                # Cache transcript and rerun to show clean result
                st.session_state[transcript_cache_key] = transcript
                st.rerun()
            else:
                st.error(
                    "❌ Transcription failed. Check your microphone and try recording again.\n\n"
                    "If the issue persists, the AssemblyAI key may be invalid or the audio format "
                    "is unsupported."
                )


# ════════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _submit_answer(q_idx: int, question: dict, transcript: str, gemini_key: str):
    """
    Store the answer, fire a background evaluation thread, advance the index.
    No UI blocking — the next question loads immediately.
    """
    # Store the transcript
    st.session_state["voice_answers"].append(transcript)

    # Placeholder `None` so list indices stay in sync; bg thread fills it in
    result_store: list = []
    st.session_state["voice_results"].append(None)  # placeholder
    answer_index = len(st.session_state["voice_answers"]) - 1

    def _bg_eval():
        """Background thread: evaluate and patch back into session results."""
        evaluate_voice_answer_async(question, transcript, gemini_key, result_store)
        result = result_store[0] if result_store else {"score": 0, "reasoning": "Failed."}
        # Patch the placeholder at the correct index
        try:
            st.session_state["voice_results"][answer_index] = result
        except Exception:
            pass  # session may have been reset

    t = threading.Thread(target=_bg_eval, daemon=True)
    t.start()
    st.session_state["voice_eval_threads"].append(t)

    # Advance immediately — no waiting
    st.session_state["voice_q_idx"] += 1
    st.rerun()


def _wait_for_evaluations_and_advance():
    """
    Called when all questions are answered.
    Waits (max 30 s) for any still-running background evaluation threads,
    then resolves any remaining None placeholders before navigating to the report.
    """
    threads: list = st.session_state.get("voice_eval_threads", [])
    alive = [t for t in threads if t.is_alive()]

    if alive:
        with st.spinner("🎊 Voice Interview Complete! Finalising your evaluation..."):
            deadline = time.time() + 30
            while time.time() < deadline:
                if not any(t.is_alive() for t in alive):
                    break
                time.sleep(0.5)

    # Patch any None results that failed
    results: list = st.session_state.get("voice_results", [])
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"score": 0, "reasoning": "Evaluation timed out."}
    st.session_state["voice_results"] = results

    # Finalise Stage 1 & 2 in DB
    session_id = st.session_state.get("interview_session_id")
    if session_id:
        # ── 1. Get Stage 1 (Text) results from DB
        stage1_data = get_interview_results(session_id)
        s1_answers  = stage1_data.get("answers", [])
        avg_s1      = (sum(a["score"] for a in s1_answers) / len(s1_answers)) if s1_answers else 0.0

        # ── 2. Get Stage 2 (Voice) results from Session
        v_results   = st.session_state.get("voice_results", [])
        avg_v       = (sum(r["score"] for r in v_results) / len(v_results)) if v_results else 0.0

        # ── 3. Combine Interview Scores (Stage 2 part of the user's formula)
        # Normalize to 100: (Avg 0-10 + Avg 0-10) / 2 * 10
        interview_score_100 = ((avg_s1 + avg_v) / 2) * 10

        # ── 4. Retrieve Stage 1 (ATS) Score
        ats_res = st.session_state.get("ats_result", {})
        ats_score = ats_res.get("ats_score", 0.0)
        
        # [DEBUG LOGGING]
        print(f"[STAGE 1 - SAVE] Session: {session_id} | ATS Score: {ats_score}%")
        print(f"[STAGE 2 - SAVE] Session: {session_id} | Final Score: {interview_score_100}%")

        # ── 5. Finalise in DB
        finalize_interview(session_id, final_score=interview_score_100, ats_score=ats_score)

        
        st.session_state["stage1_answers"] = s1_answers

    reset_monitor_state()

    st.session_state["current_page"] = "final_report"
    st.rerun()


def _handle_proctoring_violation():
    """Called when 2 proctoring strikes are detected — reset and restart."""
    reset_monitor_state()
    _reset_voice_state()
    st.session_state["current_q_idx"]     = 0
    st.session_state["interview_answers"] = {}
    st.session_state.pop("interview_mode", None)
    st.session_state["current_page"] = "interview_mode"
    time.sleep(2)
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
#  COMBINED FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════════

def render_combined_report():
    """
    Unified Stage 1 + Stage 2 performance dashboard.
    Called by the 'final_report' route in app.py.
    """
    _apply_voice_styles()

    gemini_key = st.session_state.get("GEMINI_API_KEY", "")
    session_id = st.session_state.get("interview_session_id")
    candidate  = st.session_state.get("candidate_name", "Candidate")

    # ── Gather Stage 1 data ───────────────────────────────────────────────────
    s1_answers = st.session_state.get("stage1_answers", [])
    if not s1_answers and session_id:
        s1_answers = get_interview_results(session_id).get("answers", [])

    s2_results      = st.session_state.get("voice_results", [])
    voice_questions = st.session_state.get("voice_questions", [])

    if not s1_answers and not s2_results:
        st.error("No interview data found. Please complete the interview first.")
        if st.button("← Return to Dashboard"):
            role = st.session_state.get("user_role", "candidate")
            st.session_state["current_page"] = f"{role}_dashboard"
            st.rerun()
        return

    # ── Build combined report (cached) ────────────────────────────────────────
    report_key = "combined_report_data"
    if report_key not in st.session_state:
        with st.spinner("📊 Generating your comprehensive report..."):
            report = generate_combined_report(
                stage1_answers  = s1_answers,
                stage2_results  = [r for r in s2_results if r],
                voice_questions = voice_questions,
                api_key         = gemini_key,
            )
        st.session_state[report_key] = report
    else:
        report = st.session_state[report_key]

    s1       = report["stage1"]
    s2       = report["stage2"]
    combined = report["combined"]
    ai       = report.get("ai_analysis", {})

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="voice-header">
        <h2 style="margin:0;">📊 HireMind AI — Final Performance Report</h2>
        <p style="margin:0.4rem 0 0; color:#94a3b8;">
            Candidate: <strong style="color:white;">{candidate}</strong>
            &nbsp;|&nbsp; Comprehensive Two-Stage Evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Verdict Banner ────────────────────────────────────────────────────────
    vc = combined["verdict_color"]
    st.markdown(f"""
    <div class="verdict-banner" style="background:{vc}22;border:2px solid {vc};">
        <div style="font-size:3rem;">{combined["verdict_icon"]}</div>
        <h1 style="margin:0.5rem 0;color:{vc};font-size:2.5rem;font-weight:800;">
            {combined["verdict"]}
        </h1>
        <p style="font-size:1.4rem;color:#1e293b;font-weight:700;">
            Combined Score: {combined["percentage"]}%
        </p>
        <p style="color:#64748b;font-size:0.95rem;">
            Stage 1 (60%) · Stage 2 (40%) weighted evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Combined",    f"{combined['percentage']}%")
    m2.metric("📝 Stage 1",     f"{s1['percentage']}%",  delta=f"{s1['total']}/{s1['max']} pts")
    m3.metric("🎙️ Stage 2",    f"{s2['percentage']}%" if s2["results"] else "N/A",
              delta=f"Avg {s2['avg_score']}/10" if s2["results"] else "")
    m4.metric("📋 Questions",   f"{len(s1_answers)} + {len(s2_results)}")

    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_s1, tab_s2, tab_ai = st.tabs([
        "🏆 Combined Analysis",
        "📝 Stage 1: Technical Interview",
        "🎙️ Stage 2: Voice Interview",
        "🧠 AI Hiring Recommendation",
    ])

    # Overview ─────────────────────────────────────────────────────────────────
    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("**Stage 1 — Text / Coding Interview** (60% weight)")
                st.progress(s1["percentage"] / 100)
                st.metric("Score", f"{s1['percentage']}%", f"{s1['total']}/{s1['max']} pts")
        with c2:
            with st.container(border=True):
                st.markdown("**Stage 2 — Voice Interview** (40% weight)")
                if s2["results"]:
                    st.progress(s2["percentage"] / 100)
                    st.metric("Score", f"{s2['percentage']}%", f"Avg {s2['avg_score']}/10")
                else:
                    st.info("Stage 2 not completed")

        cs, cw = st.columns(2)
        with cs:
            st.success(f"**✅ Strengths ({len(s1['strengths'])})**")
            for item in s1["strengths"]:
                st.markdown(f"- {item}…")
        with cw:
            st.warning(f"**⚠️ Areas to Improve ({len(s1['weaknesses'])})**")
            for item in s1["weaknesses"]:
                st.markdown(f"- {item}…")

    # Stage 1 detail ───────────────────────────────────────────────────────────
    with tab_s1:
        st.markdown("### 📝 Stage 1: Technical & Coding Interview")
        if not s1_answers:
            st.info("No Stage 1 answers found.")
        else:
            for i, ans in enumerate(s1_answers):
                score  = ans.get("score", 0)
                q_type = ans.get("type", "unknown")
                icon   = "🟢" if score >= 7 else ("🟡" if score >= 5 else "🔴")
                with st.expander(
                    f"Q{i+1}: {ans.get('question_text','')[:65]}…  {icon} {score}/10",
                    expanded=False,
                ):
                    st.markdown(f'<span class="badge-{q_type}">{q_type.title()}</span>',
                                unsafe_allow_html=True)
                    lang = "python" if q_type == "coding" else None
                    st.markdown("**Your Answer:**")
                    st.code(ans.get("answer_text") or "(No answer provided)", language=lang)
                    if ans.get("expected_answer"):
                        with st.expander("See Ideal Answer"):
                            st.code(ans["expected_answer"], language=lang)
                    st.caption(f"**Evaluation:** {ans.get('evaluation', '')}")

    # Stage 2 detail ───────────────────────────────────────────────────────────
    with tab_s2:
        st.markdown("### 🎙️ Stage 2: Voice Interview")
        if not s2["results"]:
            st.info("Stage 2 was not completed or no results found.")
        else:
            voice_answers = st.session_state.get("voice_answers", [])
            for i, res in enumerate(s2["results"]):
                if not res:
                    continue
                score  = res.get("score", 0)
                icon   = "🟢" if score >= 7 else ("🟡" if score >= 5 else "🔴")
                q_data = voice_questions[i] if i < len(voice_questions) else {}
                q_type = q_data.get("type", "general")
                q_text = q_data.get("question", f"Question {i+1}")
                with st.expander(f"V{i+1}: {q_text[:65]}…  {icon} {score}/10", expanded=False):
                    badge = f"badge-{q_type}"
                    st.markdown(f'<span class="{badge}">{q_type.replace("_"," ").title()}</span>',
                                unsafe_allow_html=True)
                    st.markdown("**Question:**")
                    st.markdown(f"> {q_text}")
                    if i < len(voice_answers):
                        st.markdown("**Your recorded answer:**")
                        st.markdown(
                            f'<div class="transcript-box">💬 {voice_answers[i]}</div>',
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**Score:** {score}/10")
                    st.markdown(f"**AI Evaluation:** {res.get('reasoning', '')}")

    # AI Recommendation ────────────────────────────────────────────────────────
    with tab_ai:
        st.markdown("### 🧠 AI Hiring Recommendation")
        if not ai:
            if not gemini_key:
                st.info("Add a Gemini API key to unlock AI-powered hiring recommendations.")
            else:
                st.info("AI analysis could not be generated. Check API connectivity.")
        else:
            st.markdown(f"""
            <div style="background:{vc}15;border:1px solid {vc};
                        border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
                <h3 style="margin:0;color:{vc};">
                    {combined["verdict_icon"]} {combined["verdict"]}
                </h3>
                <p style="color:#374151;margin:0.5rem 0 0;">
                    {ai.get("overall_summary","No summary available.")}
                </p>
            </div>
            """, unsafe_allow_html=True)

            k_col, w_col = st.columns(2)
            with k_col:
                st.success("**Key Strengths**")
                for s in ai.get("key_strengths", []):
                    st.markdown(f"✅ {s}")
            with w_col:
                st.warning("**Key Weaknesses**")
                for w in ai.get("key_weaknesses", []):
                    st.markdown(f"⚠️ {w}")

            if ai.get("skill_gaps"):
                with st.container(border=True):
                    st.markdown("**📉 Skill Gaps**")
                    for g in ai["skill_gaps"]:
                        st.markdown(f"- {g}")

            if ai.get("hiring_recommendation"):
                with st.container(border=True):
                    st.markdown("**📋 Hiring Recommendation**")
                    st.write(ai["hiring_recommendation"])

            if ai.get("development_areas"):
                with st.container(border=True):
                    st.markdown("**🛠️ Development Areas**")
                    for a in ai["development_areas"]:
                        st.markdown(f"- {a}")

    # Footer ───────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_r, col_d = st.columns(2)
    with col_r:
        if st.button("⬅️ Return to Dashboard", use_container_width=True):
            role = st.session_state.get("user_role", "candidate")
            st.session_state["current_page"] = f"{role}_dashboard"
            st.rerun()
    with col_d:
        if st.button("🔄 Start New Interview", use_container_width=True):
            for key in [
                "voice_questions", "voice_q_idx", "voice_answers",
                "voice_results",   "voice_eval_threads",
                "interview_answers", "interview_mode", "interview_session_id",
                "stage1_answers",  "combined_report_data", "deep_ai_report",
            ]:
                st.session_state.pop(key, None)
            # Clear transcript caches
            for k in list(st.session_state.keys()):
                if k.startswith("voice_transcript_") or k.startswith("voice_tts_"):
                    del st.session_state[k]
            reset_monitor_state()
            st.session_state["current_page"] = "interview_mode"
            st.rerun()
