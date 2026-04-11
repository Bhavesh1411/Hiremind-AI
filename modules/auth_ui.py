import streamlit as st
from modules.auth_db import (
    authenticate_user, create_user, get_all_jobs, add_job, apply_for_job
)
from modules.candidate_db import (
    get_all_candidates, get_job_applicants, update_hiring_status
)
from modules.email_service import send_hiring_email


def render_role_selection():
    st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1 style='color: #1e293b; font-size: 2.5rem;'>🧠 HireMind AI</h1>
            <p style='color: #64748b; font-size: 1.1rem;'>Autonomous HR Operating System — Identity & Access</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("### 🎓 Candidate")
        st.write("Find jobs, apply with your resume, and complete AI interviews.")
        if st.button("Access Candidate Portal", use_container_width=True, type="primary"):
            st.session_state["auth_role"] = "candidate"
            st.session_state["current_page"] = "login"
            st.rerun()

    with col2:
        st.warning("### 💼 Admin / Recruiter")
        st.write("Manage job postings, review candidate results, and monitor system performance.")
        if st.button("Access Admin Dashboard", use_container_width=True):
            st.session_state["auth_role"] = "admin"
            st.session_state["current_page"] = "login"
            st.rerun()

def render_login_page():
    role = st.session_state.get("auth_role", "candidate")
    st.markdown(f"### {'🎓 Candidate' if role == 'candidate' else '💼 Admin'} Login")
    
    if role == "admin":
        st.info("Signup is disabled for admins. Please contact the system owner for credentials.")
    
    with st.form(f"login_form_central"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            submit = st.form_submit_button("Login", use_container_width=True)
        with col_btn2:
            if st.form_submit_button("← Back", use_container_width=True):
                st.session_state["current_page"] = "role_selection"
                st.rerun()
                
        if submit:
            user = authenticate_user(email, password)
            if user:
                if user["role"] == role:
                    st.session_state["logged_in"] = True
                    st.session_state["user_data"] = user
                    st.session_state["user_role"] = role
                    st.session_state["current_page"] = f"{role}_dashboard"
                    st.success(f"Welcome back, {user['name']}!")
                    st.rerun()
                else:
                    st.error(f"Account exists but role is not {role}.")
            else:
                st.error("Invalid email or password.")

    if role == "candidate":
        st.markdown("---")
        st.subheader("New here? Create an account")
        render_signup("candidate")

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
    with st.form(f"signup_form_{role}"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")
        if st.form_submit_button("Sign Up"):
            if password != confirm:
                st.error("Passwords do not match.")
            elif not name or not email or not password:
                st.error("Please fill all fields.")
            else:
                user_id = create_user(name, email, password, role)
                if user_id:
                    st.success("Account created! Please login.")
                else:
                    st.error("Email already exists.")

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
