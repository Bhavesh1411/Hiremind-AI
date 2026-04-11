"""
candidate_db.py
-----------------
Local SQLite database for HireMind AI — Stage 2 candidate identity storage.

Tables
------
candidates : Stores verified candidate identity records, including live photo.
"""

import sqlite3
import base64
import os
from pathlib import Path
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path("data") / "candidates.db"


# ── Initialisation ─────────────────────────────────────────────────────────────
def init_db() -> None:
    """Create the database file and schema if they don't already exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id          INTEGER  PRIMARY KEY AUTOINCREMENT,
            full_name   TEXT     NOT NULL,
            email       TEXT     NOT NULL,
            phone       TEXT     NOT NULL,
            photo_b64   TEXT,
            timestamp   TEXT     NOT NULL,
            session_id  TEXT     DEFAULT ''
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER,
            job_id       INTEGER DEFAULT 0,
            mode         TEXT, -- 'normal' or 'ai'
            ats_score    REAL DEFAULT 0, -- Stage 1 Score
            final_score  REAL DEFAULT 0, -- Stage 2 Avg Score
            hiring_status TEXT DEFAULT 'pending', -- 'pending', 'hired'
            status       TEXT DEFAULT 'in_progress',
            timestamp    TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(id)
        )
    """)


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interview_answers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER,
            question_idx   INTEGER,
            question_text  TEXT,
            answer_text    TEXT,
            expected_answer TEXT,
            score          REAL,
            similarity     REAL,
            evaluation     TEXT,
            type           TEXT,
            timestamp      TEXT,
            FOREIGN KEY(session_id) REFERENCES interview_sessions(id)
        )
    """)
    conn.commit()
    conn.close()


# ── Write ──────────────────────────────────────────────────────────────────────
def store_candidate_data(
    full_name: str,
    email: str,
    phone: str,
    photo_bytes: bytes = None,
    session_id: str = "",
) -> int:
    """
    Persist a verified candidate record.

    Parameters
    ----------
    full_name   : Candidate's legal full name.
    email       : Validated email address.
    phone       : Validated phone number.
    photo_bytes : Raw bytes of the captured webcam photo (JPEG / PNG).
    session_id  : Optional Streamlit session identifier for tracking.

    Returns
    -------
    int : Primary-key ID of the newly inserted row.
    """
    init_db()

    photo_b64  = base64.b64encode(photo_bytes).decode("utf-8") if photo_bytes else ""
    timestamp  = datetime.now().isoformat(sep=" ", timespec="seconds")

    conn   = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO candidates (full_name, email, phone, photo_b64, timestamp, session_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (full_name, email, phone, photo_b64, timestamp, session_id),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


# ── Read ───────────────────────────────────────────────────────────────────────
def get_candidate_by_email(email: str) -> dict:
    """
    Fetch the most recent candidate record matching the given email.

    Returns an empty dict if no record is found.
    """
    init_db()
    conn   = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, full_name, email, phone, timestamp
        FROM   candidates
        WHERE  email = ?
        ORDER  BY id DESC
        LIMIT  1
        """,
        (email,),
    )
    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            "id":        row[0],
            "full_name": row[1],
            "email":     row[2],
            "phone":     row[3],
            "timestamp": row[4],
        }
    return {}


def get_all_candidates() -> list:
    """
    Return a list of all candidate records (without photo data, for dashboards).
    """
    init_db()
    conn   = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, full_name, email, phone, timestamp FROM candidates ORDER BY id DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {"id": r[0], "full_name": r[1], "email": r[2], "phone": r[3], "timestamp": r[4]}
        for r in rows
    ]


# ── INTERVIEW STORAGE ──────────────────────────────────────────────────────────

def create_interview_session(candidate_id: int, mode: str, job_id: int = 0, ats_score: float = 0.0) -> int:
    """Initialize a new interview session record with Stage 1 scores."""
    init_db()
    timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interview_sessions (candidate_id, mode, job_id, ats_score, timestamp) VALUES (?, ?, ?, ?, ?)",
        (candidate_id, mode, job_id, ats_score, timestamp)
    )
    conn.commit()
    session_id = cursor.lastrowid

    conn.close()
    return session_id


def add_interview_answer(
    session_id: int,
    idx: int,
    question: str,
    answer: str,
    score: float,
    evaluation: str,
    q_type: str,
    similarity: float = 0.0,
    expected_answer: str = "",
):
    """Store an individual answer with silent evaluation results."""
    init_db()
    ts   = datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO interview_answers
        (session_id, question_idx, question_text, answer_text, expected_answer,
         score, similarity, evaluation, type, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, idx, question, answer, expected_answer,
         score, similarity, evaluation, q_type, ts)
    )
    conn.commit()
    conn.close()


def finalize_interview(session_id: int, final_score: float, ats_score: float = 0.0):
    """Mark an interview session as completed and store scores."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE interview_sessions SET final_score = ?, ats_score = ?, status = 'completed' WHERE id = ?",
        (final_score, ats_score, session_id)
    )
    conn.commit()
    conn.close()


def update_hiring_status(session_id: int, status: str):
    """Update the hiring status (e.g., 'pending', 'hired')."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE interview_sessions SET hiring_status = ? WHERE id = ?",
        (status, session_id)
    )
    conn.commit()
    conn.close()


def get_job_applicants(job_id: int) -> list:
    """Fetch all completed interview sessions for a specific job."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, c.full_name, c.email, c.phone 
        FROM interview_sessions s
        JOIN candidates c ON s.candidate_id = c.id
        WHERE s.job_id = ? AND s.status = 'completed'
        ORDER BY (s.ats_score + s.final_score * 10) DESC -- Rough sorting, will refine in Python
    """, (job_id,))
    applicants = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return applicants


def get_interview_results(session_id: int) -> dict:
    """Retrieve full session data and all answers for report generation."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get session info
    cursor.execute("SELECT * FROM interview_sessions WHERE id = ?", (session_id,))
    session = dict(cursor.fetchone())
    
    # Get answers
    cursor.execute("SELECT * FROM interview_answers WHERE session_id = ? ORDER BY question_idx ASC", (session_id,))
    answers = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    return {"session": session, "answers": answers}
