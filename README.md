# 🧠 HireMind AI: 🎯 Autonomous HR Operating System

HireMind AI is a comprehensive, intelligent platform designed to transform resume screening from a basic matching exercise into a deep, multi-dimensional analytical process. It serves as an **Autonomous HR Operating System (OS)**, integrating semantic search, integrity analysis, and candidate engagement into a single workspace.

---

## 🚀 Key Modules & Intelligence Layers

### 1. 📥 Proactive Data Ingestion
- High-fidelity text extraction from **PDF**, **DOCX**, and **TXT** files.
- Automated document cleaning and structural normalization.

### 2. 🧩 NLP Parsing & Entities
- Advanced entity extraction (Skills, Experience, Education, Contact).
- Converts raw resume text into structured **JSON schemas** for downstream analysis.

### 3. 🔢 Semantic Vector Matcher
- Powered by **FAISS (Facebook AI Similarity Search)**.
- Goes beyond keyword matching to find **contextual alignment** between candidate skills and job requirements.
- Uses vector embeddings (Gemini/OpenAI) to rank candidates by true competence.

### 4. 📊 ATS Integrity Scorer
- Analyzes formatting, section presence, and keyword density.
- Provides a deterministic **ATS Score (0–100)** with actionable suggestions to improve resume visibility.

### 5. 🕵️ Fraud & Integrity Intelligence
- Detects **Keyword Stuffing** (hidden or dense skills blocks).
- Identifies **Hidden Text** (white ink manipulation).
- Checks for **Skill Inconsistencies** (skills listed but not demonstrated) and **Temporal Logic** (impossible experience dates).

### 6. 🎯 Recommendation & Career Engine
- Automates **Skill Gap Analysis** (What's missing?).
- Maps tailored **Learning Paths** (Recommended courses, projects, and certifications).
- Suggests **Alternative Roles** based on the candidate's existing skill set.

### 7. 🚀 Interview UI & Access Control (v0.7)
- **Eligibility Filter**: Automated check for Interview Entry (Match ≥ 80%, ATS ≥ 75, Fraud = "Low").
- **AI Interview Engine**: A dedicated interface for digital screening rounds with progress tracking and answer capture.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Modern Interface)
- **Vector DB**: FAISS (Meta AI)
- **AI/LLM**: Google Gemini 1.5 Pro / OpenAI GPT-4
- **Speech Processing**: Whisper (STT), gTTS / ElevenLabs (TTS)
- **Computer Vision**: OpenCV, DeepFace (Behavioral Intelligence)
- **Backend**: FastAPI, Python 3.13
- **Analytics**: Librosa, pyAudioAnalysis, NLP SpaCy

---

## ⚙️ Quick Start & Deployment

### 1. Environment Setup
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_ai_key
LLM_PROVIDER=gemini    # or 'openai'
OPENAI_API_KEY=your_openai_key (optional)
```

### 2. Installation
```powershell
pip install -r requirements.txt
```

### 3. Run Locally
```powershell
streamlit run app.py
```

---

## 🛣️ Roadmap: Future Intelligence Phases
- [ ] **Phase 2-4**: Advanced Voice & Emotion Analysis (librosa, DeepFace).
- [ ] **Phase 5**: Adaptive Questioning (Real-time difficulty scaling).
- [ ] **Phase 6**: Automated Hiring Verdicts & Multi-Dimensional Reporting.

---
**Developed by Bhavesh** | *HireMind AI: The Future of Autonomous Recruitment.*
