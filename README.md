# ⚖️ LexAI — Bharat Legal SaaS: AI Legal Intelligence & Advisory Platform

> An enterprise-grade, multi-stage AI legal intelligence platform for the Indian Legal Framework (including **BNS 2024**, **BNSS 2024**, and **BSA 2024**). Features cluster-aware **Pinecone RAG**, calibrated **Qwen 2.5** Indian law fallback, **Alibaba Qwen3-TTS 1.7B** voice narration, and an interactive dark glassmorphic Streamlit SaaS dashboard.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflow%20DAG-orange)](https://github.com/langchain-ai/langgraph)
[![Qwen 2.5](https://img.shields.io/badge/LLM-Qwen%202.5%2072B%20/%20Groq-purple)](https://openrouter.ai)
[![Qwen3-TTS](https://img.shields.io/badge/Voice-Qwen3--TTS%201.7B-blue)](https://github.com/QwenLM)
[![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-000000?logo=pinecone)](https://www.pinecone.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents
1. [Overview & Key Capabilities](#-overview--key-capabilities)
2. [System Architecture & Execution DAG](#-system-architecture--execution-dag)
3. [📋 Detailed Implementation Plan & Roadmap](#-detailed-implementation-plan--roadmap)
4. [Core Modules & Pipeline Components](#-core-modules--pipeline-components)
5. [Confidence-Based Fallback Architecture](#-confidence-based-fallback-architecture)
6. [Voice Engine: Alibaba Qwen3-TTS 1.7B](#-voice-engine-alibaba-qwen3-tts-17b)
7. [Repository Structure](#-repository-structure)
8. [Installation & Setup](#-installation--setup)
9. [Verification & Test Results](#-verification--test-results)
10. [Author & License](#-author--license)

---

## 🚀 Overview & Key Capabilities

**LexAI** transforms statutory Indian legal codes and procedures into accessible, structured guidance for citizens, advocates, and legal researchers.

### 🎯 Key Capabilities
- 🏛️ **Indian Criminal Law Overhaul Ready**: Native mapping across the 2024 criminal codes:
  - **BNS 2024** ⇄ **IPC 1860** (e.g. BNS § 318 ⇄ IPC § 420 Cheating; BNS § 103 ⇄ IPC § 302 Murder)
  - **BNSS 2024** ⇄ **CrPC 1973** (e.g. BNSS § 482 ⇄ CrPC § 438 Anticipatory Bail; BNSS § 35 ⇄ CrPC § 41A Notice)
  - **BSA 2024** ⇄ **IEA 1872** (Digital evidence, electronic certificate standards)
- 📚 **Cluster-Aware Vector Retrieval**: 2-stage Pinecone retrieval scoped to KMeans statutory document clusters.
- ⚡ **Confidence-Based Qwen 2.5 Fallback**: Evaluates retrieval confidence score; if confidence is low ($< 0.50$), key is missing, or Pinecone is offline, routes seamlessly to fine-tuned Indian Law reasoning via OpenRouter Qwen 2.5 / Groq.
- 🔊 **Voice Accessibility (Qwen3-TTS 1.7B)**: In-app voice playback synthesizing natural legal advisories for audio accessibility.
- 🗺️ **Interactive Action Roadmap**: Persistent checklist tracking evidence gathering, legal notice dispatch, and court filings.
- 📄 **Multi-Format Export**: One-click consultation report generation in Markdown and publication-ready PDF.
- 🛡️ **Zero Telemetry Data Sovereignty**: 100% private local persistence in SQLite (`data/lexai.db`).

---

## 🧠 System Architecture & Execution DAG

```mermaid
flowchart TD
    U(["👤 Citizen / Legal Query"]) --> DC["🔍 Domain Classifier Node"]
    DC -->|Non-Law Query| GL["💬 General LLM Node"] --> FIN(["🏁 Final System Output"])
    DC -->|Indian Law Query| QR["✏️ Query Refinement (Statutory Mapping)"]
    
    QR --> RC["⚠️ Risk Classifier (Low / Medium / High)"]
    RC --> RAG{"📚 Pinecone Vector Search<br>Confidence Score >= 0.50?"}
    
    RAG -->|Yes: High Confidence| RM["🗺️ Procedural Action Roadmap"]
    RM --> LE["⚖️ Legal Expert Synthesis (Groq/Llama 3)"] --> FIN
    
    RAG -->|No: Low Confidence / Key Missing / Offline| FB["⚡ Indian Law Fallback Engine (Qwen 2.5 / HF)"]
    FB --> TTS["🔊 Alibaba Qwen3-TTS 1.7B Voice Synthesis"]
    TTS --> FIN
```

---

## 📋 Detailed Implementation Plan & Roadmap

The development of LexAI is structured across **seven comprehensive implementation phases**, adhering to strict modular engineering, safety guardrails, and automated verification:

### Phase 1: Requirement Analysis & Architectural Blueprint
- **1.1 Statutory Scope Definition**: Mapped target legal domains (Tenancy, Criminal Defense, Cybercrime, Constitutional Writs, Family Disputes, Consumer Redressal).
- **1.2 Legal Transition Mapping**: Built the cross-reference mapping between colonial statutes (IPC, CrPC, Evidence Act) and the new 2024 enactments (BNS, BNSS, BSA).
- **1.3 LangGraph State Graph Design**: Defined `GraphState` TypedDict to encapsulate query tokens, cluster IDs, confidence scores, risk categories, and response metadata.

### Phase 2: Data Engineering & Cluster-Aware Vector Indexing
- **2.1 Document Ingestion**: Processed Indian statutory legal texts, landmark Supreme Court rulings, and model draft notices via `PyPDF`.
- **2.2 Semantic Chunking**: Implemented recursive chunking with 800-character segments and 150-character overlap preserving statutory sections.
- **2.3 Vector Embedding**: Embedded chunks using `sentence-transformers/all-MiniLM-L6-v2` into 384-dimensional dense vectors.
- **2.4 KMeans Clustering**: Partitioned vector space into 20 distinct clusters (`NUM_CLUSTERS = 20`) to enable 2-stage cluster-gated search in Pinecone.

### Phase 3: Multi-Stage Reasoning Pipeline
- **3.1 Domain Classifier Node**: Fast keyword and semantic filtering to distinguish legal inquiries from general conversations.
- **3.2 Query Refinement Node**: Translates colloquial citizen grievances into legally precise search queries citing governing Indian Acts.
- **3.3 Risk Assessment Node**: Evaluates urgency and personal liberty impact (e.g., imminent arrest threats, eviction cutoffs) and classifies risk as `LOW`, `MEDIUM`, or `HIGH`.
- **3.4 Structured Output Synthesis**: Enforces strict markdown sectioning (`Disclaimer`, `Lawyer Needed`, `Legal Summary`, `Relevant Laws`, `Action Roadmap`, `Important Rights`, `Final Note`).

### Phase 4: Calibrated Confidence Routing & Qwen 2.5 Fallback
- **4.1 Retrieval Confidence Scoring**: Calibrated cosine similarity score into a normalized $[0.0, 1.0]$ confidence index.
- **4.2 Gated Routing Logic**:
  $$\text{If } (\text{Score} < 0.50) \lor (\text{Status} \in \{\text{key\_missing}, \text{connection\_error}, \text{no\_docs}\}) \implies \text{Trigger Fallback}$$
- **4.3 Multi-Tier Fallback Hierarchy**:
  1. *Tier 1*: **OpenRouter Qwen 2.5** (`qwen/qwen-2.5-72b-instruct` / `32b` / `7b:free`) — Top open-weight model for legal reasoning and statutory precision.
  2. *Tier 2*: **Hugging Face Inference API** (`law-ai/InLegalBERT` or custom models).
  3. *Tier 3*: **Groq Qwen** (`qwen/qwen3.8-27b`) ultra-fast fallback.
  4. *Tier 4*: **Safe Grounded Synthesis** — Offline emergency fallback guaranteeing zero application crashes.
- **4.4 Telemetry Tracking**: Automatically records `answer_source`, `confidence_score`, `fallback_reason`, `retrieved_documents_count`, and `response_status`.

### Phase 5: Voice Accessibility Engine (Alibaba Qwen3-TTS 1.7B)
- **5.1 TTS Architecture**: Integrated Alibaba Qwen3-TTS 1.7B parameter multilingual speech generation model (`core/tts.py`).
- **5.2 Dual-Tier Voice Delivery**:
  - *Cloud Synthesis*: Posts to Hugging Face / OpenRouter audio endpoint for neural waveform generation.
  - *Browser Web Speech Synthesis*: Instant zero-latency client-side speech playback across desktop and mobile browsers.
- **5.3 Disk Caching**: Caches synthesized audio in `data/audio_cache/` using MD5 query hashing to prevent redundant API latency.
- **5.4 Smart Guardrail**: Prevents runtime type errors if a user specifies a TTS model as their text LLM by automatically delegating text to Qwen 2.5 Instruct while reserving Qwen3-TTS for audio playback.

### Phase 6: Streamlit UI & Glassmorphism Dashboard Rebuild
- **6.1 Design System**: Built a modern dark glassmorphic UI using custom CSS (`ui/styles.css`) with smooth hover micro-interactions, glowing metric cards, and responsive grids.
- **6.2 Dashboard & KPI Analytics**: Real-time telemetry cards showing Total Consultations, High Risk Alerts, Knowledge Base Size, and Database Footprint.
- **6.3 BNS ⇄ IPC Matrix Page**: Interactive lookup matrix for statutory conversions under the 2024 overhaul.
- **6.4 Legal Draft Studio**: Automated document drafter for Legal Notices, RTI Applications, Tenant Deposit Demands, and Consumer Complaints.
- **6.5 Document Analyzer**: Upload PDF/TXT agreements for automated clause risk analysis and red-flag detection.

### Phase 7: Security, Testing, Data Sovereignty & Deployment
- **7.1 Secret Scanning & Git Protection**: Clean separation of sensitive API credentials into local `.env` (git-ignored) with sanitized `.env.example`.
- **7.2 Automated Verification Suite**: Built `verify_app.py` testing database operations, fallback routing, prompt formatting, and exports.
- **7.3 Production Deployment**: Git remote setup and automated push tracking on GitHub `main` branch.

---

## 🧩 Core Modules & Pipeline Components

| Module | File | Responsibility |
| :--- | :--- | :--- |
| **App Entry Point** | `app.py` | Master Streamlit dashboard, navigation, and session state manager. |
| **Pipeline DAG** | `core/pipeline.py` | LangGraph workflow DAG and interactive step generator `run_query()`. |
| **Vector RAG** | `core/rag.py` | Pinecone 2-stage cluster search and calibrated confidence evaluator. |
| **Indian Law Fallback** | `core/hf_fallback.py` | Multi-tier Qwen 2.5, Hugging Face, and grounded statutory synthesis. |
| **Voice TTS Engine** | `core/tts.py` | Alibaba Qwen3-TTS 1.7B voice synthesis and audio cache manager. |
| **Database Client** | `db/database.py` | SQLite persistence for consultations, messages, bookmarks, and checklist progress. |
| **Dashboard UI** | `ui/dashboard.py` | KPI metric cards, quick-action chips, and system status indicators. |
| **Analytics Telemetry**| `ui/analytics.py` | Visual telemetry charts for risk distribution, pipeline latency, and feedback. |
| **UI Components** | `ui/components.py` | Glassmorphic cards, statutory pills, interactive checklist, and SOS drawer. |
| **Export Engine** | `utils/export.py` | Encoded Markdown and PDF generation using `fpdf2`. |

---

## ⚡ Confidence-Based Fallback Architecture

| Condition | Metric / Status | Routing Decision | Primary Engine |
| :--- | :---: | :---: | :---: |
| **High Document Relevance** | Score $\ge 0.50$ | Verified Knowledge Base RAG | Pinecone + Groq LLM |
| **Weak Vector Match** | Score $< 0.50$ | Low-Confidence Fallback | OpenRouter / Qwen 2.5 72B |
| **Unconfigured Pinecone Key**| `key_missing` | Automatic Fallback | Qwen 2.5 / Groq Qwen |
| **Pinecone Network Failure** | `connection_error` | Automatic Fallback | Offline Grounded Indian Law |
| **General Non-Legal Inquiry** | Domain = `general` | General Conversational LLM | Fast Direct LLM Response |

---

## 🔊 Voice Engine: Alibaba Qwen3-TTS 1.7B

LexAI incorporates **Alibaba Qwen3-TTS (1.7B Variant)** for voice accessibility:
- **Model Identifier**: `qwen/qwen3-tts-1.7b` (or `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- **Interactive UI**: Every consultation response includes a **"🔊 Listen to Legal Advice (Qwen3-TTS 1.7B)"** button.
- **Dual Synthesis**: High-fidelity neural TTS when network API keys are present, with seamless client-side browser speech synthesis as a zero-latency fallback.

---

## 📂 Repository Structure

```
indian_law-/
├── app.py                       # Master Streamlit SaaS application
├── verify_app.py                # Automated 7-point verification test suite
├── requirements.txt             # Production dependencies
├── README.md                    # Platform documentation & implementation plan
├── PROJECT_OUTPUT_REPORT.md     # Comprehensive project output & test report
├── PROJECT_OUTPUT_REPORT.html   # Styled printable HTML submission report
├── .env.example                 # Sanitized configuration template
├── .gitignore                   # Excludes .env, *.db, audio cache, logs
│
├── core/                        # Intelligence & Pipeline Engine
│   ├── config.py                # Configuration loader and threshold settings
│   ├── pipeline.py              # LangGraph multi-stage DAG & interactive generator
│   ├── rag.py                   # Pinecone cluster search & confidence scoring
│   ├── hf_fallback.py           # Qwen 2.5 & Hugging Face Indian law fallback
│   ├── tts.py                   # Alibaba Qwen3-TTS 1.7B voice synthesis
│   ├── llm.py                   # Groq & OpenRouter inference handlers
│   ├── prompts.py               # Indian statutory prompt templates & guidelines
│   └── search.py                # SerpAPI live web search integration
│
├── ui/                          # Modern Glassmorphic Frontend
│   ├── components.py            # Answer cards, status pills, voice player, SOS
│   ├── dashboard.py             # SaaS overview dashboard & KPI metric cards
│   ├── analytics.py             # Telemetry visualizations & consultation insights
│   └── styles.css               # Dark theme CSS styling & micro-animations
│
├── db/                          # Local Data Sovereignty
│   └── database.py              # SQLite client (chats, bookmarks, roadmap progress)
│
├── utils/                       # Exporters & Helpers
│   ├── export.py                # PDF (fpdf2) and Markdown export generators
│   └── helpers.py               # Input sanitizers and response parsers
│
└── data/                        # Local Storage (Git-Ignored)
    ├── audio_cache/             # Synthesized Qwen3-TTS audio files
    └── lexai.db                 # Private SQLite consultation database
```

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/himanshu9330/indian_law-.git
cd indian_law-
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```

```dotenv
# === Core Primary Keys ===
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
SERPAPI_API_KEY=your_serpapi_api_key_here

# === OpenRouter Qwen Indian Law Reasoning ===
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=qwen/qwen-2.5-72b-instruct

# === Alibaba Qwen3-TTS Voice Engine ===
QWEN_TTS_MODEL=qwen/qwen3-tts-1.7b
ENABLE_TTS=true

# === Pipeline Confidence Thresholds ===
RAG_CONFIDENCE_THRESHOLD=0.5
RAG_TOP_K=3
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 🧪 Verification & Test Results

Run the automated test suite to verify pipeline integrity, fallback routing, and database operations:
```bash
python verify_app.py
```

### Expected Output:
```text
======================================================================
LexAI Automated Verification Suite
======================================================================
1. All modules (including hf_fallback and tts) imported successfully.
2. Database created with tables: ['users', 'chats', 'messages', 'feedback', 'bookmarks', 'roadmap_progress']
3. Stored and fetched test consultation messages.
4. Qwen3-TTS Voice Component verified successfully.
5. Fallback Engine generated response via 'openrouter/qwen-2.5-72b-instruct'.
6. Markdown export generated (772 bytes).
7. PDF export generated (40 bytes).

ALL TESTS PASSED! LexAI backend, Qwen fallback, and Qwen3-TTS are fully operational.
Exit Code: 0 (OK)
```

---

## 👨‍💻 Author & Contributions

**Himanshu**  
- GitHub: [@himanshu9330](https://github.com/himanshu9330)  
- Email: [himanshugupta9330@email.com](mailto:himanshugupta9330@email.com)  
- Project Repository: [https://github.com/himanshu9330/indian_law-](https://github.com/himanshu9330/indian_law-)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

> ⚠️ **Legal Disclaimer**: LexAI is an artificial intelligence platform designed for legal awareness, procedural education, and statutory research under Indian Law. It does not constitute formal legal representation or an attorney-client relationship under the Advocates Act, 1961. Always consult an enrolled advocate for formal judicial filings.
