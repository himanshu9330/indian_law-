"""
Application configuration — loads .env and exposes all settings.
Every module imports from here rather than reading os.environ directly.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Try loading .env from multiple candidate locations
candidate_env_paths = [
    BASE_DIR / ".env",
    BASE_DIR.parent / ".env",
    Path.cwd() / ".env",
]
for env_path in candidate_env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

# Fallback: check if user added key to .env.example
if not os.getenv("GROQ_API_KEY"):
    for ex_path in [BASE_DIR / ".env.example", BASE_DIR.parent / ".env.example"]:
        if ex_path.exists():
            with open(ex_path, "r", encoding="utf-8") as f:
                content = f.read()
                m = re.search(r"GROQ_API_KEY\s*=\s*(gsk_[A-Za-z0-9_]+)", content)
                if m:
                    os.environ["GROQ_API_KEY"] = m.group(1).strip()
                    break

# ── API Keys ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip("\"' ")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "").strip("\"' ")
SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "").strip("\"' ")
HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY", os.getenv("HF_TOKEN", "")).strip("\"' ")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip("\"' ")

# ── Model settings ───────────────────────────────────────────────────────
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b").strip("\"' ")
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
PINECONE_INDEX: str = "law-suggestion-index"
HUGGINGFACE_MODEL: str = os.getenv("HUGGINGFACE_MODEL", "law-ai/InLegalBERT").strip("\"' ")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-72b-instruct").strip("\"' ")
QWEN_TTS_MODEL: str = os.getenv("QWEN_TTS_MODEL", "qwen/qwen3-tts-1.7b").strip("\"' ")
ENABLE_TTS: bool = os.getenv("ENABLE_TTS", "true").lower() in ("true", "1", "yes")
LLM_TEMPERATURE: float = 0.3


def save_env_key(key: str, value: str) -> None:
    """Save an API key to the local .env file."""
    env_file = BASE_DIR / ".env"
    lines = []
    found = False
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[i] = f"{key}={value.strip()}\n"
                found = True
                break
    if not found:
        lines.append(f"{key}={value.strip()}\n")
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.environ[key] = value.strip()

# ── RAG settings ─────────────────────────────────────────────────────────
RAG_THRESHOLD: float = float(os.getenv("RAG_THRESHOLD", "0.5"))
RAG_CONFIDENCE_THRESHOLD: float = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", os.getenv("RAG_THRESHOLD", "0.5")))
RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
NUM_CLUSTERS: int = 20

# ── Database ─────────────────────────────────────────────────────────────
DB_PATH: str = str(DATA_DIR / "lexai.db")

# ── Domain classification keywords ───────────────────────────────────────
LAW_KEYWORDS: list[str] = [
    # Criminal & Police
    "law", "ipc", "crpc", "bns", "bnss", "bsa", "fir", "police", "arrest",
    "bail", "jail", "convict", "accused", "suspect", "offence", "crime",
    # Courts & Procedure
    "court", "judge", "magistrate", "justice", "litigation", "case", "suit",
    "petition", "writ", "affidavit", "summon", "warrant", "bench",
    # Professionals
    "legal", "lawyer", "advocate", "counsel", "solicitor", "attorney",
    # Family & Personal
    "divorce", "marriage", "custody", "alimony", "maintenance", "succession",
    # Property & Civil
    "property", "lease", "tenant", "rent", "registry", "agreement", "contract",
    "compensation", "damages", "debt", "recovery", "landlord", "evict", "eviction", "deposit",
    # Constitutional & Rights
    "constitution", "fundamental rights", "article", "pil", "rights",
    # Crimes & Specific Acts
    "fraud", "scam", "harassment", "defamation", "posh", "dowry", "sc/st",
    "theft", "murder", "assault", "cybercrime", "cheating",
    # Labour
    "salary", "wages", "employer", "employee", "termination", "workplace",
    # Consumer
    "consumer", "refund", "complaint", "product", "service",
    # Procedural & Evidence (Crucial for follow-up inquiries)
    "document", "documents", "evidence", "proof", "fee", "fees", "cost", "costs",
    "limitation", "deadline", "period", "procedure", "process", "step", "steps",
    "action", "notice", "claim", "sue", "relief", "remedy", "section", "act",
    "clause", "statute", "tribunal", "jurisdiction", "appeal", "file", "filing",
]

# ── Suggested questions (by category) ────────────────────────────────────
STARTER_QUESTIONS: dict[str, list[str]] = {
    "🏠 Tenant & Property": [
        "My landlord is refusing to return my security deposit. What can I do?",
        "What are my rights as a tenant under Indian rent control laws?",
    ],
    "👩 Women's Rights": [
        "What are the rights of women during police investigation?",
        "How do I file a complaint under the POSH Act?",
    ],
    "🛡️ Criminal & FIR": [
        "How do I file an FIR if the police refuse to register one?",
        "What are my rights if I am arrested by the police?",
    ],
    "💻 Cyber Crime": [
        "Someone is blackmailing me online. What legal action can I take?",
        "How to report a cybercrime in India?",
    ],
    "🛒 Consumer Rights": [
        "I received a defective product and the company won't refund me.",
        "How do I file a complaint in consumer court?",
    ],
    "👨‍👩‍👧 Family Law": [
        "What is the process of filing for divorce in India?",
        "How is child custody decided in Indian courts?",
    ],
    "👷 Labour & Employment": [
        "My employer hasn't paid my salary for 3 months. What can I do?",
        "Can I be fired without notice in India?",
    ],
    "📜 General Legal": [
        "What is an RTI application and how do I file one?",
        "What are fundamental rights under the Indian Constitution?",
    ],
}

# ── Emergency helplines ──────────────────────────────────────────────────
HELPLINES: list[dict] = [
    {"name": "Police Emergency", "number": "112", "icon": "🚔", "desc": "All emergencies"},
    {"name": "Women Helpline", "number": "181", "icon": "👩", "desc": "Women in distress"},
    {"name": "Cyber Crime", "number": "1930", "icon": "💻", "desc": "Report cyber fraud"},
    {"name": "NALSA Free Legal Aid", "number": "15100", "icon": "⚖️", "desc": "Free legal assistance"},
    {"name": "Consumer Helpline", "number": "1915", "icon": "🛒", "desc": "Consumer complaints"},
    {"name": "Child Helpline", "number": "1098", "icon": "👶", "desc": "Child abuse/distress"},
    {"name": "Senior Citizen", "number": "14567", "icon": "🧓", "desc": "Elder abuse helpline"},
]

# ── BNS 2024 vs IPC 1860 & BNSS vs CrPC Verified Cross-Reference ────────
BNS_IPC_MAPPING: list[dict] = [
    {
        "topic": "Cheating & Financial Fraud",
        "old_law": "Section 420 IPC",
        "new_law": "Section 318 BNS",
        "severity": "Cognizable / Non-Bailable",
        "desc": "Dishonestly inducing delivery of property or alteration of valuable security.",
    },
    {
        "topic": "FIR Registration & e-FIR",
        "old_law": "Section 154 CrPC",
        "new_law": "Section 173 BNSS",
        "severity": "Mandatory Duty",
        "desc": "Mandatory registration of FIR for cognizable crimes; statutory right to e-FIR.",
    },
    {
        "topic": "Zero FIR Guarantee",
        "old_law": "Judicial Precedents",
        "new_law": "Section 173(1) BNSS",
        "severity": "Statutory Guarantee",
        "desc": "Any police station must register FIR irrespective of jurisdiction and transfer.",
    },
    {
        "topic": "Arrest Safeguards & Notice",
        "old_law": "Section 41 CrPC",
        "new_law": "Section 35 BNSS",
        "severity": "Civil Liberty",
        "desc": "Safeguards against arbitrary arrest; mandatory permission for elderly/infirm.",
    },
    {
        "topic": "Sexual Harassment",
        "old_law": "Section 354A IPC",
        "new_law": "Section 75 BNS",
        "severity": "Cognizable / Bailable",
        "desc": "Unwelcome physical contact, advances, demanding sexual favours, or pornography.",
    },
    {
        "topic": "Bail for First-Time Undertrials",
        "old_law": "Section 436A CrPC",
        "new_law": "Section 479 BNSS",
        "severity": "Statutory Bail",
        "desc": "Mandatory bail if first-time offender has served 1/3rd of maximum sentence.",
    },
    {
        "topic": "Hit and Run with Escape",
        "old_law": "Section 304A IPC",
        "new_law": "Section 106(2) BNS",
        "severity": "Up to 10 Yrs Imprisonment",
        "desc": "Causing death by rash driving and fleeing scene without reporting to police.",
    },
    {
        "topic": "Theft & Community Service",
        "old_law": "Section 378/379 IPC",
        "new_law": "Section 303 BNS",
        "severity": "Community Service Option",
        "desc": "Dishonest taking of property; introduces community service for theft under ₹5,000.",
    },
    {
        "topic": "Criminal Breach of Trust",
        "old_law": "Section 405/406 IPC",
        "new_law": "Section 316 BNS",
        "severity": "Cognizable / Non-Bailable",
        "desc": "Misappropriation of property entrusted to an agent, banker, carrier, or employee.",
    },
    {
        "topic": "Defamation",
        "old_law": "Section 499/500 IPC",
        "new_law": "Section 356 BNS",
        "severity": "Community Service Option",
        "desc": "Harming reputation by words, signs, or publications; community service reform.",
    },
]

# ── Legal glossary (mini) ────────────────────────────────────────────────
LEGAL_GLOSSARY: dict[str, str] = {
    "FIR": "First Information Report — the first step to register a criminal complaint with the police.",
    "IPC": "Indian Penal Code — the main criminal law code (now replaced by BNS in 2023).",
    "BNS": "Bharatiya Nyaya Sanhita — the new criminal law code replacing the IPC from July 2024.",
    "CrPC": "Code of Criminal Procedure — governs the process of criminal law enforcement (now replaced by BNSS).",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita — new code replacing CrPC from July 2024.",
    "BSA": "Bharatiya Sakshya Adhiniyam — new evidence act replacing the Indian Evidence Act.",
    "PIL": "Public Interest Litigation — a case filed for the benefit of the public, not just the petitioner.",
    "Bail": "Temporary release of an accused person while awaiting trial, with conditions.",
    "Cognizable Offence": "An offence where the police can arrest without a warrant and start investigation.",
    "Non-Cognizable Offence": "An offence where the police need a court's permission to investigate.",
    "Writ Petition": "A formal written order issued by a High Court or Supreme Court.",
    "Affidavit": "A written sworn statement of fact used as evidence in court.",
    "POSH Act": "Prevention of Sexual Harassment at Workplace Act, 2013.",
    "RTI": "Right to Information — a law that lets citizens request information from public authorities.",
    "Anticipatory Bail": "Bail obtained in advance, before an arrest is made.",
    "Summons": "A legal document ordering a person to appear in court.",
    "Adjournment": "Postponement of a court hearing to another date.",
    "Acquittal": "A court judgment that a person is not guilty of the crime charged.",
    "Arbitration": "Settling a dispute outside of court using a neutral third party.",
    "Caveat": "A formal notice to the court that the other party should be heard before any order is passed.",
}
