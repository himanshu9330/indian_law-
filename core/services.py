"""
LexAI Core Legal Services
Reusable business logic for BNS-IPC Matrix, Legal Draft Studio,
Document Analyzer, Statutory Glossary, and Emergency Helplines.
"""

from typing import Any, Optional
from core.config import BNS_IPC_MAPPING, HELPLINES, LEGAL_GLOSSARY
from core.prompts import DRAFT_TEMPLATES, DOCUMENT_SUMMARY
from core.llm import call_llm


def get_bns_ipc_mappings(filter_text: str = "") -> list[dict[str, Any]]:
    """Return BNS vs IPC mappings, optionally filtered by keyword."""
    if not filter_text or not filter_text.strip():
        return BNS_IPC_MAPPING
    q = filter_text.strip().lower()
    return [
        item for item in BNS_IPC_MAPPING
        if (
            q in item.get("topic", "").lower()
            or q in item.get("old_law", "").lower()
            or q in item.get("new_law", "").lower()
            or q in item.get("desc", "").lower()
            or q in item.get("severity", "").lower()
        )
    ]


def get_statutory_glossary(filter_text: str = "") -> dict[str, str]:
    """Return statutory glossary terms, optionally filtered by keyword."""
    if not filter_text or not filter_text.strip():
        return LEGAL_GLOSSARY
    q = filter_text.strip().lower()
    return {
        term: definition
        for term, definition in LEGAL_GLOSSARY.items()
        if q in term.lower() or q in definition.lower()
    }


def get_emergency_helplines() -> list[dict[str, str]]:
    """Return verified national legal and crisis helplines in India."""
    return HELPLINES


def generate_legal_draft(draft_type: str, form_data: dict[str, str]) -> str:
    """Generate a formal legal draft based on template type and field values."""
    template = DRAFT_TEMPLATES.get(draft_type)
    if not template:
        raise ValueError(f"Unknown draft template: {draft_type}")

    # Build safe prompt mapping
    prompt_text = template["prompt"].format(**form_data)
    system_prompt = (
        "You are an expert Indian advocate specializing in legal drafting. "
        "Produce an accurate, formal, and professionally formatted legal draft. "
        "Use appropriate statutory clauses, jurisdiction phrases, and standard verification clauses."
    )
    return call_llm(prompt=prompt_text, system_prompt=system_prompt, temperature=0.2)


def analyze_legal_document(document_text: str) -> str:
    """Analyze legal document text for parties, summary, important clauses, and red flags."""
    if not document_text or not document_text.strip():
        raise ValueError("Document text cannot be empty.")

    prompt_text = DOCUMENT_SUMMARY.format(document_text=document_text[:8000])
    system_prompt = (
        "You are a Senior Legal Counsel in India specializing in document analysis and risk audit. "
        "Provide a structured, unbiased evaluation adhering strictly to Indian law."
    )
    return call_llm(prompt=prompt_text, system_prompt=system_prompt, temperature=0.2)

