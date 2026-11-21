"""
Utility / helper functions used across LexAI.
Formatting, date grouping, text sanitization, parsing.
"""

import inspect
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional


# ── Streamlit compatibility helper ───────────────────────────────────────

def get_width_kw(stretch: bool = True, fn: Optional[Callable] = None) -> dict[str, Any]:
    """
    Return kwargs for full-width layout compatible across Streamlit versions.
    Uses 'width=\"stretch\"' on newer Streamlit (1.67+), falling back to 'use_container_width=True'.
    """
    try:
        import streamlit as st
        target_fn = fn or st.button
        sig = inspect.signature(target_fn)
        if "width" in sig.parameters:
            return {"width": "stretch" if stretch else "content"}
    except Exception:
        pass
    return {"use_container_width": stretch}


# ── Date / time helpers ──────────────────────────────────────────────────

def format_timestamp(ts: str | None) -> str:
    """Convert an ISO timestamp string to a human-readable format."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt

        if diff < timedelta(minutes=1):
            return "just now"
        if diff < timedelta(hours=1):
            mins = int(diff.total_seconds() / 60)
            return f"{mins}m ago"
        if diff < timedelta(hours=24):
            hrs = int(diff.total_seconds() / 3600)
            return f"{hrs}h ago"
        if diff < timedelta(days=7):
            return dt.strftime("%A %I:%M %p")
        return dt.strftime("%b %d, %Y")
    except Exception:
        return ts[:16] if len(ts) > 16 else ts


def group_chats_by_date(chats: list[dict]) -> dict[str, list[dict]]:
    """Group chat records into Today / Yesterday / Last 7 days / Older."""
    groups: dict[str, list[dict]] = {
        "📌 Pinned": [],
        "📅 Today": [],
        "📅 Yesterday": [],
        "📅 Last 7 days": [],
        "📅 Older": [],
    }
    now = datetime.now(timezone.utc)
    today = now.date()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    for chat in chats:
        if chat.get("pinned"):
            groups["📌 Pinned"].append(chat)
            continue
        try:
            ts = chat.get("updated_at") or chat.get("created_at", "")
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).date()
            if dt == today:
                groups["📅 Today"].append(chat)
            elif dt == yesterday:
                groups["📅 Yesterday"].append(chat)
            elif dt >= week_ago:
                groups["📅 Last 7 days"].append(chat)
            else:
                groups["📅 Older"].append(chat)
        except Exception:
            groups["📅 Older"].append(chat)

    # Remove empty groups
    return {k: v for k, v in groups.items() if v}


# ── Text helpers ─────────────────────────────────────────────────────────

def truncate_text(text: str, max_len: int = 60) -> str:
    """Truncate text with ellipsis."""
    text = text.strip().replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text


def sanitize_input(text: str) -> str:
    """Basic input sanitization: strip injection markers, excessive whitespace."""
    # Remove common prompt-injection patterns
    patterns = [
        r"ignore (all |the )?(previous|above|prior) (instructions|prompts?|rules)",
        r"you are now",
        r"new instructions:",
        r"system prompt:",
        r"<\|.*?\|>",
    ]
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    # Collapse excessive whitespace
    cleaned = re.sub(r"\s{3,}", "  ", cleaned).strip()
    return cleaned


# ── Answer parsing ───────────────────────────────────────────────────────

def parse_structured_answer(raw_text: str) -> dict[str, Any]:
    """
    Parse the LLM's structured legal answer (using ### headers) into sections.
    Returns a dict with keys: disclaimer, lawyer_needed, lawyer_reason,
    legal_summary, relevant_laws, action_roadmap, important_rights, final_note.
    """
    sections: dict[str, str] = {}
    current_key = ""
    current_lines: list[str] = []

    header_map = {
        "disclaimer": "disclaimer",
        "lawyer needed": "lawyer_needed",
        "lawyer requirement": "lawyer_needed",
        "legal summary": "legal_summary",
        "relevant laws": "relevant_laws",
        "action roadmap": "action_roadmap",
        "roadmap": "action_roadmap",
        "important rights": "important_rights",
        "rights & notes": "important_rights",
        "rights and notes": "important_rights",
        "final note": "final_note",
    }

    for line in raw_text.split("\n"):
        stripped = line.strip()
        # Check for ### header
        if stripped.startswith("###"):
            # Save previous section
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            # Determine new key
            header_text = stripped.lstrip("#").strip().lower()
            current_key = header_map.get(header_text, header_text.replace(" ", "_"))
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    # Parse lawyer_needed into level + reason
    lawyer_raw = sections.get("lawyer_needed", "")
    lawyer_level = "Optional"
    lawyer_reason = lawyer_raw
    for level in ["Required", "Not Needed", "Optional"]:
        if level.lower() in lawyer_raw.lower():
            lawyer_level = level
            # Reason is everything after the level keyword
            idx = lawyer_raw.lower().find(level.lower())
            lawyer_reason = lawyer_raw[idx + len(level):].strip().lstrip(":-–—").strip()
            break

    # Parse relevant_laws into list of dicts
    laws_raw = sections.get("relevant_laws", "")
    laws_list = []
    for m in re.finditer(r"[-•]\s*\*?\*?(.+?)(?:\*?\*?):\s*(.+)", laws_raw):
        laws_list.append({"section": m.group(1).strip().strip("*"), "description": m.group(2).strip()})
    if not laws_list and laws_raw:
        # Fallback: treat each line as a law
        for line in laws_raw.split("\n"):
            line = line.strip().lstrip("-•* ")
            if line:
                laws_list.append({"section": line, "description": ""})

    # Parse action_roadmap into list of steps
    roadmap_raw = sections.get("action_roadmap", "")
    roadmap_steps = []
    for m in re.finditer(r"\d+\.\s*(.+)", roadmap_raw):
        roadmap_steps.append(m.group(1).strip())
    if not roadmap_steps and roadmap_raw:
        for line in roadmap_raw.split("\n"):
            line = line.strip().lstrip("-•* ")
            if line:
                roadmap_steps.append(line)

    # Parse important_rights into list
    rights_raw = sections.get("important_rights", "")
    rights_list = []
    for line in rights_raw.split("\n"):
        line = line.strip().lstrip("-•* ")
        if line:
            rights_list.append(line)

    return {
        "disclaimer": sections.get("disclaimer", "This is general legal information, not legal advice. Please consult a qualified lawyer."),
        "lawyer_needed": lawyer_level,
        "lawyer_reason": lawyer_reason,
        "legal_summary": sections.get("legal_summary", raw_text),
        "relevant_laws": laws_list,
        "action_roadmap": roadmap_steps,
        "important_rights": rights_list,
        "final_note": sections.get("final_note", ""),
        "_raw": raw_text,
    }


def parse_risk_response(raw_text: str) -> tuple[str, str]:
    """Parse risk classifier output into (level, reason)."""
    level = "low"
    reason = raw_text.strip()
    upper = raw_text.upper()
    if "HIGH" in upper:
        level = "high"
    elif "MEDIUM" in upper:
        level = "medium"
    # Extract reason after "REASON:" if present
    for marker in ["REASON:", "REASONING:", "BECAUSE:"]:
        idx = upper.find(marker)
        if idx >= 0:
            reason = raw_text[idx + len(marker):].strip()
            break
    return level, reason


def extract_law_references(text: str) -> list[str]:
    """Extract law section references like 'Section 420 of IPC', 'CrPC 154', etc."""
    patterns = [
        r"Section\s+\d+[A-Za-z]*(?:\s+of\s+[\w\s]+(?:Act|Code|Sanhita))?",
        r"(?:IPC|BNS|CrPC|BNSS|BSA)\s+(?:Section\s+)?\d+[A-Za-z]*",
        r"Article\s+\d+[A-Za-z]*",
    ]
    refs = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            refs.add(m.group(0).strip())
    return sorted(refs)


def format_confidence(score: float) -> tuple[str, str]:
    """Return (label, color) for a confidence score 0-1."""
    if score >= 0.7:
        return "High Confidence", "green"
    elif score >= 0.5:
        return "Moderate Confidence", "orange"
    else:
        return "Low Confidence", "red"


def risk_color(risk: str) -> str:
    """Map risk level to CSS color."""
    mapping = {"high": "#E53E3E", "medium": "#D69E2E", "low": "#38A169"}
    return mapping.get(risk.lower(), "#718096")


def risk_emoji(risk: str) -> str:
    """Map risk level to emoji."""
    mapping = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    return mapping.get(risk.lower(), "⚪")


def risk_description(risk: str) -> str:
    """Map risk level to short description."""
    mapping = {
        "high": "Urgent — rights at risk, lawyer strongly recommended",
        "medium": "Moderate — legal consultation advisable",
        "low": "Informational — general guidance sufficient",
    }
    return mapping.get(risk.lower(), "Unknown risk level")
