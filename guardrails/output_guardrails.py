"""
LexAI Output Guardrails
Sequential safety and verification checks executed right after an LLM answer
is generated, before it is displayed to the user or saved to database.

Checks:
1. System Prompt Leak Check: replace regurgitated system instructions with a safe reply.
2. PII Masking: mask Aadhaar, PAN, phone, or email appearing in generated output.
3. Overconfidence Softening: soften absolute guarantees (e.g., 'definitely win', '100% legal').
4. Citation Grounding: cross-reference cited sections against RAG sources and BNS-IPC mapping.
5. Retrieval Confidence Rule: attach general knowledge disclaimer if confidence is low or retrieval empty.
6. Single Disclaimer Line: attach standardized short disclaimer once at the bottom in muted text.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from core.config import BNS_IPC_MAPPING
from guardrails.config import (
    AADHAAR_REGEX,
    BANK_ACC_REGEX,
    EMAIL_REGEX,
    GUARDRAILS_CONFIG,
    KNOWN_VALID_STATUTES,
    PAN_REGEX,
    PHONE_REGEX,
)
from guardrails.pipeline import register_output_guardrail
from guardrails.types import GuardrailResult

logger = logging.getLogger("lexai.guardrails.output")


# ── 1. System Prompt Leak Guardrail ───────────────────────────────────────────
@register_output_guardrail
def check_prompt_leak(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 1: If the answer contains fragments of the internal system prompt,
    replace it with a safe, polite legal assistant greeting.
    """
    if not GUARDRAILS_CONFIG.enable_prompt_leak_check:
        return GuardrailResult.allow(answer)

    for pat in GUARDRAILS_CONFIG.prompt_leak_patterns:
        if re.search(pat, answer):
            logger.warning(f"System prompt leak detected matching pattern: {pat}")
            return GuardrailResult.modify(
                sanitized_text=GUARDRAILS_CONFIG.safe_prompt_leak_message,
                reason_code="prompt_leak_blocked",
            )

    return GuardrailResult.allow(answer)


# ── 2. Output PII Masking Guardrail ──────────────────────────────────────────
@register_output_guardrail
def check_pii_output(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 2: Mask any Aadhaar, PAN, Phone, or Email patterns appearing in the generated text.
    """
    if not GUARDRAILS_CONFIG.enable_pii_output_masking:
        return GuardrailResult.allow(answer)

    text = answer
    was_modified = False

    # Mask Aadhaar
    def _mask_aadhaar(m: re.Match) -> str:
        nonlocal was_modified
        was_modified = True
        raw = re.sub(r"[\s-]", "", m.group(0))
        return f"XXXXXXXX{raw[-4:]}"

    text = re.sub(AADHAAR_REGEX, _mask_aadhaar, text)

    # Mask PAN
    def _mask_pan(m: re.Match) -> str:
        nonlocal was_modified
        was_modified = True
        raw = m.group(0)
        return f"XXXXX{raw[5:9]}{raw[-1]}"

    text = re.sub(PAN_REGEX, _mask_pan, text)

    # Mask Phone
    def _mask_phone(m: re.Match) -> str:
        nonlocal was_modified
        was_modified = True
        return "+91 XXXXX XXXXX"

    text = re.sub(PHONE_REGEX, _mask_phone, text)

    # Mask Email
    def _mask_email(m: re.Match) -> str:
        nonlocal was_modified
        was_modified = True
        return "***@***.***"

    text = re.sub(EMAIL_REGEX, _mask_email, text)

    if was_modified:
        return GuardrailResult.modify(sanitized_text=text, reason_code="pii_masked_output")

    return GuardrailResult.allow(answer)


# ── 3. Overconfidence & Guarantee Softening Guardrail ────────────────────────
@register_output_guardrail
def check_overconfidence(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 3: If the answer contains guarantee phrases such as 'you will definitely win'
    or 'this is 100% legal', soften the language and append an outcome disclaimer.
    """
    if not GUARDRAILS_CONFIG.enable_overconfidence_check:
        return GuardrailResult.allow(answer)

    has_overconfidence = any(re.search(pat, answer) for pat in GUARDRAILS_CONFIG.overconfidence_patterns)
    if not has_overconfidence:
        return GuardrailResult.allow(answer)

    text = answer
    # Soften specific absolute statements
    text = re.sub(r"(?i)\byou\s+will\s+(?:definitely|certainly|surely)\s+win\b", "you may have a strong legal argument", text)
    text = re.sub(r"(?i)\bthis\s+is\s+100%\s+legal\b", "this appears legally permissible under statutory rules", text)
    text = re.sub(r"(?i)\b(?:100%\s+guaranteed|guaranteed\s+to\s+win)\b", "strongly supported by statutory provisions", text)
    text = re.sub(r"(?i)\bcannot\s+lose\s+this\s+case\b", "have a defensible legal position", text)
    text = re.sub(r"(?i)\bthe\s+court\s+will\s+definitely\s+(?:rule\s+in\s+your\s+favor|acquit\s+you)\b", "the court will evaluate the matter on its merits", text)

    note = "> ⚠️ *Important: In legal proceedings, outcomes depend on specific facts, evidence, and judicial discretion. No legal outcome can be guaranteed.*"
    if note not in text:
        text = f"{text}\n\n{note}"

    return GuardrailResult.modify(sanitized_text=text, reason_code="overconfidence_softened")


# ── 4. Citation Grounding Guardrail ──────────────────────────────────────────
def _extract_statute_citations(text: str) -> List[Dict[str, str]]:
    """
    Extract statute and article references from text.
    Returns list of dicts with statute code (bns/ipc/crpc/bnss/constitution) and section number.
    """
    citations = []

    # 1. Matches: "BNS Section 103", "IPC Section 302", "BNSS Section 173", "Section 103 BNS", "Section 302 IPC"
    p1 = re.compile(
        r"(?i)\b(?:(BNS|IPC|CrPC|BNSS|BSA)\s+(?:Section|Sec\.?)\s*(\d+[A-Za-z]?(?:\(\d+\))*)|"
        r"(?:Section|Sec\.?)\s*(\d+[A-Za-z]?(?:\(\d+\))*)\s+(?:of\s+(?:the\s+)?(BNS|IPC|CrPC|BNSS|BSA|Indian\s+Penal\s+Code|Bharatiya\s+Nyaya\s+Sanhita|Bharatiya\s+Nagarik\s+Suraksha\s+Sanhita)))\b"
    )
    for m in p1.finditer(text):
        if m.group(1) and m.group(2):
            law = m.group(1).lower()
            sec = m.group(2).lower()
            citations.append({"law": law, "section": sec, "raw": f"{m.group(1)} Section {m.group(2)}"})
        elif m.group(3) and m.group(4):
            raw_law = m.group(4).lower()
            law = "ipc" if "penal" in raw_law or "ipc" in raw_law else ("bns" if "nyaya" in raw_law or "bns" in raw_law else ("bnss" if "suraksha" in raw_law or "bnss" in raw_law else raw_law[:4]))
            sec = m.group(3).lower()
            citations.append({"law": law, "section": sec, "raw": f"Section {m.group(3)} of {m.group(4)}"})

    # 2. Matches: "IPC 302", "BNS 103", "CrPC 154", "BNSS 173"
    p2 = re.compile(r"(?i)\b(BNS|IPC|CrPC|BNSS)\s+(\d+[A-Za-z]?)\b")
    for m in p2.finditer(text):
        citations.append({"law": m.group(1).lower(), "section": m.group(2).lower(), "raw": f"{m.group(1)} {m.group(2)}"})

    # 3. Matches: "Article 21", "Article 19", "Article 14"
    p3 = re.compile(r"(?i)\bArticle\s+(\d+[A-Za-z]?)(?:\s+of\s+(?:the\s+)?Constitution)?\b")
    for m in p3.finditer(text):
        citations.append({"law": "constitution", "section": m.group(1).lower(), "raw": f"Article {m.group(1)}"})

    return citations


@register_output_guardrail
def check_citation_grounding(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 4: Compare cited sections against retrieved Pinecone sources and BNS-IPC mapping.
    If a section is not grounded or recognized, append a small verification note.
    Does NOT delete the answer.
    """
    if not GUARDRAILS_CONFIG.enable_citation_grounding_check:
        return GuardrailResult.allow(answer)

    citations = _extract_statute_citations(answer)
    if not citations:
        return GuardrailResult.allow(answer)

    # Build corpus of known grounded sources
    retrieved_chunks = (context.get("retrieved_chunks") or []) if context else []
    sources_text = " ".join(
        (c if isinstance(c, str) else str(c.get("text", "") or c.get("content", "")))
        for c in retrieved_chunks
    ).lower()

    # Known mapped sections from BNS_IPC_MAPPING
    mapping_text = " ".join(
        f"{item.get('old_law', '')} {item.get('new_law', '')} {item.get('desc', '')}"
        for item in BNS_IPC_MAPPING
    ).lower()

    unverified = []
    for cit in citations:
        sec = cit["section"]
        law = cit["law"]
        raw = cit["raw"]

        # Check 1: In retrieved chunks?
        if sec in sources_text:
            continue

        # Check 2: In BNS-IPC mapping table?
        if sec in mapping_text:
            continue

        # Check 3: In known foundational statutes?
        valid_secs = KNOWN_VALID_STATUTES.get(law, [])
        # Check base section number without sub-clauses (e.g. '106' from '106(2)')
        base_sec = re.sub(r"\(.*?\)", "", sec).strip()
        if base_sec in valid_secs or sec in valid_secs:
            continue

        # If not verified anywhere, record for verification note
        if raw not in unverified:
            unverified.append(raw)

    if unverified:
        unverified_str = ", ".join(unverified)
        logger.warning(f"Unverified statute citations detected: {unverified_str}")
        note = f"> ⚠️ *Please verify this section reference: {unverified_str}.*"
        if note not in answer:
            updated = f"{answer}\n\n{note}"
            return GuardrailResult.modify(sanitized_text=updated, reason_code="citation_unverified")

    return GuardrailResult.allow(answer)


# ── 5. Retrieval Confidence Guardrail ────────────────────────────────────────
@register_output_guardrail
def check_retrieval_confidence(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 5: If confidence data exists and is below the configured threshold,
    or retrieval returned nothing, add a short note that the answer is based on
    general knowledge and should be verified.
    If confidence data is missing (None), do NOT show any note and never show 0%.
    """
    if not GUARDRAILS_CONFIG.enable_retrieval_confidence_check:
        return GuardrailResult.allow(answer)

    if not context or "confidence_score" not in context or context["confidence_score"] is None:
        # Confidence data is missing: do not show any note
        return GuardrailResult.allow(answer)

    confidence = context.get("confidence_score")
    retrieved_chunks = context.get("retrieved_chunks")

    is_empty_retrieval = not retrieved_chunks
    is_low_confidence = (confidence < GUARDRAILS_CONFIG.rag_confidence_threshold)

    if is_empty_retrieval or is_low_confidence:
        note = GUARDRAILS_CONFIG.low_confidence_note
        if note not in answer:
            updated = f"{answer}\n\n{note}"
            return GuardrailResult.modify(sanitized_text=updated, reason_code="low_confidence_note")

    return GuardrailResult.allow(answer)


# ── 6. Single Disclaimer Guardrail ───────────────────────────────────────────
@register_output_guardrail
def ensure_single_disclaimer(answer: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 6: Add a single short disclaimer line at the end of answers:
    'This is general legal information, not a substitute for advice from a qualified advocate.'
    Shown once per answer in small muted text, not as a big banner.
    """
    if not GUARDRAILS_CONFIG.enable_output_disclaimer:
        return GuardrailResult.allow(answer)

    disclaimer_sentence = "This is general legal information, not a substitute for advice from a qualified advocate."
    if disclaimer_sentence in answer:
        # Already present once; do not duplicate
        return GuardrailResult.allow(answer)

    disclaimer_line = f"---\n*{disclaimer_sentence}*"
    updated = f"{answer.rstrip()}\n\n{disclaimer_line}"
    return GuardrailResult.modify(sanitized_text=updated, reason_code="disclaimer_appended")
