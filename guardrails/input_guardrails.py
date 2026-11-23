"""
LexAI Input Guardrails
Sequential safety layers executed prior to LLM/RAG processing:
1. Validation (empty, length limits, control characters)
2. Prompt Injection & Jailbreak Defense
3. Safety-Critical Emergency Routing (Crisis detection & supportive helpline routing)
4. Scope & Domain Checking (Heuristics + Fast LLM Fallback)
5. Harmful Intent & Crime Facilitation Interceptor
6. PII Masking (Aadhaar, PAN, Phone, Email, Bank Accounts)
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from core.llm import call_llm
from guardrails.config import (
    AADHAAR_REGEX,
    BANK_ACC_REGEX,
    EMAIL_REGEX,
    GUARDRAILS_CONFIG,
    LEGAL_KEYWORDS,
    PAN_REGEX,
    PHONE_REGEX,
)
from guardrails.emergency_router import emergency_routing_guardrail
from guardrails.pipeline import register_input_guardrail
from guardrails.rate_limiter import check_rate_limits
from guardrails.types import GuardrailResult

logger = logging.getLogger("lexai.guardrails.input")


# ── 1. Validation Guardrail ──────────────────────────────────────────────────
@register_input_guardrail
def validate_input(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 1: Reject empty input, enforce max length, and strip non-printable control characters.
    """
    if not text or not text.strip():
        return GuardrailResult.block(
            reason_code="empty_input",
            user_message=GUARDRAILS_CONFIG.get_refusal("empty_input"),
        )

    # Remove non-printable control characters (keep standard \n, \r, \t)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    cleaned = cleaned.strip()

    if not cleaned:
        return GuardrailResult.block(
            reason_code="empty_input",
            user_message=GUARDRAILS_CONFIG.get_refusal("empty_input"),
        )

    if len(cleaned) > GUARDRAILS_CONFIG.max_message_length:
        return GuardrailResult.block(
            reason_code="message_too_long",
            user_message=GUARDRAILS_CONFIG.get_refusal("message_too_long"),
        )

    if cleaned != text:
        return GuardrailResult.modify(
            sanitized_text=cleaned,
            reason_code="control_chars_removed",
        )

    return GuardrailResult.allow(cleaned)


# ── 2. Rate Limiting & Abuse Protection Guardrail ─────────────────────────────
# Enforces per-minute, per-day, per-conversation, and repeated block cooldowns
register_input_guardrail(check_rate_limits)


# ── 3. Prompt Injection & Jailbreak Guardrail ────────────────────────────────
@register_input_guardrail
def detect_prompt_injection(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 2: Detect system prompt leaks, instruction overrides, and jailbreak personas.
    """
    if not GUARDRAILS_CONFIG.enable_injection_check:
        return GuardrailResult.allow(text)

    for pattern in GUARDRAILS_CONFIG.injection_patterns:
        if re.search(pattern, text):
            logger.warning(f"Prompt injection detected matching pattern: {pattern}")
            return GuardrailResult.block(
                reason_code="prompt_injection",
                user_message=GUARDRAILS_CONFIG.get_refusal("prompt_injection"),
            )

    return GuardrailResult.allow(text)


# ── 3. Safety-Critical Emergency Routing ─────────────────────────────────────
# Register emergency_routing_guardrail as Check 3
register_input_guardrail(emergency_routing_guardrail)


# ── 4. Scope & Domain Guardrail ──────────────────────────────────────────────
@register_input_guardrail
def check_scope(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 4: Enforce Indian legal assistant scope.
    Uses fast heuristics first; falls back to an LLM classifier for ambiguous inputs.
    Fails open (allows message) if classifier fails.
    """
    if not GUARDRAILS_CONFIG.enable_scope_check:
        return GuardrailResult.allow(text)

    # If marked as an active crisis, it must be handled immediately
    if context and context.get("emergency_type"):
        return GuardrailResult.allow(text)

    lower = text.lower()

    # Fast heuristic: If legal keywords are present as whole words, it is in scope
    has_legal_keyword = any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in LEGAL_KEYWORDS)

    # Check for clearly off-topic topics (recipes, coding, sports/gossip, math)
    is_clearly_offtopic = any(re.search(pat, text) for pat in GUARDRAILS_CONFIG.off_topic_patterns)

    if is_clearly_offtopic and not has_legal_keyword:
        return GuardrailResult.block(
            reason_code="out_of_scope",
            user_message=GUARDRAILS_CONFIG.get_refusal("out_of_scope"),
        )

    # If legal keywords are explicitly present, allow immediately
    if has_legal_keyword:
        return GuardrailResult.allow(text)

    # For ambiguous short queries, run fast LLM classifier
    if len(text.split()) > 3:
        try:
            prompt = f"""You are a scope classifier for LexAI, an Indian legal AI assistant.
Determine whether the user query is relevant to legal matters, rights, police, courts, contracts, tenant issues, employment disputes, statutory rules, civic procedures, government regulations, or offences under Indian law.
Educational questions about laws and crimes ARE in scope.
Respond ONLY with a JSON object: {{"in_scope": true}} or {{"in_scope": false}}

USER QUERY:
{text}
"""
            raw_eval = call_llm(
                prompt=prompt,
                system_prompt="You are a strict JSON classifier. Output only valid JSON.",
                temperature=0.0,
            )
            if raw_eval and raw_eval.strip():
                clean_json = raw_eval.strip()
                match = re.search(r"\{.*?\}", clean_json, re.DOTALL)
                if match:
                    clean_json = match.group(0)
                data = json.loads(clean_json)
                if data.get("in_scope") is False:
                    return GuardrailResult.block(
                        reason_code="out_of_scope",
                        user_message=GUARDRAILS_CONFIG.get_refusal("out_of_scope"),
                    )
        except Exception as e:
            # Rule: If the classifier fails, ALLOW the message
            logger.info(f"Scope classifier error (failing open): {e}")

    return GuardrailResult.allow(text)


# ── 5. Harmful Intent & Crime Facilitation Guardrail ──────────────────────────
@register_input_guardrail
def check_harmful_intent(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 5: Block requests seeking assistance to commit crimes, evade police, destroy evidence,
    or forge documents. Neutral legal explanations (e.g. 'Explain BNS Section 103') MUST pass.
    """
    if not GUARDRAILS_CONFIG.enable_harmful_intent_check:
        return GuardrailResult.allow(text)

    # If marked as an active crisis (e.g. self-harm or victim of violence), do not block as crime
    if context and context.get("emergency_type"):
        return GuardrailResult.allow(text)

    # 1. Neutral / educational questions explicitly pass
    for benign_pat in GUARDRAILS_CONFIG.benign_legal_patterns:
        if re.search(benign_pat, text):
            return GuardrailResult.allow(text)

    # 2. Block actionable criminal assistance / evasion / evidence tampering
    for harmful_pat in GUARDRAILS_CONFIG.harmful_intent_patterns:
        if re.search(harmful_pat, text):
            logger.warning(f"Harmful intent detected matching pattern: {harmful_pat}")
            return GuardrailResult.block(
                reason_code="crime_intent",
                user_message=GUARDRAILS_CONFIG.get_refusal("crime_intent"),
            )

    return GuardrailResult.allow(text)


# ── 6. PII Masking Guardrail ─────────────────────────────────────────────────
@register_input_guardrail
def mask_pii(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 6: Detect and mask Aadhaar, PAN, phone numbers, emails, and bank accounts.
    Masked BEFORE passing text to LLM or saving to guardrail logs.
    """
    if not GUARDRAILS_CONFIG.enable_pii_masking:
        return GuardrailResult.allow(text)

    masked = text
    pii_found = False

    # 1. Mask Email addresses
    if re.search(EMAIL_REGEX, masked):
        masked = re.sub(EMAIL_REGEX, "[REDACTED EMAIL]", masked)
        pii_found = True

    # 2. Mask Aadhaar numbers (12 digits, first digit 2-9)
    if re.search(AADHAAR_REGEX, masked):
        masked = re.sub(AADHAAR_REGEX, "[REDACTED AADHAAR]", masked)
        pii_found = True

    # 3. Mask PAN cards (5 letters, 4 digits, 1 letter)
    if re.search(PAN_REGEX, masked):
        masked = re.sub(PAN_REGEX, "[REDACTED PAN]", masked)
        pii_found = True

    # 4. Mask Phone numbers
    if re.search(PHONE_REGEX, masked):
        masked = re.sub(PHONE_REGEX, "[REDACTED PHONE]", masked)
        pii_found = True

    # 5. Mask Bank Account numbers
    if re.search(BANK_ACC_REGEX, masked):
        masked = re.sub(BANK_ACC_REGEX, "Account [REDACTED BANK ACCOUNT]", masked)
        pii_found = True

    if pii_found:
        if context is not None:
            context["pii_masked"] = True
        return GuardrailResult.modify(
            sanitized_text=masked,
            reason_code="pii_masked",
            user_message=GUARDRAILS_CONFIG.pii_notice,
        )

    return GuardrailResult.allow(text)
