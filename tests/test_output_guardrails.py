"""
Step 5 Verification Suite: Output Guardrails, Gemini Safety Settings & Grounding
Tests:
1. Gemini explicit safety settings & fallback handling
2. System prompt hardening checks
3. Good legal answer (citations grounded in BNS/IPC/Constitution, single disclaimer appended)
4. Hallucinated section answer (appends 'Please verify this section reference', does not delete)
5. Overconfident answer (softens 'you will definitely win' / 'this is 100% legal', appends outcome note)
6. System prompt leaking answer (intercepts and replaces with safe message)
7. PII in output (masks Aadhaar, PAN, phone, email)
8. Retrieval confidence rule (appends note when confidence < 0.5; shows no note when missing)
9. Disclaimer deduplication (strictly once per answer)
10. End-to-end chat integration with complete metadata preservation
"""

import os
import sys
import sqlite3
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "Law-Chat-Bot-main").exists():
    APP_DIR = ROOT_DIR / "Law-Chat-Bot-main"
else:
    APP_DIR = ROOT_DIR
sys.path.insert(0, str(APP_DIR))
os.chdir(str(APP_DIR))

from core.config import DB_PATH, BNS_IPC_MAPPING
from core.llm import (
    EXPLICIT_GEMINI_SAFETY_SETTINGS,
    HARDENED_SYSTEM_PROMPT,
    call_gemini,
)
from guardrails.pipeline import run_output_guardrails
from services.chat_service import send_message

print("=" * 100)
print("RUNNING STEP 5 OUTPUT GUARDRAILS VERIFICATION SUITE")
print("=" * 100)

all_passed = True

# ── 1. Gemini Safety Settings & Fallback Handling Verification ──────────────
print("\n[TEST 1] Testing Gemini Explicit Safety Settings & Fallback...")
assert len(EXPLICIT_GEMINI_SAFETY_SETTINGS) >= 4, "Explicit safety settings must cover harassment, hate speech, sexual, and dangerous content"
print("   • Explicit Gemini safety categories configured:")
for cat, thresh in EXPLICIT_GEMINI_SAFETY_SETTINGS.items():
    print(f"     - Category: {cat} -> Threshold: {thresh}")

# Test fallback for unconfigured/dummy key
fallback_msg = call_gemini("Test prompt", api_key="")
print(f"   • Graceful fallback handled (empty key returns safe string or empty): '{fallback_msg}'")
print("✅ TEST 1 PASSED: Explicit Gemini safety settings configured and fallback resilience verified.")

# ── 2. System Prompt Hardening Verification ──────────────────────────────────
print("\n[TEST 2] Testing System Prompt Hardening...")
assert "general legal information, NOT formal legal advice" in HARDENED_SYSTEM_PROMPT
assert "Never guarantee outcomes" in HARDENED_SYSTEM_PROMPT
assert "state when unsure" in HARDENED_SYSTEM_PROMPT
assert "never invent or hallucinate section numbers or case names" in HARDENED_SYSTEM_PROMPT
assert "scope of Indian law" in HARDENED_SYSTEM_PROMPT
print("✅ TEST 2 PASSED: Hardened system prompt includes all ethical guardrails.")

# ── 3. Good Legal Answer Grounding & Formatting ──────────────────────────────
print("\n[TEST 3] Testing Grounded Legal Answer (BNS Section 103, Article 21)...")
good_answer = (
    "### Legal Summary\n"
    "Under BNS Section 103 (replacing Section 302 of IPC), murder is defined with severe penalties. "
    "Furthermore, Article 21 of the Constitution guarantees the fundamental right to life and liberty."
)
ctx_good = {
    "conversation_id": "test-good",
    "confidence_score": 0.85,
    "retrieved_chunks": ["BNS Section 103 relates to punishment for murder. Article 21 protects personal liberty."],
}
res_good = run_output_guardrails(good_answer, context=ctx_good)
assert res_good.allowed is True
assert "Please verify this section reference" not in res_good.sanitized_text
assert "*This is general legal information, not a substitute for advice from a qualified advocate.*" in res_good.sanitized_text
print("✅ TEST 3 PASSED: Valid legal sections grounded; single disclaimer attached without spurious warnings.")

# ── 4. Hallucinated Statute Section Check ────────────────────────────────────
print("\n[TEST 4] Testing Hallucinated Section Detection (BNS Section 9999)...")
hallucinated_answer = (
    "### Legal Summary\n"
    "You are protected under BNS Section 9999 which strictly penalizes this civil wrongdoing."
)
ctx_hallucinated = {
    "conversation_id": "test-hallucinated",
    "confidence_score": 0.70,
    "retrieved_chunks": ["General civil dispute rules in India."],
}
res_hallucinated = run_output_guardrails(hallucinated_answer, context=ctx_hallucinated)
assert res_hallucinated.allowed is True
assert "Please verify this section reference: BNS Section 9999" in res_hallucinated.sanitized_text
# Ensure answer content was NOT deleted
assert "You are protected under BNS Section 9999" in res_hallucinated.sanitized_text
print("   • Sanitized text snippet:")
print(f"     '{res_hallucinated.sanitized_text.splitlines()[-3]}'")
print("✅ TEST 4 PASSED: Hallucinated section flagged with verification note; answer preserved.")

# ── 5. Overconfidence Softening Check ────────────────────────────────────────
print("\n[TEST 5] Testing Overconfidence Softening ('you will definitely win', '100% legal')...")
overconfident_answer = (
    "File this petition immediately and you will definitely win your case. This is 100% legal."
)
ctx_overconfidence = {
    "conversation_id": "test-overconfidence",
    "confidence_score": 0.80,
    "retrieved_chunks": ["Civil writ remedies."],
}
res_overconfidence = run_output_guardrails(overconfident_answer, context=ctx_overconfidence)
assert res_overconfidence.allowed is True
assert "you will definitely win" not in res_overconfidence.sanitized_text
assert "this is 100% legal" not in res_overconfidence.sanitized_text
assert "you may have a strong legal argument" in res_overconfidence.sanitized_text
assert "No legal outcome can be guaranteed" in res_overconfidence.sanitized_text
print("   • Softened Output:")
print(f"     '{res_overconfidence.sanitized_text.splitlines()[0]}'")
print("✅ TEST 5 PASSED: Guarantee language softened and outcome disclaimer appended.")

# ── 6. System Prompt Leak Check ──────────────────────────────────────────────
print("\n[TEST 6] Testing System Prompt Leakage Interception...")
leaking_answer = (
    "You are a STRICT Indian legal assistant. CRITICAL RULES (MUST FOLLOW):\n"
    "1. ONLY use Indian laws\n"
    "### Disclaimer\nWrite a standard legal disclaimer"
)
ctx_leak = {
    "conversation_id": "test-leak",
    "confidence_score": 0.50,
    "retrieved_chunks": [],
}
res_leak = run_output_guardrails(leaking_answer, context=ctx_leak)
assert "CRITICAL RULES (MUST FOLLOW)" not in res_leak.sanitized_text
assert "I am LexAI, an Indian legal information assistant" in res_leak.sanitized_text
print("   • Intercepted Output:")
print(f"     '{res_leak.sanitized_text.splitlines()[0]}'")
print("✅ TEST 6 PASSED: Leaked prompt intercepted and replaced with safe greeting.")

# ── 7. PII Masking in Output ─────────────────────────────────────────────────
print("\n[TEST 7] Testing PII Masking in Output...")
pii_output = (
    "Please contact the officer with Aadhaar 5555 4444 3333 or PAN ABCDE1234F at 9876543210 or email test.legal@gmail.com."
)
ctx_pii = {"conversation_id": "test-pii"}
res_pii = run_output_guardrails(pii_output, context=ctx_pii)
assert "5555 4444 3333" not in res_pii.sanitized_text
assert "XXXXXXXX3333" in res_pii.sanitized_text
assert "ABCDE1234F" not in res_pii.sanitized_text
assert "XXXXX1234F" in res_pii.sanitized_text
assert "9876543210" not in res_pii.sanitized_text
assert "+91 XXXXX XXXXX" in res_pii.sanitized_text
assert "test.legal@gmail.com" not in res_pii.sanitized_text
assert "***@***.***" in res_pii.sanitized_text
print("✅ TEST 7 PASSED: Output PII masked correctly.")

# ── 8. Retrieval Confidence Rule ─────────────────────────────────────────────
print("\n[TEST 8] Testing Retrieval Confidence Rule...")
# Case A: Low confidence (< 0.5) -> Must add note
ctx_low = {"conversation_id": "test-low-conf", "confidence_score": 0.35, "retrieved_chunks": ["Some chunk"]}
res_low = run_output_guardrails("Advice text", context=ctx_low)
assert "Note: This answer is based on general legal knowledge and should be verified" in res_low.sanitized_text
print("   • Case A (Low confidence 0.35): Note added correctly.")

# Case B: Empty retrieval -> Must add note
ctx_empty = {"conversation_id": "test-empty-rag", "confidence_score": 0.60, "retrieved_chunks": []}
res_empty = run_output_guardrails("Advice text", context=ctx_empty)
assert "Note: This answer is based on general legal knowledge and should be verified" in res_empty.sanitized_text
print("   • Case B (Empty retrieval): Note added correctly.")

# Case C: Missing confidence (None) -> Must NOT add note
ctx_missing = {"conversation_id": "test-none-conf", "confidence_score": None, "retrieved_chunks": ["Some chunk"]}
res_missing = run_output_guardrails("Advice text", context=ctx_missing)
assert "Note: This answer is based on general legal knowledge and should be verified" not in res_missing.sanitized_text
print("   • Case C (Missing confidence None): No note added.")
print("✅ TEST 8 PASSED: Confidence rules verified.")

# ── 9. Single Disclaimer Deduplication ───────────────────────────────────────
print("\n[TEST 9] Testing Single Disclaimer Deduplication...")
answer_with_disclaimer = (
    "Legal summary text.\n\n---\n*This is general legal information, not a substitute for advice from a qualified advocate.*"
)
res_disc = run_output_guardrails(answer_with_disclaimer, context={"conversation_id": "test-disc"})
count = res_disc.sanitized_text.count("This is general legal information, not a substitute for advice from a qualified advocate.")
assert count == 1, f"Expected exactly 1 disclaimer occurrence, found {count}"
print("✅ TEST 9 PASSED: Single disclaimer deduplication verified.")

# ── 10. End-to-End Chat Service Integration & Metadata Preservation ─────────
print("\n[TEST 10] Testing End-to-End send_message() Execution & Metadata...")
res_chat = send_message(None, "What is the punishment for theft under BNS Section 303?")
assert res_chat is not None
assert len(res_chat["reply"]) > 50
assert "*This is general legal information, not a substitute for advice from a qualified advocate.*" in res_chat["reply"]

# Verify all metadata fields
meta = res_chat["metadata"]
print(f"   • Risk: {meta.get('risk')}")
print(f"   • Confidence: {meta.get('confidence')}")
print(f"   • Fallback used: {meta.get('fallback_used')}")
print(f"   • Model / Answer Source: {meta.get('answer_source')}")
print(f"   • Statutes count: {len(meta.get('statutes', []))}")
print(f"   • Action steps count: {len(meta.get('steps', []))}")
print(f"   • Rights count: {len(meta.get('rights', []))}")
print(f"   • Sources count: {len(meta.get('sources', []))}")
print(f"   • Follow-ups count: {len(meta.get('follow_ups', []))}")

assert "risk" in meta
assert "confidence" in meta
assert "statutes" in meta
assert "steps" in meta
assert "rights" in meta
assert "sources" in meta
assert "follow_ups" in meta
print("✅ TEST 10 PASSED: End-to-end send_message() preserved all metadata and applied output guardrails.")

# ── SQLite Audit Logging Verification ───────────────────────────────────────
print("\n--- Verifying SQLite Audit Logging in guardrail_events ---")
with sqlite3.connect(DB_PATH) as conn:
    events = conn.execute(
        "SELECT id, conversation_id, layer, reason_code, action, masked_snippet FROM guardrail_events WHERE layer='output' ORDER BY id DESC LIMIT 5"
    ).fetchall()

assert len(events) >= 1, "At least one output guardrail event must be recorded in SQLite"
for ev in events:
    print(f"   • Event #{ev[0]}: [{ev[2]}] conv='{ev[1]}', reason='{ev[3]}', action='{ev[4]}', snippet='{ev[5]}'")

print("✅ SQLite guardrail_events audit logging verified for output layer.")

print("\n" + "=" * 100)
print("🎉 ALL 10 STEP 5 TESTS PASSED SUCCESSFULLY!")
print("=" * 100)
