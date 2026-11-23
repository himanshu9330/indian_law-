"""
Unit Tests for Guardrails Foundation
Tests GuardrailResult types, SQLite audit logging, privacy masking,
and sequential fail-safe pipeline execution.
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

from guardrails.types import GuardrailResult
from guardrails.config import GUARDRAILS_CONFIG
from guardrails.logger import (
    init_guardrail_db,
    log_guardrail_event,
    mask_snippet,
    get_connection,
)
from guardrails.pipeline import (
    run_input_guardrails,
    run_output_guardrails,
    register_input_guardrail,
    register_output_guardrail,
    clear_guardrails,
)

print("=" * 60)
print("RUNNING GUARDRAILS FOUNDATION UNIT TESTS")
print("=" * 60)

# 1. Test GuardrailResult Factory Methods
print("\n[TEST 1] Testing GuardrailResult factories...")
res_allow = GuardrailResult.allow("clean legal text")
assert res_allow.allowed is True
assert res_allow.action == "allow"
assert res_allow.reason_code == "ok"
assert res_allow.sanitized_text == "clean legal text"

res_block = GuardrailResult.block("crime_intent", "Blocked intent message")
assert res_block.allowed is False
assert res_block.action == "block"
assert res_block.reason_code == "crime_intent"
assert res_block.user_message == "Blocked intent message"

res_modify = GuardrailResult.modify("sanitized version", reason_code="filtered_leak")
assert res_modify.allowed is True
assert res_modify.action == "modify"
assert res_modify.sanitized_text == "sanitized version"

res_warn = GuardrailResult.warn("high_risk_topic", "Warning note")
assert res_warn.allowed is True
assert res_warn.action == "warn"
assert res_warn.reason_code == "high_risk_topic"
print("✅ TEST 1 PASSED: GuardrailResult allow, block, modify, and warn factories verified.")

# 2. Test Privacy Masking
print("\n[TEST 2] Testing Privacy Masking...")
sensitive_text = "Call me at 9876543210 about murder conspiracy case details"
snippet = mask_snippet(sensitive_text, max_len=30)
assert "9876543210" not in snippet, "Digits must be masked"
assert "*" in snippet, "Digits should be replaced with asterisks"
assert len(snippet) <= 35, f"Snippet too long ({len(snippet)})"
print(f"✅ TEST 2 PASSED: Snippet masked to: '{snippet}'")

# 3. Test SQLite Audit Logging
print("\n[TEST 3] Testing SQLite guardrail_events logging...")
test_db = str(APP_DIR / "data" / "test_guardrails.db")
init_guardrail_db(test_db)

raw_query = "Help me forge a fake sale deed document with property id 402"
logged = log_guardrail_event(
    layer="input",
    reason_code="forgery_intent",
    action="block",
    conversation_id="conv-test-123",
    raw_text=raw_query,
    db_path=test_db,
)
assert logged is True, "log_guardrail_event must return True"

with get_connection(test_db) as conn:
    row = conn.execute(
        "SELECT conversation_id, layer, reason_code, action, masked_snippet FROM guardrail_events WHERE reason_code = ?",
        ("forgery_intent",)
    ).fetchone()

assert row is not None, "Event must be saved in database"
assert row[0] == "conv-test-123"
assert row[1] == "input"
assert row[2] == "forgery_intent"
assert row[3] == "block"
# Verify privacy: raw query is NOT stored in the database!
assert row[4] != raw_query, "Raw full text must NOT be stored in DB"
assert "402" not in row[4], "Numbers in snippet must be masked"

print(f"✅ TEST 3 PASSED: Event recorded with action='{row[3]}' and masked snippet='{row[4]}'. Raw text protected.")

# Clean up test DB
if Path(test_db).exists():
    try:
        Path(test_db).unlink()
    except Exception:
        pass

# 4. Test Pipeline: Sequential checks and stopping on first block
print("\n[TEST 4] Testing Pipeline: stopping on first block...")
clear_guardrails()

check_1_executed = False
check_2_executed = False
check_3_executed = False

@register_input_guardrail
def check_1_pass(text, context):
    global check_1_executed
    check_1_executed = True
    return GuardrailResult.allow(text)

@register_input_guardrail
def check_2_block(text, context):
    global check_2_executed
    check_2_executed = True
    return GuardrailResult.block("blocked_rule_2", "Stopped by Check 2")

@register_input_guardrail
def check_3_never_reached(text, context):
    global check_3_executed
    check_3_executed = True
    return GuardrailResult.allow(text)

result = run_input_guardrails("Test query", context={"conversation_id": "conv-pipeline-test"})

assert check_1_executed is True, "Check 1 must run"
assert check_2_executed is True, "Check 2 must run"
assert check_3_executed is False, "Check 3 must NOT run after block"
assert result.allowed is False
assert result.action == "block"
assert result.reason_code == "blocked_rule_2"
assert result.user_message == "Stopped by Check 2"

print("✅ TEST 4 PASSED: Pipeline halted on first block action; subsequent checks skipped.")

# 5. Test Pipeline: Fail-safe behavior (no crashes if a check throws)
print("\n[TEST 5] Testing Pipeline: Fail-safe exception resilience...")
clear_guardrails()

check_after_error_executed = False

@register_input_guardrail
def check_faulty(text, context):
    raise RuntimeError("Simulated unexpected guardrail exception")

@register_input_guardrail
def check_post_fault(text, context):
    global check_after_error_executed
    check_after_error_executed = True
    return GuardrailResult.allow(text)

safe_result = run_input_guardrails("Hello LexAI")
assert safe_result.allowed is True
assert check_after_error_executed is True, "Check after faulty rule must still execute"
print("✅ TEST 5 PASSED: Faulty guardrail caught safely; pipeline continued without crashing.")

# 6. Test Output Guardrails Pipeline & Modification
print("\n[TEST 6] Testing Output Guardrail modification...")
clear_guardrails()

@register_output_guardrail
def disclaimer_enforcer(answer, context):
    if "Disclaimer" not in answer:
        modified = f"{answer}\n\n### Disclaimer\nStandard Indian legal advice disclaimer."
        return GuardrailResult.modify(modified, reason_code="disclaimer_appended")
    return GuardrailResult.allow(answer)

out_res = run_output_guardrails("Section 302 of IPC covers punishment for murder.")
assert out_res.action == "modify"
assert "Standard Indian legal advice disclaimer" in out_res.sanitized_text
print("✅ TEST 6 PASSED: Output guardrail successfully modified response.")

clear_guardrails()
print("\n" + "=" * 60)
print("🎉 ALL GUARDRAILS FOUNDATION TESTS PASSED!")
print("=" * 60)
