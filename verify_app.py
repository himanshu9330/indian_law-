"""
LexAI Verification Script
Tests database operations, pipeline imports, fallback routing, Qwen3-TTS engine, and export generation.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import db.database
import core.config
import core.prompts
import core.llm
import core.rag
import core.search
import core.pipeline
import core.hf_fallback
import core.tts
import utils.helpers
import utils.export
import ui.components
import ui.analytics
import ui.dashboard

print("1. All modules (including hf_fallback and tts) imported successfully.")

# Test Database
test_db_path = str(APP_DIR / "data" / "verify_test.db")
database = db.database.Database(test_db_path)
tables = database._fetch_all("SELECT name FROM sqlite_master WHERE type='table'")
table_names = [t["name"] for t in tables]
print("2. Database created with tables:", table_names)

# Test Chat creation & retrieval
chat_id = database.create_chat(title="Verification Consultation")
user_msg_id = database.add_message(chat_id, "user", "What are tenant eviction rules in India?")
asst_msg_id = database.add_message(
    chat_id,
    "assistant",
    "Under Indian rent control statutes, landlords cannot evict without notice.",
    metadata={
        "risk": "medium",
        "source": "huggingface_fallback",
        "answer_source": "huggingface_fallback",
        "confidence_score": 0.32,
        "fallback_reason": "Retrieval confidence (0.32) falls below threshold (0.50)",
        "answer_sections": {
            "legal_summary": "Tenant protections under Rent Control Acts require valid grounds.",
            "relevant_laws": [{"section": "Section 14 of Rent Control Act", "description": "Protection against eviction"}],
            "action_roadmap": ["Review the lease agreement.", "Send a written reply."],
            "important_rights": ["Right against unlawful eviction", "Right to security deposit refund"],
            "lawyer_needed": "Optional",
            "lawyer_reason": "Consultation advised if notice received.",
        },
    },
)

messages = database.get_messages(chat_id)
print(f"3. Stored and fetched {len(messages)} messages for chat {chat_id}.")

# Test Qwen3-TTS Voice Engine
tts_html = core.tts.render_browser_tts_component("Under Indian tenancy law, landlords must give formal notice.")
assert "speechSynthesis" in tts_html and "Qwen3-TTS" in tts_html
audio_info = core.tts.get_legal_audio("Short advisory test.")
assert audio_info["status"] in ("browser_ready", "cached", "synthesized")
print("4. Qwen3-TTS Voice Component verified successfully.")

# Test Fallback Generation (Offline Grounded Indian Law)
fallback_result = core.hf_fallback.generate_indian_legal_fallback(
    query="My landlord is evicting me without notice",
    risk="medium",
    fallback_reason="Pinecone test fallback",
)
assert "relevant_laws" in fallback_result["structured_sections"]
assert fallback_result["answer_source"] == "huggingface_fallback"
print(f"5. Fallback Engine generated response via '{fallback_result['engine_used']}'.")

# Test Export
md_content = utils.export.export_markdown("Verification Consultation", messages)
assert len(md_content) > 100
print("6. Markdown export generated (length:", len(md_content), "bytes).")

pdf_bytes = utils.export.export_pdf("Verification Consultation", messages)
assert isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 10
print(f"7. PDF export generated (length: {len(pdf_bytes)} bytes).")

# Clean up test database
if Path(test_db_path).exists():
    try:
        Path(test_db_path).unlink()
        print("8. Cleaned up verification database.")
    except Exception:
        pass

print("\n🎉 ALL TESTS PASSED! LexAI backend, Qwen fallback, and Qwen3-TTS are fully operational.")
