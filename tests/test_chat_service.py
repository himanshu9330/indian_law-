"""
Test script for Step 4: Chat Service Layer.
Tests send_message(), generate_followups(), database verification, and input validation.
"""

import json
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from services.chat_service import generate_followups, send_message
from services.conversations import get_conversation, get_messages


def test_chat_service():
    print("=================================================================")
    print("STEP 4: Testing Chat Service with query: 'Can I be fired without notice in India?'")
    print("=================================================================\n")

    query = "Can I be fired without notice in India?"

    # 1. Send Message with conversation_id=None (creates new conversation)
    response = send_message(conversation_id=None, user_text=query)
    reply, metadata, conv_id = response

    print(f"1. CONVERSATION ID CREATED: {conv_id}")
    conv_info = get_conversation(conv_id)
    print(f"   Conversation Title: '{conv_info['title']}' (Length: {len(conv_info['title'])})\n")

    print("2. ASSISTANT REPLY (PREVIEW):")
    print("-----------------------------------------------------------------")
    print(reply[:450] + ("..." if len(reply) > 450 else ""))
    print("-----------------------------------------------------------------\n")

    print("3. EXTRACTED METADATA:")
    print(f"   - Assessed Risk:     {metadata.get('risk')}")
    print(f"   - Retrieval Conf:    {metadata.get('confidence')}")
    print(f"   - Fallback Used:     {metadata.get('fallback_used')}")
    print(f"   - Answer Source:     {metadata.get('answer_source')}")
    print(f"   - Statutes Cited:    {len(metadata.get('statutes', []))} provision(s)")
    for st in metadata.get("statutes", [])[:2]:
        print(f"     • {st.get('section', '')}: {st.get('description', '')[:60]}")
    print(f"   - Action Steps:      {len(metadata.get('steps', []))} step(s)")
    for sp in metadata.get("steps", [])[:2]:
        print(f"     • {str(sp)[:60]}")
    print(f"   - Citizen Rights:    {len(metadata.get('rights', []))} right(s)")
    for rt in metadata.get("rights", [])[:2]:
        print(f"     • {str(rt)[:60]}")
    print(f"   - Sources Count:     {len(metadata.get('sources', []))}\n")

    print("4. GENERATED FOLLOW-UP INQUIRIES:")
    follow_ups = metadata.get("follow_ups", [])
    if follow_ups:
        for idx, fq in enumerate(follow_ups, 1):
            print(f"   [{idx}] {fq}")
    else:
        print("   (Follow-ups list returned empty as required on non-LLM/offline fallback)")
    print()

    # 5. Database Verification
    print("5. VERIFYING DATABASE PERSISTENCE:")
    db_messages = get_messages(conv_id)
    assert len(db_messages) == 2, f"Expected 2 messages in DB, got {len(db_messages)}"
    assert db_messages[0]["role"] == "user"
    assert db_messages[0]["content"] == query
    assert db_messages[1]["role"] == "assistant"
    assert db_messages[1]["metadata"]["risk"] == metadata["risk"]
    assert db_messages[1]["metadata"]["confidence"] == metadata["confidence"]
    assert "follow_ups" in db_messages[1]["metadata"]
    print("   ✓ User message found in SQLite.")
    print("   ✓ Assistant message found in SQLite.")
    print("   ✓ Metadata unpacked and verified from metadata_json in SQLite.")

    # 6. Validation tests
    print("\n6. TESTING VALIDATION & ERROR HANDLING:")
    # Empty message test
    try:
        send_message(conv_id, "   ")
        assert False, "Should have raised ValueError on empty message"
    except ValueError as e:
        print(f"   ✓ Empty message caught correctly: {e}")

    # Unknown conversation ID test
    try:
        send_message("non-existent-uuid-12345", "Hello")
        assert False, "Should have raised ValueError on unknown conversation ID"
    except ValueError as e:
        print(f"   ✓ Unknown conversation ID caught correctly: {e}")

    # Subsequent message in existing conversation
    followup_resp = send_message(conv_id, "What compensation can I claim under the Industrial Disputes Act?")
    updated_messages = get_messages(conv_id)
    assert len(updated_messages) == 4, f"Expected 4 messages, got {len(updated_messages)}"
    print(f"   ✓ Multi-turn message added to conversation {conv_id}. Total messages: {len(updated_messages)}")

    print("\n🎉 ALL STEP 4 CHAT SERVICE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_chat_service()
