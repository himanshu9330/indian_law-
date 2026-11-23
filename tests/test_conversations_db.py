"""
Test suite for Step 3: Database & Conversation Persistence.
Tests all 7 core functions, UUID generation, message isolation, metadata serialization, and cascade deletion.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from services.conversations import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_messages,
    init_db,
    list_conversations,
    rename_conversation,
    update_message_metadata,
)


def run_tests():
    # Use a clean temporary SQLite database for test isolation
    temp_dir = tempfile.mkdtemp()
    test_db = os.path.join(temp_dir, "test_conversations.db")
    print(f"--- Running Conversation Persistence Tests on: {test_db} ---")

    init_db(test_db)

    # 1. create_conversation
    conv_id_1 = create_conversation(title="Tenant Eviction Inquiry", db_path=test_db)
    assert isinstance(conv_id_1, str) and len(conv_id_1) == 36, f"Invalid UUID: {conv_id_1}"
    print(f"✓ create_conversation passed. Created ID (UUID): {conv_id_1}")

    # 2. list_conversations
    convs = list_conversations(db_path=test_db)
    assert len(convs) == 1, f"Expected 1 conversation, got {len(convs)}"
    assert convs[0]["id"] == conv_id_1
    assert convs[0]["title"] == "Tenant Eviction Inquiry"
    print(f"✓ list_conversations passed. Found {len(convs)} conversation(s).")

    # 3. get_conversation
    meta = get_conversation(conv_id_1, db_path=test_db)
    assert meta is not None and meta["title"] == "Tenant Eviction Inquiry"
    print(f"✓ get_conversation passed. Title: '{meta['title']}'")

    # 4. add_message (User)
    user_msg_id = add_message(
        conversation_id=conv_id_1,
        role="user",
        content="My landlord wants to evict me without notice. What can I do?",
        db_path=test_db,
    )
    assert user_msg_id > 0
    print(f"✓ add_message (user) passed. Message ID: {user_msg_id}")

    # 5. add_message (Assistant with full legal metadata)
    metadata_payload = {
        "risk": "medium",
        "confidence": 0.88,
        "fallback_used": False,
        "statutes": ["Rent Control Act Section 14", "Article 21 Constitution of India"],
        "steps": ["Review rent agreement", "Send written reply to landlord", "File petition in Rent Tribunal"],
        "rights": ["Right to statutory notice", "Right against arbitrary cutoff of electricity/water"],
        "sources": ["Pinecone cluster #4"],
        "follow_ups": [
            "What if the landlord disconnects water or power?",
            "How much notice is mandatory before eviction?",
            "Can the landlord keep the security deposit?",
        ],
    }
    asst_msg_id = add_message(
        conversation_id=conv_id_1,
        role="assistant",
        content="Under Indian rent control statutes, a landlord cannot evict a tenant without statutory notice and court decree.",
        metadata=metadata_payload,
        db_path=test_db,
    )
    assert asst_msg_id > user_msg_id
    print(f"✓ add_message (assistant with metadata) passed. Message ID: {asst_msg_id}")

    # 6. get_messages & metadata unpack verification
    messages = get_messages(conv_id_1, db_path=test_db)
    assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["metadata"]["risk"] == "medium"
    assert messages[1]["metadata"]["confidence"] == 0.88
    assert len(messages[1]["metadata"]["follow_ups"]) == 3
    print(f"✓ get_messages passed. Retrieved {len(messages)} messages with verified unpacked metadata.")

    # 7. update_message_metadata
    updated_meta = dict(metadata_payload)
    updated_meta["risk"] = "high"
    updated_meta["notes"] = "Urgent consultation requested"
    ok = update_message_metadata(asst_msg_id, updated_meta, db_path=test_db)
    assert ok is True
    re_msgs = get_messages(conv_id_1, db_path=test_db)
    assert re_msgs[1]["metadata"]["risk"] == "high"
    assert re_msgs[1]["metadata"]["notes"] == "Urgent consultation requested"
    print("✓ update_message_metadata passed. Verified metadata update in database.")

    # 8. rename_conversation
    renamed = rename_conversation(conv_id_1, "Urgent Eviction & Security Deposit Dispute", db_path=test_db)
    assert renamed is True
    meta_renamed = get_conversation(conv_id_1, db_path=test_db)
    assert meta_renamed["title"] == "Urgent Eviction & Security Deposit Dispute"
    print(f"✓ rename_conversation passed. New title: '{meta_renamed['title']}'")

    # 9. Isolation test: Create 2nd conversation and verify message isolation
    conv_id_2 = create_conversation(title="Cyber Fraud Consultation", db_path=test_db)
    add_message(conv_id_2, "user", "I was scammed online for 50,000 INR.", db_path=test_db)

    msgs_1 = get_messages(conv_id_1, db_path=test_db)
    msgs_2 = get_messages(conv_id_2, db_path=test_db)
    assert len(msgs_1) == 2, f"Expected 2 messages in conv 1, got {len(msgs_1)}"
    assert len(msgs_2) == 1, f"Expected 1 message in conv 2, got {len(msgs_2)}"
    assert msgs_2[0]["content"] == "I was scammed online for 50,000 INR."
    print("✓ Conversation isolation test passed. Conversations 1 & 2 maintain strict message isolation.")

    # 10. delete_conversation & cascade verification
    del_ok = delete_conversation(conv_id_1, db_path=test_db)
    assert del_ok is True
    assert get_conversation(conv_id_1, db_path=test_db) is None
    # Verify cascade deletion of messages
    orphan_msgs = get_messages(conv_id_1, db_path=test_db)
    assert len(orphan_msgs) == 0, f"Expected 0 messages after cascade delete, got {len(orphan_msgs)}"

    # Conv 2 must still remain intact
    convs_remaining = list_conversations(db_path=test_db)
    assert len(convs_remaining) == 1 and convs_remaining[0]["id"] == conv_id_2
    print("✓ delete_conversation with CASCADE delete passed. Conv 1 and its messages deleted; Conv 2 remains intact.")

    print("\n🎉 ALL 10 DATABASE AND PERSISTENCE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
