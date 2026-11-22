"""
Database and Conversation Persistence Layer for LexAI.
Exposes the conversation management functions required for the ChatGPT-style interface.
"""

from services.conversations import (
    add_message,
    create_conversation,
    delete_conversation,
    get_connection,
    get_conversation,
    get_messages,
    init_db,
    list_conversations,
    rename_conversation,
    update_message_metadata,
)

__all__ = [
    "init_db",
    "get_connection",
    "create_conversation",
    "list_conversations",
    "get_conversation",
    "rename_conversation",
    "delete_conversation",
    "add_message",
    "get_messages",
    "update_message_metadata",
]
