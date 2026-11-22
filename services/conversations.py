"""
LexAI Conversation Persistence Service
Thread-safe SQLite storage for conversations and isolated messages.

Tables:
- conversations: id (UUID), title, created_at, updated_at
- messages: id, conversation_id (FK cascade), role, content, metadata_json, created_at
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.config import DB_PATH

logger = logging.getLogger("lexai.conversations")


def _now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a configured SQLite connection with foreign keys and WAL enabled."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    """Initialize conversations and messages tables if they do not exist, handling schema evolution."""
    with get_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id          TEXT    PRIMARY KEY,
                title       TEXT    NOT NULL DEFAULT 'New Consultation',
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );
        """)

        # Migrate any legacy chats into conversations if present
        tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "chats" in tables:
            conn.execute("""
                INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at)
                SELECT CAST(id AS TEXT), COALESCE(title, 'Consultation'), COALESCE(created_at, datetime('now')), COALESCE(updated_at, datetime('now'))
                FROM chats;
            """)

        # Migrate legacy messages table if it still has old chat_id constraint
        cols = [c[1] for c in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if not cols:
            conn.execute("""
                CREATE TABLE messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT    NOT NULL,
                    role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content         TEXT    NOT NULL,
                    metadata_json   TEXT,
                    created_at      TEXT    NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)
        elif "chat_id" in cols:
            conn.execute("""
                CREATE TABLE messages_new (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT    NOT NULL,
                    role            TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content         TEXT    NOT NULL,
                    metadata_json   TEXT,
                    created_at      TEXT    NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
            """)
            conn.execute("""
                INSERT OR IGNORE INTO messages_new (id, conversation_id, role, content, metadata_json, created_at)
                SELECT id, COALESCE(conversation_id, CAST(chat_id AS TEXT)), role, content, metadata_json, COALESCE(created_at, datetime('now'))
                FROM messages;
            """)
            conn.execute("DROP TABLE messages;")
            conn.execute("ALTER TABLE messages_new RENAME TO messages;")
        elif "conversation_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE;")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);")


# Auto-initialize database tables on module import
init_db()


def create_conversation(title: str = "New Consultation", db_path: Optional[str] = None) -> str:
    """
    Create a new isolated conversation with a unique UUID.
    Returns the conversation UUID string.
    """
    conv_id = str(uuid.uuid4())
    now = _now()
    clean_title = (title or "New Consultation").strip()

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, clean_title, now, now),
            )
        return conv_id
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        raise


def list_conversations(limit: int = 100, offset: int = 0, db_path: Optional[str] = None) -> list[dict[str, Any]]:
    """
    List past conversations ordered by updated_at descending.
    Includes message count and last message preview.
    """
    sql = """
        SELECT 
            c.id,
            c.title,
            c.created_at,
            c.updated_at,
            COUNT(m.id) AS message_count,
            (SELECT m2.content FROM messages m2 WHERE m2.conversation_id = c.id ORDER BY m2.id DESC LIMIT 1) AS last_message
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT ? OFFSET ?
    """
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(sql, (limit, offset)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        return []


def get_conversation(conversation_id: str, db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Fetch metadata for a single conversation."""
    sql = "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?"
    try:
        with get_connection(db_path) as conn:
            row = conn.execute(sql, (conversation_id,)).fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to get conversation {conversation_id}: {e}")
        return None


def rename_conversation(conversation_id: str, new_title: str, db_path: Optional[str] = None) -> bool:
    """Rename a conversation and update its timestamp."""
    clean_title = (new_title or "").strip()
    if not clean_title:
        return False
    now = _now()
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, conversation_id),
            )
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to rename conversation {conversation_id}: {e}")
        return False


def delete_conversation(conversation_id: str, db_path: Optional[str] = None) -> bool:
    """Delete a conversation and all its messages via foreign key cascade."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete conversation {conversation_id}: {e}")
        return False


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Append a message to an isolated conversation and touch the conversation's updated_at.
    Returns the inserted message ID.
    """
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"Invalid role: {role}. Must be 'user', 'assistant', or 'system'.")

    meta_json = json.dumps(metadata or {}, ensure_ascii=False) if metadata is not None else None
    now = _now()

    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, meta_json, now),
            )
            msg_id = cur.lastrowid

            # Touch the conversation timestamp
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            return msg_id
    except Exception as e:
        logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
        raise


def get_messages(conversation_id: str, db_path: Optional[str] = None) -> list[dict[str, Any]]:
    """
    Retrieve all messages belonging to a conversation in chronological order.
    Unpacks metadata_json into a native dict.
    """
    sql = """
        SELECT id, conversation_id, role, content, metadata_json, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(sql, (conversation_id,)).fetchall()
            messages = []
            for r in rows:
                d = dict(r)
                meta_raw = d.pop("metadata_json", None)
                if meta_raw:
                    try:
                        d["metadata"] = json.loads(meta_raw)
                    except Exception:
                        d["metadata"] = {}
                else:
                    d["metadata"] = {}
                messages.append(d)
            return messages
    except Exception as e:
        logger.error(f"Failed to get messages for conversation {conversation_id}: {e}")
        return []


def update_message_metadata(message_id: int, metadata: dict[str, Any], db_path: Optional[str] = None) -> bool:
    """Update metadata_json for a specific message."""
    meta_json = json.dumps(metadata or {}, ensure_ascii=False)
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "UPDATE messages SET metadata_json = ? WHERE id = ?",
                (meta_json, message_id),
            )
            return cur.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to update metadata for message {message_id}: {e}")
        return False
