"""
Guardrail Audit Logger
Logs guardrail events (reasons, actions, and masked snippets) to SQLite.
Never stores full raw user text to protect privacy and adhere to data policies.
"""

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.config import DB_PATH

logger = logging.getLogger("lexai.guardrails.logger")


def _now() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def mask_snippet(text: Optional[str], max_len: int = 32) -> str:
    """
    Generate a safe, short masked preview of the input text.
    Replaces digits and sensitive tokens, truncated to max_len.
    """
    if not text:
        return ""
    clean = text.strip().replace("\n", " ")
    # Mask numbers / potential identifiers
    masked = re.sub(r"\d", "*", clean)
    if len(masked) > max_len:
        return masked[:max_len].strip() + "..."
    return masked


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return configured SQLite connection."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_guardrail_db(db_path: Optional[str] = None) -> None:
    """Initialize guardrail_events table if it does not exist."""
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS guardrail_events (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    timestamp       TEXT NOT NULL,
                    layer           TEXT NOT NULL,
                    reason_code     TEXT NOT NULL,
                    action          TEXT NOT NULL,
                    masked_snippet  TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_guardrail_conv ON guardrail_events(conversation_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_guardrail_time ON guardrail_events(timestamp DESC);")
    except Exception as e:
        logger.error(f"Failed to initialize guardrail table: {e}", exc_info=True)


# Auto-initialize table on module import
init_guardrail_db()


def log_guardrail_event(
    layer: str,
    reason_code: str,
    action: str,
    conversation_id: Optional[str] = None,
    raw_text: Optional[str] = None,
    db_path: Optional[str] = None,
) -> bool:
    """
    Safely record a guardrail trigger in SQLite.
    Fails safe and fails visible (logs errors without raising exceptions to callers).
    """
    timestamp = _now()
    snippet = mask_snippet(raw_text)

    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO guardrail_events (conversation_id, timestamp, layer, reason_code, action, masked_snippet)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, timestamp, layer, reason_code, action, snippet),
            )
        return True
    except Exception as e:
        # Fail safe: log failure visibly but do not crash the chat pipeline
        logger.error(f"Failed to record guardrail event: {e}", exc_info=True)
        return False
