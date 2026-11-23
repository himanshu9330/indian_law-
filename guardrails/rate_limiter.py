"""
LexAI Rate Limiter & Abuse Protection
Enforces per-session, per-conversation, and abuse cooldown limits stored in SQLite
so limits persist across page refreshes and browser restarts.

Guarantees:
- Per-minute limit: prevents spam bursts (friendly waiting message).
- Per-day limit: prevents daily quota exhaustion.
- Per-conversation limit: prevents runaway unbounded threads.
- Repeated block cooldown: 5 safety blocks in 10 minutes triggers 1-minute timeout.
- Clean and friendly user feedback without crashing the chat.
"""

import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from core.config import DB_PATH
from guardrails.config import GUARDRAILS_CONFIG
from guardrails.types import GuardrailResult

logger = logging.getLogger("lexai.guardrails.rate_limiter")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a configured SQLite connection for rate limit records."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_rate_limiter_db(db_path: Optional[str] = None) -> None:
    """Create rate_limit_records table and index if they do not exist."""
    try:
        with get_connection(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_records (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier  TEXT NOT NULL,
                    scope       TEXT NOT NULL,
                    timestamp   REAL NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_limit_ident_time
                ON rate_limit_records(identifier, timestamp);
            """)
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter table: {e}")


# Initialize table on module import
init_rate_limiter_db()


def record_rate_limit_event(identifier: str, scope: str = "session", db_path: Optional[str] = None) -> None:
    """Record an action timestamp in SQLite."""
    now = time.time()
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO rate_limit_records (identifier, scope, timestamp) VALUES (?, ?, ?)",
                (identifier, scope, now),
            )
            # Periodic cleanup of records older than 24 hours
            conn.execute("DELETE FROM rate_limit_records WHERE timestamp < ?", (now - 86400,))
    except Exception as e:
        logger.error(f"Failed to record rate limit event: {e}")


def check_repeated_blocks_cooldown(
    conversation_id: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[bool, Optional[float]]:
    """
    Check if the user has triggered >= repeated_block_threshold guardrail blocks
    within repeated_block_window_seconds (default: 5 blocks in 10 minutes).
    If so, checks if within block_cooldown_seconds (default: 60s).
    Returns (is_in_cooldown, remaining_seconds).
    """
    now = time.time()
    cutoff_epoch = now - GUARDRAILS_CONFIG.repeated_block_window_seconds

    try:
        with get_connection(db_path) as conn:
            # Query recent blocks for this conversation or session
            if conversation_id:
                query = """
                    SELECT timestamp FROM guardrail_events
                    WHERE action = 'block' AND conversation_id = ?
                    ORDER BY id DESC LIMIT 100
                """
                rows = conn.execute(query, (conversation_id,)).fetchall()
            else:
                query = """
                    SELECT timestamp FROM guardrail_events
                    WHERE action = 'block'
                    ORDER BY id DESC LIMIT 100
                """
                rows = conn.execute(query).fetchall()

            recent_blocks = 0
            most_recent_epoch = None

            for row in rows:
                ts_str = row[0]
                try:
                    clean_ts = ts_str.replace(" ", "T")
                    # If no timezone is present, assume UTC
                    if not clean_ts.endswith("Z") and "+" not in clean_ts and "-" not in clean_ts[10:]:
                        clean_ts += "+00:00"
                    dt = datetime.fromisoformat(clean_ts)
                    epoch = dt.timestamp()
                except Exception:
                    epoch = now

                if epoch >= cutoff_epoch:
                    recent_blocks += 1
                    if most_recent_epoch is None:
                        most_recent_epoch = epoch

            if recent_blocks >= GUARDRAILS_CONFIG.repeated_block_threshold and most_recent_epoch:
                elapsed = now - most_recent_epoch
                if elapsed < GUARDRAILS_CONFIG.block_cooldown_seconds:
                    remaining = GUARDRAILS_CONFIG.block_cooldown_seconds - elapsed
                    return True, remaining
    except Exception as e:
        logger.error(f"Error checking block cooldown: {e}")

    return False, None


def check_rate_limits(text: str, context: Optional[Dict[str, Any]] = None) -> GuardrailResult:
    """
    Check 0 (Abuse Prevention & Rate Limits):
    1. Repeated blocks cooldown (5 blocks in 10 mins -> 60s cooldown).
    2. Per-minute limit (max messages/min).
    3. Per-day limit (max messages/day).
    4. Per-conversation limit (max messages/conv).
    """
    if not GUARDRAILS_CONFIG.enable_rate_limiting:
        return GuardrailResult.allow(text)

    conv_id = context.get("conversation_id") if context else None
    session_id = context.get("session_id") or conv_id or "default_session"
    db_path = context.get("db_path") if context else None
    now = time.time()

    # 1. Check Repeated Blocks Cooldown
    in_cooldown, remaining = check_repeated_blocks_cooldown(conv_id, db_path=db_path)
    if in_cooldown:
        wait_sec = int(remaining or 60)
        user_msg = (
            f"Multiple requests were flagged by safety guardrails in a short period. "
            f"Please wait {wait_sec} seconds before trying again."
        )
        return GuardrailResult.block(
            reason_code="repeated_blocks_cooldown",
            user_message=user_msg,
        )

    # 2. Check Per-Minute Limit (session_id or conversation_id)
    minute_ago = now - 60.0
    try:
        with get_connection(db_path) as conn:
            count_min = conn.execute(
                "SELECT COUNT(*) FROM rate_limit_records WHERE identifier = ? AND timestamp >= ?",
                (session_id, minute_ago),
            ).fetchone()[0]

            if count_min >= GUARDRAILS_CONFIG.max_messages_per_minute:
                logger.warning(f"Rate limit per minute exceeded for session '{session_id}': {count_min} msgs")
                return GuardrailResult.block(
                    reason_code="rate_limit_minute",
                    user_message=GUARDRAILS_CONFIG.get_refusal("rate_limit_minute"),
                )

            # 3. Check Per-Day Limit
            day_ago = now - 86400.0
            count_day = conn.execute(
                "SELECT COUNT(*) FROM rate_limit_records WHERE identifier = ? AND timestamp >= ?",
                (session_id, day_ago),
            ).fetchone()[0]

            if count_day >= GUARDRAILS_CONFIG.max_messages_per_day:
                logger.warning(f"Rate limit per day exceeded for session '{session_id}': {count_day} msgs")
                return GuardrailResult.block(
                    reason_code="rate_limit_day",
                    user_message=GUARDRAILS_CONFIG.get_refusal("rate_limit_day"),
                )

            # 4. Check Per-Conversation Limit
            if conv_id:
                count_conv = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user'",
                    (conv_id,),
                ).fetchone()[0]

                if count_conv >= GUARDRAILS_CONFIG.max_messages_per_conversation:
                    logger.warning(f"Max conversation messages reached for '{conv_id}': {count_conv}")
                    return GuardrailResult.block(
                        reason_code="rate_limit_conversation",
                        user_message=GUARDRAILS_CONFIG.get_refusal("rate_limit_conversation"),
                    )

    except Exception as e:
        logger.error(f"Rate limit evaluation error (failing open): {e}")

    # Passed rate limit checks: Record this message timestamp
    record_rate_limit_event(session_id, scope="session", db_path=db_path)
    return GuardrailResult.allow(text)


def reset_rate_limits(identifier: Optional[str] = None, db_path: Optional[str] = None) -> None:
    """Reset rate limit counters (useful for unit testing and admin controls)."""
    try:
        with get_connection(db_path) as conn:
            if identifier:
                conn.execute("DELETE FROM rate_limit_records WHERE identifier = ?", (identifier,))
            else:
                conn.execute("DELETE FROM rate_limit_records;")
    except Exception as e:
        logger.error(f"Failed to reset rate limits: {e}")
