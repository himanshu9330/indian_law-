"""
SQLite database layer for LexAI.
Handles: users, chats, messages, feedback, bookmarks, roadmap progress.
All timestamps are UTC ISO-8601 strings.
"""

import hashlib
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import bcrypt
    _HAS_BCRYPT = True
except ImportError:
    _HAS_BCRYPT = False

from core.config import DB_PATH


def _hash_password(password: str) -> str:
    """Hash password using bcrypt if available, otherwise SHA-256 with salt."""
    if _HAS_BCRYPT:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"sha256${salt}${hashed}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash."""
    if stored_hash.startswith("sha256$"):
        parts = stored_hash.split("$")
        if len(parts) == 3:
            salt, expected = parts[1], parts[2]
            computed = hashlib.sha256((salt + password).encode()).hexdigest()
            return computed == expected
        return False
    if _HAS_BCRYPT:
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except Exception:
            return False
    return False


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


class Database:
    """Thread-safe SQLite wrapper for LexAI persistence."""

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_tables()

    # ── connection management ────────────────────────────────────────────

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cur = self._conn.execute(sql, params)
        self._conn.commit()
        return cur

    def _fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    # ── schema ───────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    UNIQUE NOT NULL,
                password_hash   TEXT    NOT NULL,
                created_at      TEXT    DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chats (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         INTEGER,
                title           TEXT    DEFAULT 'New Chat',
                pinned          INTEGER DEFAULT 0,
                created_at      TEXT    DEFAULT (datetime('now')),
                updated_at      TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id         INTEGER NOT NULL,
                role            TEXT    NOT NULL CHECK(role IN ('user','assistant','system')),
                content         TEXT    NOT NULL,
                metadata_json   TEXT,
                created_at      TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL,
                rating          INTEGER NOT NULL CHECK(rating IN (-1, 1)),
                comment         TEXT,
                created_at      TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS bookmarks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL UNIQUE,
                user_id         INTEGER,
                note            TEXT,
                created_at      TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id)    REFERENCES users(id)    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS roadmap_progress (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id      INTEGER NOT NULL,
                step_index      INTEGER NOT NULL,
                completed       INTEGER DEFAULT 0,
                updated_at      TEXT    DEFAULT (datetime('now')),
                FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
                UNIQUE(message_id, step_index)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
            CREATE INDEX IF NOT EXISTS idx_chats_user    ON chats(user_id);
            CREATE INDEX IF NOT EXISTS idx_feedback_msg  ON feedback(message_id);
        """)

    # ═══════════════════════════════════════════════════════════════════════
    # USERS
    # ═══════════════════════════════════════════════════════════════════════

    def create_user(self, username: str, password: str) -> int:
        pw_hash = _hash_password(password)
        cur = self._exec(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, pw_hash),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def authenticate_user(self, username: str, password: str) -> Optional[dict]:
        user = self._fetch_one("SELECT * FROM users WHERE username = ?", (username,))
        if user and _verify_password(password, user["password_hash"]):
            return user
        return None

    def get_user(self, user_id: int) -> Optional[dict]:
        return self._fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))

    # ═══════════════════════════════════════════════════════════════════════
    # CHATS
    # ═══════════════════════════════════════════════════════════════════════

    def create_chat(self, user_id: Optional[int] = None, title: str = "New Chat") -> int:
        now = _now()
        cur = self._exec(
            "INSERT INTO chats (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, title, now, now),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def get_chats(self, user_id: Optional[int] = None) -> list[dict]:
        if user_id is not None:
            return self._fetch_all(
                "SELECT * FROM chats WHERE user_id = ? ORDER BY pinned DESC, updated_at DESC",
                (user_id,),
            )
        return self._fetch_all(
            "SELECT * FROM chats WHERE user_id IS NULL ORDER BY pinned DESC, updated_at DESC"
        )

    def get_chat(self, chat_id: int) -> Optional[dict]:
        return self._fetch_one("SELECT * FROM chats WHERE id = ?", (chat_id,))

    def update_chat_title(self, chat_id: int, title: str) -> None:
        self._exec(
            "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), chat_id),
        )

    def touch_chat(self, chat_id: int) -> None:
        self._exec("UPDATE chats SET updated_at = ? WHERE id = ?", (_now(), chat_id))

    def pin_chat(self, chat_id: int, pinned: bool = True) -> None:
        self._exec(
            "UPDATE chats SET pinned = ? WHERE id = ?", (1 if pinned else 0, chat_id)
        )

    def delete_chat(self, chat_id: int) -> None:
        self._exec("DELETE FROM chats WHERE id = ?", (chat_id,))

    def search_chats(self, query: str, user_id: Optional[int] = None) -> list[dict]:
        q = f"%{query}%"
        if user_id is not None:
            return self._fetch_all(
                """SELECT DISTINCT c.* FROM chats c
                   LEFT JOIN messages m ON m.chat_id = c.id
                   WHERE c.user_id = ? AND (c.title LIKE ? OR m.content LIKE ?)
                   ORDER BY c.updated_at DESC""",
                (user_id, q, q),
            )
        return self._fetch_all(
            """SELECT DISTINCT c.* FROM chats c
               LEFT JOIN messages m ON m.chat_id = c.id
               WHERE c.user_id IS NULL AND (c.title LIKE ? OR m.content LIKE ?)
               ORDER BY c.updated_at DESC""",
            (q, q),
        )

    # ═══════════════════════════════════════════════════════════════════════
    # MESSAGES
    # ═══════════════════════════════════════════════════════════════════════

    def add_message(
        self,
        chat_id: int,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> int:
        meta_str = json.dumps(metadata) if metadata else None
        cur = self._exec(
            "INSERT INTO messages (chat_id, role, content, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, role, content, meta_str, _now()),
        )
        self.touch_chat(chat_id)
        return cur.lastrowid  # type: ignore[return-value]

    def get_messages(self, chat_id: int) -> list[dict]:
        rows = self._fetch_all(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        )
        for r in rows:
            r["metadata"] = json.loads(r["metadata_json"]) if r.get("metadata_json") else {}
        return rows

    def get_message(self, message_id: int) -> Optional[dict]:
        row = self._fetch_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        if row:
            row["metadata"] = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return row

    def get_last_assistant_message(self, chat_id: int) -> Optional[dict]:
        row = self._fetch_one(
            "SELECT * FROM messages WHERE chat_id = ? AND role = 'assistant' ORDER BY created_at DESC LIMIT 1",
            (chat_id,),
        )
        if row:
            row["metadata"] = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
        return row

    # ═══════════════════════════════════════════════════════════════════════
    # FEEDBACK
    # ═══════════════════════════════════════════════════════════════════════

    def add_feedback(self, message_id: int, rating: int, comment: str = "") -> int:
        cur = self._exec(
            "INSERT OR REPLACE INTO feedback (message_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
            (message_id, rating, comment, _now()),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def get_feedback(self, message_id: int) -> Optional[dict]:
        return self._fetch_one(
            "SELECT * FROM feedback WHERE message_id = ?", (message_id,)
        )

    # ═══════════════════════════════════════════════════════════════════════
    # BOOKMARKS
    # ═══════════════════════════════════════════════════════════════════════

    def add_bookmark(self, message_id: int, user_id: Optional[int] = None, note: str = "") -> int:
        cur = self._exec(
            "INSERT OR IGNORE INTO bookmarks (message_id, user_id, note, created_at) VALUES (?, ?, ?, ?)",
            (message_id, user_id, note, _now()),
        )
        return cur.lastrowid  # type: ignore[return-value]

    def get_bookmarks(self, user_id: Optional[int] = None) -> list[dict]:
        if user_id is not None:
            rows = self._fetch_all(
                """SELECT b.*, m.content, m.metadata_json, m.created_at as msg_time, c.title as chat_title
                   FROM bookmarks b
                   JOIN messages m ON m.id = b.message_id
                   JOIN chats c ON c.id = m.chat_id
                   WHERE b.user_id = ?
                   ORDER BY b.created_at DESC""",
                (user_id,),
            )
        else:
            rows = self._fetch_all(
                """SELECT b.*, m.content, m.metadata_json, m.created_at as msg_time, c.title as chat_title
                   FROM bookmarks b
                   JOIN messages m ON m.id = b.message_id
                   JOIN chats c ON c.id = m.chat_id
                   WHERE b.user_id IS NULL
                   ORDER BY b.created_at DESC"""
            )
        for r in rows:
            r["metadata"] = json.loads(r["metadata_json"]) if r.get("metadata_json") else {}
        return rows

    def remove_bookmark(self, bookmark_id: int) -> None:
        self._exec("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))

    def is_bookmarked(self, message_id: int) -> bool:
        row = self._fetch_one("SELECT id FROM bookmarks WHERE message_id = ?", (message_id,))
        return row is not None

    # ═══════════════════════════════════════════════════════════════════════
    # ROADMAP PROGRESS
    # ═══════════════════════════════════════════════════════════════════════

    def update_roadmap_step(self, message_id: int, step_index: int, completed: bool) -> None:
        self._exec(
            """INSERT INTO roadmap_progress (message_id, step_index, completed, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(message_id, step_index)
               DO UPDATE SET completed = excluded.completed, updated_at = excluded.updated_at""",
            (message_id, step_index, 1 if completed else 0, _now()),
        )

    def get_roadmap_progress(self, message_id: int) -> dict[int, bool]:
        rows = self._fetch_all(
            "SELECT step_index, completed FROM roadmap_progress WHERE message_id = ?",
            (message_id,),
        )
        return {r["step_index"]: bool(r["completed"]) for r in rows}

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYTICS QUERIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_total_queries(self, user_id: Optional[int] = None) -> int:
        if user_id is not None:
            row = self._fetch_one(
                """SELECT COUNT(*) as cnt FROM messages m
                   JOIN chats c ON c.id = m.chat_id
                   WHERE m.role = 'user' AND c.user_id = ?""",
                (user_id,),
            )
        else:
            row = self._fetch_one("SELECT COUNT(*) as cnt FROM messages WHERE role = 'user'")
        return row["cnt"] if row else 0

    def get_risk_distribution(self, user_id: Optional[int] = None) -> list[dict]:
        base = """SELECT
                    json_extract(m.metadata_json, '$.risk') as risk,
                    COUNT(*) as count
                  FROM messages m
                  JOIN chats c ON c.id = m.chat_id
                  WHERE m.role = 'assistant' AND m.metadata_json IS NOT NULL"""
        if user_id is not None:
            base += " AND c.user_id = ?"
            base += " GROUP BY risk"
            return self._fetch_all(base, (user_id,))
        base += " GROUP BY risk"
        return self._fetch_all(base)

    def get_source_distribution(self, user_id: Optional[int] = None) -> list[dict]:
        base = """SELECT
                    json_extract(m.metadata_json, '$.source') as source,
                    COUNT(*) as count
                  FROM messages m
                  JOIN chats c ON c.id = m.chat_id
                  WHERE m.role = 'assistant' AND m.metadata_json IS NOT NULL"""
        if user_id is not None:
            base += " AND c.user_id = ?"
            base += " GROUP BY source"
            return self._fetch_all(base, (user_id,))
        base += " GROUP BY source"
        return self._fetch_all(base)

    def get_confidence_over_time(self, user_id: Optional[int] = None, limit: int = 50) -> list[dict]:
        base = """SELECT
                    m.created_at,
                    CAST(json_extract(m.metadata_json, '$.confidence_score') AS REAL) as score
                  FROM messages m
                  JOIN chats c ON c.id = m.chat_id
                  WHERE m.role = 'assistant' AND m.metadata_json IS NOT NULL
                    AND json_extract(m.metadata_json, '$.confidence_score') IS NOT NULL"""
        if user_id is not None:
            base += " AND c.user_id = ?"
            base += f" ORDER BY m.created_at DESC LIMIT {limit}"
            return self._fetch_all(base, (user_id,))
        base += f" ORDER BY m.created_at DESC LIMIT {limit}"
        return self._fetch_all(base)

    def get_feedback_stats(self, user_id: Optional[int] = None) -> dict:
        base = """SELECT
                    SUM(CASE WHEN f.rating = 1 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN f.rating = -1 THEN 1 ELSE 0 END) as negative,
                    COUNT(*) as total
                  FROM feedback f
                  JOIN messages m ON m.id = f.message_id
                  JOIN chats c ON c.id = m.chat_id"""
        if user_id is not None:
            base += " WHERE c.user_id = ?"
            row = self._fetch_one(base, (user_id,))
        else:
            row = self._fetch_one(base)
        return {
            "positive": row["positive"] or 0,
            "negative": row["negative"] or 0,
            "total": row["total"] or 0,
        } if row else {"positive": 0, "negative": 0, "total": 0}

    def get_queries_by_date(self, user_id: Optional[int] = None, days: int = 30) -> list[dict]:
        base = """SELECT
                    DATE(m.created_at) as date,
                    COUNT(*) as count
                  FROM messages m
                  JOIN chats c ON c.id = m.chat_id
                  WHERE m.role = 'user'
                    AND m.created_at >= datetime('now', ?)"""
        param = f"-{days} days"
        if user_id is not None:
            base += " AND c.user_id = ?"
            base += " GROUP BY DATE(m.created_at) ORDER BY date"
            return self._fetch_all(base, (param, user_id))
        base += " GROUP BY DATE(m.created_at) ORDER BY date"
        return self._fetch_all(base, (param,))

    # ═══════════════════════════════════════════════════════════════════════
    # DATA MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def clear_all_data(self, user_id: Optional[int] = None) -> None:
        if user_id is not None:
            chat_ids = [r["id"] for r in self._fetch_all(
                "SELECT id FROM chats WHERE user_id = ?", (user_id,)
            )]
            for cid in chat_ids:
                self._exec("DELETE FROM chats WHERE id = ?", (cid,))
            self._exec("DELETE FROM users WHERE id = ?", (user_id,))
        else:
            self._conn.executescript("""
                DELETE FROM roadmap_progress;
                DELETE FROM bookmarks;
                DELETE FROM feedback;
                DELETE FROM messages;
                DELETE FROM chats;
                DELETE FROM users;
            """)

    def get_db_size(self) -> str:
        size = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"
