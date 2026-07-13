import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from config.settings import AUTH_DB_PATH
from utils.logger import logger

MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(AUTH_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    logger.info("Initializing auth/chat-history database...")
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            );
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                trace_json TEXT,
                created_at TEXT NOT NULL
            );
        """)
        # Migration for databases created before login-lockout support existed —
        # CREATE TABLE IF NOT EXISTS above won't add columns to an already-existing
        # users table, so add them here if missing (harmless no-op otherwise).
        for statement in (
            "ALTER TABLE users ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE users ADD COLUMN locked_until TEXT"
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass  # column already exists
    logger.info("Auth/chat-history database ready.")


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_user(username, password_hash, salt):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, salt, _now())
        )
        return cur.lastrowid


def get_user_by_username(username):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def record_failed_login(username):
    with get_conn() as conn:
        row = conn.execute("SELECT failed_attempts FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return
        attempts = row["failed_attempts"] + 1
        locked_until = None
        if attempts >= MAX_FAILED_ATTEMPTS:
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
        conn.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
            (attempts, locked_until, username)
        )


def reset_failed_login(username):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?",
            (username,)
        )


def create_auth_session(user_id):
    token = uuid.uuid4().hex + uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO auth_sessions (token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, _now())
        )
    return token


def get_user_by_token(token):
    if not token:
        return None
    with get_conn() as conn:
        row = conn.execute(
            """SELECT users.* FROM auth_sessions
                  JOIN users ON users.id = auth_sessions.user_id
                  WHERE auth_sessions.token = ?""",
            (token,)
        ).fetchone()
        return dict(row) if row else None


def delete_auth_session(token):
    with get_conn() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))


def create_chat_session(user_id, title):
    session_id = uuid.uuid4().hex
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, title, created_at) VALUES (?, ?, ?, ?)",
            (session_id, user_id, title, _now())
        )
    return session_id


def add_chat_message(session_id, role, text, trace_json=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, text, trace_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, text, trace_json, _now())
        )


def list_chat_sessions(user_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM chat_sessions WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_chat_session(session_id):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None


def list_chat_messages(session_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT role, text, trace_json, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def delete_chat_session(session_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
