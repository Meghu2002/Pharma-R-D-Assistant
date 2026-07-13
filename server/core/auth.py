import hashlib
import os
from datetime import datetime, timezone

from fastapi import Request

from core import auth_db

SESSION_COOKIE_NAME = "session_token"
PBKDF2_ITERATIONS = 260000


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return candidate == password_hash


def signup(username: str, password: str) -> dict:
    username = username.strip()
    if not username or len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if auth_db.get_user_by_username(username):
        raise ValueError("That username is already taken.")

    password_hash, salt = hash_password(password)
    user_id = auth_db.create_user(username, password_hash, salt)
    token = auth_db.create_auth_session(user_id)
    return {"user_id": user_id, "username": username, "token": token}


def login(username: str, password: str) -> dict:
    username = username.strip()
    user = auth_db.get_user_by_username(username)

    if user and user["locked_until"]:
        locked_until = datetime.fromisoformat(user["locked_until"])
        if locked_until > datetime.now(timezone.utc):
            minutes_left = max(1, int((locked_until - datetime.now(timezone.utc)).total_seconds() // 60) + 1)
            raise ValueError(f"Too many failed attempts. Try again in {minutes_left} minute(s).")

    if not user or not verify_password(password, user["password_hash"], user["salt"]):
        if user:
            auth_db.record_failed_login(username)
        raise ValueError("Invalid username or password.")

    auth_db.reset_failed_login(username)
    token = auth_db.create_auth_session(user["id"])
    return {"user_id": user["id"], "username": user["username"], "token": token}


def logout(token: str | None):
    if token:
        auth_db.delete_auth_session(token)


def get_current_user(request: Request) -> dict | None:
    """Returns {user_id, username} if the request carries a valid session
    cookie, otherwise None. Never raises — auth is optional for most routes,
    callers that require it check for None themselves and return 401."""
    token = request.cookies.get(SESSION_COOKIE_NAME)
    user = auth_db.get_user_by_token(token)
    if not user:
        return None
    return {"user_id": user["id"], "username": user["username"]}
