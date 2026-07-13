# Lockout is tested by calling core.auth directly rather than through the
# HTTP layer: the lockout threshold (5 failed attempts) and the /auth/login
# rate limit (5/minute) are numerically identical, which makes it ambiguous
# over HTTP which one fired. Testing the business logic directly sidesteps
# that entirely and is faster besides.
import pytest

from core import auth


def test_account_locks_after_max_failed_attempts():
    auth.signup("lockout_test_user", "correcthorse123")

    for _ in range(auth.auth_db.MAX_FAILED_ATTEMPTS):
        with pytest.raises(ValueError, match="Invalid username or password"):
            auth.login("lockout_test_user", "wrongpassword")

    # One more attempt, even with the *correct* password, should now be
    # rejected for being locked out rather than authenticated.
    with pytest.raises(ValueError, match="Too many failed attempts"):
        auth.login("lockout_test_user", "correcthorse123")


def test_successful_login_resets_failed_attempt_counter():
    auth.signup("lockout_reset_user", "correcthorse123")

    for _ in range(auth.auth_db.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(ValueError):
            auth.login("lockout_reset_user", "wrongpassword")

    # Correct password before hitting the threshold succeeds and clears the count.
    result = auth.login("lockout_reset_user", "correcthorse123")
    assert result["username"] == "lockout_reset_user"

    user = auth.auth_db.get_user_by_username("lockout_reset_user")
    assert user["failed_attempts"] == 0
    assert user["locked_until"] is None
