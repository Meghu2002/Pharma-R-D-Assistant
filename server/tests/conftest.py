import os
import tempfile

import pytest

# Point the auth/chat-history DB at a throwaway file *before* anything under
# server/ gets imported (config.settings reads AUTH_DB_PATH at import time),
# so tests never touch the real server/data/app.db or its accounts/history.
_test_db_fd, _test_db_path = tempfile.mkstemp(suffix=".db")
os.close(_test_db_fd)
os.environ["AUTH_DB_PATH"] = _test_db_path

from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402
from core.rate_limit import limiter  # noqa: E402
from core import auth_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _init_test_db():
    # Some tests (e.g. lockout) call core.auth directly rather than through
    # the HTTP layer, so they never trigger FastAPI's startup event — ensure
    # the schema exists regardless of which test runs first.
    auth_db.init_db()


@pytest.fixture(scope="session")
def _app_client():
    # Runs the FastAPI startup/shutdown lifecycle exactly once for the whole
    # test session (init_auth_db + the background vectorstore warmup thread).
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(_app_client):
    # Fresh cookie jar per test — TestClient's underlying httpx client
    # persists cookies across requests like a browser, so sharing one
    # instance across tests would leak session cookies from one test's
    # signup/login into a later test's "no cookie" assertions.
    _app_client.cookies.clear()
    return _app_client


@pytest.fixture(autouse=True)
def reset_rate_limits():
    # Each test gets a clean rate-limit slate — otherwise every test hitting
    # a decorated endpoint shares one global counter (TestClient requests all
    # come from the same fake IP), and call order/count across the whole
    # suite would silently affect unrelated tests.
    limiter.reset()
    yield


def pytest_sessionfinish(session, exitstatus):
    try:
        os.remove(_test_db_path)
    except OSError:
        pass
