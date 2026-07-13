def test_signup_creates_account_and_sets_cookie(client):
    res = client.post("/auth/signup", json={"username": "alice_test", "password": "correcthorse123"})
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["username"] == "alice_test"
    assert "session_token" in res.cookies


def test_signup_duplicate_username_rejected(client):
    client.post("/auth/signup", json={"username": "bob_test", "password": "correcthorse123"})
    res = client.post("/auth/signup", json={"username": "bob_test", "password": "anotherpassword"})
    body = res.json()
    assert body["status"] == "error"
    assert "already taken" in body["message"]


def test_signup_rejects_short_password(client):
    res = client.post("/auth/signup", json={"username": "carol_test", "password": "short"})
    body = res.json()
    assert body["status"] == "error"


def test_login_success_recovers_history_access(client):
    client.post("/auth/signup", json={"username": "dave_test", "password": "correcthorse123"})
    res = client.post("/auth/login", json={"username": "dave_test", "password": "correcthorse123"})
    body = res.json()
    assert body["status"] == "success"
    assert body["data"]["username"] == "dave_test"


def test_login_wrong_password_rejected(client):
    client.post("/auth/signup", json={"username": "erin_test", "password": "correcthorse123"})
    res = client.post("/auth/login", json={"username": "erin_test", "password": "wrongpassword"})
    body = res.json()
    assert body["status"] == "error"
    assert "Invalid username or password" in body["message"]


def test_auth_me_reflects_login_state(client):
    signup_res = client.post("/auth/signup", json={"username": "frank_test", "password": "correcthorse123"})
    cookies = signup_res.cookies
    me_res = client.get("/auth/me", cookies=cookies)
    assert me_res.json()["data"]["username"] == "frank_test"


def test_auth_me_null_without_cookie(client):
    res = client.get("/auth/me")
    assert res.json()["data"]["username"] is None


def test_logout_clears_session(client):
    signup_res = client.post("/auth/signup", json={"username": "grace_test", "password": "correcthorse123"})
    cookies = signup_res.cookies
    client.post("/auth/logout", cookies=cookies)
    me_res = client.get("/auth/me", cookies=cookies)
    assert me_res.json()["data"]["username"] is None


def test_guest_chat_sessions_endpoint_requires_auth(client):
    res = client.get("/chat_sessions")
    body = res.json()
    assert body["status"] == "error"
    assert "Not authenticated" in body["message"]
