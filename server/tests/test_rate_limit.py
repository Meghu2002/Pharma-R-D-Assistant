def test_signup_endpoint_rate_limited_after_five_per_minute(client):
    # Distinct usernames each call so every request would otherwise succeed —
    # isolates "rate limiting fired" from any account-already-exists error.
    for i in range(5):
        res = client.post("/auth/signup", json={
            "username": f"ratelimit_user_{i}",
            "password": "correcthorse123"
        })
        assert res.status_code == 200

    sixth = client.post("/auth/signup", json={
        "username": "ratelimit_user_5",
        "password": "correcthorse123"
    })
    assert sixth.status_code == 429
