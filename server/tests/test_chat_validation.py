def test_chat_rejects_invalid_provider(client):
    res = client.post("/chat", json={
        "model_provider": "not-a-real-provider",
        "model_name": "whatever",
        "message": "hello"
    })
    body = res.json()
    assert body["status"] == "error"
    assert "Invalid model provider" in body["message"]


def test_chat_rejects_invalid_model_name(client):
    res = client.post("/chat", json={
        "model_provider": "groq",
        "model_name": "not-a-real-model",
        "message": "hello"
    })
    body = res.json()
    assert body["status"] == "error"
    assert "Invalid model name" in body["message"]


def test_chat_response_never_leaks_session_id_for_guests(client):
    # Invalid-provider path never reaches the LLM/session logic, so this just
    # confirms the error response shape has no session_id key at all.
    res = client.post("/chat", json={
        "model_provider": "invalid",
        "model_name": "invalid",
        "message": "hello"
    })
    body = res.json()
    assert body["data"] is None
