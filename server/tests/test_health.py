def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert body["data"] == "ok"


def test_llm_providers_lists_groq(client):
    res = client.get("/llm")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert "Groq" in body["data"]


def test_llm_models_for_groq(client):
    res = client.get("/llm/groq")
    body = res.json()
    assert body["status"] == "success"
    assert "openai/gpt-oss-20b" in body["data"]


def test_llm_models_invalid_provider(client):
    res = client.get("/llm/not-a-real-provider")
    body = res.json()
    assert body["status"] == "error"
