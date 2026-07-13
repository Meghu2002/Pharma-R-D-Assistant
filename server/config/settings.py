import os
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TEMPFILE_UPLOAD_DIRECTORY = "./temp/uploaded_files"
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "./data/app.db")

# Deployment settings. Defaults preserve today's local-dev behavior
# (auto-reload, HTTP-only cookies, 127.0.0.1) — set ENVIRONMENT=production
# (and optionally HOST/PORT) when actually deploying.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"
HOST = os.getenv("HOST", "0.0.0.0" if IS_PRODUCTION else "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))
MODEL_OPTIONS = {
    "groq": {
        "playground": "https://console.groq.com",
        "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    }
    # "gemini" temporarily disabled: its vectorstore has never been ingested
    # into (0 chunks), so it currently returns no local retrieval results.
    # Re-add here once ingest_data.py has been run against the gemini
    # embedding space too:
    # "gemini": {
    #   "playground": "https://ai.google.dev",
    #   "models": ["gemini-2.0-flash", "gemini-2.5-flash"]
    # }
}
VECTORSTORE_DIRECTORY = {
    key.lower(): f"./data/{key.lower()}_vector_store"
    for key in MODEL_OPTIONS.keys()
}