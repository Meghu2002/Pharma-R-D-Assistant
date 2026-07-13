# 🤖 RAG PDFBot - Server

This is the FastAPI backend for the RAG PDFBot. It handles document processing, vectorstore embedding, LLM chain execution, live internet retrieval, ML tool inference, optional account auth, and API endpoints — and it also serves the static frontend (`../client/static/`) directly, so this is the only process you need to run.

---

## Features

- ✅ Upload and process documents (PDF, DOCX, CSV, HTML, XLSX, JSON, TXT)
- 🧠 Chat with LLM using vectorstore retrieval, always combined with a live PubMed/ClinicalTrials.gov fetch
- 🧬 Adverse-event severity and clinical-trial phase prediction via small local ML models
- 🔍 Inspect document chunks via similarity search, with scores and a source citation viewer — no LLM call
- 🌐 Groq active (Gemini supported in code, disabled until its vectorstore is ingested — see `config/settings.py`)
- 👤 Optional account auth — guests chat with no login required; logging in links chat history to the account

---

## Project Structure

```
server/
├── api/                        # FastAPI routes and schemas
├── config/                     # Environment and constants
├── core/                       # LLM logic, vectorstore, agent tools, ML tools, auth
├── ml/                         # Trained model artifacts + training scripts
├── data/                       # Vectorstores, curated data, app.db (gitignored)
├── utils/                      # Logger
├── ingest_data.py              # One-off bulk data ingestion script
├── main.py                     # App entry point — also mounts the static frontend
```

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/Meghu2002/Pharma-R-D-Assistant.git
cd Pharma-R-D-Assistant
```

2. **Create a virtual environment (optional)**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
cd server

pip3 install -r requirements.txt
```

---

## Configuration

Set your API keys in a `.env` file in `server/` (`config/settings.py` reads them via `python-dotenv` — don't hardcode keys directly in `settings.py`):

- **Groq**: [console.groq.com](https://console.groq.com/)
- **Gemini**: [ai.google.dev](https://ai.google.dev)

```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
```

---

## ▶️ Usage

Run the app:

```bash
cd rag-bot-fastapi/server

uvicorn main:app --reload
```

---

## API Endpoints

- `/health`
- `/llm`
- `/llm/{provider}`
- `/upload_and_process_pdfs`
- `/vector_store/count/{provider}`
- `/vector_store/search`
- `/vector_store/document/{provider}` — full page text + page list, for citation paging
- `/chat` — always combines local retrieval with a live internet fetch
- `/auth/signup`, `/auth/login`, `/auth/logout`, `/auth/me`
- `/chat_sessions`, `/chat_sessions/{id}/messages` — account-linked chat history (requires login)

## Logging

Logs are printed to the console and controlled via `utils/logger.py`.
