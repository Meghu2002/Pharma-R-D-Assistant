# Pharma R&D Assistant — End-to-End Implementation Document

## 1. Purpose & scope

A retrieval-augmented generation (RAG) assistant for pharma R&D research. Users ask natural-language questions and get answers grounded in three combined sources: a locally-indexed knowledge base (clinical trial records, FDA adverse event reports, PubMed-style literature abstracts), a live internet fetch (PubMed + ClinicalTrials.gov, called on every query), and two small purpose-built ML models (adverse-event severity, clinical-trial phase prediction). A separate Inspector view exposes the raw retriever output directly, for verifying what the system surfaces before any LLM ever sees it.

This document describes the system as currently implemented, end to end: data → ingestion → backend → ML models → frontend → API → deployment.

## 2. High-level architecture

```
                        ┌─────────────────────────────────────┐
                        │         FastAPI app (main.py)         │
                        │  single process, single port :8000    │
                        │                                        │
   Browser  ───HTTP───▶ │  ┌────────────┐      ┌──────────────┐ │
                        │  │ StaticFiles │      │  API routes   │ │
                        │  │  mount "/"  │      │  (routes.py)  │ │
                        │  └────────────┘      └──────┬───────┘ │
                        │   serves index.html          │         │
                        └───────────────────────────────┼─────────┘
                                                         │
                        ┌────────────────────────────────┼─────────────────┐
                        │                core/ (business logic)             │
                        │                                                    │
                        │  vector_database.py   llm_chain_factory.py         │
                        │  (Chroma, embeddings)  (LLM call, agent orchestration)
                        │         │                       │  │              │
                        │         │              agent_tools.py  ml_tools.py │
                        │         │              (live fetch)   (severity/   │
                        │         │                              phase models)│
                        └─────────┼───────────────────────┼──────┼──────────┘
                                  │                       │      │
                          ┌───────▼────────┐   ┌──────────▼──┐  ┌▼─────────────┐
                          │  ChromaDB       │   │ Groq API /  │  │ local .pkl/  │
                          │  (on-disk,      │   │ Gemini API  │  │ .pt models   │
                          │  per-provider)  │   │ (external)  │  │ (in-process) │
                          └─────────────────┘   └─────────────┘  └──────────────┘
                                                        │
                                          ┌─────────────▼─────────────┐
                                          │ PubMed / ClinicalTrials.gov │
                                          │ (external, live per query) │
                                          └─────────────────────────────┘
```

One process serves everything — there is no separate frontend server. `client/static/index.html` is a single self-contained static file (HTML + CSS + vanilla JS, no build tooling, no framework) mounted at `/` by FastAPI's `StaticFiles`.

## 3. Data pipeline

### 3.1 Raw sources (`server/data/raw/`, not committed to git)
- **Clinical trials:** 214,129 raw JSON files (ClinicalTrials.gov study records), nested in subfolders.
- **FAERS:** 6 large JSON files (FDA Adverse Event Reporting System exports).
- **PubMed abstracts:** a flat text file of abstracts.

### 3.2 Curation (`server/data/curate_data.py`)
A one-off sampling script, run manually, not part of the API:
- Copies the PubMed abstracts file as-is (small, kept whole).
- Randomly samples 40,000 of the 214,129 clinical trial files (kept small enough to embed in reasonable time while remaining representative).
- Copies 3 of the 6 FAERS files (each ~600-700MB, so only a subset is used).
- Output goes to `server/data/curated/`.

### 3.3 Ingestion (`server/ingest_data.py`)
Also a one-off manual script (`python ingest_data.py`, not an API endpoint):
- **Clinical trials:** parses `protocolSection` → builds a text block per trial (title, NCT ID, status, phase, conditions, summary). Metadata: `{"source": filename, "type": "clinical_trial"}`.
- **FAERS:** takes up to 3,000 records per file, extracts drug names, reactions, seriousness flag, patient age/sex into a formatted text block. Metadata: `{"source": filename, "type": "adverse_event"}`.
- **PubMed:** splits the abstracts file on blank-line boundaries, keeps any block over 50 characters. Metadata: `{"source": "pubmed", "type": "literature"}`.
- All documents are then split with `TokenTextSplitter(chunk_size=300, chunk_overlap=30)` and embedded into Chroma in batches of 200, hardcoded to the `"groq"` provider's embedding space (`sentence-transformers/all-MiniLM-L12-v2`).
- **Current state:** 80,172 chunks indexed for Groq; 0 for Gemini — ingestion has only ever been run once, against the Groq embedding space. Gemini's vectorstore exists but is empty.

### 3.4 Runtime uploads (`server/core/document_processor.py`)
Separate from the bulk pipeline above — this is what the `/upload_and_process_pdfs` endpoint uses for user-uploaded files at runtime:
- Accepts `.pdf .txt .docx .csv .html .htm .json .xlsx .xls`, max 200MB per file.
- Per-extension loader: `PyPDFLoader`, `TextLoader`, `Docx2txtLoader`, `CSVLoader`, `UnstructuredHTMLLoader`, `UnstructuredExcelLoader`.
- Splits with `TokenTextSplitter(chunk_size=500, chunk_overlap=50)` — a different chunk size than the bulk-ingestion pipeline (500/50 vs. 300/30); the two pipelines aren't harmonized, just built at different times for different purposes.
- Appends into whichever provider's vectorstore the user has selected (`upsert_vectorstore_from_pdfs`), creating it fresh if it doesn't exist yet, or calling `.add_documents()` on the cached instance if it does.

## 4. Vectorstore & embeddings (`server/core/vector_database.py`)

- **Vector DB:** ChromaDB, persisted to disk at `server/data/{provider}_vector_store/`. One separate collection per LLM provider, because embeddings differ per provider.
- **Embeddings:**
  - `groq` → `HuggingFaceEmbeddings("sentence-transformers/all-MiniLM-L12-v2")` — runs locally on CPU.
  - `gemini` → `GoogleGenerativeAIEmbeddings("models/embedding-001")` — Google's API, no local compute.
- **Caching (added this project, not in the original template):**
  - `get_embeddings()` is `@lru_cache`d per provider — without this, a brand-new HuggingFace model was being loaded from scratch (~5.7s) on *every single* chat/search/upload request.
  - `_vectorstore_cache` (module-level dict) caches the Chroma connection object per provider, so it's opened once and reused rather than reopened per request.
  - `initialize_empty_vectorstores()` (run at app startup, in a background thread — see §7) proactively warms both the embeddings cache and the vectorstore cache, including running a throwaway `similarity_search("warmup", k=1)` query, because Chroma lazily loads its on-disk index on the *first query* rather than on construction — without this warmup query, the first real request after boot still paid a multi-second cost even with caching in place.
- **Search:** `find_similar_chunks()` uses `similarity_search_with_relevance_scores(query, k=5)`, returning `{page_content, metadata, score}` per result (score clamped to `[0, 1]`) — used by both `/vector_store/search` (Inspector) and available for citation display.
- **Citation paging:** `get_document_page(provider, source, page)` — given a source filename and a page number, fetches *all* chunks for that source via Chroma's metadata filter (`vectorstore.get(where={"source": source})`), figures out the full set of distinct pages that document has, and returns the concatenated text for the requested page plus the full page list — this is what powers the Inspector's "flip through the source document" citation panel.

## 5. RAG + agent orchestration (`server/core/llm_chain_factory.py`)

`run_agent_query(model_provider, model, vectorstore, message)` is the core pipeline behind `/chat`:

1. **Local retrieval:** `vectorstore.similarity_search(message, k=4)` — top 4 chunks, labeled `[Local knowledge base result]` in the context fed to the LLM.
2. **Live internet fetch — always runs, unconditionally** (this was originally gated on freshness keywords like "latest"/"recent" or an empty local result set; that gating was removed per explicit request so every query gets both local and live context regardless of phrasing):
   - `fetch_live_pubmed(query)` and `fetch_live_clinicaltrials(query)` (`server/core/agent_tools.py`) run **in parallel** via `ThreadPoolExecutor(max_workers=2)`.
   - PubMed: `esearch` → get up to 3 PMIDs → `esummary` (titles) + `efetch` (abstract text, rettype=abstract) → returns `{"texts": [...], "sources": [{"title", "url": "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}]}`.
   - ClinicalTrials.gov: queries the v2 studies API (`query.term`, `pageSize=3`) → returns `{"texts": [...], "sources": [{"title", "url": "https://clinicaltrials.gov/study/{nctId}"}]}`.
   - Both are wrapped in try/except returning empty results on any failure (network error, timeout, malformed response) — a failed live fetch never breaks the chat response, it just contributes nothing.
3. **ML tool triggers** (keyword-gated, unlike live fetch):
   - Severity prediction triggers on `risk|severity|serious|how dangerous|how severe` in the query; extracts drug count / reaction count / age from the query text via regex, then runs the decision tree model.
   - Phase prediction triggers on `what phase|predict phase|which phase|classify phase`; runs the neural net on the raw query text.
4. **Context assembly:** local + live + ML-tool text blocks are concatenated and stuffed into a single prompt (`create_stuff_documents_chain` pattern, though the actual call path bypasses the LangChain retrieval chain object and does the LLM call directly via `llm.invoke(formatted_prompt)` for full control over the manually-assembled context).
5. **LLM call:** `get_llm(provider, model)` — `ChatGroq` or `ChatGoogleGenerativeAI`, `@lru_cache`d per (provider, model) pair so the client object isn't reconstructed every request.
6. **Response:** `{answer, trace}` where `trace` = `{used_live_fetch: true, used_severity_model, used_phase_model, local_chunks_found, live_chunks_found, live_sources: [...], sources: [...]}` (`sources` = local chunk metadata, `live_sources` = internet reference titles+URLs).

**System prompt** (`get_prompt()`): instructs the model that context may include retrieved documents, live search results, and ML predictions (labeled accordingly), to synthesize all of it, and to only say "I don't know" if the context truly has nothing relevant.

## 6. ML models (`server/ml/`)

Two small models, trained offline by standalone scripts (not part of the API — run manually, artifacts committed to `server/ml/models/`):

### 6.1 Severity classifier (`train_decision_tree.py` → `severity_model.pkl`)
- **Type:** `sklearn.tree.DecisionTreeClassifier` (`max_depth=6, min_samples_leaf=10`).
- **Features:** `[num_drugs, num_reactions, age, sex]`, extracted per-FAERS-record.
- **Label:** binary — `serious` flag from the FAERS record (1 = Serious, 0 = Not Serious).
- **Inference (`ml_tools.predict_severity`):** given drug count / reaction count / age (extracted from the user's natural-language query via regex in `agent_tools.extract_severity_features`), returns `{prediction: "Serious"|"Not Serious", confidence: %}`.

### 6.2 Trial-phase classifier (`train_neural_net.py` → `phase_classifier.pt` + `phase_vectorizer.pkl` + `phase_label_encoder.pkl` + `phase_model_config.pkl`)
- **Type:** small PyTorch feed-forward net — `Linear → ReLU → Dropout(0.2) → Linear`, hidden size 64.
- **Features:** bag-of-words via `sklearn.feature_extraction.text.CountVectorizer` (max 1000 features, English stopwords removed) over `{trial title} {trial summary}`.
- **Label:** the trial's first listed phase (multi-class, encoded via `sklearn.LabelEncoder`).
- **Training:** 15 epochs, Adam optimizer (`lr=0.001`), cross-entropy loss, 80/20 train/test split.
- **Inference (`ml_tools.predict_trial_phase`):** vectorizes the raw query text with the *same* fitted vectorizer, runs the net, softmaxes, returns `{prediction: <phase label>, confidence: %}`.

Both models are loaded lazily on first use and cached in module-level globals (`_severity_model`, `_phase_model`, etc.) in `ml_tools.py` — loaded once per process, not per request.

## 7. Backend app & performance

`server/main.py`:
- Mounts the API router, then mounts `client/static/` at `/` via `StaticFiles(html=True)` — API routes are matched first (registered first), the static mount catches everything else including `/` itself (serving `index.html`).
- **Startup:** `initialize_empty_vectorstores()` runs in a **background daemon thread**, not awaited inline. This was a deliberate fix: originally it ran synchronously in the `startup` event, which warmed caches correctly but made the server take ~15-20s to even start accepting connections. Running it in a background thread lets the port open immediately; a request that lands before warmup finishes just pays the (same, pre-optimization) cost inline instead of every launch blocking on it.
- Run via `python main.py` (uses `uvicorn.run(..., reload=True)`) or `uvicorn main:app --reload`.

**Caching summary (all added this project):**
| What | Where | Why |
|---|---|---|
| Embeddings model per provider | `@lru_cache` on `get_embeddings()` | Avoid reloading a local sentence-transformers model (~5.7s) every request |
| Chroma connection per provider | `_vectorstore_cache` dict | Avoid reopening the persisted collection every request |
| LLM client per (provider, model) | `@lru_cache` on `get_llm()` | Avoid reconstructing the API client every request |
| Vectorstore index warmup | throwaway query at startup | Chroma loads its on-disk index lazily on first query, not on construction |
| Live fetch parallelism | `ThreadPoolExecutor(max_workers=2)` | PubMed + ClinicalTrials.gov fetched concurrently instead of sequentially |

**Measured impact:** vectorstore search went from ~8s cold to ~0.3-0.5s; a plain local-only chat round-trip from 8+ seconds of pure overhead to ~2.8s total. (Live-fetch chat responses now run ~6-14s regardless, since that path always makes 2 external API calls — an explicit, accepted tradeoff for always surfacing internet sources.)

## 8. Frontend (`client/static/index.html`)

Single file, ~1000 lines, no build step, no framework — plain HTML/CSS + one `<script>` IIFE managing a small hand-rolled state/render loop (a `State` object, an `Actions` object of event handlers, and a `render()` function that re-syncs the DOM to `State` after every action).

**Design origin:** built to pixel-match a provided design mockup (`Pharma RAG Assistant v2.dc.html`), then wired to the real backend in place of the mockup's canned fake data.

**Views:**
- **Chat** — sidebar (logo, New chat, recent sessions persisted in `localStorage`, Settings menu with Reset/Clear/Undo/Export-CSV), empty state with 3 suggestion pills, message thread (user bubbles right-aligned, AI messages rendered as sanitized Markdown), expandable "How I got this answer" trace per AI message showing two distinct labeled groups — **"From local data"** (clickable, opens the citation viewer) and **"From the internet"** (clickable, real PubMed/ClinicalTrials.gov links) — plus a composer with an attach (upload) popover and a model-provider/model popover.
- **Inspector** — documents-indexed count, active model, a test-retrieval search box, ranked/scored matching chunks, and a slide-in citation panel with page-by-page navigation through a source document (backed by `/vector_store/document/{provider}`). No LLM call happens in this view at all — pure retrieval diagnostics.

**Markdown rendering:** AI answers often contain real Markdown (tables, bold, lists) from the LLM. Rendered via `marked` (parser) + `DOMPurify` (XSS-safe sanitizer), both loaded from CDN with `defer` (non-blocking — an earlier version without `defer` could make the whole page appear to hang if the CDN was slow to respond).

**Color palette — "Clinical Slate & Teal"** (an original palette, replacing the mockup's default warm cream/orange, chosen to read as professional/clinical rather than a generic AI-tool look): primary text `#16232D`, accent `#0E7C74` (deep teal), surfaces `#ECF2F4`, borders in the `#DCE6EB → #B7C7D1` range. Applied as literal hex values per CSS rule (no CSS custom properties in this file).

**Notable bugs fixed during implementation** (all in this single file):
- Provider→models cache was pre-seeded with empty arrays; since `[]` is truthy in JS, the "should I fetch this provider's models" check never fired, so reasoning models never populated and the Send button was permanently disabled. Fixed by restructuring the cache to only populate lazily on first real selection.
- The composer wasn't pinned to the bottom of the page before a model was selected, because the empty-state block (the layout's only flex-grow spacer) was fully hidden (`display:none`) whenever `chatReady` was false, collapsing the flex layout. Fixed by always rendering the empty-state container and only toggling its *inner content* based on readiness.
- Inspector view didn't scroll — it was toggled to `display:"block"` instead of `display:"flex"`, so its scrollable child never got a bounded height from `flex:1` and content was clipped by the parent's `overflow:hidden` instead of scrolling.

## 9. API reference

| Method | Path | Request | Response `data` |
|---|---|---|---|
| GET | `/health` | — | `"ok"` |
| GET | `/llm` | — | `["Groq", "Gemini"]` |
| GET | `/llm/{provider}` | — | list of model names for that provider |
| POST | `/upload_and_process_pdfs` | multipart: `files[]`, `model_provider` | `"PDFs processed successfully."` |
| GET | `/vector_store/count/{provider}` | — | integer chunk count |
| POST | `/vector_store/search` | `{model_provider, query}` | `[{page_content, metadata, score}]` |
| GET | `/vector_store/document/{provider}?source=&page=` | — | `{text, page, total_pages, pages}` |
| POST | `/chat` | `{model_provider, model_name, message}` | `{answer, trace}` |

All responses share the envelope `{status: "success"|"error", data, message}` (`server/api/schemas.py: StandardAPIResponse`).

## 10. Configuration & secrets (`server/config/settings.py`, `server/.env`)

```python
MODEL_OPTIONS = {
  "groq":   {"models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]},
  "gemini": {"models": ["gemini-2.0-flash", "gemini-2.5-flash"]}
}
```
- `.env` (gitignored) holds `GROQ_API_KEY` (real, working), `GOOGLE_API_KEY` (currently a **placeholder** — Gemini chat will not work until replaced with a real key), `OPENAI_API_KEY` (loaded but never referenced anywhere in the code — dead config, harmless).
- Vectorstore paths: `server/data/{provider}_vector_store/`.
- Upload staging directory: `server/temp/uploaded_files/`.

## 11. Security notes

- **No authentication anywhere** — anyone who can reach the server (or a shared tunnel URL) can chat, upload documents, and consume the configured API keys. Acceptable for local dev/UAT; not acceptable as-is for any real internet-facing deployment.
- **XSS:** AI-generated Markdown is rendered via `innerHTML`, but always passed through `DOMPurify.sanitize()` first — chosen specifically because the content is LLM output, not literal user-authored HTML, but still an external/untrusted string.
- **Secrets:** `.env` is gitignored and confirmed never committed.
- **Uploads:** validated by extension allowlist and a 200MB size cap (`document_processor.validate_file`); no virus/content scanning beyond that.

## 12. Deployment & UAT

**Local run:**
```powershell
cd server
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
Open `http://127.0.0.1:8000/` — one process serves both API and frontend.

**UAT sharing (current setup):** `cloudflared` (Cloudflare Tunnel), installed via `winget`, run as `cloudflared tunnel --protocol http2 --url http://localhost:8000`. This is an **anonymous "quick tunnel"** — no Cloudflare account, gives a random `*.trycloudflare.com` URL that changes every time the tunnel restarts, and depends entirely on the laptop staying powered on, connected, and not asleep. (The default QUIC protocol was found to silently die after periods of network inactivity and get stuck in an unrecoverable retry loop; `--protocol http2` was used instead for reliability.) Locking the screen alone does not stop anything (Windows keeps background processes running while locked); the laptop's power plan sleep timer is what actually kills it if the machine sits idle too long.

**Not yet done, discussed as future options:**
- A Cloudflare *named* tunnel (needs a free account) for a stable, non-random URL — still laptop-dependent.
- A real hosted deployment (Render/Railway/Fly.io/etc.) for laptop-independent uptime — would need containerization, moving the vectorstore + ML model files over, API keys as host secrets, and a tier with persistent (non-ephemeral) disk. **No GPU is required** for this app under any hosting option — all LLM inference already happens on Groq's/Google's infrastructure via API; local compute is limited to a small CPU-bound embeddings model and two tiny ML models, none of which benefit from a GPU at this scale.

## 13. Known gaps / limitations

- **Gemini vectorstore is empty** (0 chunks) — ingestion has only ever targeted the Groq embedding space. Selecting Gemini in the UI gives zero local retrieval results; chat would fall back to live-fetch + ML tools only.
- **Live internet fetch runs on every chat message, unconditionally** — adds a few seconds of latency to every response and makes two external API calls per message, by design (explicit tradeoff, not an oversight).
- **No automated tests** anywhere in the codebase — all verification during implementation was manual (curl checks, log inspection, visual review).
- **Two different chunking configurations** exist side by side (bulk ingestion: 300/30 tokens; runtime upload: 500/50 tokens) and were never reconciled.
- **`OPENAI_API_KEY`** is loaded into settings but referenced nowhere else — safe to remove if cleaning up config.
- The old Streamlit-based client (this project's original interface) was deleted from the working tree; it is recoverable from git commit `ef47f96` if ever needed for reference (`git checkout ef47f96 -- client/app.py`, etc.), but the active, documented frontend is the static file described in §8.
