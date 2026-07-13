import threading
import uvicorn

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.routes import router
from core.vector_database import initialize_empty_vectorstores
from core.auth_db import init_db as init_auth_db
from core.rate_limit import limiter
from config.settings import HOST, PORT, IS_PRODUCTION
from utils.logger import logger


app = FastAPI(title="RAG PDFBot", description="Chat with multiple PDFs :books:")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent.parent / "client" / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up app...")
    init_auth_db()
    # Warm the embeddings/vectorstore caches in the background instead of
    # blocking startup — the server can accept connections immediately, and
    # requests that arrive before warmup finishes just pay that cost inline
    # (same as before caching was added), instead of every launch waiting on it.
    threading.Thread(target=initialize_empty_vectorstores, daemon=True).start()
    logger.info("Startup complete (cache warmup running in background).")

if __name__ == "__main__":
    logger.info(f"Running app... (environment={'production' if IS_PRODUCTION else 'development'})")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=not IS_PRODUCTION)
