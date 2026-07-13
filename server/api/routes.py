import json

from fastapi import APIRouter, UploadFile, File, Form, Request, Response

from config.settings import MODEL_OPTIONS, IS_PRODUCTION
from core.vector_database import (
    get_collections_count,
    find_similar_chunks,
    get_document_page,
    upsert_vectorstore_from_pdfs,
    load_vectorstore
)
from core.llm_chain_factory import build_llm_chain, run_agent_query
from core import auth, auth_db
from core.rate_limit import limiter
from api.schemas import SearchQueryRequest, ChatRequest, StandardAPIResponse, SignupRequest, LoginRequest
from utils.logger import logger

router = APIRouter()

SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


@router.get("/health", response_model=StandardAPIResponse)
def health_check():
    logger.debug("Health check requested")
    return StandardAPIResponse(
        status="success",
        data="ok",
        message="Service is healthy"
    )

@router.get("/llm", response_model=StandardAPIResponse)
async def get_llm_options():
    logger.debug("Fetching LLM providers.")
    return StandardAPIResponse(
        status="success",
        data=[provider.title() for provider in MODEL_OPTIONS.keys()]
    )

@router.get("/llm/{model_provider}", response_model=StandardAPIResponse)
async def get_llm_models(model_provider: str):
    model_provider = model_provider.lower()
    if model_provider not in MODEL_OPTIONS:
        logger.warning(f"Invalid model provider: {model_provider}")
        return StandardAPIResponse(status="error", message="Invalid model provider.")

    logger.debug(f"Fetching models for provider: {model_provider}")
    return StandardAPIResponse(
        status="success",
        data=MODEL_OPTIONS[model_provider]["models"]
    )

@router.post("/upload_and_process_pdfs", response_model=StandardAPIResponse)
@limiter.limit("5/minute")
async def upload_and_process_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
    model_provider: str = Form(...)
):
    try:
        model_provider = model_provider.lower()
        logger.info(f"Received {len(files)} files for model provider: {model_provider}")
        await upsert_vectorstore_from_pdfs(files, model_provider)
        logger.info("Files processed successfully")
        return StandardAPIResponse(status="success", data="PDFs processed successfully.")
    except Exception as e:
        logger.exception("Error while uploading and processing files")
        return StandardAPIResponse(status="error", message=str(e))

@router.get("/vector_store/count/{model_provider}", response_model=StandardAPIResponse)
async def get_vectorstore_count(model_provider: str):
    try:
        model_provider = model_provider.lower()
        logger.info(f"Getting collection count for provider: {model_provider}")
        count = get_collections_count(model_provider)
        return StandardAPIResponse(status="success", data=count)
    except Exception as e:
        logger.exception("Error getting collection count")
        return StandardAPIResponse(status="error", message=str(e))

@router.post("/vector_store/search", response_model=StandardAPIResponse)
@limiter.limit("30/minute")
async def get_vectorstore_search(request: Request, payload: SearchQueryRequest):
    try:
        model_provider = payload.model_provider.lower()
        logger.info(f"Search requested with query: {payload.query} for provider: {payload.model_provider}")
        results = find_similar_chunks(model_provider, payload.query)
        return StandardAPIResponse(status="success", data=results)
    except Exception as e:
        logger.exception("Error during similarity search")
        return StandardAPIResponse(status="error", message=str(e))

@router.get("/vector_store/document/{model_provider}", response_model=StandardAPIResponse)
async def get_vectorstore_document_page(model_provider: str, source: str, page: int = 0):
    try:
        model_provider = model_provider.lower()
        logger.info(f"Fetching document page for source: {source}, page: {page}")
        result = get_document_page(model_provider, source, page)
        return StandardAPIResponse(status="success", data=result)
    except Exception as e:
        logger.exception("Error fetching document page")
        return StandardAPIResponse(status="error", message=str(e))


@router.post("/auth/signup", response_model=StandardAPIResponse)
@limiter.limit("5/minute")
async def signup(request: Request, payload: SignupRequest, response: Response):
    try:
        result = auth.signup(payload.username, payload.password)
        response.set_cookie(
            auth.SESSION_COOKIE_NAME, result["token"],
            httponly=True, samesite="lax", max_age=SESSION_COOKIE_MAX_AGE,
            secure=IS_PRODUCTION
        )
        logger.info(f"New account created: {result['username']}")
        return StandardAPIResponse(status="success", data={"username": result["username"]})
    except ValueError as e:
        return StandardAPIResponse(status="error", message=str(e))
    except Exception as e:
        logger.exception("Signup failed")
        return StandardAPIResponse(status="error", message=str(e))

@router.post("/auth/login", response_model=StandardAPIResponse)
@limiter.limit("5/minute")
async def login(request: Request, payload: LoginRequest, response: Response):
    try:
        result = auth.login(payload.username, payload.password)
        response.set_cookie(
            auth.SESSION_COOKIE_NAME, result["token"],
            httponly=True, samesite="lax", max_age=SESSION_COOKIE_MAX_AGE,
            secure=IS_PRODUCTION
        )
        logger.info(f"User logged in: {result['username']}")
        return StandardAPIResponse(status="success", data={"username": result["username"]})
    except ValueError as e:
        return StandardAPIResponse(status="error", message=str(e))
    except Exception as e:
        logger.exception("Login failed")
        return StandardAPIResponse(status="error", message=str(e))

@router.post("/auth/logout", response_model=StandardAPIResponse)
async def logout(request: Request, response: Response):
    token = request.cookies.get(auth.SESSION_COOKIE_NAME)
    auth.logout(token)
    response.delete_cookie(auth.SESSION_COOKIE_NAME)
    return StandardAPIResponse(status="success", data="Logged out.")

@router.get("/auth/me", response_model=StandardAPIResponse)
async def get_me(request: Request):
    user = auth.get_current_user(request)
    return StandardAPIResponse(status="success", data={"username": user["username"] if user else None})


@router.get("/chat_sessions", response_model=StandardAPIResponse)
async def get_chat_sessions(request: Request):
    user = auth.get_current_user(request)
    if not user:
        return StandardAPIResponse(status="error", message="Not authenticated.")
    sessions = auth_db.list_chat_sessions(user["user_id"])
    return StandardAPIResponse(status="success", data=sessions)

@router.get("/chat_sessions/{session_id}/messages", response_model=StandardAPIResponse)
async def get_chat_session_messages(session_id: str, request: Request):
    user = auth.get_current_user(request)
    if not user:
        return StandardAPIResponse(status="error", message="Not authenticated.")
    session = auth_db.get_chat_session(session_id)
    if not session or session["user_id"] != user["user_id"]:
        return StandardAPIResponse(status="error", message="Chat session not found.")
    messages = auth_db.list_chat_messages(session_id)
    for m in messages:
        m["trace"] = json.loads(m.pop("trace_json")) if m.get("trace_json") else None
    return StandardAPIResponse(status="success", data=messages)

@router.delete("/chat_sessions/{session_id}", response_model=StandardAPIResponse)
async def delete_chat_session_route(session_id: str, request: Request):
    user = auth.get_current_user(request)
    if not user:
        return StandardAPIResponse(status="error", message="Not authenticated.")
    session = auth_db.get_chat_session(session_id)
    if not session or session["user_id"] != user["user_id"]:
        return StandardAPIResponse(status="error", message="Chat session not found.")
    auth_db.delete_chat_session(session_id)
    return StandardAPIResponse(status="success", data="Deleted.")


@router.post("/chat", response_model=StandardAPIResponse)
@limiter.limit("20/minute")
async def chat(request: Request, payload: ChatRequest):
    try:
        message = payload.message
        model_name = payload.model_name
        model_provider = payload.model_provider.lower()
        logger.debug(f"Chat request for model: {payload.model_name} (provider: {payload.model_provider})")

        if model_provider not in MODEL_OPTIONS:
            logger.warning("Invalid model provider.")
            return StandardAPIResponse(status="error", message="Invalid model provider.")
        if model_name not in MODEL_OPTIONS[model_provider]["models"]:
            logger.warning("Invalid model name.")
            return StandardAPIResponse(status="error", message="Invalid model name.")

        vectorstore = load_vectorstore(model_provider)
        answer, trace = run_agent_query(model_provider, model_name, vectorstore, message)

        logger.debug(f"Chat response generated. Used live fetch: {trace['used_live_fetch']}")

        session_id = None
        user = auth.get_current_user(request)
        if user:
            session_id = payload.session_id
            if not session_id or not auth_db.get_chat_session(session_id):
                title = message[:50] + ("..." if len(message) > 50 else "")
                session_id = auth_db.create_chat_session(user["user_id"], title)
            auth_db.add_chat_message(session_id, "user", message)
            auth_db.add_chat_message(session_id, "ai", answer, json.dumps(trace))

        data = {"answer": answer, "trace": trace}
        if session_id:
            data["session_id"] = session_id
        return StandardAPIResponse(status="success", data=data)
    except Exception as e:
        logger.exception("Chat endpoint encountered an error")
        return StandardAPIResponse(status="error", message=str(e))
