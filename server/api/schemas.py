from pydantic import BaseModel
from typing import Any, Optional, Literal


class SearchQueryRequest(BaseModel):
    model_provider: str
    query: str

class ChatRequest(BaseModel):
    model_provider: str
    model_name: str
    message: str
    session_id: Optional[str] = None

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class StandardAPIResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[Any] = None
    message: Optional[str] = None
