"""The FastAPI application. Spec, Component 5.

    GET  /health              is the server up
    GET  /collections/{role}  what this role may read
    POST /login               credentials in, role-tagged token out
    POST /chat                a question in, an Answer out

The role /chat acts on comes out of the signed token, never out of the request body. The app is a thin shell: every
decision about retrieval and access lives in medibot.retrieval, so the endpoints stay
small enough to read.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated

import groq
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from medibot.api.auth import authenticate, create_token, decode_token
from medibot.config import ROLE_COLLECTIONS
from medibot.retrieval.chat import chat

app = FastAPI(
    title="MediBot",
    description="Role-aware assistant over MediAssist's documents and operational database.",
    version="0.1.0",
)


# The frontend is served from a different port, which makes every call it sends a
# cross-origin request that the browser blocks unless we say otherwise. An explicit list
# rather than "*": nothing here needs to be callable from an arbitrary page, and the
# tokens travel in an Authorization header, so widening this would buy nothing.
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = os.getenv("MEDIBOT_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(groq.RateLimitError)
def rate_limited(request: Request, error: groq.RateLimitError) -> JSONResponse:
    """The model provider caps tokens per minute and per day. Left unhandled this
    surfaced as a 500 with a stack trace in the log and a vague message to the user.
    503 is the honest status, and the message says what to do rather than what broke;
    the provider's own text is not passed through, since it names the organisation."""
    logger.warning("groq rate limit on %s", request.url.path)
    return JSONResponse(
        status_code=503,
        content={"detail": "MediBot is busy right now. Wait a few seconds and ask again."},
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/collections/{role}")
def collections(role: str) -> dict[str, object]:
    """Straight out of the role matrix in config; no retrieval happens here.

    The frontend shows this beside the role badge so a user can see what they may ask
    about before they ask, and so a refusal is never a surprise.
    """
    if role not in ROLE_COLLECTIONS:
        raise HTTPException(status_code=404, detail=f"unknown role: {role}")
    return {"role": role, "collections": ROLE_COLLECTIONS[role]}


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str


@app.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    """Credentials in, role-tagged token out. Spec, Component 5.

    The role is returned alongside the token so the frontend can show the badge without
    decoding a JWT, but it is not what /chat trusts: that reads the role out of the
    signed token, so editing this field client-side buys nothing.

    One message for both failures. Saying "no such user" for one and "wrong password"
    for the other would turn this endpoint into a way to discover valid usernames.
    """
    role = authenticate(request.username, request.password)
    if role is None:
        raise HTTPException(status_code=401, detail="incorrect username or password")
    return LoginResponse(token=create_token(role), role=role)


def current_role(authorization: Annotated[str | None, Header()] = None) -> str:
    """The role from the Authorization header, or 401.

    Everything downstream trusts this value: it selects the RBAC filter inside the
    Qdrant query and decides whether SQL RAG runs at all. It is read from the signature
    and from nowhere else, so no field a client can set has any influence on it.
    """
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="missing bearer token")
    try:
        return decode_token(token.strip())
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


class ChatRequest(BaseModel):
    # A ceiling as well as a floor. Without one, a 50,000-character body was accepted and
    # forwarded to the model, which is somebody else's bill and a denial-of-service shape.
    question: str = Field(min_length=1, max_length=1000)
    # No role field. One appearing in the body is ignored, which a test pins.


class Source(BaseModel):
    source_document: str
    section_title: str
    collection: str


class ChatResponse(BaseModel):
    """Spec, Component 5. The four fields it names, plus the SQL when there was any."""

    answer: str
    sources: list[Source]
    retrieval_type: str
    role: str
    sql: str | None = None
    refusal: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, role: Annotated[str, Depends(current_role)]) -> ChatResponse:
    """Hands the question to the retrieval layer and returns what comes back.

    Deliberately thin. The routing, the role gate, the RBAC filter and the score gate all
    live in medibot.retrieval and are covered by their own tests; duplicating any of that
    logic here would give it a second place to drift.
    """
    result = chat(request.question, role)
    return ChatResponse(
        answer=result.answer,
        sources=[Source(**source) for source in result.sources],
        retrieval_type=result.retrieval_type,
        role=result.role,
        sql=result.sql,
        refusal=result.refusal,
    )
