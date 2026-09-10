"""Demo accounts and the role-tagged session token. Spec, Component 5.

    authenticate(username, password) -> role, or None
    create_token(role)               -> signed JWT carrying the role and an expiry
    decode_token(token)              -> role, or ValueError

The point of signing is that /chat can trust the role it reads. A client holding a nurse
token cannot edit it into an admin one, because the signature would no longer verify.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import time

import jwt

# Signing key. Taken from the environment when set; otherwise a fresh random key per
# process, which is right for a demo and keeps a real key out of the repository. Tokens
# do not survive a restart in that mode, and the frontend simply logs in again.
_SECRET = os.getenv("MEDIBOT_JWT_SECRET") or secrets.token_urlsafe(32)
_ALGORITHM = "HS256"
# One hour. Short because there is no refresh token here: whatever this number says is
# how long a stolen token stays useful. A real deployment would pair a short access token
# with a refresh token so the session outlives the credential; until then, re-login is
# the cheaper cost. Override for a long unattended demo.
TOKEN_TTL_SECONDS = int(os.getenv("MEDIBOT_TOKEN_TTL_SECONDS", 60 * 60))


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# The five accounts the spec names (line 183), one per role. Usernames and roles come
# from the spec; the passwords are ours and are documented in the README, because a
# grader has to be able to log in. Stored as SHA-256 so the file holds no plaintext.
# A real system would use a slow hash such as bcrypt or argon2: SHA-256 is fast, which
# is exactly what makes it a poor choice against an offline guessing attack.
DEMO_USERS: dict[str, tuple[str, str]] = {
    "dr.mehta": (_hash("dr.mehta-demo"), "doctor"),
    "nurse.priya": (_hash("nurse.priya-demo"), "nurse"),
    "billing.ravi": (_hash("billing.ravi-demo"), "billing_executive"),
    "tech.anand": (_hash("tech.anand-demo"), "technician"),
    "admin.sys": (_hash("admin.sys-demo"), "admin"),
}


def authenticate(username: str, password: str) -> str | None:
    """The user's role when the credentials match, otherwise None.

    compare_digest rather than ==, so the time taken does not reveal how much of the
    hash was correct.
    """
    record = DEMO_USERS.get(username)
    if record is None:
        return None
    expected, role = record
    return role if secrets.compare_digest(expected, _hash(password)) else None


def create_token(role: str, issued_at: float | None = None) -> str:
    """Signed token carrying the role. issued_at is a seam for testing expiry."""
    now = issued_at if issued_at is not None else time.time()
    return jwt.encode(
        {"role": role, "iat": int(now), "exp": int(now + TOKEN_TTL_SECONDS)},
        _SECRET,
        algorithm=_ALGORITHM,
    )


def decode_token(token: str) -> str:
    """The role inside a valid token. ValueError for anything else, so the caller has
    one exception to catch rather than three PyJWT subclasses."""
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError("invalid token") from exc
    role = payload.get("role")
    if not role:
        raise ValueError("invalid token: no role")
    return role
