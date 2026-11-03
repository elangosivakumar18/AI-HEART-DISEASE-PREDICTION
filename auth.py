import secrets
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

import database

bearer = HTTPBearer(auto_error=False)


class AuthUser:
    def __init__(self, id: str, email: str, full_name: str, is_anonymous: bool = False):
        self.id = id
        self.email = email
        self.full_name = full_name
        self.is_anonymous = is_anonymous

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "is_anonymous": self.is_anonymous,
        }


class GuestUser:
    def __init__(self):
        self.id = "guest"
        self.email = "guest@cardio.ai"
        self.full_name = "Guest User"
        self.is_anonymous = True

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "is_anonymous": True,
        }


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> AuthUser:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please log in.",
        )

    user_dict = database.get_user_by_session_token(credentials.credentials)
    if not user_dict:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session. Please log in again.",
        )

    return AuthUser(
        id=user_dict["id"],
        email=user_dict["email"],
        full_name=user_dict["full_name"],
        is_anonymous=user_dict["is_anonymous"],
    )


def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> AuthUser | GuestUser:
    if credentials is None or not credentials.credentials:
        return GuestUser()

    user_dict = database.get_user_by_session_token(credentials.credentials)
    if not user_dict:
        return GuestUser()

    return AuthUser(
        id=user_dict["id"],
        email=user_dict["email"],
        full_name=user_dict["full_name"],
        is_anonymous=user_dict["is_anonymous"],
    )


# --------------------------------------------------
# Auth Helper Functions
# --------------------------------------------------

def register_user(email: str, password: str, full_name: str):
    email_clean = email.strip().lower()
    if not email_clean or "@" not in email_clean:
        raise ValueError("Invalid email format.")

    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    existing = database.get_user_by_email(email_clean)
    if existing:
        raise ValueError("An account with this email already exists.")

    user = database.create_user(
        email=email_clean,
        password=password,
        full_name=full_name or "User",
        is_anonymous=False
    )
    return user


def login_user(email: str, password: str):
    email_clean = email.strip().lower()
    user = database.get_user_by_email(email_clean)
    if not user:
        raise ValueError("Invalid email or password.")

    if not database.verify_password(password, user["password_hash"], user["salt"]):
        raise ValueError("Invalid email or password.")

    token = database.create_session(user["id"])
    return {
        "access_token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "full_name": user["full_name"],
            "is_anonymous": bool(user["is_anonymous"]),
        }
    }


def create_guest_session():
    # Create or reuse a guest user
    guest_email = f"guest_{secrets.token_hex(4)}@cardio.ai"
    user = database.create_user(
        email=guest_email,
        password=secrets.token_urlsafe(16),
        full_name="Guest User",
        is_anonymous=True
    )
    token = database.create_session(user["id"])
    return {
        "access_token": token,
        "user": user
    }


def logout_session(token: str):
    if token:
        database.delete_session(token)
