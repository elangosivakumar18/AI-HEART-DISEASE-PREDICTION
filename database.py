import json
import os
import sqlite3
import secrets
import hashlib
import hmac
from datetime import datetime, timezone, timedelta
from pathlib import Path
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "cardio_ai.db"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                is_anonymous INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                patient_data TEXT NOT NULL,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON prediction_history(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_created_at ON prediction_history(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_user_id ON chat_history(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_created_at ON chat_history(created_at);")
        conn.commit()


# Initialize database schema on module load
init_db()





# --------------------------------------------------
# Password Security (PBKDF2-HMAC-SHA256)
# --------------------------------------------------

def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    ).hex()
    return key, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    key, _ = hash_password(password, salt)
    return hmac.compare_digest(key, stored_hash)


# --------------------------------------------------
# User & Session Management
# --------------------------------------------------

def create_user(email: str, password: str, full_name: str, is_anonymous: bool = False):
    email_clean = email.strip().lower()
    user_id = secrets.token_hex(16)
    pwd_hash, salt = hash_password(password)

    with get_db() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO users (id, email, full_name, password_hash, salt, is_anonymous)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (user_id, email_clean, full_name.strip(), pwd_hash, salt, 1 if is_anonymous else 0)
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("An account with this email already exists.") from exc

    return {
        "id": user_id,
        "email": email_clean,
        "full_name": full_name.strip(),
        "is_anonymous": is_anonymous
    }


def get_user_by_email(email: str):
    email_clean = email.strip().lower()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, email, full_name, is_anonymous, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None




def create_session(user_id: str, days_valid: int = 30):
    token = secrets.token_urlsafe(32)
    expires_at = utc_now() + timedelta(days=days_valid)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires_at.isoformat())
        )
        conn.commit()

    return token


def get_user_by_session_token(token: str):
    if not token:
        return None

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT u.id, u.email, u.full_name, u.is_anonymous, s.expires_at
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.token = ?
            """,
            (token,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Check expiry
        if row["expires_at"]:
            try:
                raw_exp = row["expires_at"]
                exp = datetime.fromisoformat(raw_exp)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if utc_now() > exp:
                    cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
                    conn.commit()
                    return None
            except Exception:
                pass

        return {
            "id": row["id"],
            "email": row["email"],
            "full_name": row["full_name"],
            "is_anonymous": bool(row["is_anonymous"])
        }


def delete_session(token: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# --------------------------------------------------
# Prediction History
# --------------------------------------------------

def save_prediction(user_id: str, patient_data: dict, prediction: dict):
    record_id = secrets.token_hex(12)
    now = utc_now().isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history (id, user_id, patient_data, prediction, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                str(user_id),
                json.dumps(patient_data),
                json.dumps(prediction, default=str),
                now
            )
        )
        conn.commit()

    return {
        "id": record_id,
        "user_id": str(user_id),
        "patient_data": patient_data,
        "prediction": prediction,
        "created_at": now
    }


def get_user_history(user_id: str, limit: int = 20, offset: int = 0):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, patient_data, prediction, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (str(user_id), limit, offset)
        )
        rows = cursor.fetchall()

    results = []
    for r in rows:
        try:
            p_data = json.loads(r["patient_data"]) if isinstance(r["patient_data"], str) else r["patient_data"]
            pred = json.loads(r["prediction"]) if isinstance(r["prediction"], str) else r["prediction"]
        except Exception:
            p_data = r["patient_data"]
            pred = r["prediction"]

        results.append({
            "id": r["id"],
            "patient_data": p_data,
            "prediction": pred,
            "created_at": r["created_at"]
        })

    return results


def clear_user_history(user_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_history WHERE user_id = ?", (str(user_id),))
        conn.commit()
    return True


# --------------------------------------------------
# Chat History
# --------------------------------------------------

def save_chat_message(user_id: str, role: str, text: str):
    record_id = secrets.token_hex(12)
    now = utc_now().isoformat()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO chat_history (id, user_id, role, text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record_id, str(user_id), role, text, now)
        )
        conn.commit()
    return {
        "id": record_id,
        "user_id": str(user_id),
        "role": role,
        "text": text,
        "created_at": now
    }


def get_chat_history(user_id: str, limit: int = 100):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, role, text, created_at
            FROM chat_history
            WHERE user_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (str(user_id), limit)
        )
        rows = cursor.fetchall()

    return [{
        "id": r["id"],
        "role": r["role"],
        "text": r["text"],
        "created_at": r["created_at"]
    } for r in rows]


def clear_chat_history(user_id: str):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE user_id = ?", (str(user_id),))
        conn.commit()
    return True
