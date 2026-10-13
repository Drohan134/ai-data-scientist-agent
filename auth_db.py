"""
auth_db.py — Supabase REST backend (direct HTTP, no supabase-py client quirks)
All auth operations call the Supabase PostgREST API directly via requests.
Falls back to local SQLite if Supabase is not configured.
"""

import os
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Tuple

from dotenv import load_dotenv

_DOTENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_DOTENV, override=True)

try:
    import bcrypt
    _HAS_BCRYPT = True
except ImportError:
    _HAS_BCRYPT = False

def _get_setting(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is not None and val != "":
        return str(val).strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


_raw_sb_url = _get_setting("SUPABASE_URL").rstrip("/")
if _raw_sb_url.endswith("/rest/v1"):
    _raw_sb_url = _raw_sb_url[:-8].rstrip("/")
SUPABASE_URL         = _raw_sb_url
SUPABASE_SERVICE_KEY = _get_setting("SUPABASE_SERVICE_KEY")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.db")


# ─────────────────────────────────────────────────────────────────────────────
# Backend selector
# ─────────────────────────────────────────────────────────────────────────────

def _is_supabase() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Supabase REST helpers (direct requests — no supabase-py client)
# ─────────────────────────────────────────────────────────────────────────────

def _headers(prefer: str = "") -> dict:
    h = {
        "apikey":        SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _rest(table: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{table}"


def _sb_select(table: str, filters: dict = None, columns: str = "*", limit: int = None) -> list:
    """SELECT from a Supabase table. filters = {col: val} (all AND, eq only)."""
    import requests
    params = {"select": columns}
    for col, val in (filters or {}).items():
        params[col] = f"eq.{val}"
    if limit:
        params["limit"] = limit
    r = requests.get(_rest(table), headers=_headers(), params=params, timeout=10)
    if r.status_code >= 400:
        raise Exception(r.json())
    return r.json() if isinstance(r.json(), list) else []


def _sb_insert(table: str, data: dict) -> dict:
    """INSERT one row and return the inserted record."""
    import requests
    r = requests.post(
        _rest(table), headers=_headers("return=representation"),
        json=data, timeout=10,
    )
    if r.status_code >= 400:
        raise Exception(r.json())
    result = r.json()
    return result[0] if isinstance(result, list) and result else data


def _sb_update(table: str, filters: dict, data: dict) -> None:
    """UPDATE rows matching filters."""
    import requests
    params = {}
    for col, val in filters.items():
        params[col] = f"eq.{val}"
    r = requests.patch(
        _rest(table), headers=_headers("return=minimal"),
        params=params, json=data, timeout=10,
    )
    if r.status_code >= 400:
        raise Exception(r.json())


def _sb_count(table: str, filters: dict) -> int:
    """Return count of rows matching filters."""
    import requests
    params = {"select": "id"}
    for col, val in filters.items():
        params[col] = f"eq.{val}"
    r = requests.get(_rest(table), headers=_headers("count=exact"), params=params, timeout=10)
    if r.status_code >= 400:
        return 0
    data = r.json()
    return len(data) if isinstance(data, list) else 0


def _sb_select_or(table: str, or_filters: list, columns: str = "*") -> list:
    """
    SELECT with OR filters: or_filters = [("col", "val"), ...]
    Uses PostgREST's 'or' query parameter.
    """
    import requests
    or_parts = ",".join(f"{col}.eq.{val}" for col, val in or_filters)
    params = {"select": columns, "or": f"({or_parts})"}
    r = requests.get(_rest(table), headers=_headers(), params=params, timeout=10)
    if r.status_code >= 400:
        raise Exception(r.json())
    return r.json() if isinstance(r.json(), list) else []


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> Tuple[str, str]:
    salt = secrets.token_hex(16)
    if _HAS_BCRYPT:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        return hashed, salt
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return hashed, salt


def verify_password(password: str, stored_hash: str, salt: str = "") -> bool:
    if not stored_hash:
        return False
    if _HAS_BCRYPT and stored_hash.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
        except Exception:
            return False
    if salt:
        test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
        if secrets.compare_digest(test, stored_hash):
            return True
    test_sha = hashlib.sha256((password + salt).encode()).hexdigest()
    return secrets.compare_digest(test_sha, stored_hash)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _future(minutes: int = 15) -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    if _is_supabase():
        _ensure_admin_supabase()
    else:
        _init_sqlite()


def _ensure_admin_supabase():
    try:
        rows = _sb_select("users", {"username": "admin"}, columns="id")
        if not rows:
            pwd_hash, salt = hash_password("AdminPassword123!")
            _sb_insert("users", {
                "username": "admin", "email": "admin@aidatascientist.com",
                "password_hash": pwd_hash, "salt": salt,
                "auth_provider": "local", "full_name": "System Administrator", "role": "admin",
            })
            print("[AUTH_DB] Admin seeded in Supabase.", flush=True)
        else:
            print("[AUTH_DB] Connected to Supabase OK.", flush=True)
    except Exception as e:
        print(f"[AUTH_DB] Supabase init warning: {e}", flush=True)


def _init_sqlite():
    import sqlite3
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
                password_hash TEXT, salt TEXT,
                auth_provider TEXT DEFAULT 'local',
                google_id TEXT, full_name TEXT, avatar_url TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_login TEXT, is_logged_in INTEGER DEFAULT 0
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                session_token TEXT UNIQUE NOT NULL,
                user_id TEXT NOT NULL, username TEXT NOT NULL,
                ip_address TEXT, user_agent TEXT,
                login_time TEXT NOT NULL DEFAULT (datetime('now')),
                last_active TEXT NOT NULL DEFAULT (datetime('now')),
                is_active INTEGER DEFAULT 1
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
                user_id TEXT NOT NULL, email TEXT NOT NULL,
                reset_code TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, used INTEGER DEFAULT 0
            )""")
        if not conn.execute("SELECT 1 FROM users WHERE username='admin'").fetchone():
            pwd_hash, salt = hash_password("AdminPassword123!")
            conn.execute("""
                INSERT INTO users (id,username,email,password_hash,salt,auth_provider,full_name,role)
                VALUES (?,?,?,?,?,'local','System Administrator','admin')
            """, (secrets.token_hex(16), "admin", "admin@aidatascientist.com", pwd_hash, salt))
    conn.close()
    print("[AUTH_DB] SQLite fallback initialised.", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_user(username: str, email: str, password: str,
                  full_name: str = "", role: str = "user") -> Tuple[bool, str, Optional[Dict]]:
    username  = username.strip()
    email     = email.strip().lower()
    full_name = full_name.strip() or username
    if len(username) < 3:
        return False, "Username must be at least 3 characters.", None
    if "@" not in email:
        return False, "Please provide a valid email address.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters.", None
    if _is_supabase():
        return _sb_register(username, email, password, full_name, role)
    return _sqlite_register(username, email, password, full_name, role)


def authenticate_user(username_or_email: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    query = username_or_email.strip()
    if not query or not password:
        return False, "Please enter your username/email and password.", None
    if _is_supabase():
        return _sb_auth_user(query, password)
    return _sqlite_auth_user(query, password)


def authenticate_google_user(email: str, full_name: str = "",
                             google_id: str = "", avatar_url: str = "") -> Tuple[bool, str, Dict]:
    email = email.strip().lower()
    if not email:
        return False, "Invalid Google account email.", {}
    if _is_supabase():
        return _sb_google(email, full_name, google_id, avatar_url)
    return _sqlite_google(email, full_name, google_id, avatar_url)


def start_user_session(user_id, username: str,
                       ip_address: str = "127.0.0.1",
                       user_agent: str = "Streamlit") -> str:
    token = secrets.token_urlsafe(32)
    if _is_supabase():
        try:
            _sb_insert("sessions", {
                "session_token": token, "user_id": str(user_id),
                "username": username, "ip_address": ip_address, "user_agent": user_agent,
            })
            _sb_update("users", {"id": str(user_id)}, {"is_logged_in": True, "last_login": _now()})
        except Exception as e:
            print(f"[AUTH_DB] start_session error: {e}")
    return token


def heartbeat_user_session(session_token: str):
    if not session_token or not _is_supabase():
        return
    try:
        _sb_update("sessions", {"session_token": session_token, "is_active": "true"},
                   {"last_active": _now()})
    except Exception:
        pass


def end_user_session(session_token: str):
    if not session_token:
        return
    if _is_supabase():
        try:
            rows = _sb_select("sessions", {"session_token": session_token}, columns="user_id")
            if rows:
                uid = rows[0]["user_id"]
                _sb_update("sessions", {"session_token": session_token}, {"is_active": False})
                active = _sb_count("sessions", {"user_id": uid, "is_active": "true"})
                if active == 0:
                    _sb_update("users", {"id": uid}, {"is_logged_in": False})
        except Exception as e:
            print(f"[AUTH_DB] end_session error: {e}")


def create_password_reset_link(email_or_username: str, base_url: str = None) -> Tuple[bool, str, Optional[Dict]]:
    query = email_or_username.strip()
    if not query:
        return False, "Please enter your email or username.", None
    user = _get_user_by_identity(query)
    if not user:
        return False, f"No account found for '{query}'.", None

    token = secrets.token_urlsafe(32)
    expires_at = _future(60)  # 60 minute expiry for link

    if _is_supabase():
        try:
            _sb_insert("password_resets", {
                "user_id": str(user["id"]),
                "email": user["email"],
                "reset_code": token,
                "expires_at": expires_at,
                "used": False,
            })
        except Exception as e:
            return False, f"Could not generate reset link: {e}", None
    else:
        _sqlite_insert_reset(user["id"], user["email"], token, expires_at)

    app_base = (base_url or os.getenv("APP_BASE_URL", "http://localhost:8501")).strip().rstrip("/")
    reset_link = f"{app_base}/?reset_token={token}"

    smtp_ok, smtp_msg = _dispatch_reset_email(
        to_email=user["email"],
        username=user.get("full_name") or user.get("username", "User"),
        reset_link=reset_link,
    )

    return True, reset_link, {
        "user_id": user["id"],
        "email": user["email"],
        "username": user["username"],
        "reset_link": reset_link,
        "token": token,
        "smtp_sent": smtp_ok,
        "smtp_msg": smtp_msg,
    }


def verify_reset_token(token: str) -> Tuple[bool, str, Optional[Dict]]:
    if not token or not token.strip():
        return False, "No reset token provided.", None
    clean_token = token.strip()
    if _is_supabase():
        row = _sb_verify_reset_token(clean_token)
    else:
        row = _sqlite_verify_reset_token(clean_token)
    if not row:
        return False, "This password reset link is invalid, has expired, or has already been used.", None
    return True, "Valid token", row


def reset_password_with_token(token: str, new_password: str) -> Tuple[bool, str]:
    if not token or not new_password:
        return False, "Please fill in all fields."
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    clean_token = token.strip()
    if _is_supabase():
        return _sb_reset_pw_token(clean_token, new_password)
    return _sqlite_reset_pw_token(clean_token, new_password)


def create_password_reset(email_or_username: str) -> Tuple[bool, str, Optional[Dict]]:
    """Backward compatible wrapper returning reset link and token."""
    ok, link_or_err, info = create_password_reset_link(email_or_username)
    if ok and info:
        return True, info["token"], info
    return False, link_or_err, None


def reset_password_with_code(email_or_username: str, code: str,
                             new_password: str) -> Tuple[bool, str]:
    return reset_password_with_token(code, new_password)


def get_all_users() -> List[Dict]:
    if _is_supabase():
        try:
            return _sb_select("users", columns=(
                "id,username,email,auth_provider,full_name,avatar_url,"
                "role,created_at,last_login,is_logged_in"
            ))
        except Exception as e:
            print(f"[AUTH_DB] get_all_users: {e}")
            return []
    return _sqlite_all_users()


def get_active_sessions() -> List[Dict]:
    if _is_supabase():
        try:
            return _sb_select("sessions", {"is_active": "true"})
        except Exception as e:
            print(f"[AUTH_DB] get_active_sessions: {e}")
            return []
    return _sqlite_active_sessions()


def force_logout_user(user_id):
    if _is_supabase():
        try:
            _sb_update("sessions", {"user_id": str(user_id)}, {"is_active": False})
            _sb_update("users", {"id": str(user_id)}, {"is_logged_in": False})
        except Exception as e:
            print(f"[AUTH_DB] force_logout: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Supabase implementations
# ─────────────────────────────────────────────────────────────────────────────

def _get_user_by_identity(query: str) -> Optional[Dict]:
    """Fetch user by username or email (tries both)."""
    if _is_supabase():
        try:
            # Try email first
            rows = _sb_select("users", {"email": query.lower()})
            if rows:
                return rows[0]
            # Try username
            rows = _sb_select("users", {"username": query})
            return rows[0] if rows else None
        except Exception as e:
            print(f"[AUTH_DB] _get_user_by_identity error: {e}")
            return None
    return _sqlite_get_user_by_identity(query)


def _sb_register(username, email, password, full_name, role):
    try:
        if _sb_select("users", {"username": username}, columns="id"):
            return False, f"Username '{username}' is already taken.", None
        if _sb_select("users", {"email": email}, columns="id"):
            return False, f"Email '{email}' is already registered.", None
        pwd_hash, salt = hash_password(password)
        row = _sb_insert("users", {
            "username": username, "email": email,
            "password_hash": pwd_hash, "salt": salt,
            "auth_provider": "local", "full_name": full_name, "role": role,
        })
        row.pop("password_hash", None); row.pop("salt", None)
        return True, "Registered!", row
    except Exception as e:
        return False, f"Registration error: {e}", None


def _sb_auth_user(query, password):
    try:
        row = None
        rows = _sb_select("users", {"email": query.lower()})
        if rows:
            row = rows[0]
        else:
            rows = _sb_select("users", {"username": query})
            if rows:
                row = rows[0]
        if not row:
            return False, "Invalid username or password.", None
        if not verify_password(password, row.get("password_hash", ""), row.get("salt", "")):
            return False, "Invalid username or password.", None
        row.pop("password_hash", None); row.pop("salt", None)
        return True, "Login successful!", row
    except Exception as e:
        return False, f"Auth error: {e}", None


def _sb_google(email, full_name, google_id, avatar_url):
    try:
        rows = _sb_select("users", {"email": email})
        if rows:
            row = rows[0]
            _sb_update("users", {"id": row["id"]}, {
                "auth_provider": "google",
                "google_id":     google_id or row.get("google_id") or "",
                "avatar_url":    avatar_url or row.get("avatar_url") or "",
                "full_name":     full_name  or row.get("full_name") or "",
                "is_logged_in":  True,
                "last_login":    _now(),
            })
            row["auth_provider"] = "google"
            row["avatar_url"]    = avatar_url or row.get("avatar_url", "")
            row["full_name"]     = full_name  or row.get("full_name", "")
        else:
            base = email.split("@")[0].replace(".", "_").replace("+", "_")
            if _sb_select("users", {"username": base}, columns="id"):
                base += f"_{secrets.token_hex(2)}"
            row = _sb_insert("users", {
                "username":      base,
                "email":         email,
                "auth_provider": "google",
                "google_id":     google_id,
                "full_name":     full_name or base,
                "avatar_url":    avatar_url or "",
                "role":          "user",
                "is_logged_in":  True,
                "last_login":    _now(),
            })
        row.pop("password_hash", None)
        row.pop("salt", None)
        return True, "Google auth successful!", row
    except Exception as e:
        return False, f"Google auth error: {e}", {}


def _sb_verify_reset_token(token: str) -> Optional[Dict]:
    try:
        import requests as req
        now = _now()
        params = {
            "select": "id,user_id,email,expires_at,used",
            "reset_code": f"eq.{token}",
            "used": "eq.false",
            "expires_at": f"gt.{now}",
            "limit": "1",
            "order": "id.desc",
        }
        r = req.get(_rest("password_resets"), headers=_headers(), params=params, timeout=10)
        if r.status_code == 200 and isinstance(r.json(), list) and r.json():
            return r.json()[0]
        return None
    except Exception as e:
        print(f"[AUTH_DB] _sb_verify_reset_token error: {e}")
        return None


def _sb_reset_pw_token(token: str, new_password: str) -> Tuple[bool, str]:
    try:
        row = _sb_verify_reset_token(token)
        if not row:
            return False, "This password reset link is invalid, has expired, or has already been used."
        _sb_update("password_resets", {"id": row["id"]}, {"used": True})
        pwd_hash, salt = hash_password(new_password)
        _sb_update("users", {"id": str(row["user_id"])}, {"password_hash": pwd_hash, "salt": salt})
        return True, "Password updated successfully!"
    except Exception as e:
        return False, f"Password reset failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# SQLite fallback
# ─────────────────────────────────────────────────────────────────────────────

def _sqlite_conn():
    import sqlite3
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def _sqlite_get_user_by_identity(query):
    c = _sqlite_conn()
    try:
        r = c.execute("SELECT * FROM users WHERE email=? OR username=?",
                      (query.lower(), query)).fetchone()
        return dict(r) if r else None
    finally:
        c.close()

def _sqlite_register(username, email, password, full_name, role):
    c = _sqlite_conn()
    try:
        with c:
            if c.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone():
                return False, f"Username '{username}' is taken.", None
            if c.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
                return False, f"Email '{email}' already registered.", None
            pwd_hash, salt = hash_password(password)
            uid = secrets.token_hex(16)
            c.execute("""
                INSERT INTO users (id,username,email,password_hash,salt,auth_provider,full_name,role)
                VALUES (?,?,?,?,?,'local',?,?)
            """, (uid, username, email, pwd_hash, salt, full_name, role))
            return True, "Registered!", {"id": uid, "username": username, "email": email,
                                         "full_name": full_name, "role": role, "auth_provider": "local"}
    except Exception as e:
        return False, str(e), None
    finally:
        c.close()

def _sqlite_auth_user(query, password):
    c = _sqlite_conn()
    try:
        r = c.execute("SELECT * FROM users WHERE email=? OR username=?",
                      (query.lower(), query)).fetchone()
        if not r or not verify_password(password, r["password_hash"] or "", r["salt"] or ""):
            return False, "Invalid username or password.", None
        d = dict(r); d.pop("password_hash", None); d.pop("salt", None)
        return True, "Login successful!", d
    finally:
        c.close()

def _sqlite_google(email, full_name, google_id, avatar_url):
    c = _sqlite_conn()
    try:
        with c:
            r = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            if r:
                c.execute("""
                    UPDATE users SET auth_provider='google',google_id=?,avatar_url=?,
                    full_name=COALESCE(NULLIF(?,''),full_name),last_login=?,is_logged_in=1
                    WHERE id=?
                """, (google_id, avatar_url, full_name, _now(), r["id"]))
                d = dict(r)
            else:
                base = email.split("@")[0].replace(".", "_")
                if c.execute("SELECT id FROM users WHERE username=?", (base,)).fetchone():
                    base += f"_{secrets.token_hex(2)}"
                uid = secrets.token_hex(16)
                c.execute("""
                    INSERT INTO users (id,username,email,auth_provider,google_id,full_name,
                    avatar_url,role,is_logged_in,last_login)
                    VALUES (?,?,?,'google',?,?,?,'user',1,?)
                """, (uid, base, email, google_id, full_name or base, avatar_url or "", _now()))
                d = {"id": uid, "username": base, "email": email,
                     "full_name": full_name or base, "auth_provider": "google", "role": "user"}
            d.pop("password_hash", None); d.pop("salt", None)
            return True, "Google auth successful!", d
    except Exception as e:
        return False, str(e), {}
    finally:
        c.close()

def _sqlite_insert_reset(user_id, email, code, expires_at):
    c = _sqlite_conn()
    with c:
        c.execute("""
            INSERT INTO password_resets (id,user_id,email,reset_code,created_at,expires_at)
            VALUES (?,?,?,?,?,?)
        """, (secrets.token_hex(16), user_id, email, code, _now(), expires_at))
    c.close()

def _sqlite_verify_reset_token(token: str) -> Optional[Dict]:
    c = _sqlite_conn()
    try:
        r = c.execute("""
            SELECT id, user_id, email, expires_at, used
            FROM password_resets
            WHERE reset_code=? AND used=0 AND expires_at>=?
            ORDER BY id DESC LIMIT 1
        """, (token, _now())).fetchone()
        return dict(r) if r else None
    finally:
        c.close()

def _sqlite_reset_pw_token(token: str, new_password: str) -> Tuple[bool, str]:
    c = _sqlite_conn()
    try:
        with c:
            r = c.execute("""
                SELECT id, user_id FROM password_resets
                WHERE reset_code=? AND used=0 AND expires_at>=?
                ORDER BY id DESC LIMIT 1
            """, (token, _now())).fetchone()
            if not r:
                return False, "This password reset link is invalid, has expired, or has already been used."
            c.execute("UPDATE password_resets SET used=1 WHERE id=?", (r["id"],))
            pwd_hash, salt = hash_password(new_password)
            c.execute("UPDATE users SET password_hash=?,salt=? WHERE id=?", (pwd_hash, salt, r["user_id"]))
        return True, "Password updated successfully!"
    finally:
        c.close()

def _sqlite_all_users():
    c = _sqlite_conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT id,username,email,auth_provider,full_name,avatar_url,role,"
            "created_at,last_login,is_logged_in FROM users ORDER BY created_at"
        ).fetchall()]
    finally:
        c.close()

def _sqlite_active_sessions():
    c = _sqlite_conn()
    try:
        return [dict(r) for r in c.execute(
            "SELECT s.*,u.email,u.role FROM sessions s "
            "JOIN users u ON s.user_id=u.id WHERE s.is_active=1 ORDER BY s.last_active DESC"
        ).fetchall()]
    finally:
        c.close()


# ─────────────────────────────────────────────────────────────────────────────
# Email
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch_reset_email(to_email: str, username: str, reset_link: str) -> Tuple[bool, str]:
    smtp_user     = _get_setting("SMTP_USER")
    smtp_pass     = (_get_setting("SMTP_PASS") or _get_setting("SMTP_PASSWORD")).replace(" ", "")
    smtp_host     = _get_setting("SMTP_HOST")
    smtp_port_raw = _get_setting("SMTP_PORT")

    # Smart default for Gmail
    if not smtp_host and smtp_user.lower().endswith("@gmail.com"):
        smtp_host = "smtp.gmail.com"

    smtp_port  = int(smtp_port_raw) if smtp_port_raw else 587
    raw_from   = _get_setting("SMTP_FROM")
    if raw_from and not (smtp_host == "smtp.gmail.com" and "aidatascientist.com" in raw_from):
        from_email = raw_from
    else:
        from_email = f"AI Data Scientist <{smtp_user}>"

    if not smtp_host or not smtp_user or not smtp_pass:
        return False, "SMTP credentials (SMTP_USER, SMTP_PASS) not configured in .env."

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Reset Your Password — AI Data Scientist"
        msg["From"]    = f"AI Data Scientist <{from_email}>"
        msg["To"]      = to_email

        plain_text = f"""Hello {username},

We received a request to reset your password for AI Data Scientist.
Click or paste the link below to set a new password:

{reset_link}

This link is valid for 60 minutes.
If you did not request this, please disregard this email.

— AI Data Scientist Team
"""

        html_text = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #06070f; color: #dde3f0; margin: 0; padding: 24px; }}
    .card {{ max-width: 480px; margin: 0 auto; background: #0e1122; border: 1px solid rgba(99,102,241,0.3); border-radius: 16px; padding: 32px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }}
    .brand {{ font-size: 20px; font-weight: 800; color: #818cf8; margin-bottom: 20px; text-align: center; }}
    .title {{ font-size: 18px; font-weight: 700; color: #ffffff; margin-bottom: 12px; }}
    .body-text {{ font-size: 14px; line-height: 1.6; color: #94a3b8; margin-bottom: 24px; }}
    .btn-container {{ text-align: center; margin: 28px 0; }}
    .btn {{ display: inline-block; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); color: #ffffff !important; padding: 14px 32px; text-decoration: none; border-radius: 10px; font-weight: 700; font-size: 15px; }}
    .link-alt {{ font-size: 12px; color: #64748b; word-break: break-all; margin-top: 24px; padding-top: 16px; border-top: 1px solid rgba(99,102,241,0.15); }}
    .footer {{ font-size: 12px; color: #475569; margin-top: 20px; text-align: center; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="brand">🧠 AI Data Scientist</div>
    <div class="title">Reset Your Password</div>
    <div class="body-text">
      Hello <b>{username}</b>,<br><br>
      We received a request to reset the password for your AI Data Scientist account. Click the button below to choose a new password:
    </div>
    <div class="btn-container">
      <a href="{reset_link}" class="btn" target="_blank" style="background:#6366f1;color:#ffffff;padding:14px 28px;text-decoration:none;border-radius:10px;font-weight:bold;display:inline-block;">Reset Password</a>
    </div>
    <div class="body-text" style="font-size: 13px; color: #64748b;">
      ⏱️ This link is valid for <b>60 minutes</b>.<br>
      If you did not request this, you can safely ignore this email — your password will not change.
    </div>
    <div class="link-alt">
      Button not working? Paste this link in your browser:<br>
      <a href="{reset_link}" style="color: #818cf8;">{reset_link}</a>
    </div>
  </div>
  <div class="footer">AI Data Scientist Agent · Autonomous Intelligence Platform</div>
</body>
</html>"""

        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_text, "html"))

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as s:
                s.login(smtp_user, smtp_pass)
                s.sendmail(from_email, [to_email], msg.as_string())
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as s:
                s.starttls()
                s.login(smtp_user, smtp_pass)
                s.sendmail(from_email, [to_email], msg.as_string())

        return True, "Email sent successfully!"
    except Exception as e:
        return False, f"SMTP error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
init_db()
