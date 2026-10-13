"""
google_oauth.py
Real Google OAuth2 for Streamlit — uses raw requests, no PKCE.

Flow:
  1. build_auth_url()  -> user redirected to Google account chooser
  2. exchange_code()   -> exchanges ?code= for access token + user info
"""

import os
import urllib.parse
from dotenv import load_dotenv

# Must be set before any oauthlib import
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Load .env from the same directory as this file
_DOTENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_DOTENV, override=True)

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


CLIENT_ID     = _get_setting("GOOGLE_CLIENT_ID")
CLIENT_SECRET = _get_setting("GOOGLE_CLIENT_SECRET")
REDIRECT_URI  = _get_setting("GOOGLE_REDIRECT_URI", "http://localhost:8501/")

def get_client_id() -> str:
    return _get_setting("GOOGLE_CLIENT_ID") or CLIENT_ID

def get_client_secret() -> str:
    return _get_setting("GOOGLE_CLIENT_SECRET") or CLIENT_SECRET

def get_redirect_uri() -> str:
    uri = _get_setting("GOOGLE_REDIRECT_URI")
    if uri:
        return uri
    base = _get_setting("APP_BASE_URL")
    if base:
        return base.rstrip("/") + "/"
    return REDIRECT_URI

_AUTH_ENDPOINT  = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_USERINFO_URL   = "https://www.googleapis.com/oauth2/v3/userinfo"

SCOPES = "openid email profile"


def is_configured() -> bool:
    cid = get_client_id()
    csecret = get_client_secret()
    return bool(cid and csecret and ".apps.googleusercontent.com" in cid)


def build_auth_url(state: str = "google_oauth") -> str:
    """
    Build the Google OAuth2 authorization URL manually (no PKCE, no wrappers).
    """
    params = {
        "client_id":     get_client_id(),
        "redirect_uri":  get_redirect_uri(),
        "response_type": "code",
        "scope":         SCOPES,
        "state":         state,
        "access_type":   "offline",
        "prompt":        "select_account",
    }
    return _AUTH_ENDPOINT + "?" + urllib.parse.urlencode(params)


def exchange_code(code: str) -> tuple:
    """
    Exchange the authorization code for an access token, then fetch user info.
    Returns (user_info_dict, error_string).
    user_info keys: sub, email, name, given_name, family_name, picture
    """
    import requests

    cid = get_client_id()
    csecret = get_client_secret()
    r_uri = get_redirect_uri()

    # Step 1: Exchange code for access token
    try:
        token_resp = requests.post(
            _TOKEN_ENDPOINT,
            data={
                "code":          code,
                "client_id":     cid,
                "client_secret": csecret,
                "redirect_uri":  r_uri,
                "grant_type":    "authorization_code",
            },
            timeout=15,
        )
    except Exception as exc:
        return None, f"Token request network error: {exc}"

    if token_resp.status_code != 200:
        return None, f"Token exchange failed: HTTP {token_resp.status_code} — {token_resp.text[:300]}"

    token_data   = token_resp.json()
    access_token = token_data.get("access_token")

    if not access_token:
        return None, f"No access_token in response: {token_data}"

    # Step 2: Fetch user profile
    try:
        info_resp = requests.get(
            _USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
    except Exception as exc:
        return None, f"Userinfo network error: {exc}"

    if info_resp.status_code != 200:
        return None, f"Userinfo failed: HTTP {info_resp.status_code} — {info_resp.text[:200]}"

    return info_resp.json(), ""
