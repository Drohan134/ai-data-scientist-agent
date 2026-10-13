"""
auth_ui.py
Authentication UI for AI Data Scientist.
Supports:
  • Real Google OAuth2 (redirect flow via google_oauth.py)
  • Username / Password login
  • Register new account
  • Forgot Password (reset code)
"""

import os
import sys
import streamlit as st
import auth_db

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def is_google_available() -> bool:
    try:
        import google_oauth
        return google_oauth.is_configured()
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Shared CSS
# ─────────────────────────────────────────────────────────────────────────────
_AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"], .main {
    background: #06070f !important;
}
.main .block-container {
    max-width: 440px !important;
    padding: 60px 24px 40px !important;
    margin: 0 auto !important;
}

/* Title */
.auth-logo     { text-align: center; margin-bottom: 6px; }
.auth-logo-icon{ font-size: 2.6rem; line-height: 1; }
.auth-title {
    font-size: 1.7rem; font-weight: 800; letter-spacing: -0.035em;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; text-align: center; margin-bottom: 4px;
}
.auth-subtitle {
    font-size: 0.83rem; color: #4b5680;
    text-align: center; margin-bottom: 32px;
}

/* Google button */
.g-btn button {
    background: #fff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 12px !important;
    color: #3c4043 !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 12px 20px !important;
    transition: box-shadow 0.2s ease !important;
}
.g-btn button:hover {
    box-shadow: 0 2px 10px rgba(0,0,0,0.18) !important;
    background: #f8f9fa !important;
}

/* OR divider */
.or-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 20px 0; color: #2e3554;
    font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.or-divider::before, .or-divider::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(99,102,241,0.18);
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 60%, #06b6d4 100%) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 13px 28px !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
    transition: all 0.22s !important; letter-spacing: 0.015em !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.5) !important;
    filter: brightness(1.08) !important;
}

/* Secondary (nav) buttons */
.main .stButton > button:not([kind="primary"]) {
    background: transparent !important;
    border: none !important;
    color: #6366f1 !important;
    font-size: 0.82rem !important;
    padding: 4px 0 !important;
    font-weight: 500 !important;
    text-decoration: underline !important;
    box-shadow: none !important;
}
.main .stButton > button:not([kind="primary"]):hover {
    color: #a5b4fc !important;
    transform: none !important;
}

/* Inputs */
.stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 10px !important;
    color: #dde3f0 !important;
    font-size: 0.88rem !important; padding: 10px 14px !important;
}
.stTextInput input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stTextInput label { color: #7283a8 !important; font-size: 0.8rem !important; }

/* Info box */
.oauth-waiting {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px; padding: 16px 18px;
    font-size: 0.83rem; color: #7283a8; text-align: center;
    margin-bottom: 12px;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# OAuth callback handler (runs on every page load)
# ─────────────────────────────────────────────────────────────────────────────
def _handle_oauth_callback():
    """
    Check URL query params for ?code=... returned by Google.
    If found, exchange it for user info and log in.
    """
    params = st.query_params
    code  = params.get("code")
    state = params.get("state", "")

    if not code or state != "google_oauth":
        return False

    # Clear code from URL immediately to prevent re-use on refresh
    st.query_params.clear()

    with st.spinner("Signing you in with Google…"):
        user_info, error = google_oauth.exchange_code(code)

    if not user_info:
        st.error(f"❌ Google sign-in failed: {error or 'Unknown error. Check that http://localhost:8501/ is an Authorized Redirect URI in Google Cloud Console.'}")
        return True

    email     = user_info.get("email", "")
    full_name = user_info.get("name", email.split("@")[0])
    picture   = user_info.get("picture", "")
    google_id = user_info.get("sub", "")

    # Register or retrieve the user in our database
    ok, msg, user_dict = auth_db.authenticate_google_user(
        email, full_name=full_name, google_id=google_id, avatar_url=picture
    )

    if ok and user_dict:
        token = auth_db.start_user_session(user_dict["id"], user_dict["username"])
        st.session_state["authenticated_user"] = user_dict
        st.session_state["session_token"]      = token
        st.session_state.pop("_auth_view", None)
        st.toast(f"Welcome, {full_name}! 🎉", icon="🌐")
        st.rerun()
    else:
        st.error(f"❌ {msg}")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main auth page
# ─────────────────────────────────────────────────────────────────────────────
def render_auth_page():
    """Render the full-screen authentication portal."""

    # Hide sidebar
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none!important}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    # ── Handle Google OAuth callback first ─────────────────────────
    if is_google_available():
        handled = _handle_oauth_callback()
        if handled:
            return   # either logged in or showed error

    # ── Handle Password Reset Link (?reset_token=...) ───────────────
    reset_token = st.query_params.get("reset_token")
    if reset_token:
        _render_reset_password_page(reset_token)
        return

    # ── Title ──────────────────────────────────────────────────────
    st.markdown('<div class="auth-logo"><span class="auth-logo-icon">🧠</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">AI Data Scientist</div>', unsafe_allow_html=True)

    view = st.session_state.get("_auth_view", "login")

    subtitles = {
        "login":    "Sign in to your workspace",
        "register": "Create a new account",
        "forgot":   "Reset your password",
    }
    st.markdown(f'<div class="auth-subtitle">{subtitles.get(view, "")}</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # LOGIN VIEW
    # ══════════════════════════════════════════════════════════════
    if view == "login":
        if "_reset_success_msg" in st.session_state:
            st.success(st.session_state.pop("_reset_success_msg"))

        # ── Google button ────────────────────────────────────────
        if is_google_available():
            import google_oauth
            auth_url = google_oauth.build_auth_url(state="google_oauth")
            # Render as a real hyperlink styled as a button
            st.markdown(f"""
            <a href="{auth_url}" target="_self" style="text-decoration:none;display:block;width:100%">
              <div style="
                display:flex;align-items:center;justify-content:center;gap:10px;
                background:#fff;border:1px solid #dadce0;border-radius:12px;
                padding:12px 20px;cursor:pointer;width:100%;
                font-family:'Inter',sans-serif;font-size:0.9rem;font-weight:600;color:#3c4043;
                transition:box-shadow 0.2s;
              " onmouseover="this.style.boxShadow='0 2px 10px rgba(0,0,0,0.18)'"
                 onmouseout="this.style.boxShadow='none'">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48" width="20" height="20">
                  <path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/>
                  <path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/>
                  <path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/>
                  <path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.18 1.48-4.97 2.31-8.16 2.31-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/>
                  <path fill="none" d="M0 0h48v48H0z"/>
                </svg>
                Continue with Google
              </div>
            </a>
            """, unsafe_allow_html=True)

            st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Google OAuth not configured. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to Streamlit Secrets (or `.env` locally).")

        # ── Username / Password form ─────────────────────────────
        with st.form("login_form", clear_on_submit=False):
            username_in = st.text_input("Username or Email", placeholder="admin", key="login_user")
            password_in = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            submitted   = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submitted:
                if not username_in or not password_in:
                    st.error("Please enter your username/email and password.")
                else:
                    ok, msg, user_dict = auth_db.authenticate_user(username_in, password_in)
                    if ok and user_dict:
                        token = auth_db.start_user_session(user_dict["id"], user_dict["username"])
                        st.session_state["authenticated_user"] = user_dict
                        st.session_state["session_token"]      = token
                        st.toast(f"Welcome back, {user_dict.get('full_name') or user_dict['username']}!", icon="🎉")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        # ── Bottom links ─────────────────────────────────────────
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button("Forgot password?", key="btn_goto_forgot"):
                st.session_state["_auth_view"] = "forgot"
                st.rerun()
        with col_r:
            if st.button("Create an account →", key="btn_goto_register"):
                st.session_state["_auth_view"] = "register"
                st.rerun()

    # ══════════════════════════════════════════════════════════════
    # REGISTER VIEW
    # ══════════════════════════════════════════════════════════════
    elif view == "register":
        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Full Name", placeholder="Alex Morgan",         key="reg_name")
            username  = st.text_input("Username",  placeholder="alex_scientist",      key="reg_user")
            email     = st.text_input("Email",     placeholder="alex@example.com",    key="reg_email")
            password  = st.text_input("Password",  type="password",
                                      placeholder="At least 6 characters",            key="reg_pass")
            confirm   = st.text_input("Confirm Password", type="password",
                                      placeholder="Re-enter password",                key="reg_confirm")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submitted:
                if not username or not email or not password:
                    st.error("Please fill in all required fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, msg, new_user = auth_db.register_user(username, email, password, full_name=full_name)
                    if ok and new_user:
                        token = auth_db.start_user_session(new_user["id"], new_user["username"])
                        st.session_state["authenticated_user"] = new_user
                        st.session_state["session_token"]      = token
                        st.session_state["_auth_view"]         = "login"
                        st.toast("Account created! Welcome aboard 🎉", icon="✨")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        if st.button("← Back to Sign In", key="btn_back_login"):
            st.session_state["_auth_view"] = "login"
            st.rerun()

    # ══════════════════════════════════════════════════════════════
    # FORGOT PASSWORD VIEW
    # ══════════════════════════════════════════════════════════════
    elif view == "forgot":
        st.markdown(
            '<p style="font-size:0.83rem;color:#7283a8;margin-bottom:18px">'
            'Enter your registered email address or username and we will send you a secure link to reset your password.'
            '</p>',
            unsafe_allow_html=True,
        )
        with st.form("forgot_form"):
            identity  = st.text_input("Registered Email or Username",
                                      placeholder="you@example.com", key="fp_id")
            submitted = st.form_submit_button("Send Reset Link", type="primary",
                                              use_container_width=True)
            if submitted:
                if not identity:
                    st.error("Please enter your email or username.")
                else:
                    ok, link_or_err, info = auth_db.create_password_reset_link(identity)
                    if ok and info:
                        st.session_state["_reset_info"] = info
                        st.rerun()
                    else:
                        st.error(f"❌ {link_or_err}")

        info = st.session_state.get("_reset_info")
        if info:
            if info.get("smtp_sent"):
                st.success(f"📬 **Password reset link sent!** We have sent an email to **{info['email']}**. Please check your inbox and click the link to reset your password.")
                st.caption("Valid for 60 minutes. Check your spam folder if it doesn't appear within a minute.")
            else:
                st.warning(
                    "⚠️ **Email delivery (SMTP) is not configured in `.env`.**\n\n"
                    "To send real emails to your users, add your SMTP credentials (`SMTP_USER`, `SMTP_PASS`) to `.env`."
                )
                st.markdown(f"""
                <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:16px;margin:14px 0">
                  <div style="font-size:0.8rem;color:#a5b4fc;font-weight:700;margin-bottom:8px">🔗 Generated Reset Link (click to test):</div>
                  <a href="{info['reset_link']}" target="_self" style="color:#38bdf8;font-size:0.84rem;word-break:break-all;text-decoration:underline">Click here to reset your password</a>
                </div>
                """, unsafe_allow_html=True)
                st.code(info['reset_link'], language="text")

        if st.button("← Back to Sign In", key="btn_fp_back"):
            st.session_state.pop("_reset_info", None)
            st.session_state["_auth_view"] = "login"
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Password Reset Page (Triggered by ?reset_token=...)
# ─────────────────────────────────────────────────────────────────────────────
def _render_reset_password_page(token: str):
    """Render the password reset form when a user opens their emailed reset link."""
    st.markdown('<div class="auth-logo"><span class="auth-logo-icon">🔑</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Set New Password</div>', unsafe_allow_html=True)

    ok, msg, token_info = auth_db.verify_reset_token(token)
    if not ok or not token_info:
        st.error(
            "❌ **Invalid or Expired Link**\n\n"
            "This password reset link is invalid, has expired, or has already been used."
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Request New Link", key="btn_tok_req_new", type="primary"):
                st.query_params.clear()
                st.session_state["_auth_view"] = "forgot"
                st.rerun()
        with col2:
            if st.button("Back to Sign In", key="btn_tok_back_sign"):
                st.query_params.clear()
                st.session_state["_auth_view"] = "login"
                st.rerun()
        return

    email = token_info.get("email", "")
    st.markdown(
        f'<div class="auth-subtitle">Create a new password for <b style="color:#a5b4fc">{email}</b></div>',
        unsafe_allow_html=True,
    )

    with st.form("set_new_password_form", clear_on_submit=False):
        new_pwd = st.text_input("New Password", type="password", placeholder="At least 6 characters", key="rp_new")
        confirm_pwd = st.text_input("Confirm New Password", type="password", placeholder="Re-enter password", key="rp_confirm")
        submitted = st.form_submit_button("Update Password", type="primary", use_container_width=True)

        if submitted:
            if not new_pwd or not confirm_pwd:
                st.error("Please fill in both password fields.")
            elif new_pwd != confirm_pwd:
                st.error("Passwords do not match.")
            elif len(new_pwd) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                success, upd_msg = auth_db.reset_password_with_token(token, new_pwd)
                if success:
                    st.query_params.clear()
                    st.session_state["_reset_success_msg"] = "✅ Password updated successfully! Please sign in."
                    st.session_state["_auth_view"] = "login"
                    st.rerun()
                else:
                    st.error(f"❌ {upd_msg}")

    if st.button("Cancel & Return to Sign In", key="btn_tok_cancel"):
        st.query_params.clear()
        st.session_state["_auth_view"] = "login"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Users & Database Management Tab
# ─────────────────────────────────────────────────────────────────────────────
def render_users_management_tab(current_user: dict):
    """Render the Users & Database inspector tab."""
    st.markdown("""
    <div class="hero-title" style="font-size:1.65rem">👥 User Management & Database Inspector</div>
    <div class="hero-sub">Direct SQLite connection · active sessions · user credentials & authentication logs</div>
    <div style="margin-bottom:22px"></div>
    """, unsafe_allow_html=True)

    users    = auth_db.get_all_users()
    sessions = auth_db.get_active_sessions()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:1.5rem">👥</div>
            <div style="font-size:.78rem;color:#7283a8;text-transform:uppercase;margin-top:6px">Total Users</div>
            <div style="font-size:1.6rem;font-weight:800;color:#dde3f0">{len(users)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        online_count = sum(1 for u in users if u.get("is_logged_in") == 1)
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:1.5rem">🟢</div>
            <div style="font-size:.78rem;color:#7283a8;text-transform:uppercase;margin-top:6px">Currently Online</div>
            <div style="font-size:1.6rem;font-weight:800;color:#34d399">{online_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:1.5rem">⚡</div>
            <div style="font-size:.78rem;color:#7283a8;text-transform:uppercase;margin-top:6px">Active Sessions</div>
            <div style="font-size:1.6rem;font-weight:800;color:#38bdf8">{len(sessions)}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        db_size_kb = 0
        if os.path.exists(auth_db.DB_PATH):
            db_size_kb = round(os.path.getsize(auth_db.DB_PATH) / 1024, 1)
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:1.5rem">🗄️</div>
            <div style="font-size:.78rem;color:#7283a8;text-transform:uppercase;margin-top:6px">Database Storage</div>
            <div style="font-size:1.6rem;font-weight:800;color:#a5b4fc">{db_size_kb} KB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Active sessions table
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon">🟢</span><h2>Active Logged-In Sessions</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
    if sessions:
        import pandas as pd
        sess_rows = []
        for s in sessions:
            sess_rows.append({
                "Session ID":  s["session_token"][:16] + "...",
                "Username":    s["username"],
                "Email":       s["email"],
                "Role":        s["role"].upper(),
                "Login Time":  s["login_time"],
                "Last Active": s["last_active"],
                "Status":      "🟢 Online",
            })
        st.dataframe(pd.DataFrame(sess_rows), use_container_width=True)
    else:
        st.info("No active sessions currently recorded.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Registered users table
    st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon">📋</span><h2>Registered Users in Database</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
    if users:
        import pandas as pd
        user_rows = []
        for u in users:
            is_online = "🟢 Online" if u.get("is_logged_in") == 1 else "⚪ Offline"
            user_rows.append({
                "ID":            u["id"],
                "Username":      u["username"],
                "Full Name":     u.get("full_name") or u["username"],
                "Email":         u["email"],
                "Provider":      u.get("auth_provider", "local").upper(),
                "Role":          u.get("role", "user").upper(),
                "Status":        is_online,
                "Registered At": u.get("created_at", "N/A"),
                "Last Login":    u.get("last_login") or "Never",
            })
        st.dataframe(pd.DataFrame(user_rows), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_act1, col_act2 = st.columns([1, 1])
    with col_act1:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon">🛡️</span><h2>Session Termination</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
        user_options = {f"{u['username']} ({u['email']})": u['id'] for u in users}
        if user_options:
            selected_user_label = st.selectbox("Select user to force logout:", list(user_options.keys()), key="select_force_logout")
            if st.button("🚫 Terminate All Sessions", key="btn_force_logout"):
                uid = user_options[selected_user_label]
                auth_db.force_logout_user(uid)
                st.success(f"Terminated all active sessions for {selected_user_label}!")
                st.rerun()

    with col_act2:
        st.markdown('<div class="sec-hdr"><span class="sec-hdr-icon">⚡</span><h2>Database Actions</h2><div class="sec-line"></div></div>', unsafe_allow_html=True)
        if st.button("🔄 Refresh Database State", use_container_width=True, key="btn_refresh_db"):
            st.rerun()
        google_status = "✅ Configured" if is_google_available() else "❌ Not configured"
        st.markdown(f"""
        <div style="font-size:0.8rem;color:#7283a8;line-height:1.6;margin-top:10px">
        • <b>Database File:</b> <code>{auth_db.DB_PATH}</code><br>
        • <b>Tables:</b> <code>users</code>, <code>sessions</code>, <code>password_resets</code><br>
        • <b>Security:</b> Bcrypt password hashing + salted tokens.<br>
        • <b>Google OAuth:</b> {google_status}
        </div>
        """, unsafe_allow_html=True)
