-- ============================================================
-- Supabase schema for AI Data Scientist auth system
-- Run this in Supabase → SQL Editor → New Query → Run
-- ============================================================

-- 1. USERS TABLE
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT UNIQUE NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    password_hash   TEXT,
    salt            TEXT,
    auth_provider   TEXT NOT NULL DEFAULT 'local',
    google_id       TEXT,
    full_name       TEXT,
    avatar_url      TEXT,
    role            TEXT NOT NULL DEFAULT 'user',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login      TIMESTAMPTZ,
    is_logged_in    BOOLEAN NOT NULL DEFAULT FALSE
);

-- 2. SESSIONS TABLE
CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token   TEXT UNIQUE NOT NULL,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    username        TEXT NOT NULL,
    ip_address      TEXT DEFAULT '127.0.0.1',
    user_agent      TEXT DEFAULT 'Streamlit',
    login_time      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_active     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

-- 3. PASSWORD RESETS TABLE
CREATE TABLE IF NOT EXISTS password_resets (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    email           TEXT NOT NULL,
    reset_code      TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    used            BOOLEAN NOT NULL DEFAULT FALSE
);

-- 4. INDEXES for performance
CREATE INDEX IF NOT EXISTS idx_sessions_token    ON sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user     ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_active   ON sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_resets_user       ON password_resets(user_id);

-- 5. ROW LEVEL SECURITY
--    Users can only read their own row; service_role (admin) sees all
ALTER TABLE users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE password_resets ENABLE ROW LEVEL SECURITY;

-- Allow service_role (your backend) full access
CREATE POLICY "service_role_all_users"    ON users           FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all_sessions" ON sessions        FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE);
CREATE POLICY "service_role_all_resets"   ON password_resets FOR ALL TO service_role USING (TRUE) WITH CHECK (TRUE);

-- 6. SEED DEFAULT ADMIN ACCOUNT
--    Password = AdminPassword123! (hashed via app on first run, this is a placeholder)
INSERT INTO users (username, email, auth_provider, full_name, role)
VALUES ('admin', 'admin@aidatascientist.com', 'local', 'System Administrator', 'admin')
ON CONFLICT (username) DO NOTHING;
