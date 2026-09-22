"""
Analytics and Admin Access Management for BizInsights LLM
- Local SQLite analytics engine (zero external dependencies)
- Tracks unique visitors, active sessions, device OS, device types, browsers
- Secure admin authentication and protected dashboard routes
"""
import os
import re
import time
import secrets
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "analytics.db"
STATIC_DIR = ROOT_DIR / "static"

# Configurable admin password via environment variable (default: bizinsights2026!)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "bizinsights2026!")
COOKIE_NAME = "biz_admin_session"
SESSION_DURATION_SECONDS = 86400  # 24 hours

# Rate limiting for admin login attempts: {ip: [timestamps]}
LOGIN_ATTEMPTS: Dict[str, List[float]] = {}
MAX_LOGIN_ATTEMPTS = 5
LOGIN_WINDOW_SECONDS = 300  # 5 minutes


def mask_ip(ip: str) -> str:
    """Mask IP address for privacy compliance (e.g., 192.168.1.***)."""
    if not ip or ip == "127.0.0.1" or ip == "localhost":
        return "127.0.0.1 (Local)"
    if ":" in ip:  # IPv6
        parts = ip.split(":")
        return ":".join(parts[:3]) + ":****:****"
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.***"
    return "Hidden IP"


def parse_user_agent(ua_string: str) -> Dict[str, str]:
    """Parse User-Agent string to detect OS, Browser, and Device category."""
    if not ua_string:
        return {"os": "Unknown", "browser": "Unknown", "device": "Desktop"}

    ua = ua_string.lower()

    # 1. Operating System
    if "iphone" in ua or "ipod" in ua:
        os_name = "iOS"
    elif "ipad" in ua:
        os_name = "iPadOS"
    elif "android" in ua:
        os_name = "Android"
    elif "windows nt 10.0" in ua or "windows nt 11.0" in ua or "windows" in ua:
        os_name = "Windows"
    elif "macintosh" in ua or "mac os x" in ua:
        os_name = "macOS"
    elif "cros" in ua:
        os_name = "Chrome OS"
    elif "linux" in ua:
        os_name = "Linux"
    else:
        os_name = "Other"

    # 2. Device Category
    if "ipad" in ua or "tablet" in ua or ("android" in ua and "mobile" not in ua):
        device_type = "Tablet"
    elif "mobile" in ua or "iphone" in ua or "ipod" in ua or "android" in ua:
        device_type = "Mobile"
    else:
        device_type = "Desktop"

    # 3. Browser
    if "edg/" in ua or "edge/" in ua:
        browser = "Microsoft Edge"
    elif "opr/" in ua or "opera" in ua:
        browser = "Opera"
    elif "samsungbrowser" in ua:
        browser = "Samsung Internet"
    elif "firefox/" in ua or "fxios/" in ua:
        browser = "Firefox"
    elif "chrome/" in ua or "crios/" in ua:
        browser = "Chrome"
    elif "safari/" in ua and "chrome" not in ua:
        browser = "Safari"
    else:
        browser = "Other"

    return {"os": os_name, "browser": browser, "device": device_type}


class AnalyticsDB:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for high concurrency read/write
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            conn.execute("PRAGMA busy_timeout=10000;")
        except sqlite3.OperationalError:
            pass
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS visitors (
                    visitor_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    os TEXT,
                    device_type TEXT,
                    browser TEXT,
                    screen_res TEXT,
                    masked_ip TEXT,
                    visit_count INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    visitor_id TEXT NOT NULL,
                    started_at TIMESTAMP,
                    last_seen TIMESTAMP,
                    pageviews INTEGER DEFAULT 1,
                    os TEXT,
                    device_type TEXT,
                    browser TEXT,
                    masked_ip TEXT
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    visitor_id TEXT,
                    event_type TEXT,
                    path TEXT,
                    os TEXT,
                    device_type TEXT,
                    browser TEXT,
                    masked_ip TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS admin_tokens (
                    token TEXT PRIMARY KEY,
                    created_at REAL,
                    expires_at REAL
                );

                CREATE INDEX IF NOT EXISTS idx_visitors_last_seen ON visitors(last_seen);
                CREATE INDEX IF NOT EXISTS idx_sessions_last_seen ON sessions(last_seen);
                CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
            """)

    def record_visit(
        self,
        visitor_id: str,
        session_id: str,
        ip: str,
        ua_string: str,
        client_os: Optional[str] = None,
        client_device: Optional[str] = None,
        client_browser: Optional[str] = None,
        screen_res: Optional[str] = None,
        path: str = "/",
        event_type: str = "pageview"
    ):
        """Record or update visitor, session, and event."""
        now = datetime.utcnow().isoformat()
        masked = mask_ip(ip)

        # Parse UA as fallback or primary
        parsed = parse_user_agent(ua_string)
        os_name = client_os if client_os and client_os != "Unknown" else parsed["os"]
        device_type = client_device if client_device and client_device != "Unknown" else parsed["device"]
        browser = client_browser if client_browser and client_browser != "Unknown" else parsed["browser"]
        screen = screen_res or "Unknown"

        with self._get_connection() as conn:
            # 1. Update or insert visitor
            cursor = conn.cursor()
            cursor.execute("SELECT visitor_id, visit_count FROM visitors WHERE visitor_id = ?", (visitor_id,))
            row = cursor.fetchone()
            if row:
                conn.execute("""
                    UPDATE visitors
                    SET last_seen = ?,
                        os = COALESCE(?, os),
                        device_type = COALESCE(?, device_type),
                        browser = COALESCE(?, browser),
                        screen_res = COALESCE(?, screen_res),
                        masked_ip = ?,
                        visit_count = visit_count + (CASE WHEN ? = 'pageview' THEN 1 ELSE 0 END)
                    WHERE visitor_id = ?
                """, (now, os_name, device_type, browser, screen, masked, event_type, visitor_id))
            else:
                conn.execute("""
                    INSERT INTO visitors (visitor_id, first_seen, last_seen, os, device_type, browser, screen_res, masked_ip, visit_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """, (visitor_id, now, now, os_name, device_type, browser, screen, masked))

            # 2. Update or insert session
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            s_row = cursor.fetchone()
            if s_row:
                conn.execute("""
                    UPDATE sessions
                    SET last_seen = ?,
                        pageviews = pageviews + (CASE WHEN ? = 'pageview' THEN 1 ELSE 0 END)
                    WHERE session_id = ?
                """, (now, event_type, session_id))
            else:
                conn.execute("""
                    INSERT INTO sessions (session_id, visitor_id, started_at, last_seen, pageviews, os, device_type, browser, masked_ip)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?)
                """, (session_id, visitor_id, now, now, os_name, device_type, browser, masked))

            # 3. Insert event (log all pageviews and chat events, skip pure heartbeats from bloated events table)
            if event_type != "heartbeat":
                conn.execute("""
                    INSERT INTO events (session_id, visitor_id, event_type, path, os, device_type, browser, masked_ip, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, visitor_id, event_type, path, os_name, device_type, browser, masked, "{}", now))

    def record_chat_event(
        self,
        visitor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip: str = "",
        ua_string: str = "",
        metadata: str = "{}"
    ):
        """Record an LLM chat inference request with device/browser resolution."""
        now = datetime.utcnow().isoformat()
        masked = mask_ip(ip)

        os_name = "Unknown"
        device_type = "Unknown"
        browser = "Unknown"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Inherit from active session if known
            if session_id and session_id != "unknown":
                cursor.execute("SELECT os, device_type, browser FROM sessions WHERE session_id = ?", (session_id,))
                s_row = cursor.fetchone()
                if s_row:
                    os_name = s_row["os"] or "Unknown"
                    device_type = s_row["device_type"] or "Unknown"
                    browser = s_row["browser"] or "Unknown"

            # 2. Inherit from visitor record if still unknown
            if os_name == "Unknown" and visitor_id and visitor_id != "unknown":
                cursor.execute("SELECT os, device_type, browser FROM visitors WHERE visitor_id = ?", (visitor_id,))
                v_row = cursor.fetchone()
                if v_row:
                    os_name = v_row["os"] or "Unknown"
                    device_type = v_row["device_type"] or "Unknown"
                    browser = v_row["browser"] or "Unknown"

            # 3. Fall back to parsing User-Agent string directly
            if (os_name == "Unknown" or browser == "Unknown") and ua_string:
                parsed = parse_user_agent(ua_string)
                if os_name == "Unknown":
                    os_name = parsed["os"]
                if device_type == "Unknown":
                    device_type = parsed["device"]
                if browser == "Unknown":
                    browser = parsed["browser"]

            conn.execute("""
                INSERT INTO events (session_id, visitor_id, event_type, path, os, device_type, browser, masked_ip, metadata, created_at)
                VALUES (?, ?, 'chat_request', '/api/chat', ?, ?, ?, ?, ?, ?)
            """, (session_id or "unknown", visitor_id or "unknown", os_name, device_type, browser, masked, metadata, now))

    def get_summary(self) -> Dict[str, Any]:
        """Aggregate analytics metrics for the admin dashboard."""
        now_ts = datetime.utcnow()
        # Active in last 5 minutes (300 seconds)
        active_cutoff = datetime.utcfromtimestamp(now_ts.timestamp() - 300).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Active users (unique visitors seen in last 5 minutes)
            cursor.execute("SELECT COUNT(DISTINCT visitor_id) FROM visitors WHERE last_seen >= ?", (active_cutoff,))
            active_users = cursor.fetchone()[0] or 0

            # Total unique visitors
            cursor.execute("SELECT COUNT(*) FROM visitors")
            total_visitors = cursor.fetchone()[0] or 0

            # Total sessions
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total_sessions = cursor.fetchone()[0] or 0

            # Total pageviews
            cursor.execute("SELECT SUM(pageviews) FROM sessions")
            total_pageviews = cursor.fetchone()[0] or 0

            # Total chat events
            cursor.execute("SELECT COUNT(*) FROM events WHERE event_type = 'chat_request'")
            total_chats = cursor.fetchone()[0] or 0

            # OS breakdown
            cursor.execute("""
                SELECT os, COUNT(*) as cnt
                FROM visitors
                WHERE os IS NOT NULL AND os != ''
                GROUP BY os
                ORDER BY cnt DESC
            """)
            os_rows = cursor.fetchall()
            os_list = []
            for r in os_rows:
                pct = round((r["cnt"] / max(total_visitors, 1)) * 100, 1)
                os_list.append({"name": r["os"], "count": r["cnt"], "percentage": pct})

            # Device breakdown
            cursor.execute("""
                SELECT device_type, COUNT(*) as cnt
                FROM visitors
                WHERE device_type IS NOT NULL AND device_type != ''
                GROUP BY device_type
                ORDER BY cnt DESC
            """)
            device_rows = cursor.fetchall()
            device_list = []
            for r in device_rows:
                pct = round((r["cnt"] / max(total_visitors, 1)) * 100, 1)
                device_list.append({"name": r["device_type"], "count": r["cnt"], "percentage": pct})

            # Browser breakdown
            cursor.execute("""
                SELECT browser, COUNT(*) as cnt
                FROM visitors
                WHERE browser IS NOT NULL AND browser != ''
                GROUP BY browser
                ORDER BY cnt DESC
            """)
            browser_rows = cursor.fetchall()
            browser_list = []
            for r in browser_rows:
                pct = round((r["cnt"] / max(total_visitors, 1)) * 100, 1)
                browser_list.append({"name": r["browser"], "count": r["cnt"], "percentage": pct})

            # Recent activity stream (last 50 events)
            cursor.execute("""
                SELECT id, session_id, visitor_id, event_type, path, os, device_type, browser, masked_ip, created_at
                FROM events
                ORDER BY id DESC
                LIMIT 50
            """)
            recent_events = [dict(r) for r in cursor.fetchall()]

            # Active sessions detail
            cursor.execute("""
                SELECT visitor_id, session_id, os, device_type, browser, masked_ip, started_at, last_seen, pageviews
                FROM sessions
                WHERE last_seen >= ?
                ORDER BY last_seen DESC
                LIMIT 20
            """, (active_cutoff,))
            active_sessions = [dict(r) for r in cursor.fetchall()]

        return {
            "active_users": active_users,
            "total_visitors": total_visitors,
            "total_sessions": total_sessions,
            "total_pageviews": total_pageviews,
            "total_chats": total_chats,
            "os_breakdown": os_list,
            "device_breakdown": device_list,
            "browser_breakdown": browser_list,
            "recent_events": recent_events,
            "active_sessions": active_sessions,
            "server_time": now_ts.strftime("%Y-%m-%d %H:%M:%S UTC")
        }

    def clear_data(self):
        """Clear all analytics logs."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM events;")
            conn.execute("DELETE FROM sessions;")
            conn.execute("DELETE FROM visitors;")

    def create_admin_token(self) -> str:
        token = secrets.token_hex(32)
        now = time.time()
        expires = now + SESSION_DURATION_SECONDS
        with self._get_connection() as conn:
            conn.execute("INSERT INTO admin_tokens (token, created_at, expires_at) VALUES (?, ?, ?)", (token, now, expires))
        return token

    def validate_admin_token(self, token: Optional[str]) -> bool:
        if not token:
            return False
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT expires_at FROM admin_tokens WHERE token = ?", (token,))
            row = cursor.fetchone()
            if row and row["expires_at"] > now:
                return True
            if row:
                # Remove expired token
                conn.execute("DELETE FROM admin_tokens WHERE token = ?", (token,))
        return False

    def revoke_admin_token(self, token: Optional[str]):
        if not token:
            return
        with self._get_connection() as conn:
            conn.execute("DELETE FROM admin_tokens WHERE token = ?", (token,))


# Initialize database instance
db = AnalyticsDB()


def is_rate_limited(ip: str) -> bool:
    """Check if IP has exceeded login attempt threshold."""
    now = time.time()
    attempts = [t for t in LOGIN_ATTEMPTS.get(ip, []) if now - t < LOGIN_WINDOW_SECONDS]
    # Evict expired keys so spoofed IPs cannot grow memory forever.
    # Overflow of fresh keys drops the oldest: throttle only, never data.
    if len(LOGIN_ATTEMPTS) > 5000:
        for key in [k for k, v in LOGIN_ATTEMPTS.items() if not any(t > now - LOGIN_WINDOW_SECONDS for t in v)]:
            LOGIN_ATTEMPTS.pop(key, None)
    while len(LOGIN_ATTEMPTS) > 5000:
        LOGIN_ATTEMPTS.pop(next(iter(LOGIN_ATTEMPTS)), None)
    LOGIN_ATTEMPTS[ip] = attempts
    return len(attempts) >= MAX_LOGIN_ATTEMPTS


def _peer_is_loopback(request: Request) -> bool:
    try:
        host = (request.client.host if request.client else "") or ""
    except Exception:
        return False
    return host.startswith("127.") or host in ("::1", "::ffff:127.0.0.1", "localhost")


def get_client_ip(request: Request) -> str:
    """Extract real client IP, trusting proxy headers only from loopback.

    The Cloudflare tunnel lands on 127.0.0.1, so headers stay trusted
    there. Direct LAN clients use the socket peer, which stops header
    spoofing from defeating the login and upload rate limits.
    """
    if _peer_is_loopback(request):
        cf_ip = request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def record_failed_attempt(ip: str):
    now = time.time()
    if ip not in LOGIN_ATTEMPTS:
        LOGIN_ATTEMPTS[ip] = []
    LOGIN_ATTEMPTS[ip].append(now)


def is_authenticated(request: Request) -> bool:
    """Verify admin session token from cookie or Authorization header."""
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
    return db.validate_admin_token(token)


# Pydantic models
class CollectPayload(BaseModel):
    visitor_id: str
    session_id: str
    event_type: Optional[str] = "pageview"
    path: Optional[str] = "/"
    os: Optional[str] = None
    device_type: Optional[str] = None
    browser: Optional[str] = None
    screen_res: Optional[str] = None


class LoginPayload(BaseModel):
    password: str


# Analytics APIRouter
router = APIRouter()


@router.post("/api/analytics/collect")
async def collect_analytics(payload: CollectPayload, request: Request):
    """Public analytics beacon sent by client on page load / heartbeat."""
    ip = get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    try:
        db.record_visit(
            visitor_id=payload.visitor_id,
            session_id=payload.session_id,
            ip=ip,
            ua_string=ua,
            client_os=payload.os,
            client_device=payload.device_type,
            client_browser=payload.browser,
            screen_res=payload.screen_res,
            path=payload.path or "/",
            event_type=payload.event_type or "pageview"
        )
        return {"status": "ok"}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/admin/login", response_class=HTMLResponse)
async def serve_admin_login(request: Request):
    """Serve Admin Login page. If already authenticated, redirect to /admin."""
    if is_authenticated(request):
        return RedirectResponse(url="/admin", status_code=303)

    login_file = STATIC_DIR / "admin_login.html"
    if login_file.is_file():
        return HTMLResponse(login_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Admin Login</h1><form method='POST'><input type='password' name='password'/><button>Log in</button></form>")


@router.post("/admin/login")
async def handle_admin_login(request: Request):
    """Authenticate administrator via password and issue session cookie."""
    ip = get_client_ip(request)

    if is_rate_limited(ip):
        return JSONResponse(
            {"success": False, "error": "Too many failed attempts. Please wait 5 minutes before trying again."},
            status_code=429
        )

    # Support JSON or Form body
    password_entered = ""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            password_entered = body.get("password", "")
        except Exception:
            password_entered = ""
    else:
        form = await request.form()
        password_entered = form.get("password", "")

    # Constant-time comparison
    if secrets.compare_digest(password_entered, ADMIN_PASSWORD):
        token = db.create_admin_token()
        response = JSONResponse({"success": True, "redirect": "/admin"})
        # Set HttpOnly, SameSite=Lax cookie
        response.set_cookie(
            key=COOKIE_NAME,
            value=token,
            max_age=SESSION_DURATION_SECONDS,
            httponly=True,
            samesite="lax",
            secure=False  # Allow local network HTTP access
        )
        return response
    else:
        record_failed_attempt(ip)
        return JSONResponse(
            {"success": False, "error": "Invalid password. Access denied."},
            status_code=401
        )


@router.post("/admin/logout")
async def handle_admin_logout(request: Request):
    """Revoke admin session and redirect to login."""
    token = request.cookies.get(COOKIE_NAME)
    if token:
        db.revoke_admin_token(token)
    response = JSONResponse({"success": True, "redirect": "/admin/login"})
    response.delete_cookie(COOKIE_NAME)
    return response


@router.get("/admin", response_class=HTMLResponse)
async def serve_admin_dashboard(request: Request):
    """Serve Admin Dashboard if authenticated, else redirect to login."""
    if not is_authenticated(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    admin_file = STATIC_DIR / "admin.html"
    if admin_file.is_file():
        return HTMLResponse(admin_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BizInsights Admin</h1><p>admin.html not found.</p>")


@router.get("/api/admin/analytics")
async def get_admin_analytics(request: Request):
    """Return aggregated analytics data. Strictly protected by admin auth."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized: Admin access required.")
    return db.get_summary()


@router.post("/api/admin/clear")
async def clear_admin_analytics(request: Request):
    """Clear analytics data. Strictly protected by admin auth."""
    if not is_authenticated(request):
        raise HTTPException(status_code=401, detail="Unauthorized: Admin access required.")
    db.clear_data()
    return {"success": True, "message": "All analytics logs cleared."}
