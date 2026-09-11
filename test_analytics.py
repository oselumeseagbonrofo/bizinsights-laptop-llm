"""
Comprehensive test suite for BizInsights Analytics & Admin Access Control
"""
import sys
import os
import time
from pathlib import Path

# Ensure virtualenv libraries can be imported
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from server import app
from analytics import (
    parse_user_agent,
    AnalyticsDB,
    ADMIN_PASSWORD,
    COOKIE_NAME
)

client = TestClient(app)

def test_user_agent_parser():
    print("[1] Testing User-Agent parser...")

    # Windows 11 Chrome
    ua_win = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    res = parse_user_agent(ua_win)
    assert res["os"] == "Windows", f"Expected Windows, got {res['os']}"
    assert res["device"] == "Desktop", f"Expected Desktop, got {res['device']}"
    assert res["browser"] == "Chrome", f"Expected Chrome, got {res['browser']}"

    # iPhone iOS Safari
    ua_ios = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1"
    res = parse_user_agent(ua_ios)
    assert res["os"] == "iOS", f"Expected iOS, got {res['os']}"
    assert res["device"] == "Mobile", f"Expected Mobile, got {res['device']}"
    assert res["browser"] == "Safari", f"Expected Safari, got {res['browser']}"

    # Android Mobile Chrome
    ua_android = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.6261.94 Mobile Safari/537.36"
    res = parse_user_agent(ua_android)
    assert res["os"] == "Android", f"Expected Android, got {res['os']}"
    assert res["device"] == "Mobile", f"Expected Mobile, got {res['device']}"
    assert res["browser"] == "Chrome", f"Expected Chrome, got {res['browser']}"

    # macOS Safari
    ua_mac = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15"
    res = parse_user_agent(ua_mac)
    assert res["os"] == "macOS", f"Expected macOS, got {res['os']}"
    assert res["device"] == "Desktop", f"Expected Desktop, got {res['device']}"
    assert res["browser"] == "Safari", f"Expected Safari, got {res['browser']}"

    # Linux Firefox
    ua_linux = "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0"
    res = parse_user_agent(ua_linux)
    assert res["os"] == "Linux", f"Expected Linux, got {res['os']}"
    assert res["device"] == "Desktop", f"Expected Desktop, got {res['device']}"
    assert res["browser"] == "Firefox", f"Expected Firefox, got {res['browser']}"

    print("  [PASS] User-Agent parsing works accurately across OS and device types.")


def test_analytics_collection_and_metrics():
    print("[2] Testing Analytics collection endpoint...")

    # Clear previous test data first via direct db
    from analytics import db
    db.clear_data()

    # Visitor 1: Windows Desktop
    payload_win = {
        "visitor_id": "v_test_win",
        "session_id": "s_test_win_1",
        "event_type": "pageview",
        "path": "/",
        "os": "Windows",
        "device_type": "Desktop",
        "browser": "Chrome",
        "screen_res": "1920x1080"
    }
    r1 = client.post("/api/analytics/collect", json=payload_win)
    assert r1.status_code == 200, f"Failed collect: {r1.text}"

    # Visitor 2: Android Mobile
    payload_android = {
        "visitor_id": "v_test_android",
        "session_id": "s_test_android_1",
        "event_type": "pageview",
        "path": "/",
        "os": "Android",
        "device_type": "Mobile",
        "browser": "Chrome",
        "screen_res": "390x844"
    }
    r2 = client.post("/api/analytics/collect", json=payload_android)
    assert r2.status_code == 200

    # Visitor 3: iOS iPhone
    payload_ios = {
        "visitor_id": "v_test_ios",
        "session_id": "s_test_ios_1",
        "event_type": "pageview",
        "path": "/",
        "os": "iOS",
        "device_type": "Mobile",
        "browser": "Safari",
        "screen_res": "393x852"
    }
    r3 = client.post("/api/analytics/collect", json=payload_ios)
    assert r3.status_code == 200

    # Visitor 1 sends heartbeat
    payload_heartbeat = {
        "visitor_id": "v_test_win",
        "session_id": "s_test_win_1",
        "event_type": "heartbeat",
        "path": "/",
        "os": "Windows",
        "device_type": "Desktop",
        "browser": "Chrome"
    }
    client.post("/api/analytics/collect", json=payload_heartbeat)

    # Chat event
    db.record_chat_event(visitor_id="v_test_win", session_id="s_test_win_1", ip="127.0.0.1")

    # Verify summary metrics
    summary = db.get_summary()
    assert summary["total_visitors"] == 3, f"Expected 3 visitors, got {summary['total_visitors']}"
    assert summary["active_users"] == 3, f"Expected 3 active users, got {summary['active_users']}"
    assert summary["total_sessions"] == 3
    assert summary["total_chats"] == 1

    os_dict = {item["name"]: item for item in summary["os_breakdown"]}
    assert "Windows" in os_dict, "Windows missing from OS breakdown"
    assert "Android" in os_dict, "Android missing from OS breakdown"
    assert "iOS" in os_dict, "iOS missing from OS breakdown"
    assert os_dict["Windows"]["count"] == 1
    assert round(os_dict["Windows"]["percentage"]) == 33

    print(f"  [PASS] Analytics summary: {summary['active_users']} active, {summary['total_visitors']} visitors, OS distribution: {list(os_dict.keys())}")
    
    # Clean up test rows so database remains clean for real visitors
    db.clear_data()


def test_admin_access_control():
    print("[3] Testing Admin access control and security...")

    # A. Unauthenticated access to /admin must redirect to /admin/login
    resp = client.get("/admin", follow_redirects=False)
    assert resp.status_code in [302, 303, 307], f"Expected redirect, got {resp.status_code}"
    assert "/admin/login" in resp.headers.get("location", "")
    print("  [PASS] Unauthenticated access to /admin is blocked and redirected to /admin/login.")

    # B. Direct access to /api/admin/analytics must return 401 Unauthorized
    resp_api = client.get("/api/admin/analytics")
    assert resp_api.status_code == 401, f"Expected 401 Unauthorized, got {resp_api.status_code}"
    print("  [PASS] /api/admin/analytics is strictly protected (returned 401 Unauthorized).")

    # C. Login with wrong password must be rejected
    resp_bad_login = client.post("/admin/login", json={"password": "wrong_password_123"})
    assert resp_bad_login.status_code == 401
    assert resp_bad_login.json()["success"] is False
    print("  [PASS] Invalid password rejected.")

    # D. Login with correct password must succeed and set session cookie
    resp_good_login = client.post("/admin/login", json={"password": ADMIN_PASSWORD})
    assert resp_good_login.status_code == 200
    assert resp_good_login.json()["success"] is True
    assert COOKIE_NAME in resp_good_login.cookies
    token = resp_good_login.cookies[COOKIE_NAME]
    print("  [PASS] Correct password authenticated, issued secure session cookie.")

    # E. Accessing /admin with session cookie serves dashboard
    resp_admin = client.get("/admin", cookies={COOKIE_NAME: token})
    assert resp_admin.status_code == 200
    assert "BizInsights AI — Admin Analytics Dashboard" in resp_admin.text
    print("  [PASS] Authenticated admin successfully access /admin dashboard.")

    # F. Accessing /api/admin/analytics with session cookie returns data
    resp_admin_data = client.get("/api/admin/analytics", cookies={COOKIE_NAME: token})
    assert resp_admin_data.status_code == 200
    data = resp_admin_data.json()
    assert "os_breakdown" in data
    assert "device_breakdown" in data
    assert "browser_breakdown" in data
    assert "active_users" in data
    print("  [PASS] Authenticated admin successfully retrieved metrics from /api/admin/analytics.")

    # G. Logout revokes token
    resp_logout = client.post("/admin/logout", cookies={COOKIE_NAME: token})
    assert resp_logout.status_code == 200

    # H. Post-logout access to /api/admin/analytics fails with 401
    resp_after_logout = client.get("/api/admin/analytics", cookies={COOKIE_NAME: token})
    assert resp_after_logout.status_code == 401
    print("  [PASS] Logout successfully revoked admin session.")


if __name__ == "__main__":
    test_user_agent_parser()
    test_analytics_collection_and_metrics()
    test_admin_access_control()
    print("\n=======================================================")
    print("  ALL TESTS PASSED SUCCESSFULLY! ANALYTICS ARE SECURE.")
    print("=======================================================")
