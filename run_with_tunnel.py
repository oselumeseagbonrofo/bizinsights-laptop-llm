"""
Launch BizInsights Local Server with an encrypted public Cloudflare Tunnel.
Allows external access from any mobile phone or device over the internet
without needing port forwarding or cloud accounts.
"""
import sys
import os
import re
import time
import atexit
import subprocess
from pathlib import Path
import qrcode

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
CLOUDFLARED_BIN = ROOT_DIR / "bin" / "cloudflared.exe"
PYTHON_BIN = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

server_proc = None
tunnel_proc = None


def cleanup():
    global server_proc, tunnel_proc
    if tunnel_proc and tunnel_proc.poll() is None:
        try:
            tunnel_proc.terminate()
        except Exception:
            pass
    if server_proc and server_proc.poll() is None:
        try:
            server_proc.terminate()
        except Exception:
            pass


atexit.register(cleanup)


def main():
    global server_proc, tunnel_proc

    if not CLOUDFLARED_BIN.is_file():
        print(f"[ERROR] cloudflared.exe not found at {CLOUDFLARED_BIN}")
        sys.exit(1)

    print("=" * 64)
    print("  Starting BizInsights Assistant + Public Internet Tunnel")
    print("=" * 64)

    # 1. Start local server
    print("[INFO] Launching local model server...")
    server_proc = subprocess.Popen(
        [str(PYTHON_BIN), "-u", "server.py"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # Wait for server to announce listening
    server_ready = False
    for line in iter(server_proc.stdout.readline, ""):
        print("  " + line.rstrip())
        if "Model engine successfully loaded" in line or "Host PC Access" in line:
            server_ready = True
            break
        if server_proc.poll() is not None:
            print("[ERROR] Server failed to start.")
            sys.exit(1)

    # 2. Launch cloudflared tunnel
    print("\n[INFO] Creating secure public internet tunnel via Cloudflare...")
    tunnel_cmd = [
        str(CLOUDFLARED_BIN),
        "tunnel",
        "--url", "http://127.0.0.1:8000",
        "--no-autoupdate"
    ]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for line in iter(tunnel_proc.stdout.readline, ""):
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break
        if tunnel_proc.poll() is not None:
            print("[ERROR] Tunnel failed to start.")
            sys.exit(1)

    if not public_url:
        print("[ERROR] Could not extract public tunnel URL.")
        sys.exit(1)

    admin_pass = os.environ.get("ADMIN_PASSWORD", "bizinsights2026!")
    print("\n" + "=" * 64)
    print("  🚀 PUBLIC INTERNET ACCESS IS LIVE!")
    print("=" * 64)
    print(f"  🌐 Public HTTPS URL : {public_url}")
    print(f"  💻 Local PC URL     : http://127.0.0.1:8000")
    print(f"  🔒 Admin Dashboard  : {public_url}/admin")
    print(f"                        (or http://127.0.0.1:8000/admin)")
    print(f"  🔑 Admin Password   : {admin_pass}")
    print("=" * 64)
    print("  📲 Scan with ANY mobile phone camera on cellular data or any Wi-Fi:")
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(public_url)
        qr.print_ascii(invert=True)
    except Exception:
        pass
    print("=" * 64)
    print("  Anyone in the world can now access your model via this URL.")
    print("  Only you can access the admin dashboard using your admin password.")
    print("  Press Ctrl+C to stop.\n")

    try:
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down tunnel and server...")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
