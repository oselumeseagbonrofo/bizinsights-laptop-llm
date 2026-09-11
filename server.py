import os
import sys

# Ensure UTF-8 output encoding on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import time
import socket
import atexit
import shutil
import io
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

import httpx
import qrcode
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from analytics import router as analytics_router, db as analytics_db, ADMIN_PASSWORD, get_client_ip

# Configuration defaults
DEFAULT_PORT = 8000
DEFAULT_LLAMA_PORT = 8081
ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "qwen3.5-2B-lora-Instruct-Q8_0.gguf"
STATIC_DIR = ROOT_DIR / "static"

DEFAULT_SYSTEM_PROMPT = (
    "You are BizInsights AI, an intelligent customer support and operations assistant "
    "tailored for retail and enterprise SMEs. You analyze customer tickets, identify root causes, "
    "evaluate customer sentiment, categorize operational issues, and draft polite, concise, and "
    "empathetic resolution emails."
)

app = FastAPI(title="BizInsights LLM Assistant")
app.include_router(analytics_router)
llama_process: Optional[subprocess.Popen] = None
LLAMA_PORT = DEFAULT_LLAMA_PORT


def get_local_ip() -> str:
    """Detect local network IPv4 address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 53))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_llama_server_binary() -> Optional[Path]:
    """Find llama-server binary on the system."""
    env_path = os.environ.get("LLAMA_SERVER_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # Common Windows locations (including Docker inference runtime)
    candidates = [
        Path(r"C:\Users\ZBOOK STUDIO G5\.docker\bin\inference\llama-server.exe"),
        ROOT_DIR / "llama-server.exe",
        ROOT_DIR / "bin" / "llama-server.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    which_path = shutil.which("llama-server.exe") or shutil.which("llama-server")
    if which_path:
        return Path(which_path)

    return None


def print_banner(ip: str, port: int):
    """Print connection URLs and ASCII QR code to terminal."""
    target_url = PUBLIC_DOMAIN if PUBLIC_DOMAIN else f"http://{ip}:{port}"
    local_url = f"http://127.0.0.1:{port}"
    admin_url = f"{target_url}/admin"

    print("\n" + "=" * 64)
    print("  BizInsights LLM -- Mobile & Network Server")
    print("=" * 64)
    print(f"  Public Domain  : {target_url}")
    print(f"  Host PC Access : {local_url}")
    print(f"  Admin Portal   : {admin_url}")
    print(f"  Admin Password : {ADMIN_PASSWORD}")
    print("=" * 64)
    print("  Scan with mobile phone camera:")
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(target_url)
        qr.print_ascii(invert=True)
    except Exception as e:
        print(f"  (QR code render skipped: {e})")
    print("=" * 64)
    print("  Keep this window open to continue running the model.")
    print("  Press Ctrl+C to stop the server.\n")


def stop_llama_server():
    """Terminate the background llama-server process cleanly."""
    global llama_process
    if llama_process and llama_process.poll() is None:
        print("\nStopping llama-server background engine...")
        try:
            llama_process.terminate()
            llama_process.wait(timeout=3)
        except Exception:
            llama_process.kill()
        llama_process = None


atexit.register(stop_llama_server)


def cleanup_existing_llama_ports(port: int):
    """Clean up any orphan llama-server that might be occupying the port."""
    try:
        with httpx.Client(timeout=0.5) as client:
            res = client.get(f"http://127.0.0.1:{port}/health")
            if res.status_code == 200:
                print(f"[INFO] Existing llama-server active on port {port}.")
                return True
    except Exception:
        pass
    return False


def start_llama_server(model_path: Path, port: int, threads: int = 4, ctx_size: int = 2048, reasoning: str = "off"):
    """Start llama-server in the background and wait for it to be healthy."""
    global llama_process, LLAMA_PORT
    LLAMA_PORT = port

    # If already running and healthy, reuse it
    if cleanup_existing_llama_ports(port):
        return

    binary = find_llama_server_binary()
    if not binary:
        raise RuntimeError(
            "llama-server.exe could not be found. Please ensure llama-server is installed or set LLAMA_SERVER_PATH."
        )

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    cmd = [
        str(binary),
        "-m", str(model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", str(ctx_size),
        "-t", str(threads),
        "--reasoning", reasoning
    ]

    print(f"[INFO] Starting model engine: {binary.name} with {model_path.name}...")
    log_file = ROOT_DIR / "llama_server.log"
    log_handle = open(log_file, "w", encoding="utf-8")

    llama_process = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT_DIR)
    )

    print("[INFO] Waiting for model to load into memory (typically 5-10s)...")
    health_url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    while time.time() - start_time < 45:
        if llama_process.poll() is not None:
            raise RuntimeError(f"llama-server exited unexpectedly. Check {log_file} for details.")
        try:
            with httpx.Client(timeout=1.0) as client:
                res = client.get(health_url)
                if res.status_code == 200:
                    print("[OK] Model engine successfully loaded and ready for inference!")
                    return
        except (httpx.RequestError, httpx.HTTPStatusError):
            time.sleep(0.8)

    raise TimeoutError("Timed out waiting for llama-server to become ready.")


# Pydantic models for chat API
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = True


@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """Serve the mobile-friendly web UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>BizInsights AI</h1><p>UI loading error: index.html not found.</p>")


PUBLIC_DOMAIN = os.environ.get("PUBLIC_DOMAIN", "https://bizinsight.oselumeseagbonrofo.dev").rstrip("/")


@app.get("/api/info")
async def get_info():
    """Return system & model status metadata."""
    local_ip = get_local_ip()
    is_ready = (llama_process and llama_process.poll() is None) or cleanup_existing_llama_ports(LLAMA_PORT)
    public_url = PUBLIC_DOMAIN if PUBLIC_DOMAIN else f"http://{local_ip}:{DEFAULT_PORT}"
    return {
        "status": "ready" if is_ready else "offline",
        "ip": local_ip,
        "port": DEFAULT_PORT,
        "url": public_url,
        "model_name": "qwen3.5-2B-lora-Instruct-Q8_0",
        "parameters": "2B (Q8_0 Quantized GGUF)",
        "team_id": "bizinsight-nfmgvw",
        "discipline": "Retail SME Customer Support & Complaint Analytics"
    }


@app.get("/api/qr.png")
async def get_qr_png():
    """Generate and return a PNG QR code of the mobile connection URL."""
    target_url = PUBLIC_DOMAIN if PUBLIC_DOMAIN else f"http://{get_local_ip()}:{DEFAULT_PORT}"
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(target_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#0f172a", back_color="#ffffff")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest, request: Request):
    """Stream chat completions from llama-server to the client via SSE."""
    # Record chat interaction in analytics with proper client metadata
    client_ip = get_client_ip(request)
    ua = request.headers.get("user-agent", "")
    visitor_id = request.headers.get("x-visitor-id")
    session_id = request.headers.get("x-session-id")
    analytics_db.record_chat_event(
        visitor_id=visitor_id,
        session_id=session_id,
        ip=client_ip,
        ua_string=ua
    )

    formatted_messages = []
    has_system = any(m.role == "system" for m in req.messages)
    if not has_system:
        formatted_messages.append({"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

    for m in req.messages:
        formatted_messages.append({"role": m.role, "content": m.content})

    payload = {
        "messages": formatted_messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "stream": True,
    }

    async def event_generator():
        client = httpx.AsyncClient(timeout=120.0)
        try:
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {{\"error\": \"Inference failed: {error_text.decode('utf-8')}\"}}\n\n"
                    return

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield f"{line}\n\n"
                        if line.strip() == "data: [DONE]":
                            break
        except Exception as e:
            yield f"data: {{\"error\": \"Streaming exception: {str(e)}\"}}\n\n"
        finally:
            await client.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    global DEFAULT_PORT
    parser = argparse.ArgumentParser(description="BizInsights Local Network LLM Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind to (0.0.0.0 for LAN)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--llama-port", type=int, default=8081, help="Internal port for llama-server")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context window size in tokens")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to GGUF model")
    parser.add_argument("--reasoning", default="off", choices=["on", "off", "auto"], help="Reasoning mode (default: off for faster support responses)")
    args = parser.parse_args()

    DEFAULT_PORT = args.port
    model_path = Path(args.model)

    # Start the underlying llama-server
    start_llama_server(
        model_path=model_path,
        port=args.llama_port,
        threads=args.threads,
        ctx_size=args.ctx_size,
        reasoning=args.reasoning
    )

    # Print connection banner & QR code
    local_ip = get_local_ip()
    print_banner(local_ip, args.port)

    import uvicorn
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        stop_llama_server()


if __name__ == "__main__":
    main()
