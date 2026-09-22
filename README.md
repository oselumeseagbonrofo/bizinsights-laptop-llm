

# BizInsight - Offline SME Customer Support Assistant
This is team BizInsights official repository for the Africa Deep Tech Challenge 2026 Laptop LLM track.

---

## Instructions: Running the Server

### 1. Prerequisites
- **Python:** Version 3.10 or higher.
- **llama.cpp Binary:** `llama-server` (or `llama-server.exe` on Windows) installed and available on your system `PATH`, or specify its path via the `LLAMA_SERVER_PATH` environment variable.

### 2. Install Dependencies
Install all required Python dependencies:
```bash
pip install -r requirements.txt
```

### 3. Download the Quantized Model
Download the activation-aware quantized model weights (`Q4_K_M` GGUF, ~1.3 GB) using the provided download script:
```bash
bash download_model.sh
```
*(On Windows without bash, you can download the model URL specified in `download_model.sh` directly to `model/qwen3.5-2B-bitext-imatrix-Q4_K_M.gguf`)*.

### 4. Start the Application Server
Run `server.py` to start the backend and automatically launch the underlying `llama-server` inference engine:
```bash
python server.py --port 8000 --threads 4
```

#### Server Options:
- `--host`: Network interface to bind to (default: `0.0.0.0` for local network and mobile access).
- `--port`: Web server port (default: `8000`).
- `--llama-port`: Internal port for `llama-server` (default: `8081`).
- `--threads`: Number of CPU threads to allocate for inference (default: `4`).
- `--ctx-size`: Context window size in tokens (default: `2048`).
- `--model`: Path to GGUF model file (default: `model/qwen3.5-2B-bitext-imatrix-Q4_K_M.gguf`).
- `--reasoning`: Reasoning mode: `off`, `on`, or `auto` (default: `off` for rapid support responses).

---

## Accessing the Web Interface & Features

- **Chat Interface:** Open [http://localhost:8000](http://localhost:8000) in your browser. When launched, the terminal also displays an ASCII QR code for connecting directly from a mobile device on the same local network.
- **Admin & Analytics Portal:** Visit [http://localhost:8000/admin](http://localhost:8000/admin) to view real-time inference latency, generation speed (tokens/sec), and token usage statistics.

---

## Shop Knowledge (Offline RAG)

BizInsights includes a 100% offline Retrieval-Augmented Generation (RAG) system:
1. Click the **Book icon** in the chat header to open the **Shop Knowledge** panel.
2. Upload business documents in `.pdf`, `.docx`, `.txt`, `.md`, or `.csv` format (up to 10 MB per file).
3. A sample African retail business draft document is included in [`sample_business/Beauty Biz Business Document Draft.pdf`](sample_business/Beauty%20Biz%20Business%20Document%20Draft.pdf) for immediate testing.
4. Enable the **"Use shop docs in answers"** toggle to ground responses with citations to uploaded files.

---

## Running Verification Tests

Run the test suites for RAG indexing and analytics tracking:
```bash
python test_rag.py
python test_analytics.py
```

---

These additional files were added after the deadline to enable running of the system UI.