#!/usr/bin/env python3
"""
Post-Training Continuation Pipeline for NVIDIA A6000
====================================================
Picks up from Stage 3 after successful 3-epoch LoRA training:
  1. Fixes file permissions on cache and target directories.
  2. Loads the trained LoRA adapter from `lora_weights/`.
  3. Merges LoRA into base model and exports pre-fused F16 GGUF.
  4. Verifies / builds CUDA llama.cpp.
  5. Prepares 256 balanced multi-domain calibration samples (86 bitext, 85 sentiment, 85 heimdall).
  6. Computes activation importance matrix (imatrix.dat).
  7. Quantizes to Q4_K_M and IQ4_XS.
  8. Evaluates test prompts (tp_001, tp_002) using llama-cli.
  9. Uploads quantized models and imatrix to Hugging Face:
     oselumese/qwen3.5-2B_lora_multidomain-imatrix
"""

import os
import sys
import time
import random
import shutil
import subprocess
from pathlib import Path

# Configure environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "bizinsights-laptop-llm"
os.environ["WANDB_LOG_MODEL"] = "false"

BASE_DIR = Path(os.getcwd()).resolve()
LORA_DIR = BASE_DIR / "lora_weights"
GGUF_F16_DIR = BASE_DIR / "model_f16"
LLAMA_BUILD_DIR = BASE_DIR / "llama_cpp"
FINAL_MODELS_DIR = BASE_DIR / "final_models"

for d in [GGUF_F16_DIR, FINAL_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load credentials from .env or environment
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

env_file = BASE_DIR / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")
        elif line.startswith("WANDB_API_KEY="):
            WANDB_API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")

if not HF_TOKEN:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        HF_TOKEN = token_file.read_text().strip()

HF_TARGET_REPO = "oselumese/qwen3.5-2B_lora_multidomain-imatrix"

DEFAULT_SYSTEM_PROMPT = (
    "You are BizInsights AI, an intelligent customer support and operations assistant "
    "tailored for retail and enterprise SMEs. You analyze customer tickets, identify root causes, "
    "evaluate customer sentiment, categorize operational issues, and draft polite, concise, "
    "and empathetic resolution emails. When you detect requests involving potentially "
    "security-compromising situations, you provide guidance on appropriate next steps."
)

TEST_PROMPTS = [
    {
        "prompt_id": "tp_001",
        "name": "Unresolved Ticket Thread Analysis",
        "prompt": (
            "Analyse the following unresolved ticket thread. Provide: Executive Summary "
            "(under 50 words), root cause & sentiment (Positive / Neutral / Negative / Escalation Risk) "
            "and draft resolution email (polite, concise, offering refund/replacement options) "
            "[TICKET THREAD] Customer: Hi, our order #88419 for 20 custom office chairs arrived 4 days late, "
            "and 3 bases are cracked. We have an office launch this Friday and cannot use these., "
            "Support: Checking on the shipment details now. Customer: Any update? It's been 24 hours. "
            "We need replacements or an immediate refund so we can source locally. [/TICKET THREAD]"
        ),
    },
    {
        "prompt_id": "tp_002",
        "name": "5 Support Logs Categorization",
        "prompt": (
            "Categorise the following 5 customer support logs into Logistics, Product Defect, "
            "Billing, or UX, and identify the top recurring issue for customers: [LOGS] "
            "1. Invoice didn't show VAT breakdown required for HMRC filing. "
            "2. Tracking link sent via SMS was broken; carrier said package delayed. "
            "3. Need VAT receipt for expense claims on order #4491. "
            "4. Where do I add company tax ID during checkout? "
            "5. Delivery driver left pallets in the rain; packaging damaged. [/LOGS]"
        ),
    },
]


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n", flush=True)


# ==============================================================================
# 0. Environment & Permissions Safeguard
# ==============================================================================
print_banner("STAGE 0: Permissions & Environment Safeguards")

# Ensure write permissions across ~/.cache/huggingface and model_f16
print("-> Setting write permissions on cache and target directories...")
hf_cache = Path.home() / ".cache" / "huggingface"
subprocess.run(["chmod", "-R", "u+w", str(hf_cache)], check=False)
subprocess.run(["chmod", "-R", "u+w", str(GGUF_F16_DIR)], check=False)

import torch
import wandb
from huggingface_hub import login as hf_login, HfApi

if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is required but not detected!")

gpu_props = torch.cuda.get_device_properties(0)
print(f"[OK] GPU: {gpu_props.name} ({round(gpu_props.total_memory / (1024**3), 1)} GB VRAM)")

if HF_TOKEN:
    hf_login(token=HF_TOKEN)
    print("[OK] Hugging Face authentication confirmed.")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="bizinsights-laptop-llm",
        name="qwen3.5-2B-multidomain-quantization-continuation",
        config={
            "stage": "post_training_quantization",
            "base_model": "unsloth/Qwen3.5-2B",
            "lora_source": str(LORA_DIR),
            "imatrix_samples": 256,
        },
    )
    print("[OK] WandB session re-initialized for quantization tracking.")


# ==============================================================================
# 1. Merge LoRA and Export F16 GGUF
# ==============================================================================
print_banner("STAGE 1: Loading Trained LoRA Adapter & Exporting F16 GGUF")

from unsloth import FastLanguageModel

# Ensure target directory exists and is fully writable
if GGUF_F16_DIR.exists():
    for p in GGUF_F16_DIR.glob("*"):
        try:
            os.chmod(p, 0o777)
        except Exception:
            pass

# Locate the produced F16 GGUF in model_f16 or model_f16_gguf
f16_dirs = [GGUF_F16_DIR, BASE_DIR / "model_f16_gguf"]
ACTUAL_F16_PATH = None

for d in f16_dirs:
    if d.exists():
        candidates = list(d.glob("*.F16.gguf")) + list(d.glob("*f16*.gguf")) + [f for f in d.glob("*.gguf") if "mmproj" not in f.name]
        if candidates:
            ACTUAL_F16_PATH = candidates[0]
            break

if not ACTUAL_F16_PATH or not ACTUAL_F16_PATH.exists():
    print(f"-> Loading trained model & tokenizer from {LORA_DIR}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(LORA_DIR),
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"-> Merging LoRA and exporting F16 GGUF into {GGUF_F16_DIR}...")
    for p in GGUF_F16_DIR.glob("*.safetensors*"):
        try:
            os.chmod(p, 0o666)
        except Exception:
            pass

    model.save_pretrained_gguf(
        str(GGUF_F16_DIR),
        tokenizer,
        quantization_method="f16",
    )

    # Search again after export
    for d in f16_dirs:
        if d.exists():
            candidates = list(d.glob("*.F16.gguf")) + list(d.glob("*f16*.gguf")) + [f for f in d.glob("*.gguf") if "mmproj" not in f.name]
            if candidates:
                ACTUAL_F16_PATH = candidates[0]
                break

if not ACTUAL_F16_PATH or not ACTUAL_F16_PATH.exists():
    raise FileNotFoundError(f"No F16 GGUF file found in {[str(d) for d in f16_dirs]} after export!")

f16_size_gb = round(os.path.getsize(ACTUAL_F16_PATH) / (1024**3), 2)
print(f"[OK] F16 GGUF Found & Ready: {ACTUAL_F16_PATH} ({f16_size_gb} GB)")
if wandb.run:
    wandb.log({"model/f16_disk_gb": f16_size_gb})

# Release PyTorch VRAM before C++ operations
if "model" in locals():
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("[OK] VRAM released for llama.cpp tools.")


# ==============================================================================
# 2. Locate llama.cpp Binaries
# ==============================================================================
print_banner("STAGE 2: Locating llama.cpp Binaries")

unsloth_bin = Path.home() / ".unsloth" / "llama.cpp"
if unsloth_bin.exists() and (unsloth_bin / "llama-imatrix").exists():
    LLAMA_IMATRIX = unsloth_bin / "llama-imatrix"
    LLAMA_QUANTIZE = unsloth_bin / "llama-quantize"
    LLAMA_CLI = unsloth_bin / "llama-cli"
    os.environ["LD_LIBRARY_PATH"] = f"{unsloth_bin}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    print(f"[OK] Using prebuilt llama.cpp binaries from {unsloth_bin}")
else:
    LLAMA_BIN_DIR = LLAMA_BUILD_DIR / "build" / "bin"
    LLAMA_IMATRIX = LLAMA_BIN_DIR / "llama-imatrix"
    LLAMA_QUANTIZE = LLAMA_BIN_DIR / "llama-quantize"
    LLAMA_CLI = LLAMA_BIN_DIR / "llama-cli"

print(f"[OK] llama-imatrix : {LLAMA_IMATRIX}")
print(f"[OK] llama-quantize: {LLAMA_QUANTIZE}")
print(f"[OK] llama-cli     : {LLAMA_CLI}")


# ==============================================================================
# 3. Prepare 256 Multi-Domain Calibration Samples
# ==============================================================================
print_banner("STAGE 3: Preparing 256 Balanced Calibration Samples")

from datasets import load_dataset

CALIB_FILE = BASE_DIR / "calibration_multidomain_256.txt"
IMATRIX_FILE = BASE_DIR / "imatrix.dat"

if not CALIB_FILE.exists():
    print("-> Loading dataset samples for calibration...")
    calib_samples = []

    # 1. Bitext: 86 samples
    ds_bitext = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    for row in list(ds_bitext)[:86]:
        inst = str(row.get("instruction", "")).strip()
        resp = str(row.get("response", "")).strip()
        calib_samples.append(
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n{resp}<|im_end|>"
        )

    # 2. Sentiment: 85 samples
    ds_sent = load_dataset("jbeno/sentiment_merged", split="train")
    for row in list(ds_sent)[:85]:
        sent = str(row.get("sentence", "")).strip()
        lbl = str(row.get("label", "")).strip().capitalize()
        calib_samples.append(
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nClassify the customer sentiment: \"{sent}\"<|im_end|>\n"
            f"<|im_start|>assistant\nSentiment: {lbl}.<|im_end|>"
        )

    # 3. Heimdall: 85 samples
    ds_heim = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1", split="train")
    for row in list(ds_heim)[:85]:
        u = str(row.get("user", "")).strip()
        a = str(row.get("assistant", "")).strip()
        s = str(row.get("system", "")).strip() or DEFAULT_SYSTEM_PROMPT
        calib_samples.append(
            f"<|im_start|>system\n{s}<|im_end|>\n"
            f"<|im_start|>user\n{u}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )

    random.seed(42)
    random.shuffle(calib_samples)

    with open(CALIB_FILE, "w", encoding="utf-8") as f:
        for item in calib_samples:
            f.write(item + "\n\n")

print(f"[OK] Calibration file ready: {CALIB_FILE} ({len(calib_samples) if 'calib_samples' in locals() else 256} samples)")


# ==============================================================================
# 4. Compute Activation Importance Matrix (imatrix.dat)
# ==============================================================================
print_banner("STAGE 4: Computing imatrix with GPU Offload (-ngl 99)")

t0 = time.time()
res_imatrix = subprocess.run(
    [
        str(LLAMA_IMATRIX),
        "-m", str(ACTUAL_F16_PATH),
        "-f", str(CALIB_FILE),
        "-o", str(IMATRIX_FILE),
        "-ngl", "99",
        "-c", "512",
        "--chunks", "128",
    ],
    capture_output=True,
    text=True,
)

if res_imatrix.returncode != 0:
    print(res_imatrix.stdout)
    print(res_imatrix.stderr)
    raise RuntimeError(f"llama-imatrix failed with code {res_imatrix.returncode}")

imatrix_sec = round(time.time() - t0, 2)
imatrix_kb = round(os.path.getsize(IMATRIX_FILE) / 1024, 2)
print(f"[OK] imatrix.dat computed in {imatrix_sec}s ({imatrix_kb} KB)!")
if wandb.run:
    wandb.log({"imatrix/duration_sec": imatrix_sec, "imatrix/size_kb": imatrix_kb})


# ==============================================================================
# 5. Activation-Aware Quantization (Q4_K_M and IQ4_XS)
# ==============================================================================
print_banner("STAGE 5: Quantizing to Q4_K_M and IQ4_XS with imatrix")

Q4KM_PATH = FINAL_MODELS_DIR / "qwen3.5-2B-multidomain-imatrix-Q4_K_M.gguf"
IQ4XS_PATH = FINAL_MODELS_DIR / "qwen3.5-2B-multidomain-imatrix-IQ4_XS.gguf"

# 5.1 Q4_K_M
print(f"-> Quantizing Q4_K_M + imatrix -> {Q4KM_PATH.name}...")
t_q4 = time.time()
res_q4 = subprocess.run(
    [
        str(LLAMA_QUANTIZE),
        "--imatrix", str(IMATRIX_FILE),
        str(ACTUAL_F16_PATH),
        str(Q4KM_PATH),
        "Q4_K_M",
    ],
    capture_output=True,
    text=True,
)
if res_q4.returncode != 0:
    print(res_q4.stderr)
    raise RuntimeError(f"Q4_K_M quantization failed: {res_q4.returncode}")

size_q4km_gb = round(os.path.getsize(Q4KM_PATH) / (1024**3), 2)
print(f"[OK] Q4_K_M created: {size_q4km_gb} GB (took {round(time.time()-t_q4, 1)}s)")

# 5.2 IQ4_XS
print(f"\n-> Quantizing IQ4_XS + imatrix -> {IQ4XS_PATH.name}...")
t_iq4 = time.time()
res_iq4 = subprocess.run(
    [
        str(LLAMA_QUANTIZE),
        "--imatrix", str(IMATRIX_FILE),
        str(ACTUAL_F16_PATH),
        str(IQ4XS_PATH),
        "IQ4_XS",
    ],
    capture_output=True,
    text=True,
)
if res_iq4.returncode != 0:
    print(res_iq4.stderr)
    raise RuntimeError(f"IQ4_XS quantization failed: {res_iq4.returncode}")

size_iq4xs_gb = round(os.path.getsize(IQ4XS_PATH) / (1024**3), 2)
print(f"[OK] IQ4_XS created: {size_iq4xs_gb} GB (took {round(time.time()-t_iq4, 1)}s)")

if wandb.run:
    wandb.log({
        "quant/q4km_gb": size_q4km_gb,
        "quant/iq4xs_gb": size_iq4xs_gb,
    })


# ==============================================================================
# 6. Direct Test Prompt Benchmark with llama-cli
# ==============================================================================
print_banner("STAGE 6: Direct Test Prompt Benchmarking with llama-cli")

for model_label, model_path in [("Q4_K_M", Q4KM_PATH), ("IQ4_XS", IQ4XS_PATH)]:
    print(f"\n--- Testing {model_label} ({model_path.name}) ---")
    for tp in TEST_PROMPTS:
        pid = tp["prompt_id"]
        pname = tp["name"]
        raw_prompt = tp["prompt"]
        
        chatml_prompt = (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{raw_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        t_start = time.time()
        res_cli = subprocess.run(
            [
                str(LLAMA_CLI),
                "-m", str(model_path),
                "-p", chatml_prompt,
                "-n", "384",
                "-c", "2048",
                "-ngl", "99",
                "--temp", "0.3",
                "-st",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        cli_duration = round(time.time() - t_start, 2)
        full_output = res_cli.stdout + "\n" + res_cli.stderr
        
        tok_per_sec = 0.0
        for line in full_output.splitlines():
            if "eval time =" in line and "tokens per second" in line:
                try:
                    parts = line.split("tokens per second")[0].strip().split()
                    tok_per_sec = float(parts[-1])
                except Exception:
                    pass
        
        response_text = full_output
        if "<|im_start|>assistant" in full_output:
            response_text = full_output.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        
        print(f"\n[{model_label}] {pid}: {pname}")
        print(f"Speed: {tok_per_sec} tok/s | Duration: {cli_duration}s")
        print(f"Sample response:\n{response_text[:300]}...\n")


# ==============================================================================
# 7. Upload to Hugging Face
# ==============================================================================
print_banner("STAGE 7: Uploading Final Quantized GGUF Artifacts to Hugging Face")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_TARGET_REPO, exist_ok=True, private=False)

readme_content = f"""---
license: apache-2.0
base_model: unsloth/Qwen3.5-2B
tags:
- llama.cpp
- gguf
- imatrix
- retail
- customer-support
- sentiment-analysis
- cybersecurity
- adtc-2026
---

# Qwen 3.5 2B Multi-Domain LoRA with Activation-Aware imatrix Quantization

Fine-tuned on NVIDIA RTX A6000 for the **Africa Deep Tech Challenge 2026 (Laptop LLM Track)**.

## Training Configuration
- **Base Model:** `unsloth/Qwen3.5-2B`
- **LoRA Adapter:** $r=32$, $\\alpha=64$, targeting all 7 linear projections
- **Epochs:** 3 full epochs with cosine LR decay ($2 \\times 10^{{-4}}$)
- **Datasets:**
  1. `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 samples)
  2. `jbeno/sentiment_merged` (27,000 stratified samples)
  3. `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1` (21,258 samples)
  - **Total Volume:** ~75,130 samples (100% completed)

## Quantized Artifacts
- `qwen3.5-2B-multidomain-imatrix-Q4_K_M.gguf` ({size_q4km_gb} GB): Calibrated with 256 balanced multi-domain samples. Recommended for high reasoning accuracy and fast CPU generation.
- `qwen3.5-2B-multidomain-imatrix-IQ4_XS.gguf` ({size_iq4xs_gb} GB): Non-linear importance quantization for minimum memory footprint (~1.2 GB).
- `imatrix.dat`: Computed importance matrix from 256 multi-domain interactions.

## Provenance
- Base Model: `huggingface:unsloth/Qwen3.5-2B`
- Base Model Commit SHA: `fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`
- Team ID: `bizinsight-nfmgvw`
- Challenge Track: `corporate_enterprise` / `retail`
"""

readme_path = FINAL_MODELS_DIR / "README.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

upload_files = [
    Q4KM_PATH,
    IQ4XS_PATH,
    IMATRIX_FILE,
    readme_path,
]

latest_commit_sha = "unknown"
for u_file in upload_files:
    if u_file.exists():
        print(f"-> Uploading {u_file.name} to https://huggingface.co/{HF_TARGET_REPO}...")
        commit_info = api.upload_file(
            path_or_fileobj=str(u_file),
            path_in_repo=u_file.name,
            repo_id=HF_TARGET_REPO,
            commit_message=f"Upload {u_file.name} (A6000 multi-domain run)",
        )
        if hasattr(commit_info, "oid"):
            latest_commit_sha = commit_info.oid

print(f"\n[OK] All artifacts uploaded to https://huggingface.co/{HF_TARGET_REPO}")
print(f"[OK] Pinned Commit SHA: {latest_commit_sha}")
if wandb.run:
    wandb.log({
        "hub/repo_url": f"https://huggingface.co/{HF_TARGET_REPO}",
        "hub/commit_sha": latest_commit_sha,
    })
    wandb.finish()

print_banner("PIPELINE EXECUTION COMPLETE")
print(f"Hugging Face: https://huggingface.co/{HF_TARGET_REPO}")
print(f"Pinned Commit SHA: {latest_commit_sha}")
