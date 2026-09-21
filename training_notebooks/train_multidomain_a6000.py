#!/usr/bin/env python3
"""
End-to-End Multi-Domain Fine-Tuning & imatrix Quantization Pipeline for NVIDIA A6000
===================================================================================
Target Model: Qwen 3.5 2B (unsloth/Qwen3.5-2B)
Datasets:
  1. bitext/Bitext-customer-support-llm-chatbot-training-dataset (26,872 samples)
  2. jbeno/sentiment_merged (27,000 stratified samples)
  3. AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1 (21,258 samples)
Total: ~75,130 samples

Outputs:
  - LoRA adapter weights (r=32, alpha=64)
  - Pre-fused F16 GGUF
  - Multi-domain activation importance matrix (imatrix.dat with 256 samples)
  - Calibrated Q4_K_M GGUF
  - Calibrated IQ4_XS GGUF
  - ARC-Easy accuracy sanity evaluation
  - Test prompts benchmark using llama-cli
  - Automatic push to HuggingFace: oselumese/qwen3.5-2B_lora_multidomain-imatrix
  - WandB experiment tracking
"""

import os
import sys
import time
import math
import random
import shutil
import subprocess
from pathlib import Path

# Configure environment variables before imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "bizinsights-laptop-llm"
os.environ["WANDB_LOG_MODEL"] = "false"

# Credentials & Repositories
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "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")
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        HF_TOKEN = token_file.read_text().strip()
if not HF_TOKEN:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                HF_TOKEN = line.split("=", 1)[1].strip().strip('"').strip("'")


HF_TARGET_REPO = "oselumese/qwen3.5-2B_lora_multidomain-imatrix"
HF_LORA_REPO = "oselumese/qwen3.5-2B_lora_multidomain-weights"

# Directory Structure
BASE_DIR = Path(os.getcwd()).resolve()
DATA_DIR = BASE_DIR / "data_cache"
OUTPUT_DIR = BASE_DIR / "training_outputs"
LORA_DIR = BASE_DIR / "lora_weights"
GGUF_F16_DIR = BASE_DIR / "model_f16"
LLAMA_BUILD_DIR = BASE_DIR / "llama_cpp"
FINAL_MODELS_DIR = BASE_DIR / "final_models"

for d in [DATA_DIR, OUTPUT_DIR, LORA_DIR, GGUF_F16_DIR, FINAL_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# System Prompt matching server.py
DEFAULT_SYSTEM_PROMPT = (
    "You are BizInsights AI, an intelligent customer support and operations assistant "
    "tailored for retail and enterprise SMEs. You analyze customer tickets, identify root causes, "
    "evaluate customer sentiment, categorize operational issues, and draft polite, concise, "
    "and empathetic resolution emails. When you detect requests involving potentially "
    "security-compromising situations, you provide guidance on appropriate next steps."
)

# Test Prompts for verification
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
# 0. Environment & Verification
# ==============================================================================
print_banner("STAGE 0: Environment Verification")

import torch
import wandb
from huggingface_hub import login as hf_login, HfApi

if not torch.cuda.is_available():
    raise SystemError("CUDA GPU is required but not detected! Please verify NVIDIA driver.")

gpu_props = torch.cuda.get_device_properties(0)
gpu_vram_gb = round(gpu_props.total_memory / (1024**3), 2)
print(f"[OK] GPU: {gpu_props.name}")
print(f"[OK] VRAM: {gpu_vram_gb} GB")
print(f"[OK] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

# Login
hf_login(token=HF_TOKEN)
print("[OK] Logged into Hugging Face.")

wandb.login(key=WANDB_API_KEY)
wandb.init(
    project="bizinsights-laptop-llm",
    name="qwen3.5-2B-multidomain-a6000",
    config={
        "base_model": "unsloth/Qwen3.5-2B",
        "gpu": gpu_props.name,
        "vram_gb": gpu_vram_gb,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0,
        "learning_rate": 2e-4,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.05,
        "epochs": 3,
        "per_device_batch_size": 8,
        "grad_accum_steps": 2,
        "max_seq_length": 2048,
        "imatrix_samples": 256,
        "datasets": [
            "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
            "jbeno/sentiment_merged",
            "AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1",
        ],
    },
)
print("[OK] WandB initialized.")


# ==============================================================================
# 1. Dataset Preparation & Multi-Task Mixing
# ==============================================================================
print_banner("STAGE 1: Loading & Mixing 3 Datasets")

from datasets import load_dataset, Dataset

formatted_records = []

# --- 1.1 Bitext Customer Support Dataset ---
print("-> Loading bitext/Bitext-customer-support-llm-chatbot-training-dataset...")
ds_bitext = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
bitext_count = 0
for row in ds_bitext:
    inst = str(row.get("instruction", "")).strip()
    resp = str(row.get("response", "")).strip()
    if inst and resp:
        text = (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n{resp}<|im_end|>"
        )
        formatted_records.append({"text": text, "source": "bitext"})
        bitext_count += 1
print(f"[OK] Added {bitext_count} Bitext customer support samples.")

# --- 1.2 Sentiment Merged Dataset (Downsampled to 27,000 balanced) ---
print("-> Loading jbeno/sentiment_merged...")
ds_sentiment = load_dataset("jbeno/sentiment_merged", split="train")

sentiment_templates = [
    "Classify the sentiment of the following customer text: \"{sentence}\"",
    "Analyze the sentiment of this customer statement as positive, neutral, or negative: \"{sentence}\"",
    "Determine whether the sentiment expressed in this message is positive, negative, or neutral: \"{sentence}\"",
    "What is the customer sentiment in this review? \"{sentence}\"",
]

# Separate by label for stratified sampling
sent_by_label = {"positive": [], "neutral": [], "negative": []}
for row in ds_sentiment:
    sent = str(row.get("sentence", "")).strip()
    lbl = str(row.get("label", "")).strip().lower()
    if sent and lbl in sent_by_label:
        sent_by_label[lbl].append(sent)

target_per_label = 9000  # 9,000 * 3 = 27,000
random.seed(42)
sentiment_count = 0
for lbl, sentences in sent_by_label.items():
    sampled = random.sample(sentences, min(len(sentences), target_per_label))
    for s in sampled:
        tpl = random.choice(sentiment_templates).format(sentence=s)
        resp = f"Sentiment: {lbl.capitalize()}."
        text = (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{tpl}<|im_end|>\n"
            f"<|im_start|>assistant\n{resp}<|im_end|>"
        )
        formatted_records.append({"text": text, "source": "sentiment"})
        sentiment_count += 1
print(f"[OK] Added {sentiment_count} balanced Sentiment samples (9k positive, 9k neutral, 9k negative).")

# --- 1.3 Heimdall Cybersecurity Dataset ---
print("-> Loading AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1...")
ds_heimdall = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1", split="train")
heimdall_count = 0
for row in ds_heimdall:
    u = str(row.get("user", "")).strip()
    a = str(row.get("assistant", "")).strip()
    s = str(row.get("system", "")).strip() or DEFAULT_SYSTEM_PROMPT
    if u and a:
        text = (
            f"<|im_start|>system\n{s}<|im_end|>\n"
            f"<|im_start|>user\n{u}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>"
        )
        formatted_records.append({"text": text, "source": "heimdall"})
        heimdall_count += 1
print(f"[OK] Added {heimdall_count} Cybersecurity Heimdall samples.")

# Shuffle all combined data
random.shuffle(formatted_records)
total_samples = len(formatted_records)
print(f"\n[OK] Combined Dataset Ready: {total_samples} total samples")
print(f"      - Bitext:        {bitext_count} ({round(bitext_count/total_samples*100, 1)}%)")
print(f"      - Sentiment:     {sentiment_count} ({round(sentiment_count/total_samples*100, 1)}%)")
print(f"      - Cybersecurity: {heimdall_count} ({round(heimdall_count/total_samples*100, 1)}%)")

wandb.log({
    "data/total_samples": total_samples,
    "data/bitext_samples": bitext_count,
    "data/sentiment_samples": sentiment_count,
    "data/cybersecurity_samples": heimdall_count,
})

combined_dataset = Dataset.from_list(formatted_records)


# ==============================================================================
# 2. Fine-Tuning with Unsloth
# ==============================================================================
print_banner("STAGE 2: Unsloth LoRA Fine-Tuning (r=32, alpha=64, 3 epochs)")

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq

MAX_SEQ_LENGTH = 2048

print("-> Loading base model unsloth/Qwen3.5-2B in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-2B",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detects bfloat16 for Ampere A6000
    load_in_4bit=True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-instruct",
)

print("-> Attaching LoRA adapters (r=32, alpha=64, all 7 linear projections)...")
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=combined_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        output_dir=str(OUTPUT_DIR),
        report_to="wandb",
        run_name="qwen3.5-2B-multidomain-a6000",
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)

trainer = train_on_responses_only(trainer)

print("\n-> Launching training on NVIDIA A6000...")
train_start = time.time()
train_stats = trainer.train()
train_duration = round(time.time() - train_start, 2)

print(f"\n[OK] Training completed in {train_duration}s ({round(train_duration/60, 2)} minutes)!")
wandb.log({
    "train/runtime_sec": train_duration,
    "train/train_loss": train_stats.training_loss,
})

# Save LoRA adapter locally and to HF
print("-> Saving LoRA adapter checkpoint...")
model.save_pretrained(str(LORA_DIR))
tokenizer.save_pretrained(str(LORA_DIR))

try:
    print(f"-> Pushing LoRA weights to {HF_LORA_REPO}...")
    model.push_to_hub(HF_LORA_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_LORA_REPO, token=HF_TOKEN)
    print(f"[OK] LoRA adapter pushed to https://huggingface.co/{HF_LORA_REPO}")
except Exception as e:
    print(f"[WARN] Failed pushing LoRA adapter to Hub: {e}")


# ==============================================================================
# 3. Export to Pre-Fused F16 GGUF
# ==============================================================================
print_banner("STAGE 3: Merging LoRA & Exporting Pre-Fused F16 GGUF")

F16_GGUF_PATH = GGUF_F16_DIR / "Qwen3.5-2B.F16.gguf"

print(f"-> Exporting fused F16 GGUF to {GGUF_F16_DIR}...")
model.save_pretrained_gguf(
    str(GGUF_F16_DIR),
    tokenizer,
    quantization_method="f16",
)

# Unsloth usually saves as Qwen3.5-2B.F16.gguf or similar in that directory
f16_files = list(GGUF_F16_DIR.glob("*.gguf")) + list(GGUF_F16_DIR.glob("*f16*.gguf"))
if not f16_files:
    raise FileNotFoundError(f"No F16 GGUF file found in {GGUF_F16_DIR}")

ACTUAL_F16_PATH = f16_files[0]
f16_size_gb = round(os.path.getsize(ACTUAL_F16_PATH) / (1024**3), 2)
print(f"[OK] Fused F16 GGUF Created: {ACTUAL_F16_PATH} ({f16_size_gb} GB)")
wandb.log({"model/f16_disk_gb": f16_size_gb})

# Free Python GPU memory before C++ llama.cpp operations
del model
del trainer
torch.cuda.empty_cache()
print("[OK] VRAM released for llama.cpp tools.")


# ==============================================================================
# 4. Build llama.cpp with CUDA
# ==============================================================================
print_banner("STAGE 4: Verifying / Building llama.cpp with CUDA")

LLAMA_BIN_DIR = LLAMA_BUILD_DIR / "build" / "bin"
LLAMA_IMATRIX = LLAMA_BIN_DIR / "llama-imatrix"
LLAMA_QUANTIZE = LLAMA_BIN_DIR / "llama-quantize"
LLAMA_CLI = LLAMA_BIN_DIR / "llama-cli"

def binaries_ready():
    return LLAMA_IMATRIX.exists() and LLAMA_QUANTIZE.exists() and LLAMA_CLI.exists()

if not binaries_ready():
    print("-> Cloning llama.cpp repository...")
    if not LLAMA_BUILD_DIR.exists():
        subprocess.run(["git", "clone", "https://github.com/ggml-org/llama.cpp", str(LLAMA_BUILD_DIR)], check=True)
    
    print("-> Compiling llama.cpp with CUDA support (-DGGML_CUDA=ON)...")
    cpu_cores = os.cpu_count() or 4
    subprocess.run(
        ["cmake", "-B", "build", "-DGGML_CUDA=ON"],
        cwd=str(LLAMA_BUILD_DIR),
        check=True
    )
    subprocess.run(
        ["cmake", "--build", "build", f"-j{cpu_cores}", "--target", "llama-imatrix", "llama-quantize", "llama-cli"],
        cwd=str(LLAMA_BUILD_DIR),
        check=True
    )

if not binaries_ready():
    # Check alternate build output layout
    alt_bin = LLAMA_BUILD_DIR / "build"
    for b in ["llama-imatrix", "llama-quantize", "llama-cli"]:
        candidate = alt_bin / b
        if candidate.exists():
            shutil.copy(candidate, LLAMA_BIN_DIR / b)

print(f"[OK] llama-imatrix : {LLAMA_IMATRIX}")
print(f"[OK] llama-quantize: {LLAMA_QUANTIZE}")
print(f"[OK] llama-cli     : {LLAMA_CLI}")


# ==============================================================================
# 5. Multi-Domain Activation Importance Matrix (imatrix.dat)
# ==============================================================================
print_banner("STAGE 5: Multi-Domain imatrix Calibration (256 samples)")

CALIB_FILE = BASE_DIR / "calibration_multidomain_256.txt"
IMATRIX_FILE = BASE_DIR / "imatrix.dat"

# Select 86 Bitext, 85 Sentiment, 85 Cybersecurity
calib_bitext = [r["text"] for r in formatted_records if r["source"] == "bitext"][:86]
calib_sent = [r["text"] for r in formatted_records if r["source"] == "sentiment"][:85]
calib_heimdall = [r["text"] for r in formatted_records if r["source"] == "heimdall"][:85]

all_calib = calib_bitext + calib_sent + calib_heimdall
random.shuffle(all_calib)

with open(CALIB_FILE, "w", encoding="utf-8") as f:
    for item in all_calib:
        f.write(item + "\n\n")

calib_kb = round(os.path.getsize(CALIB_FILE) / 1024, 2)
print(f"[OK] Prepared {CALIB_FILE} with {len(all_calib)} multi-domain samples ({calib_kb} KB).")

print(f"-> Computing importance matrix on GPU offload (-ngl 99 -c 512 --chunks 128)...")
t0 = time.time()
res = subprocess.run(
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

if res.returncode != 0:
    print(res.stdout)
    print(res.stderr)
    raise RuntimeError(f"llama-imatrix failed with code {res.returncode}")

imatrix_duration = round(time.time() - t0, 2)
imatrix_kb = round(os.path.getsize(IMATRIX_FILE) / 1024, 2)
print(f"[OK] imatrix.dat computed in {imatrix_duration}s ({imatrix_kb} KB)!")

wandb.log({
    "imatrix/duration_sec": imatrix_duration,
    "imatrix/size_kb": imatrix_kb,
    "imatrix/samples": len(all_calib),
})


# ==============================================================================
# 6. Activation-Aware Quantization (Q4_K_M and IQ4_XS)
# ==============================================================================
print_banner("STAGE 6: Quantizing to Q4_K_M and IQ4_XS with imatrix")

Q4KM_PATH = FINAL_MODELS_DIR / "qwen3.5-2B-multidomain-imatrix-Q4_K_M.gguf"
IQ4XS_PATH = FINAL_MODELS_DIR / "qwen3.5-2B-multidomain-imatrix-IQ4_XS.gguf"

# 6.1 Q4_K_M
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

# 6.2 IQ4_XS
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

compression_q4 = round((1 - size_q4km_gb / 2.08) * 100, 1)
compression_iq4 = round((1 - size_iq4xs_gb / 2.08) * 100, 1)

print("\n" + "=" * 50)
print(f"Model Comparison vs Previous Q8_0 (2.08 GB):")
print(f"  - Q4_K_M + imatrix: {size_q4km_gb} GB (~{compression_q4}% smaller)")
print(f"  - IQ4_XS + imatrix: {size_iq4xs_gb} GB (~{compression_iq4}% smaller)")
print("=" * 50)

wandb.log({
    "quant/q4km_gb": size_q4km_gb,
    "quant/iq4xs_gb": size_iq4xs_gb,
    "quant/compression_q4km_pct": compression_q4,
    "quant/compression_iq4xs_pct": compression_iq4,
})


# ==============================================================================
# 7. Quick Accuracy Sanity Evaluation (ARC-Easy)
# ==============================================================================
print_banner("STAGE 7: Accuracy Evaluation (ARC-Easy 50 Samples)")

arc_scores = {}
try:
    import lm_eval
    print("-> Running lm_eval ARC-Easy (50 samples)...")
    for model_label, model_path in [("Q4_K_M", Q4KM_PATH), ("IQ4_XS", IQ4XS_PATH)]:
        try:
            results = lm_eval.simple_evaluate(
                model="llama_cpp",
                model_args=f"model_path={model_path},n_ctx=2048,n_gpu_layers=99",
                tasks=["arc_easy"],
                limit=50,
            )
            acc_norm = results["results"]["arc_easy"].get("acc_norm,none", results["results"]["arc_easy"].get("acc_norm"))
            arc_scores[model_label] = round(float(acc_norm), 4)
            print(f"[OK] {model_label} ARC-Easy acc_norm: {arc_scores[model_label]}")
            wandb.log({f"eval/arc_easy_{model_label.lower()}": arc_scores[model_label]})
        except Exception as e:
            print(f"[WARN] lm-eval run for {model_label} failed: {e}")
except ImportError:
    print("[INFO] lm_eval is not installed. To run official accuracy benchmark, install: pip install lm-eval[llama-cpp]")


# ==============================================================================
# 8. Direct Benchmark with llama-cli
# ==============================================================================
print_banner("STAGE 8: Direct Test Prompt Benchmarking with llama-cli")

eval_table = wandb.Table(columns=["model", "prompt_id", "name", "tok_per_sec", "duration_sec", "response"])

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
            ],
            capture_output=True,
            text=True,
        )
        cli_duration = round(time.time() - t_start, 2)
        full_output = res_cli.stdout + "\n" + res_cli.stderr
        
        # Parse tokens per second from llama-cli output
        tok_per_sec = 0.0
        for line in full_output.splitlines():
            if "eval time =" in line and "tokens per second" in line:
                try:
                    parts = line.split("tokens per second")[0].strip().split()
                    tok_per_sec = float(parts[-1])
                except Exception:
                    pass
        
        # Extract assistant response portion
        response_text = full_output
        if "<|im_start|>assistant" in full_output:
            response_text = full_output.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        
        print(f"\n[{model_label}] {pid}: {pname}")
        print(f"Speed: {tok_per_sec} tok/s | Duration: {cli_duration}s")
        print(f"Sample response:\n{response_text[:300]}...\n")
        
        eval_table.add_data(model_label, pid, pname, tok_per_sec, cli_duration, response_text[:1000])
        wandb.log({
            f"cli/{model_label.lower()}_{pid}_tok_s": tok_per_sec,
            f"cli/{model_label.lower()}_{pid}_duration_s": cli_duration,
        })

wandb.log({"cli/benchmark_results": eval_table})


# ==============================================================================
# 9. Push Artifacts to Hugging Face
# ==============================================================================
print_banner("STAGE 9: Pushing Quantized GGUF Artifacts to Hugging Face")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_TARGET_REPO, exist_ok=True, private=False)

# Create a clean Model Card
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
- **Fine-Tuning:** LoRA rank $r=32$, $\\alpha=64$, targeting all 7 linear projections
- **Epochs:** 3 full epochs with cosine LR decay ($2 \\times 10^{{-4}}$)
- **Datasets:**
  1. `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 samples)
  2. `jbeno/sentiment_merged` (27,000 stratified samples)
  3. `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1` (21,258 samples)
  - **Total Training Volume:** ~75,130 samples

## Quantized Artifacts
- `qwen3.5-2B-multidomain-imatrix-Q4_K_M.gguf` ({size_q4km_gb} GB): Calibrated with 256 balanced multi-domain samples. Recommended for best balance of reasoning accuracy and CPU generation speed.
- `qwen3.5-2B-multidomain-imatrix-IQ4_XS.gguf` ({size_iq4xs_gb} GB): Non-linear importance quantization for minimum memory footprint (~1.2 GB).
- `imatrix.dat`: Computed importance matrix from 256 multi-domain interactions.

## Provenance
- Base Model: `huggingface:unsloth/Qwen3.5-2B`
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
wandb.log({
    "hub/repo_url": f"https://huggingface.co/{HF_TARGET_REPO}",
    "hub/commit_sha": latest_commit_sha,
})

# Summary
print_banner("EXECUTION COMPLETE")
print(f"Target HF Repository: https://huggingface.co/{HF_TARGET_REPO}")
print(f"Pinned Commit SHA:    {latest_commit_sha}")
print(f"Local Q4_K_M:         {Q4KM_PATH} ({size_q4km_gb} GB)")
print(f"Local IQ4_XS:         {IQ4XS_PATH} ({size_iq4xs_gb} GB)")
print("\nWandB Run successfully recorded.")
wandb.finish()
