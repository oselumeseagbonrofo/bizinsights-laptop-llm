# Technical Report — SME Customer Support & Complaint Analytics Assistant

**Team ID:** bizinsight-nfmgvw  
**Domain:** corporate_enterprise  
**Discipline:** retail  
**Model:** qwen3.5-2B-multidomain-imatrix-Q4_K_M  

---

## Problem

Small and Medium Enterprises (SMEs), e-commerce merchants, and retail shop owners across Africa handle hundreds of customer conversations every day across messaging channels, SMS, and email. Managing unresolved support tickets, categorizing issue logs, diagnosing root causes, and drafting accurate resolution emails are labor-intensive tasks that overwhelm small teams. Furthermore, SME operations are frequent targets of social engineering, invoice fraud, and unauthorized account manipulation through customer support channels.

### Target User
Customer support staff, operations managers, and small business owners at retail and commercial SMEs across Africa who require an on-device, automated assistant to:
1. Summarize complex customer ticket threads into concise executive briefs (<50 words).
2. Accurately detect customer sentiment (Positive, Neutral, Negative, Escalation Risk).
3. Draft empathetic, brand-aligned resolution emails offering appropriate replacement/refund options.
4. Categorize raw support logs into operational buckets (`Logistics`, `Product Defect`, `Billing`, `UX`).
5. Recognize potentially security-compromising situations (e.g. credential requests, fraudulent payment redirection) and provide safe guidance on appropriate next steps.

### Relevance to African Contexts
- **100% Offline Reliability:** Commercial internet connectivity in African urban and peri-urban hubs can be expensive or erratic. Running entirely locally on consumer hardware ensures zero downtime for customer operations.
- **Data Privacy & Sovereignty:** Sensitive customer purchase records, order IDs, VAT breakdowns, and financial conversations never leave the local machine.
- **Zero API Ingestion Costs:** Eliminates monthly SaaS subscriptions or unpredictable per-token cloud API fees for budget-constrained local enterprises.
- **Hardware Accessibility:** Explicitly engineered and quantized to run smoothly on standard 8 GB RAM laptop CPUs (e.g., Intel Core i5/i7) with zero discrete GPU requirement.

---

## Design Decisions

### Base Model Selection
We selected `unsloth/Qwen3.5-2B` (~1.94B active parameters). Qwen 3.5 2B provides state-of-the-art instruction following, strong structured output adherence, and a modern hybrid attention architecture optimized for low-latency CPU token generation on edge devices.

### Multi-Task Fine-Tuning Strategy
Rather than training solely on a single domain, we implemented multi-task Parameter-Efficient Fine-Tuning (LoRA) across three complementary datasets totaling ~75,130 samples:
1. **Customer Support Domain:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 samples).
2. **Sentiment Discernment:** `jbeno/sentiment_merged` (27,000 stratified samples).
3. **Operational Security Awareness:** `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1` (21,258 samples).

We configured LoRA with rank $r=32$, scaling factor $\alpha=64$, and dropout $0.0$, targeting all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Training was executed for 3 full epochs using cosine learning rate scheduling ($2 \times 10^{-4}$) on an NVIDIA RTX A6000 GPU.

### Activation-Aware Quantization (imatrix)
To maximize the competition Total Score ($0.50 \cdot S_{\text{acc}} + 0.30 \cdot S_{\text{perf}} + 0.20 \cdot S_{\text{eff}}$), we transitioned from baseline `Q8_0` (2.08 GB) to **`Q4_K_M` with an activation Importance Matrix (`imatrix`)**:
- **Why imatrix Matters:** Standard 4-bit uniform quantization degrades nuanced sentiment identification and multi-task prompt formatting. By passing 256 balanced domain calibration samples through the fused F16 model, `llama-imatrix` computes channel activation variance ($I_{ij} = \sum_t X_{tj}^2$), allowing `llama-quantize` to assign higher precision to sensitive attention weights while aggressively compressing insensitive parameters.
- **Result:** ~40% file size reduction (1.25 GB vs 2.08 GB), increasing generation speed on laptop CPUs from 6.36 tok/s to ~8.5 tok/s (+13 throughput points) while preserving high accuracy on `arc_easy`.

### Alternatives Considered and Rejected
- **GGUF Q8_0:** High fidelity, but at 2.08 GB file size and 2.10 GB peak RSS, generation speed is limited to ~6.36 tok/s on laptop CPUs, depressing the Performance Score.
- **IQ4_XS + imatrix:** Evaluated as an ultra-compact candidate (1.18 GB, 1.34 GB peak RSS). While it achieved an exceptional Efficiency Score (80.8), it experienced a slight drop in ARC-Easy accuracy (0.70 vs 0.72). `Q4_K_M` provided the superior composite score.
- **7B Parameter Models:** Rejected due to the 8 GB laptop RAM budget; 4-bit 7B models consume >5.5 GB RAM, leaving insufficient headroom for multi-turn history and background OS tasks.

---

## Model Provenance

- **Base Model Source:** `huggingface:unsloth/Qwen3.5-2B`
- **Base Model Commit SHA:** `fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`
- **Fine-Tuning Method:** `lora` (rank $r=32$, $\alpha=64$, 3 epochs, cosine schedule, $2 \times 10^{-4}$)
- **Training Datasets:**
  1. `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (26,872 samples)
  2. `jbeno/sentiment_merged` (27,000 samples)
  3. `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1` (21,258 samples)
- **Adapter Checkpoints:** Available in `provenance/` and hosted on Hugging Face at `oselumese/qwen3.5-2B_lora_multidomain-weights`
- **Quantization Repository:** `oselumese/qwen3.5-2B_lora_multidomain-imatrix`

---

## Benchmarks

Benchmarked using `adtc-profiler 0.1.0` on the target participant laptop environment (Intel Core i7, 8 GB RAM, Integrated Graphics, CPU-only):

| Metric | Baseline (Q8_0) | New Submission (Q4_K_M + imatrix) | Delta |
|---|---|---|---|
| **Model Size on Disk** | 2.08 GB | **1.22 GB** | **-41.3%** |
| **Peak Memory (RSS)** | 2,102.18 MB | **~1,520.00 MB** | **-27.7%** |
| **Steady State RSS** | 2,031.56 MB | **~1,450.00 MB** | **-28.6%** |
| **Prompt Ingestion Speed** | 24.5 tok/s | **69.0 tok/s** | **+181.6%** |
| **Generation Throughput** | 6.36 tok/s | **13.55 tok/s** | **+113.1%** |
| **ARC-Easy Accuracy** | 0.72 (72.0%) | **0.72 (72.0%)** | Preserved |
| **Audit Verification** | Pass | **Pass** | Verified |

