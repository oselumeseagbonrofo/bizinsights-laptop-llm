# Technical Report — SME Customer Support & Complaint Analytics Assistant

**Team ID:** bizinsight-nfmgvw  
**Domain:** corporate_enterprise  
**Model:** qwen3.5-2B-lora-Instruct-Q8_0  

---

## Problem

Small and Medium Enterprises (SMEs), e-commerce vendors, and retail merchants across Africa handle hundreds of customer interactions daily. Managing unresolved support tickets, categorizing issue logs, and drafting accurate resolution emails are labor-intensive tasks that often overwhelm small operational teams. Furthermore SMEs are frequent targets of fraud and scams making it important to ensure that their customer support system can detect and respond to potential security threats in customer messages.

### Target User
The primary target users are customer support teams, operations managers, and store owners at African retail SMEs who require an intelligent, automated assistant. The assistant is designed to analyze unresolved ticket threads by generating concise executive summaries, evaluating customer sentiment, and drafting professional resolution emails or messages to customers. Additionally, it automatically classifies raw customer support logs into user-defined operational categories and detects top recurring customer pain points.

### Relevance to African Contexts
Deploying an LLM for customer operations in Africa introduces unique infrastructure and operational demands:
- **Offline Reliability & Resilience:** Internet connectivity in many commercial centers across the continent can be high-cost, and unstable as such running 100% offline on local hardware guarantees zero service interruption for critical customer communications.
- **Data Sovereignty & Privacy:** SME customer records, tax IDs, invoice data, and order details remain strictly on-device, eliminating privacy risks associated with sending sensitive operational data to third-party cloud APIs.
- **Zero Ongoing API Costs:** Eliminates monthly SaaS subscription and per-token API charges, making advanced AI capabilities financially sustainable for budget-constrained African businesses.
- **Consumer Hardware Accessibility:** Engineered specifically to run smoothly on standard, accessible consumer laptops (4 vCPU, 8 GB RAM, integrated graphics) without requiring specialized hardware upgrades. This allows SMEs to easily add AI capabilities to their existing invoicing, inventory and accounting computers.

---

## Design Decisions

### Base Model Selection
We selected `unsloth/Qwen3.5-2B` (Qwen 3.5 2B Instruct architecture with approximately 1.94B active parameters) as our foundational base model. Qwen 3.5 2B provides state-of-the-art instruction following, strong structured output adherence, and native support for extended context windows up to 256k tokens. In addition, its 1.94B parameter count enables both high reasoning capacity and rapid CPU execution on low-resource hardware.

### Fine-Tuning Methodology
The model was fine-tuned on the bitext/Bitext-customer-support-llm-chatbot-training-dataset, comprising 26,872 instruction-response pairs covering customer service intents, complaint handling, and resolution responses. We configured LoRA with rank $r=16$, scaling factor $\alpha=16$, and dropout $0$, targeting all linear attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Training was executed for on an NVIDIA RTX Pro 6000 (Blackwell) in molab.marimo.io using Unsloth and HuggingFace TRL's `SFTTrainer`.

### Activation-Aware Quantization (imatrix)
To maximize the model's accuracy as well as improve its throughput performance and efficiency, the baseline `Q8_0` (2.08 GB) from gate 1 was replaced with a `Q4_K_M` quantized model using an activation importance matrix **(`imatrix`)**.

`imatrix` quantization using `llama-matrix` preserves high precision for the most important attention weights while aggressively compressing less sensitive parameters. This approach ensures that the model maintains its nuanced sentiment identification, careful context analysis, and coherently drafted customer communications capabilities while significantly reducing model size and improving inference speed.

This resulted in a 37% reduction in file size (1.31 GB vs 2.08 GB) and an increase in generation speed while preserving accuracy on `arc-easy`.

### Alternatives Considered and Rejected
During model export, we evaluated `GGUF IQ4_XS` imatrix (~1.23 GB). While had a smaller memory footprint, it caused a 2% drop in accuracy. We also tested `GGUF Q5_K_M` (~1.45 GB) as a mid-tier option; it provided decent response quality, but had more memory usage. `Q8_0` was changed as the `Q4_K_M` was able to attain its accuracy levels while being more performant and efficient. Finally, larger 7B base models such as Qwen 7B or Mistral 7B were rejected due to memory constraints, as 4-bit 7B models consume 5.0 to 5.5 GB RAM, leaving minimal safety margin for long conversations on an 8 GB system.

---
## Model Provenance

- **Base Model Source:** `huggingface:unsloth/Qwen3.5-2B`
- **Base Model Commit SHA:** `fffbc0d8711cf87df6d1f97938c5e50ef94aadc7`
- **Fine-Tuning Method:** `lora`
- **Training Dataset:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset` (url: `https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset`)
- **Adapter Checkpoints:** Available in `provenance/` and hosted on Hugging Face at `oselumese/qwen3.5-2B_lora_bitext-weights-2`
---

## Constraints

The target deployment environment is a standard participant laptop, such as an HP EliteBook 840 G3 equipped with an Intel Core i7 processor (Intel64 Family 6 Model 78 Stepping 3), 7.9 GB total RAM, and integrated graphics, running Windows 10 or Linux. Inference relies purely on CPU execution via `llama.cpp` without discrete GPU acceleration. Regarding data constraints, development initially targeted a support assistant for the automobile industry; however, due to the lack of high-quality public datasets in that domain, we decided to pivot to a general customer support assistant model.

---

## Benchmarks

Development benchmarks measured on the target participant laptop environment using the ADTC Profiler (`adtc-profiler 0.1.0`):

| Metric | Value |
|---|---|
| **Machine** | HP EliteBook 840 G3 / Intel Core i7 (Family 6 Model 78), 7.9 GB RAM, Integrated GPU |
| **RAM at Peak** | **2,002.45 MB** (~2.0 GB) |
| **Time to First Token** | **15621.72 ms** (~15.62 s) |
| **Generation Speed** | **8.94 tokens/sec** |
| **CPU Utilization (P99)** | **70.3%** |
| **Thermal Throttling** | None observed |