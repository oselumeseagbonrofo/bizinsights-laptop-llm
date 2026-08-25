# Technical Report — SME Customer Support & Complaint Analytics Assistant

**Team ID:** bizinsight-nfmgvw  
**Domain:** corporate_enterprise  
**Model:** qwen3.5-2B-lora-Instruct-Q8_0  

---

## Problem

Small and Medium Enterprises (SMEs), e-commerce vendors, and retail merchants across Africa handle hundreds of incoming customer interactions daily. Managing unresolved support tickets, categorizing issue logs, and drafting accurate resolution emails are labor-intensive tasks that often overwhelm small operational teams.

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
We selected `unsloth/Qwen3.5-2B` (Qwen 3.5 2B Instruct architecture with approximately 1.94B active parameters) as our foundational base model. Qwen 3.5 2B provides state-of-the-art instruction following, strong structural outputs including JSON schema adherence and structured text formatting, and native support for extended context windows up to 256k tokens. In addition, its 1.94B parameter count enables both high reasoning capacity and rapid CPU execution on low-resource hardware.

### Fine-Tuning Methodology
The model was fine-tuned on the `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, comprising 26,872 instruction-response pairs covering customer service intents, complaint handling, and resolution responses. Fine-tuning was performed using Unsloth and HuggingFace TRL's `SFTTrainer` with an `adamw_8bit` optimizer on a Tesla T4 GPU in Google Colab. We implemented Parameter-Efficient Fine-Tuning via LoRA with rank $r=16$, alpha $\alpha=16$, and dropout $0$, targeting all linear attention and MLP projection layers including `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

### Quantization & Runtime Format
For offline execution, the model was converted to GGUF format for `llama.cpp` and quantized to 8-bit precision using `GGUF Q8_0`. Eight-bit quantization preserves high precision for nuanced sentiment identification, careful context analysis, and coherently drafted customer communications. At a file size of 2.08 GB and approximately 2.10 GB peak RAM usage, `Q8_0` fits comfortably inside our 8 GB RAM target profile while leaving over 5.8 GB of system RAM free for other desktop applications.

### Alternatives Considered and Rejected
During model export, we evaluated `GGUF Q4_K_M` (~1.31 GB). While it reduced memory footprint by approximately 770 MB, 4-bit quantization exhibited slight formatting degradation when executing complex multi-task prompts such as simultaneous summary generation, sentiment analysis, and resolution email drafting. We also tested `GGUF Q5_K_M` (~1.45 GB) as a mid-tier option; it provided decent response quality, but given the substantial headroom remaining within the 8 GB RAM budget, `Q8_0` was selected to maximize output fidelity. Finally, larger 7B base models such as Qwen 7B or Mistral 7B were rejected due to memory constraints, as 4-bit 7B models consume 5.0 to 5.5 GB RAM, leaving minimal safety margin for long conversations on an 8 GB system.

---

## Constraints

The target deployment environment is a standard participant laptop, such as an HP EliteBook 840 G3 equipped with an Intel Core i7 processor (Intel64 Family 6 Model 78 Stepping 3), 7.9 GB total RAM, and integrated graphics, running Windows 10 or Linux. Inference relies purely on CPU execution via `llama.cpp` without discrete GPU acceleration. Regarding data constraints, development initially targeted a support assistant for the automobile industry; however, due to the lack of high-quality public datasets in that domain, we decided to pivot to a general customer support assistant model.

---

## Benchmarks

Development benchmarks measured on the target participant laptop environment using the ADTC Profiler (`adtc-profiler 0.1.0`):

| Metric | Value |
|---|---|
| **Machine** | HP EliteBook 840 G3 / Intel Core i7 (Family 6 Model 78), 7.9 GB RAM, Integrated GPU |
| **RAM at Peak** | **2,102.18 MB** (~2.10 GB) |
| **Time to First Token** | **22,782.81 ms** (~22.78 s) |
| **Generation Speed** | **6.36 tokens/sec** |
| **CPU Utilization (P99)** | **65.6%** |
| **Thermal Throttling** | None observed |