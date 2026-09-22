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

### Comparison Prompts
#### Test Prompt 1: Unresolved Ticket Thread Analysis (`tp_001`)

##### Input Prompt
> Analyse the following unresolved ticket thread. Provide: Executive Summary (under 50 words), root cause & sentiment (Positive / Neutral / Negative / Escalation Risk) and draft resolution email (polite, concise, offering refund/replacement options) [TICKET THREAD] Customer: Hi, our order #88419 for 20 custom office chairs arrived 4 days late, and 3 bases are cracked. We have an office launch this Friday and cannot use these., Support: Checking on the shipment details now. Customer: Any update? It's been 24 hours. We need replacements or an immediate refund so we can source locally. [/TICKET THREAD]

##### Output: Before (Default Base Model — Qwen3.5-2B-Q4_K_M)
**Executive Summary:**
Order #88419 is 4 days late with 3 cracked chairs. Customer demands immediate replacement or refund due to a critical office launch.

**Root Cause & Sentiment:**
*   **Root Cause:** Supply chain delay causing product damage (cracked bases).
*   **Sentiment:** **Negative** (due to damage and deadline breach).
*   **Escalation Risk:** **High** (Customer explicitly threatens to source locally, indicating dissatisfaction).

**Resolution Email Draft:**

**Subject:** Urgent: Resolution for Order #88419 – Immediate Replacement Options

Dear [Customer Name],

Thank you for your patience regarding the delay with Order #88419. We sincerely apologize for the inconvenience and the damaged chairs, especially given your critical office launch.

We understand you need replacements immediately. To resolve this, we can offer the following options:
1.  **Immediate Replacement:** We will ship 3 new chairs today to ensure you can launch on time.
2.  **Full Refund:** We can issue a full refund for the damaged items.

Please let us know which option you prefer, or simply reply with "replacement" and I will process the shipping right away. We are committed to getting this resolved for you immediately.

Best regards,

[Your Name]  
[Your Title]  
[Company Name]

##### Output: After (Final Fine-Tuned Model — qwen3.5-2B-bitext-imatrix-Q4_K_M)
*(Inference Latency: 43.4s | Generation: ~9.7 tok/s | CPU 4 Threads)*

**Executive Summary:**
Your office chairs are 4 days late, with 3 bases cracked, causing a critical delay to your office launch. We acknowledge the frustration and will prioritize your replacement request immediately.

**Root Cause & Sentiment:**
The root cause appears to be a supply chain delay. The sentiment is **Negative**, as the delay and damage directly impact your ability to launch on time.

**Resolution Email Draft:**

Dear Valued Customer,

Thank you for bringing this to our attention. We sincerely apologize for the inconvenience caused by the delayed arrival of your office chairs and the damage to the bases. We understand how critical it is for you to launch your office this Friday, and we are fully committed to resolving this issue for you.

To address your concerns, please allow us a moment to investigate and arrange for immediate replacements. We will ensure that you receive new chairs that meet your specifications and are in good condition. If you prefer, we can also offer a full refund for the damaged items so you can source replacements locally.

We appreciate your patience and understanding as we work to resolve this matter. If you have any further questions or need additional assistance, please don't hesitate to reach out.

Thank you for your cooperation.

Best regards,

[Your Name]  
Customer Support Team  
BizInsights AI


#### Test Prompt 2: 5 Support Logs Categorization (`tp_002`)

##### Input Prompt
> Categorise the following 5 customer support logs into Logistics, Product Defect, Billing, or UX, and identify the top recurring issue for customers: [LOGS] 1. Invoice didn't show VAT breakdown required for HMRC filing. 2. Tracking link sent via SMS was broken; carrier said package delayed. 3. Need VAT receipt for expense claims on order #4491. 4. Where do I add company tax ID during checkout? 5. Delivery driver left pallets in the rain; packaging damaged. [/LOGS]

##### Output: Before (Default Base Model — Qwen3.5-2B-Q4_K_M)
Here is the categorization and analysis of your customer support logs based on BizInsights AI protocols.

### 📊 Log Categorization

| Log # | Category | Reasoning |
| :--- | :--- | :--- |
| **1** | **Billing** | The customer is requesting a breakdown of VAT, which is a financial calculation and tax compliance issue. |
| **2** | **Logistics** | The issue involves tracking links, carrier communication, and package delivery status. |
| **3** | **Billing** | The request is for a VAT receipt to support expense claims, directly impacting financial records. |
| **4** | **UX** | The question "Where do I add company tax ID" is a procedural instruction error regarding the checkout process. |
| **5** | **Logistics** | The damage to packaging and the condition of the delivery driver's pallets are physical logistics failures. |

---

### 🚨 Security & Compliance Alert

**Critical Issue Detected:**
*   **Log #1** explicitly mentions "HMRC filing."
*   **Log #3** requires a "VAT receipt" for expense claims.

**Security & Compliance Guidance:**
These logs indicate a potential compliance gap regarding **HMRC (UK Revenue) filing requirements**. Customers are explicitly asking for VAT breakdowns and receipts, which suggests their current invoice format may not meet their specific tax filing needs.

*   **Immediate Action Required:** Do not simply send a standard invoice. You must verify that the invoice includes the specific VAT breakdown required for HMRC.
*   **Next Steps:**
    1.  **Audit the Invoice:** Ensure the invoice clearly states the VAT amount and breakdown (e.g., "VAT Included" or "VAT Excluded" with the correct rate).
    2.  **Customer Communication:** Proactively inform the customer that their current invoice format does not meet HMRC requirements and that a corrected VAT breakdown is needed for their filing.
    3.  **Escalation:** If the customer insists on a specific format not provided, escalate this to your Finance/Compliance team to review the invoice generation settings before the next shipment.

*Note: While this is primarily a billing/compliance issue, it can also be categorized as a **UX** issue if the checkout flow makes it difficult to locate the tax details.*


##### Output: After (Final Fine-Tuned Model — qwen3.5-2B-bitext-imatrix-Q4_K_M)
Thank you for providing the five customer support logs. I've analyzed each one and categorized them as follows:

1. **Invoice didn't show VAT breakdown required for HMRC filing.** -> **Billing**
2. **Tracking link sent via SMS was broken; carrier said package delayed.** -> **Logistics**
3. **Need VAT receipt for expense claims on order #4491.** -> **Billing**
4. **Where do I add company tax ID during checkout?** -> **UX**
5. **Delivery driver left pallets in the rain; packaging damaged.** -> **Logistics**

Based on this analysis, the **top recurring issue** for our customers is **Billing**. There are three instances of billing-related concerns (Invoice VAT breakdown, VAT receipt for expense claims, and a general question about adding company tax ID), which suggests a need for further review of our billing process and communication regarding invoices.

If you have any additional details or specific concerns about these categories, please let me know. I'm here to assist you further.

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