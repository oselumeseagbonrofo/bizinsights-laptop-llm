# Dataset Provenance & Information

This document details the datasets utilized for Parameter-Efficient Fine-Tuning (LoRA) of `unsloth/Qwen3.5-2B` for the Africa Deep Tech Challenge 2026 (Laptop LLM Track).

---

## 1. Bitext Customer Support Dataset
- **Hugging Face ID:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **URL:** [https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Role in Training:** Customer support instruction-following, ticket triage, order inquiry resolution, complaint handling, and polite refund/replacement communication.
- **Samples Used:** 26,872 instruction-response pairs (100% of train split)
- **Format:** Single-turn instruction-response pairs converted to Qwen ChatML formatting:
- **Datatset commit checksum:** 430d1a89bd93bd1fa23c16f29dd53e73f0087443

- **Calibration Subset for imatrix:** 256 samples