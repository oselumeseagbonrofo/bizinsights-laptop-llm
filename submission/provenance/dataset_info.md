# Dataset Provenance & Information

This document details the datasets utilized for multi-task Parameter-Efficient Fine-Tuning (LoRA) of `unsloth/Qwen3.5-2B` for the Africa Deep Tech Challenge 2026 (Laptop LLM Track).

---

## 1. Bitext Customer Support Dataset
- **Hugging Face ID:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **URL:** [https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- **Role in Training:** Primary customer support instruction-following, ticket triage, order inquiry resolution, complaint handling, and polite refund/replacement communication.
- **Samples Used:** 26,872 instruction-response pairs (100% of train split)
- **Format:** Single-turn instruction-response pairs converted to Qwen ChatML formatting:
  ```text
  <|im_start|>system\n{BizInsights System Prompt}<|im_end|>\n
  <|im_start|>user\n{instruction}<|im_end|>\n
  <|im_start|>assistant\n{response}<|im_end|>
  ```
- **License:** Open data for training conversational assistants.

---

## 2. Sentiment Merged Dataset
- **Hugging Face ID:** `jbeno/sentiment_merged`
- **URL:** [https://huggingface.co/datasets/jbeno/sentiment_merged](https://huggingface.co/datasets/jbeno/sentiment_merged)
- **Role in Training:** Fine-grained sentiment discernment (positive, neutral, negative) across adversarial and real-world customer reviews (incorporating Stanford Sentiment Treebank and DynaSent).
- **Samples Used:** 27,000 samples (stratified balanced sample: 9,000 positive, 9,000 neutral, 9,000 negative).
- **Format:** Converted into multi-template sentiment classification requests to instill robust customer sentiment detection.
- **License:** Research & commercial use open dataset.

---

## 3. Heimdall Defensive Cybersecurity Dataset
- **Hugging Face ID:** `AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1`
- **URL:** [https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Heimdall-v1.1)
- **Role in Training:** Providing the model with awareness of potentially security-compromising situations (e.g. social engineering, invoice spoofing, suspicious links, credential theft) and instructing it to flag risky actions and provide guidance on appropriate next steps.
- **Samples Used:** 21,258 conversational samples (100% of train split).
- **Format:** Conversational triplets with defensive remediation.
- **License:** Open access.

---

## Combined Dataset Summary
- **Total Training Instances:** 75,130 samples
- **Data Shuffling:** Global deterministic seed `3407` / `42`
- **Calibration Subset for imatrix:** 256 balanced multi-domain samples (86 Bitext, 85 Sentiment, 85 Cybersecurity)
