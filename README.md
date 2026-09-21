# ADTC 2026 — Submission

This is team BizInsights official repository for the **Africa Deep Tech Challenge 2026** Laptop LLM track.

## AWQ-informed GGUF research

The reproducible FP16/BF16-to-native-GGUF sensitivity-aware quantisation pipeline is in [quantization/README.md](quantization/README.md). It is isolated from the application runtime and never quantises from AWQ-packed weights.

## Shop knowledge (offline RAG)

The chat UI has a Shop knowledge panel (book icon in the header). Upload
`.txt`, `.md`, `.csv`, `.pdf`, or `.docx` files up to 10 MB each. Text is
split into 750-char overlapping chunks and stored in `rag_store.db`
(SQLite FTS5 plus pure-Python BM25 rerank, no new service, no torch).

- Answers cite sources as `filename (chunk N/M)` under the reply.
- Toggle per chat with the "Use shop docs in answers" checkbox (saved in
  the browser). API clients send `"use_rag": true` (default) or `false`
  on `POST /api/chat`.
- Debug endpoints: `GET /api/knowledge/list`,
  `GET /api/knowledge/search?q=...&k=3` (k clamped 1-10, q capped 500 chars),
  `DELETE /api/knowledge/{doc_id}`.
- Limits: 10 MB per file, 200 MB total store, 20 uploads per 10 minutes
  per IP. PDF capped at 50 pages, DOCX at 5000 paragraphs and 50 MB
  expanded, text at 200k chars, 200 chunks per doc. Top 3 chunks, max
  1500 chars of context, inside the 2048 ctx window. Chat clamps
  temperature to 0-2 and max tokens to 1-1024.
- Run `python test_rag.py` for RAG tests, `python test_analytics.py` for
  analytics tests. New deps: `python-multipart pypdf python-docx`.
