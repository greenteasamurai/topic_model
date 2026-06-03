# Narrative Analysis Tool — Critical Fixes

STATUS: FINALIZED

## Vision

A Python/FastAPI tool that ingests text, runs NLP analysis (mood, entities, topics, arcs, character impact), and returns quantitative narrative analysis plus Claude-powered interpretation.

The current system works for well-formed UTF-8 `.txt` novels with standard chapter markers. This project addresses the bugs and gaps that cause crashes, produce incorrect output, or block real-world use.

## Current Architecture (as-built)

- **Ingestion**: `.txt` only, UTF-8 assumed, chapter/act/scene regex splitting
- **Analysis pipeline**: spaCy NER → BERTopic → VADER/TextBlob → SentenceTransformer arcs → character impact (Cohen's d) → articulation graph
- **Output**: 7 PNG charts + Markdown report + Claude LLM interpretation
- **API**: FastAPI `POST /analyze` (multipart form: file + domain)
- **Frontend**: React/Vite, drag-and-drop `.txt` upload

## Must-Have Requirements (this project)

### Phase 1 — Crash & Correctness Bugs
1. LLM client initialized at module level crashes the server on startup if `ANTHROPIC_API_KEY` is missing
2. LLM API call has no try/except — any network/auth error returns unhandled HTTP 500
3. `visualization.py:164` uses substring match (`in`) to detect entity presence; `articulation_analysis.py:18` uses word-boundary regex — inconsistency produces wrong chart edges
4. All analyses write to the same output filenames — concurrent uploads corrupt each other's charts

### Phase 2 — Ingestion Hardening
5. `read_file()` (CLI path) uses `open(..., encoding="utf-8")` with no fallback — non-UTF-8 files crash with `UnicodeDecodeError`
6. API upload path uses `errors="replace"` — silently corrupts non-UTF-8 content instead of detecting encoding
7. No file size limit on API — memory exhaustion / DoS risk
8. `_FALSE_POSITIVES` entity filter is Shakespeare-specific (`verona`, `mantua`, `rome`, `italy`, archaic pronouns) — suppresses valid entities in court/meeting transcripts

### Phase 3 — Format Expansion
9. PDF files (`.pdf`) cannot be ingested at all
10. Conversation transcripts have no segmentation — speaker-label and timestamp patterns are not recognized as segment boundaries; domain parameter changes only the LLM prompt framing, not any processing

## Nice-to-Have (out of scope for this project)
- EPUB / DOCX support
- Multi-file corpus comparison
- OCR for scanned PDFs
- Streaming / incremental processing for large files

## Technical Constraints
- Python backend (FastAPI + existing analysis stack)
- Do not break the existing pipeline for `.txt` novels
- `chardet` for encoding detection (add to requirements)
- `pdfplumber` for PDF parsing (add to requirements)
- No changes to frontend until Phase 3 (then minimal: accept PDF, show domain selector)
- Output directory isolation: `outputs/<uuid>/` per request

## Success Criteria

### Phase 1
- Server starts with missing `ANTHROPIC_API_KEY` without crashing
- LLM failure returns a graceful error message (not 500 stack trace) and still returns analysis data
- `visualization.py` entity-presence detection uses identical logic to `articulation_analysis.py`
- Two simultaneous uploads produce independent chart sets with no file collisions

### Phase 2
- Non-UTF-8 `.txt` files (Latin-1, Windows-1252) are decoded correctly, not corrupted
- Uploads over 20MB are rejected with HTTP 413 before reading into memory
- `_FALSE_POSITIVES` is a dict keyed by domain; `"book"` domain gets current set, `"court"` and `"meeting"` get minimal generic stopwords

### Phase 3
- A valid PDF file produces the same analysis output structure as a `.txt` file
- A transcript file with `SPEAKER:` or `[00:01:23]` patterns is split into per-speaker/timestamp segments instead of one blob