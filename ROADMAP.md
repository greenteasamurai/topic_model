# Roadmap — Narrative Analysis Critical Fixes

## Phase 1: Crash & Correctness Bugs ✓
*Fix things that produce wrong output or crash the server*

- [x] Lazy-init Anthropic client; wrap LLM call in try/except with graceful fallback
- [x] Fix entity-presence detection in `visualization.py:164` to use word-boundary regex (match `articulation_analysis.py`)
- [x] Add per-request output isolation (`outputs/<uuid>/`) so concurrent uploads don't collide

## Phase 2: Ingestion Hardening ✓
*Fix things that silently corrupt input or allow abuse*

- [x] Add encoding detection (`chardet`) in `read_file()` and API upload path
- [x] Add 20MB file size limit to `POST /analyze` before reading body into memory
- [x] Make `_FALSE_POSITIVES` a per-domain dict; `"court"` and `"meeting"` domains get minimal stopword set

## Phase 3: Format Expansion ✓
*Support PDF ingestion and LLM-detected segmentation for any text structure*

- [x] Add PDF ingestion via `pdfplumber`; extract text and route through existing pipeline
- [x] Add LLM-detected segmentation: sample first 150 lines with Haiku, return observed delimiter strings, build regex from escaped literals; fall back to paragraph chunking on failure or empty result
- [x] Frontend: accept `.pdf` uploads; add domain selector (book / court / meeting)
