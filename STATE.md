# State

## Current Phase
Phase 3 — Format Expansion

## Status
COMPLETE

## Completed Tasks
- [x] Phase 1: Fix LLM client init and add error handling
- [x] Phase 1: Fix entity-presence substring match bug in visualization
- [x] Phase 1: Add per-request output isolation
- [x] Phase 2: Add encoding detection in read_file() and API upload path
- [x] Phase 2: Add 20 MB file size limit to POST /analyze
- [x] Phase 2: Make _FALSE_POSITIVES a per-domain dict; thread domain through call chain
- [x] Phase 3: Add PDF ingestion via pdfplumber
- [x] Phase 3: Add LLM-detected segmentation with pre-check and fallback chain
- [x] Phase 3: Frontend accepts .pdf uploads (domain selector was already implemented)

## In Progress
(none)

## Blockers
(none)

## Notes
- PDF: read_pdf() in utils.py opens with pdfplumber, joins pages with \n\n, collapses excess blank lines. API writes bytes to tempfile, calls read_pdf(), deletes tempfile in finally block.
- Segmentation: backend/core/segmentation.py — split_into_segments(text, domain). Pre-check fires for ≥3 chapter/act markers in first 3000 chars (no LLM call). Otherwise calls Haiku with first 150 lines/4000 chars; parses JSON; builds regex from re.escape'd delimiters. Any LLM failure (exception, bad JSON, empty list, <2 segments) falls through to paragraph chunking (>200 char blocks). Final fallback: [text.strip()].
- Frontend: accept changed to ".txt,.pdf"; handleFile guard updated; drop hint text updated. Domain selector and api.ts FormData were already complete.
