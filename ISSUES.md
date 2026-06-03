# Issues

## CRITICAL — Crash or Silent Data Corruption

### ISS-001: Server crashes at startup if ANTHROPIC_API_KEY is missing
- **File**: `backend/narrative_llm.py:5`
- **Code**: `_CLIENT = anthropic.Anthropic()` runs at module import time
- **Impact**: Entire API process dies before serving any request
- **Fix in**: Phase 1, Task 1
- **Status**: OPEN

### ISS-002: LLM API call has no error handling
- **File**: `backend/narrative_llm.py:100-128`
- **Code**: `_CLIENT.messages.create(...)` — no try/except
- **Impact**: Any API error (rate limit, network, auth) propagates as unhandled HTTP 500; all computed analysis data is lost in the response
- **Fix in**: Phase 1, Task 1
- **Status**: OPEN

### ISS-003: Entity-presence detection inconsistency between visualization and analysis
- **File**: `backend/core/visualization.py:164`
- **Code**: `present = [n for n in names if n.lower() in chapter.content.lower()]`
- **Correct version**: `backend/analysis/articulation_analysis.py:18` uses `re.search(r"\b" + re.escape(n.lower()) + r"\b", ...)`
- **Impact**: Character network chart uses different edge set than the analysis that computed articulation points — visual output is incorrect. Short entity names (e.g. "Al", "Ed") produce false-positive edges.
- **Fix in**: Phase 1, Task 2
- **Status**: OPEN

### ISS-004: Concurrent uploads overwrite each other's output files
- **Files**: All `visualize_*` functions in `backend/core/visualization.py`, `backend/core/summary_report.py`
- **Code**: All charts saved to fixed names under module-level `_OUT = .../outputs/`
- **Impact**: Two simultaneous API requests corrupt each other's PNG charts and summary report; second response returns charts from first request
- **Fix in**: Phase 1, Task 3
- **Status**: OPEN

---

## HIGH — Incorrect Behavior on Valid Input

### ISS-005: CLI crashes on non-UTF-8 encoded .txt files
- **File**: `backend/core/utils.py:8`
- **Code**: `open(file_path, "r", encoding="utf-8")` — no error handling
- **Impact**: Any Windows-1252 or Latin-1 encoded file throws `UnicodeDecodeError`; common with Gutenberg downloads and older literature sources
- **Fix in**: Phase 2, Task 1
- **Status**: OPEN

### ISS-006: API silently corrupts non-UTF-8 file content
- **File**: `backend/api.py:31`
- **Code**: `(await file.read()).decode("utf-8", errors="replace")`
- **Impact**: Non-UTF-8 bytes replaced with `�` — content is destroyed before analysis, no warning to user
- **Fix in**: Phase 2, Task 1
- **Status**: OPEN

### ISS-007: No file size limit — memory exhaustion / DoS risk
- **File**: `backend/api.py:23-35`
- **Code**: `await file.read()` with no prior size check
- **Impact**: Arbitrarily large uploads are fully read into memory; can exhaust server RAM or stall the event loop
- **Fix in**: Phase 2, Task 2
- **Status**: OPEN

### ISS-008: Shakespeare-specific entity false-positive filter applied to all domains
- **File**: `backend/analysis/entity_extraction.py:14-21`
- **Code**: `_FALSE_POSITIVES` is a flat set including "verona", "mantua", "rome", "italy" and archaic pronouns
- **Impact**: Court transcripts and meeting notes suppress valid place names and common words that happen to match; domain parameter has no effect on filtering
- **Fix in**: Phase 2, Task 3
- **Status**: OPEN

---

## MEDIUM — Format / Segmentation Gaps

### ISS-009: PDF files cannot be ingested
- **Files**: `backend/api.py:28`, `frontend/src/components/FileUpload.tsx`
- **Impact**: Most research papers, legal filings, and many books are distributed as PDF; hard-blocked at upload
- **Fix in**: Phase 3, Task 1
- **Status**: OPEN

### ISS-010: Conversation transcripts produce no segmentation
- **Files**: `backend/core/utils.py:split_into_chapters()`
- **Impact**: Transcripts with `SPEAKER:`, `[00:01:23]`, or `Q:` / `A:` patterns return as a single segment; mood flow, arc detection, and entity tracking are meaningless with one data point
- **Fix in**: Phase 3, Task 2
- **Status**: OPEN

### ISS-011: Domain parameter changes only LLM framing, not processing
- **File**: `backend/narrative_llm.py:51-56` (only consumer of `domain`)
- **Impact**: User selects "court" or "meeting" domain but gets identical analysis to "book" — same chapter splitting, same entity filters, same segmentation
- **Fix in**: Phase 3, Task 2 (segmentation) + Phase 2, Task 3 (entity filters)
- **Status**: OPEN

---

## LOW — Minor Correctness Issues

### ISS-012: `read_file()` CLI path has no error message on failure
- **File**: `backend/core/utils.py:7-9`
- **Impact**: UnicodeDecodeError stack trace with no user-friendly message
- **Fix alongside**: ISS-005

### ISS-013: `split_into_chapters()` returns single chunk with no warning when no markers found
- **File**: `backend/core/utils.py:12-20`
- **Impact**: Unstructured text silently degrades to single-segment analysis with no indication to user
- **Note**: Low priority; addressing ISS-010 (transcript segmentation) partially mitigates this

### ISS-014: Output directory not cleaned up between runs
- **Files**: `backend/core/visualization.py`, `backend/core/summary_report.py`
- **Impact**: Old charts accumulate on disk indefinitely; disk space leak
- **Note**: Mitigated once ISS-004 is fixed (UUID dirs isolate runs); add cleanup as follow-on
