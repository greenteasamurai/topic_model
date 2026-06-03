# Plan — Phase 2: Ingestion Hardening
*(active)*

```xml
<task>
  <name>Add encoding detection to read_file and API upload</name>
  <files>backend/core/utils.py, backend/api.py</files>
  <actions>
    1. In utils.py, rewrite read_file() to read raw bytes with Path.read_bytes(), run
       chardet.detect() on them, decode with detected encoding (fallback to utf-8)
    2. In api.py, replace `await file.read()` with a size-checked read: read up to
       MAX_UPLOAD_BYTES+1 bytes; if over limit raise HTTP 413
    3. After the size check, run chardet.detect() on the raw bytes and decode with
       detected encoding (fallback to utf-8) instead of hardcoded utf-8 with errors=replace
  </actions>
  <verification>
    py -3.14 -c "from core.utils import read_file" from backend/ — import OK.
    Manually pass a Latin-1 bytes object through chardet to confirm encoding detected correctly.
  </verification>
  <success>
    read_file() decodes non-UTF-8 files without UnicodeDecodeError.
    API upload detects encoding from bytes rather than assuming UTF-8.
    Files over 20 MB return HTTP 413 before content is read into memory.
  </success>
</task>

<task>
  <name>Make _FALSE_POSITIVES per-domain</name>
  <files>backend/analysis/entity_extraction.py, backend/main.py</files>
  <actions>
    1. In entity_extraction.py, replace _FALSE_POSITIVES flat set with
       _FALSE_POSITIVES_BY_DOMAIN dict keyed by domain string:
       "book" gets current set; "court" and "meeting" get small generic stopword sets
    2. Add domain: str = "book" parameter to extract_important_entities()
    3. Inside the function, resolve false_positives = _FALSE_POSITIVES_BY_DOMAIN.get(domain, set())
    4. In main.py, add domain: str = "book" parameter to analyze_book() and pass it
       through to both extract_important_entities() calls
    5. In api.py, pass domain=domain to analyze_book()
  </actions>
  <verification>
    py -3.14 -c "from analysis.entity_extraction import extract_important_entities; print('OK')"
    Confirm "verona" and "rome" are in the "book" false-positive set but not "court".
  </verification>
  <success>
    extract_important_entities() accepts domain parameter.
    "verona"/"rome" suppressed for domain="book", not for domain="court".
    analyze_book() and api.py correctly thread domain through the call chain.
  </success>
</task>
```

---

# Plan — Phase 1: Crash & Correctness Bugs
*(complete)*

```xml
<task>
  <name>Fix LLM client init and add error handling</name>
  <files>backend/narrative_llm.py</files>
  <actions>
    1. Remove module-level `_CLIENT = anthropic.Anthropic()` assignment
    2. Create a `_get_client()` function that instantiates and returns `anthropic.Anthropic()` lazily
    3. Wrap the `_CLIENT.messages.create(...)` call in `analyze_with_llm()` with try/except catching `anthropic.APIError` and `Exception`
    4. On error, return a fallback string describing the failure (e.g. "LLM analysis unavailable: {error}") so the API route can still return analysis data
  </actions>
  <verification>
    Start backend with ANTHROPIC_API_KEY unset; confirm server starts without error.
    Call POST /analyze with a valid .txt file; confirm response includes analysis data even when LLM call fails.
  </verification>
  <success>
    Server starts without ANTHROPIC_API_KEY set.
    LLM failure returns a string fallback in `narrative_analysis` field, not an HTTP 500.
  </success>
</task>

<task>
  <name>Fix entity-presence substring match bug in visualization</name>
  <files>backend/core/visualization.py</files>
  <actions>
    1. Add `import re` at top of file (if not already present)
    2. Locate line 164: `present = [n for n in names if n.lower() in chapter.content.lower()]`
    3. Replace with: `present = [n for n in names if re.search(r"\b" + re.escape(n.lower()) + r"\b", chapter.content.lower())]`
    4. Confirm no other `in chapter.content` substring checks exist in the file
  </actions>
  <verification>
    Run `python backend/main.py` against RomeoAndJuliet.txt; confirm articulation_network.png generates without error.
    Manually verify: entity "Al" does not appear as present in chapters containing only "also" or "already".
  </verification>
  <success>
    `visualize_articulation_structure` uses word-boundary regex matching identical in logic to `articulation_analysis.py:18`.
    No regression in chart generation for existing sample books.
  </success>
</task>

<task>
  <name>Add per-request output isolation</name>
  <files>backend/api.py, backend/main.py, backend/core/visualization.py, backend/core/summary_report.py</files>
  <actions>
    1. In `analyze_book()` signature (main.py), add `output_dir: Path | None = None` parameter
    2. Pass `output_dir` through to `visualize_*` functions and `generate_summary_report()`
    3. In each visualization function and `generate_summary_report`, replace the module-level `_OUT` path with the passed-in `output_dir` (fall back to `_OUT` if None, preserving CLI behavior)
    4. In `api.py POST /analyze`, generate `output_dir = _OUTPUTS / str(uuid.uuid4())` before calling `analyze_book()`; mkdir it
    5. In the API response `charts` dict, replace `/charts/filename.png` with `/charts/<uuid>/filename.png`
    6. Add `import uuid` to api.py
  </actions>
  <verification>
    Start the API server. Send two simultaneous POST /analyze requests (two different .txt files).
    Confirm each response has a different UUID prefix in chart URLs.
    Confirm both sets of charts exist in separate subdirectories under outputs/.
  </verification>
  <success>
    Concurrent uploads produce charts in separate `outputs/<uuid>/` directories.
    CLI `python backend/main.py` still writes to `outputs/` (no UUID, backward-compatible).
    Chart URLs in API response correctly point to the per-request subdirectory.
  </success>
</task>
```

---

# Plan — Phase 3: Format Expansion
*(pending Phase 2 completion)*

```xml
<task>
  <name>Add PDF ingestion</name>
  <files>backend/core/utils.py, backend/api.py, requirements.txt (or pyproject.toml)</files>
  <actions>
    1. Add `pdfplumber` to project dependencies
    2. In `utils.py`, add `read_pdf(file_path: str) -> str` that opens with pdfplumber,
       concatenates `page.extract_text() or ""` for each page, strips excessive whitespace
    3. In `read_file()`, detect `.pdf` extension and delegate to `read_pdf()`
    4. In `api.py`, change the extension check from `endswith(".txt")` to
       `endswith((".txt", ".pdf"))`; decode branch: if `.pdf`, write bytes to a temp file,
       call `read_pdf()`, then delete temp file
  </actions>
  <verification>
    Upload a PDF of one of the sample books (convert RomeoAndJuliet.txt to PDF).
    Confirm POST /analyze returns a valid response with chapters, mood_flow, and charts.
    Confirm a .txt upload still works unchanged.
  </verification>
  <success>
    PDF upload produces equivalent analysis output to the same text as .txt.
    .txt path has no regression.
    Temp file is deleted after extraction regardless of success or failure.
  </success>
</task>

<task>
  <name>Add LLM-detected segmentation</name>
  <files>backend/core/utils.py, backend/narrative_llm.py (or new backend/core/segmentation.py)</files>
  <actions>
    1. Create `detect_segment_strategy(sample: str) -> dict` in a new
       `backend/core/segmentation.py`; returns
       `{"doc_type": str, "segment_label": str, "delimiters": list[str]}`
    2. Prompt: send first 150 lines to `claude-haiku-4-5-20251001` with instruction to
       return only strings observed verbatim in the sample that serve as segment boundaries;
       use `max_tokens=200`; parse response as JSON
    3. Add a fast pre-check in `split_into_segments()`: if the existing chapter/act regex
       finds ≥ 3 matches in `text[:3000]`, skip LLM and use current logic
    4. Replace `split_into_chapters(text)` with `split_into_segments(text, domain)`:
       — run pre-check; if passes, use existing regex path
       — otherwise call `detect_segment_strategy(text[:4000])`
       — if `delimiters` non-empty, build pattern with `"|".join(re.escape(d) for d in delimiters)`
         and `re.split(pattern, text, flags=re.IGNORECASE)`
       — if zero or one segment results, fall back to paragraph chunking:
         `[s.strip() for s in re.split(r"\n{2,}", text) if len(s.strip()) > 200]`
    5. Update callers in `main.py` and `api.py` to pass `domain` to `split_into_segments()`
    6. LLM call must be wrapped in try/except; any failure goes straight to paragraph fallback
  </actions>
  <verification>
    Upload a plain court transcript .txt with DIRECT EXAMINATION / CROSS-EXAMINATION markers.
    Confirm segments correspond to examination blocks, not a single blob.
    Upload a meeting transcript with "Agenda Item" headers; confirm segments per agenda item.
    Upload RomeoAndJuliet.txt; confirm pre-check fires and existing regex path is used (no LLM call).
    Kill ANTHROPIC_API_KEY mid-test; confirm paragraph-chunking fallback produces > 1 segment.
  </verification>
  <success>
    Court transcript with examination markers produces ≥ 3 segments aligned to examination blocks.
    Novel bypasses LLM via pre-check; segmentation output identical to current behavior.
    Any LLM failure (API error, malformed JSON, empty delimiters) falls back gracefully;
    analysis still completes with ≥ 1 segment.
    Haiku model used (not Sonnet) to keep classification latency and cost minimal.
  </success>
</task>

<task>
  <name>Frontend: PDF upload and domain selector</name>
  <files>frontend/src/components/FileUpload.tsx, frontend/src/api.ts</files>
  <actions>
    1. In `FileUpload.tsx`, change the `accept` attribute from `".txt"` to `".txt,.pdf"`
    2. Add a `<select>` dropdown with options: Book, Court Transcript, Meeting Transcript;
       default to Book; store selection in component state as `domain`
    3. Pass `domain` into `analyzeFile()` call alongside the file
    4. In `api.ts`, include `domain` as a FormData field in the POST body
    5. Display the `domain` value returned in the API response alongside the title
  </actions>
  <verification>
    Open frontend dev server. Upload a PDF; confirm it is accepted and analysis renders.
    Change domain selector to "Court Transcript"; confirm value is sent in request
    (check Network tab: FormData should include domain=court).
    Confirm .txt upload still works with all three domain options.
  </verification>
  <success>
    PDF files accepted by file picker without error.
    Domain selector value reaches the API as the `domain` form field.
    No regression on .txt uploads or existing chart rendering.
  </success>
</task>
```
