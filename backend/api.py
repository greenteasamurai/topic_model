import asyncio
import uuid
import tempfile
from pathlib import Path
import chardet
import nltk
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from core.utils import read_pdf, turning_point_index
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from main import analyze_book
from narrative_llm import analyze_with_llm

for _corpus in ("punkt", "punkt_tab", "stopwords", "vader_lexicon"):
    try:
        nltk.download(_corpus, quiet=True, raise_on_error=True)
    except Exception:
        pass

_OUTPUTS = Path(__file__).parent.parent / "outputs"
_OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="Narrative Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount("/charts", StaticFiles(directory=str(_OUTPUTS)), name="charts")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    domain: str = Form(default="book"),
):
    fname = file.filename or ""
    if not fname.endswith((".txt", ".pdf")):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    _MAX_BYTES = 20 * 1024 * 1024  # 20 MB
    raw = await file.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 20 MB limit.")

    if fname.endswith(".pdf"):
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            text = read_pdf(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        text = raw.decode(encoding, errors="replace")

    title = Path(fname).stem

    request_id = str(uuid.uuid4())
    output_dir = _OUTPUTS / request_id
    output_dir.mkdir(parents=True, exist_ok=True)

    book = await asyncio.to_thread(analyze_book, text, title=title, output_dir=output_dir, domain=domain)
    narrative = await asyncio.to_thread(analyze_with_llm, book, domain=domain)

    scores = [ch.mood.vader_sentiment["compound"] for ch in book.chapters]
    tp_idx = turning_point_index(scores)
    turning_point_scene = tp_idx + 1 if tp_idx is not None else None

    return {
        "title": book.title,
        "domain": domain,
        "narrative_analysis": narrative,
        "characters": [
            {
                "name": imp.name,
                "cohens_d": round(imp.cohens_d, 4),
                "lagged_delta": round(imp.lagged_delta, 4),
                "mood_delta": round(imp.mood_delta, 4),
                "presence_avg": round(imp.presence_avg, 4),
                "scene_count": imp.scene_count,
                "crisis_scene_count": imp.crisis_scene_count,
                "role": next(
                    (cr.role for cr in (book.character_roles or []) if cr.name == imp.name),
                    None,
                ),
            }
            for imp in book.character_impacts
        ],
        "mood_flow": [
            {
                "scene": ch.number,
                "score": round(ch.mood.vader_sentiment["compound"], 4),
                "arc": ch.arc_label,
                "entities": [e.name for e in ch.entities[:2]],
                "dominant_emotion": max(ch.mood.emotions, key=ch.mood.emotions.get),
            }
            for ch in book.chapters
        ],
        "topics": [
            {"id": t.id, "keywords": t.keywords}
            for t in book.themes
        ],
        "turning_point_scene": turning_point_scene,
        "charts": {
            "mood_flow": f"/charts/{request_id}/mood_flow.png",
            "character_network": f"/charts/{request_id}/character_network.png",
            "articulation_network": f"/charts/{request_id}/articulation_network.png",
            "entity_flow": f"/charts/{request_id}/entity_flow.png",
            "emotion_distribution": f"/charts/{request_id}/emotion_distribution.png",
            "character_mood_impact": f"/charts/{request_id}/character_mood_impact.png",
            "character_impact_scatter": f"/charts/{request_id}/character_impact_scatter.png",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}

_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
