from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from main import analyze_book
from narrative_llm import analyze_with_llm

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
    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    text = (await file.read()).decode("utf-8", errors="replace")
    title = Path(file.filename).stem

    book = analyze_book(text, title=title)
    narrative = analyze_with_llm(book, domain=domain)

    scores = [ch.mood.vader_sentiment["compound"] for ch in book.chapters]
    diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    turning_point_scene = diffs.index(min(diffs)) + 1 if diffs else None

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
            "mood_flow": "/charts/mood_flow.png",
            "character_network": "/charts/character_network.png",
            "articulation_network": "/charts/articulation_network.png",
            "entity_flow": "/charts/entity_flow.png",
            "emotion_distribution": "/charts/emotion_distribution.png",
            "character_mood_impact": "/charts/character_mood_impact.png",
            "character_impact_scatter": "/charts/character_impact_scatter.png",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}
