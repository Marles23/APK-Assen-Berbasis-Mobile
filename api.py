from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

from youtube_service import get_comments_for_video
from sentiment_service import analyze_list

app = FastAPI()

# CORS CONFIG
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # DEV MODE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YoutubeRequest(BaseModel):
    video_id: str
    max_comments: int | None = 300   # opsional, default 100 komentar

@app.get("/")
def root():
    return {"message": "Backend YouTube Sentiment API siap"}

@app.post("/analyze-youtube")
def analyze_youtube(req: YoutubeRequest) -> Dict:
    video_id = req.video_id.strip()
    if not video_id:
        raise HTTPException(status_code=400, detail="video_id tidak boleh kosong")

    max_comments = req.max_comments or 300
    if max_comments < 1:
        max_comments = 1
    if max_comments > 300:
        max_comments = 300

    # 1. Ambil komentar dari YouTube
    comments_raw = get_comments_for_video(video_id, max_comments=max_comments)

    if not comments_raw:
        return {
            "summary": {"positive": 0, "neutral": 0, "negative": 0},
            "comments": [],
        }

    texts = [c["text"] for c in comments_raw]

    # 2. Analisis sentimen pakai SVM
    analyzed = analyze_list(texts)

    # 3. Gabungkan & hitung summary
    results: List[Dict] = []
    pos = neu = neg = 0

    for raw, sent in zip(comments_raw, analyzed):
        sentiment = sent["sentiment"]
        s_lower = sentiment.lower()

        if s_lower.startswith("pos"):
            pos += 1
        elif s_lower.startswith("neg"):
            neg += 1
        else:
            neu += 1

        results.append({
            "author": raw["author"],
            "text": raw["text"],
            "sentiment": sentiment,  # akan tampil di badge Flutter
            "score": sent["score"],
        })

    summary = {"positive": pos, "neutral": neu, "negative": neg}

    return {
        "summary": summary,
        "comments": results,
    }
