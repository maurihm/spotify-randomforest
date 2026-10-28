import pathlib
import pandas as pd
from joblib import load
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(
    title="Spotify Track Popularity Prediction",
    description="API para predecir la popularidad de canciones en Spotify utilizando Random Forest",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Cargar modelo serializado
model_file = pathlib.Path(__file__).resolve().parent / "model" / "spotify-rf-v1.joblib"
model = load(model_file)

html_file = pathlib.Path(__file__).resolve().parent / "index.html"


class InputData(BaseModel):
    duration_ms: int = 230666
    danceability: float = 0.676
    energy: float = 0.461
    key: int = 1
    loudness: float = -6.746
    mode: int = 0
    speechiness: float = 0.143
    acousticness: float = 0.0322
    instrumentalness: float = 0.000001
    liveness: float = 0.358
    valence: float = 0.715
    tempo: float = 87.917
    time_signature: int = 4


class OutputData(BaseModel):
    score: float = 73.0


@app.get("/", response_class=HTMLResponse)
def read_root():
    if html_file.exists():
        return html_file.read_text(encoding="utf-8")
    return "<h1>Spotify Popularity Predictor API</h1><p>Visita <a href='/docs'>/docs</a> para ver la documentación interactiva.</p>"


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Spotify Track Popularity Prediction API",
        "docs_url": "/docs"
    }


@app.post("/score", response_model=OutputData)
def score(data: InputData):
    df_input = pd.DataFrame([{
        "duration_ms": data.duration_ms,
        "danceability": data.danceability,
        "energy": data.energy,
        "key": data.key,
        "loudness": data.loudness,
        "mode": data.mode,
        "speechiness": data.speechiness,
        "acousticness": data.acousticness,
        "instrumentalness": data.instrumentalness,
        "liveness": data.liveness,
        "valence": data.valence,
        "tempo": data.tempo,
        "time_signature": data.time_signature
    }])
    prediction = float(model.predict(df_input)[0])
    return {"score": round(prediction, 2)}
