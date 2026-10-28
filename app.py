import io
import pathlib
import threading
from datetime import datetime, timezone

import pandas as pd
import requests
from joblib import load, dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, BackgroundTasks, HTTPException
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

DATASET_URLS = {
    "25": "https://res.cloudinary.com/dlnympwl9/raw/upload/v1793165520/spotify-tracks_25_c9fzi2.csv",
    "50": "https://res.cloudinary.com/dlnympwl9/raw/upload/v1793165527/spotify-tracks_50_e9nhyy.csv",
    "100": "https://res.cloudinary.com/dlnympwl9/raw/upload/v1793165532/spotify-tracks_100_qrltzm.csv",
}
FEATURES = [
    "duration_ms", "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "time_signature",
]

train_state = {
    "status": "idle",  # idle | running | done | error
    "dataset": None,
    "rows": None,
    "mae": None,
    "message": "Modelo inicial cargado desde la imagen.",
    "finished_at": None,
}
train_lock = threading.Lock()

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


def _train(dataset: str):
    """Descarga el CSV de Cloudinary, entrena y reemplaza el modelo en memoria."""
    global model
    try:
        resp = requests.get(DATASET_URLS[dataset], timeout=120)
        resp.raise_for_status()
        data = pd.read_csv(io.BytesIO(resp.content)).dropna(axis=0)

        X_train, X_val, y_train, y_val = train_test_split(
            data[FEATURES], data["popularity"], test_size=0.2, random_state=0
        )
        new_model = RandomForestRegressor(
            n_estimators=50, max_depth=12, random_state=1, n_jobs=-1
        )
        new_model.fit(X_train, y_train)
        mae = float(mean_absolute_error(y_val, new_model.predict(X_val)))

        model = new_model
        try:
            dump(new_model, model_file, compress=3)
        except OSError:
            pass  # el modelo en memoria ya quedo activo

        train_state.update(
            status="done", dataset=dataset, rows=int(len(data)), mae=round(mae, 2),
            message=f"Modelo reentrenado con el dataset {dataset}%.",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:  # noqa: BLE001
        train_state.update(status="error", dataset=dataset, message=str(exc))


@app.post("/train")
def train(background_tasks: BackgroundTasks, dataset: str = "100"):
    if dataset not in DATASET_URLS:
        raise HTTPException(status_code=400, detail="dataset debe ser 25, 50 o 100")
    with train_lock:
        if train_state["status"] == "running":
            raise HTTPException(status_code=409, detail="Ya hay un entrenamiento en curso")
        train_state.update(
            status="running", dataset=dataset, rows=None, mae=None,
            message=f"Entrenando con el dataset {dataset}%...", finished_at=None,
        )
    background_tasks.add_task(_train, dataset)
    return train_state


@app.get("/train/status")
def train_status():
    return train_state
