import shutil
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODEL_PATH
from src.landmarks import create_hands, extract_frame_landmarks
from src.predictor import predictor

app = FastAPI(
    title="Assistive Communication Backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Assistive Communication backend is running",
        "model_ready": MODEL_PATH.exists(),
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.post("/predict/video")
async def predict_video(video: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained yet.")

    suffix = Path(video.filename or "video.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to read uploaded video.")

    predictor.reset()
    last_result = None

    try:
        with create_hands() as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                features = extract_frame_landmarks(frame, hands)
                result = predictor.add_features(features)
                if result.get("success"):
                    last_result = result

        if last_result:
            return last_result

        return {
            "success": False,
            "message": "No supported sign detected or video was too short."
        }
    finally:
        cap.release()
        Path(temp_path).unlink(missing_ok=True)
