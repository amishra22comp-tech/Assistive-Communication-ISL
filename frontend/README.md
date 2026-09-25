# ISL Frontend

A Streamlit frontend for the `Assistive-Communication-ISL` backend
(`backend/src/main.py`). Records a short webcam clip, uploads it to the
backend, shows the predicted word/sentence, builds a spoken sentence, and
speaks it with text-to-speech.

## How it talks to the backend

- `GET /health` — checked on load, to show whether the backend is reachable
  and whether the model is trained.
- `POST /predict/video` — a recorded clip (`multipart/form-data`, field name
  `video`) is uploaded here. Response:
  ```json
  {"success": true, "sentence": "hello", "confidence": 92.4}
  ```
  or, on failure:
  ```json
  {"success": false, "message": "No supported sign detected"}
  ```

All MediaPipe landmark extraction and model inference happen **on the
backend** — this frontend just records and uploads video clips.

## Setup

```bash
cd frontend
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
streamlit run app.py
```

Make sure the backend is running first (`uvicorn src.main:app --reload` from
the `backend/` folder), and edit `.streamlit/secrets.toml` if it's not on
`http://localhost:8000`.

## ⚠️ Backend issues found while building this

While wiring the frontend up to the real `main.py`, two things in the
current backend will stop predictions from working — worth fixing there
before testing end-to-end:

1. **Feature-count mismatch.** `backend/src/config.py` sets
   `FEATURES_PER_FRAME = 225` (2 hands + full body pose), but
   `backend/src/landmarks.py`'s `extract_frame_landmarks()` only extracts
   the 2 hands (126 features). The `assert features.shape[0] ==
   FEATURES_PER_FRAME` in that function will raise on every request until
   either the config is changed to 126, or pose extraction is added to
   match 225.

2. **Model not trained yet.** `config.py`'s `MODEL_PATH` points to
   `models/isl_sentence_model.keras`, which isn't in the repo — only the
   word-level model (`models/isl_word_model.keras`, 10 words) exists. Until
   the sentence model is trained, `/predict/video` will return `503 Model
   is not trained yet.` The frontend already handles this gracefully (shows
   a warning banner), so you can still confirm the recording/upload flow
   works even before training finishes.
