# Backend - Assistive Communication App

## Pipeline
Video/Webcam -> MediaPipe landmarks -> 126 features per frame -> Conv1D + LSTM -> 10 ISL sentence classes -> FastAPI response

## IMPORTANT
1. Do not train until you place the dataset in `dataset/raw/`.
2. Run `python -m src.inspect_dataset` first to understand the downloaded dataset.
3. If the dataset's sentence labels do not match our chosen 10 classes, update `src/config.py`.
4. Convert/extract the videos into landmark `.npy` sequences with `extract_landmarks.py`.
5. Train with `train_model.py`.
6. Run the API with `uvicorn src.main:app --reload`.

## Local setup (Windows PowerShell)
```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Development commands
```powershell
python -m src.inspect_dataset
python -m src.extract_landmarks
python -m src.train_model
python -m src.evaluate_model
uvicorn src.main:app --reload
```
