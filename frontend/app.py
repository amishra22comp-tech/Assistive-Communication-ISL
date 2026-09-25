"""
Assistive Communication - ISL to English (Frontend)

Matches the REAL backend contract (backend/src/main.py):

  POST /predict/video
    multipart/form-data, field name "video" -> a short video clip (mp4)
    Response: {"success": bool, "sentence": str, "confidence": float}
              or {"success": false, "message": str}

  GET /health
    Response: {"status": "ok", "model_ready": bool}

All landmark extraction and model inference happen on the BACKEND.
This frontend only needs to: preview the webcam, record a short clip,
upload it, and speak the result.
"""

import io
import time

import av
import cv2
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from gtts import gTTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{BACKEND_URL}/predict/video"
HEALTH_ENDPOINT = f"{BACKEND_URL}/health"

RECORD_SECONDS = 2.5     # how long a clip to record per sign
REQUEST_TIMEOUT = 15.0   # uploading + server-side mediapipe processing takes longer than a JSON call

st.set_page_config(page_title="ISL Assistive Communication", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
defaults = {
    "sentence_words": [],
    "recording": False,
    "record_start_time": None,
    "frame_buffer": [],
    "frame_size": None,
    "last_result_message": None,
    "backend_status": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def check_backend_health():
    try:
        resp = requests.get(HEALTH_ENDPOINT, timeout=3.0)
        resp.raise_for_status()
        st.session_state.backend_status = resp.json()
    except Exception as e:
        st.session_state.backend_status = {"status": "unreachable", "error": str(e)}


class RecorderProcessor:
    """Shows a live preview and, while recording is active, buffers raw frames."""

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        if st.session_state.recording:
            st.session_state.frame_buffer.append(img.copy())
            st.session_state.frame_size = (img.shape[1], img.shape[0])  # (w, h)

            elapsed = time.time() - st.session_state.record_start_time
            remaining = max(0.0, RECORD_SECONDS - elapsed)
            cv2.putText(
                img, f"REC {remaining:.1f}s", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA,
            )
            if elapsed >= RECORD_SECONDS:
                st.session_state.recording = False
        else:
            cv2.putText(
                img, "Ready", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    return st.session_state.recorder.process(frame)


if "recorder" not in st.session_state:
    st.session_state.recorder = RecorderProcessor()


def frames_to_mp4_bytes(frames, size, fps=20) -> bytes:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = "/tmp/isl_clip.mp4"
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, size)
    for f in frames:
        writer.write(f)
    writer.release()
    with open(tmp_path, "rb") as f:
        return f.read()


def send_clip_to_backend(video_bytes: bytes):
    files = {"video": ("clip.mp4", video_bytes, "video/mp4")}
    try:
        resp = requests.post(PREDICT_ENDPOINT, files=files, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 503:
            return {"success": False, "message": "Backend model is not trained yet (503)."}
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"success": False, "message": f"Request failed: {e}"}


def text_to_speech_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    gTTS(text=text, lang="en").write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🤟 ISL Assistive Communication")
st.caption("Record a short clip of a sign → backend predicts it → build a sentence → speak it aloud.")

if st.session_state.backend_status is None:
    check_backend_health()

status = st.session_state.backend_status or {}
if status.get("status") == "ok":
    if status.get("model_ready"):
        st.success("Backend connected — model is trained and ready.")
    else:
        st.warning(
            "Backend is connected, but the model isn't trained yet "
            "(`models/isl_sentence_model.keras` not found). Predictions will fail until it's trained."
        )
else:
    st.error(f"Can't reach backend at {BACKEND_URL}. Is `uvicorn src.main:app --reload` running?")

if st.button("🔄 Recheck backend status"):
    check_backend_health()
    st.rerun()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera")
    webrtc_streamer(
        key="isl-stream",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if st.button("🔴 Record sign", disabled=st.session_state.recording):
        st.session_state.recording = True
        st.session_state.record_start_time = time.time()
        st.session_state.frame_buffer = []

    if st.session_state.recording:
        st.info(f"Recording for {RECORD_SECONDS}s — hold the sign steady.")
    st.caption(f"Backend: {BACKEND_URL}")

with col2:
    st.subheader("Last prediction")
    if st.session_state.last_result_message:
        st.write(st.session_state.last_result_message)

    st.subheader("Sentence")
    sentence = " ".join(st.session_state.sentence_words)
    st.text_area("Built from recognized signs:", value=sentence, height=100, key="sentence_box")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🔊 Speak"):
            if sentence.strip():
                audio_bytes = text_to_speech_bytes(sentence)
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.warning("No words yet.")
    with b2:
        if st.button("⌫ Undo word"):
            if st.session_state.sentence_words:
                st.session_state.sentence_words.pop()
                st.rerun()
    with b3:
        if st.button("🗑 Clear"):
            st.session_state.sentence_words = []
            st.rerun()

# After a recording just finished (buffer full, recording flag turned off by the
# callback), send it to the backend once.
if (
    not st.session_state.recording
    and st.session_state.frame_buffer
    and st.session_state.frame_size
):
    with st.spinner("Uploading clip and waiting for prediction..."):
        video_bytes = frames_to_mp4_bytes(
            st.session_state.frame_buffer, st.session_state.frame_size
        )
        result = send_clip_to_backend(video_bytes)

    st.session_state.frame_buffer = []  # clear so we don't resend

    if result.get("success"):
        word = result["sentence"]
        confidence = result.get("confidence", 0)
        st.session_state.last_result_message = f"✅ **{word}** ({confidence}% confidence)"
        st.session_state.sentence_words.append(word)
    else:
        st.session_state.last_result_message = f"⚠️ {result.get('message', 'No prediction.')}"

    st.rerun()

st.divider()
with st.expander("How this works"):
    st.markdown(
        """
        1. Click **Record sign** and hold the sign steady in front of the camera for
           about 2.5 seconds.
        2. The recorded clip is uploaded to the backend's `POST /predict/video`.
        3. The backend runs MediaPipe landmark extraction and the trained model on the
           clip, and returns the recognized word/sentence with a confidence score.
        4. A successful prediction is appended to the sentence on the right.
        5. Press **Speak** to convert the built sentence to audio.

        All ML work (landmark extraction + inference) happens server-side — this
        frontend just records and uploads clips.
        """
    )
