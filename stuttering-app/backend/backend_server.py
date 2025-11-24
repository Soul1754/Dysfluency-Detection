import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
import os
import uuid
import sys
from pydub import AudioSegment 

# Import our custom engines
# Ensure inference_engine.py and transcription_engine.py are in the same folder
from inference_engine import StutterPredictor
from transcription_engine import SpeechCorrector

# --- CONFIGURATION ---
MODEL_PATH = r"../../models/stutter_model_epoch_10.pth" 
UPLOAD_DIR = "temp_uploads"

app = FastAPI(title="FluencyFlow Backend")

# CORS: Allow connection from React (running on port 3000 or 5173 usually)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- STARTUP CHECKS ---
print("\n--- SYSTEM STARTUP CHECKS ---")

ffmpeg_status = "✓ Found" if shutil.which("ffmpeg") else "MISSING"
model_status = "✓ Found" if os.path.exists(MODEL_PATH) else "MISSING (Using random weights)"

print(f"FFmpeg: {ffmpeg_status}")
print(f"Model:  {model_status}")

if ffmpeg_status == "MISSING":
    print("CRITICAL WARNING: Audio conversion will fail. Install FFmpeg and add to PATH.")

# Initialize Engines
try:
    stutter_engine = StutterPredictor(MODEL_PATH)
    transcription_engine = SpeechCorrector("openai/whisper-tiny.en")
    print("--- SYSTEM READY ---\n")
except Exception as e:
    print(f"CRITICAL ERROR during model initialization: {e}")
    sys.exit(1)

# --- DATA MODELS ---
class StutterEvent(BaseModel):
    type: str
    start: float
    end: float

class AnalysisResponse(BaseModel):
    filename: str
    original_transcript: str
    corrected_transcript: str
    fluency_score: int
    events: List[StutterEvent]
    feedback: List[str]

# --- HELPERS ---
def convert_to_wav(input_path):
    """Converts WebM/Ogg (Browser Audio) to 16kHz WAV"""
    print(f"Converting file: {input_path}")
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        output_path = input_path + "_clean.wav"
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"FFmpeg Error: {e}")
        # Return None to signal failure
        return None

def generate_feedback(events):
    feedback = []
    counts = {"block": 0, "prolongation": 0, "repetition": 0}
    for e in events:
        if e['type'] == 'block': counts['block'] += 1
        elif e['type'] == 'prolongation': counts['prolongation'] += 1
        elif 'rep' in e['type']: counts['repetition'] += 1

    if counts['prolongation'] > 0: feedback.append("Prolongation detected. Technique: Use 'Easy Onset' (start vowels gently).")
    if counts['block'] > 0: feedback.append("Block detected. Technique: Use 'Pull-out' (stop, relax, stretch sound).")
    if counts['repetition'] > 0: feedback.append("Repetition detected. Technique: Use 'Pacing' (slow down rhythm).")
    
    if not feedback: feedback.append("Excellent fluency! Keep maintaining this rhythm.")
    return feedback

# --- ENDPOINTS ---

@app.get("/")
def root():
    return {"status": "online", "ffmpeg": ffmpeg_status}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    temp_path = os.path.join(UPLOAD_DIR, f"{file_id}_raw") 
    clean_path = None
    
    print(f"\n[Request] Analyzing: {file.filename}")

    try:
        # 1. Save Raw File
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Convert Audio
        clean_path = convert_to_wav(temp_path)
        if not clean_path:
            # Raise HTTP 500 to frontend if conversion fails
            raise HTTPException(status_code=500, detail="Server Error: Audio conversion failed. Is FFmpeg installed on the backend?")

        # 3. Analyze
        try:
            events = stutter_engine.predict(clean_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model Inference Error: {str(e)}")

        # 4. Transcribe
        try:
            transcript, corrected = transcription_engine.transcribe(clean_path)
        except Exception as e:
            print(f"Transcription warning: {e}")
            transcript, corrected = ("(Transcription Unavailable)", "(Correction Unavailable)")

        # 5. Score & Feedback
        penalty = sum([min((e['end'] - e['start']) * 10, 10) for e in events])
        score = max(0, int(100 - penalty))
        
        return {
            "filename": file.filename,
            "original_transcript": transcript,
            "corrected_transcript": corrected,
            "fluency_score": score,
            "events": events,
            "feedback": generate_feedback(events)
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions so FastAPI sends the right code
        raise he
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected Server Error: {str(e)}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path): os.remove(temp_path)
        if clean_path and os.path.exists(clean_path): os.remove(clean_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)