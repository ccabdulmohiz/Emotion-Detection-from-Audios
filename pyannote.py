import os
import json
import whisper
import torch
from pyannote.audio import Pipeline
import librosa
import warnings
from dotenv import load_dotenv
import tempfile
import soundfile as sf
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION", "us-central1")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GEMINI_SERVICE_ACCOUNT_PATH")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file!")

# =============================
# PREPROCESS AUDIO FOR STABILITY
# =============================
def preprocess_audio(input_path):
    """
    Converts input audio to a consistent 16kHz mono WAV to avoid tensor size mismatch.
    """
    audio_data, sr = librosa.load(input_path, sr=16000, mono=True)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_wav.name, audio_data, 16000)
    return tmp_wav.name

# =============================
# MAIN PROCESSING FUNCTION
# =============================
def process_audio_file(audio_path, output_path=None, save_as_text=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    try:
        print("[INFO] Preprocessing audio for diarization...")
        processed_path = preprocess_audio(audio_path)

        print("[INFO] Loading speaker diarization model...")
        diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        diarization.to(torch.device(device))

        print(f"[INFO] Performing diarization on {os.path.basename(audio_path)}...")
        diarization_results = diarization(processed_path)

        print("[INFO] Loading Whisper model...")
        whisper_model = whisper.load_model("medium", device=device)

        audio_data, sr = librosa.load(processed_path, sr=16000)
        transcript = []

        speakers = {}
        speaker_idx = 0

        for turn, _, speaker in diarization_results.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end

            if speaker not in speakers:
                speakers[speaker] = "Agent" if speaker_idx == 0 else "Customer"
                speaker_idx += 1

            start_frame = int(start_time * sr)
            end_frame = min(int(end_time * sr), len(audio_data))
            segment = audio_data[start_frame:end_frame]

            # Skip segments shorter than 1 second
            if len(segment) < sr:
                print(f"[DEBUG] Skipping short segment ({len(segment)/sr:.2f}s)")
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav:
                sf.write(tmp_wav.name, segment, sr)
                seg_result = whisper_model.transcribe(tmp_wav.name)
                seg_text = seg_result["text"].strip()

            if seg_text:
                transcript.append({
                    "speaker": speakers[speaker],
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": seg_text
                })

        print("\n[TRANSCRIPT]")
        for t in transcript:
            print(f"{t['speaker']}: {t['text']}")

        # Save transcript
        if output_path:
            if save_as_text:
                output_file = output_path if output_path.endswith(".txt") else os.path.splitext(output_path)[0] + ".txt"
                with open(output_file, "w") as f:
                    for t in transcript:
                        f.write(f"{t['speaker']}: {t['text']}\n\n")
                print(f"[INFO] Transcript saved to {output_file}")
            else:
                output_file = output_path if output_path.endswith(".json") else os.path.splitext(output_path)[0] + ".json"
                with open(output_file, "w") as f:
                    json.dump({"audio_file": audio_path, "transcript": transcript}, f, indent=2)
                print(f"[INFO] Transcript JSON saved to {output_file}")

        # Analyze with Gemini
        analysis = analyze_with_gemini(transcript)
        print("\n[GEMINI ANALYSIS]")
        print(json.dumps(analysis, indent=2))

        # Save analysis
        if output_path:
            analysis_path = os.path.splitext(output_path)[0] + "_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"[INFO] Gemini analysis saved to {analysis_path}")

        return transcript, analysis

    except Exception as e:
        print(f"[ERROR] {e}")
        return None, None

# =============================
# GEMINI POST-PROCESSING
# =============================
def analyze_with_gemini(transcript):
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel("gemini-1.5-pro")

        transcript_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])

        prompt = f"""
        You are a call center conversation analyst.
        Given the transcript below, return a JSON object with:
        - "summary": a concise summary of the conversation
        - "sentiment": overall customer sentiment ("positive", "neutral", or "negative")
        - "resolved": true/false if the customer's issue was resolved
        - "empathy_score": a number from 0 to 10 rating agent's empathy
        Transcript:
        {transcript_text}
        """

        response = model.generate_content(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"summary": response.text.strip()}

    except Exception as e:
        print(f"[ERROR] Gemini analysis failed: {e}")
        return {}

# =============================
# MAIN ENTRY POINT
# =============================
if __name__ == "__main__":
    manual_audio_path = "/home/abdulmohiz@BAGH.MTBC.COM/Downloads/Emots/Test Audio/250611_4871.MP3"
    output_dir = os.path.join(os.path.dirname(manual_audio_path), "Transcripts")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(manual_audio_path))[0] + "_transcript.txt"
    )

    process_audio_file(manual_audio_path, output_file, save_as_text=True)
