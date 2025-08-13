import os
import json
import torch
from pyannote.audio import Pipeline
import librosa
import warnings
from dotenv import load_dotenv
import tempfile
import soundfile as sf
import openai
import traceback

# Suppress warnings
warnings.filterwarnings("ignore")


prompt = """
    
            You are a **Medical Documentation Assistant**. Your primary task is to **meticulously correct errors** in the provided medical transcription.
 
            **Core Principles for Correction:**
            - **Preserve Originality:** Do NOT alter the original wording, meaning, or sentence structure.
            - **No Additions:** Do NOT introduce any new information or content.
            - **Accuracy:** Focus on correcting spelling, grammar, punctuation, and medical terminology. Ensure medical terms are accurate while retaining the original intended meaning.
            - **Clarity:** Format the text for readability with appropriate spacing and punctuation.
            **IMPORTANT:**
            - Do NOT change any names, dates, or medical terms. If you are unsure about a word, leave it unchanged.
            - Only correct punctuation, spacing, and obvious grammar mistakes.
            - Do not change the original wording or structure of the sentences.
            - Most of the times, the mistake that happens is that you make the word 'Consult' into 'Councel' or 'Cancer', so be careful with that.
            - Also there is mistake with the word 'Period', you write 'Peter' instead of 'Period', so be careful with that.
            - Sometimes, 'please write' is written as 'please right', while it means you should the text next to it understanding the context in bullet lines and do not write the keyword, so be careful with that.
            - Sometimes, 'please add' is also written as it is instead of making bullet lines of the points and not writing the keyword, Carefully understand the context first in this case as well.
 
            **Specific Formatting Instructions (Keyword-based Replacements):**
            - **Paragraphs:** If "paragraph" or "next paragraph" appears, replace it with two newline characters (`\\n\\n`) and remove the keyword phrase.
            - **Periods:** If "period" appears, replace it with a period character (`.`) and remove the keyword. Do NOT add a newline after the period unless a "newline" or "paragraph" keyword explicitly follows.
            - **Newlines:** If "newline" appears, replace it with a single newline character (`\\n`) and remove the keyword.
            - **Next:** If "next" or "next number" appears, replace it with a single newline character (`\\n`), remove the keyword and make bullet points with numbering. If the context indicates a list of points or items, format them as bullet points (e.g., "- ", or numbered list if appropriate) in the output.
            - **Commas:** If "comma" appears, replace it with a comma character (`,`) and remove the keyword.
            - **Numerals:** Convert spelled-out numbers to their digit equivalents (e.g., "one" becomes "1", "twenty-five" becomes "25").
            - **Abbreviations:** If "abbreviation" appears, replace it with the appropriate abbreviation (e.g., "mg" for milligrams, "ml" for milliliters) and remove the keyword.
 
            **Thing that can be changed:**
            - **Age Correction:** If age is mentioned anywhere in the transcription, it may be incorrectly transcribed from audio. **CRITICALLY IMPORTANT:**
              * First, locate the date of birth (DOB) in the transcription
              * Calculate the correct age using the DOB and the consultation/dictation date
              * **Replace ALL instances of age throughout the ENTIRE text** - check every occurrence of age-related terms like "year-old", "years old", "age", etc.
              * Be thorough - scan the complete transcription from beginning to end for any age mentions
              * Ensure mathematical accuracy when calculating age (use the exact dates provided)
              * Do NOT change any other information - only correct the age values
              * Example: If DOB is 03/11/1997 and consultation date is 05/07/2025, the correct age is 28 years old - replace ALL age references with "28"
    """


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file!")
openai.api_key = OPENAI_API_KEY

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file!")

def preprocess_audio(input_file):
    # input_file is a file-like object
    input_file.seek(0)
    audio_data, sr = librosa.load(input_file, sr=16000, mono=True)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_wav.name, audio_data, 16000)
    return tmp_wav.name

def transcribe_with_openai(audio_path):
    with open(audio_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="gpt-4o-transcribe",  
            file=audio_file
        )
    return response.text.strip()

def correct_transcription_with_llm(text, prompt=None):
    if prompt is None:
        prompt = prompt
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ],
        max_tokens=2048
    )
    return response.choices[0].message.content.strip()

def summarize_transcript(transcript_text):
    summary_prompt = (
        "Summarize the following medical call transcript in 3-5 concise bullet points, "
        "focusing on the main issues discussed, actions taken, and any follow-up required. "
        "Do not add any information not present in the transcript.\n\n"
        f"{transcript_text}"
    )
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": summary_prompt}
        ],
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def save_transcripts(audio_path, transcript, corrected_text):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = "Transcripts"
    os.makedirs(output_dir, exist_ok=True)

    # Save raw transcript
    raw_path = os.path.join(output_dir, f"{base_name}_raw.txt")
    with open(raw_path, "w") as f:
        for t in transcript:
            f.write(f"{t['speaker']}: {t['text']}\n")
    print(f"[INFO] Raw transcript saved to {raw_path}")

    # Save corrected transcript
    corrected_path = os.path.join(output_dir, f"{base_name}_corrected.txt")
    with open(corrected_path, "w") as f:
        f.write(corrected_text)
    print(f"[INFO] Corrected transcript saved to {corrected_path}")

    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
    with open(json_path, "w") as f:
        json.dump({"audio_file": audio_path, "transcript": transcript, "corrected": corrected_text}, f, indent=2)
    print(f"[INFO] Transcript JSON saved to {json_path}")

def process_audio_file(audio_file_obj, correction_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    try:
        print("[INFO] Preprocessing audio for diarization...")
        processed_path = preprocess_audio(audio_file_obj)

        print("[INFO] Loading speaker diarization model...")
        diarization = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        diarization.to(torch.device(device))

        print(f"[INFO] Performing diarization on {os.path.basename(processed_path)}...")
        diarization_results = diarization(processed_path)

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
                seg_text = transcribe_with_openai(tmp_wav.name)

            if seg_text:
                transcript.append({
                    "speaker": speakers[speaker],
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "text": seg_text
                })

        print("\n[RAW TRANSCRIPT]")
        for t in transcript:
            print(f"{t['speaker']}: {t['text']}")

        # Combine transcript for LLM correction
        full_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in transcript])
        corrected_text = correct_transcription_with_llm(full_text, correction_prompt)

        print("\n[CORRECTED TRANSCRIPT]")
        print(corrected_text)

        # Generate and print summary
        summary = summarize_transcript(corrected_text)
        print("\n[CALL SUMMARY]")
        print(summary)

        # Save transcripts
        save_transcripts(processed_path, transcript, corrected_text)

        return transcript, corrected_text, summary

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return None, None, None
import fileinput
if __name__ == "__main__":
    audio = fileinput.input('Audio file')
    
    with open(audio, "rb") as audio_file:
        process_audio_file(audio_file, prompt)

            

