import os
import csv
import json
import time
import sys
import google.generativeai as genai
from tabulate import tabulate
from google.oauth2 import service_account
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Clear any API_KEY from environment to avoid conflicts with service account
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

SERVICE_ACCOUNT_FILE = os.getenv("GEMINI_SERVICE_ACCOUNT_PATH", 
                               "")
AUDIO_DIR = os.getenv("AUDIO_DIR", 
                    "Test Audio")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", 
                     "call_analysis_results.csv")
TRANSCRIPT_DIR = "Transcripts"

print(f"Using service account file: {SERVICE_ACCOUNT_FILE}")
print(f"Using audio directory: {AUDIO_DIR}")
print(f"Using transcript directory: {TRANSCRIPT_DIR}")
print(f"Output will be saved to: {OUTPUT_CSV}")

# Set up credentials from the service account file
try:
    with open(SERVICE_ACCOUNT_FILE, 'r') as f:
        service_account_info = json.load(f)

    
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    genai.configure(credentials=credentials, api_key=None)  
except Exception as e:
    print(f"Error loading credentials: {e}")
    sys.exit(1)

LABELS = [
    "Friendly and empathetic",
    # "Clear and effective communication",                                    
    "Actively listened",
    "Professional and respectful",
    "Unengaged or disinterested",
    "Lacked empathy",
    "Unclear or confusing information",
    "Interruptive or dismissive"
]

def get_transcript_for_audio(audio_filename):
    """Find the corresponding transcript file for an audio file"""
    base_name = os.path.splitext(audio_filename)[0]
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.txt")
    
    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r') as f:
                lines = f.readlines()
            

            transcript = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if ":" in line:
                    parts = line.split(":", 1)
                    speaker = parts[0].strip()
                    text = parts[1].strip() if len(parts) > 1 else ""
                    
                    transcript.append({
                        "speaker": speaker,
                        "text": text
                    })
            
            if not transcript:
                print(f"Warning: Transcript for {audio_filename} appears empty or improperly formatted")
                
            return {"transcript": transcript}
        except Exception as e:
            print(f"Error reading transcript for {audio_filename}: {e}")
    

    json_path = os.path.join(TRANSCRIPT_DIR, f"{base_name}_transcript.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON transcript for {audio_filename}: {e}")
            
    return None

def analyze_call_with_transcript(audio_file, transcript_data):
    """Analyze the call using both audio and transcript data, but just return a label"""
    print(f"Analyzing: {os.path.basename(audio_file)}...")
    
    
    transcript_text = ""
    for segment in transcript_data.get("transcript", []):
        transcript_text += f"{segment['speaker']}: {segment['text']}\n"
    
    few_shot_examples = """
EXAMPLE 1:
[Transcript excerpt]
Customer: Hello, I need help with my internet connection. It's been down for two days.
Agent: I understand that's frustrating. Let me check your account and see what's happening. Can you please verify your account number?
Customer: It's 12345678.
Agent: Thank you. I see the outage in your area. We have technicians working on it and expect service to be restored by tomorrow morning. Is there anything else I can help you with?
Customer: No, that's all. Thank you for the information.
Agent: You're welcome. Thank you for your patience, and please reach out if you need anything else.

Analysis:
The agent was empathetic about the customer's frustration, clearly communicated the status of the outage, provided a timeline for resolution, and ended the call professionally.
LABEL: Friendly and empathetic
SCORE: 9

EXAMPLE 2:
[Transcript excerpt]
Customer: Hi, I've been charged twice on my last bill. Can you help me?
Agent: Account number?
Customer: It's 87654321.
Agent: I see the duplicate charge. It might be fixed in the next billing cycle.
Customer: But I need that money now. It's a significant amount.
Agent: Well, that's our policy. You'll have to wait.
Customer: Can I speak to a supervisor?
Agent: Fine, I'll transfer you. Hold on.

Analysis:
The agent was abrupt, showed no empathy regarding the financial impact on the customer, provided unclear information about resolution, and was dismissive when the customer expressed concern.
LABEL: Lacked empathy
SCORE: 4.5
"""

    prompt = f"""
You are a call center quality analyst. Given a call audio and its transcript, determine the SINGLE most appropriate label for the agent's performance from this list: {', '.join(LABELS)}.
The transcript could a little bugged or messed up by sequence so make sure to understand the context first and then proceed with careful evaluation.
Instructions:
1. Listen carefully to the audio for tone, emotion, and engagement
2. Review the transcript to understand the conversation flow and content
3. Pay special attention to how the agent:
   - Responds to customer concerns
   - Demonstrates understanding and empathy
   - Provides clear information
   - Maintains professionalism throughout the call
   - Resolves the customer's issue or provides a path forward

Below are examples of how to analyze calls:
{few_shot_examples}

Here's the transcript for the current call:
{transcript_text}

Please analyze both the audio and transcript, then respond with ONLY:
1. The single most appropriate label from the list
2. A score from 1-10 (where 10 is excellent customer experience)
3. A one-line reason for your rating

Format your response exactly as:
LABEL: [single label]
SCORE: [number 1-10]
REASON: [one line reason]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
"""
    
    def read_bytes(path):
        with open(path, "rb") as f:
            return f.read()
            
    # Determine the mime type based on file extension
    extension = os.path.splitext(audio_file)[1].lower()
    if extension == '.mp3':
        mime_type = "audio/mpeg"
    elif extension == '.opus':
        mime_type = "audio/opus"
    else:
        mime_type = "audio/wav"  # Default for .wav and others
    
    # Add retry mechanism for API calls
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel("gemini-2.5-pro")
            parts = [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": read_bytes(audio_file)}}
            ]
            
            # Generate with safety settings that allow call center audio analysis
            response = model.generate_content(
                parts,
                generation_config={"temperature": 0.2}  
            )
            return response.text.strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Error during API call (attempt {attempt+1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to analyze call after {max_retries} attempts: {e}")
                return None

def extract_label_score_reason(analysis_text):
    """Extract the label, score, and reason from the analysis text"""
    if not analysis_text:
        return None, None, None

    label = None
    score = None
    reason = None

    for line in analysis_text.split('\n'):
        if line.startswith('LABEL:'):
            label = line.replace('LABEL:', '').strip().strip('"\'')
        elif line.startswith('SCORE:'):
            try:
                score = int(line.replace('SCORE:', '').strip())
            except ValueError:
                pass
        elif line.startswith('REASON:'):
            reason = line.replace('REASON:', '').strip()

    # Validate that the label is from our predefined list
    if label and label not in LABELS:
        print(f"Warning: LLM returned an invalid label: '{label}'")
        closest_label = min(LABELS, key=lambda x: abs(len(x) - len(label)))
        print(f"Using closest match: '{closest_label}'")
        label = closest_label

    return label, score, reason

def write_results_to_csv(results):
    """Write analysis results to CSV file"""
    # Check if file exists and read existing results
    existing_results = {}
    if os.path.exists(OUTPUT_CSV):
        try:
            with open(OUTPUT_CSV, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader, None)  # Skip header
                if headers:
                    for row in reader:
                        if len(row) >= 4:
                            existing_results[row[0]] = row[1:]
        except Exception as e:
            print(f"Warning: Couldn't read existing CSV: {e}")

    try:
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File Name', 'Label', 'Score', 'Reason'])

            # Add new results
            for result in results:
                filename, label, score, reason = result
                csv_writer.writerow([filename, label, score, reason])
                if filename in existing_results:
                    del existing_results[filename]

            # Add existing results that weren't overwritten
            for filename, data in existing_results.items():
                row = [filename] + data
                csv_writer.writerow(row)

        print(f"\nResults saved to: {OUTPUT_CSV}")
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        # Write to backup file
        backup_file = OUTPUT_CSV + ".backup"
        with open(backup_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['File Name', 'Label', 'Score', 'Reason'])
            for result in results:
                csv_writer.writerow(result)
        print(f"Results saved to backup file: {backup_file}")

def verify_audio_playable(audio_file):
    """Check if the audio file can be played correctly"""
    try:
        import wave
        if audio_file.lower().endswith('.wav'):
            with wave.open(audio_file, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                duration = frames / float(rate)
                return duration > 0
        return True  # For non-WAV files, assume they're OK
    except Exception:
        # If we can't verify, assume it's OK
        return True

if __name__ == "__main__":
    all_results = []
    
    # Ensure transcript directory exists
    if not os.path.exists(TRANSCRIPT_DIR):
        print(f"Transcript directory not found: {TRANSCRIPT_DIR}")
        exit(1)
    
    # Get the list of audio files
    audio_files = [f for f in os.listdir(AUDIO_DIR) 
                  if f.lower().endswith(('.wav', '.mp3', '.opus'))]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for fname in audio_files:
        call_file = os.path.join(AUDIO_DIR, fname)
        
        # Verify the audio file
        if not verify_audio_playable(call_file):
            print(f"Warning: Audio file {fname} appears to be corrupted")
        
        # Get the matching transcript file
        transcript_data = get_transcript_for_audio(fname)
        if not transcript_data:
            print(f"No transcript found for {fname}, skipping...")
            continue
        
        # Check if transcript has content
        if not transcript_data.get("transcript"):
            print(f"Empty transcript for {fname}, skipping...")
            continue
        
        # Analyze the call
        analysis = analyze_call_with_transcript(call_file, transcript_data)
        if not analysis:
            print(f"Failed to analyze {fname}, skipping...")
            continue
        
        # Extract label and score
        label, score, reason = extract_label_score_reason(analysis)
        
        # Print formatted output
        table = [
            ["Call File", fname],
            ["Label", label if label else "Unknown"],
            ["Score", score if score else "Unknown"],
            ["Reason", reason if reason else "No reason provided"]
        ]
        print(tabulate(table, tablefmt="fancy_grid"))

        # Store results for CSV
        if label and score:
            all_results.append([fname, label, score, reason])
    
    # Write all results to CSV
    if all_results:
        write_results_to_csv(all_results)
    else:
        print("No results to save.")

