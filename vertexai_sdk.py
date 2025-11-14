import os
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import ffmpeg
from dotenv import load_dotenv
from pydub import AudioSegment
import google.genai as genai
from google.genai.types import Part, GenerateContentConfig
from prompt_transcribe import transcript_prompt
from response_schema import response_schema

# ------------------ Setup Logging ------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ------------------ Load Environment ------------------
load_dotenv()
GCP_API_KEY = os.getenv("GCP_API_KEY")
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")

# ------------------ Client Initialization ------------------
def get_genai_client(mode="sdk", api_key=None, creds_path=None, project=None, location=None):
    """
    Returns a genai.Client configured for either:
    - 'sdk': API key based Gemini SDK
    - 'vertex': Vertex AI integration
    """
    if mode == "sdk":
        key = api_key or os.getenv("GCP_API_KEY")
        if not key:
            raise ValueError("GCP_API_KEY not provided or found in environment")
        return genai.Client(api_key=key)
    
    elif mode == "vertex":
        if creds_path:
            if not os.path.exists(creds_path):
                raise FileNotFoundError(f"Google credentials file not found: {creds_path}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        if not project or not location:
            raise ValueError("PROJECT_ID and LOCATION are required for Vertex AI")
        return genai.Client(vertexai=True, project=project, location=location)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")



# ------------------ Utility Functions ------------------
def format_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02}:{secs:02}"

def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None

# ------------------ Audio Splitting ------------------
def split_audio_ffmpeg(input_file, chunk_length_sec=360):
    out_files = []
    audio = AudioSegment.from_file(input_file)
    duration = len(audio) / 1000
    base_name, ext = os.path.splitext(input_file)

    for i, start in enumerate(range(0, int(duration), chunk_length_sec)):
        this_chunk_len = min(chunk_length_sec, duration - start)
        out_file = f"{base_name}_chunk{i+1}.mp3"
        (
            ffmpeg.input(input_file, ss=start, t=this_chunk_len)
            .output(out_file)
            .run(overwrite_output=True, quiet=True)
        )
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            out_files.append(out_file)
        else:
            logger.warning(f"Chunk {i+1} not created or empty → skipped")
    return out_files, duration

# ------------------ Transcript Extractor ------------------
def extract_transcript(response):
    try:
        candidate = response.candidates[0] if response.candidates else None
        if candidate and candidate.content.parts:
            return candidate.content.parts[0].text
        logger.warning("Empty or malformed response from Gemini.")
        return ""
    except Exception as e:
        logger.error(f"Failed to extract transcript: {e}")
        return ""

# ------------------ Transcription ------------------
def transcribe_chunk(client, audio_file_path, chunk_number, chunk_length_sec, total_duration, model_name="gemini-2.5-flash", max_retries=3, retry_delay=5):
    import time
    with open(audio_file_path, "rb") as f:
        byte_data = f.read()

    audio_content = Part.from_bytes(
        data=byte_data,
        mime_type="audio/mpeg" if audio_file_path.endswith(".wav") else "audio/mp3"
    )

    logger.debug(f"[Chunk {chunk_number}] Sending audio to Gemini {model_name}")
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[transcript_prompt, audio_content],
                config=GenerateContentConfig(
                    temperature=0,
                    top_p=0.9,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )
            chunk_text = extract_transcript(response)
            start_time = (chunk_number - 1) * chunk_length_sec
            end_time = min(start_time + chunk_length_sec, total_duration)

            entries = []
            parsed = safe_json_parse(chunk_text)
            if parsed:
                for e in parsed:
                    def adjust(ts: str) -> str:
                        mm, ss = map(int, ts.split(":"))
                        total_ms = (mm * 60 + ss) * 1000 + (start_time * 1000)
                        adj_mm, adj_ss = divmod(total_ms // 1000, 60)
                        return f"{int(adj_mm):02d}:{int(adj_ss):02d}"
                    e["start_time"] = adjust(e["start_time"])
                    e["end_time"] = adjust(e["end_time"])
                    entries.append(e)

            # Show classified agent and speaker if present
            agent_info = None
            speaker_info = None
            for entry in entries:
                if "agent" in entry and entry["agent"]:
                    agent_info = entry["agent"]
                if "speaker" in entry and entry["speaker"]:
                    speaker_info = entry["speaker"]
                if agent_info and speaker_info:
                    break
            if agent_info:
                logger.info(f"[Chunk {chunk_number}] Classified agent: {agent_info}")
            if speaker_info:
                logger.info(f"[Chunk {chunk_number}] Classified speaker: {speaker_info}")

            chunk_result = {
                "chunk_number": chunk_number,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "start_time": format_time(start_time),
                "end_time": format_time(end_time),
                "entries": entries,
                "raw_text": chunk_text,
                "classified_agent": agent_info if agent_info else None,
                "classified_speaker": speaker_info if speaker_info else None
            }

            chunk_file = f"{os.path.splitext(audio_file_path)[0]}_chunk{chunk_number}.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(chunk_result, f, indent=2, ensure_ascii=False)

            logger.info(f"[Chunk {chunk_number}] Transcript saved → {chunk_file}")
            return chunk_result

        except Exception as e:
            attempt += 1
            # Check for 503 error
            if hasattr(e, 'args') and len(e.args) > 0 and '503' in str(e.args[0]):
                logger.warning(f"[Chunk {chunk_number}] 503 error, retrying in {retry_delay} seconds (attempt {attempt}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                logger.error(f"[Chunk {chunk_number}] Transcription failed: {e}")
                break
    # If all retries failed
    logger.error(f"[Chunk {chunk_number}] Transcription failed after {max_retries} attempts.")
    start_time = (chunk_number - 1) * chunk_length_sec
    end_time = min(start_time + chunk_length_sec, total_duration)
    return {
        "chunk_number": chunk_number,
        "start_time_sec": start_time,
        "end_time_sec": end_time,
        "start_time": format_time(start_time),
        "end_time": format_time(end_time),
        "entries": [],
        "raw_text": "",
        "classified_agent": None,
        "classified_speaker": None
    }

# ------------------ Adjust Transcript ------------------
def adjust_chunk_timestamps(transcripts):
    corrected_entries = []
    for chunk in transcripts:
        if chunk.get("entries"):
            for e in chunk["entries"]:
                corrected_entries.append({
                    "start_time": e.get("start_time", chunk["start_time"]),
                    "end_time": e.get("end_time", chunk["end_time"]),
                    "speaker": e.get("speaker"),
                    "utterance": e.get("utterance") or e.get("content") or "",
                    "loudness": e.get("loudness"),
                    "sentiment": e.get("sentiment")
                })
        else:
            corrected_entries.append({
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "speaker": None,
                "utterance": chunk.get("raw_text", ""),
                "loudness": None,
                "sentiment": None
            })
    return corrected_entries

# ------------------ Main Parallel Transcription ------------------
def transcribe_audio_parallel(audio_path, client, chunk_length_sec=360, max_workers=4):
    logger.info("Splitting audio into chunks...")
    chunk_files, duration = split_audio_ffmpeg(audio_path, chunk_length_sec)
    logger.info(f"Created {len(chunk_files)} chunks (total duration {format_time(duration)}).")

    transcripts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(transcribe_chunk, client, chunk, idx+1, chunk_length_sec, duration): chunk
            for idx, chunk in enumerate(chunk_files)
        }
        for future in as_completed(future_to_chunk):
            transcripts.append(future.result())

    transcripts = sorted(transcripts, key=lambda x: x["chunk_number"])
    corrected_entries = adjust_chunk_timestamps(transcripts)

    final_file = "final_transcript.json"
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(corrected_entries, f, indent=2, ensure_ascii=False)

    logger.info(f"Final transcript saved → {final_file}")
    return corrected_entries

# ------------------ Example Run ------------------
if __name__ == "__main__":
    audio_path = "audio30.aac"

    # Choose mode: 'sdk' or 'vertex'
    mode = "sdk"  # or "sdk"
    creds_path = "./creds.json"  # only for vertex
    # client = get_genai_client(mode="sdk")  # Gemini SDK
    # client = get_genai_client(mode="vertex", creds_path="./creds.json", project=PROJECT_ID, location=LOCATION)
    client = get_genai_client(mode=mode, creds_path=creds_path, project=PROJECT_ID, location=LOCATION)

    transcript_json = transcribe_audio_parallel(audio_path, client, chunk_length_sec=360, max_workers=4)
    print(f"Transcript JSON with {len(transcript_json)} entries generated.")