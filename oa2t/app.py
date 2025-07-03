#!/usr/bin/env python3
import os
import math
import argparse
import openai
from pydub import AudioSegment

# ─── Configuration ────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-transcribe"
#MODEL = "whisper-1"
MAX_MB = 16  # maximum file size per request (in megabytes)

# ─── Transcription logic ──────────────────────────────────────────────────
def transcribe_large_file(path: str, language: str | None = None) -> str:
    audio = AudioSegment.from_file(path)
    file_size = os.path.getsize(path)
    duration_ms = len(audio)

    def call_api(file_handle):
        params = {
            "file": file_handle,
            "model": MODEL,
            "prompt": "以繁體中文回答，每句話結束時換行"
        }
        if language:
            params["language"] = language
        return openai.audio.transcriptions.create(**params).text

    # Single-request if under size limit
    if file_size <= MAX_MB * 1024 * 1024:
        with open(path, "rb") as f:
            return call_api(f)

    # Calculate chunk size in ms
    bytes_per_ms = file_size / duration_ms
    max_chunk_bytes = MAX_MB * 1024 * 1024
    max_chunk_ms = math.floor(max_chunk_bytes / bytes_per_ms)

    transcripts = []
    for start in range(0, duration_ms, max_chunk_ms):
        end = min(start + max_chunk_ms, duration_ms)
        chunk = audio[start:end]
        tmp_filename = f".chunk_{start//1000}_{end//1000}.mp4"
        chunk.export(tmp_filename, format="mp4")

        with open(tmp_filename, "rb") as f:
            transcripts.append(call_api(f))
        os.remove(tmp_filename)

    return "\n".join(transcripts)

# ─── Command-line interface ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Transcribe large audio files via OpenAI Whisper API (auto-split >25MB)"
    )
    parser.add_argument(
        "-f", "--file", required=True,
        help="Path to the input audio file (e.g. path/to/audio.m4a)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Optional path to save the transcript. Defaults to <audio_filename>.txt"
    )
    parser.add_argument(
        "-l", "--language", default=None,
        help="ISO language code of the audio (e.g. 'en', 'zh', 'ja')"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.file):
        parser.error(f"File not found: {args.file}")

    print(f"Transcribing: {args.file} …")
    transcript = transcribe_large_file(args.file, args.language)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base, _ = os.path.splitext(args.file)
        out_path = f"{base}.txt"

    # Save transcript to file
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(transcript)
    print(f"Transcript saved to: {out_path}")

if __name__ == "__main__":
    main()
