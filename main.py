import os
import argparse
import whisper
import subprocess
from pathlib import Path


def extract_audio(video_path, audio_path):
    """Extracts audio from the video file using FFmpeg."""
    print(f"Extracting audio to {audio_path}...")
    command = [
        "ffmpeg", "-i", video_path,
        "-q:a", "0", "-map", "a",
        audio_path, "-y", "-loglevel", "error"
    ]
    subprocess.run(command, check=True)


def generate_subtitles(audio_path, model_size="base", language="fa"):
    """Transcribes audio using Whisper and generates an SRT file."""
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Transcribing audio (Language: {language})...")
    result = model.transcribe(audio_path, language=language)

    srt_path = Path(audio_path).with_suffix('.srt')
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start_time = format_timestamp(segment["start"])
            end_time = format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    print(f"Subtitles saved to {srt_path}")
    return srt_path


def format_timestamp(seconds):
    """Formats seconds into SRT timestamp format (HH:MM:SS,ms)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millisecs:03}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoSub-Reels: Automated Subtitle Generator")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--model", default="base",
                        help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--lang", default="fa",
                        help="Language code for transcription (default: fa for Persian)")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print("Error: Video file not found.")
        exit(1)

    temp_audio = "temp_audio.wav"
    try:
        extract_audio(args.video, temp_audio)
        generate_subtitles(
            temp_audio, model_size=args.model, language=args.lang)
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
