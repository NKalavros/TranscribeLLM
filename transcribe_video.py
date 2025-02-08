import subprocess
import whisper
import os
import tempfile
import yt_dlp as youtube_dl  # Switch to yt_dlp
import uuid

def transcribe_video(video_path, output_txt_path="transcription.txt", model_size="base"):
    temp_audio_path = os.path.abspath(f"audio_{uuid.uuid4().hex[:8]}.wav")
    try:
        # Verify the existence of the video file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Run ffmpeg command to extract audio into a WAV file
        subprocess.run([
            "ffmpeg", "-i", video_path, "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le", temp_audio_path
        ], check=True)
        print("Beginning whisper model loading")
        # Load Whisper model 
        model = whisper.load_model(model_size)
        print("Beginning whisper run")
        # Transcribe
        result = model.transcribe(temp_audio_path)
        
        # Save output
        with open(output_txt_path, "w") as f:
            f.write(result["text"])

        print(f"Transcription saved to {output_txt_path}")

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(video_path):
            os.remove(video_path)

def download_and_transcribe_youtube_video(youtube_url, output_txt_path="transcription.txt", model_size="base"):
    temp_video_path = os.path.abspath(f"video_{uuid.uuid4().hex[:8]}.mp4")
    try:
        # Download video using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_video_path,
            'overwrites': True,  # Ensure file is overwritten if it exists
            'ignoreerrors': True,
            'nocheckcertificate': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:6.0) Gecko/20100101 Firefox/60.0',
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        # Verify the existence of the video file
        if not os.path.exists(temp_video_path):
            raise FileNotFoundError(f"Downloaded video file not found: {temp_video_path}")

        # Transcribe the downloaded video
        transcribe_video(temp_video_path, output_txt_path, model_size)

    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <input.mp4> [output.txt]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "transcription.txt"
    transcribe_video(input_file, output_file)
