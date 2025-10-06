# venv/bin/modal run transcribe.py

file_path = "https://www.youtube.com/shorts/xkD4JkcuP9M"

# MODEL_NAME = "openai/whisper-large-v3"
MODEL_NAME = "vrclc/Whisper-medium-Malayalam"

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "ffmpeg")
    .uv_pip_install("librosa", "transformers", "torch", "accelerate", "huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App("whisper", image=image)

@app.cls(
    gpu="L4",
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import torch
        from transformers import pipeline
        import io
        import librosa


        pipe = pipeline(
            task="automatic-speech-recognition",
            model=MODEL_NAME,
            dtype=torch.float16,
            device="cuda",
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"},
        )

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        result = pipe(audio)

        return result["text"]


@app.local_entrypoint()
def main():
    local_file_path = file_path
    if "youtube.com" in file_path or "youtu.be" in file_path:
        from yt_dlp import YoutubeDL
        ydl_opts = {'postprocessors':[{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'outtmpl': 'audio'}
        YoutubeDL(ydl_opts).download([file_path])
        local_file_path = "audio.mp3"
    path = Path(local_file_path)
    transcription_text = Model().transcribe.remote(path.read_bytes())

    output_file_path = path.with_stem(f"{path.stem}_transcription").with_suffix(".txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)
    print(f"Transcription saved to {output_file_path}")