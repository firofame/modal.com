# venv/bin/modal run whisper_x.py
# yt-dlp -x -o "audio.%(ext)s" "Ln3CGhx8DcI"

local_file_path = "/Users/firozahmed/Downloads/audio.mp4"

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "ffmpeg","libcudnn8")
    .uv_pip_install("whisperx", "huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App("whisperx", image=image)

@app.cls(
    gpu="L40s",
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
    scaledown_window=60 * 10,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import whisperx
        import tempfile

        model = whisperx.load_model("large-v3", "cuda", compute_type="float16", download_root="/cache")

        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            audio = whisperx.load_audio(temp_file.name)
            result = model.transcribe(audio, batch_size=16)
            return result["segments"]


@app.local_entrypoint()
def main():
    path = Path(local_file_path)
    segments = Model().transcribe.remote(path.read_bytes())
    transcription_text = "\n".join(segment["text"].strip() for segment in segments)
    output_file_path = path.with_stem(f"{path.stem}_transcription").with_suffix(".txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(transcription_text)
    print(f"Transcription saved to {output_file_path}")