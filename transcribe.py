# venv/bin/modal run transcribe.py

file_path = "/Users/firozahmed/Downloads/audio.opus"

MODEL_NAME = "openai/whisper-large-v3"
# MODEL_NAME = "vrclc/Whisper-medium-Malayalam"

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libsndfile1")
    .uv_pip_install("librosa", "transformers", "torch", "accelerate", "huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App("whisper-malayalam", image=image)

with image.imports():
    import torch
    from transformers import pipeline
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

@app.cls(
    gpu="L4",
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.enter()
    def setup(self):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_NAME,
            device="cuda",
            chunk_length_s=30,
            return_timestamps=True,
            batch_size=8,
            generate_kwargs={"task": "transcribe"},
            ignore_warning=True,
        )

    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import io
        import librosa

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
        result = self.pipe(audio)

        return result["text"]


@app.local_entrypoint()
def main():
    path = Path(file_path)
    text = Model().transcribe.remote(path.read_bytes())
    print(f"Transcription: {text}")