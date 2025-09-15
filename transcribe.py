# venv/bin/modal run transcribe.py --file-path /Users/firozahmed/Downloads/audio.mpeg

from pathlib import Path
import modal

app = modal.App(name="whisper-transcribe-openai")

MODEL_DIR = "/model"
MODEL_NAME = "large-v3"
volume = modal.Volume.from_name("whisper-model-vol", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "openai-whisper",
    "librosa",
)

@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    text = Transcribe().transcribe.remote(path.read_bytes())
    print(f"Transcription: {text}")

@app.cls(
    image=image,
    gpu="L40s",
    volumes={MODEL_DIR: volume},
    scaledown_window=240,
    timeout=600,
)
class Transcribe:
    @modal.enter()
    def load_model(self):
        import whisper
        from pathlib import Path

        model_path = Path(MODEL_DIR) / f"{MODEL_NAME}.pt"
        if not model_path.exists():
            whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)
            volume.commit()
        self.model = whisper.load_model(MODEL_NAME, download_root=MODEL_DIR)

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
    ) -> str:
        import io
        import librosa

        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        transcription = self.model.transcribe(audio_data, language="ml", verbose=True)["text"]

        return transcription