# venv/bin/modal run transcribe.py --file-path /Users/firozahmed/Downloads/audio.mpeg

from pathlib import Path
import modal

app = modal.App(name="whisper-transcribe-openai")

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "openai-whisper==20250625",
    "librosa==0.11.0",
)

@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    text = Transcribe().transcribe.remote(path.read_bytes())
    print(f"Transcription: {text}")

@app.cls(
    image=image,
    gpu="L40s",
)
class Transcribe:
    @modal.enter()
    def load_model(self):
        import whisper

        self.model = whisper.load_model("large-v3")

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
    ) -> str:
        import io
        import librosa

        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        transcription = self.model.transcribe(audio_data, language="ml")["text"]

        return transcription