# venv/bin/modal run transcribe.py --file-path /Users/firozahmed/Downloads/audio.mpeg

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libcudnn8", "libcudnn8-dev")
    .pip_install("whisperx", "librosa")
)
app = modal.App("whisperx", image=image)

GPU_CONFIG = "T4"

CACHE_DIR = "/cache"
cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)

@app.cls(
    gpu=GPU_CONFIG,
    volumes={CACHE_DIR: cache_vol},
    scaledown_window=60 * 10,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.enter()
    def setup(self):
        import whisperx

        device = "cuda"
        self.model = whisperx.load_model("large-v2", device, compute_type="float16", download_root=CACHE_DIR)

    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import io
        import librosa

        audio_data, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        transcription = self.model.transcribe(audio_data, language="ml", verbose=True, batch_size=16)["segments"]

        return transcription


@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    text = Model().transcribe.remote(path.read_bytes())
    print(f"Transcription: {text}")