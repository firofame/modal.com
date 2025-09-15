# venv/bin/modal run transcribe.py --file-path /Users/firozahmed/Downloads/audio.mpeg

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg", "libcudnn8", "libcudnn8-dev")
    .uv_pip_install("whisperx", "librosa", "numpy")
)

app = modal.App("whisperx", image=image)

CACHE_DIR = "/cache"
device = "cuda"

with image.imports():
    import whisperx

@app.cls(
    gpu="L40s",
    volumes={CACHE_DIR: modal.Volume.from_name("whisper-cache", create_if_missing=True)},
    scaledown_window=60 * 10,
    timeout=60 * 60,
)
@modal.concurrent(max_inputs=15)
class Model:
    @modal.enter()
    def setup(self):
        self.model = whisperx.load_model("large-v3", device, compute_type="float16", download_root=CACHE_DIR)

    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import io
        import librosa

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        result = self.model.transcribe(audio, batch_size=16)

        # model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        # result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        return result["segments"]


@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    text = Model().transcribe.remote(path.read_bytes())
    print(f"Transcription: {text}")