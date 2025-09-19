# venv/bin/modal run whisper_x.py --file-path /Users/firozahmed/Downloads/audio.opus

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "libcudnn8", "libcudnn8-dev")
    .uv_pip_install("whisperx", "librosa")
)

app = modal.App("whisperx", image=image)

with image.imports():    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

@app.cls(
    gpu="T4",
    image=image,
    volumes={"/cache": modal.Volume.from_name("whisper-cache", create_if_missing=True)},
    timeout=30 * 60,
)
class Model:
    @modal.enter()
    def setup(self):
        import whisperx
        self.model = whisperx.load_model("large-v3", "cuda", compute_type="float16", download_root="/cache")

    @modal.method()
    def transcribe(self, audio_bytes: bytes):
        import io
        import librosa

        audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        result = self.model.transcribe(audio, batch_size=16, task="translate")

        return " ".join([segment["text"].strip() for segment in result["segments"]])


@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    output_path = path.with_suffix(".txt")

    transcription = Model().transcribe.remote(path.read_bytes())

    output_path.write_text(transcription)
    print(f"Transcription saved to: {output_path}")