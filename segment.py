# venv/bin/modal run segment.py --file-path /Users/firozahmed/Downloads/Bayan.ogg

from io import BytesIO
from pathlib import Path
import tempfile

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "ffmpeg")
    .uv_pip_install(
        "pyannote.audio",
        "pydub",
        "torchaudio",
        "hf-transfer",
    )
)

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(CACHE_DIR)})


app = modal.App("audio-segment")

with image.imports():
    import os
    import torch
    from pyannote.audio import Pipeline
    from pydub import AudioSegment
    import whisper
    import torchaudio

@app.cls(
    image=image, gpu="L40s", volumes=volumes, secrets=secrets, scaledown_window=240, timeout=600
)
class Model:
    @modal.enter()
    def enter(self):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(DEVICE)

    @modal.method()
    def inference(self, input_bytes: bytes, suffix:str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(input_bytes)
            temp_file.flush()
            
            # Run diarization
            diarization = self.pipeline(temp_file.name)

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
            
            # Load audio with pydub to easily slice it
            audio = AudioSegment.from_file(temp_file.name)
            
            # Get all segments for the first speaker
            first_speaker_segments = AudioSegment.empty()
            # The first speaker is usually the one with the most speech time.
            # We can get the label of the first speaker from the diarization result.
            try:
                first_speaker_label = diarization.labels()[0]
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker == first_speaker_label:
                        # pydub uses milliseconds for slicing
                        first_speaker_segments += audio[turn.start * 1000:turn.end * 1000]
            except IndexError:
                # If no speakers are found, return the original audio
                return input_bytes
            
            buffer = BytesIO()
            first_speaker_segments.export(buffer, format="mp3", parameters=["-ac", "1", "-b:a", "32k"])
            return buffer.getvalue()

@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    output_bytes = Model().inference.remote(input_bytes=path.read_bytes(), suffix=path.suffix)
    
    output_path = path.with_stem(f"{path.stem}_segment").with_suffix(".mp3")
    
    output_path.write_bytes(output_bytes)
    print(f"Audio saved to {output_path}")
