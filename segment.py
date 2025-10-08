# venv/bin/modal run segment.py

local_file_path = "/Users/firozahmed/Downloads/audio.ogg"

from pathlib import Path
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "ffmpeg")
    .uv_pip_install("pydub", "pyannote.audio", "huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App("audio-segment", image=image)

@app.cls(
    gpu="L4",
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Model:
    @modal.method()
    def inference(self, input_bytes: bytes, suffix:str) -> bytes:
        import tempfile
        from io import BytesIO
        from pydub import AudioSegment
        import torch
        import torchaudio
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        pipeline.to(torch.device("cuda"))

        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(input_bytes)
            temp_file.flush()
            
            waveform, sample_rate = torchaudio.load(temp_file.name)
            # Run diarization on the resampled audio
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}).speaker_diarization

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
def main():
    path = Path(local_file_path)
    output_bytes = Model().inference.remote(input_bytes=path.read_bytes(), suffix=path.suffix)
    
    output_path = path.with_stem(f"{path.stem}_segment").with_suffix(".mp3")
    
    output_path.write_bytes(output_bytes)
    print(f"Audio saved to {output_path}")
