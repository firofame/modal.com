# venv/bin/modal run denoise.py --file-path /Users/firozahmed/Downloads/audio.ogg

from io import BytesIO
from pathlib import Path
import tempfile

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "git-lfs", "ffmpeg")
    .uv_pip_install("resemble-enhance")
)

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": str(CACHE_DIR)})


app = modal.App("resemble-enhance")

with image.imports():
    import torchaudio
    from resemble_enhance.enhancer.inference import denoise

@app.cls(
    image=image, gpu="T4", volumes=volumes, secrets=secrets, scaledown_window=240
)
class Model:
    @modal.enter()
    def enter(self):
        self.device = "cuda"

    @modal.method()
    def inference(self, input_bytes: bytes, suffix:str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(input_bytes)
            temp_file.flush()

            dwav, sr = torchaudio.load(temp_file.name)
            dwav = dwav.mean(dim=0)

            enhanced_wav, new_sr = denoise(dwav, sr, self.device)

            buffer = BytesIO()
            torchaudio.save(buffer, enhanced_wav.unsqueeze(0).cpu(), new_sr, format="opus")
            return buffer.getvalue()


@app.local_entrypoint()
def main(file_path: str):
    path = Path(file_path)
    output_bytes = Model().inference.remote(input_bytes=path.read_bytes(), suffix=path.suffix)
    
    output_path = path.with_stem(f"{path.stem}_denoise").with_suffix(".opus")
    
    output_path.write_bytes(output_bytes)
    print(f"Enhanced audio saved to {output_path}")
