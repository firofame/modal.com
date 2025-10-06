# venv/bin/modal run denoise.py

file_path = "/Users/firozahmed/Downloads/audio.mpeg"

from io import BytesIO
from pathlib import Path
import tempfile

import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "git-lfs", "ffmpeg")
    .uv_pip_install("resemble-enhance")
)

app = modal.App("resemble-enhance")

@app.cls(image=image, gpu="T4")
class Model:
    @modal.method()
    def inference(self, input_bytes: bytes, suffix:str) -> bytes:
        import torchaudio
        from torchaudio.io import CodecConfig
        from resemble_enhance.enhancer.inference import denoise

        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(input_bytes)
            temp_file.flush()

            dwav, sr = torchaudio.load(temp_file.name)
            dwav = dwav.mean(dim=0)

            enhanced_wav, new_sr = denoise(dwav, sr, "cuda")

            buffer = BytesIO()
            torchaudio.save(buffer, enhanced_wav.unsqueeze(0).cpu(), new_sr, format="opus")
            return buffer.getvalue()


@app.local_entrypoint()
def main():
    path = Path(file_path)
    output_bytes = Model().inference.remote(input_bytes=path.read_bytes(), suffix=path.suffix)
    
    output_path = path.with_stem(f"{path.stem}_denoise").with_suffix(".opus")
    
    output_path.write_bytes(output_bytes)
    print(f"Enhanced audio saved to {output_path}")
