# venv/bin/modal run chatterbox_tts.py --text "This is a test"

from io import BytesIO
from pathlib import Path
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .uv_pip_install("numpy")
    .run_commands("git clone https://github.com/resemble-ai/chatterbox.git /chatterbox")
    .run_commands("pip install -e /chatterbox")
)

app = modal.App("chatterbox-tts", image=image)

with image.imports():
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

@app.cls(gpu="T4", enable_memory_snapshot=True)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    @modal.enter()
    def load(self):
        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    @modal.method()
    def generate(self, text:str) -> bytes:
        wav = self.model.generate(text)
        buffer = BytesIO()
        ta.save(buffer, wav, self.model.sr, format="opus")
        return buffer.getvalue()

@app.local_entrypoint()
def main(text: str):
    output_bytes = Chatterbox().generate.remote(text=text)
    output_path = Path("/Users/firozahmed/Downloads/audio.opus")
    output_path.write_bytes(output_bytes)
    print(f"TTS audio saved to {output_path}")
