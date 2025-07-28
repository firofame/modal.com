import subprocess
from pathlib import Path
import modal

PORT = 8675

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]")
    .run_commands("git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI")
    .run_commands("cd /ComfyUI && pip install -r requirements.txt")
)

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("comfy-ui", image=image, volumes=volumes, secrets=secrets)

@app.function(
    gpu="T4",
    cpu=2,
    memory=1024,
    timeout=3600,
    min_containers=1,  # Keep at least one instance of the server running.
)
@modal.concurrent(max_inputs=100)  # Allow 100 concurrent requests per container.
@modal.web_server(port=PORT, startup_timeout=300)
def run():
    subprocess.Popen("cd /ComfyUI && python main.py --listen --port 8675", shell=True)