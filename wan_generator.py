# venv/bin/modal serve wan_generator.py

from io import BytesIO
from pathlib import Path
import modal
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "wget", "ninja-build")
    .env({"TORCH_CUDA_ARCH_LIST": "8.9"})
    .uv_pip_install("setuptools", "wheel")
    .run_commands(
        "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129",
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt update",
        "apt -y install cuda-toolkit-12-9",
        "pip install git+https://github.com/winggan/SageAttention.git@patch-1",
    )
    .run_commands("git clone https://github.com/deepbeepmeep/Wan2GP.git /Wan2GP")
    .run_commands("pip install -r /Wan2GP/requirements.txt")
)

app = modal.App("Wan2GP", image=image)

with image.imports():    
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

@app.function(max_containers=1, gpu="L4", volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)})
@modal.concurrent(max_inputs=10)
@modal.web_server(7860, startup_timeout=600)
def ui():
    subprocess.Popen("cd /Wan2GP && python wgp.py --server-name 0.0.0.0", shell=True)