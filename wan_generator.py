# venv/bin/modal serve wan_generator.py

from io import BytesIO
from pathlib import Path
import modal
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands("git clone https://github.com/deepbeepmeep/Wan2GP.git /Wan2GP")
    .run_commands("pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128")
    .run_commands("pip install -r /Wan2GP/requirements.txt")
)

app = modal.App("Wan2GP", image=image)

with image.imports():    
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

@app.function(max_containers=1, gpu="T4", volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)})
@modal.concurrent(max_inputs=10)
@modal.web_server(7860, startup_timeout=600)
def ui():
    subprocess.Popen("cd /Wan2GP && python wgp.py --server-name 0.0.0.0", shell=True)