# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

photo = "photo.png"
gpu = "T4"

import subprocess
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
)

def download_models():
    from huggingface_hub import hf_hub_download

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])

app = modal.App(name="comfy", image=image, volumes={"/cache": volume})
@app.function(max_containers=1, gpu=gpu)
@modal.concurrent(max_inputs=10)
@modal.web_server(8188, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)