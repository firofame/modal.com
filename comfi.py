# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

photo = "photo.png"
gpu = "L4"

import subprocess
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
)

def download_models():
    from huggingface_hub import hf_hub_download

    flux1_dev_kontext_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/flux1-kontext-dev_ComfyUI", filename="split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{flux1_dev_kontext_fp8_scaled}' '/root/comfy/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors'", shell=True, check=True)

    t5xxl_fp8_e4m3fn_scaled = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp8_e4m3fn_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{t5xxl_fp8_e4m3fn_scaled}' '/root/comfy/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors'", shell=True, check=True)

    clip_l = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{clip_l}' '/root/comfy/ComfyUI/models/text_encoders/clip_l.safetensors'", shell=True, check=True)

    ae = hf_hub_download(repo_id="Comfy-Org/Lumina_Image_2.0_Repackaged", filename="split_files/vae/ae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{ae}' '/root/comfy/ComfyUI/models/vae/ae.safetensors'", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")

app = modal.App(name="comfy", image=image, volumes={"/cache": volume})
@app.function(max_containers=1, gpu=gpu)
@modal.concurrent(max_inputs=10)
@modal.web_server(8188, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)