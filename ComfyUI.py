import subprocess
import modal
import os

image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "unzip")
image = image.pip_install("comfy-cli").run_commands("comfy --skip-prompt install --nvidia")
image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/comfy/ComfyUI/models")
image = image.run_commands("comfy node install https://github.com/crystian/ComfyUI-Crystools")
image = image.run_commands("comfy node install https://github.com/city96/ComfyUI-GGUF")

app = modal.App(name="ComfyUI", image=image)

vol = modal.Volume.from_name("ComfyUI-models", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download
    local_dir = f"/root/models/{model_type}"
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    
    if filename.endswith(".zip"):
        subprocess.run(["unzip", f"{local_dir}/{filename}", "-d", local_dir], check=True)

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf", "unet"),
        ("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors", "clip"),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip"),
        ("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "vae"),
        ("firofame/firoz", "firoz.safetensors", "loras"),
        ("firofame/thilakan", "thilakan.safetensors", "loras"),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(allow_concurrent_inputs=2, concurrency_limit=1, container_idle_timeout=1200, gpu="L4", volumes={"/root/comfy/ComfyUI/models": vol})
@modal.web_server(8188, startup_timeout=300)
def ui():
    subprocess.Popen("comfy launch -- --listen", shell=True)