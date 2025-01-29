import subprocess
import modal
import os

image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0", "unzip")
image = image.run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
image = image.pip_install("comfy-cli").run_commands("comfy --skip-prompt install --nvidia")
image = image.pip_install("insightface", "onnxruntime")
image = image.run_commands("comfy node install https://github.com/Gourieff/ComfyUI-ReActor")
image = image.run_commands("comfy node install https://github.com/cubiq/ComfyUI_IPAdapter_plus")

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/comfy/ComfyUI/models")

app = modal.App(name="ComfyUI", image=image)

vol = modal.Volume.from_name("ComfyUI-models", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def rm_file():
    subprocess.run("rm -rf /root/models/vae/pytorch_lora_weights.safetensors", shell=True)

@app.function(volumes={"/root/models": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])
def hf_download(repo_id: str, filename: str, model_type: str):
    HF_TOKEN = os.environ["HF_TOKEN"]
    from huggingface_hub import hf_hub_download
    local_dir = f"/root/models/{model_type}"
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, token=HF_TOKEN)
    
    if filename.endswith(".zip"):
        subprocess.run(["unzip", f"{local_dir}/{filename}", "-d", local_dir], check=True)

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("cyberdelia/CyberRealistic", "CyberRealistic_V7.0_FP32.safetensors", "checkpoints"),
        ("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.safetensors", "vae"),
        ("ezioruan/inswapper_128.onnx", "inswapper_128.onnx", "insightface"),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(allow_concurrent_inputs=100, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*60*2, gpu="T4", volumes={"/root/comfy/ComfyUI/models": vol})
@modal.web_server(8188, startup_timeout=60*5)
def ui():
    subprocess.Popen("comfy launch -- --listen", shell=True)
