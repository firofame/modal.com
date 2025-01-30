import subprocess
import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
    .pip_install("comfy-cli", "onnxruntime-gpu")
    .run_commands("comfy --skip-prompt install --nvidia")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy node install https://github.com/Gourieff/ComfyUI-ReActor")
    .run_commands("comfy node install https://github.com/cubiq/ComfyUI_IPAdapter_plus")
    .run_commands("comfy node install https://github.com/city96/ComfyUI-GGUF")
)

def hf_download():
    from huggingface_hub import hf_hub_download, login
    
    HF_TOKEN=os.environ["HF_TOKEN"]
    login(HF_TOKEN)

    flux_gguf = hf_hub_download("city96/FLUX.1-dev-gguf", "flux1-dev-Q8_0.gguf")
    subprocess.run(f"ln -s {flux_gguf} /root/comfy/ComfyUI/models/unet/flux1-dev-Q8_0.gguf", shell=True, check=True)

    flux_vae = hf_hub_download("black-forest-labs/FLUX.1-dev", "ae.safetensors")
    subprocess.run(f"ln -s {flux_vae} /root/comfy/ComfyUI/models/vae/ae.safetensors", shell=True, check=True)

    clip_l = hf_hub_download("comfyanonymous/flux_text_encoders", "clip_l.safetensors")
    subprocess.run(f"ln -s {clip_l} /root/comfy/ComfyUI/models/clip/clip_l.safetensors", shell=True, check=True)

    t5xxl_fp8_e4m3fn_scaled = hf_hub_download("comfyanonymous/flux_text_encoders", "t5xxl_fp8_e4m3fn_scaled.safetensors")
    subprocess.run(f"ln -s {t5xxl_fp8_e4m3fn_scaled} /root/comfy/ComfyUI/models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors", shell=True, check=True)

    firoz_lora = hf_hub_download("firofame/firoz", "firoz.safetensors")
    subprocess.run(f"ln -s {firoz_lora} /root/comfy/ComfyUI/models/loras/firoz.safetensors", shell=True, check=True)

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

image = image.run_function(hf_download, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])

app = modal.App(name="comfyui", image=image)

@app.function(allow_concurrent_inputs=10, concurrency_limit=1, gpu="T4", volumes={"/cache": vol})
@modal.web_server(8188, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen", shell=True)