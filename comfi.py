from pathlib import Path
import subprocess
import modal

# https://registry.comfy.org/

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .run_commands("comfy node install ComfyUI-Manager")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

def hf_download():
    from huggingface_hub import hf_hub_download

    firoz = hf_hub_download(repo_id="firofame/firoz", filename="firoz.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {firoz} /root/comfy/ComfyUI/models/loras/firoz.safetensors", shell=True, check=True)

    manjuw = hf_hub_download(repo_id="firofame/manjuw", filename="manjuw.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {manjuw} /root/comfy/ComfyUI/models/loras/manjuw.safetensors", shell=True, check=True)

    RealESRGAN_x2 = hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x2.pth", cache_dir="/cache")
    subprocess.run(f"ln -s {RealESRGAN_x2} /root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x2.pth", shell=True, check=True)

    CyberRealisticXLPlay_V5 = hf_hub_download(repo_id="cyberdelia/CyberRealisticXL", filename="CyberRealisticXLPlay_V5.8.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {CyberRealisticXLPlay_V5} /root/comfy/ComfyUI/models/checkpoints/CyberRealisticXLPlay_V5.8.safetensors", shell=True, check=True)

    Realistic_Vision_V5 = hf_hub_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", filename="Realistic_Vision_V5.1.ckpt", cache_dir="/cache")
    subprocess.run(f"ln -s {Realistic_Vision_V5} /root/comfy/ComfyUI/models/checkpoints/Realistic_Vision_V5.1.ckpt", shell=True, check=True)

    Realistic_Vision_V5_inpainting = hf_hub_download(repo_id="SG161222/Realistic_Vision_V5.1_noVAE", filename="Realistic_Vision_V5.1-inpainting.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {Realistic_Vision_V5_inpainting} /root/comfy/ComfyUI/models/checkpoints/Realistic_Vision_V5.1-inpainting.safetensors", shell=True, check=True)

    dev = hf_hub_download(repo_id="black-forest-labs/FLUX.1-dev", filename="flux1-dev.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {dev} /root/comfy/ComfyUI/models/diffusion_models/flux1-dev.safetensors", shell=True, check=True)

    fill = hf_hub_download(repo_id="black-forest-labs/FLUX.1-Fill-dev", filename="flux1-fill-dev.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {fill} /root/comfy/ComfyUI/models/diffusion_models/flux1-fill-dev.safetensors", shell=True, check=True)

    kontext = hf_hub_download(repo_id="black-forest-labs/FLUX.1-Kontext-dev", filename="flux1-kontext-dev.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {kontext} /root/comfy/ComfyUI/models/diffusion_models/flux1-kontext-dev.safetensors", shell=True, check=True)

    ae = hf_hub_download(repo_id="black-forest-labs/FLUX.1-Kontext-dev", filename="ae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {ae} /root/comfy/ComfyUI/models/vae/ae.safetensors", shell=True, check=True)

    clip_l = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {clip_l} /root/comfy/ComfyUI/models/text_encoders/clip_l.safetensors", shell=True, check=True)

    t5xxl_fp16 = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {t5xxl_fp16} /root/comfy/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors", shell=True, check=True)

secrets = [modal.Secret.from_name("huggingface-secret")]
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(hf_download, volumes={"/cache": vol}, secrets=secrets)

app = modal.App(name="comfy-ui", image=image)

@app.function(max_containers=1, gpu="T4", volumes={"/cache": vol})
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)