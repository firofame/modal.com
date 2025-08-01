from pathlib import Path
import subprocess
import modal

# https://registry.comfy.org/

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "gcc", "g++", "libgl1", "libglib2.0-0")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"CC": "gcc", "CXX": "g++"}) # Install insightface with explicit compiler environment variables
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia")
    .run_commands("comfy node install ComfyUI-Manager")
    .run_commands("comfy node install ComfyUI-Crystools")
    .run_commands("comfy node install comfyui-reactor")
    .run_commands("comfy node install comfyui_ipadapter_plus")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

def hf_download():
    from huggingface_hub import hf_hub_download

    t5xxl_fp8_e4m3fn_scaled = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp8_e4m3fn_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {t5xxl_fp8_e4m3fn_scaled} /root/comfy/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn_scaled.safetensors", shell=True, check=True)

    kontext_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/flux1-kontext-dev_ComfyUI", filename="split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {kontext_fp8_scaled} /root/comfy/ComfyUI/models/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors", shell=True, check=True)

    wan2_2_2_ti2v_5B_fp16 = hf_hub_download(repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename="split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {wan2_2_2_ti2v_5B_fp16} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors", shell=True, check=True)

    wan_2_2_vae = hf_hub_download(repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename="split_files/vae/wan2.2_vae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {wan_2_2_vae} /root/comfy/ComfyUI/models/vae/wan2.2_vae.safetensors", shell=True, check=True)

    wan_2_1_vae = hf_hub_download(repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename="split_files/vae/wan_2.1_vae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {wan_2_1_vae} /root/comfy/ComfyUI/models/vae/wan_2.1_vae.safetensors", shell=True, check=True)

    umt5_xxl_fp16 = hf_hub_download(repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename="split_files/text_encoders/umt5_xxl_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {umt5_xxl_fp16} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors", shell=True, check=True)

    umt5_xxl_fp8_e4m3fn_scaled = hf_hub_download(repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged", filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {umt5_xxl_fp8_e4m3fn_scaled} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", shell=True, check=True)

    Path("/root/comfy/ComfyUI/models/facerestore_models").mkdir(parents=True, exist_ok=True)
    GPEN512 = hf_hub_download(repo_id="martintomov/comfy", filename="facerestore_models/GPEN-BFR-512.onnx", cache_dir="/cache")
    subprocess.run(f"ln -s {GPEN512} /root/comfy/ComfyUI/models/facerestore_models/GPEN-BFR-512.onnx", shell=True, check=True)

    codeformer = hf_hub_download(repo_id="martintomov/comfy", filename="facerestore_models/codeformer.pth", cache_dir="/cache")
    subprocess.run(f"ln -s {codeformer} /root/comfy/ComfyUI/models/facerestore_models/codeformer.pth", shell=True, check=True)

    GFPGANv14 = hf_hub_download(repo_id="martintomov/comfy", filename="facerestore_models/GFPGANv1.4.pth", cache_dir="/cache")
    subprocess.run(f"ln -s {GFPGANv14} /root/comfy/ComfyUI/models/facerestore_models/GFPGANv1.4.pth", shell=True, check=True)

    GFPGANv13 = hf_hub_download(repo_id="martintomov/comfy", filename="facerestore_models/GFPGANv1.3.pth", cache_dir="/cache")
    subprocess.run(f"ln -s {GFPGANv13} /root/comfy/ComfyUI/models/facerestore_models/GFPGANv1.3.pth", shell=True, check=True)

    Path("/root/comfy/ComfyUI/models/ipadapter").mkdir(parents=True, exist_ok=True)
    plusv2_sdxl_ipadapter = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sdxl.bin", cache_dir="/cache")
    subprocess.run(f"ln -s {plusv2_sdxl_ipadapter} /root/comfy/ComfyUI/models/ipadapter/ip-adapter-faceid-plusv2_sdxl.bin", shell=True, check=True)

    plusv2_sdxl_lora = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid-plusv2_sdxl_lora.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {plusv2_sdxl_lora} /root/comfy/ComfyUI/models/loras/ip-adapter-faceid-plusv2_sdxl_lora.safetensors", shell=True, check=True)

    clip_vit = hf_hub_download(repo_id="h94/IP-Adapter", filename="models/image_encoder/model.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {clip_vit} /root/comfy/ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors", shell=True, check=True)

    SDXL4steps = hf_hub_download(repo_id="ByteDance/Hyper-SD", filename="Hyper-SDXL-4steps-lora.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {SDXL4steps} /root/comfy/ComfyUI/models/loras/Hyper-SDXL-4steps-lora.safetensors", shell=True, check=True)

    firoz = hf_hub_download(repo_id="firofame/firoz", filename="firoz.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {firoz} /root/comfy/ComfyUI/models/loras/firoz.safetensors", shell=True, check=True)

    manjuw = hf_hub_download(repo_id="firofame/manjuw", filename="manjuw.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {manjuw} /root/comfy/ComfyUI/models/loras/manjuw.safetensors", shell=True, check=True)

    RealESRGAN_x2 = hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename="RealESRGAN_x2.pth", cache_dir="/cache")
    subprocess.run(f"ln -s {RealESRGAN_x2} /root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x2.pth", shell=True, check=True)

    CyberRealisticXLPlay_V6 = hf_hub_download(repo_id="cyberdelia/CyberRealisticXL", filename="CyberRealisticXLPlay_V6.0.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {CyberRealisticXLPlay_V6} /root/comfy/ComfyUI/models/checkpoints/CyberRealisticXLPlay_V6.0.safetensors", shell=True, check=True)

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

@app.function(max_containers=1, gpu="L4", volumes={"/cache": vol})
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)