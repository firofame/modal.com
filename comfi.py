# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

from pathlib import Path
import subprocess
import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "gcc", "g++", "libgl1", "libglib2.0-0", "wget")
    .uv_pip_install("huggingface-hub[hf-transfer]", "git+https://github.com/Comfy-Org/comfy-cli")
    .env({"CC": "gcc", "CXX": "g++", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129")
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt update",
        "apt -y install cuda-toolkit-12-9",
    )
    .run_commands("git clone https://github.com/thu-ml/SageAttention /SageAttention")
    .run_commands(
        "cd /SageAttention && "
        "sed -i 's/raise RuntimeError(\"No GPUs found.*\")/print(\"⚠️ No GPUs at build time, using TORCH_CUDA_ARCH_LIST\")/' setup.py && "
        "export TORCH_CUDA_ARCH_LIST=\"8.6\" EXT_PARALLEL=4 NVCC_APPEND_FLAGS=\"--threads 8\" MAX_JOBS=32 && "
        "pip install -e ."
    )
    .run_commands("comfy --skip-prompt install --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
    .run_commands("comfy node install comfyui-reactor")
    .run_commands("comfy node install comfyui_ipadapter_plus")
    .run_commands("comfy node install comfyui-mmaudio")
    .run_commands("comfy node install comfyui-videohelpersuite")
    .run_commands("comfy node install ComfyUI-WanVideoWrapper")
    .run_commands("comfy node install comfyui-kjnodes")
    .run_commands("comfy node install ComfyUI-MelBandRoFormer")
    .run_commands("comfy node install ComfyUI-VibeVoice")
)

def hf_download():
    from huggingface_hub import hf_hub_download

    qwen_image_vae = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/vae/qwen_image_vae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_vae} /root/comfy/ComfyUI/models/vae/qwen_image_vae.safetensors", shell=True, check=True)

    qwen_2_5_vl_7b_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_2_5_vl_7b_fp8_scaled} /root/comfy/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", shell=True, check=True)

    qwen_image_fp8_e4m3fn = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_fp8_e4m3fn} /root/comfy/ComfyUI/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors", shell=True, check=True)

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

    Path("/root/comfy/ComfyUI/models/mmaudio").mkdir(parents=True, exist_ok=True)
    mmaudio_large = hf_hub_download(repo_id="Kijai/MMAudio_safetensors", filename="mmaudio_large_44k_v2_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {mmaudio_large} /root/comfy/ComfyUI/models/mmaudio/mmaudio_large_44k_v2_fp16.safetensors", shell=True, check=True)

    mmaudio_synchformer = hf_hub_download(repo_id="Kijai/MMAudio_safetensors", filename="mmaudio_synchformer_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {mmaudio_synchformer} /root/comfy/ComfyUI/models/mmaudio/mmaudio_synchformer_fp16.safetensors", shell=True, check=True)

    apple_dfn5b = hf_hub_download(repo_id="Kijai/MMAudio_safetensors", filename="apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {apple_dfn5b} /root/comfy/ComfyUI/models/mmaudio/apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors", shell=True, check=True)

    mmaudio_vae = hf_hub_download(repo_id="Kijai/MMAudio_safetensors", filename="mmaudio_vae_44k_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {mmaudio_vae} /root/comfy/ComfyUI/models/mmaudio/mmaudio_vae_44k_fp16.safetensors", shell=True, check=True)

    melband_roformer = hf_hub_download(repo_id="Kijai/MelBandRoFormer_comfy", filename="MelBandRoformer_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {melband_roformer} /root/comfy/ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors", shell=True, check=True)

    lightx2v = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {lightx2v} /root/comfy/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", shell=True, check=True)

    wan2_1_infinitetalk = hf_hub_download(repo_id="Kijai/WanVideo_comfy_GGUF", filename="InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf", cache_dir="/cache")
    subprocess.run(f"ln -s {wan2_1_infinitetalk} /root/comfy/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf", shell=True, check=True)

    clip_vision_h = hf_hub_download(repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged", filename="split_files/clip_vision/clip_vision_h.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {clip_vision_h} /root/comfy/ComfyUI/models/clip_vision/clip_vision_h.safetensors", shell=True, check=True)

    wan2_1_i2v_14b_480p = hf_hub_download(repo_id="city96/Wan2.1-I2V-14B-480P-gguf", filename="wan2.1-i2v-14b-480p-Q8_0.gguf", cache_dir="/cache")
    subprocess.run(f"ln -s {wan2_1_i2v_14b_480p} /root/comfy/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf", shell=True, check=True)

    umt5_xxl_enc_bf16 = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="umt5-xxl-enc-bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {umt5_xxl_enc_bf16} /root/comfy/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors", shell=True, check=True)

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(hf_download, volumes={"/cache": vol}, secrets=[modal.Secret.from_name("huggingface-secret")])

app = modal.App(name="comfy-ui", image=image)

@app.function(max_containers=1, gpu="L4", volumes={"/cache": vol})
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)