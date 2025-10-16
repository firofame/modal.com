import os
import subprocess
from pathlib import Path

civit_token = os.environ.get("civit_token")

models_list = {
        "Qwen_Image_Edit_Rapid_AIO_v5": {
            "name": "Qwen-Rapid-AIO-v5.safetensors", 
            "subdir": "checkpoints",
            "url": f"https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v5/Qwen-Rapid-AIO-NSFW-v5.safetensors"
        },
        "epiCRealism_XL": {
            "name": "epiCRealism_XL.safetensors", 
            "subdir": "checkpoints",
            "url": f"https://civitai.com/api/download/models/1920523?type=Model&format=SafeTensor&token={civit_token}"
        },
        "GFPGANv1_4": {
            "name": "GFPGANv1.4.pth",
            "subdir": "upscale_models",
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        },
        "4x_Ultrasharp": {
            "name": "4x-UltraSharp.pth",
            "subdir": "upscale_models",
            "url": "https://civitai.com/api/download/models/125843?type=Model&format=PickleTensor&token={civit_token}"
        },
        "Wan2_1_InfiniteTalk_Single_Q8": {
            "name": "Wan2_1-InfiniteTalk_Single_Q8.gguf",
            "subdir": "unet",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"
        },
        "wan2_1_i2v_14b_480p_Q8_0": {
            "name": "wan2.1-i2v-14b-480p-Q8_0.gguf",
            "subdir": "unet",
            "url": "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf"
        },
        "Wan2_1_VAE_bf16": {
            "name": "Wan2_1_VAE_bf16.safetensors",
            "subdir": "vae",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
        },
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16": {
            "name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
            "subdir": "loras",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
        },
        "clip_vision_h": {
            "name": "clip_vision_h.safetensors",
            "subdir": "clip_vision",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
        },
        "umt5_xxl_enc_bf16": {
            "name": "umt5-xxl-enc-bf16.safetensors",
            "subdir": "text_encoders",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
        },
        "MelBandRoformer_fp16": {
            "name": "MelBandRoformer_fp16.safetensors",
            "subdir": "diffusion_models",
            "url": "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors"
        },
        "wav2vec2_chinese_base_fp16": {
            "name": "wav2vec2-chinese-base_fp16.safetensors",
            "subdir": "wav2vec2",
            "url": "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
        },
    }

def download_and_link(model_id):
    model_info = models_list.get(model_id)
    model_name = model_info["name"]
    model_subdir = model_info["subdir"]
    model_url = model_info["url"]

    # Define paths using pathlib for better cross-platform handling
    cache_path = Path("/cache") / model_name
    link_dir = Path("/root/comfy/ComfyUI/models") / model_subdir
    link_path = link_dir / model_name
    
    # 1. Ensure the target link directory exists
    link_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Download the file to the cache if it doesn't exist
    if not cache_path.exists():
        print(f"Downloading {model_name}...")
        try:
            subprocess.run([
                "aria2c", "-x", "8", "-c",
                "-o", str(cache_path.name),
                "-d", str(cache_path.parent),
                model_url
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {model_name}: {e.stderr}")
            return # Stop processing this model if download fails
    else:
        print(f"'{model_name}' already found in cache. Skipping download.")

    # 3. Create a symbolic link if it doesn't already exist
    if not link_path.is_symlink():
        # Important: Ensure the target file actually exists before linking
        if cache_path.exists():
            print(f"Linking '{cache_path}' to '{link_path}'")
            os.symlink(cache_path, link_path)
        else:
            print(f"Cannot create link. Source file '{cache_path}' not found.")
    else:
        print(f"Link for '{model_name}' already exists. Skipping.")