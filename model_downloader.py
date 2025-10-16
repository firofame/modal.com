import os
import subprocess
from pathlib import Path

civit_token = os.environ.get("civit_token")

models_list = {
        "epiCRealism_XL": {
            "name": "epiCRealism_XL.safetensors", 
            "subdir": "checkpoints",
            "url": f"https://civitai.com/api/download/models/1920523?type=Model&format=SafeTensor&token={civit_token}"
        },
        "GFPGANv1.4": {
            "name": "GFPGANv1.4.pth",
            "subdir": "upscale_models",
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        },
        "4x-Ultrasharp": {
            "name": "4x-UltraSharp.pth",
            "subdir": "upscale_models",
            "url": "https://civitai.com/api/download/models/125843?type=Model&format=PickleTensor&token={civit_token}"
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