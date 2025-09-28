# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

workflow_api={"3":{"inputs":{"seed":913629450015772,"steps":4,"cfg":1,"sampler_name":"euler","scheduler":"simple","denoise":1,"model":["75",0],"positive":["111",0],"negative":["110",0],"latent_image":["88",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"8":{"inputs":{"samples":["3",0],"vae":["39",0]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"37":{"inputs":{"unet_name":"qwen_image_edit_2509_fp8_e4m3fn.safetensors","weight_dtype":"default"},"class_type":"UNETLoader","_meta":{"title":"Load Diffusion Model"}},"38":{"inputs":{"clip_name":"qwen_2.5_vl_7b_fp8_scaled.safetensors","type":"qwen_image","device":"default"},"class_type":"CLIPLoader","_meta":{"title":"Load CLIP"}},"39":{"inputs":{"vae_name":"qwen_image_vae.safetensors"},"class_type":"VAELoader","_meta":{"title":"Load VAE"}},"60":{"inputs":{"filename_prefix":"ComfyUI","images":["8",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"66":{"inputs":{"shift":3,"model":["89",0]},"class_type":"ModelSamplingAuraFlow","_meta":{"title":"ModelSamplingAuraFlow"}},"75":{"inputs":{"strength":1,"model":["66",0]},"class_type":"CFGNorm","_meta":{"title":"CFGNorm"}},"78":{"inputs":{"image":"photo.jpeg"},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"88":{"inputs":{"pixels":["93",0],"vae":["39",0]},"class_type":"VAEEncode","_meta":{"title":"VAE Encode"}},"89":{"inputs":{"lora_name":"Qwen-Image-Lightning-4steps-V1.0.safetensors","strength_model":1,"model":["115",0]},"class_type":"LoraLoaderModelOnly","_meta":{"title":"LoraLoaderModelOnly"}},"93":{"inputs":{"upscale_method":"lanczos","megapixels":1,"image":["78",0]},"class_type":"ImageScaleToTotalPixels","_meta":{"title":"Scale Image to Total Pixels"}},"110":{"inputs":{"prompt":"","clip":["38",0],"vae":["39",0],"image1":["93",0]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus"}},"111":{"inputs":{"prompt":"remove clothing","clip":["38",0],"vae":["39",0],"image1":["93",0]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus"}},"112":{"inputs":{"width":1024,"height":1024,"batch_size":1},"class_type":"EmptySD3LatentImage","_meta":{"title":"EmptySD3LatentImage"}},"115":{"inputs":{"lora_name":"qwen_image_edit_ mannequin-clipper_v1.0.safetensors","strength_model":1,"model":["37",0]},"class_type":"LoraLoaderModelOnly","_meta":{"title":"LoraLoaderModelOnly"}}}

photo = "photo.jpeg"

from pathlib import Path
import subprocess
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "ninja-build", "libgl1", "libglib2.0-0", "wget")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    # .uv_pip_install("setuptools", "wheel", "ninja")
    # .run_commands('TORCH_CUDA_ARCH_LIST="8.9" pip install --use-pep517 --no-build-isolation git+https://github.com/winggan/SageAttention.git@patch-1')
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
)

def download_models():
    from huggingface_hub import hf_hub_download

    mannequin_clipper = hf_hub_download(repo_id="drbaph/Qwen-Image-Edit-Mannequin-Clipper-LoRA", filename="qwen_image_edit_ mannequin-clipper_v1.0.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{mannequin_clipper}' '/root/comfy/ComfyUI/models/loras/qwen_image_edit_ mannequin-clipper_v1.0.safetensors'", shell=True, check=True)

    Qwen_Lora_4steps = hf_hub_download(repo_id="lightx2v/Qwen-Image-Lightning", filename="Qwen-Image-Lightning-4steps-V1.0.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {Qwen_Lora_4steps} /root/comfy/ComfyUI/models/loras/Qwen-Image-Lightning-4steps-V1.0.safetensors", shell=True, check=True)

    qwen_image_edit_2509_fp8_e4m3fn = hf_hub_download(repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI", filename="split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_edit_2509_fp8_e4m3fn} /root/comfy/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", shell=True, check=True)

    qwen_image_vae = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/vae/qwen_image_vae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_vae} /root/comfy/ComfyUI/models/vae/qwen_image_vae.safetensors", shell=True, check=True)

    qwen_2_5_vl_7b_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_2_5_vl_7b_fp8_scaled} /root/comfy/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")

app = modal.App(name="comfy-qwen-edit", image=image, volumes={"/cache": volume})
@app.function(max_containers=1, gpu="L4")
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)