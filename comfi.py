# venv/bin/modal run comfi.py
# https://registry.comfy.org/

prompt = "change the background to green"
photo = "photo.jpg"

import subprocess
import modal
from pathlib import Path

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2", "ffmpeg")
    .uv_pip_install("opencv-python-headless", "huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools qweneditutils")
)

def download_models():
    from huggingface_hub import hf_hub_download
    import os

    models = [
        {"id": "1920523", "name": "epiCRealism_XL.safetensors", "subdir": "checkpoints"},
    ]
    cache_dir = "/cache"
    comfy_dir = "/root/comfy/ComfyUI/models/"
    token = os.environ["CIVIT_TOKEN"]
    for m in models:
        url = f"https://civitai.com/api/download/models/{m['id']}?type=Model&format=SafeTensor&token={token}"
        dest = os.path.join(cache_dir, m["name"])
        link = os.path.join(comfy_dir, m["subdir"], m["name"])
        if not os.path.exists(dest):
            subprocess.run(f"aria2c -x 8 -c -o {os.path.basename(dest)} -d {os.path.dirname(dest)} '{url}'", shell=True, check=True)        
        subprocess.run(f"ln -s '{dest}' '{link}'", shell=True, check=True)

    Qwen_Rapid_AIO_v4 = hf_hub_download(repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO", filename="Qwen-Rapid-AIO-v4.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{Qwen_Rapid_AIO_v4}' '/root/comfy/ComfyUI/models/checkpoints/Qwen-Rapid-AIO-v4.safetensors'", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("custom-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")

app = modal.App(name="comfy", image=image, volumes={"/cache": volume})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=60)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(gpu="L40s")
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        import json

        seed = random.randint(0, 2**32 - 1)
        workflow_api={"1":{"inputs":{"ckpt_name":"Qwen-Rapid-AIO-v4.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"2":{"inputs":{"seed":seed,"steps":4,"cfg":1,"sampler_name":"sa_solver","scheduler":"beta","denoise":1,"model":["1",0],"positive":["57",0],"negative":["11",0],"latent_image":["57",6]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"5":{"inputs":{"samples":["2",0],"vae":["1",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"8":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Input Image"}},"11":{"inputs":{"conditioning":["57",0]},"class_type":"ConditioningZeroOut","_meta":{"title":"ConditioningZeroOut"}},"12":{"inputs":{"filename_prefix":"ComfyUI","images":["5",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"57":{"inputs":{"prompt":prompt,"enable_resize":True,"enable_vl_resize":True,"skip_first_image_resize":False,"upscale_method":"lanczos","crop":"disabled","instruction":"Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.","clip":["1",1],"vae":["1",2],"image1":["8",0]},"class_type":"TextEncodeQwenImageEditPlus_lrzjason","_meta":{"title":"TextEncodeQwenImageEditPlus 小志Jason(xiaozhijason)"}}}
        with open("/root/workflow_api.json", "w") as f:
            json.dump(workflow_api, f)
        subprocess.run("comfy launch --background", shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose", shell=True, check=True)
        return Path("/root/comfy/ComfyUI/output/ComfyUI_00001_.png").read_bytes()

@app.local_entrypoint()
def main():
    import datetime

    output_bytes = ComfyUI().infer.remote()
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"/Users/firozahmed/Downloads/comfy_{datetime}.png")
    output_path.write_bytes(output_bytes)
    print(f"Output saved to {output_path}")