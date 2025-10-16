# venv/bin/modal run comfi.py
# https://registry.comfy.org/

photo = "photo.jpeg"

def download_models():
    from model_downloader import download_and_link
    download_and_link("epiCRealism_XL")
    download_and_link("SkinDiffDetail")

import modal
import subprocess
from pathlib import Path

volume = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2")
    .uv_pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
    .add_local_file("./model_downloader.py", remote_path="/root/model_downloader.py", copy=True)
    .run_function(download_models, volumes={"/cache": volume})
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")
)
app = modal.App(name="comfy", image=image, volumes={"/cache": volume})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=60)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(gpu="T4")
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        import json

        seed = random.randint(0, 2**32 - 1)
        workflow_api={"205":{"inputs":{"seed":seed,"steps":20,"cfg":3.5,"sampler_name":"dpmpp_2m_sde","scheduler":"karras","denoise":0.03,"model":["219",0],"positive":["207",0],"negative":["208",0],"latent_image":["210",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"207":{"inputs":{"text":"Realistic skin detail, blemishes, imperfections, pores","clip":["219",1]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}},"208":{"inputs":{"text":"","clip":["219",1]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}},"210":{"inputs":{"pixels":["214",0],"vae":["219",2]},"class_type":"VAEEncode","_meta":{"title":"VAE Encode"}},"212":{"inputs":{"samples":["205",0],"vae":["219",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"214":{"inputs":{"upscale_method":"lanczos","scale_by":1.5,"image":["225",0]},"class_type":"ImageScaleBy","_meta":{"title":"Upscale Image By"}},"216":{"inputs":{"upscale_method":"lanczos","megapixels":1,"image":["212",0]},"class_type":"ImageScaleToTotalPixels","_meta":{"title":"Scale Image to Total Pixels"}},"219":{"inputs":{"ckpt_name":"epiCRealism_XL.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"220":{"inputs":{"upscale_model":["221",0],"image":["216",0]},"class_type":"ImageUpscaleWithModel","_meta":{"title":"Upscale Image (using Model)"}},"221":{"inputs":{"model_name":"1x-ITF-SkinDiffDetail-Lite-v1.pth"},"class_type":"UpscaleModelLoader","_meta":{"title":"Load Upscale Model"}},"223":{"inputs":{"filename_prefix":"ComfyUI","images":["220",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"225":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}}}
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