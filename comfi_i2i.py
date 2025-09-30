# venv/bin/modal run comfi_i2i.py
# https://registry.comfy.org/

prompt = "photograph of a cute muslim boy smiling, 4k"
photo = "photo.jpeg"
gpu = "T4"

from pathlib import Path
import subprocess
import json
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
)

def download_models():
    import os
    from huggingface_hub import hf_hub_download

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

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("custom-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")

app = modal.App(name="comfy-comfi_i2i", image=image, volumes={"/cache": volume})

@app.cls(gpu=gpu)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        seed = random.randint(0, 2**32 - 1)
        workflow_api={"3":{"inputs":{"seed":seed,"steps":30,"cfg":5,"sampler_name":"dpmpp_2m","scheduler":"normal","denoise":0.35,"model":["14",0],"positive":["6",0],"negative":["7",0],"latent_image":["12",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"6":{"inputs":{"text":prompt,"clip":["14",1]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}},"7":{"inputs":{"text":"watermark, text\n","clip":["14",1]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}},"8":{"inputs":{"samples":["3",0],"vae":["14",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"9":{"inputs":{"filename_prefix":"ComfyUI","images":["8",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"10":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"12":{"inputs":{"pixels":["18",0],"vae":["14",2]},"class_type":"VAEEncode","_meta":{"title":"VAE Encode"}},"14":{"inputs":{"ckpt_name":"epiCRealism_XL.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"18":{"inputs":{"upscale_method":"nearest-exact","megapixels":1,"image":["10",0]},"class_type":"ImageScaleToTotalPixels","_meta":{"title":"Scale Image to Total Pixels"}}}
        with open("/root/workflow_api.json", "w") as f:
            json.dump(workflow_api, f)
        subprocess.run("comfy launch --background -- --port 8000", shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose", shell=True, check=True)
        
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]

        for f in Path("/root/comfy/ComfyUI/output").iterdir():
            if f.name.startswith(file_prefix):
                return f.read_bytes()

@app.local_entrypoint()
def main():
    output_bytes = ComfyUI().infer.remote()
    output_path = Path("/Users/firozahmed/Downloads/comfy_output.png")
    output_path.write_bytes(output_bytes)
    print(f"Image saved to {output_path}")