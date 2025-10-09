# venv/bin/modal run comfi_krea.py
# https://registry.comfy.org/

prompt = "A cute, chubby Muslim woman standing confidently facing forward, head held straight, smiling with a happy and warm expression. She is wearing modest, stylish Islamic clothing (such as a colorful hijab and a long tunic or abaya), and the image is captured in a realistic photo style showing her from head to toe."
width = 768
height = 1024
gpu = "L4"

from pathlib import Path
import subprocess
import json
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
)

def download_models():
    from huggingface_hub import hf_hub_download

    flux1_krea_dev_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/FLUX.1-Krea-dev_ComfyUI", filename="split_files/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{flux1_krea_dev_fp8_scaled}' '/root/comfy/ComfyUI/models/diffusion_models/flux1-krea-dev_fp8_scaled.safetensors'", shell=True, check=True)

    ae = hf_hub_download(repo_id="Comfy-Org/Lumina_Image_2.0_Repackaged", filename="split_files/vae/ae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{ae}' '/root/comfy/ComfyUI/models/vae/ae.safetensors'", shell=True, check=True)

    t5xxl_fp16 = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{t5xxl_fp16}' '/root/comfy/ComfyUI/models/text_encoders/t5xxl_fp16.safetensors'", shell=True, check=True)

    clip_l = hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="clip_l.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{clip_l}' '/root/comfy/ComfyUI/models/text_encoders/clip_l.safetensors'", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret")])

app = modal.App(name="comfi-krea", image=image, volumes={"/cache": volume})

@app.cls(gpu=gpu)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        seed = random.randint(0, 2**32 - 1)
        workflow_api={"8":{"inputs":{"samples":["31",0],"vae":["39",0]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"9":{"inputs":{"filename_prefix":"flux_krea/flux_krea","images":["8",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"27":{"inputs":{"width":width,"height":height,"batch_size":1},"class_type":"EmptySD3LatentImage","_meta":{"title":"EmptySD3LatentImage"}},"31":{"inputs":{"seed":seed,"steps":20,"cfg":1,"sampler_name":"euler","scheduler":"simple","denoise":1,"model":["38",0],"positive":["45",0],"negative":["42",0],"latent_image":["27",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"38":{"inputs":{"unet_name":"flux1-krea-dev_fp8_scaled.safetensors","weight_dtype":"default"},"class_type":"UNETLoader","_meta":{"title":"Load Diffusion Model"}},"39":{"inputs":{"vae_name":"ae.safetensors"},"class_type":"VAELoader","_meta":{"title":"Load VAE"}},"40":{"inputs":{"clip_name1":"clip_l.safetensors","clip_name2":"t5xxl_fp16.safetensors","type":"flux","device":"default"},"class_type":"DualCLIPLoader","_meta":{"title":"DualCLIPLoader"}},"42":{"inputs":{"conditioning":["45",0]},"class_type":"ConditioningZeroOut","_meta":{"title":"ConditioningZeroOut"}},"45":{"inputs":{"text":prompt,"clip":["40",0]},"class_type":"CLIPTextEncode","_meta":{"title":"CLIP Text Encode (Prompt)"}}}
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

        output_dir = Path("/root/comfy/ComfyUI/output")
        for f in output_dir.rglob(f"{Path(file_prefix).name}*"):
            if str(f.relative_to(output_dir)).startswith(file_prefix):
                return f.read_bytes()

@app.local_entrypoint()
def main():
    output_bytes = ComfyUI().infer.remote()
    output_path = Path("/Users/firozahmed/Downloads/comfy_output.png")
    output_path.write_bytes(output_bytes)
    print(f"Image saved to {output_path}")