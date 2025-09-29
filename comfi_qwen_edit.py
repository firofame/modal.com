# venv/bin/modal run comfi_qwen_edit.py
# https://registry.comfy.org/

prompt = "change the background to country side"
photo = "image.jpg"
gpu = "L4"

from pathlib import Path
import subprocess
import json
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2")
    .uv_pip_install("opencv-python-headless", "huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install qweneditutils")
)

def download_models():
    import os
    from huggingface_hub import hf_hub_download

    token = os.environ["CIVIT_TOKEN"]
    url = f"https://civitai.com/api/download/models/2256755?type=Model&format=SafeTensor&token={token}"
    consistence_edit_v2 = "/cache/consistence_edit_v2.safetensors"
    if not os.path.exists(consistence_edit_v2):
        subprocess.run(f"aria2c -x 8 -c -o {os.path.basename(consistence_edit_v2)} -d {os.path.dirname(consistence_edit_v2)} '{url}'", shell=True, check=True)
    subprocess.run(f"ln -s '{consistence_edit_v2}' '/root/comfy/ComfyUI/models/loras/{os.path.basename(consistence_edit_v2)}'", shell=True, check=True)

    mannequin_clipper = hf_hub_download(repo_id="drbaph/Qwen-Image-Edit-Mannequin-Clipper-LoRA", filename="qwen_image_edit_ mannequin-clipper_v1.0.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s '{mannequin_clipper}' '/root/comfy/ComfyUI/models/loras/qwen_image_edit_ mannequin-clipper_v1.0.safetensors'", shell=True, check=True)

    Qwen_Image_Edit_Lightning_8steps_V1_0_bf16 = hf_hub_download(repo_id="lightx2v/Qwen-Image-Lightning", filename="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {Qwen_Image_Edit_Lightning_8steps_V1_0_bf16} /root/comfy/ComfyUI/models/loras/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors", shell=True, check=True)

    qwen_image_edit_2509_fp8_e4m3fn = hf_hub_download(repo_id="Comfy-Org/Qwen-Image-Edit_ComfyUI", filename="split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_edit_2509_fp8_e4m3fn} /root/comfy/ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors", shell=True, check=True)

    qwen_image_vae = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/vae/qwen_image_vae.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_image_vae} /root/comfy/ComfyUI/models/vae/qwen_image_vae.safetensors", shell=True, check=True)

    qwen_2_5_vl_7b_fp8_scaled = hf_hub_download(repo_id="Comfy-Org/Qwen-Image_ComfyUI", filename="split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {qwen_2_5_vl_7b_fp8_scaled} /root/comfy/ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("custom-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")

app = modal.App(name="comfy-qwen-edit", image=image, volumes={"/cache": volume})

@app.cls(gpu=gpu)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        workflow_api={"1":{"inputs":{"unet_name":"qwen_image_edit_2509_fp8_e4m3fn.safetensors","weight_dtype":"default"},"class_type":"UNETLoader","_meta":{"title":"Load Diffusion Model"}},"2":{"inputs":{"clip_name":"qwen_2.5_vl_7b_fp8_scaled.safetensors","type":"qwen_image","device":"default"},"class_type":"CLIPLoader","_meta":{"title":"Load CLIP"}},"3":{"inputs":{"vae_name":"qwen_image_vae.safetensors"},"class_type":"VAELoader","_meta":{"title":"Load VAE"}},"4":{"inputs":{"lora_name":"Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors","strength_model":1.0000000000000002,"model":["1",0]},"class_type":"LoraLoaderModelOnly","_meta":{"title":"LoraLoaderModelOnly"}},"5":{"inputs":{"conditioning":["84",0]},"class_type":"ConditioningZeroOut","_meta":{"title":"ConditioningZeroOut"}},"21":{"inputs":{"lora_name":"consistence_edit_v2.safetensors","strength_model":0.5000000000000001,"model":["4",0]},"class_type":"LoraLoaderModelOnly","_meta":{"title":"LoraLoaderModelOnly"}},"25":{"inputs":{"seed":810617380969497,"steps":8,"cfg":1,"sampler_name":"er_sde","scheduler":"beta","denoise":1,"model":["21",0],"positive":["84",0],"negative":["5",0],"latent_image":["84",1]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"27":{"inputs":{"samples":["25",0],"vae":["3",0]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"84":{"inputs":{"prompt":prompt,"target_size":1024,"target_vl_size":384,"upscale_method":"lanczos","crop":"center","instruction":"Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.","clip":["2",0],"vae":["3",0],"vl_resize_image1":["107",0]},"class_type":"TextEncodeQwenImageEditPlusAdvance_lrzjason","_meta":{"title":"TextEncodeQwenImageEditPlusAdvance 小志Jason(xiaozhijason)"}},"106":{"inputs":{"filename_prefix":"ComfyUI","images":["27",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"107":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}}}
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