import subprocess
import modal

image = modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libglib2.0-0", "unzip")
image = image.pip_install("comfy-cli").run_commands("comfy --skip-prompt install --nvidia")
image = image.run_commands("comfy node install https://github.com/crystian/ComfyUI-Crystools")
image = image.run_commands("comfy node install https://github.com/city96/ComfyUI-GGUF")
image = image.run_commands("comfy node install https://github.com/kijai/ComfyUI-SUPIR")
image = image.run_commands("comfy node install https://github.com/Gourieff/ComfyUI-ReActor")

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/comfy/ComfyUI/models")

app = modal.App(name="ComfyUI", image=image)

vol = modal.Volume.from_name("ComfyUI-models", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download
    local_dir = f"/root/models/{model_type}"
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    
    if filename.endswith(".zip"):
        subprocess.run(["unzip", f"{local_dir}/{filename}", "-d", local_dir], check=True)

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("AdamCodd/vit-base-nsfw-detector", "config.json", "nsfw_detector/vit-base-nsfw-detector"),
        ("AdamCodd/vit-base-nsfw-detector", "confusion_matrix.png", "nsfw_detector/vit-base-nsfw-detector"),
        ("AdamCodd/vit-base-nsfw-detector", "model.safetensors", "nsfw_detector/vit-base-nsfw-detector"),
        ("AdamCodd/vit-base-nsfw-detector", "preprocessor_config.json", "nsfw_detector/vit-base-nsfw-detector"),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(allow_concurrent_inputs=2, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*60*2, gpu="L4", volumes={"/root/comfy/ComfyUI/models": vol})
@modal.web_server(8188, startup_timeout=60*5)
def ui():
    subprocess.Popen("comfy launch -- --listen", shell=True)