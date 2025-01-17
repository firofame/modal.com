import subprocess
from modal import (App, Image, web_server, Secret, Volume)

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("git clone https://github.com/lllyasviel/stable-diffusion-webui-forge /root/forge")
    .pip_install("insightface")
)
image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/forge/models")

app = App("forge", image=image)

vol = Volume.from_name("forge-models", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download
    local_dir = f"/root/models/{model_type}"
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("lllyasviel/flux1-dev-bnb-nf4", "flux1-dev-bnb-nf4-v2.safetensors", "Stable-diffusion"),
    ]
    list(hf_download.starmap(models_to_download))


@app.function(allow_concurrent_inputs=2, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*60*2, gpu="L4", volumes={"/root/forge/models": vol})
@web_server(8000, startup_timeout=60*10)
def ui():
    subprocess.Popen("cd /root/forge && python launch.py --listen --port 8000", shell=True)