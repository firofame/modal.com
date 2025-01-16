import subprocess
import os
from modal import (App, Image, web_server, Secret, Volume)

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("git clone https://github.com/ostris/ai-toolkit.git /root/ai-toolkit")
    .run_commands("cd /root/ai-toolkit && git submodule update --init --recursive")
    .run_commands("pip install -r /root/ai-toolkit/requirements.txt")
)

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/ai-toolkit/FLUX.1-dev")

app = App("ai-toolkit", image=image, secrets=[Secret.from_name("huggingface-secret")])

vol = Volume.from_name("toolkit-flux", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo_id, local_dir="/root/models")

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("black-forest-labs/FLUX.1-dev",),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(allow_concurrent_inputs=2, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*60, gpu="L40S", volumes={"/root/ai-toolkit/FLUX.1-dev": vol})
@web_server(7860, startup_timeout=60*5)
def ui():
    subprocess.Popen("cd /root/ai-toolkit && python flux_train_ui.py", shell=True)