import subprocess
import os
from modal import (App, Image, web_server, Secret, Volume)

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("git clone https://github.com/ostris/ai-toolkit.git /root/ai-toolkit")
    .run_commands("cd /root/ai-toolkit && git submodule update --init --recursive")
    .run_commands("pip install torch")
    .run_commands("pip install -r /root/ai-toolkit/requirements.txt")
    .add_local_file("config.yaml", "/root/config.yaml")
    .add_local_file("images.zip", "/root/images.zip")
    .run_commands("unzip /root/images.zip -d /root/images && rm /root/images.zip")
    .add_local_file("metadata.jsonl", "/root/images/metadata.jsonl")
)

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/ai-toolkit/FLUX.1-dev")

app = App("ai-toolkit", image=image, secrets=[Secret.from_name("huggingface-secret")])

vol = Volume.from_name("toolkit-flux", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir="/root/models")

@app.function(timeout=60*60, gpu="L40S", volumes={"/root/ai-toolkit/FLUX.1-dev": vol})
def start():
    os.chdir("/root/ai-toolkit")
    from toolkit.job import get_job
    job = get_job('/root/config.yaml')
    job.run()