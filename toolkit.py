import subprocess
import os
from modal import (App, Image, web_server, Secret, Volume)

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
    .run_commands("git clone -b fix/gradient-checkpointing --single-branch https://github.com/jhj0517/ai-toolkit.git /root/ai-toolkit")
    .run_commands("cd /root/ai-toolkit && git submodule update --init --recursive")
    .run_commands("pip install -r /root/ai-toolkit/requirements.txt")
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = App("ai-toolkit", image=image, secrets=[Secret.from_name("huggingface-secret")])

vol = Volume.from_name("hf-cache", create_if_missing=True)

@app.local_entrypoint()
def main():
    print("install complete")

@app.function(gpu="L40s", allow_concurrent_inputs=10, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*30, volumes={"/cache": vol})
@web_server(7860, startup_timeout=60*5)
def ui():
    subprocess.Popen("cd /root/ai-toolkit && python flux_train_ui.py", shell=True)