import sys
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git")
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126")
    .run_commands("git clone https://github.com/ostris/ai-toolkit /root/ai-toolkit")
    .run_commands("cd /root/ai-toolkit && git submodule update --init --recursive")
    .pip_install("sentencepiece", "peft", "transformers", "accelerate", "git+https://github.com/huggingface/diffusers")
    .add_local_dir("data", remote_path="/data", copy=True)
    .add_local_file("config.yaml", remote_path="/config.yaml", copy=True)
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App("ai-toolkit", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="L40s", volumes={"/cache": vol}, timeout=60*30)
def train():
    from accelerate.utils import write_basic_config
    write_basic_config(mixed_precision="bf16")

    sys.path.insert(0, "/root/ai-toolkit")
    from toolkit.job import get_job
    job = get_job("/config.yaml")
    job.run()

@app.local_entrypoint()
def main():
    train.remote()