import os
import subprocess
from modal import App, Image, Secret, Volume

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip", "wget")
    .run_commands("wget -P /root https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py")
    .run_commands("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 -U -q")
    .run_commands("pip install git+https://github.com/huggingface/diffusers -U -q")
    .run_commands("pip install git+https://github.com/huggingface/transformers -U -q")
    .run_commands("pip install git+https://github.com/huggingface/accelerate -U -q")
    .run_commands("pip install -r https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/requirements_flux.txt -U -q")
    .run_commands("pip install prodigyopt -U -q")
    .add_local_file("images.zip", "/root/images.zip", copy=True)
    .run_commands("unzip /root/images.zip -d /root/images && rm /root/images.zip")
)

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/models")

app = App("dreambooth", image=image, secrets=[Secret.from_name("huggingface-secret")])

vol = Volume.from_name("dreambooth-models", create_if_missing=True)

@app.function(timeout=60*60, volumes={"/root/models": vol}, gpu="L40S")
def run_training():
    from accelerate.utils import write_basic_config
    write_basic_config()

    command = [
        "accelerate", "launch", "train_dreambooth_lora_flux.py",
        "--pretrained_model_name_or_path=/root/models/flux",
        "--instance_data_dir=/root/images",
        "--instance_prompt=manjuw woman",
        "--optimizer=prodigy",
        "--learning_rate=1.",
        "--max_train_steps=1000",
        "--mixed_precision=bf16",
        "--train_batch_size=2",
        "--push_to_hub"
    ]
    subprocess.run(command, check=True)

@app.local_entrypoint()
def main():
    run_training.remote()

@app.function(volumes={"/root/models": vol})
def download():
    from huggingface_hub import snapshot_download, hf_hub_download
    snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir='/root/models/flux', token=os.environ["HF_TOKEN"])
    subprocess.run(["ls", "/root/models/flux"], check=True)
