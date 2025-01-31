import modal
import os
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("torch", "torchvision", "torchaudio", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu126")
    .pip_install("sentencepiece", "peft", "transformers", "accelerate", "git+https://github.com/huggingface/diffusers")
    .pip_install("datasets", "prodigyopt")
    .run_commands("wget -P /root https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py")
    .add_local_dir("data", remote_path="/data", copy=True)
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App(name="train", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="H100", volumes={"/cache": vol}, timeout=60*30)
def train_lora():
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])

    subprocess.run(["rm -f /cache/accelerate/default_config.yaml"], shell=True, check=True)
    from accelerate.utils import write_basic_config
    write_basic_config(mixed_precision="bf16")

    LORA_NAME = "NirmalaSitharaman"

    subprocess.run([
        "accelerate", "launch", "train_dreambooth_lora_flux.py",
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",
        "--dataset_name", "/data",
        "--caption_column=prompt",
        "--output_dir", LORA_NAME,
        "--mixed_precision", "bf16",
        "--instance_prompt", "NirmalaSitharaman woman",
        "--resolution", "512",
        "--train_batch_size", "5",
        "--gradient_accumulation_steps", "1",
        "--optimizer", "prodigy",
        "--learning_rate", "1.",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", "1000",
        "--push_to_hub"
    ], check=True)

@app.local_entrypoint()
def main():
    train_lora.remote()