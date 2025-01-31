import modal
import os
import subprocess

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("torch", "torchvision", "torchaudio", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu126")
    .pip_install("sentencepiece", "peft", "transformers", "accelerate", "git+https://github.com/huggingface/diffusers")
    .pip_install("wandb")
    .run_commands("wget -P /root https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py")
    .add_local_dir("data", remote_path="/data", copy=True)
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App(name="train", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="L4", volumes={"/cache": vol})
def train_lora():
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])

    import wandb
    wandb.login(key="20aa4ebf67f6fb866cacbdb4376a59c403377b6b")

    subprocess.run(["accelerate config default"], shell=True, check=True)
    subprocess.run(["accelerate env"], shell=True, check=True)

    LORA_NAME = "trained-flux"

    subprocess.run([
        "accelerate", "launch", "train_dreambooth_lora_flux.py",
        "--pretrained_model_name_or_path", "black-forest-labs/FLUX.1-dev",
        "--instance_data_dir", "/data",
        "--output_dir", LORA_NAME,
        "--mixed_precision", "bf16",
        "--instance_prompt", "a photo of sks woman",
        "--resolution", "1024",
        "--train_batch_size", "1",
        "--guidance_scale", "1",
        "--gradient_accumulation_steps", "4",
        "--optimizer", "prodigy",
        "--learning_rate", "1.",
        "--report_to", "wandb",
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", "500",
        "--validation_prompt", "A photo of sks woman in a bucket",
        "--validation_epochs", "25",
        "--seed", "0",
        "--push_to_hub"
    ], check=True)

@app.local_entrypoint()
def main():
    train_lora.remote()