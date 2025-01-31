import modal
import os

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget")
    .pip_install("accelerate", "transformers", "diffusers[torch]", "huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_flux.py")
    .run_commands("accelerate config default")
)

app = modal.App(name="train", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="T4", volumes={"/cache": vol})
def train_lora() -> bytes:
    HF_TOKEN=os.environ["HF_TOKEN"]
    from huggingface_hub import login
    login(HF_TOKEN)

    MODEL_NAME = "black-forest-labs/FLUX.1-dev"
    INSTANCE_DIR = "dog"
    OUTPUT_DIR = "trained-flux"

    import subprocess
    subprocess.run([
        "accelerate", "launch", "train_dreambooth_lora_flux.py",
        "--pretrained_model_name_or_path", MODEL_NAME,
        "--instance_data_dir", INSTANCE_DIR,
        "--output_dir", OUTPUT_DIR,
        "--mixed_precision", "bf16",
        "--instance_prompt", "a photo of sks dog",
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
        "--validation_prompt", "A photo of sks dog in a bucket",
        "--validation_epochs", "25",
        "--seed", "0",
        "--push_to_hub"
    ], check=True)

    print(f"Model trained and saved to {OUTPUT_DIR}")
    return b"Model training complete"

@app.local_entrypoint()
def main():
    train_lora.remote()