import subprocess
import os
from modal import App, Image, Secret, Volume

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "git", "unzip")
    .run_commands("git clone --recursive --branch sd3-flux.1 https://github.com/bmaltais/kohya_ss /root/kohya_ss")
    .run_commands("cd /root/kohya_ss/sd-scripts && pip install --use-pep517 --upgrade -r requirements.txt")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .run_commands("python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 torchvision==0.19.0")
    .add_local_file("dataset_config.toml", "/root/dataset_config.toml", copy=True)
    .add_local_file("images.zip", "/root/images.zip", copy=True)
    .run_commands("unzip /root/images.zip -d /root/images && rm /root/images.zip")
)

image = image.pip_install("huggingface_hub[hf_transfer]").env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}).run_commands("rm -rf /root/models")

app = App("kohya_ss", image=image, secrets=[Secret.from_name("huggingface-secret")])

vol = Volume.from_name("kohya_ss-models", create_if_missing=True)

@app.function(volumes={"/root/models": vol})
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download
    local_dir = f"/root/models/{model_type}"
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

@app.local_entrypoint()
def download_models():
    models_to_download = [
        ("black-forest-labs/FLUX.1-schnell", "ae.safetensors", "flux"),
    ]
    list(hf_download.starmap(models_to_download))

@app.function(volumes={"/root/models": vol}, gpu="T4")
def run_training():
    from accelerate.utils import write_basic_config
    write_basic_config(mixed_precision='bf16')

    os.chdir("/root/kohya_ss/sd-scripts")

    command = [
        "accelerate", "launch", "flux_train_network.py",
        "--pretrained_model_name_or_path", "/root/models/flux/flux1-dev-fp8.safetensors",
        "--clip_l", "/root/models/flux/clip_l.safetensors",
        "--t5xxl", "/root/models/flux/t5xxl_fp8_e4m3fn.safetensors",
        "--ae", "/root/models/flux/ae.safetensors",
        "--cache_latents_to_disk",
        "--save_model_as", "safetensors",
        "--sdpa",
        "--persistent_data_loader_workers",
        "--max_data_loader_n_workers", "2",
        "--seed", "42",
        "--gradient_checkpointing",
        "--mixed_precision", "bf16",
        "--save_precision", "bf16",
        "--network_module", "networks.lora_flux",
        "--network_args", "train_double_block_indices=none", "train_single_block_indices=7,20", "single_mod_dim=0",
        "--cache_text_encoder_outputs",
        "--network_train_unet_only",
        "--optimizer_type", "adafactor",
        "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
        "--lr_scheduler", "constant_with_warmup",
        "--max_grad_norm", "0.0",
        "--learning_rate", "1e-4",
        "--highvram",
        "--fp8_base",
        "--max_train_epochs", "10",
        "--save_every_n_epochs", "5",
        "--dataset_config", "/root/dataset_config.toml",
        "--output_dir", "/root/models",
        "--output_name", "flux-lora-name",
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "3.1582",
        "--model_prediction_type", "raw",
        "--guidance_scale", "1.0",
    ]

    subprocess.run(command, check=True)

    subprocess.run("cd /root/models && ls", shell=True)

@app.local_entrypoint()
def main():
    run_training.remote()