import modal
import subprocess
import os

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "git")
    .pip_install("torch", "torchvision", "torchaudio", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu126")
    .run_commands("wget -P /root https://raw.githubusercontent.com/fpgaminer/joycaption/main/scripts/batch-caption.py")
    .add_local_dir("data", remote_path="/data", copy=True)
    .pip_install("accelerate", "transformers", "huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App(name="train", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="L4", volumes={"/cache": vol}, timeout=60*30)
def run_caption():

    os.chdir("/root")
    # subprocess.run("ls ../data", shell=True)
    subprocess.run([
        "python", 
        "batch-caption.py", 
        "--glob",
        "../data/*.png",  # Changed the glob pattern to match only .png files
        "--prompt", 
        "Write a descriptive caption for this image in a formal tone."
    ], shell=False, check=True) 

@app.local_entrypoint()
def main():
    run_caption.remote()