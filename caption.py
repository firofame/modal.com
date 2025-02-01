import modal
import subprocess
import os
import json

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

@app.function(gpu="L4", volumes={"/cache": vol})
def run_caption():
    name = "manjuw"

    os.chdir("/root")
    subprocess.run([
        "python", "batch-caption.py",
        "--glob", "../data/*.png",
        "--batch-size", "3",
        "--prompt", f"Write a descriptive caption for this image in a formal tone. If there is a person/character in the image you must refer to them as {name}."
    ], shell=False, check=True)

    metadata = []
    for filename in os.listdir("/data"):
        if filename.endswith(".txt"):
            image_filename = filename.replace(".txt", ".png")
            with open(os.path.join("/data", filename), "r") as f:
                prompt = f.read().strip()
            metadata.append({"file_name": image_filename, "prompt": prompt})

    return metadata

@app.local_entrypoint()
def main():
    metadata = run_caption.remote()
    with open("data/metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")