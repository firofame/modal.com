import modal
import os
from io import BytesIO
from pathlib import Path

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sentencepiece", "peft", "transformers", "diffusers[torch]", "huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App(name="flux", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(gpu="L40s", volumes={"/cache": vol})
def inference() -> bytes:
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])

    from diffusers import AutoPipelineForText2Image
    import torch

    pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    pipe.load_lora_weights("firofame/firoz", weight_name="firoz.safetensors")
    pipe.fuse_lora(lora_scale=1.1)
    pipe.to("cuda")
    prompt = "firofame man"
    img = pipe(prompt=prompt, guidance_scale=3.5, height=768, width=512, num_inference_steps=30).images[0]
    byte_stream = BytesIO()
    img.save(byte_stream, format="PNG")
    return byte_stream.getvalue()

@app.local_entrypoint()
def main():
    image_bytes = inference.remote()
    output_path = Path("./output.png")
    output_path.write_bytes(image_bytes)