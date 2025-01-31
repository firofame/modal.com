import modal
import os
from io import BytesIO
from pathlib import Path

app = modal.App("flux-modal")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("sentencepiece", "peft", "transformers", "diffusers[torch]", "huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

@app.function(
    gpu="L40s",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
)
def inference(prompt: str) -> bytes:
    from diffusers import FluxPipeline
    import torch

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, token=os.environ["HF_TOKEN"])
    pipe.load_lora_weights("firofame/firoz", weight_name="firoz.safetensors")
    img = pipe(prompt=prompt, guidance_scale=3.5, height=512, width=768, num_inference_steps=50).images[0]
    byte_stream = BytesIO()
    img.save(byte_stream, format="PNG")
    return byte_stream.getvalue()

@app.local_entrypoint()
def main(prompt: str = "firofame man"):
    image_bytes = inference.remote(prompt)
    output_path = Path("./output.png")
    output_path.write_bytes(image_bytes)