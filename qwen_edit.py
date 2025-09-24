# venv/bin/modal run qwen_edit.py

prompt = "cute woman"
file_path = "/Users/firozahmed/Downloads/photo.png"

import io
import os
import pathlib
from typing import Optional

import modal

app = modal.App("qwen-image-edit")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install("torchvision", "transformers", "accelerate", "git+https://github.com/huggingface/diffusers", "huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1200,
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
)
def edit_image_remote(image_bytes: bytes) -> bytes:
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline

    DEVICE = "cuda"

    PIPE = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.float16).to(DEVICE)

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    SEED = 42
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Run the edit
    out = PIPE(
        image=[pil_image],
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=40,
        true_cfg_scale=4.0,
        num_images_per_prompt=1,
        generator=generator,
        height=None,  # keep original image size
        width=None,   # keep original image size
    ).images

    # Return the first image as PNG bytes
    buf = io.BytesIO()
    out[0].save(buf, format="PNG")
    return buf.getvalue()


@app.local_entrypoint()
def main():
    in_path = pathlib.Path(file_path).expanduser().resolve()
    image_bytes = in_path.read_bytes()

    # Call the remote GPU function
    result_bytes = edit_image_remote.remote(image_bytes=image_bytes)

    # Save result next to input file
    out_path = in_path.with_name(f"{in_path.stem}_edited.png")
    out_path.write_bytes(result_bytes)
    print(f"Done. Saved: {out_path}")