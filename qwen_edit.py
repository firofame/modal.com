# venv/bin/modal run qwen_edit.py

prompt = "cute muslim woman"
file_path = "/Users/firozahmed/Downloads/photo.png"

import io
import pathlib

import modal

app = modal.App("qwen-image-edit")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git")
    .uv_pip_install(
        "torchvision",
        "transformers",
        "accelerate",
        "git+https://github.com/huggingface/diffusers",
        "huggingface-hub[hf-transfer]",
        "kernels",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache", "PYTHONPATH": "/root"})
    .add_local_dir("qwenimage", remote_path="/root/qwenimage")
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=1200,
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)},
)
def edit_image_remote(image_bytes: bytes) -> bytes:
    import math
    import torch
    from PIL import Image
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
    from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
    from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

    DEVICE = "cuda"
    ONE_MEGAPIXEL = 1024 * 1024

    pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16).to(DEVICE)
    pipe.transformer.__class__ = QwenImageTransformer2DModel
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = pil_image.size
    if w * h > ONE_MEGAPIXEL:
        print(f"Image is larger than 1MP. Resizing from {w}x{h}...")
        aspect_ratio = w / h
        new_h = int(math.sqrt(ONE_MEGAPIXEL / aspect_ratio))
        new_w = int(new_h * aspect_ratio)
        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"Resized to {new_w}x{new_h}")
    SEED = torch.randint(0, 100000, (1,)).item()  # Generate a random seed
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)

    # Run the edit
    out = pipe(
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
    out_path = in_path.with_name(f"{in_path.stem}_edited_{prompt}.png")
    out_path.write_bytes(result_bytes)
    print(f"Done. Saved: {out_path}")