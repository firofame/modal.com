from io import BytesIO
from pathlib import Path
import random

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]")
    .uv_pip_install("accelerate", "transformers", "git+https://github.com/huggingface/diffusers")
    .uv_pip_install("sentencepiece")
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("huggingface-secret")]


image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)


app = modal.App("image-to-image")

with image.imports():
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    import numpy as np

@app.cls(
    image=image, gpu="L40S", volumes=volumes, secrets=secrets, scaledown_window=240
)
class Model:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        MAX_SEED = np.iinfo(np.int32).max
        self.seed = random.randint(0, MAX_SEED)
        self.device = "cuda"

        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=CACHE_DIR,
        ).to(self.device)

    @modal.method()
    def inference(
        self,
        image_bytes: bytes,
    ) -> bytes:
        init_image = load_image(Image.open(BytesIO(image_bytes)))

        image = self.pipe(
            image=init_image,
            prompt="Enhance the image of the man to appear as a professional headshot. He should be wearing a black suit.",
            guidance_scale=2.5,
            num_inference_steps=30,
            output_type="pil",
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        ).images[0]

        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes

@app.local_entrypoint()
def main():
    input_image_bytes = Path("/Users/firozqburst/Downloads/firoz.jpg").read_bytes()
    output_image_bytes = Model().inference.remote(input_image_bytes)
    Path("./output.png").write_bytes(output_image_bytes)