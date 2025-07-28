import time
from io import BytesIO
from pathlib import Path

import modal

flux_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .uv_pip_install("huggingface-hub[hf-transfer]")
    .uv_pip_install("accelerate", "transformers", "git+https://github.com/huggingface/diffusers")
    .uv_pip_install("peft", "sentencepiece")
)

flux_image = flux_image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(Path("/cache")),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("flux", image=flux_image)

with flux_image.imports():
    import torch
    from diffusers import AutoPipelineForText2Image

MINUTES = 60  # seconds
secrets = [modal.Secret.from_name("huggingface-secret")]

@app.cls(
    gpu="L40S",
    secrets=secrets,
    scaledown_window=20 * MINUTES,
    timeout=60 * MINUTES,  # leave plenty of time for compilation
    volumes={  # add Volumes to store serializable compilation artifacts, see section on torch.compile below
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)

class Model:
    compile: bool = (  # see section on torch.compile below for details
        modal.parameter(default=False)
    )

    @modal.enter()
    def enter(self):
        pipe = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
        pipe.load_lora_weights("firofame/manjuw", weight_name="manjuw.safetensors")
        pipe.fuse_lora(lora_scale=1.1)
        self.pipe = pipe

    @modal.method()
    def inference(self, prompt: str) -> bytes:
        print("🎨 generating image...")
        out = self.pipe(
            prompt,
            output_type="pil",
            num_inference_steps=30,
        ).images[0]

        byte_stream = BytesIO()
        out.save(byte_stream, format="JPEG")
        return byte_stream.getvalue()

@app.local_entrypoint()
def main(
    prompt: str = "Photograph of manjuw, A cute woman holding a sign that says Flux",
):
    t0 = time.time()
    image_bytes = Model().inference.remote(prompt)
    print(f"🎨 first inference latency: {time.time() - t0:.2f} seconds")

    output_path = Path("./output.jpg")
    print(f"🎨 saving output to {output_path}")
    output_path.write_bytes(image_bytes)

