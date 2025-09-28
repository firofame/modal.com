# venv/bin/modal serve facefusion_ui.py

import modal
import subprocess

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04", add_python="3.12")
    .apt_install("git-all", "ffmpeg", "curl")
    .uv_pip_install("huggingface-hub[hf-transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache", "GRADIO_SERVER_NAME": "0.0.0.0"})
    .run_commands("git clone https://github.com/facefusion/facefusion.git /facefusion_app")
    .workdir("/facefusion_app")
    .run_commands("python install.py --skip-conda --onnxruntime cuda")
)

app = modal.App("facefusion-ui")

minute = 60

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True)}
)
@modal.concurrent(max_inputs=10)
@modal.web_server(7860, startup_timeout=minute * 2)
def ui():
    subprocess.Popen("python facefusion.py run --download-providers huggingface", shell=True)