# venv/bin/modal run comfi.py
# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

prompt = "ugly woman"
photo = "photo.png"
width = 288
height = 512
audio = "audio.wav"
seconds = 3

import modal
from comfi_helper import install_dependencies, download_models, run_comfy

volume = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2", "libgl1", "libglib2.0-0", "ninja-build")
    .uv_pip_install("setuptools", "wheel", "ninja", "comfy-cli")
    .run_commands('TORCH_CUDA_ARCH_LIST="8.9" pip install --use-pep517 --no-build-isolation git+https://github.com/winggan/SageAttention.git@patch-1')
    .run_function(install_dependencies)
    .run_function(download_models, volumes={"/cache": volume})
    .add_local_file("/Users/firozahmed/Desktop/modal.com/comfi_helper.py", remote_path="/root/comfi_helper.py")
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")
    .add_local_file(f"/Users/firozahmed/Downloads/{audio}", remote_path=f"/root/comfy/ComfyUI/input/{audio}")
)
app = modal.App(name="comfy", image=image, volumes={"/cache": volume})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=60)
# def ui():
#     import subprocess
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(gpu="L40s", timeout=seconds*70)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.method()
    def infer(self) -> tuple[bytes, str]:
        return run_comfy("infinite_talk", prompt, photo, width, height, audio, seconds)

@app.local_entrypoint()
def main():
    import datetime
    from pathlib import Path

    output_bytes, file_suffix = ComfyUI().infer.remote()
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(f"/Users/firozahmed/Downloads/comfy_{dt_string}{file_suffix}").write_bytes(output_bytes)
    print(f"Output saved")