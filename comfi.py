# venv/bin/modal run comfi.py
# venv/bin/modal serve comfi.py
# https://registry.comfy.org/

prompt = "cute woman"
photo = "photo.png"
width = 288
height = 512
audio = "audio.wav"
seconds = 3

import modal
import subprocess
from pathlib import Path
from comfi_helper import download_models, launch_comfy_background, run_workflow

volume = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2", "libgl1", "libglib2.0-0", "ninja-build")
    .uv_pip_install("setuptools", "wheel", "ninja")
    .run_commands('TORCH_CUDA_ARCH_LIST="8.9" pip install --use-pep517 --no-build-isolation git+https://github.com/winggan/SageAttention.git@patch-1')
    .uv_pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
    .run_commands("comfy node install comfyui-impact-pack comfyui-impact-subpack") # face_detailer
    .run_commands("comfy node install ComfyUI-WanVideoWrapper comfyui-kjnodes comfyui-videohelpersuite ComfyUI-MelBandRoFormer") # infinite_talk
    .run_function(download_models, volumes={"/cache": volume})
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")
    .add_local_file("/Users/firozahmed/Desktop/modal.com/comfi_helper.py", remote_path="/root/comfi_helper.py")
    .add_local_file(f"/Users/firozahmed/Downloads/{audio}", remote_path=f"/root/comfy/ComfyUI/input/{audio}")
)
app = modal.App(name="comfy", image=image, volumes={"/cache": volume})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=60)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(gpu="L40s")
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        launch_comfy_background("face_detailer", prompt, photo, width, height, audio, seconds)
    @modal.method()
    def infer(self) -> tuple[bytes, str]:
        return run_workflow()

@app.local_entrypoint()
def main():
    import datetime

    output_bytes, file_suffix = ComfyUI().infer.remote()
    dt_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    Path(f"/Users/firozahmed/Downloads/comfy_{dt_string}{file_suffix}").write_bytes(output_bytes)
    print(f"Output saved")