# venv/bin/modal run comfi.py
# https://registry.comfy.org/

prompt = ""
photo = "photo.png"

def download_models():
    from model_downloader import download_and_link
    download_and_link("epiCRealism_XL")
    download_and_link("GFPGANv1.4")
    download_and_link("4x-Ultrasharp")

import modal
import subprocess
from pathlib import Path

volume = modal.Volume.from_name("my-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "aria2", "libgl1", "libglib2.0-0")
    .uv_pip_install("comfy-cli")
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools")
    .run_commands("comfy node install comfyui-impact-pack comfyui-impact-subpack")
    .add_local_file("./model_downloader.py", remote_path="/root/model_downloader.py", copy=True)
    .run_function(download_models, volumes={"/cache": volume})
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}")
)
app = modal.App(name="comfy", image=image, volumes={"/cache": volume})

# @app.function(max_containers=1, gpu="T4")
# @modal.concurrent(max_inputs=10)
# @modal.web_server(8188, startup_timeout=60)
# def ui():
#     subprocess.Popen("comfy launch -- --listen 0.0.0.0", shell=True)

@app.cls(gpu="T4")
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        import random
        import json

        seed = random.randint(0, 2**32 - 1)
        workflow_api={"4":{"inputs":{"ckpt_name":"epiCRealism_XL.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"5":{"inputs":{"text":prompt,"clip":["4",1]},"class_type":"CLIPTextEncode","_meta":{"title":"Positive"}},"6":{"inputs":{"text":"","clip":["4",1]},"class_type":"CLIPTextEncode","_meta":{"title":"Negative"}},"16":{"inputs":{"model_name":"sam_vit_b_01ec64.pth","device_mode":"AUTO"},"class_type":"SAMLoader","_meta":{"title":"SAMLoader (Impact)"}},"51":{"inputs":{"guide_size":360,"guide_size_for":True,"max_size":768,"seed":seed,"steps":20,"cfg":8,"sampler_name":"euler","scheduler":"normal","denoise":0.5,"feather":5,"noise_mask":True,"force_inpaint":False,"bbox_threshold":0.5,"bbox_dilation":15,"bbox_crop_factor":3,"sam_detection_hint":"center-1","sam_dilation":0,"sam_threshold":0.93,"sam_bbox_expansion":0,"sam_mask_hint_threshold":0.7,"sam_mask_hint_use_negative":"False","drop_size":10,"wildcard":"","cycle":1,"inpaint_model":False,"noise_mask_feather":20,"tiled_encode":False,"tiled_decode":False,"image":["62",0],"model":["4",0],"clip":["4",1],"vae":["4",2],"positive":["5",0],"negative":["6",0],"bbox_detector":["53",0],"sam_model_opt":["16",0]},"class_type":"FaceDetailer","_meta":{"title":"FaceDetailer"}},"53":{"inputs":{"model_name":"bbox/face_yolov8m.pt"},"class_type":"UltralyticsDetectorProvider","_meta":{"title":"UltralyticsDetectorProvider"}},"62":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"63":{"inputs":{"filename_prefix":"ComfyUI","images":["51",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}}}
        with open("/root/workflow_api.json", "w") as f:
            json.dump(workflow_api, f)
        subprocess.run("comfy launch --background", shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout 1200 --verbose", shell=True, check=True)
        
        for file in Path("/root/comfy/ComfyUI/output").iterdir():
            if file.suffix == ".png":
                return file.read_bytes()
        

@app.local_entrypoint()
def main():
    import datetime

    output_bytes = ComfyUI().infer.remote()
    datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(f"/Users/firozahmed/Downloads/comfy_{datetime}.png")
    output_path.write_bytes(output_bytes)
    print(f"Output saved to {output_path}")