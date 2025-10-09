# venv/bin/modal run comfi_infinitetalk.py
# https://registry.comfy.org/

prompt = "a woman is singing"
photo = "photo.png"
audio = "audio.m4a"
seconds = 3
gpu = "L40s"

from pathlib import Path
import subprocess
import json
import modal

image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel")
    .run_commands("apt update")
    .apt_install("git", "ninja-build", "libgl1", "libglib2.0-0")
    .uv_pip_install("setuptools", "wheel", "ninja")
    .run_commands('TORCH_CUDA_ARCH_LIST="8.9" pip install --use-pep517 --no-build-isolation git+https://github.com/winggan/SageAttention.git@patch-1')
    .uv_pip_install("huggingface-hub[hf-transfer]", "comfy-cli")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
    .run_commands("comfy --skip-prompt install --version latest --nvidia --skip-torch-or-directml")
    .run_commands("comfy node install ComfyUI-Crystools ComfyUI-WanVideoWrapper comfyui-kjnodes comfyui-videohelpersuite ComfyUI-MelBandRoFormer")
)

def download_models():
    import os
    from huggingface_hub import hf_hub_download, snapshot_download

    wav2vec_repo = "TencentGameMate/chinese-wav2vec2-base"
    wav2vec_local_path = snapshot_download(repo_id=wav2vec_repo, cache_dir="/cache")
    subprocess.run(f"mkdir -p /root/comfy/ComfyUI/models/transformers/TencentGameMate && ln -s {wav2vec_local_path} /root/comfy/ComfyUI/models/transformers/{wav2vec_repo}", shell=True, check=True)
    
    lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16 = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16} /root/comfy/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", shell=True, check=True)

    MelBandRoformer_fp16 = hf_hub_download(repo_id="Kijai/MelBandRoFormer_comfy", filename="MelBandRoformer_fp16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {MelBandRoformer_fp16} /root/comfy/ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors", shell=True, check=True)
    
    umt5_xxl_enc_bf16 = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="umt5-xxl-enc-bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {umt5_xxl_enc_bf16} /root/comfy/ComfyUI/models/text_encoders/umt5-xxl-enc-bf16.safetensors", shell=True, check=True)

    clip_vision_h = hf_hub_download(repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged", filename="split_files/clip_vision/clip_vision_h.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {clip_vision_h} /root/comfy/ComfyUI/models/clip_vision/clip_vision_h.safetensors", shell=True, check=True)
    
    Wan2_1_VAE_bf16 = hf_hub_download(repo_id="Kijai/WanVideo_comfy", filename="Wan2_1_VAE_bf16.safetensors", cache_dir="/cache")
    subprocess.run(f"ln -s {Wan2_1_VAE_bf16} /root/comfy/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors", shell=True, check=True)

    wan2_1_i2v_14b_480p_Q8_0 = hf_hub_download(repo_id="city96/Wan2.1-I2V-14B-480P-gguf", filename="wan2.1-i2v-14b-480p-Q8_0.gguf", cache_dir="/cache")
    subprocess.run(f"ln -s {wan2_1_i2v_14b_480p_Q8_0} /root/comfy/ComfyUI/models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf", shell=True, check=True)

    Wan2_1_InfiniteTalk_Single_Q8 = hf_hub_download(repo_id="Kijai/WanVideo_comfy_GGUF", filename="InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf", cache_dir="/cache")
    subprocess.run(f"ln -s '{Wan2_1_InfiniteTalk_Single_Q8}' '/root/comfy/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf'", shell=True, check=True)

volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = image.run_function(download_models, volumes={"/cache": volume}, secrets=[modal.Secret.from_name("huggingface-secret")]) \
    .add_local_file(f"/Users/firozahmed/Downloads/{photo}", remote_path=f"/root/comfy/ComfyUI/input/{photo}") \
    .add_local_file(f"/Users/firozahmed/Downloads/{audio}", remote_path=f"/root/comfy/ComfyUI/input/{audio}")

app = modal.App(name="comfi-infinitetalk", image=image, volumes={"/cache": volume})

@app.cls(gpu=gpu, timeout=seconds*60)
@modal.concurrent(max_inputs=5)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        true=True
        false=False
        workflow_api={"120":{"inputs":{"model":"Wan2_1-InfiniteTalk_Single_Q8.gguf"},"class_type":"MultiTalkModelLoader","_meta":{"title":"Multi/InfiniteTalk Model Loader"}},"122":{"inputs":{"model":"wan2.1-i2v-14b-480p-Q8_0.gguf","base_precision":"fp16_fast","quantization":"disabled","load_device":"offload_device","attention_mode":"sageattn","block_swap_args":["134",0],"lora":["138",0],"multitalk_model":["120",0]},"class_type":"WanVideoModelLoader","_meta":{"title":"WanVideo Model Loader"}},"125":{"inputs":{"audio":audio,"audioUI":""},"class_type":"LoadAudio","_meta":{"title":"LoadAudio"}},"128":{"inputs":{"steps":6,"cfg":1.0000000000000002,"shift":11.000000000000002,"seed":2,"force_offload":true,"scheduler":"dpm++_sde","riflex_freq_index":0,"denoise_strength":1,"batched_cfg":false,"rope_function":"comfy","start_step":0,"end_step":-1,"add_noise_to_samples":true,"model":["122",0],"image_embeds":["192",0],"text_embeds":["241",0],"multitalk_embeds":["194",0]},"class_type":"WanVideoSampler","_meta":{"title":"WanVideo Sampler"}},"129":{"inputs":{"model_name":"Wan2_1_VAE_bf16.safetensors","precision":"bf16"},"class_type":"WanVideoVAELoader","_meta":{"title":"WanVideo VAE Loader"}},"131":{"inputs":{"frame_rate":25,"loop_count":0,"filename_prefix":"WanVideo2_1_InfiniteTalk","format":"video/h264-mp4","pix_fmt":"yuv420p","crf":19,"save_metadata":true,"trim_to_audio":false,"pingpong":false,"save_output":true,"images":["309",0],"audio":["125",0]},"class_type":"VHS_VideoCombine","_meta":{"title":"Video Combine 🎥🅥🅗🅢"}},"134":{"inputs":{"blocks_to_swap":20,"offload_img_emb":false,"offload_txt_emb":false,"use_non_blocking":true,"vace_blocks_to_swap":0,"prefetch_blocks":1,"block_swap_debug":false},"class_type":"WanVideoBlockSwap","_meta":{"title":"WanVideo Block Swap"}},"137":{"inputs":{"model":"TencentGameMate/chinese-wav2vec2-base","base_precision":"fp16","load_device":"main_device"},"class_type":"DownloadAndLoadWav2VecModel","_meta":{"title":"(Down)load Wav2Vec Model"}},"138":{"inputs":{"lora":"lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors","strength":1,"low_mem_load":false,"merge_loras":false},"class_type":"WanVideoLoraSelect","_meta":{"title":"WanVideo Lora Select"}},"177":{"inputs":{"backend":"inductor","fullgraph":false,"mode":"default","dynamic":false,"dynamo_cache_size_limit":64,"compile_transformer_blocks_only":true,"dynamo_recompile_limit":128},"class_type":"WanVideoTorchCompileSettings","_meta":{"title":"WanVideo Torch Compile Settings"}},"192":{"inputs":{"width":["291",1],"height":["291",2],"frame_window_size":81,"motion_frame":9,"force_offload":false,"colormatch":"disabled","tiled_vae":false,"mode":"infinitetalk","output_path":"","vae":["129",0],"start_image":["291",0],"clip_embeds":["237",0]},"class_type":"WanVideoImageToVideoMultiTalk","_meta":{"title":"WanVideo Long I2V Multi/InfiniteTalk"}},"194":{"inputs":{"normalize_loudness":true,"num_frames":["270",0],"fps":25,"audio_scale":1,"audio_cfg_scale":1,"multi_audio_type":"para","wav2vec_model":["137",0],"audio_1":["302",0]},"class_type":"MultiTalkWav2VecEmbeds","_meta":{"title":"Multi/InfiniteTalk Wav2vec2 Embeds"}},"237":{"inputs":{"strength_1":1,"strength_2":1,"crop":"center","combine_embeds":"average","force_offload":true,"tiles":0,"ratio":0.5,"clip_vision":["238",0],"image_1":["291",0]},"class_type":"WanVideoClipVisionEncode","_meta":{"title":"WanVideo ClipVision Encode"}},"238":{"inputs":{"clip_name":"clip_vision_h.safetensors"},"class_type":"CLIPVisionLoader","_meta":{"title":"Load CLIP Vision"}},"241":{"inputs":{"model_name":"umt5-xxl-enc-bf16.safetensors","precision":"bf16","positive_prompt":prompt,"negative_prompt":"bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards","quantization":"disabled","use_disk_cache":false,"device":"gpu"},"class_type":"WanVideoTextEncodeCached","_meta":{"title":"WanVideo TextEncode Cached"}},"245":{"inputs":{"value":640},"class_type":"INTConstant","_meta":{"title":"Width"}},"246":{"inputs":{"value":640},"class_type":"INTConstant","_meta":{"title":"Height"}},"270":{"inputs":{"value":25*seconds},"class_type":"INTConstant","_meta":{"title":"Max frames"}},"281":{"inputs":{"width":["245",0],"height":["246",0],"upscale_method":"lanczos","keep_proportion":"crop","pad_color":"0, 0, 0","crop_position":"center","divisible_by":16,"device":"cpu","image":["284",0]},"class_type":"ImageResizeKJv2","_meta":{"title":"Resize Image v2"}},"284":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"291":{"inputs":{"image":["281",0]},"class_type":"GetImageSizeAndCount","_meta":{"title":"Get Image Size & Count"}},"293":{"inputs":{"preview":"","source":["194",2]},"class_type":"PreviewAny","_meta":{"title":"Preview Any"}},"300":{"inputs":{"model":"wav2vec2-chinese-base_fp16.safetensors","base_precision":"fp16","load_device":"main_device"},"class_type":"Wav2VecModelLoader","_meta":{"title":"Wav2vec2 Model Loader"}},"301":{"inputs":{"model_name":"MelBandRoformer_fp16.safetensors"},"class_type":"MelBandRoFormerModelLoader","_meta":{"title":"Mel-Band RoFormer Model Loader"}},"302":{"inputs":{"model":["301",0],"audio":["125",0]},"class_type":"MelBandRoFormerSampler","_meta":{"title":"Mel-Band RoFormer Sampler"}},"309":{"inputs":{"samples":["128",0]},"class_type":"WanVideoPassImagesFromSamples","_meta":{"title":"WanVideo Pass Images From Samples"}}}
        with open("/root/workflow_api.json", "w") as f:
            json.dump(workflow_api, f)
        subprocess.run("comfy launch --background -- --port 8000", shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        subprocess.run(f"comfy run --workflow {workflow_path} --wait --timeout {20*60} --verbose", shell=True, check=True)
        
        workflow = json.loads(Path(workflow_path).read_text())
        
        # 1. Find the filename_prefix from the workflow
        filename_prefix = None
        for node in workflow.values():
            if node.get("class_type") == "VHS_VideoCombine":  # 修改这里：从 "SaveVideo" 改为 "VHS_VideoCombine"
                filename_prefix = node["inputs"].get("filename_prefix")
                break
        
        if not filename_prefix:
            raise ValueError("Could not find 'filename_prefix' in a VHS_VideoCombine node in the workflow.")  # 修改错误信息

        # The prefix can be a path like "video/ComfyUI", we need the base name "ComfyUI"
        file_basename = Path(filename_prefix).name
        
        # 2. Find the output file using the extracted prefix
        output_dir = Path("/root/comfy/ComfyUI/output")
        valid_exts = {".mp4", ".webm", ".gif", ".png", ".jpg", ".jpeg"}

        candidates = [
            p for p in output_dir.rglob(f"{file_basename}*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ]
        
        if not candidates:
            # Optional: dump tree for debugging
            subprocess.run(f"ls -R {output_dir}", shell=True)
            raise FileNotFoundError(
                f"No output file found with prefix '{file_basename}'. "
                f"Make sure VHS_VideoCombine.save_output=true and "
                f"that files are written to {output_dir}."
            )

        latest_file = max(candidates, key=lambda p: p.stat().st_mtime)
        
        # 3. Return both filename and bytes
        return {
            "filename": latest_file.name,
            "bytes": latest_file.read_bytes()
        }

@app.local_entrypoint()
def main():
    # --- MODIFIED SECTION ---
    result = ComfyUI().infer.remote()
    
    output_filename = result["filename"]
    output_bytes = result["bytes"]
    
    # Use the original filename from the remote execution
    output_path = Path(f"/Users/firozahmed/Downloads/{output_filename}")
    output_path.write_bytes(output_bytes)
    
    print(f"Video saved to {output_path}")