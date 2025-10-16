import os
import subprocess
from pathlib import Path

def get_workflow_api(prompt, photo, width, height, audio, seconds):
    import random
    seed = random.randint(0, 2**32 - 1)
    return {
        "infinite_talk":{"120":{"inputs":{"model":"Wan2_1-InfiniteTalk_Single_Q8.gguf"},"class_type":"MultiTalkModelLoader","_meta":{"title":"Multi/InfiniteTalk Model Loader"}},"122":{"inputs":{"model":"wan2.1-i2v-14b-480p-Q8_0.gguf","base_precision":"fp16_fast","quantization":"disabled","load_device":"offload_device","attention_mode":"sageattn","rms_norm_function":"default","block_swap_args":["134",0],"lora":["138",0],"multitalk_model":["120",0]},"class_type":"WanVideoModelLoader","_meta":{"title":"WanVideo Model Loader"}},"125":{"inputs":{"audio":audio,"audioUI":""},"class_type":"LoadAudio","_meta":{"title":"Load Audio"}},"128":{"inputs":{"steps":6,"cfg":1.0000000000000002,"shift":11.000000000000002,"seed":2,"force_offload":True,"scheduler":"dpm++_sde","riflex_freq_index":0,"denoise_strength":1,"batched_cfg":False,"rope_function":"comfy","start_step":0,"end_step":-1,"add_noise_to_samples":True,"model":["122",0],"image_embeds":["192",0],"text_embeds":["241",0],"multitalk_embeds":["194",0]},"class_type":"WanVideoSampler","_meta":{"title":"WanVideo Sampler"}},"129":{"inputs":{"model_name":"Wan2_1_VAE_bf16.safetensors","precision":"bf16"},"class_type":"WanVideoVAELoader","_meta":{"title":"WanVideo VAE Loader"}},"131":{"inputs":{"frame_rate":25,"loop_count":0,"filename_prefix":"ComfyUI","format":"video/h264-mp4","pix_fmt":"yuv420p","crf":19,"save_metadata":True,"trim_to_audio":True,"pingpong":False,"save_output":True,"images":["309",0],"audio":["125",0]},"class_type":"VHS_VideoCombine","_meta":{"title":"Video Combine 🎥🅥🅗🅢"}},"134":{"inputs":{"blocks_to_swap":20,"offload_img_emb":False,"offload_txt_emb":False,"use_non_blocking":True,"vace_blocks_to_swap":0,"prefetch_blocks":1,"block_swap_debug":False},"class_type":"WanVideoBlockSwap","_meta":{"title":"WanVideo Block Swap"}},"137":{"inputs":{"model":"TencentGameMate/chinese-wav2vec2-base","base_precision":"fp16","load_device":"main_device"},"class_type":"DownloadAndLoadWav2VecModel","_meta":{"title":"(Down)load Wav2Vec Model"}},"138":{"inputs":{"lora":"lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors","strength":1,"low_mem_load":False,"merge_loras":False},"class_type":"WanVideoLoraSelect","_meta":{"title":"WanVideo Lora Select"}},"177":{"inputs":{"backend":"inductor","fullgraph":False,"mode":"default","dynamic":False,"dynamo_cache_size_limit":64,"compile_transformer_blocks_only":True,"dynamo_recompile_limit":128},"class_type":"WanVideoTorchCompileSettings","_meta":{"title":"WanVideo Torch Compile Settings"}},"192":{"inputs":{"width":["291",1],"height":["291",2],"frame_window_size":81,"motion_frame":9,"force_offload":False,"colormatch":"disabled","tiled_vae":False,"mode":"infinitetalk","output_path":"","vae":["129",0],"start_image":["291",0],"clip_embeds":["237",0]},"class_type":"WanVideoImageToVideoMultiTalk","_meta":{"title":"WanVideo Long I2V Multi/InfiniteTalk"}},"194":{"inputs":{"normalize_loudness":True,"num_frames":["270",0],"fps":25,"audio_scale":1,"audio_cfg_scale":1,"multi_audio_type":"para","wav2vec_model":["300",0],"audio_1":["302",0]},"class_type":"MultiTalkWav2VecEmbeds","_meta":{"title":"Multi/InfiniteTalk Wav2vec2 Embeds"}},"237":{"inputs":{"strength_1":1,"strength_2":1,"crop":"center","combine_embeds":"average","force_offload":True,"tiles":0,"ratio":0.5,"clip_vision":["238",0],"image_1":["291",0]},"class_type":"WanVideoClipVisionEncode","_meta":{"title":"WanVideo ClipVision Encode"}},"238":{"inputs":{"clip_name":"clip_vision_h.safetensors"},"class_type":"CLIPVisionLoader","_meta":{"title":"Load CLIP Vision"}},"241":{"inputs":{"model_name":"umt5-xxl-enc-bf16.safetensors","precision":"bf16","positive_prompt":"a woman is talking","negative_prompt":"bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards","quantization":"disabled","use_disk_cache":False,"device":"gpu"},"class_type":"WanVideoTextEncodeCached","_meta":{"title":"WanVideo TextEncode Cached"}},"245":{"inputs":{"value":width},"class_type":"INTConstant","_meta":{"title":"Width"}},"246":{"inputs":{"value":height},"class_type":"INTConstant","_meta":{"title":"Height"}},"270":{"inputs":{"value":seconds*25},"class_type":"INTConstant","_meta":{"title":"Max frames"}},"281":{"inputs":{"width":["245",0],"height":["246",0],"upscale_method":"lanczos","keep_proportion":"crop","pad_color":"0, 0, 0","crop_position":"center","divisible_by":16,"device":"cpu","per_batch":0,"image":["284",0]},"class_type":"ImageResizeKJv2","_meta":{"title":"Resize Image v2"}},"284":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"291":{"inputs":{"image":["281",0]},"class_type":"GetImageSizeAndCount","_meta":{"title":"Get Image Size & Count"}},"293":{"inputs":{"preview":"","source":["194",2]},"class_type":"PreviewAny","_meta":{"title":"Preview Any"}},"300":{"inputs":{"model":"wav2vec2-chinese-base_fp16.safetensors","base_precision":"fp16","load_device":"main_device"},"class_type":"Wav2VecModelLoader","_meta":{"title":"Wav2vec2 Model Loader"}},"301":{"inputs":{"model_name":"MelBandRoformer_fp16.safetensors"},"class_type":"MelBandRoFormerModelLoader","_meta":{"title":"Mel-Band RoFormer Model Loader"}},"302":{"inputs":{"model":["301",0],"audio":["125",0]},"class_type":"MelBandRoFormerSampler","_meta":{"title":"Mel-Band RoFormer Sampler"}},"309":{"inputs":{"samples":["128",0]},"class_type":"WanVideoPassImagesFromSamples","_meta":{"title":"WanVideo Pass Images From Samples"}}},
        "qwen_edit":{"1":{"inputs":{"ckpt_name":"Qwen-Rapid-AIO-v5.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"2":{"inputs":{"seed":seed,"steps":4,"cfg":1,"sampler_name":"sa_solver","scheduler":"beta","denoise":1,"model":["1",0],"positive":["3",0],"negative":["4",0],"latent_image":["12",0]},"class_type":"KSampler","_meta":{"title":"KSampler"}},"3":{"inputs":{"prompt":prompt,"clip":["1",1],"vae":["1",2],"image1":["7",0]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus Input Prompt"}},"4":{"inputs":{"prompt":"","clip":["1",1],"vae":["1",2]},"class_type":"TextEncodeQwenImageEditPlus","_meta":{"title":"TextEncodeQwenImageEditPlus Negative (leave blank)"}},"5":{"inputs":{"samples":["2",0],"vae":["1",2]},"class_type":"VAEDecode","_meta":{"title":"VAE Decode"}},"7":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Optional Input Image"}},"10":{"inputs":{"filename_prefix":"ComfyUI","images":["5",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}},"11":{"inputs":{"upscale_method":"nearest-exact","megapixels":1,"image":["7",0]},"class_type":"ImageScaleToTotalPixels","_meta":{"title":"Scale Image to Total Pixels"}},"12":{"inputs":{"pixels":["11",0],"vae":["1",2]},"class_type":"VAEEncode","_meta":{"title":"VAE Encode"}}},
        "face_detailer":{"4":{"inputs":{"ckpt_name":"epiCRealism_XL.safetensors"},"class_type":"CheckpointLoaderSimple","_meta":{"title":"Load Checkpoint"}},"5":{"inputs":{"text":prompt,"clip":["4",1]},"class_type":"CLIPTextEncode","_meta":{"title":"Positive"}},"6":{"inputs":{"text":"","clip":["4",1]},"class_type":"CLIPTextEncode","_meta":{"title":"Negative"}},"16":{"inputs":{"model_name":"sam_vit_b_01ec64.pth","device_mode":"AUTO"},"class_type":"SAMLoader","_meta":{"title":"SAMLoader (Impact)"}},"51":{"inputs":{"guide_size":360,"guide_size_for":True,"max_size":768,"seed":seed,"steps":20,"cfg":8,"sampler_name":"euler","scheduler":"normal","denoise":0.5,"feather":5,"noise_mask":True,"force_inpaint":False,"bbox_threshold":0.5,"bbox_dilation":15,"bbox_crop_factor":3,"sam_detection_hint":"center-1","sam_dilation":0,"sam_threshold":0.93,"sam_bbox_expansion":0,"sam_mask_hint_threshold":0.7,"sam_mask_hint_use_negative":"False","drop_size":10,"wildcard":"","cycle":1,"inpaint_model":False,"noise_mask_feather":20,"tiled_encode":False,"tiled_decode":False,"image":["62",0],"model":["4",0],"clip":["4",1],"vae":["4",2],"positive":["5",0],"negative":["6",0],"bbox_detector":["53",0],"sam_model_opt":["16",0]},"class_type":"FaceDetailer","_meta":{"title":"FaceDetailer"}},"53":{"inputs":{"model_name":"bbox/face_yolov8m.pt"},"class_type":"UltralyticsDetectorProvider","_meta":{"title":"UltralyticsDetectorProvider"}},"62":{"inputs":{"image":photo},"class_type":"LoadImage","_meta":{"title":"Load Image"}},"63":{"inputs":{"filename_prefix":"ComfyUI","images":["51",0]},"class_type":"SaveImage","_meta":{"title":"Save Image"}}}
    }

civit_token = os.environ.get("civit_token")

models_list = {
        "Qwen_Image_Edit_Rapid_AIO_v5": {
            "name": "Qwen-Rapid-AIO-v5.safetensors", 
            "subdir": "checkpoints",
            "url": f"https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO/resolve/main/v5/Qwen-Rapid-AIO-NSFW-v5.safetensors"
        },
        "epiCRealism_XL": {
            "name": "epiCRealism_XL.safetensors", 
            "subdir": "checkpoints",
            "url": f"https://civitai.com/api/download/models/1920523?type=Model&format=SafeTensor&token={civit_token}"
        },
        "GFPGANv1_4": {
            "name": "GFPGANv1.4.pth",
            "subdir": "upscale_models",
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
        },
        "4x_Ultrasharp": {
            "name": "4x-UltraSharp.pth",
            "subdir": "upscale_models",
            "url": "https://civitai.com/api/download/models/125843?type=Model&format=PickleTensor&token={civit_token}"
        },
        "Wan2_1_InfiniteTalk_Single_Q8": {
            "name": "Wan2_1-InfiniteTalk_Single_Q8.gguf",
            "subdir": "unet",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf"
        },
        "wan2_1_i2v_14b_480p_Q8_0": {
            "name": "wan2.1-i2v-14b-480p-Q8_0.gguf",
            "subdir": "unet",
            "url": "https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf"
        },
        "Wan2_1_VAE_bf16": {
            "name": "Wan2_1_VAE_bf16.safetensors",
            "subdir": "vae",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors"
        },
        "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16": {
            "name": "lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors",
            "subdir": "loras",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"
        },
        "clip_vision_h": {
            "name": "clip_vision_h.safetensors",
            "subdir": "clip_vision",
            "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
        },
        "umt5_xxl_enc_bf16": {
            "name": "umt5-xxl-enc-bf16.safetensors",
            "subdir": "text_encoders",
            "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors"
        },
        "MelBandRoformer_fp16": {
            "name": "MelBandRoformer_fp16.safetensors",
            "subdir": "diffusion_models",
            "url": "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors"
        },
        "wav2vec2_chinese_base_fp16": {
            "name": "wav2vec2-chinese-base_fp16.safetensors",
            "subdir": "wav2vec2",
            "url": "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors"
        },
    }

def download_and_link(model_id):
    model_info = models_list.get(model_id)
    model_name = model_info["name"]
    model_subdir = model_info["subdir"]
    model_url = model_info["url"]

    # Define paths using pathlib for better cross-platform handling
    cache_path = Path("/cache") / model_name
    link_dir = Path("/root/comfy/ComfyUI/models") / model_subdir
    link_path = link_dir / model_name
    
    # 1. Ensure the target link directory exists
    link_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Download the file to the cache if it doesn't exist
    if not cache_path.exists():
        print(f"Downloading {model_name}...")
        try:
            subprocess.run([
                "aria2c", "-x", "8", "-c",
                "-o", str(cache_path.name),
                "-d", str(cache_path.parent),
                model_url
            ], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {model_name}: {e.stderr}")
            return # Stop processing this model if download fails
    else:
        print(f"'{model_name}' already found in cache. Skipping download.")

    # 3. Create a symbolic link if it doesn't already exist
    if not link_path.is_symlink():
        # Important: Ensure the target file actually exists before linking
        if cache_path.exists():
            print(f"Linking '{cache_path}' to '{link_path}'")
            os.symlink(cache_path, link_path)
        else:
            print(f"Cannot create link. Source file '{cache_path}' not found.")
    else:
        print(f"Link for '{model_name}' already exists. Skipping.")

def download_models():
    for model_id in models_list:
        download_and_link(model_id)