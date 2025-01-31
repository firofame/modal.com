import modal
import json
import os

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "torchaudio", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu126")
    .pip_install("accelerate", "transformers", "Pillow")
    .add_local_dir("data", remote_path="/data", copy=True) # Changed to add_local_dir
    .pip_install("huggingface_hub[hf_transfer]")
    .env({"HF_DATASETS_TRUST_REMOTE_CODE": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

app = modal.App(name="joy-caption", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.function(image=image, gpu="L4", volumes={"/cache": vol})
def generate_caption(image_path: str, name: str, prompt_template: str) -> str:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_name = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

    processor = AutoProcessor.from_pretrained(model_name)
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    llava_model.eval()

    with torch.no_grad():
        image = Image.open(image_path)
        
        prompt = prompt_template.format(name=name)

        convo = [{"role": "system", "content": "You are a helpful image captioner."}, {"role": "user", "content": prompt}]

        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        generate_ids = llava_model.generate(**inputs, max_new_tokens=300, do_sample=True, suppress_tokens=None, use_cache=True, temperature=0.6, top_k=None, top_p=0.9)[0]

        generate_ids = generate_ids[inputs["input_ids"].shape[1] :]

        caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        caption = caption.strip()
        return caption

@app.local_entrypoint()
def main():
    local_data_dir = "data"  # Local directory containing images
    remote_data_dir = "/data" # Remote directory where images will be copied
    name = "NirmalaSitharaman"
    prompt_template = "Write a long descriptive caption for this image in a formal tone. If there is a person/character in the image you must refer to them as {name}."

    metadata_entries = []

    for filename in os.listdir(local_data_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Add other image extensions if needed
            local_image_path = os.path.join(local_data_dir, filename)
            remote_image_path = os.path.join(remote_data_dir, filename) 
            
            caption = generate_caption.remote(remote_image_path, name, prompt_template) # Use remote path

            metadata_entries.append({"file_name": filename, "prompt": caption})

    with open("metadata.jsonl", "w") as f:
        for entry in metadata_entries:
            json.dump(entry, f)
            f.write("\n")