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
    from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
    import torchvision.transforms.functional as TVF

    model_name = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"
    llava_model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    llava_model.tie_weights()

    llava_model = llava_model.to("cuda")

    assert isinstance(llava_model, LlavaForConditionalGeneration)

    processor = AutoProcessor.from_pretrained(model_name, use_fast=True) # Use fast processor
    llava_model.eval()

    with torch.no_grad():
        image = Image.open(image_path)

        prompt = prompt_template.format(name=name)

        convo = [{"role": "system", "content": "You are a helpful image captioner."}, {"role": "user", "content": prompt}]

        convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to("cuda")

        vision_device = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.device
        language_device = llava_model.language_model.get_input_embeddings().weight.device

        # Move to GPU
        pixel_values = inputs["pixel_values"].to(vision_device, non_blocking=True)
        input_ids = inputs["input_ids"].to(language_device, non_blocking=True)
        attention_mask = inputs["attention_mask"].to(language_device, non_blocking=True)

        inputs["pixel_values"] = pixel_values
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

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

    with open("data/metadata.jsonl", "w") as f:
        for entry in metadata_entries:
            json.dump(entry, f)
            f.write("\n")