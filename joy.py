import modal
import json

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "torchaudio", pre=True, index_url="https://download.pytorch.org/whl/nightly/cu126")
    .pip_install("accelerate", "transformers", "Pillow")
    .add_local_file("output.png", remote_path="/output.png", copy=True)
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
    image_path = "/output.png"  # Path to your image
    file_name = "01.png" # Desired file name in metadata
    name = "NirmalaSitharaman"
    prompt_template = "Write a long descriptive caption for this image in a formal tone. If there is a person/character in the image you must refer to them as {name}."
    caption = generate_caption.remote(image_path, name, prompt_template)

    metadata = {"file_name": file_name, "prompt": caption}

    with open("metadata.jsonl", "w") as f:
        json.dump(metadata, f)