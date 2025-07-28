import subprocess
import modal

image = (
    modal.Image.from_registry("ostris/aitoolkit:latest")
    .apt_install("git")
    .run_commands("git clone https://github.com/ostris/ai-toolkit.git /aitoolkit")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache"})
)

secrets = [modal.Secret.from_name("huggingface-secret")]
vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

app = modal.App(name="ai-toolkit", image=image, secrets=secrets)

@app.function(max_containers=1, gpu="T4", volumes={"/cache": vol})
@modal.concurrent(max_inputs=10)
@modal.web_server(port=8675, startup_timeout=300)
def run():
    subprocess.Popen("cd /aitoolkit/ui && npm run build_and_start", shell=True)