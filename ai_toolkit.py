import subprocess
import modal
import os

os.environ["MODAL_ENVIRONMENT"] = "ComfyUI"
os.environ["GRADIO_SERVER_PORT"] = "8000"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"

image = modal.Image.debian_slim(python_version="3.11").apt_install("git")
image = image.run_commands("git clone https://github.com/ostris/ai-toolkit /toolkit")
image = image.run_commands("cd /toolkit && pip install -r requirements.txt")

app = modal.App(name="app", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])

@app.function(allow_concurrent_inputs=2, concurrency_limit=1, container_idle_timeout=1200, timeout=1800, gpu="T4")
@modal.web_server(8000, startup_timeout=1800)
def ui():
    subprocess.Popen("cd /toolkit && python flux_train_ui.py", shell=True)