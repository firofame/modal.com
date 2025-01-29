import subprocess
from modal import (App, Image, web_server, Secret, Volume)
import os

os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = "7860"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git-all", "curl", "ffmpeg")
    .run_commands("git clone https://github.com/facefusion/facefusion /root/facefusion")
    .run_commands("cd /root/facefusion && python install.py --onnxruntime cuda --skip-conda")
)

app = App("facefusion", image=image)

@app.function(allow_concurrent_inputs=100, concurrency_limit=1, container_idle_timeout=60*20, timeout=60*60, gpu="T4")
@web_server(7860, startup_timeout=60*10)
def ui():
    subprocess.Popen("cd /root/facefusion && python facefusion.py run --open-browser", shell=True)