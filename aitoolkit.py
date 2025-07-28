import subprocess
import modal

PORT = 8675

image = (
    modal.Image.from_registry("ostris/aitoolkit:latest")
    .apt_install("git")
    .run_commands("git clone https://github.com/ostris/ai-toolkit.git /aitoolkit")
)

app = modal.App("aitoolkit", image=image)

@app.function(
    gpu="T4",
    cpu=2,
    memory=1024,
    timeout=3600,
    min_containers=1,  # Keep at least one instance of the server running.
)
@modal.concurrent(max_inputs=100)  # Allow 100 concurrent requests per container.
@modal.web_server(port=PORT, startup_timeout=300)
def run():
    subprocess.Popen("cd /aitoolkit/ui && npm run build_and_start", shell=True)