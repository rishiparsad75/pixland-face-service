import multiprocessing
import os

# Gunicorn configuration for heavy DeepFace startup
bind = os.environ.get("PORT", "8080")
if ":" not in bind:
    bind = f"0.0.0.0:{bind}"

workers = 1  # Keep it to 1 to save memory on B1 plan
timeout = 600  # 10 minute timeout for heavy model loading
loglevel = "info"
accesslog = "-"
errorlog = "-"
capture_output = True
enable_stdio_inheritance = True
