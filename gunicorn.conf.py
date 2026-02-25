import multiprocessing
import os

# Gunicorn configuration for heavy DeepFace startup
bind = "0.0.0.0:8000"
workers = 1  # Keep it to 1 to save memory on B1 plan
timeout = 600  # 10 minute timeout for heavy model loading
loglevel = "info"
accesslog = "-"
errorlog = "-"
capture_output = True
enable_stdio_inheritance = True
