# =============================================================================
# Qwen Fine-Tuning Lab
# =============================================================================
# Multi-stage Dockerfile for training and inference
#
# Build:   docker build -t qwen-lab .
# Train:   docker run --gpus all -it -v $(pwd)/outputs:/app/outputs qwen-lab
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.01-py3

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements_clean.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_clean.txt

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python --force-reinstall

# Copy project files
COPY . .

# Create directories for outputs
RUN mkdir -p /app/outputs /app/models/gguf /app/evaluation_results

# Default: run training in interactive mode
ENTRYPOINT ["python", "train.py"]
