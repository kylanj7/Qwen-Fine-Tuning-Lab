#!/usr/bin/env python3
"""
Convert base HuggingFace model to GGUF format (no LoRA merging).
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
CACHE_DIR = Path(__file__).parent / "models" / "base_cache"
GGUF_OUTPUT_DIR = Path(__file__).parent / "models" / "gguf"
QUANTIZATION = "q4_k_m"

def main():
    print("=" * 70)
    print("BASE MODEL -> GGUF CONVERSION")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print(f"Quantization: {QUANTIZATION}")
    print()

    # Ensure directories exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download model if needed
    print("[1/4] Downloading/caching base model...")
    model_path = snapshot_download(
        MODEL_ID,
        local_dir=CACHE_DIR / "Qwen2.5-14B-Instruct",
        local_dir_use_symlinks=False,
    )
    print(f"Model cached at: {model_path}")

    # Check llama.cpp
    if not os.path.exists("llama.cpp"):
        print("\n[2/4] Cloning llama.cpp...")
        os.system("git clone https://github.com/ggerganov/llama.cpp")
    else:
        print("\n[2/4] llama.cpp already exists")

    # Build if needed
    quantize_bin = "llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print("\n[3/4] Building llama.cpp...")
        os.makedirs("llama.cpp/build", exist_ok=True)
        result = os.system("cd llama.cpp/build && cmake .. -DLLAMA_CUBLAS=ON && cmake --build . --config Release -j")
        if result != 0:
            print("CUDA build failed, trying CPU...")
            os.system("rm -rf llama.cpp/build && mkdir -p llama.cpp/build")
            os.system("cd llama.cpp/build && cmake .. && cmake --build . --config Release -j")
    else:
        print("\n[3/4] llama.cpp already built")

    # Convert to GGUF
    print("\n[4/4] Converting to GGUF...")
    
    convert_script = "llama.cpp/convert_hf_to_gguf.py"
    if not os.path.exists(convert_script):
        convert_script = "llama.cpp/convert.py"

    fp16_output = str(GGUF_OUTPUT_DIR / "qwen2.5-14b-instruct-base.fp16.gguf")
    final_output = str(GGUF_OUTPUT_DIR / f"qwen2.5-14b-instruct-base-{QUANTIZATION}.gguf")

    # Convert to FP16 first
    print("Converting to FP16 GGUF...")
    result = os.system(f"python {convert_script} {model_path} --outfile {fp16_output} --outtype f16")
    if result != 0:
        print("ERROR: FP16 conversion failed")
        sys.exit(1)

    # Quantize
    print(f"\nQuantizing to {QUANTIZATION}...")
    result = os.system(f"./llama.cpp/build/bin/llama-quantize {fp16_output} {final_output} {QUANTIZATION}")
    if result != 0:
        print("ERROR: Quantization failed")
        sys.exit(1)

    # Cleanup FP16
    if os.path.exists(fp16_output):
        os.remove(fp16_output)

    # Done
    size_gb = os.path.getsize(final_output) / (1024**3)
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Output: {final_output}")
    print(f"Size: {size_gb:.2f} GB")
    print("=" * 70)

if __name__ == "__main__":
    main()
