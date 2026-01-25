#!/usr/bin/env python3
"""
LoRA Merge + GGUF Conversion Pipeline
======================================
Merge LoRA adapters with base models and convert to GGUF format.

Run interactively (recommended):
    python merge_and_convert_gguff.py

Or with arguments:
    python merge_and_convert_gguff.py --adapter-path outputs/checkpoint-100
"""

import torch
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Directories
SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
GGUF_OUTPUT_DIR = SCRIPT_DIR / "models" / "gguf"

# Quantization options
QUANT_OPTIONS = {
    "q4_k_m": "Best balance of quality/size (recommended)",
    "q5_k_m": "Higher quality, ~20% larger",
    "q6_k": "Near-lossless, ~40% larger",
    "q8_0": "Highest quality, largest size",
    "q4_k_s": "Smaller than q4_k_m, slightly lower quality",
    "q3_k_m": "Smallest usable size, lower quality",
}

def merge_lora_adapter(base_model_name, adapter_path, output_path):
    """
    Merge LoRA adapter with base model to create a standalone model
    
    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-14B-Instruct")
        adapter_path: Path to the saved LoRA adapter
        output_path: Where to save the merged model
    """
    print("=" * 80)
    print("STEP 1: MERGING LORA ADAPTER WITH BASE MODEL")
    print("=" * 80)
    
    print(f"\nBase Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    print(f"Output Path: {output_path}")
    
    # Load base model in FP16 (no quantization for merging)
    print("\n[1/4] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("Base model loaded!")
    
    # Load tokenizer
    print("\n[2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    print("Tokenizer loaded!")
    
    # Load and merge LoRA adapter
    print("\n[3/4] Loading LoRA adapter and merging...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print("LoRA adapter merged successfully!")
    
    # Save merged model
    print("\n[4/4] Saving merged model...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Merged model saved to: {output_path}")
    
    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"Total size: {round(total_size / 1024 / 1024 / 1024, 2)} GB")
    
    return output_path


def convert_to_gguf(model_path, quantization_method="q4_k_m"):
    """
    Convert merged HuggingFace model to GGUF format
    
    Args:
        model_path: Path to the merged model
        quantization_method: GGUF quantization method
    """
    print("\n" + "=" * 80)
    print("STEP 2: CONVERTING TO GGUF FORMAT")
    print("=" * 80)
    
    ALLOWED_QUANTS = {
        "q2_k": "Uses Q4_K for attention.vw and feed_forward.w2 tensors, Q2_K for others.",
        "q3_k_l": "Uses Q5_K for attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
        "q3_k_m": "Uses Q4_K for attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
        "q3_k_s": "Uses Q3_K for all tensors",
        "q4_0": "Original quant method, 4-bit.",
        "q4_1": "Higher accuracy than q4_0 but not as high as q5_0.",
        "q4_k_m": "Uses Q6_K for half of attention.wv and feed_forward.w2 tensors, else Q4_K",
        "q4_k_s": "Uses Q4_K for all tensors",
        "q5_0": "Higher accuracy, higher resource usage and slower inference.",
        "q5_1": "Even higher accuracy, resource usage and slower inference.",
        "q5_k_m": "Uses Q6_K for half of attention.wv and feed_forward.w2 tensors, else Q5_K",
        "q5_k_s": "Uses Q5_K for all tensors",
        "q6_k": "Uses Q8_K for all tensors",
        "q8_0": "Almost indistinguishable from float16. High resource use and slow.",
    }
    
    if quantization_method not in ALLOWED_QUANTS.keys():
        error = f"Quantization method = [{quantization_method}] not supported. Choose from:\n"
        for key, value in ALLOWED_QUANTS.items():
            error += f"  [{key}] => {value}\n"
        raise ValueError(error)
    
    print(f"\nModel Path: {model_path}")
    print(f"Quantization Method: {quantization_method}")
    print(f"Description: {ALLOWED_QUANTS[quantization_method]}")
    
    # Check/clone llama.cpp
    if not os.path.exists("llama.cpp"):
        print("\n[1/5] Cloning llama.cpp repository...")
        result = os.system("git clone https://github.com/ggerganov/llama.cpp")
        if result != 0:
            raise RuntimeError("Failed to clone llama.cpp repository")
        print("llama.cpp cloned successfully!")
    else:
        print("\n[1/5] llama.cpp repository already exists")
    
    # Build llama.cpp
    quantize_bin = "llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        print("\n[2/5] Building llama.cpp with CMake (this may take several minutes)...")
        os.makedirs("llama.cpp/build", exist_ok=True)
        
        build_commands = [
            "cd llama.cpp/build",
            "cmake .. -DLLAMA_CUBLAS=ON",
            "cmake --build . --config Release -j"
        ]
        
        result = os.system(" && ".join(build_commands))
        if result != 0:
            print("WARNING: Build with CUDA failed, trying CPU-only build...")
            os.system("rm -rf llama.cpp/build")
            os.makedirs("llama.cpp/build", exist_ok=True)
            
            build_commands = [
                "cd llama.cpp/build",
                "cmake ..",
                "cmake --build . --config Release -j"
            ]
            result = os.system(" && ".join(build_commands))
            
            if result != 0:
                raise RuntimeError("Failed to build llama.cpp")
        
        print("llama.cpp built successfully!")
    else:
        print("\n[2/5] llama.cpp already built")
    
    # Install dependencies
    print("\n[3/5] Installing GGUF dependencies...")
    os.system("pip install gguf protobuf -q")
    
    # Convert to FP16 GGUF
    print("\n[4/5] Converting to FP16 GGUF format...")
    fp16_output = f"{model_path}.fp16.gguf"
    
    convert_script = (
        "llama.cpp/convert_hf_to_gguf.py" 
        if os.path.exists("llama.cpp/convert_hf_to_gguf.py") 
        else "llama.cpp/convert.py"
    )
    
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")
    
    result = os.system(
        f"python {convert_script} {model_path} "
        f"--outfile {fp16_output} --outtype f16"
    )
    
    if result != 0:
        raise RuntimeError("FP16 GGUF conversion failed!")
    
    print(f"FP16 GGUF created: {fp16_output}")
    print(f"Size: {round(os.path.getsize(fp16_output) / 1024 / 1024 / 1024, 2)} GB")
    
    # Quantize to target format
    print(f"\n[5/5] Quantizing to {quantization_method}...")
    final_output = f"{model_path}-{quantization_method}.gguf"
    
    result = os.system(
        f"./llama.cpp/build/bin/llama-quantize "
        f"{fp16_output} {final_output} {quantization_method}"
    )
    
    if result != 0:
        raise RuntimeError("Quantization failed!")
    
    # Clean up intermediate FP16 file
    if os.path.exists(fp16_output):
        os.remove(fp16_output)
        print(f"\nCleaned up intermediate file: {fp16_output}")
    
    # Final statistics
    final_size = os.path.getsize(final_output)
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"Output File: {final_output}")
    print(f"Final Size: {round(final_size / 1024 / 1024 / 1024, 2)} GB")
    print(f"Quantization: {quantization_method}")
    print("=" * 80)
    
    return final_output


# =============================================================================
# Interactive Selection Functions
# =============================================================================

def print_header():
    print()
    print("=" * 70)
    print("LORA MERGE + GGUF CONVERSION")
    print("=" * 70)
    print()


def find_adapters() -> List[Dict]:
    """Find all LoRA adapters in the outputs directory."""
    adapters = []

    if not OUTPUTS_DIR.exists():
        return adapters

    # Search for adapter_config.json files
    for adapter_config in OUTPUTS_DIR.rglob("adapter_config.json"):
        adapter_dir = adapter_config.parent

        # Load adapter config for info
        try:
            with open(adapter_config, 'r') as f:
                config = json.load(f)
        except:
            config = {}

        # Check for run metadata
        run_metadata = {}
        for meta_path in [adapter_dir / "run_metadata.json", adapter_dir.parent / "run_metadata.json"]:
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f:
                        run_metadata = json.load(f)
                    break
                except:
                    pass

        # Calculate adapter size
        adapter_size = sum(
            f.stat().st_size for f in adapter_dir.glob("*.safetensors")
        ) / (1024 * 1024)  # MB

        adapters.append({
            "path": str(adapter_dir),
            "name": adapter_dir.name,
            "base_model": config.get("base_model_name_or_path", run_metadata.get("base_model", "Unknown")),
            "r": config.get("r", "?"),
            "lora_alpha": config.get("lora_alpha", "?"),
            "size_mb": adapter_size,
            "run_metadata": run_metadata,
        })

    # Sort by modification time (newest first)
    adapters.sort(key=lambda x: Path(x["path"]).stat().st_mtime, reverse=True)

    return adapters


def select_from_list(options: List[str], prompt: str) -> int:
    """Interactive selection from a list."""
    print(f"{prompt}")
    print("-" * 50)

    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    print()
    while True:
        try:
            choice = input(f"Select [1-{len(options)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"Please enter 1-{len(options)}")
        except ValueError:
            print("Enter a number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)


def get_text_input(prompt: str, default: str = "") -> str:
    """Get text input with optional default."""
    try:
        default_str = f" [{default}]" if default else ""
        value = input(f"{prompt}{default_str}: ").strip()
        return value if value else default
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Get yes/no input."""
    default_str = "Y/n" if default else "y/N"
    try:
        value = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not value:
            return default
        return value in ('y', 'yes')
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


def interactive_mode() -> Dict:
    """Run interactive selection mode."""
    print_header()

    # Find available adapters
    adapters = find_adapters()

    if not adapters:
        print("ERROR: No LoRA adapters found in outputs/")
        print("Train a model first with: python train.py")
        sys.exit(1)

    print(f"Found {len(adapters)} adapter(s)\n")

    # Select adapter
    adapter_displays = []
    for a in adapters:
        base_short = a["base_model"].split("/")[-1] if "/" in a["base_model"] else a["base_model"]
        display = f"{a['name']} (r={a['r']}, {a['size_mb']:.1f}MB, base: {base_short})"
        adapter_displays.append(display)

    selected_idx = select_from_list(adapter_displays, "Select adapter to convert:")
    selected_adapter = adapters[selected_idx]
    print(f"  -> {selected_adapter['name']}\n")

    # Select quantization
    print()
    quant_displays = [f"{k} - {v}" for k, v in QUANT_OPTIONS.items()]
    quant_idx = select_from_list(quant_displays, "Select quantization method:")
    selected_quant = list(QUANT_OPTIONS.keys())[quant_idx]
    print(f"  -> {selected_quant}\n")

    # Output name
    print()
    default_name = selected_adapter["run_metadata"].get("run_name", selected_adapter["name"])
    output_name = get_text_input("Output name for GGUF file", default=default_name)
    print()

    # Base model (auto-detect or override)
    base_model = selected_adapter["base_model"]
    if base_model == "Unknown" or not base_model:
        base_model = get_text_input("Base model (HuggingFace name)", default="Qwen/Qwen2.5-14B-Instruct")
    else:
        print(f"Base model: {base_model}")
        if not get_yes_no("Use this base model?", default=True):
            base_model = get_text_input("Base model (HuggingFace name)")

    print()

    # Confirmation
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"  Adapter:      {selected_adapter['path']}")
    print(f"  Base Model:   {base_model}")
    print(f"  Output Name:  {output_name}")
    print(f"  Quantization: {selected_quant}")
    print(f"  Output File:  models/gguf/{output_name}-{selected_quant}.gguf")
    print("=" * 70)
    print()

    if not get_yes_no("Start conversion?", default=True):
        print("Cancelled.")
        sys.exit(0)

    return {
        "adapter_path": selected_adapter["path"],
        "base_model": base_model,
        "output_name": output_name,
        "quantization": selected_quant,
    }


def load_run_metadata(adapter_path: str) -> dict:
    """
    Load run metadata from the training output directory.

    Args:
        adapter_path: Path to the LoRA adapter (e.g., outputs/run-name/final_adapter)

    Returns:
        Run metadata dict or empty dict if not found
    """
    # Check in adapter path parent (the run directory)
    adapter_dir = Path(adapter_path)

    # Try parent directory (outputs/run-name/)
    metadata_path = adapter_dir.parent / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)

    # Try adapter directory itself
    metadata_path = adapter_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)

    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and convert to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    python merge_and_convert_gguff.py

  With arguments:
    python merge_and_convert_gguff.py --adapter-path outputs/checkpoint-100
    python merge_and_convert_gguff.py --adapter-path outputs/checkpoint-100 --quantization q5_k_m
"""
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name from HuggingFace (auto-detected from run metadata if not specified)"
    )

    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to the LoRA adapter (e.g., outputs/checkpoint-100)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output name for GGUF file (auto-generated from adapter name if not specified)"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        choices=[
            "q2_k", "q3_k_l", "q3_k_m", "q3_k_s",
            "q4_0", "q4_1", "q4_k_m", "q4_k_s",
            "q5_0", "q5_1", "q5_k_m", "q5_k_s",
            "q6_k", "q8_0"
        ],
        help="GGUF quantization method (default: q4_k_m)"
    )

    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merging step (use if model is already merged)"
    )

    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep the merged model directory (default: delete after GGUF conversion)"
    )

    args = parser.parse_args()

    # If no adapter path provided, run interactive mode
    if args.adapter_path is None:
        config = interactive_mode()
        adapter_path = config["adapter_path"]
        base_model = config["base_model"]
        output_name = config["output_name"]
        quantization = config["quantization"]
        skip_merge = False
        keep_merged = False
    else:
        # CLI mode
        adapter_path = args.adapter_path
        quantization = args.quantization
        skip_merge = args.skip_merge
        keep_merged = args.keep_merged

        # Load run metadata
        run_metadata = load_run_metadata(adapter_path)

        # Determine base model
        if args.base_model:
            base_model = args.base_model
        elif run_metadata.get('base_model'):
            base_model = run_metadata['base_model']
        else:
            base_model = "Qwen/Qwen2.5-14B-Instruct"
            print(f"WARNING: No base model specified, using default: {base_model}")

        # Determine output name
        if args.output_name:
            output_name = args.output_name
        elif run_metadata.get('run_name'):
            output_name = run_metadata['run_name']
        else:
            # Fallback: use adapter directory name
            output_name = Path(adapter_path).name

        print("=" * 80)
        print("LORA MERGE + GGUF CONVERSION PIPELINE")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Base Model: {base_model}")
        print(f"  Adapter Path: {adapter_path}")
        print(f"  Output Name: {output_name}")
        print(f"  Quantization: {quantization}")
        print(f"  Skip Merge: {skip_merge}")
        if run_metadata:
            print(f"\nRun Metadata:")
            if isinstance(run_metadata.get('run_index'), int):
                print(f"  Run Index: #{run_metadata.get('run_index', 'N/A'):03d}")
            else:
                print(f"  Run Index: {run_metadata.get('run_index', 'N/A')}")
            print(f"  Model Size: {run_metadata.get('model_size', 'N/A')}")
            print(f"  Dataset: {run_metadata.get('dataset', 'N/A')}")
        print()

    # Create temporary merged model path
    merged_path = f"_temp_merged_{output_name}"

    # Ensure output directory exists
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Merge LoRA adapter with base model
        if not skip_merge:
            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

            merged_path = merge_lora_adapter(
                base_model,
                adapter_path,
                merged_path
            )
        else:
            print("Skipping merge step...")
            if not os.path.exists(merged_path):
                raise FileNotFoundError(f"Merged model not found: {merged_path}")

        # Step 2: Convert to GGUF
        temp_gguf_path = convert_to_gguf(merged_path, quantization)

        # Step 3: Move GGUF to models/gguf/ with proper name
        final_gguf_name = f"{output_name}-{quantization}.gguf"
        final_gguf_path = GGUF_OUTPUT_DIR / final_gguf_name

        print(f"\nMoving GGUF to: {final_gguf_path}")
        shutil.move(temp_gguf_path, final_gguf_path)

        # Step 4: Clean up merged model directory (unless --keep-merged)
        if not keep_merged and os.path.exists(merged_path):
            print(f"\nCleaning up merged model directory: {merged_path}")
            shutil.rmtree(merged_path)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nGGUF Model: {final_gguf_path}")
        print(f"Size: {final_gguf_path.stat().st_size / 1024**3:.2f} GB")
        print(f"\nYou can now use this model with:")
        print(f"  streamlit run app.py")
        print(f"  ollama create {output_name} -f Modelfile")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        print(f"\n{type(e).__name__}: {str(e)}")
        print("\nPlease check the error message above and try again.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
