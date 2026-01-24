import torch
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Output directory for GGUF models
GGUF_OUTPUT_DIR = Path(__file__).parent / "models" / "gguf"

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
        description="Merge LoRA adapter with base model and convert to GGUF format"
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
        required=True,
        help="Path to the LoRA adapter (e.g., outputs/qwen25-14b-chemistry-20260124-001/final_adapter)"
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output name for GGUF file (auto-generated from run metadata if not specified)"
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

    # Load run metadata
    run_metadata = load_run_metadata(args.adapter_path)

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
        output_name = Path(args.adapter_path).parent.name

    # Create temporary merged model path
    merged_path = f"_temp_merged_{output_name}"

    print("=" * 80)
    print("LORA MERGE + GGUF CONVERSION PIPELINE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Base Model: {base_model}")
    print(f"  Adapter Path: {args.adapter_path}")
    print(f"  Output Name: {output_name}")
    print(f"  Quantization: {args.quantization}")
    print(f"  Skip Merge: {args.skip_merge}")
    if run_metadata:
        print(f"\nRun Metadata:")
        print(f"  Run Index: #{run_metadata.get('run_index', 'N/A'):03d}" if isinstance(run_metadata.get('run_index'), int) else f"  Run Index: {run_metadata.get('run_index', 'N/A')}")
        print(f"  Model Size: {run_metadata.get('model_size', 'N/A')}")
        print(f"  Dataset: {run_metadata.get('dataset', 'N/A')}")
    print()

    # Ensure output directory exists
    GGUF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Merge LoRA adapter with base model
        if not args.skip_merge:
            if not os.path.exists(args.adapter_path):
                raise FileNotFoundError(f"Adapter path not found: {args.adapter_path}")

            merged_path = merge_lora_adapter(
                base_model,
                args.adapter_path,
                merged_path
            )
        else:
            print("Skipping merge step...")
            if not os.path.exists(merged_path):
                raise FileNotFoundError(f"Merged model not found: {merged_path}")

        # Step 2: Convert to GGUF
        temp_gguf_path = convert_to_gguf(merged_path, args.quantization)

        # Step 3: Move GGUF to models/gguf/ with proper name
        final_gguf_name = f"{output_name}-{args.quantization}.gguf"
        final_gguf_path = GGUF_OUTPUT_DIR / final_gguf_name

        print(f"\nMoving GGUF to: {final_gguf_path}")
        shutil.move(temp_gguf_path, final_gguf_path)

        # Step 4: Clean up merged model directory (unless --keep-merged)
        if not args.keep_merged and os.path.exists(merged_path):
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
