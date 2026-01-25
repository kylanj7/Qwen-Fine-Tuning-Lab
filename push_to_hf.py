#!/usr/bin/env python3
"""
Upload fine-tuned models to HuggingFace Hub
Supports uploading GGUF files and/or merged HuggingFace models
"""

import os
import sys
import argparse
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from pathlib import Path


def validate_paths(args):
    """Validate that specified paths exist"""
    issues = []
    
    if args.gguf_file and not os.path.exists(args.gguf_file):
        issues.append(f"GGUF file not found: {args.gguf_file}")
    
    if args.merged_model_dir and not os.path.exists(args.merged_model_dir):
        issues.append(f"Merged model directory not found: {args.merged_model_dir}")
    
    if not args.gguf_file and not args.merged_model_dir:
        issues.append("Must specify at least --gguf-file or --merged-model-dir")
    
    return issues


def create_model_card(args):
    """Generate a README.md model card"""
    
    # Determine what's being uploaded
    uploads = []
    if args.gguf_file:
        gguf_name = os.path.basename(args.gguf_file)
        uploads.append(f"- **GGUF**: `{gguf_name}` - Quantized for efficient inference with llama.cpp")
    if args.merged_model_dir:
        uploads.append(f"- **HuggingFace Format**: Full merged model compatible with `transformers`")
    
    uploads_text = "\n".join(uploads)
    
    model_card = f"""---
language:
- en
license: apache-2.0
tags:
- qwen2.5
- fine-tuned
- lora
- chemistry
base_model: Qwen/Qwen2.5-14B-Instruct
---

# {args.repo_id.split('/')[-1]}

This model is a fine-tuned version of [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) 
using LoRA (Low-Rank Adaptation) on a chemistry dataset.

## Model Description

{args.description if args.description else "Fine-tuned Qwen2.5-14B model for chemistry domain tasks."}

## Available Formats

{uploads_text}

## Usage

### Using GGUF (with llama.cpp, Ollama, LM Studio, etc.)

```bash
# Download the GGUF file
huggingface-cli download {args.repo_id} {gguf_name if args.gguf_file else 'model.gguf'}

# Use with llama.cpp
./llama.cpp/build/bin/llama-cli -m {gguf_name if args.gguf_file else 'model.gguf'} -p "Your prompt here"
```

### Using HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{args.repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{args.repo_id}")

prompt = "What is the IUPAC name for..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

## Training Details

- **Base Model**: Qwen/Qwen2.5-14B-Instruct
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: {args.dataset if args.dataset else "Chemistry Q&A dataset"}
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Limitations

This model inherits the limitations of the base Qwen2.5-14B-Instruct model and may have 
additional domain-specific limitations due to the fine-tuning dataset.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{args.repo_id.split('/')[-1].lower().replace('-', '_')},
  author = {{{args.author if args.author else "Your Name"}}},
  title = {{{args.repo_id.split('/')[-1]}}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{args.repo_id}}}
}}
```

## License

This model is released under the Apache 2.0 license, consistent with the base Qwen model.
"""
    
    return model_card


def upload_to_hf(args):
    """Main upload function"""
    
    print("=" * 80)
    print("HUGGINGFACE MODEL UPLOAD")
    print("=" * 80)
    
    # Validate paths
    print("\n[1/5] Validating paths...")
    issues = validate_paths(args)
    if issues:
        print("\n❌ Validation errors:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    print("✓ All paths validated")
    
    # Initialize HF API
    print("\n[2/5] Initializing HuggingFace API...")
    if not args.token:
        print("⚠️  No HF token provided. Checking for HF_TOKEN environment variable...")
        args.token = os.environ.get("HF_TOKEN")
        if not args.token:
            print("\n❌ No HuggingFace token found!")
            print("\nPlease either:")
            print("  1. Pass --token YOUR_TOKEN")
            print("  2. Set HF_TOKEN environment variable")
            print("  3. Run 'huggingface-cli login' first")
            sys.exit(1)
    
    api = HfApi(token=args.token)
    print("✓ API initialized")
    
    # Create repository
    print(f"\n[3/5] Creating repository: {args.repo_id}")
    try:
        repo_url = create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"⚠️  Repository creation: {e}")
        print("Continuing with upload...")
    
    # Generate and upload model card
    print("\n[4/5] Creating model card (README.md)...")
    model_card = create_model_card(args)
    
    readme_path = "temp_README.md"
    with open(readme_path, "w") as f:
        f.write(model_card)
    
    try:
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            token=args.token,
        )
        print("✓ Model card uploaded")
    finally:
        if os.path.exists(readme_path):
            os.remove(readme_path)
    
    # Upload files
    print("\n[5/5] Uploading model files...")
    
    uploaded_files = []
    
    # Upload GGUF file
    if args.gguf_file:
        print(f"\n  Uploading GGUF: {args.gguf_file}")
        gguf_name = os.path.basename(args.gguf_file)
        file_size_gb = os.path.getsize(args.gguf_file) / (1024**3)
        print(f"  Size: {file_size_gb:.2f} GB")
        
        try:
            upload_file(
                path_or_fileobj=args.gguf_file,
                path_in_repo=gguf_name,
                repo_id=args.repo_id,
                token=args.token,
            )
            print(f"  ✓ GGUF uploaded: {gguf_name}")
            uploaded_files.append(gguf_name)
        except Exception as e:
            print(f"  ❌ Failed to upload GGUF: {e}")
    
    # Upload merged model directory
    if args.merged_model_dir:
        print(f"\n  Uploading merged model from: {args.merged_model_dir}")
        
        # Calculate total size
        total_size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, dirs, files in os.walk(args.merged_model_dir)
            for f in files
        )
        print(f"  Total size: {total_size / (1024**3):.2f} GB")
        
        try:
            upload_folder(
                folder_path=args.merged_model_dir,
                repo_id=args.repo_id,
                token=args.token,
                ignore_patterns=["*.gguf", "outputs/*", "__pycache__/*", ".git/*"]
            )
            print(f"  ✓ Merged model uploaded")
            uploaded_files.append("merged model files")
        except Exception as e:
            print(f"  ❌ Failed to upload merged model: {e}")
    
    # Success summary
    print("\n" + "=" * 80)
    print("✓ UPLOAD COMPLETE!")
    print("=" * 80)
    print(f"\nRepository: https://huggingface.co/{args.repo_id}")
    print(f"\nUploaded files:")
    for f in uploaded_files:
        print(f"  - {f}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload only GGUF file
  python upload_to_huggingface.py \\
      --repo-id username/qwen-chemistry-q4 \\
      --gguf-file qwen_chemistry_merged-q4_k_m.gguf \\
      --token YOUR_HF_TOKEN

  # Upload merged model directory
  python upload_to_huggingface.py \\
      --repo-id username/qwen-chemistry \\
      --merged-model-dir qwen_chemistry_merged \\
      --token YOUR_HF_TOKEN

  # Upload both GGUF and merged model
  python upload_to_huggingface.py \\
      --repo-id username/qwen-chemistry-full \\
      --gguf-file qwen_chemistry_merged-q4_k_m.gguf \\
      --merged-model-dir qwen_chemistry_merged \\
      --private \\
      --token YOUR_HF_TOKEN

  # Use environment variable for token
  export HF_TOKEN=your_token_here
  python upload_to_huggingface.py \\
      --repo-id username/qwen-chemistry \\
      --gguf-file qwen_chemistry_merged-q4_k_m.gguf
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (format: username/repo-name)"
    )
    
    # Model files
    parser.add_argument(
        "--gguf-file",
        type=str,
        help="Path to GGUF file to upload"
    )
    
    parser.add_argument(
        "--merged-model-dir",
        type=str,
        help="Path to merged model directory (HuggingFace format)"
    )
    
    # Authentication
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token (or set HF_TOKEN env variable)"
    )
    
    # Repository options
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    
    # Model card metadata
    parser.add_argument(
        "--description",
        type=str,
        help="Model description for the model card"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="camel-ai/chemistry",
        help="Dataset used for training (default: camel-ai/chemistry)"
    )
    
    parser.add_argument(
        "--author",
        type=str,
        help="Author name for citation"
    )
    
    args = parser.parse_args()
    
    # Run upload
    try:
        upload_to_hf(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"\n{type(e).__name__}: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
