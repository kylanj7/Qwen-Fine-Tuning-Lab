#!/usr/bin/env python3
"""
Unified Qwen Model Fine-Tuning System

A configuration-driven training pipeline that supports multiple models and datasets.
Select model, dataset, and training configuration interactively or via command line.

Usage:
    python train.py                          # Interactive mode
    python train.py --model qwen2.5-14b-instruct --dataset quantum --training default
"""

import os
import sys
import yaml
import json
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import wandb


# =============================================================================
# Configuration Discovery and Loading
# =============================================================================

def discover_configs(config_type: str) -> Dict[str, Path]:
    """
    Discover available config files of a given type.

    Args:
        config_type: One of 'models', 'datasets', or 'training'

    Returns:
        Dict mapping config name (without extension) to full path
    """
    config_dir = Path(__file__).parent / "configs" / config_type
    configs = {}

    if config_dir.exists():
        for yaml_file in sorted(config_dir.glob("*.yaml")):
            name = yaml_file.stem  # filename without extension
            configs[name] = yaml_file

    return configs


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file

    Returns:
        Parsed configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# Training Log and Naming
# =============================================================================

TRAINING_LOG_FILE = Path(__file__).parent / "training_log.md"
TRAINING_INDEX_FILE = Path(__file__).parent / ".training_index.json"


def get_next_training_index() -> int:
    """Get the next training index and increment the counter."""
    if TRAINING_INDEX_FILE.exists():
        with open(TRAINING_INDEX_FILE, 'r') as f:
            data = json.load(f)
            index = data.get('next_index', 1)
    else:
        index = 1

    # Save incremented index
    with open(TRAINING_INDEX_FILE, 'w') as f:
        json.dump({'next_index': index + 1}, f)

    return index


def generate_run_name(model_config: Dict, dataset_config: Dict) -> Tuple[str, int, str]:
    """
    Generate a unique run name with the format:
    {modelname}-{size}-{dataset}-{timestamp}-{index}

    Returns:
        Tuple of (run_name, index, timestamp)
    """
    # Extract components
    model_name = model_config.get('name', 'model').lower().replace(' ', '-').replace('.', '')
    model_size = model_config.get('size', '').lower()
    dataset_name = dataset_config.get('name', 'dataset').lower().replace(' ', '-')

    # Generate timestamp and get index
    timestamp = datetime.now().strftime("%Y%m%d")
    index = get_next_training_index()

    # Build run name
    run_name = f"{model_name}-{model_size}-{dataset_name}-{timestamp}-{index:03d}"

    return run_name, index, timestamp


def log_training_run(
    run_name: str,
    index: int,
    model_config: Dict,
    dataset_config: Dict,
    training_config: Dict,
    output_dir: Path,
    trainer_stats: Any = None,
) -> None:
    """
    Log training run parameters to training_log.md

    Args:
        run_name: The generated run name
        index: Training run index
        model_config: Model configuration
        dataset_config: Dataset configuration
        training_config: Training configuration
        output_dir: Output directory path
        trainer_stats: Optional training statistics from trainer
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create header if file doesn't exist
    if not TRAINING_LOG_FILE.exists():
        header = """# Training Log

This file tracks all fine-tuning runs with their configurations and results.

---

"""
        with open(TRAINING_LOG_FILE, 'w') as f:
            f.write(header)

    # Build log entry
    entry = f"""
## Run #{index:03d}: {run_name}

**Date:** {timestamp}

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Name | {model_config.get('name', 'N/A')} |
| HuggingFace ID | `{model_config.get('model_name', 'N/A')}` |
| Size | {model_config.get('size', 'N/A')} |
| Max Seq Length | {model_config.get('max_seq_length', 'N/A')} |
| LoRA Rank (r) | {model_config.get('lora', {}).get('r', 'N/A')} |
| LoRA Alpha | {model_config.get('lora', {}).get('lora_alpha', 'N/A')} |
| Quantization | {model_config.get('quantization', {}).get('bnb_4bit_quant_type', 'N/A')} |

### Dataset Configuration
| Parameter | Value |
|-----------|-------|
| Name | {dataset_config.get('name', 'N/A')} |
| HuggingFace ID | `{dataset_config.get('dataset_name', 'N/A')}` |
| Domain | {dataset_config.get('domain', 'N/A')} |
| Train/Val/Test Split | {dataset_config.get('train_val_test_split', {}).get('train', 'N/A')} / {dataset_config.get('train_val_test_split', {}).get('val', 'N/A')} / {dataset_config.get('train_val_test_split', {}).get('test', 'N/A')} |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | {training_config.get('per_device_train_batch_size', 'N/A')} |
| Gradient Accumulation | {training_config.get('gradient_accumulation_steps', 'N/A')} |
| Effective Batch Size | {training_config.get('per_device_train_batch_size', 1) * training_config.get('gradient_accumulation_steps', 1)} |
| Learning Rate | {training_config.get('learning_rate', 'N/A')} |
| LR Scheduler | {training_config.get('lr_scheduler_type', 'N/A')} |
| Warmup Steps | {training_config.get('warmup_steps', 'N/A')} |
| Epochs | {training_config.get('num_train_epochs', 'N/A')} |
| Max Steps | {training_config.get('max_steps', 'N/A')} |
| Optimizer | {training_config.get('optim', 'N/A')} |
| Weight Decay | {training_config.get('weight_decay', 'N/A')} |

### Output
| Item | Path |
|------|------|
| Output Directory | `{output_dir}` |
| LoRA Adapter | `{output_dir}/final_adapter` |
"""

    # Add training stats if available
    if trainer_stats:
        entry += f"""
### Training Results
| Metric | Value |
|--------|-------|
| Total Steps | {getattr(trainer_stats, 'global_step', 'N/A')} |
| Training Loss | {getattr(trainer_stats, 'training_loss', 'N/A'):.4f if hasattr(trainer_stats, 'training_loss') else 'N/A'} |
| Training Runtime | {getattr(trainer_stats, 'metrics', {}).get('train_runtime', 'N/A')} |
"""

    entry += """
---
"""

    # Append to log file
    with open(TRAINING_LOG_FILE, 'a') as f:
        f.write(entry)

    print(f"Training run logged to: {TRAINING_LOG_FILE}")


# =============================================================================
# Interactive Selection
# =============================================================================

def print_header():
    """Print the application header."""
    print("=" * 80)
    print("QWEN MODEL FINE-TUNING SYSTEM")
    print("=" * 80)
    print()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {gpu_mem:.1f} GB")
    else:
        print("WARNING: No CUDA device detected!")
    print()


def interactive_select(options: Dict[str, Any], prompt: str, config_type: str) -> Tuple[str, Any]:
    """
    Display options and get user selection.

    Args:
        options: Dict of config name -> config path or data
        prompt: The prompt to display
        config_type: Type of config for display purposes

    Returns:
        Tuple of (selected name, loaded config dict)
    """
    print(f"{prompt}")
    print("-" * 40)

    option_list = list(options.items())

    for i, (name, path) in enumerate(option_list, 1):
        # Load config to get display name
        if isinstance(path, Path):
            config = load_config(path)
            display_name = config.get('name', name)
            if config_type == 'models':
                size = config.get('size', '')
                print(f"  [{i}] {display_name} ({size})")
            elif config_type == 'datasets':
                dataset_name = config.get('dataset_name', '')
                print(f"  [{i}] {display_name} ({dataset_name})")
            else:
                desc = config.get('description', '')
                print(f"  [{i}] {name} - {desc}")
        else:
            print(f"  [{i}] {name}")

    print()

    while True:
        try:
            choice = input(f"Enter choice [1-{len(option_list)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(option_list):
                name, path = option_list[idx]
                config = load_config(path) if isinstance(path, Path) else path
                return name, config
            else:
                print(f"Please enter a number between 1 and {len(option_list)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)


def confirm_selection(model_config: Dict, dataset_config: Dict, training_config: Dict) -> bool:
    """Show summary and confirm before training."""
    print()
    print("=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Model:    {model_config.get('name')} ({model_config.get('model_name')})")
    print(f"Dataset:  {dataset_config.get('name')} ({dataset_config.get('dataset_name')})")
    print(f"Training: {training_config.get('name')} - {training_config.get('description', '')}")
    print()

    # Show key training parameters
    print("Training Parameters:")
    print(f"  - Batch size: {training_config.get('per_device_train_batch_size')}")
    print(f"  - Gradient accumulation: {training_config.get('gradient_accumulation_steps')}")
    print(f"  - Learning rate: {training_config.get('learning_rate')}")
    print(f"  - Max steps: {training_config.get('max_steps', 'full epoch')}")
    print(f"  - Scheduler: {training_config.get('lr_scheduler_type')}")
    print()

    while True:
        confirm = input("Proceed with training? [Y/n]: ").strip().lower()
        if confirm in ('', 'y', 'yes'):
            return True
        elif confirm in ('n', 'no'):
            return False
        print("Please enter 'y' or 'n'")


# =============================================================================
# Model Loading
# =============================================================================

def load_model_and_tokenizer(model_config: Dict) -> Tuple[Any, Any]:
    """
    Load the model and tokenizer with quantization and LoRA.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Tuple of (model, tokenizer)
    """
    model_name = model_config['model_name']
    quant_config = model_config.get('quantization', {})
    lora_config = model_config.get('lora', {})
    tokenizer_config = model_config.get('tokenizer', {})

    print()
    print("=" * 80)
    print(f"LOADING MODEL: {model_name}")
    print("=" * 80)

    # Configure quantization
    compute_dtype = torch.bfloat16 if quant_config.get('bnb_4bit_compute_dtype') == 'bfloat16' else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config.get('load_in_4bit', True),
        bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True),
    )

    print(f"Quantization: 4-bit {quant_config.get('bnb_4bit_quant_type', 'nf4')}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=tokenizer_config.get('trust_remote_code', True),
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer_config.get('padding_side', 'right')

    print("Tokenizer loaded.")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Model loaded.")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    print()
    print("Applying LoRA configuration...")

    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('lora_alpha', 16),
        lora_dropout=lora_config.get('lora_dropout', 0),
        bias=lora_config.get('bias', 'none'),
        task_type=lora_config.get('task_type', 'CAUSAL_LM'),
        target_modules=lora_config.get('target_modules', [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ]),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


# =============================================================================
# Dataset Loading and Formatting
# =============================================================================

def create_formatting_function(dataset_config: Dict):
    """
    Create a formatting function for the specific dataset.

    Args:
        dataset_config: Dataset configuration dictionary

    Returns:
        Function that formats examples into prompts
    """
    fields = dataset_config.get('fields', {})
    prompt_template = dataset_config.get('prompt_template', '')
    context_format = dataset_config.get('context_format')

    instruction_field = fields.get('instruction', 'instruction')
    response_field = fields.get('response', 'response')
    context_fields = fields.get('context_fields', [])
    options = fields.get('options', [])
    correct_option = fields.get('correct_option')

    def formatting_func(examples):
        """Format dataset examples into the prompt template."""
        texts = []

        # Get the number of examples in this batch
        num_examples = len(examples[instruction_field])

        for i in range(num_examples):
            instruction = examples[instruction_field][i]
            response = examples[response_field][i]

            # Handle MCQ datasets (like MedMCQA)
            if options and correct_option:
                opt_values = [examples[opt][i] for opt in options]
                correct_idx = examples[correct_option][i]
                # Build the correct answer text
                if isinstance(correct_idx, int) and 0 <= correct_idx < len(opt_values):
                    answer_letter = chr(65 + correct_idx)  # 0->A, 1->B, etc.
                    response = f"{answer_letter}) {opt_values[correct_idx]}"

                # Format MCQ prompt
                if context_fields:
                    context_values = {field: examples[field][i] for field in context_fields if field in examples}
                    context = context_format.format(**context_values) if context_format else ""
                else:
                    context = ""

                # MCQ-specific formatting
                text = prompt_template.format(
                    instruction,
                    opt_values[0] if len(opt_values) > 0 else "",
                    opt_values[1] if len(opt_values) > 1 else "",
                    opt_values[2] if len(opt_values) > 2 else "",
                    opt_values[3] if len(opt_values) > 3 else "",
                    context,
                    response
                )

            # Handle datasets with context fields
            elif context_fields and context_format:
                context_values = {}
                for field in context_fields:
                    if field in examples:
                        context_values[field] = examples[field][i]

                try:
                    context = context_format.format(**context_values)
                except KeyError:
                    context = ""

                text = prompt_template.format(instruction, context, response)

            # Handle simple Q&A datasets (no context)
            else:
                # Simple template without context
                text = prompt_template.format(instruction, response)

            texts.append(text)

        return {"text": texts}

    return formatting_func


def load_and_prepare_dataset(dataset_config: Dict) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
    """
    Load and prepare the dataset with train/val/test splits.

    Args:
        dataset_config: Dataset configuration dictionary

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset_name = dataset_config['dataset_name']
    split = dataset_config.get('split', 'train')
    split_config = dataset_config.get('train_val_test_split')

    print()
    print("=" * 80)
    print(f"LOADING DATASET: {dataset_name}")
    print("=" * 80)

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    print(f"Dataset loaded: {len(dataset)} examples")

    # Create formatting function and apply it
    formatting_func = create_formatting_function(dataset_config)
    dataset = dataset.map(formatting_func, batched=True)
    print("Dataset formatted.")

    # Split if configured
    val_dataset = None
    test_dataset = None

    if split_config:
        train_ratio = split_config.get('train', 0.6)
        val_ratio = split_config.get('val', 0.2)
        test_ratio = split_config.get('test', 0.2)
        seed = split_config.get('seed', 3407)

        # First split: train vs (val + test)
        temp_test_size = val_ratio + test_ratio
        split1 = dataset.train_test_split(test_size=temp_test_size, seed=seed)
        train_dataset = split1['train']

        # Second split: val vs test
        if test_ratio > 0:
            val_test_ratio = test_ratio / temp_test_size
            split2 = split1['test'].train_test_split(test_size=val_test_ratio, seed=seed)
            val_dataset = split2['train']
            test_dataset = split2['test']
        else:
            val_dataset = split1['test']

        print(f"Split: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset) if test_dataset else 0}")

        # Save test dataset for later evaluation
        if test_dataset:
            test_save_path = Path("outputs") / "test_dataset"
            test_save_path.parent.mkdir(parents=True, exist_ok=True)
            test_dataset.save_to_disk(str(test_save_path))
            print(f"Test dataset saved to: {test_save_path}")
    else:
        train_dataset = dataset
        print("Using full dataset for training (no split)")

    return train_dataset, val_dataset, test_dataset


# =============================================================================
# Trainer Creation
# =============================================================================

def create_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset],
    training_config: Dict,
    model_config: Dict,
    dataset_config: Dict,
    run_name: str,
) -> Tuple[Any, Path]:
    """
    Create the SFTTrainer with the merged configuration.

    Args:
        model: The LoRA-wrapped model
        tokenizer: The tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        training_config: Training configuration
        model_config: Model configuration
        dataset_config: Dataset configuration
        run_name: Unique run name for this training run

    Returns:
        Tuple of (SFTTrainer, output_dir)
    """
    # Determine precision
    auto_precision = training_config.get('auto_precision', True)
    if auto_precision:
        use_bf16 = torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16
    else:
        use_bf16 = training_config.get('bf16', False)
        use_fp16 = training_config.get('fp16', False)

    # Build output directory using run_name
    output_dir = Path(training_config.get('output_dir', 'outputs')) / run_name

    # Create training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        warmup_steps=training_config.get('warmup_steps', 10),
        num_train_epochs=training_config.get('num_train_epochs', 1),
        max_steps=training_config.get('max_steps', -1),
        learning_rate=training_config.get('learning_rate', 5e-5),
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=training_config.get('logging_steps', 1),
        optim=training_config.get('optim', 'adamw_8bit'),
        weight_decay=training_config.get('weight_decay', 0.01),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        seed=training_config.get('seed', 3407),
        output_dir=str(output_dir),
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config.get('save_steps', 25) if training_config.get('save_strategy') == 'steps' else None,
        save_total_limit=training_config.get('save_total_limit'),
        eval_strategy=training_config.get('eval_strategy', 'no'),
        eval_steps=training_config.get('eval_steps', 25) if training_config.get('eval_strategy') == 'steps' else None,
        push_to_hub=training_config.get('push_to_hub', False),
        report_to="wandb",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    return trainer, output_dir


# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    """Main entry point for the training pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen Model Fine-Tuning System")
    parser.add_argument('--model', type=str, help='Model config name (e.g., qwen2.5-14b-instruct)')
    parser.add_argument('--dataset', type=str, help='Dataset config name (e.g., quantum)')
    parser.add_argument('--training', type=str, help='Training config name (e.g., default)')
    parser.add_argument('--no-confirm', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    # Discover available configs
    model_configs = discover_configs('models')
    dataset_configs = discover_configs('datasets')
    training_configs = discover_configs('training')

    if not model_configs:
        print("ERROR: No model configs found in configs/models/")
        sys.exit(1)
    if not dataset_configs:
        print("ERROR: No dataset configs found in configs/datasets/")
        sys.exit(1)
    if not training_configs:
        print("ERROR: No training configs found in configs/training/")
        sys.exit(1)

    print_header()

    # Select or load configs
    if args.model and args.model in model_configs:
        model_name = args.model
        model_config = load_config(model_configs[args.model])
    else:
        model_name, model_config = interactive_select(
            model_configs, "Select a model:", "models"
        )

    print()

    if args.dataset and args.dataset in dataset_configs:
        dataset_name = args.dataset
        dataset_config = load_config(dataset_configs[args.dataset])
    else:
        dataset_name, dataset_config = interactive_select(
            dataset_configs, "Select a dataset:", "datasets"
        )

    print()

    if args.training and args.training in training_configs:
        training_name = args.training
        training_config = load_config(training_configs[args.training])
    else:
        training_name, training_config = interactive_select(
            training_configs, "Select training configuration:", "training"
        )

    # Confirm selection
    if not args.no_confirm:
        if not confirm_selection(model_config, dataset_config, training_config):
            print("Training cancelled.")
            sys.exit(0)

    # Generate unique run name
    run_name, run_index, run_timestamp = generate_run_name(model_config, dataset_config)
    print()
    print("=" * 80)
    print(f"RUN NAME: {run_name}")
    print(f"RUN INDEX: #{run_index:03d}")
    print("=" * 80)

    # Initialize WandB
    wandb_template = training_config.get('wandb_project_template', '{dataset_name}-{model_name}')
    wandb_project = wandb_template.format(
        model_name=model_config.get('name', model_name).replace(' ', '-'),
        dataset_name=dataset_config.get('name', dataset_name).replace(' ', '-'),
        model_size=model_config.get('size', ''),
    )
    wandb.init(project=wandb_project, name=run_name)
    print(f"\nWandB project: {wandb_project}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Load and prepare dataset
    train_dataset, val_dataset, test_dataset = load_and_prepare_dataset(dataset_config)

    # Create trainer
    print()
    print("=" * 80)
    print("CONFIGURING TRAINER")
    print("=" * 80)

    trainer, output_dir = create_trainer(
        model, tokenizer, train_dataset, val_dataset,
        training_config, model_config, dataset_config, run_name
    )

    print("Trainer configured.")

    # Display GPU stats
    print()
    print("=" * 80)
    print("GPU STATISTICS")
    print("=" * 80)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_stats = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {gpu_stats.name}")
            print(f"  Total Memory: {gpu_stats.total_memory / 1024**3:.2f} GB")
    else:
        print("No GPU available")

    # Start training
    print()
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer_stats = trainer.train()

    print()
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    # Save the fine-tuned model
    print()
    print("=" * 80)
    print("SAVING LORA ADAPTER")
    print("=" * 80)

    model_save_path = output_dir / "final_adapter"
    model_save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    print(f"LoRA adapter saved to: {model_save_path}")

    # Save run metadata for GGUF conversion
    run_metadata = {
        "run_name": run_name,
        "run_index": run_index,
        "base_model": model_config.get('model_name'),
        "model_size": model_config.get('size'),
        "dataset": dataset_config.get('name'),
    }
    with open(output_dir / "run_metadata.json", 'w') as f:
        json.dump(run_metadata, f, indent=2)

    # Log training run to training_log.md
    print()
    print("=" * 80)
    print("LOGGING TRAINING RUN")
    print("=" * 80)

    log_training_run(
        run_name=run_name,
        index=run_index,
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        output_dir=output_dir,
        trainer_stats=trainer_stats,
    )

    # Final summary
    print()
    print("=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print(f"\nRun Name: {run_name}")
    print(f"Run Index: #{run_index:03d}")
    print(f"LoRA Adapter: {model_save_path}")
    print(f"\nTo convert to GGUF format, run:")
    print(f"  python merge_and_convert_gguff.py --adapter-path {model_save_path}")
    print("=" * 80)

    wandb.finish()


if __name__ == "__main__":
    main()
