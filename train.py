import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import wandb

wandb.init(project="QUANTUM-DATASET-QWEN-2.5-14B-INSTRUCT")

print("=" * 80)
print("QWEN2.5-14B QUANTUM PHYSICS FINE-TUNING")
print("=" * 80)

# Get the major and minor version of the current CUDA device (GPU)
if torch.cuda.is_available():
    major_version, minor_version = torch.cuda.get_device_capability()
    print(f"CUDA Device Capability: {major_version}.{minor_version}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
else:
    print("WARNING: No CUDA device detected!")

# Model and Training Configuration
print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)

max_seq_length = 2048
model_name = "Qwen/Qwen2.5-14B-Instruct"

print(f"Model: {model_name}")
print(f"Max Sequence Length: {max_seq_length}")
print(f"Quantization: 4-bit NF4")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load Qwen2.5-14B model
print("\n" + "=" * 80)
print(f"LOADING {model_name} MODEL")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Model loaded successfully!")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Apply PEFT (LoRA) configuration
print("\n" + "=" * 80)
print("APPLYING PEFT (LoRA) CONFIGURATION")
print("=" * 80)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("PEFT configuration applied successfully!")

# Define prompt template for quantum physics instruction following
quantum_prompt = """Below is an instruction that describes a quantum physics task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def formatting_prompts_func(examples):
    """Format dataset examples into the quantum physics prompt template"""
    problems = examples["problem"]
    solutions = examples["solution"]
    main_domains = examples["main_domain"]
    sub_domains = examples["sub_domain"]
    
    texts = []
    for problem, solution, main_domain, sub_domain in zip(problems, solutions, main_domains, sub_domains):
        # Use domain as input context
        context = f"Domain: {main_domain} - {sub_domain}"
        text = quantum_prompt.format(problem, context, solution)
        texts.append(text)
    return {"text": texts}
    
# Load Quantum Physics Dataset
print("\n" + "=" * 80)
print("LOADING QUANTUM PHYSICS DATASET")
print("=" * 80)

dataset = load_dataset("BoltzmannEntropy/QuantumLLMInstruct", split="train")
print(f"Dataset loaded: {len(dataset)} examples")

# Apply formatting
dataset = dataset.map(formatting_prompts_func, batched=True)
print("Dataset formatted successfully!")

# Training Configuration
print("\n" + "=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=20,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="no",
        push_to_hub=False,
    ),
)

print("Trainer configured successfully!")

# Display GPU stats before training
print("\n" + "=" * 80)
print("GPU STATISTICS")
print("=" * 80)

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu_stats = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_stats.name}")
        print(f"  Total Memory: {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 2)} GB")
else:
    print("No GPU available")

# Start Training
print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

trainer_stats = trainer.train()

print("\n" + "=" * 80)
print("TRAINING COMPLETED")
print("=" * 80)

# Save the fine-tuned LoRA adapter
print("\n" + "=" * 80)
print("SAVING LORA ADAPTER")
print("=" * 80)

model_save_path = "qwen_quantum_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"LoRA adapter saved to: {model_save_path}")

print("\n" + "=" * 80)
print("TRAINING PIPELINE COMPLETED")
print("=" * 80)
print(f"\nLoRA Adapter: {model_save_path}")
print(f"\nTo convert to GGUF format, run:")
print(f"  python convert_to_gguf.py --adapter-path {model_save_path}")
print("=" * 80)

wandb.finish()
