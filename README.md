# Qwen Fine-Tuning System

A unified, configuration-driven pipeline for fine-tuning Qwen models on scientific datasets with automated evaluation.

## Overview

- **Models**: Qwen 2.5 (7B, 14B, 32B) and Qwen 3 (4B, 8B, 14B, 32B)
- **Datasets**: Chemistry, Quantum Physics, Biology, Physics, Math, Materials Science, MedMCQA, SciQ
- **Quantization**: 4-bit NF4 for training, Q4_K_M GGUF for inference
- **Framework**: Transformers + PEFT/LoRA + bitsandbytes

## Project Structure

```
.
├── train.py                    # Unified training pipeline
├── app.py                      # Streamlit test interface
├── evaluate_model.py           # Model evaluation with LLM-as-judge
├── merge_and_convert_gguff.py  # LoRA merge + GGUF conversion
├── configs/
│   ├── models/                 # Model configurations
│   │   ├── qwen2.5-14b-instruct.yaml
│   │   ├── qwen3-8b.yaml
│   │   └── ...
│   ├── datasets/               # Dataset configurations
│   │   ├── chemistry.yaml
│   │   ├── quantum.yaml
│   │   └── ...
│   └── training/               # Training configurations
│       ├── default.yaml
│       └── quick_test.yaml
├── models/
│   └── gguf/                   # GGUF models for inference
│       ├── qwen_chemistry_merged-q4_k_m.gguf
│       └── qwen_quantum_merged-q4_k_m.gguf
└── outputs/                    # Training outputs and checkpoints
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Model

Interactive mode (select model, dataset, and training config):
```bash
python train.py
```

Command-line mode:
```bash
python train.py --model qwen2.5-14b-instruct --dataset chemistry --training default
```

### 3. Convert to GGUF

After training, merge LoRA weights and convert to GGUF:
```bash
# Uses run metadata to auto-detect base model and generate proper filename
python merge_and_convert_gguff.py --adapter-path outputs/qwen25-14b-instruct-chemistry-20260124-001/final_adapter
```

Output will be saved to `models/gguf/` with naming convention:
```
{model}-{size}-{dataset}-{timestamp}-{index}-{quantization}.gguf
```

Example: `qwen25-14b-instruct-chemistry-20260124-001-q4_k_m.gguf`

### 4. Test with Streamlit

```bash
streamlit run app.py
```

### 5. Evaluate Model Accuracy

```bash
# List available datasets
python evaluate_model.py --list-datasets

# Evaluate a model
python evaluate_model.py --model qwen_chemistry:latest --dataset chemistry

# Evaluate with more samples, no web search
python evaluate_model.py --model my_model:latest --dataset quantum --max_samples 200 --no-web-search
```

## Configuration System

### Model Configs (`configs/models/`)

Define model name, quantization, LoRA settings, and max sequence length:

```yaml
name: "Qwen 2.5 14B Instruct"
model_name: "Qwen/Qwen2.5-14B-Instruct"
size: "14B"

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"

lora:
  r: 16
  lora_alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

max_seq_length: 2048
```

### Dataset Configs (`configs/datasets/`)

Define dataset source, field mappings, prompt template, and train/val/test splits:

```yaml
name: "Chemistry"
dataset_name: "camel-ai/chemistry"
domain: "Chemistry"

fields:
  instruction: "message_1"
  response: "message_2"
  context_fields: ["topic", "sub_topic"]

prompt_template: |
  Below is an instruction...
  ### Instruction:
  {}
  ### Response:
  {}

train_val_test_split:
  train: 0.6
  val: 0.2
  test: 0.2
```

### Training Configs (`configs/training/`)

Define batch size, learning rate, scheduler, checkpointing:

```yaml
name: "default"
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 5.0e-5
lr_scheduler_type: "cosine"
num_train_epochs: 1
save_strategy: "steps"
save_steps: 25
eval_strategy: "steps"
eval_steps: 25
```

## Training Log & Model Naming

### Naming Convention

Each training run generates a unique name:
```
{model}-{size}-{dataset}-{timestamp}-{index}
```

Example: `qwen25-14b-instruct-chemistry-20260124-001`

Components:
- **model**: Model name (e.g., `qwen25-14b-instruct`)
- **size**: Parameter count (e.g., `14b`)
- **dataset**: Dataset name (e.g., `chemistry`)
- **timestamp**: Date in YYYYMMDD format
- **index**: Auto-incrementing run number (001, 002, ...)

### Training Log

All training runs are logged to `training_log.md` with full configuration details:

```markdown
## Run #001: qwen25-14b-instruct-chemistry-20260124-001

**Date:** 2026-01-24 10:30:00

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Name | Qwen 2.5 14B Instruct |
| HuggingFace ID | Qwen/Qwen2.5-14B-Instruct |
| LoRA Rank (r) | 16 |
...

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Learning Rate | 5e-5 |
...
```

This makes it easy to:
- Track all experiments in one place
- Compare configurations between runs
- Reproduce previous training runs
- Document model provenance

## Model Evaluation

The evaluation system (`evaluate_model.py`) uses an **LLM-as-judge** approach:

### How It Works

1. **Load Test Data**: Uses the 20% test split from the dataset config
2. **Get Model Response**: Queries your fine-tuned model via Ollama
3. **Gather References**: Combines ground truth + optional web search results
4. **Judge Scoring**: A larger model (default: gpt-oss:120b) scores the answer:
   - **CORRECT (1.0)**: Accurate and complete
   - **PARTIAL (0.5)**: Core concept right, minor issues
   - **INCORRECT (0.0)**: Wrong facts or off-topic

### Output

```
============================================================
CHEMISTRY MODEL EVALUATION RESULTS
============================================================
Evaluation Date: 2026-01-24 10:30:00
Test Model: qwen_chemistry:latest
Judge Model: gpt-oss:120b
Dataset: camel-ai/chemistry
Total Samples: 100

------------------------------------------------------------
OVERALL SCORES
------------------------------------------------------------
Fully Correct:       72 (72.0%)
Partially Correct:   18 (18.0%)
Incorrect:           10 (10.0%)

RAW ACCURACY SCORE: 81.0%

------------------------------------------------------------
SCORES BY TOPIC
------------------------------------------------------------
Organic Chemistry                    87.5% (n=24)
Physical Chemistry                   82.0% (n=20)
...
```

Results are saved to `evaluation_results/` as JSON (detailed) and TXT (report).

## Streamlit Test Interface

The app provides:

- **Model Selection**: Auto-detects GGUF models in `models/gguf/`
- **GPU Acceleration**: Full GPU offloading with llama-cpp-python
- **Generation Controls**: Temperature, top-p, top-k, max tokens
- **Live Metrics**: Tokens/second, total tokens, elapsed time
- **Custom System Prompts**: Adjust model behavior

## Hardware Requirements

### Training
- **GPU**: 24GB+ VRAM (RTX 3090/4090, A5000, A100)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space

### Inference (GGUF)
- **RAM/VRAM**: 16GB minimum
- **Storage**: 20GB per model

## Available Datasets

| Config Name | Dataset | Domain |
|-------------|---------|--------|
| `chemistry` | camel-ai/chemistry | Chemistry |
| `quantum` | BoltzmannEntropy/QuantumLLMInstruct | Quantum Physics |
| `physics` | camel-ai/physics | Physics |
| `biology` | camel-ai/biology | Biology |
| `math` | camel-ai/math | Mathematics |
| `materials-science` | pranked03/materials_science_dataset | Materials Science |
| `medmcqa` | openlifescienceai/medmcqa | Medical (MCQ) |
| `sciq` | allenai/sciq | Science (MCQ) |

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` in training config
- Increase `gradient_accumulation_steps`
- Use a smaller model (7B or 8B)

### Slow Inference
- Ensure GPU layers are enabled in Streamlit app
- Use Q4_K_M quantization (good balance of speed/quality)
- Reduce context length

### Evaluation Errors
- Ensure Ollama is running: `ollama serve`
- Check model is loaded: `ollama list`
- Verify judge model is available: `ollama pull gpt-oss:120b`

## References

- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)

## License

Model weights are subject to their respective licenses (Qwen, dataset licenses).
Code is provided as-is for research and educational purposes.
