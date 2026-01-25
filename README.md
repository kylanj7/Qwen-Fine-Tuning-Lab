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
├── evaluate_model.py           # Model evaluation with RAG-grounded fact-checking
├── merge_and_convert_gguff.py  # LoRA merge + GGUF conversion (interactive)
├── convert_base_to_gguf.py     # Convert base model to GGUF (no LoRA)
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
│       ├── qwen2.5-14b-instruct-base-q4_k_m.gguf  # Base model
│       ├── qwen_chemistry_merged-q4_k_m.gguf      # Fine-tuned
│       └── qwen_quantum_merged-q4_k_m.gguf        # Fine-tuned
├── evaluation_results/         # Evaluation outputs
│   ├── eval_*.json             # Detailed results
│   ├── eval_*.txt              # Text logs
│   └── articles_*.json         # RAG sources for verification
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

#### Fine-tuned Model (with LoRA merge)

Interactive mode (recommended):
```bash
python merge_and_convert_gguff.py
```

This will:
1. List all available LoRA adapters from `outputs/`
2. Let you select quantization method
3. Auto-detect base model from run metadata
4. Merge and convert to GGUF

CLI mode:
```bash
python merge_and_convert_gguff.py --adapter-path outputs/checkpoint-100 --quantization q4_k_m
```

#### Base Model (no LoRA)

Convert the base Qwen model to GGUF for comparison testing:
```bash
python convert_base_to_gguf.py
```

Output: `models/gguf/qwen2.5-14b-instruct-base-q4_k_m.gguf`

### 4. Test with Streamlit

```bash
streamlit run app.py
```

### 5. Evaluate Model Accuracy

Interactive mode:
```bash
python evaluate_model.py
```

This will guide you through:
1. Selecting a GGUF model from `models/gguf/`
2. Choosing a test dataset
3. Setting the number of samples

## Model Evaluation

The evaluation system uses **RAG-grounded LLM-as-judge** scoring:

### How It Works

1. **Load Test Data**: Uses the test split from the dataset config
2. **Get Model Response**: Runs inference on your GGUF model via llama-cpp
3. **RAG Retrieval**: Fetches relevant academic papers from Semantic Scholar
4. **Multi-dimensional Scoring**: A judge model (gpt-oss:120b) scores on three dimensions:
   - **Factual Accuracy** (50%): Do claims match academic sources?
   - **Completeness** (30%): Does the response fully address the query?
   - **Technical Precision** (20%): Are equations and terminology correct?

### Output Files

Each evaluation run generates three files in `evaluation_results/`:

| File | Description |
|------|-------------|
| `eval_*.txt` | Full text log with Q&A and scores |
| `eval_*.json` | Structured results for analysis |
| `articles_*.json` | All Semantic Scholar papers used (for manual verification) |

### Article Logging

The `articles_*.json` file contains all papers used for fact-checking:

```json
{
  "article_logs": [
    {
      "question_index": 1,
      "question": "Explain quantum entanglement...",
      "search_keywords": "quantum entanglement",
      "papers_retrieved": [
        {
          "paper_id": "abc123",
          "title": "Quantum Entanglement in Many-Body Systems",
          "year": 2023,
          "authors": ["Alice", "Bob"],
          "citation_count": 150,
          "abstract": "...",
          "semantic_scholar_url": "https://www.semanticscholar.org/paper/abc123"
        }
      ]
    }
  ]
}
```

Use this to manually verify the sources the judge used for scoring.

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

### Training Log

All training runs are logged to `training_log.md` with full configuration details.

## Available Models

| Model | Size | Type |
|-------|------|------|
| `qwen2.5-14b-instruct-base-q4_k_m.gguf` | 8.4 GB | Base (untuned) |
| `qwen_chemistry_merged-q4_k_m.gguf` | 8.4 GB | Fine-tuned on Chemistry |
| `qwen_quantum_merged-q4_k_m.gguf` | 8.4 GB | Fine-tuned on Quantum Physics |

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

## Hardware Requirements

### Training
- **GPU**: 24GB+ VRAM (RTX 3090/4090, A5000, A100)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space

### Inference (GGUF)
- **RAM/VRAM**: 16GB minimum
- **Storage**: 10GB per model

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

### Semantic Scholar Rate Limiting
- Set `SEMANTIC_SCHOLAR_API_KEY` environment variable for faster API access
- The script handles rate limiting automatically with retries

## References

- [Qwen2.5 Models](https://huggingface.co/Qwen)
- [PEFT/LoRA](https://github.com/huggingface/peft)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Ollama](https://ollama.ai/)
- [Semantic Scholar API](https://api.semanticscholar.org/)

## License

Model weights are subject to their respective licenses (Qwen, dataset licenses).
Code is provided as-is for research and educational purposes.
