# Nemotron-3-30B Quantum Physics Fine-Tuning & Inference

This project fine-tunes NVIDIA's Nemotron-3-30B model on quantum physics datasets and provides a Streamlit chatbot interface for testing.

## 📋 Overview

- **Model**: NVIDIA Nemotron-3-Nano-30B-A3B-BF16
- **Dataset**: BoltzmannEntropy/QuantumLLMInstruct (500k+ quantum computing instruction pairs)
- **Quantization**: FP4 (4-bit) for training, Q4_K_M GGUF for inference
- **Framework**: Unsloth + LoRA fine-tuning

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to fine-tune Nemotron-3 on quantum physics:

```bash
python train.py
```

**Training Process:**
- Loads Nemotron-3-30B in FP4 quantization
- Applies LoRA (Low-Rank Adaptation) fine-tuning
- Trains on BoltzmannEntropy/QuantumLLMInstruct dataset
- Converts to GGUF format automatically
- Saves GGUF model in project directory

**Output:** `nemotron3_quantum_model-q4_k_m-unsloth.gguf`

### 3. Test with Streamlit App

After training completes, run the chatbot interface:

```bash
streamlit run app.py
```

The app will automatically detect and load available GGUF models.

## 🎛️ Streamlit App Features

### Model Controls
- **Model Selection**: Choose from available GGUF models
- **Context Length**: Adjust context window (512-32768 tokens)
- **GPU Layers**: Control GPU offloading (-1 for full GPU)
- **CPU Threads**: Set number of threads for CPU inference

### Generation Parameters
- **Max Tokens**: Control response length (1-4096)
- **Temperature**: Adjust creativity (0.0-2.0)
- **Top P**: Nucleus sampling threshold (0.0-1.0)
- **Top K**: Top-K sampling limit (0-100)

### Advanced Parameters
- **Repeat Penalty**: Prevent repetition (1.0-2.0)
- **Presence Penalty**: Token presence control (0.0-2.0)
- **Frequency Penalty**: Frequency-based control (0.0-2.0)
- **Mirostat Sampling**: Advanced sampling modes (0, 1, 2)
  - Mirostat Tau: Target entropy
  - Mirostat Eta: Learning rate

### Live Metrics
- **Tokens/Second**: Real-time generation speed
- **Tokens Generated**: Total token count
- **Elapsed Time**: Time tracking
- **Streaming Output**: Live text generation

### System Prompt
Customize the model's behavior with custom system prompts for different quantum physics tasks.

## 📊 Training Configuration

### Active Parameters
```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
warmup_steps = 5
max_steps = 20
learning_rate = 2e-4
weight_decay = 0.01
lr_scheduler_type = "linear"
optimizer = "adamw_8bit"
```

### LoRA Configuration
```python
r = 16  # LoRA rank
lora_alpha = 16
lora_dropout = 0  # No dropout
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

### Additional Parameters (Currently Set to 0)
The training script includes robust parameter options that are currently disabled:
- warmup_ratio
- max_grad_norm
- evaluation_strategy
- save_strategy
- label_smoothing_factor
- And many more...

You can adjust these in `train.py` as needed for your specific use case.

## 🔧 Hardware Requirements

### Minimum for Training
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A5000, etc.)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space

### Minimum for Inference (GGUF)
- **RAM/VRAM**: 16GB (can run on CPU or GPU)
- **Storage**: 20GB free space

## 📁 Project Structure

```
.
├── train.py                    # Training script
├── app.py                      # Streamlit chatbot interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── outputs/                    # Training outputs
├── nemotron3_quantum_model/    # Saved model (safetensors)
└── nemotron3_quantum_model-q4_k_m-unsloth.gguf  # GGUF model
```

## 🧪 Dataset Information

**BoltzmannEntropy/QuantumLLMInstruct** contains:
- 500,000+ instruction-following pairs
- 90 quantum computing domains
- Topics include:
  - Hamiltonian dynamics
  - Quantum circuit optimization
  - Yang-Baxter solvability
  - Variational Quantum Eigensolvers (VQE)
  - Quantum thermodynamics
  - Quantum phase estimation
  - Trotter-Suzuki decompositions

## 🎯 Use Cases

- Quantum computing education
- Quantum algorithm development assistance
- Quantum physics problem-solving
- Research support for quantum mechanics
- Quantum circuit design help

## ⚠️ Important Notes

1. **FP4 Quantization**: Used during training to fit 30B model in limited VRAM
2. **No Dropout**: `lora_dropout=0` for cleaner fine-tuning
3. **No Data Augmentation**: Using clean dataset as-is
4. **GGUF Format**: Optimized for inference, not training
5. **Streaming**: Live token generation for better UX

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`

### Slow Inference
- Increase `n_gpu_layers` in Streamlit app
- Use smaller GGUF quantization (q4_k_m is recommended)
- Reduce `max_tokens`

### Model Not Found
- Ensure training completed successfully
- Check for `.gguf` files in project directory
- Verify GGUF conversion didn't error

## 📚 References

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Nemotron-3 Model Card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [QuantumLLMInstruct Dataset](https://huggingface.co/datasets/BoltzmannEntropy/QuantumLLMInstruct)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## 📄 License

This project uses:
- NVIDIA Nemotron Open Model License (for the model)
- Apache 2.0 (for the dataset)

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Happy Quantum Computing! ⚛️**
